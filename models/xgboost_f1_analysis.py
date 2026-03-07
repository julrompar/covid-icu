import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    brier_score_loss, roc_curve, precision_recall_curve,
    f1_score, confusion_matrix
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

output_dir = "F1-optimized"
os.makedirs(output_dir, exist_ok=True)

# ── 1. CARGA Y PREPARACIÓN ────────────────────────────────────────────────────
df = pd.read_csv("../datasets/v1/dataset_v1_clean.csv")
df["target"] = df["dod_within_30_days"].notnull().astype(int)

drop_cols = ["target", "dod_within_30_days", "subject_id", "admission_id",
             "stay_id", "is_dnr", "has_chest_tube"]

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
def add_engineered_features(df):
    df = df.copy()
    df["mean_arterial_pressure"] = (df["min_bp_systolic"] + 2 * df["min_bp_diastolic"]) / 3
    df["shock_index"]            = df["max_pulse"] / df["min_bp_systolic"].replace(0, np.nan)
    df["severe_hypoxemia"]       = (df["min_oxygen_saturation"] < 90).astype(int)
    df["fever"]                  = (df["max_temp"] > 101.0).astype(int)
    df["hypothermia"]            = (df["max_temp"] < 96.8).astype(int)
    df["tachycardia_severe"]     = (df["max_pulse"] > 130).astype(int)
    df["hypotension"]            = (df["min_bp_systolic"] < 90).astype(int)
    organ_support_cols = ["is_mechanical_ventilation", "is_crrt", "has_arterial_line",
                          "is_intubated", "is_prone_position", "has_central_line",
                          "has_hemodialysis", "has_niv", "is_ecmo"]
    existing = [c for c in organ_support_cols if c in df.columns]
    df["organ_support_count"] = df[existing].sum(axis=1)
    df["age_x_ventilation"]   = df["anchor_age"] * df["is_mechanical_ventilation"]
    df["age_x_hypoxemia"]     = df["anchor_age"] * df["severe_hypoxemia"]
    df["age_x_hypotension"]   = df["anchor_age"] * df["hypotension"]
    map_score   = pd.cut(df["mean_arterial_pressure"],
                         bins=[-np.inf, 65, 70, np.inf], labels=[2, 1, 0]).astype(float)
    resp_score  = np.where(df["min_oxygen_saturation"] < 90, 2,
                  np.where(df["min_oxygen_saturation"] < 94, 1, 0))
    renal_score = df["is_crrt"] * 3
    df["sofa_partial"] = map_score.fillna(0) + resp_score + renal_score
    return df

df_eng = add_engineered_features(df)
feature_cols = [c for c in df_eng.columns if c not in drop_cols + ["dod_within_30_days", "target"]]
X = df_eng[feature_cols]
y = df_eng["target"]

# ── 3. SPLITS ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# ── 4. ENCODING CATEGÓRICO (fit solo en train) ────────────────────────────────
cat_cols = ["gender", "marital_status", "race"]
num_cols = [col for col in X_train.columns if col not in cat_cols]

encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoder.fit(X_train_fit[cat_cols])

def encode_df(df_in, enc, cats, nums):
    enc_arr = enc.transform(df_in[cats])
    enc_df  = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(cats), index=df_in.index)
    return pd.concat([df_in[nums].reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)

X_train_enc = encode_df(X_train_fit, encoder, cat_cols, num_cols)
X_val_enc   = encode_df(X_val,       encoder, cat_cols, num_cols)
X_test_enc  = encode_df(X_test,      encoder, cat_cols, num_cols)

# ── 5. IMPUTACIÓN MICE (fit solo en train) ────────────────────────────────────
print("--- Iniciando Imputación MICE ---")
imputer = IterativeImputer(max_iter=10, random_state=42)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_enc), columns=X_train_enc.columns)
X_val_imp   = pd.DataFrame(imputer.transform(X_val_enc),       columns=X_val_enc.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test_enc),      columns=X_test_enc.columns)

ratio = float(np.sum(y_train_fit == 0) / np.sum(y_train_fit == 1))

# ── 6. OPTUNA ─────────────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": ratio,
        "random_state":     42,
        "eval_metric":      "aucpr",
    }
    model  = xgb.XGBClassifier(**params)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_imp, y_train_fit, cv=cv, scoring="average_precision")
    return scores.mean()

print("--- Iniciando búsqueda de hiperparámetros con Optuna ---")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            study_name="xgboost_f1",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params["scale_pos_weight"] = ratio
best_params["random_state"]     = 42
best_params["eval_metric"]      = "aucpr"

print(f"\nMejor AUPRC (CV): {study.best_value:.4f}")
print("Mejores hiperparámetros:", best_params)

study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)
with open(os.path.join(output_dir, "best_params.json"), "w") as fp:
    json.dump({"best_auprc_cv": study.best_value, "params": best_params}, fp, indent=2)

# ── 7. CV FINAL ───────────────────────────────────────────────────────────────
base_model = xgb.XGBClassifier(**best_params)
print("\n--- Validación Cruzada final (5-Fold) ---")
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auroc_cv = cross_val_score(base_model, X_train_imp, y_train_fit, cv=cv, scoring="roc_auc")
auprc_cv = cross_val_score(base_model, X_train_imp, y_train_fit, cv=cv, scoring="average_precision")
print(f"CV AUROC: {auroc_cv.mean():.4f} ± {auroc_cv.std():.4f}")
print(f"CV AUPRC: {auprc_cv.mean():.4f} ± {auprc_cv.std():.4f}")

# ── 8. CALIBRACIÓN ────────────────────────────────────────────────────────────
print("\n--- Calibrando modelo sobre validation set ---")
base_model.fit(X_train_imp, y_train_fit)
calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
calibrated_model.fit(X_val_imp, y_val)

# ── 9. UMBRAL ÓPTIMO EN VALIDATION ───────────────────────────────────────────
y_probs_val       = calibrated_model.predict_proba(X_val_imp)[:, 1]
thresholds_search = np.linspace(0.01, 0.99, 200)
f1_scores_val = np.array([
    f1_score(y_val, y_probs_val > thr, pos_label=1, zero_division=0)
    for thr in thresholds_search
])
optimal_idx        = np.argmax(f1_scores_val)
optimal_threshold  = float(thresholds_search[optimal_idx])

print("=" * 50)
print("OPTIMIZACIÓN POR F1 (sobre validation set)")
print("=" * 50)
print(f"Umbral óptimo: {optimal_threshold:.4f}")
print(f"F1 máximo (val): {f1_scores_val[optimal_idx]:.4f}")

with open(os.path.join(output_dir, "best_params.json"), "r") as fp:
    saved = json.load(fp)
saved["optimal_threshold_f1"] = optimal_threshold
saved["val_f1_at_threshold"]  = float(f1_scores_val[optimal_idx])
saved["cv_auroc_mean"] = float(auroc_cv.mean())
saved["cv_auprc_mean"] = float(auprc_cv.mean())
with open(os.path.join(output_dir, "best_params.json"), "w") as fp:
    json.dump(saved, fp, indent=2)

# ── 10. EVALUACIÓN FINAL EN TEST SET ─────────────────────────────────────────
y_probs        = calibrated_model.predict_proba(X_test_imp)[:, 1]
y_pred_default = (y_probs > 0.5).astype(int)
y_pred_opt     = (y_probs > optimal_threshold).astype(int)

auroc_final = roc_auc_score(y_test, y_probs)
auprc_final = average_precision_score(y_test, y_probs)
brier       = brier_score_loss(y_test, y_probs)

print("=" * 50)
print("MÉTRICAS FINALES — TEST SET")
print("=" * 50)
print(f"AUROC:       {auroc_final:.4f}")
print(f"AUPRC:       {auprc_final:.4f}")
print(f"Brier Score: {brier:.4f}")
print("\nClasificación umbral 0.5:")
print(classification_report(y_test, y_pred_default))
print(f"\nClasificación umbral F1-óptimo ({optimal_threshold:.4f}):")
print(classification_report(y_test, y_pred_opt))

saved["test_auroc"] = auroc_final
saved["test_auprc"] = auprc_final
saved["test_brier"] = brier
with open(os.path.join(output_dir, "best_params.json"), "w") as fp:
    json.dump(saved, fp, indent=2)

# ── 11. CURVAS ROC / PR / CALIBRACIÓN ────────────────────────────────────────
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUROC {auroc_final:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Tasa Falsos Positivos"); plt.ylabel("Tasa Verdaderos Positivos")
plt.title("Curva ROC"); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision, color="green", lw=2, label=f"AUPRC {auprc_final:.3f}")
plt.xlabel("Recall (Sensibilidad)"); plt.ylabel("Precisión")
plt.title("Curva Precision-Recall"); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
plt.plot(prob_pred, prob_true, "s-", color="red", label="XGBoost Calibrado")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectamente calibrado")
plt.xlabel("Probabilidad Predicha"); plt.ylabel("Fracción Real")
plt.title("Fiabilidad Calibración"); plt.legend(); plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_evaluation_base.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── 12. CURVA MÉTRICA vs UMBRAL ───────────────────────────────────────────────
plt.figure(figsize=(6, 4))
plt.plot(thresholds_search, f1_scores_val, color="purple", lw=2)
plt.axvline(optimal_threshold, color="red", linestyle="--",
            label=f"Umbral óptimo = {optimal_threshold:.3f}")
plt.xlabel("Umbral de Decisión"); plt.ylabel("F1-Score clase 1")
plt.title("Optimización del Umbral por F1-Score (validation set)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "f1_vs_threshold.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── 13. IMPORTANCIA HIPERPARÁMETROS OPTUNA ───────────────────────────────────
importances = optuna.importance.get_param_importances(study)
plt.figure(figsize=(7, 4))
plt.barh(list(importances.keys())[::-1], list(importances.values())[::-1], color="steelblue")
plt.xlabel("Importancia relativa")
plt.title("Importancia de hiperparámetros (Optuna)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "optuna_param_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── 14. MATRICES DE CONFUSIÓN ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_default = confusion_matrix(y_test, y_pred_default)
sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Pred Vivo", "Pred Muerte"], yticklabels=["Real Vivo", "Real Muerte"])
axes[0].set_title("Matriz de Confusión — Umbral 0.5")
axes[0].set_ylabel("Valor Real"); axes[0].set_xlabel("Valor Predicho")

cm_opt = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["Pred Vivo", "Pred Muerte"], yticklabels=["Real Vivo", "Real Muerte"])
axes[1].set_title(f"Matriz de Confusión — Umbral F1-óptimo ({optimal_threshold:.4f})")
axes[1].set_ylabel("Valor Real"); axes[1].set_xlabel("Valor Predicho")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrices_f1.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── 15. SHAP ──────────────────────────────────────────────────────────────────
print("\n--- Generando explicaciones SHAP ---")
explainer   = shap.TreeExplainer(calibrated_model.estimator)
shap_values = explainer.shap_values(X_test_imp)
shap.summary_plot(shap_values, X_test_imp, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nDone. Resultados guardados en:", output_dir)