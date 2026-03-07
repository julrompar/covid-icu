import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder

# ── Cargar best_params ────────────────────────────────────────────────────────
with open("F1-optimized/best_params.json") as f:
    saved = json.load(f)
best_params   = saved["params"]
opt_threshold = saved["optimal_threshold_f1"]

output_dir = "F1-optimized"

# ── 1. CARGA Y PREPARACIÓN ────────────────────────────────────────────────────
df = pd.read_csv("../datasets/v1/dataset_v1_clean.csv")
df["target"] = df["dod_within_30_days"].notnull().astype(int)
drop_cols = ["target", "dod_within_30_days", "subject_id", "admission_id", "stay_id", "is_dnr"]
features  = [col for col in df.columns if col not in drop_cols]

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
def add_engineered_features(df):
    df = df.copy()

    # mean_arterial_pressure — evitar "map" que colisiona con df.map()
    df["mean_arterial_pressure"] = (df["min_bp_systolic"] + 2 * df["min_bp_diastolic"]) / 3

    # Shock index proxy: pulso / presión sistólica (>1.0 indica inestabilidad)
    df["shock_index"] = df["max_pulse"] / df["min_bp_systolic"].replace(0, np.nan)

    # Hipoxemia severa: SpO2 < 90
    df["severe_hypoxemia"] = (df["min_oxygen_saturation"] < 90).astype(int)

    # Fiebre / hipotermia
    df["fever"]       = (df["max_temp"] > 101.0).astype(int)
    df["hypothermia"] = (df["max_temp"] < 96.8).astype(int)

    # Taquicardia severa (>130 lpm)
    df["tachycardia_severe"] = (df["max_pulse"] > 130).astype(int)

    # Hipotensión (SBP < 90 mmHg)
    df["hypotension"] = (df["min_bp_systolic"] < 90).astype(int)

    # Carga de soporte orgánico
    organ_support_cols = ["is_mechanical_ventilation", "is_crrt", "has_arterial_line",
                          "is_intubated", "is_prone_position", "has_central_line",
                          "has_hemodialysis", "has_niv", "is_ecmo"]
    existing = [c for c in organ_support_cols if c in df.columns]
    df["organ_support_count"] = df[existing].sum(axis=1)

    # Interacciones edad × complicación
    df["age_x_ventilation"] = df["anchor_age"] * df["is_mechanical_ventilation"]
    df["age_x_hypoxemia"]   = df["anchor_age"] * df["severe_hypoxemia"]
    df["age_x_hypotension"] = df["anchor_age"] * df["hypotension"]

    # SOFA parcial proxy (cardiovascular + respiratorio + renal)
    map_score = pd.cut(df["mean_arterial_pressure"],
                       bins=[-np.inf, 65, 70, np.inf],
                       labels=[2, 1, 0]).astype(float)
    resp_score  = np.where(df["min_oxygen_saturation"] < 90, 2,
                  np.where(df["min_oxygen_saturation"] < 94, 1, 0))
    renal_score = df["is_crrt"] * 3
    df["sofa_partial"] = map_score.fillna(0) + resp_score + renal_score

    return df

# Aplicar feature engineering sobre el df completo
df_eng = add_engineered_features(df)
feature_cols = [c for c in df_eng.columns if c not in drop_cols + ["dod_within_30_days", "target"]]
X = df_eng[feature_cols]
y = df_eng["target"]

# ── 3. SPLITS (mismos random_state que el script principal) ───────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

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

imputer = IterativeImputer(max_iter=10, random_state=42)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_enc), columns=X_train_enc.columns)
X_val_imp   = pd.DataFrame(imputer.transform(X_val_enc),       columns=X_val_enc.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test_enc),      columns=X_test_enc.columns)

# ── 4. REENTRENAR CON BEST PARAMS + FEATURES NUEVAS ──────────────────────────
base_model = xgb.XGBClassifier(**best_params)
base_model.fit(X_train_imp, y_train_fit)
calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
calibrated_model.fit(X_val_imp, y_val)

y_probs = calibrated_model.predict_proba(X_test_imp)[:, 1]
y_pred  = (y_probs > opt_threshold).astype(int)

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
print(f"AUROC: {roc_auc_score(y_test, y_probs):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_probs):.4f}")
print(f"F1:    {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ── 5. SHAP GLOBAL ────────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(calibrated_model.estimator)
shap_values = explainer.shap_values(X_test_imp)

shap.summary_plot(shap_values, X_test_imp, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_engineered.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── 6. SHAP WATERFALL — EJEMPLOS INDIVIDUALES ─────────────────────────────────
y_test_arr  = y_test.values
y_probs_arr = y_probs

tp_idx = np.where((y_test_arr == 1) & (y_pred == 1))[0]
tp_idx = tp_idx[np.argsort(y_probs_arr[tp_idx])[::-1]][:3]

tn_idx = np.where((y_test_arr == 0) & (y_pred == 0))[0]
tn_idx = tn_idx[np.argsort(y_probs_arr[tn_idx])][:3]

fn_idx = np.where((y_test_arr == 1) & (y_pred == 0))[0]
fn_idx = fn_idx[np.argsort(y_probs_arr[fn_idx])][:3]

fp_idx = np.where((y_test_arr == 0) & (y_pred == 1))[0]
fp_idx = fp_idx[np.argsort(y_probs_arr[fp_idx])[::-1]][:3]

groups = [
    ("Verdaderos Positivos (predice muerte ✓)", tp_idx, "shap_tp"),
    ("Verdaderos Negativos (predice vida ✓)",   tn_idx, "shap_tn"),
    ("Falsos Negativos (muere, predice vida ✗)", fn_idx, "shap_fn"),
    ("Falsos Positivos (vive, predice muerte ✗)", fp_idx, "shap_fp"),
]

for group_name, idxs, fname in groups:
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle(group_name, fontsize=13, fontweight="bold")
    for ax_i, idx in enumerate(idxs):
        sv  = shap_values[idx]
        top = np.argsort(np.abs(sv))[::-1][:12]
        feat_names = X_test_imp.columns[top]
        feat_vals  = X_test_imp.iloc[idx][feat_names].values
        sv_top     = sv[top]
        colors     = ["#d62728" if v > 0 else "#1f77b4" for v in sv_top]
        axes[ax_i].barh(range(len(sv_top)), sv_top[::-1], color=colors[::-1])
        axes[ax_i].set_yticks(range(len(sv_top)))
        axes[ax_i].set_yticklabels(
            [f"{n}={v:.2f}" for n, v in zip(feat_names[::-1], feat_vals[::-1])],
            fontsize=8
        )
        axes[ax_i].axvline(0, color="black", linewidth=0.8)
        axes[ax_i].set_title(f"Ejemplo {ax_i+1}  |  P(muerte)={y_probs_arr[idx]:.3f}", fontsize=9)
        axes[ax_i].set_xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {fname}.png")

print("\nDone.")