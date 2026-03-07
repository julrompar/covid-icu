import json
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, classification_report)

with open("F1-optimized/best_params.json") as f:
    saved = json.load(f)
best_params   = saved["params"]
opt_threshold = saved["optimal_threshold_f1"]
output_dir    = "F1-optimized"

# ── 1. CARGA ──────────────────────────────────────────────────────────────────
df = pd.read_csv("../datasets/v1/dataset_v1_clean.csv")
df["target"] = df["dod_within_30_days"].notnull().astype(int)
drop_cols = ["target", "dod_within_30_days", "subject_id", "admission_id",
             "stay_id", "is_dnr", "has_chest_tube"]   # ← has_chest_tube eliminado

# ── 2. FEATURE ENGINEERING v2 ────────────────────────────────────────────────
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

    # ✅ Interacciones suavizadas con log(edad)
    log_age = np.log1p(df["anchor_age"])
    df["age_x_ventilation"] = log_age * df["is_mechanical_ventilation"]
    df["age_x_hypoxemia"]   = log_age * df["severe_hypoxemia"]
    df["age_x_hypotension"] = log_age * df["hypotension"]

    # ✅ age_bin categórica
    df["age_bin"] = pd.cut(df["anchor_age"],
                           bins=[0, 50, 65, 75, 85, 120],
                           labels=["lt50", "50_64", "65_74", "75_84", "85plus"])

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
age_series = df_eng["anchor_age"]

# ── 3. SPLITS ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

age_test = age_series.loc[X_test.index].values

# age_bin se codifica junto con gender, marital_status, race
cat_cols = ["gender", "marital_status", "race", "age_bin"]
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

# ── 4. MODELO ─────────────────────────────────────────────────────────────────
base_model = xgb.XGBClassifier(**best_params)
base_model.fit(X_train_imp, y_train_fit)
calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
calibrated_model.fit(X_val_imp, y_val)

y_probs = calibrated_model.predict_proba(X_test_imp)[:, 1]
y_pred  = (y_probs > opt_threshold).astype(int)
y_true  = y_test.values

# ── 5. MÉTRICAS GLOBALES ──────────────────────────────────────────────────────
auroc = roc_auc_score(y_test, y_probs)
auprc = average_precision_score(y_test, y_probs)
f1    = f1_score(y_test, y_pred)

print("=" * 55)
print("MÉTRICAS GLOBALES v2 (con correcciones)")
print("=" * 55)
print(f"AUROC: {auroc:.4f}  (antes: 0.7640)")
print(f"AUPRC: {auprc:.4f}  (antes: 0.3164)")
print(f"F1:    {f1:.4f}  (antes: val 0.3979)")
print()
print(classification_report(y_test, y_pred))

# ── 6. FP POR FRANJA DE EDAD ──────────────────────────────────────────────────
tp_mask = (y_true == 1) & (y_pred == 1)
tn_mask = (y_true == 0) & (y_pred == 0)
fp_mask = (y_true == 0) & (y_pred == 1)
fn_mask = (y_true == 1) & (y_pred == 0)

bins_age   = [0, 50, 60, 70, 80, 90, 120]
labels_age = ["<50", "50-59", "60-69", "70-79", "80-89", "90+"]
fp_ages    = age_test[fp_mask]
all_neg    = age_test[y_true == 0]

# Tasas anteriores para comparar
prev_fp_rates = {
    "<50":   3.9, "50-59":  8.7, "60-69": 11.1,
    "70-79": 19.8, "80-89": 27.2, "90+":   39.3
}

print("=" * 65)
print(f"{'Franja':<8} {'FP':>5} {'Vivos':>6} {'Tasa v2':>9} {'Tasa v1':>9} {'Δ':>7}")
print("=" * 65)
for label, lo, hi in zip(labels_age, bins_age[:-1], bins_age[1:]):
    n_fp  = ((fp_ages  >= lo) & (fp_ages  < hi)).sum()
    n_neg = ((all_neg  >= lo) & (all_neg  < hi)).sum()
    rate  = n_fp / n_neg * 100 if n_neg > 0 else 0
    delta = rate - prev_fp_rates[label]
    print(f"  {label:<8} {n_fp:>5} {n_neg:>6} {rate:>8.1f}% {prev_fp_rates[label]:>8.1f}% {delta:>+7.1f}%")

# ── 7. GRÁFICA COMPARATIVA ────────────────────────────────────────────────────
franjas = labels_age
v1 = [3.9, 8.7, 11.1, 19.8, 27.2, 39.3]
v2 = []
for label, lo, hi in zip(labels_age, bins_age[:-1], bins_age[1:]):
    n_fp  = ((fp_ages >= lo) & (fp_ages < hi)).sum()
    n_neg = ((all_neg >= lo) & (all_neg < hi)).sum()
    v2.append(n_fp / n_neg * 100 if n_neg > 0 else 0)

x = np.arange(len(franjas))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, v1, width, label="v1 (original)", color="tomato",    alpha=0.8)
ax.bar(x + width/2, v2, width, label="v2 (corregido)", color="steelblue", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(franjas)
ax.set_ylabel("Tasa de Falsos Positivos (%)")
ax.set_title("Tasa FP por franja de edad: v1 vs v2")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for i, (a, b) in enumerate(zip(v1, v2)):
    ax.annotate(f"{b:.1f}%", xy=(x[i]+width/2, b+0.3), ha="center", fontsize=8, color="steelblue")
    ax.annotate(f"{a:.1f}%", xy=(x[i]-width/2, a+0.3), ha="center", fontsize=8, color="tomato")
plt.tight_layout()
plt.savefig(f"{output_dir}/fp_age_v1_vs_v2.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"\nGuardado: {output_dir}/fp_age_v1_vs_v2.png")