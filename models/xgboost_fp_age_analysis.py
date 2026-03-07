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

# ── Cargar best_params ────────────────────────────────────────────────────────
with open("F1-optimized/best_params.json") as f:
    saved = json.load(f)
best_params   = saved["params"]
opt_threshold = saved["optimal_threshold_f1"]

# ── 1. CARGA Y PREPARACIÓN ────────────────────────────────────────────────────
df = pd.read_csv("../datasets/v1/dataset_v1_clean.csv")
df["target"] = df["dod_within_30_days"].notnull().astype(int)
drop_cols = ["target", "dod_within_30_days", "subject_id", "admission_id", "stay_id", "is_dnr"]

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
    map_score   = pd.cut(df["mean_arterial_pressure"], bins=[-np.inf,65,70,np.inf], labels=[2,1,0]).astype(float)
    resp_score  = np.where(df["min_oxygen_saturation"]<90,2,np.where(df["min_oxygen_saturation"]<94,1,0))
    renal_score = df["is_crrt"] * 3
    df["sofa_partial"] = map_score.fillna(0) + resp_score + renal_score
    return df

df_eng = add_engineered_features(df)
feature_cols = [c for c in df_eng.columns if c not in drop_cols + ["dod_within_30_days","target"]]
X = df_eng[feature_cols]
y = df_eng["target"]

# Guardamos anchor_age original para el análisis (alineado con el índice del test set)
age_series = df_eng["anchor_age"]

# ── 2. SPLITS ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

age_test = age_series.loc[X_test.index]

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

# ── 3. MODELO ─────────────────────────────────────────────────────────────────
base_model = xgb.XGBClassifier(**best_params)
base_model.fit(X_train_imp, y_train_fit)
calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
calibrated_model.fit(X_val_imp, y_val)

y_probs = calibrated_model.predict_proba(X_test_imp)[:, 1]
y_pred  = (y_probs > opt_threshold).astype(int)
y_true  = y_test.values
age_arr = age_test.values

# ── 4. SEGMENTAR POR CUADRANTE ────────────────────────────────────────────────
tp_mask = (y_true == 1) & (y_pred == 1)
tn_mask = (y_true == 0) & (y_pred == 0)
fp_mask = (y_true == 0) & (y_pred == 1)
fn_mask = (y_true == 1) & (y_pred == 0)

results = {
    "Verdaderos Positivos (TP)": age_arr[tp_mask],
    "Verdaderos Negativos (TN)": age_arr[tn_mask],
    "Falsos Positivos (FP)":     age_arr[fp_mask],
    "Falsos Negativos (FN)":     age_arr[fn_mask],
}

print("=" * 55)
print(f"{'Grupo':<30} {'N':>5} {'Media':>7} {'Mediana':>8} {'P25':>6} {'P75':>6}")
print("=" * 55)
for name, ages in results.items():
    print(f"{name:<30} {len(ages):>5} {ages.mean():>7.1f} {np.median(ages):>8.1f} "
          f"{np.percentile(ages,25):>6.1f} {np.percentile(ages,75):>6.1f}")

# ── 5. DISTRIBUCIÓN DE EDAD EN FP vs RESTO ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma comparativo
bins = range(15, 100, 5)
axes[0].hist(age_arr[tn_mask], bins=bins, alpha=0.5, label="TN (vive, predice vida)", color="steelblue", density=True)
axes[0].hist(age_arr[fp_mask], bins=bins, alpha=0.6, label="FP (vive, predice muerte)", color="tomato", density=True)
axes[0].axvline(age_arr[fp_mask].mean(), color="red",  linestyle="--", linewidth=1.5, label=f"Media FP = {age_arr[fp_mask].mean():.1f}")
axes[0].axvline(age_arr[tn_mask].mean(), color="blue", linestyle="--", linewidth=1.5, label=f"Media TN = {age_arr[tn_mask].mean():.1f}")
axes[0].set_xlabel("Edad"); axes[0].set_ylabel("Densidad")
axes[0].set_title("Distribución de edad: FP vs TN")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Boxplot de los 4 grupos
axes[1].boxplot([age_arr[tp_mask], age_arr[tn_mask], age_arr[fp_mask], age_arr[fn_mask]],
                labels=["TP", "TN", "FP", "FN"],
                patch_artist=True,
                boxprops=dict(facecolor="lightgray"))
axes[1].set_ylabel("Edad")
axes[1].set_title("Boxplot edad por cuadrante")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("F1-optimized/fp_age_analysis.png", dpi=200, bbox_inches="tight")
plt.close()

# ── 6. FP POR FRANJA DE EDAD ──────────────────────────────────────────────────
print("\nFalsos Positivos por franja de edad:")
print("-" * 40)
bins_age = [0, 50, 60, 70, 80, 90, 120]
labels_age = ["<50", "50-59", "60-69", "70-79", "80-89", "90+"]
fp_ages = age_arr[fp_mask]
all_neg_ages = age_arr[(y_true == 0)]  # todos los que realmente viven

for i, (lo, hi) in enumerate(zip(bins_age[:-1], bins_age[1:])):
    n_fp  = ((fp_ages >= lo) & (fp_ages < hi)).sum()
    n_neg = ((all_neg_ages >= lo) & (all_neg_ages < hi)).sum()
    rate  = n_fp / n_neg * 100 if n_neg > 0 else 0
    print(f"  {labels_age[i]:<8} FP={n_fp:>4}  total_vivos={n_neg:>5}  tasa_FP={rate:.1f}%")

print("\nGuardado: F1-optimized/fp_age_analysis.png")
