import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
import xgboost as xgb

# 1. Carga de datos
df = pd.read_csv('../datasets/v1/dataset_v1.csv')

# 2. Variable Objetivo
df['target'] = df['dod_within_30_days'].notnull().astype(int)

# 3. Selección de Features
features = [
    'is_mechanical_ventilation', 'is_crrt', 'has_arterial_line', 'is_intubated', 
    'is_prone_position', 'has_chest_tube', 'is_dnr', 'has_central_line', 
    'has_hemodialysis', 'has_niv', 'is_ecmo', 'length_of_stay', 'min_bp_systolic', 
    'min_bp_diastolic', 'min_oxygen_saturation', 'max_pulse', 'max_temp', 'max_glucose',
    'anchor_age', 'gender', 'marital_status', 'race', 'max_lactate'
]

X = df[features].copy()
y = df['target']

# Preprocesamiento de categóricas
X = pd.get_dummies(X, columns=['gender', 'marital_status', 'race'], drop_first=True)

X.columns = X.columns.str.replace('[', '_', regex=False)
X.columns = X.columns.str.replace(']', '_', regex=False)
X.columns = X.columns.str.replace('<', '_', regex=False)
X.columns = X.columns.str.replace('>', '_', regex=False)
X.columns = X.columns.str.replace('"', '_', regex=False)
X.columns = X.columns.str.replace(',', '_', regex=False)
X.columns = X.columns.str.replace(':', '_', regex=False)

# 4. División y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calcular el ratio de clases
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# AutoML enfocado en XGBoost
automl = AutoML()
automl.fit(
    X_train, y_train, 
    task="classification",
    metric="ap",
    time_budget=600,
    estimator_list=['xgboost'],
    eval_method="cv",
    n_splits=5,
    early_stop=True,
    n_jobs=-1,
    verbose=2
)

print("\n--- Mejor modelo AutoML ---")
print(f"Algoritmo: {automl.best_estimator}")
print(f"Hiperparámetros: {automl.best_config}")
print(f"AUPRC en validación: {-automl.best_loss:.4f}")

# Ahora entrenar modelo baseline con scale_pos_weight para comparar
print("\n--- Entrenando Baseline con scale_pos_weight ---")
model_baseline = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.01,
    scale_pos_weight=ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='aucpr',
    early_stopping_rounds=100
)
model_baseline.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# 5. Evaluación AutoML
y_probs_auto = automl.predict_proba(X_test)[:, 1]
y_probs_base = model_baseline.predict_proba(X_test)[:, 1]

# 6. Umbral óptimo para AutoML
thresholds = np.linspace(0, 1, 100)
f1_scores_auto = [f1_score(y_test, y_probs_auto >= t) for t in thresholds]
opt_threshold_auto = thresholds[np.argmax(f1_scores_auto)]

# Umbral óptimo para Baseline
f1_scores_base = [f1_score(y_test, y_probs_base >= t) for t in thresholds]
opt_threshold_base = thresholds[np.argmax(f1_scores_base)]

print(f"\n--- COMPARACIÓN FINAL ---")
print(f"\nAutoML (umbral={opt_threshold_auto:.2f}):")
print(f"  AUROC: {roc_auc_score(y_test, y_probs_auto):.4f}")
print(f"  AUPRC: {average_precision_score(y_test, y_probs_auto):.4f}")

print(f"\nBaseline con scale_pos_weight (umbral={opt_threshold_base:.2f}):")
print(f"  AUROC: {roc_auc_score(y_test, y_probs_base):.4f}")
print(f"  AUPRC: {average_precision_score(y_test, y_probs_base):.4f}")

# Classification report del mejor
if average_precision_score(y_test, y_probs_auto) > average_precision_score(y_test, y_probs_base):
    print("\n--- Métricas AutoML (GANADOR) ---")
    y_pred_opt = (y_probs_auto >= opt_threshold_auto).astype(int)
else:
    print("\n--- Métricas Baseline (GANADOR) ---")
    y_pred_opt = (y_probs_base >= opt_threshold_base).astype(int)

print(classification_report(y_test, y_pred_opt))