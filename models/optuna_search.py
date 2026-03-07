import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score, make_scorer
import optuna
from optuna.samplers import TPESampler

# Cargar dataset con feature engineering
df = pd.read_csv('../datasets/v1/dataset_v1_engineered.csv')

df['target'] = df['dod_within_30_days'].notnull().astype(int)

# Usar TODOS los features
features = [col for col in df.columns if col not in ['target', 'dod_within_30_days', 'subject_id', 'hadm_id', 'stay_id']]

X = df[features].copy()
y = df['target']

# Preprocesamiento
X = pd.get_dummies(X, columns=['gender', 'marital_status', 'race'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# Función objetivo para Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 10, log=True),
        'scale_pos_weight': ratio,
        'random_state': 42,
        'eval_metric': 'aucpr'
    }
    
    model = xgb.XGBClassifier(**params)
    scorer = make_scorer(average_precision_score, response_method='predict_proba')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1)
    
    return scores.mean()

print("🚀 Optimizando XGBoost con FEATURE ENGINEERING...")
print("Esto tomará ~10-15 minutos...\n")

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study.optimize(objective, n_trials=150, show_progress_bar=True)  # Más trials

print("\n--- MEJORES HIPERPARÁMETROS (con Feature Engineering) ---")
print(study.best_params)
print(f"AUPRC en validación: {study.best_value:.4f}")

# Entrenar modelo final optimizado
best_params = study.best_params
best_params['scale_pos_weight'] = ratio
best_params['random_state'] = 42
best_params['eval_metric'] = 'aucpr'

model_optimized = xgb.XGBClassifier(**best_params)
model_optimized.fit(X_train, y_train)

# Baseline original (sin feature engineering)
model_baseline = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.01,
    scale_pos_weight=ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='aucpr'
)

# Cargar datos originales para baseline
df_original = pd.read_csv('../datasets/v1/dataset_v1.csv')
df_original['target'] = df_original['dod_within_30_days'].notnull().astype(int)
features_original = [
    'is_mechanical_ventilation', 'is_crrt', 'has_arterial_line', 'is_intubated', 
    'is_prone_position', 'has_chest_tube', 'is_dnr', 'has_central_line', 
    'has_hemodialysis', 'has_niv', 'is_ecmo', 'length_of_stay', 'min_bp_systolic', 
    'min_bp_diastolic', 'min_oxygen_saturation', 'max_pulse', 'max_temp', 'max_glucose',
    'anchor_age', 'gender', 'marital_status', 'race', 'max_lactate'
]
X_original = df_original[features_original].copy()
X_original = pd.get_dummies(X_original, columns=['gender', 'marital_status', 'race'], drop_first=True)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_original, df_original['target'], test_size=0.2, random_state=42, stratify=df_original['target']
)
model_baseline.fit(X_train_orig, y_train_orig)

# Evaluación
y_probs_opt = model_optimized.predict_proba(X_test)[:, 1]
y_probs_base = model_baseline.predict_proba(X_test_orig)[:, 1]

# Umbrales óptimos
thresholds = np.linspace(0, 1, 100)
f1_scores_opt = [f1_score(y_test, y_probs_opt >= t) for t in thresholds]
opt_threshold_opt = thresholds[np.argmax(f1_scores_opt)]

f1_scores_base = [f1_score(y_test_orig, y_probs_base >= t) for t in thresholds]
opt_threshold_base = thresholds[np.argmax(f1_scores_base)]

print(f"\n{'='*60}")
print(f"{'COMPARACIÓN FINAL':^60}")
print(f"{'='*60}")

print(f"\n📊 BASELINE Original (sin feature engineering):")
print(f"   AUROC: {roc_auc_score(y_test_orig, y_probs_base):.4f}")
print(f"   AUPRC: {average_precision_score(y_test_orig, y_probs_base):.4f}")
print(f"   Umbral: {opt_threshold_base:.2f}")

print(f"\n🚀 OPTIMIZADO (con feature engineering + Optuna):")
print(f"   AUROC: {roc_auc_score(y_test, y_probs_opt):.4f}")
print(f"   AUPRC: {average_precision_score(y_test, y_probs_opt):.4f}")
print(f"   Umbral: {opt_threshold_opt:.2f}")

mejora_auroc = (roc_auc_score(y_test, y_probs_opt) - 0.8185) / 0.8185 * 100
mejora_auprc = (average_precision_score(y_test, y_probs_opt) - 0.4007) / 0.4007 * 100

print(f"\n📈 MEJORA:")
print(f"   AUROC: +{mejora_auroc:.2f}%")
print(f"   AUPRC: +{mejora_auprc:.2f}%")

# Mostrar métricas detalladas del mejor
y_pred_opt = (y_probs_opt >= opt_threshold_opt).astype(int)
print(f"\n{'='*60}")
print(f"{'MÉTRICAS DETALLADAS DEL MODELO OPTIMIZADO':^60}")
print(f"{'='*60}")
print(classification_report(y_test, y_pred_opt))

# Guardar modelo final
import pickle
with open('best_model_final.pkl', 'wb') as f:
    pickle.dump(model_optimized, f)
print("\n✅ Modelo final guardado en: best_model_final.pkl")