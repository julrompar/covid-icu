import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
import matplotlib.pyplot as plt
import shap

# 1. Carga de datos
df = pd.read_csv('/Users/julianromero/Desktop/covid-icu/data/processed/covid_icu_dataset.csv/part-00000-93da85da-fc49-4447-b271-f9767ad4f112-c000.csv') # Cambia al nombre de tu archivo actualizado

# 2. Variable Objetivo
df['target'] = df['dod_within_30_days'].notnull().astype(int)

# 3. Selección de Features (He incluido 'Lactate' asumiendo que ya la tienes)
features = [
    'is_mechanical_ventilation', 'is_crrt', 'has_arterial_line', 'is_intubated', 
    'is_prone_position', 'has_chest_tube', 'is_dnr', 'has_central_line', 
    'has_hemodialysis', 'has_niv', 'is_ecmo', 'length_of_stay', 'min_bp_systolic', 
    'min_bp_diastolic', 'min_oxygen_saturation', 'max_pulse', 'max_temp', 'max_glucose',
    'anchor_age', 'gender', 'marital_status', 'race'
]

# Añade 'Lactate' solo si existe en tu dataframe actual
if 'Lactate' in df.columns:
    features.append('Lactate')

X = df[features].copy()
y = df['target']

# Preprocesamiento de categóricas
X = pd.get_dummies(X, columns=['gender', 'marital_status', 'race'], drop_first=True)

# 4. División y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
#Añadir RandomizedSearch y Cross Validation
#Probar automl/autosklearn

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.01, # Un paso más lento ayuda a mejorar la precisión
    scale_pos_weight=ratio,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='aucpr',
    early_stopping_rounds=100
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# 1. Encontrar el mejor umbral automáticamente
y_probs = model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_test, y_probs >= t) for t in thresholds]
opt_threshold = thresholds[np.argmax(f1_scores)]

print(f"El umbral óptimo sugerido es: {opt_threshold:.2f}")

# 2. Aplicar el nuevo umbral
y_pred_opt = (y_probs >= opt_threshold).astype(int)

print(f"\n--- Métricas con Umbral Optimizado ({opt_threshold:.2f}) ---")
print(f"AUROC: {roc_auc_score(y_test, y_probs):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_probs):.4f}")
print(classification_report(y_test, y_pred_opt))

# 3. ÚNICA GRÁFICA: SHAP Summary Plot
# Explicación visual de por qué el modelo toma estas decisiones
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(12, 8))
# El summary_plot de SHAP es la "joya de la corona" para explicar modelos médicos
shap.summary_plot(shap_values, X_test, show=False)
plt.title(f"Importancia Clínica de las Variables (SHAP Values)\nAUROC: {roc_auc_score(y_test, y_probs):.4f}")
plt.tight_layout()
plt.show()