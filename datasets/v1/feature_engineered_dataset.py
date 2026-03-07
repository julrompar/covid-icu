import pandas as pd
import numpy as np

df = pd.read_csv('./dataset_v1.csv')

# Features actuales
features_base = [
    'is_mechanical_ventilation', 'is_crrt', 'has_arterial_line', 'is_intubated', 
    'is_prone_position', 'has_chest_tube', 'is_dnr', 'has_central_line', 
    'has_hemodialysis', 'has_niv', 'is_ecmo', 'length_of_stay', 'min_bp_systolic', 
    'min_bp_diastolic', 'min_oxygen_saturation', 'max_pulse', 'max_temp', 'max_glucose',
    'anchor_age', 'gender', 'marital_status', 'race', 'max_lactate'
]

# NUEVOS FEATURES - Interacciones clínicas importantes
df['severity_score'] = (
    df['is_mechanical_ventilation'].astype(int) * 3 +
    df['is_crrt'].astype(int) * 2 +
    df['is_ecmo'].astype(int) * 4 +
    df['is_intubated'].astype(int) * 2 +
    df['is_prone_position'].astype(int) * 1
)

# Ratio de presión arterial (indicador de shock)
df['bp_ratio'] = df['min_bp_systolic'] / (df['min_bp_diastolic'] + 1)

# Índice de oxigenación
df['oxygenation_index'] = df['min_oxygen_saturation'] / (df['max_pulse'] + 1)

# Edad ajustada por comorbilidades
df['age_severity'] = df['anchor_age'] * (1 + df['severity_score'] / 10)

# Interacción lactato-ventilación (muy predictivo de mortalidad)
df['lactate_ventilation'] = df['max_lactate'] * df['is_mechanical_ventilation'].astype(int)

# Tiempo de estancia crítico (>7 días)
df['prolonged_stay'] = (df['length_of_stay'] > 7).astype(int)

# Shock séptico (lactato alto + hipotensión)
df['septic_shock_indicator'] = (
    (df['max_lactate'] > 2) & 
    (df['min_bp_systolic'] < 90)
).astype(int)

# Fallo multiorgánico
df['multi_organ_failure'] = (
    df['is_crrt'].astype(int) + 
    df['is_mechanical_ventilation'].astype(int) + 
    df['is_ecmo'].astype(int)
)

# FEATURES FINALES
features_engineered = features_base + [
    'severity_score',
    'bp_ratio', 
    'oxygenation_index',
    'age_severity',
    'lactate_ventilation',
    'prolonged_stay',
    'septic_shock_indicator',
    'multi_organ_failure'
]

print(f"Features originales: {len(features_base)}")
print(f"Features nuevos: {len(features_engineered)}")
print(f"\nNuevos features agregados:")
for f in features_engineered:
    if f not in features_base:
        print(f"  - {f}")

# Guardar dataset mejorado
df.to_csv('./dataset_v1_engineered.csv', index=False)
print(f"\n✅ Dataset guardado en: datasets/v1/dataset_v1_engineered.csv")