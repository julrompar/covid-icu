import pandas as pd
import numpy as np
import os

os.makedirs('datasets/v2', exist_ok=True)

# 1. Base dataset_v2
df_base = pd.read_parquet('data/processed/covid_icu_dataset.parquet')
df_base = df_base.drop(columns=['length_of_stay', 'max_lactate', 'max_glucose'])

df_base.to_csv('datasets/v2/dataset_v2.csv', index=False)
df_base.to_parquet('datasets/v2/dataset_v2.parquet', index=False, compression='snappy')

# 2. Clean dataset_v2_clean
# Also dropping min_bp_systolic_line, min_bp_diastolic_line as in dataset_cleaner.py
df_clean = df_base.drop(columns=['min_bp_systolic_line', 'min_bp_diastolic_line'])
df_clean.to_csv('datasets/v2/dataset_v2_clean.csv', index=False)
df_clean.to_parquet('datasets/v2/dataset_v2_clean.parquet', index=False, compression='snappy')

# 3. Engineered dataset_v2_engineered
features_base = [
    'is_mechanical_ventilation', 'is_crrt', 'has_arterial_line', 'is_intubated', 
    'is_prone_position', 'has_chest_tube', 'is_dnr', 'has_central_line', 
    'has_hemodialysis', 'has_niv', 'is_ecmo', 'min_bp_systolic', 
    'min_bp_diastolic', 'min_oxygen_saturation', 'max_pulse', 'max_temp',
    'anchor_age', 'gender', 'marital_status', 'race'
]

df_eng = df_clean.copy()

# NUEVOS FEATURES - Interacciones clinicas importantes
df_eng['severity_score'] = (
    df_eng['is_mechanical_ventilation'].astype(int) * 3 +
    df_eng['is_crrt'].astype(int) * 2 +
    df_eng['is_ecmo'].astype(int) * 4 +
    df_eng['is_intubated'].astype(int) * 2 +
    df_eng['is_prone_position'].astype(int) * 1
)

df_eng['bp_ratio'] = df_eng['min_bp_systolic'] / (df_eng['min_bp_diastolic'] + 1)
df_eng['oxygenation_index'] = df_eng['min_oxygen_saturation'] / (df_eng['max_pulse'] + 1)
df_eng['age_severity'] = df_eng['anchor_age'] * (1 + df_eng['severity_score'] / 10)

# Replaced septic shock indicator as max_lactate is missing
df_eng['severe_hypotension'] = (df_eng['min_bp_systolic'] < 90).astype(int)

# Fallo multiorganico
df_eng['multi_organ_failure'] = (
    df_eng['is_crrt'].astype(int) + 
    df_eng['is_mechanical_ventilation'].astype(int) + 
    df_eng['is_ecmo'].astype(int)
)

df_eng.to_csv('datasets/v2/dataset_v2_engineered.csv', index=False)
df_eng.to_parquet('datasets/v2/dataset_v2_engineered.parquet', index=False, compression='snappy')
print("Generados datasets v2 correctamente en datasets/v2/")
