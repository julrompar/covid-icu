import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

os.makedirs('datasets/v3', exist_ok=True)

print("Cargando dataset base...")
# El dataset v3_all_stays.parquet fue generado por notebooks/dataset.py y contiene todas las estancias
df = pd.read_parquet('data/processed/covid_icu_dataset_v3_all_stays.parquet')

lab_cols = [
    'max_lactate', 'max_creatinine', 'max_bilirubin', 'min_platelets',
    'max_bun', 'max_pt', 'max_crp', 'max_ferritin', 'max_dimer',
    'min_lymphocytes', 'max_neutrophils', 'max_troponin', 'min_pao2'
]

print("Paso 2.2 — Crear variables indicadoras de missingness")
for col in lab_cols:
    if col in df.columns:
        df[f'{col}_measured'] = df[col].notna().astype(int)

print("Paso 2.3 — Imputar valores faltantes según umbral de missingness")
missing_rates = df[lab_cols].isna().mean()
cols_to_impute = missing_rates[missing_rates < 0.70].index.tolist()
cols_indicator_only = missing_rates[missing_rates >= 0.70].index.tolist()

print(f"Imputar con mediana: {cols_to_impute}")
print(f"Solo indicadora (>70% missing): {cols_indicator_only}")

# Imputación (Nota: Aunque la guía indica fit en train y transform en test, 
# la generación del dataset consolidado requiere hacerlo de manera global para
# calcular el severity score y permitir la compactación por estancia).
imputer = SimpleImputer(strategy='median')
if cols_to_impute:
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

# Eliminar columna numérica de las features con >70% missing
df.drop(columns=cols_indicator_only, inplace=True, errors='ignore')

print("Paso 2.4 — Calcular el score de severidad compuesto")
def compute_severity_score(row):
    components = []
    weights = []

    # Respiratorio (peso 3)
    if pd.notna(row.get('min_oxygen_saturation')):
        spo2 = np.clip(row['min_oxygen_saturation'], 0, 100)
        components.append((100 - spo2) / 100)
        weights.append(3)

    if pd.notna(row.get('min_pao2')) and 'min_pao2' not in cols_indicator_only:
        pf_norm = np.clip(1 - (row['min_pao2'] / 500), 0, 1)
        components.append(pf_norm)
        weights.append(3)

    if pd.notna(row.get('max_resp_rate')):
        rr_norm = np.clip((row['max_resp_rate'] - 12) / 28, 0, 1)
        components.append(rr_norm)
        weights.append(2)

    # Hemodinámico (peso 2)
    if pd.notna(row.get('min_bp_systolic')):
        sbp_norm = np.clip(1 - (row['min_bp_systolic'] / 120), 0, 1)
        components.append(sbp_norm)
        weights.append(2)

    # Metabólico / Perfusión (peso 3)
    if pd.notna(row.get('max_lactate')) and 'max_lactate' not in cols_indicator_only:
        lactate_norm = np.clip(row['max_lactate'] / 10, 0, 1)
        components.append(lactate_norm)
        weights.append(3)

    # Renal (peso 2)
    if pd.notna(row.get('max_creatinine')) and 'max_creatinine' not in cols_indicator_only:
        creat_norm = np.clip(row['max_creatinine'] / 10, 0, 1)
        components.append(creat_norm)
        weights.append(2)

    # Hepático (peso 1)
    if pd.notna(row.get('max_bilirubin')) and 'max_bilirubin' not in cols_indicator_only:
        bili_norm = np.clip(row['max_bilirubin'] / 20, 0, 1)
        components.append(bili_norm)
        weights.append(1)

    # Coagulación (peso 2)
    if pd.notna(row.get('min_platelets')) and 'min_platelets' not in cols_indicator_only:
        plt_norm = np.clip(1 - (row['min_platelets'] / 400), 0, 1)
        components.append(plt_norm)
        weights.append(2)

    if pd.notna(row.get('max_dimer')) and 'max_dimer' not in cols_indicator_only:
        dimer_norm = np.clip(row['max_dimer'] / 10, 0, 1)
        components.append(dimer_norm)
        weights.append(2)

    # Inflamación COVID (peso 2)
    if pd.notna(row.get('max_crp')) and 'max_crp' not in cols_indicator_only:
        crp_norm = np.clip(row['max_crp'] / 300, 0, 1)
        components.append(crp_norm)
        weights.append(2)

    if pd.notna(row.get('min_lymphocytes')) and 'min_lymphocytes' not in cols_indicator_only:
        lymph_norm = np.clip(1 - (row['min_lymphocytes'] / 2000), 0, 1)
        components.append(lymph_norm)
        weights.append(2)

    if pd.notna(row.get('max_ferritin')) and 'max_ferritin' not in cols_indicator_only:
        ferritin_norm = np.clip(row['max_ferritin'] / 5000, 0, 1)
        components.append(ferritin_norm)
        weights.append(2)

    # Daño cardiaco (peso 2)
    if pd.notna(row.get('max_troponin')) and 'max_troponin' not in cols_indicator_only:
        trop_norm = np.clip(row['max_troponin'] / 10, 0, 1)
        components.append(trop_norm)
        weights.append(2)

    # Procedimientos invasivos (ya presentes en dataset)
    proc_weights = {
        'is_ecmo': 5,
        'is_mechanical_ventilation': 4,
        'is_prone_position': 3,
        'is_crrt': 3,
        'has_hemodialysis': 3,
        'is_intubated': 2,
    }
    for proc, w in proc_weights.items():
        if proc in row.index and pd.notna(row[proc]):
            components.append(float(row[proc]))
            weights.append(w)

    if not components:
        return np.nan

    return np.average(components, weights=weights)

df['severity_score'] = df.apply(compute_severity_score, axis=1)

cols_to_drop = ['min_bp_systolic_line', 'min_bp_diastolic_line', 'length_of_stay', 'is_dnr']
print(f"Eliminando columnas ruidosas: {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print("Compactación Opción C: última estancia (admittime DESC)")
df_optionC = df.sort_values(by=['subject_id', 'admittime'], ascending=[True, False]).drop_duplicates(subset=['subject_id'], keep='first')
df_optionC.to_parquet('datasets/v3/dataset_v3_optionC.parquet', index=False, compression='snappy')
df_optionC.to_csv('datasets/v3/dataset_v3_optionC.csv', index=False)
print(f"Dataset Option C guardado con {len(df_optionC)} registros.")

print("Compactación Opción B: max severity_score")
df_optionB = df.sort_values(by=['subject_id', 'severity_score'], ascending=[True, False]).drop_duplicates(subset=['subject_id'], keep='first')
df_optionB.to_parquet('datasets/v3/dataset_v3_optionB.parquet', index=False, compression='snappy')
df_optionB.to_csv('datasets/v3/dataset_v3_optionB.csv', index=False)
print(f"Dataset Option B guardado con {len(df_optionB)} registros.")

print("Generación completada en datasets/v3/")
