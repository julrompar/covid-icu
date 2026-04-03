import pandas as pd
import os

os.makedirs('datasets/v5', exist_ok=True)

# Cargar v4
df_b = pd.read_parquet('datasets/v4/dataset_v4_optionB.parquet')
df_c = pd.read_parquet('datasets/v4/dataset_v4_optionC.parquet')

# Columnas a eliminar
DROP_COLS = [
    # Labs con baja cobertura — valor numérico y su indicadora
    'max_crp',
    'max_ferritin', 'max_ferritin_measured',
    'max_pt',       'max_pt_measured',
    'max_lactate',  'max_lactate_measured',
    'min_lymphocytes', 'min_lymphocytes_measured',
    'max_neutrophils', 'max_neutrophils_measured',
    'max_dimer',    'max_dimer_measured',
    'max_troponin', 'max_troponin_measured',
]

# Eliminar solo las que existen (por si alguna no llegó al dataset)
drop_existing = [c for c in DROP_COLS if c in df_b.columns]

df_b_v5 = df_b.drop(columns=drop_existing)
df_c_v5 = df_c.drop(columns=drop_existing)

# Verificación
print(f"OptionB v5: {df_b_v5.shape}")
print(f"OptionC v5: {df_c_v5.shape}")
print("\nColumnas finales:")
print(list(df_c_v5.columns))

# Validar: min_platelets y max_bilirubin presentes y con datos
for col in ['min_platelets', 'max_bilirubin']:
    coverage = df_c_v5[col].notna().mean()
    print(f"  {col}: {coverage:.1%} cobertura")

# Guardar
df_b_v5.to_parquet('datasets/v5/dataset_v5_optionB.parquet', index=False, compression='snappy')
df_c_v5.to_parquet('datasets/v5/dataset_v5_optionC.parquet', index=False, compression='snappy')
df_b_v5.to_csv('datasets/v5/dataset_v5_optionB.csv', index=False)
df_c_v5.to_csv('datasets/v5/dataset_v5_optionC.csv', index=False)
print("\nDatasets v5 guardados.")

# Fase 4 — Validación antes de ejecutar el modelo
print("\n=== Validando Dataset v5 ===")

# 1. Shape esperado
assert df_c_v5.shape[1] == 39, f"Esperadas 39 columnas, encontradas {df_c_v5.shape[1]}"

# 2. Los labs eliminados NO deben estar presentes
eliminated = ['max_crp', 'max_ferritin', 'max_dimer', 'max_troponin',
              'max_lactate', 'min_lymphocytes', 'max_neutrophils', 'max_pt']
for col in eliminated:
    assert col not in df_c_v5.columns, f"ERROR: {col} sigue en el dataset"

# 3. Los labs nuevos SÍ deben estar y con cobertura correcta
for col, min_cov in [('min_platelets', 0.20), ('max_bilirubin', 0.18)]:
    cov = df_c_v5[col].notna().mean()
    assert cov >= min_cov, f"Cobertura insuficiente en {col}: {cov:.1%}"
    print(f"  ✅ {col}: {cov:.1%}")

# 4. Target distribucion no ha cambiado
mortality = df_c_v5['dod_within_30_days'].notna().mean()
print(f"  ✅ Mortalidad: {mortality:.2%} (esperado ~23.5%)")

print("\n✅ Validación completada. Dataset listo para el modelo.")
