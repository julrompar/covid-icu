import pandas as pd

df = pd.read_csv('dataset_v1.csv')


df = df.drop(columns=['length_of_stay','min_bp_systolic_line', 'min_bp_diastolic_line', 'max_lactate'])

df.to_csv('./dataset_v1_clean.csv', index=False)
df.to_parquet('./dataset_v1_clean.parquet', index=False, compression='snappy')