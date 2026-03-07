import pandas as pd
from tableone import TableOne
import matplotlib.pyplot as plt
import dataframe_image as dfi

df = pd.read_csv('dataset_v1.csv')

df['mortality_30d'] = df['dod_within_30_days'].notna().astype(int)
df['mortality_status'] = df['mortality_30d'].map({1: 'Fallecidos', 0: 'No fallecidos'})

# Definir variables continuas
continuous_vars = [
    'length_of_stay',
    'min_bp_systolic_line',
    'min_bp_diastolic_line',
    'min_bp_systolic',
    'min_bp_diastolic',
    'min_oxygen_saturation',
    'max_pulse',
    'max_temp',
    'max_glucose',
    'max_lactate',
    'anchor_age'
]

# Definir variables categóricas (solo las no binarias para Table One)
categorical_vars = [
    'gender'
]

# Combinar todas las columnas para la tabla
columns = continuous_vars + categorical_vars

# Crear Table One estratificada por mortalidad a 30 días
mytable = TableOne(
    df,
    columns=columns,
    categorical=categorical_vars,
    groupby='mortality_status',
    pval=True,
    missing=True,
    overall=True,
    label_suffix=False
)


print("=" * 80)
print("TABLE ONE - COVID-19 ICU COHORT")
print("Stratified by 30-day mortality")
print("=" * 80)
print(mytable.tabulate(tablefmt='grid'))


print("\n" + "-" * 80)
print("DEMOGRAPHIC VARIABLES (Missing Data Summary)")
print("-" * 80)


race_missing = df['race'].isna().sum()
race_available = df['race'].notna().sum()
marital_missing = df['marital_status'].isna().sum()
marital_available = df['marital_status'].notna().sum()

race_missing_fallecidos = df[df['mortality_status'] == 'Fallecidos']['race'].isna().sum()
race_missing_no_fallecidos = df[df['mortality_status'] == 'No fallecidos']['race'].isna().sum()
marital_missing_fallecidos = df[df['mortality_status'] == 'Fallecidos']['marital_status'].isna().sum()
marital_missing_no_fallecidos = df[df['mortality_status'] == 'No fallecidos']['marital_status'].isna().sum()

print(f"\nRace:")
print(f"  Overall: Missing = {race_missing} ({race_missing/len(df)*100:.1f}%), Available = {race_available}")
print(f"  Fallecidos: Missing = {race_missing_fallecidos}")
print(f"  No fallecidos: Missing = {race_missing_no_fallecidos}")

print(f"\nMarital Status:")
print(f"  Overall: Missing = {marital_missing} ({marital_missing/len(df)*100:.1f}%), Available = {marital_available}")
print(f"  Fallecidos: Missing = {marital_missing_fallecidos}")
print(f"  No fallecidos: Missing = {marital_missing_no_fallecidos}")
print("-" * 80)


mytable.to_csv('table_one_output.csv')
print("\n✓ Table One saved to: table_one_output.csv")


mytable.to_html('table_one_output.html')
print("✓ Table One saved to: table_one_output.html")


print("\nGenerating image output...")

table_df = mytable.tableone


styled_df = table_df.style.set_properties(**{
    'text-align': 'left',
    'font-size': '10pt',
    'border': '1px solid black'
}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#4CAF50'), 
                                   ('color', 'white'), 
                                   ('font-weight', 'bold'),
                                   ('text-align', 'center'),
                                   ('border', '1px solid black')]},
    {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
    {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#f2f2f2')]},
])


dfi.export(styled_df, 'table_one_output.png', max_rows=-1, max_cols=-1, table_conversion='matplotlib')
print("✓ Table One saved to: table_one_output.png")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total patients: {len(df)}")
print(f"Deaths within 30 days: {df['mortality_30d'].sum()} ({df['mortality_30d'].mean()*100:.2f}%)")
print(f"Survivors: {len(df) - df['mortality_30d'].sum()} ({(1-df['mortality_30d'].mean())*100:.2f}%)")
print("=" * 80)

print("\n" + "=" * 80)
print("TABLE TWO - BINARY VARIABLES DESCRIPTION")
print("Data availability and distribution (0 vs 1)")
print("=" * 80)


binary_vars = [
    'is_mechanical_ventilation',
    'is_crrt',
    'has_arterial_line',
    'is_intubated',
    'is_prone_position',
    'has_chest_tube',
    'is_dnr',
    'has_central_line',
    'has_hemodialysis',
    'has_niv',
    'is_ecmo'
]


binary_stats = []
for var in binary_vars:
    total = len(df)
    non_missing = df[var].notna().sum()
    missing = df[var].isna().sum()
    count_0 = (df[var] == 0).sum()
    count_1 = (df[var] == 1).sum()
    pct_0 = (count_0 / non_missing * 100) if non_missing > 0 else 0
    pct_1 = (count_1 / non_missing * 100) if non_missing > 0 else 0
    
    binary_stats.append({
        'Variable': var,
        'Total': total,
        'Available': non_missing,
        'Missing': missing,
        'Count_0': count_0,
        'Percent_0': f"{pct_0:.1f}%",
        'Count_1': count_1,
        'Percent_1': f"{pct_1:.1f}%"
    })

binary_df = pd.DataFrame(binary_stats)

print(binary_df.to_string(index=False))
print("=" * 80)


binary_df.to_csv('table_two_binary_variables.csv', index=False)
print("\n✓ Binary variables table saved to: table_two_binary_variables.csv")


styled_binary = binary_df.style.set_properties(**{
    'text-align': 'center',
    'font-size': '10pt',
    'border': '1px solid black'
}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#2196F3'), 
                                   ('color', 'white'), 
                                   ('font-weight', 'bold'),
                                   ('text-align', 'center'),
                                   ('border', '1px solid black')]},
    {'selector': 'td', 'props': [('border', '1px solid #ddd')]},
    {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', '#f2f2f2')]},
])

dfi.export(styled_binary, 'table_two_binary_variables.png', max_rows=-1, max_cols=-1, table_conversion='matplotlib')
print("✓ Binary variables table saved to: table_two_binary_variables.png")
print("=" * 80)
