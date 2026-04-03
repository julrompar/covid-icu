#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para generar gráficas descriptivas del dataset v5:
1. Diagrama de flujo de criterios de inclusión (CONSORT-style)
2. Table-1 con características demográficas y clínicas
3. Tabla de pacientes con múltiples estancias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar datasets
print("Cargando datasets...")
# Dataset base con todas las estancias (mismo que se usa para generar v4/v5)
df_all_stays = pd.read_parquet('data/processed/covid_icu_dataset_v3_all_stays.parquet')
df_v5_optionB = pd.read_csv('datasets/v5/dataset_v5_optionB.csv')
df_v5_optionC = pd.read_csv('datasets/v5/dataset_v5_optionC.csv')

# ============================================================================
# 1. DIAGRAMA DE FLUJO DE INCLUSIÓN (CONSORT-style)
# ============================================================================
print("\nGenerando diagrama de flujo de inclusión...")

# Calcular estadísticas para el flujo
n_total_stays = len(df_all_stays)
n_unique_patients = df_all_stays['subject_id'].nunique()
stays_per_patient = df_all_stays.groupby('subject_id').agg(
    num_stays=('stay_id', 'count'),
    first_admit=('admittime', 'min'),
    last_admit=('admittime', 'max')
).reset_index()
n_patients_multiple_stays = (stays_per_patient['num_stays'] > 1).sum()
n_patients_single_stay = n_unique_patients - n_patients_multiple_stays
n_final_optionB = len(df_v5_optionB)
n_final_optionC = len(df_v5_optionC)

# Crear figura del diagrama de flujo - REDISEÑADO para mostrar flujo de múltiples estancias
fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

def draw_box(ax, x, y, width, height, text, color='#E3F2FD', edgecolor='#1976D2', fontsize=10):
    """Dibuja una caja con texto centrado"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor=color, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            wrap=True, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color='#1976D2'):
    """Dibuja una flecha entre dos puntos"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Título
ax.text(5, 11.7, 'Criterios de Inclusión - Dataset v5', 
        ha='center', va='center', fontsize=16, fontweight='bold', color='#0D47A1')

# Caja 1: Dataset MIMIC-IV UCI
draw_box(ax, 5, 10.8, 4.5, 0.7, 
         f'Base de datos MIMIC-IV\nPacientes UCI',
         color='#BBDEFB')

draw_arrow(ax, 5, 10.8 - 0.7/2, 5, 9.4 + 0.9/2)

# Caja 2: Filtro COVID-19
draw_box(ax, 5, 9.4, 5.5, 0.9,
         f'Criterio de inclusión:\nDiagnóstico COVID-19 (ICD-10: U07.1)',
         color='#C8E6C9', edgecolor='#388E3C')

draw_arrow(ax, 5, 9.4 - 0.9/2, 5, 7.7 + 1.0/2)

# Caja 3: Total estancias COVID
draw_box(ax, 5, 7.7, 5, 1.0,
         f'Estancias UCI COVID-19\nn = {n_total_stays:,} estancias\n{n_unique_patients:,} pacientes únicos',
         color='#FFF9C4', edgecolor='#FBC02D')

draw_arrow(ax, 5, 7.7 - 1.0/2, 5, 5.9 + 1.2/2)

# Caja 4: Identificación de múltiples estancias
draw_box(ax, 5, 5.9, 6, 1.2,
         f'Pacientes con múltiples estancias UCI\n'
         f'{n_patients_multiple_stays:,} pacientes ({(n_patients_multiple_stays/n_unique_patients)*100:.1f}%)\n'
         f'Rango: 2-{stays_per_patient["num_stays"].max()} estancias por paciente',
         color='#FFECB3', edgecolor='#FF9800')

draw_arrow(ax, 5, 5.9 - 1.2/2, 5, 4.1 + 0.9/2)

# Caja 5: Compactación necesaria
draw_box(ax, 5, 4.1, 5.5, 0.9,
         f'Compactación requerida:\n1 registro por paciente',
         color='#E1BEE7', edgecolor='#7B1FA2')

# Bifurcación - flechas desde el borde inferior de la caja 5 hacia los bordes superiores de las cajas 6a y 6b
draw_arrow(ax, 5, 4.1 - 0.9/2, 3, 2.5 + 0.9/2)
draw_arrow(ax, 5, 4.1 - 0.9/2, 7, 2.5 + 0.9/2)

# Caja 6a: Opción B - Max Severity
draw_box(ax, 3, 2.5, 3.5, 0.9,
         f'Opción B:\nEstancia con máximo\nSeverity Score',
         color='#D1C4E9', edgecolor='#512DA8', fontsize=9)

draw_arrow(ax, 3, 2.5 - 0.9/2, 3, 0.85 + 0.9/2)

# Caja 7a: Dataset final B
draw_box(ax, 3, 0.85, 3.2, 0.9,
         f'Dataset v5 - Opción B\nn = {n_final_optionB:,} pacientes',
         color='#B39DDB', edgecolor='#4527A0')

# Caja 6b: Opción C - Última estancia
draw_box(ax, 7, 2.5, 3.5, 0.9,
         f'Opción C:\nÚltima estancia UCI\n(más reciente)',
         color='#B2DFDB', edgecolor='#00796B', fontsize=9)

draw_arrow(ax, 7, 2.5 - 0.9/2, 7, 0.85 + 0.9/2)

# Caja 7b: Dataset final C
draw_box(ax, 7, 0.85, 3.2, 0.9,
         f'Dataset v5 - Opción C\nn = {n_final_optionC:,} pacientes',
         color='#80CBC4', edgecolor='#004D40')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'flow_inclusion_criteria.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'flow_inclusion_criteria.pdf'), bbox_inches='tight')
print(f"✓ Diagrama de flujo guardado en {OUTPUT_DIR}/flow_inclusion_criteria.png")

# ============================================================================
# 2. TABLE-1: Características demográficas y clínicas (usando tableone)
# ============================================================================
print("\nGenerando Table-1 con tableone...")

from tableone import TableOne

df = df_v5_optionB.copy()
df['mortality_30d'] = df['dod_within_30_days'].notna().astype(int)
df['mortality_status'] = df['mortality_30d'].map({1: 'Fallecidos', 0: 'Supervivientes'})

# Filtrar outliers usando rangos clínicos razonables
clinical_ranges = {
    'max_glucose': (0, 1000),        # mg/dL
    'max_creatinine': (0, 30),       # mg/dL
    'max_bilirubin': (0, 50),        # mg/dL
    'min_platelets': (0, 1000),      # ×10³/µL
    'max_bun': (0, 200),             # mg/dL
    'min_pao2': (0, 700),            # mmHg
    'min_bp_systolic': (40, 250),    # mmHg
    'min_bp_diastolic': (20, 150),   # mmHg
    'max_pulse': (30, 250),          # lpm
    'max_temp': (90, 110),           # °F
    'max_resp_rate': (5, 80),        # rpm
    'min_oxygen_saturation': (50, 100),  # %
}

print("Filtrando outliers fuera de rangos clínicos...")
for col, (lo, hi) in clinical_ranges.items():
    if col in df.columns:
        n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} outliers reemplazados por NaN")
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

# Renombrar columnas para mejor presentación - con símbolos μ ± σ para variables continuas
rename_map = {
    'anchor_age': 'Edad, años (μ ± σ)',
    'bmi': 'IMC, kg/m² (μ ± σ)',
    'min_bp_systolic': 'Presión sistólica mín, mmHg (μ ± σ)',
    'min_bp_diastolic': 'Presión diastólica mín, mmHg (μ ± σ)',
    'min_oxygen_saturation': 'SpO2 mín, % (μ ± σ)',
    'max_pulse': 'Frecuencia cardíaca máx, lpm (μ ± σ)',
    'max_temp': 'Temperatura máx, °F (μ ± σ)',
    'max_glucose': 'Glucosa máx, mg/dL (μ ± σ)',
    'max_resp_rate': 'Frecuencia resp máx, rpm (μ ± σ)',
    'max_creatinine': 'Creatinina máx, mg/dL (μ ± σ)',
    'max_bilirubin': 'Bilirrubina máx, mg/dL (μ ± σ)',
    'min_platelets': 'Plaquetas mín, ×10³/µL (μ ± σ)',
    'max_bun': 'BUN máx, mg/dL (μ ± σ)',
    'min_pao2': 'PaO2 mín, mmHg (μ ± σ)',
    'severity_score': 'Score de severidad (μ ± σ)',
    'gender': 'Sexo, n (%)',
    'is_mechanical_ventilation': 'Ventilación mecánica, n (%)',
    'is_intubated': 'Intubación, n (%)',
    'is_crrt': 'CRRT, n (%)',
    'has_hemodialysis': 'Hemodiálisis, n (%)',
    'is_ecmo': 'ECMO, n (%)',
    'is_prone_position': 'Posición prona, n (%)',
    'has_arterial_line': 'Línea arterial, n (%)',
    'has_central_line': 'Línea central, n (%)',
    'has_niv': 'Ventilación no invasiva, n (%)',
}

df_renamed = df.rename(columns=rename_map)

# Variables continuas (nombres actualizados con símbolos)
continuous_vars = [
    'Edad, años (μ ± σ)', 'IMC, kg/m² (μ ± σ)', 'Presión sistólica mín, mmHg (μ ± σ)', 
    'Presión diastólica mín, mmHg (μ ± σ)', 'SpO2 mín, % (μ ± σ)', 'Frecuencia cardíaca máx, lpm (μ ± σ)',
    'Temperatura máx, °F (μ ± σ)', 'Glucosa máx, mg/dL (μ ± σ)', 'Frecuencia resp máx, rpm (μ ± σ)',
    'Creatinina máx, mg/dL (μ ± σ)', 'Bilirrubina máx, mg/dL (μ ± σ)', 'Plaquetas mín, ×10³/µL (μ ± σ)',
    'BUN máx, mg/dL (μ ± σ)', 'PaO2 mín, mmHg (μ ± σ)', 'Score de severidad (μ ± σ)'
]

# Variables categóricas (nombres actualizados)
categorical_vars = [
    'Sexo, n (%)', 'Ventilación mecánica, n (%)', 'Intubación, n (%)', 'CRRT, n (%)', 'Hemodiálisis, n (%)',
    'ECMO, n (%)', 'Posición prona, n (%)', 'Línea arterial, n (%)', 'Línea central, n (%)', 'Ventilación no invasiva, n (%)'
]

# Filtrar solo variables que existen
continuous_vars = [v for v in continuous_vars if v in df_renamed.columns]
categorical_vars = [v for v in categorical_vars if v in df_renamed.columns]
all_columns = continuous_vars + categorical_vars

# Crear Table One
mytable = TableOne(
    df_renamed,
    columns=all_columns,
    categorical=categorical_vars,
    continuous=continuous_vars,
    groupby='mortality_status',
    pval=True,
    missing=True,
    overall=True,
    label_suffix=False,
    decimals=1
)

# Imprimir en consola
print("\n" + "="*80)
print("TABLE ONE - COVID-19 ICU COHORT (Dataset v5)")
print("Valores continuos: Media ± Desviación Típica")
print("Valores categóricos: n (%)")
print("="*80)
print(mytable.tabulate(tablefmt='grid'))

# Guardar como CSV y HTML
mytable.to_csv(os.path.join(OUTPUT_DIR, 'table_1_demographics.csv'))
mytable.to_html(os.path.join(OUTPUT_DIR, 'table_1_demographics.html'))
print(f"✓ Table-1 CSV guardada en {OUTPUT_DIR}/table_1_demographics.csv")
print(f"✓ Table-1 HTML guardada en {OUTPUT_DIR}/table_1_demographics.html")

# Crear visualización con matplotlib - encabezados limpios
table_df = mytable.tableone

fig, ax = plt.subplots(figsize=(14, 16))
ax.axis('off')

# Título con especificación clara del formato usando símbolos
title_text = ('Table 1: Características de la Cohorte COVID-19 UCI (Dataset v5 - Opción B)\n'
              'Estratificada por mortalidad a 30 días\n'
              'Variables continuas: (μ ± σ) | Variables categóricas: n (%)')
ax.set_title(title_text, fontsize=13, fontweight='bold', pad=20)

# Resetear índice y limpiar nombres de columnas
table_display = table_df.reset_index()

# Limpiar nombres de columnas del multi-index
clean_columns = []
for col in table_display.columns:
    if isinstance(col, tuple):
        # Tomar el primer elemento no vacío
        clean_name = col[0] if col[0] else col[1]
        if clean_name == 'level_0':
            clean_name = 'Variable'
        elif clean_name == 'level_1':
            clean_name = ''
        clean_columns.append(clean_name)
    else:
        clean_columns.append(col)

# Simplificar: renombrar columnas problemáticas
col_rename = {
    'Grouped by mortality_status': 'Overall',
    'Missing': 'Missing',
    'Overall': 'Overall',
    'Fallecidos': 'Fallecidos',
    'Supervivientes': 'Supervivientes',
    'P-Value': 'P-Value'
}

table_display.columns = ['Variable', 'Categoría', 'Missing', 'Overall', 'Fallecidos', 'Supervivientes', 'P-Value']

# Crear tabla visual
table = ax.table(cellText=table_display.values,
                 colLabels=table_display.columns.tolist(),
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.22, 0.08, 0.08, 0.18, 0.18, 0.18, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 1.5)

# Estilo de encabezados
n_cols = len(table_display.columns)
for i in range(n_cols):
    table[(0, i)].set_facecolor('#1976D2')
    table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=9)

# Alternar colores de filas
for i in range(1, len(table_display) + 1):
    color = '#E3F2FD' if i % 2 == 0 else 'white'
    for j in range(n_cols):
        table[(i, j)].set_facecolor(color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'table_1_demographics.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'table_1_demographics.pdf'), bbox_inches='tight')
print(f"✓ Table-1 imagen guardada en {OUTPUT_DIR}/table_1_demographics.png")

# ============================================================================
# 3. TABLA DE PACIENTES CON MÚLTIPLES ESTANCIAS
# ============================================================================
print("\nGenerando tabla de múltiples estancias...")

# Calcular estadísticas de estancias por paciente
stays_per_patient = df_all_stays.groupby('subject_id').agg(
    num_stays=('stay_id', 'count'),
    first_admit=('admittime', 'min'),
    last_admit=('admittime', 'max')
).reset_index()

# Estadísticas generales
multi_stays_stats = {
    'Total de estancias UCI': n_total_stays,
    'Pacientes totales': n_unique_patients,
    'Pacientes con 1 estancia': n_patients_single_stay,
    'Pacientes con múltiples estancias': n_patients_multiple_stays,
    'Porcentaje múltiples estancias': f"{(n_patients_multiple_stays/n_unique_patients)*100:.1f}%",
    '---': '---',
    'Mín estancias por paciente': stays_per_patient['num_stays'].min(),
    'Máx estancias por paciente': stays_per_patient['num_stays'].max(),
    'Media estancias por paciente': f"{stays_per_patient['num_stays'].mean():.2f}",
    'Mediana estancias por paciente': stays_per_patient['num_stays'].median()
}

# Distribución de número de estancias
stays_distribution = stays_per_patient['num_stays'].value_counts().sort_index()

# Crear figura con dos subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Tabla de estadísticas
ax1 = axes[0]
ax1.axis('off')
ax1.set_title('Estadísticas de Estancias UCI por Paciente', fontsize=12, fontweight='bold', pad=10)

stats_data = [[k, str(v)] for k, v in multi_stays_stats.items()]
table1 = ax1.table(cellText=stats_data,
                   colLabels=['Métrica', 'Valor'],
                   cellLoc='left',
                   loc='center',
                   colWidths=[0.6, 0.3])

table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1.2, 2.0)

for i in range(2):
    table1[(0, i)].set_facecolor('#00796B')
    table1[(0, i)].set_text_props(color='white', fontweight='bold')

for i in range(1, len(stats_data) + 1):
    color = '#B2DFDB' if i % 2 == 0 else 'white'
    for j in range(2):
        table1[(i, j)].set_facecolor(color)

# Subplot 2: Gráfico de distribución con seaborn
ax2 = axes[1]
colors = sns.color_palette("viridis", len(stays_distribution))

bars = ax2.bar(stays_distribution.index.astype(str), stays_distribution.values, 
               color=colors, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Número de Estancias UCI', fontsize=11)
ax2.set_ylabel('Número de Pacientes', fontsize=11)
ax2.set_title('Distribución de Estancias por Paciente', fontsize=12, fontweight='bold')

# Añadir etiquetas en las barras
for bar, val in zip(bars, stays_distribution.values):
    height = bar.get_height()
    ax2.annotate(f'{val:,}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

# Añadir porcentajes
total = stays_distribution.sum()
for bar, val in zip(bars, stays_distribution.values):
    height = bar.get_height()
    pct = (val / total) * 100
    ax2.annotate(f'({pct:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 15),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, color='gray')

ax2.set_ylim(0, max(stays_distribution.values) * 1.2)
sns.despine(ax=ax2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'table_multiple_stays.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'table_multiple_stays.pdf'), bbox_inches='tight')
print(f"✓ Tabla de múltiples estancias guardada en {OUTPUT_DIR}/table_multiple_stays.png")

# Crear tabla detallada de distribución
distribution_df = pd.DataFrame({
    'Número de Estancias': stays_distribution.index,
    'Número de Pacientes': stays_distribution.values,
    'Porcentaje': [f"{(v/total)*100:.2f}%" for v in stays_distribution.values],
    'Total Estancias Acumuladas': stays_distribution.index * stays_distribution.values
})
distribution_df.to_csv(os.path.join(OUTPUT_DIR, 'stays_distribution.csv'), index=False)
print(f"✓ Distribución de estancias CSV guardada en {OUTPUT_DIR}/stays_distribution.csv")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*60)
print("RESUMEN DE GENERACIÓN")
print("="*60)
print(f"Archivos generados en: {OUTPUT_DIR}/")
print(f"  1. flow_inclusion_criteria.png/pdf - Diagrama de flujo")
print(f"  2. table_1_demographics.png/pdf/csv - Table-1")
print(f"  3. table_multiple_stays.png/pdf - Tabla múltiples estancias")
print(f"  4. stays_distribution.csv - Distribución de estancias")
print("="*60)

# Mostrar estadísticas clave
print("\nESTADÍSTICAS CLAVE DEL DATASET V5:")
print(f"  • Total estancias UCI COVID-19: {n_total_stays:,}")
print(f"  • Pacientes únicos: {n_unique_patients:,}")
print(f"  • Pacientes con múltiples estancias: {n_patients_multiple_stays:,} ({(n_patients_multiple_stays/n_unique_patients)*100:.1f}%)")
print(f"  • Rango de estancias: {stays_per_patient['num_stays'].min()} - {stays_per_patient['num_stays'].max()}")
print(f"  • Mortalidad a 30 días: {(df['mortality_30d'].sum()/len(df))*100:.1f}%")
print(f"  • Variables en v5: {len(df_v5_optionB.columns)} columnas")
print("="*60)
