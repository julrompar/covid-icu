#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para generar gráficas descriptivas del dataset v7:
1. Diagrama de flujo de criterios de inclusión (CONSORT-style)
2. Table-1 con características demográficas y clínicas
3. Tabla de pacientes con múltiples estancias

Cambios respecto a v6:
- Criterio de inclusión en dos vías: U07.1 en posición 1 (siempre incluido) o
  U07.1 en posición 2 con diagnóstico primario de comorbilidad COVID
- Ventana de extracción: 72 horas
- Variables labs/vitales: val_first, val_last, val_min, val_max, val_delta
- Opción B: estancia más grave por (SOFA + APACHE II normalizados), sin severity_score
- 22 categorías PMHX como variables binarias
- Variables clínicas binarias adicionales (bin_sofa_ge2, bin_apache2_ge10, etc.)
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
df_v7_optionB = pd.read_parquet(os.path.join(BASE_DIR, 'dataset_v7_optionB.parquet'))
df_v7_optionC = pd.read_parquet(os.path.join(BASE_DIR, 'dataset_v7_optionC.parquet'))

# ============================================================================
# 1. DIAGRAMA DE FLUJO DE INCLUSIÓN (CONSORT-style)
# ============================================================================
print("\nGenerando diagrama de flujo de inclusión...")

# Estadísticas para el flujo (aproximadas desde los parquets finales)
n_final_optionB = len(df_v7_optionB)
n_final_optionC = len(df_v7_optionC)

# Crear figura del diagrama de flujo
fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
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


def draw_lateral_box(ax, x, y, width, height, text, color='#FFEBEE', edgecolor='#C62828', fontsize=9):
    """Dibuja caja de exclusión lateral (sin flecha)"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.03,rounding_size=0.1",
                         facecolor=color, edgecolor=edgecolor, linewidth=1.5,
                         linestyle='--')
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, style='italic')


# Título
ax.text(5, 15.7, 'Criterios de Inclusión — Dataset v7\n(Ventana de extracción: 72 horas)',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#0D47A1')

# ── Caja 1: Base MIMIC-IV ────────────────────────────────────────────────────
draw_box(ax, 5, 14.6, 5, 0.8,
         'Base de datos MIMIC-IV\nPacientes UCI (NWICU)',
         color='#BBDEFB')

draw_arrow(ax, 5, 14.6 - 0.8/2, 5, 13.4 + 0.9/2)

# ── Caja 2: Filtro ICD-10 U07.1 ─────────────────────────────────────────────
draw_box(ax, 5, 13.4, 6, 0.9,
         'Diagnóstico ICD-10: U07.1 (COVID-19)\npresente en cualquier posición de la cadena diagnóstica',
         color='#C8E6C9', edgecolor='#388E3C')

# Bifurcación: dos vías de inclusión
draw_arrow(ax, 5, 13.4 - 0.9/2, 3, 11.8 + 1.1/2)
draw_arrow(ax, 5, 13.4 - 0.9/2, 7, 11.8 + 1.1/2)

# Etiquetas de bifurcación
ax.text(3.2, 12.7, 'Vía 1', ha='center', va='center', fontsize=9,
        color='#1976D2', fontweight='bold')
ax.text(6.8, 12.7, 'Vía 2', ha='center', va='center', fontsize=9,
        color='#7B1FA2', fontweight='bold')

# ── Vía 1: U07.1 en posición 1 (diagnóstico primario) ───────────────────────
draw_box(ax, 3, 11.8, 4.2, 1.1,
         'U07.1 en posición 1\n(diagnóstico primario)\n→ Siempre incluido',
         color='#DCEDC8', edgecolor='#558B2F', fontsize=9)

# ── Vía 2: U07.1 en posición 2 con comorbilidad COVID como primario ──────────
draw_box(ax, 7, 11.8, 4.2, 1.1,
         'U07.1 en posición 2\n→ Solo si posición 1 es\ncomorbilidad COVID (22 categorías)',
         color='#E1BEE7', edgecolor='#7B1FA2', fontsize=9)

# Caja lateral: categorías de comorbilidades
draw_lateral_box(ax, 8.8, 10.4, 2.8, 1.4,
                 '22 categorías PMHX:\nCardíacas, Pulmonares, Renales,\nDiabetes, Oncológicas,\nNeurológicas, Inmunes...',
                 color='#F3E5F5', edgecolor='#9C27B0', fontsize=8)
ax.annotate('', xy=(7.5, 11.8 - 1.1/2), xytext=(8.8, 10.4 + 1.4/2),
            arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5, linestyle='dashed'))

# ── Merge: Unión de ambas vías ───────────────────────────────────────────────
draw_arrow(ax, 3, 11.8 - 1.1/2, 5, 10.2 + 0.9/2)
draw_arrow(ax, 7, 11.8 - 1.1/2, 5, 10.2 + 0.9/2)

draw_box(ax, 5, 10.2, 5.5, 0.9,
         'Estancias UCI elegibles\n(pool de estancias por paciente)',
         color='#FFF9C4', edgecolor='#FBC02D')

draw_arrow(ax, 5, 10.2 - 0.9/2, 5, 8.8 + 1.0/2)

# ── Caja: Múltiples estancias ────────────────────────────────────────────────
draw_box(ax, 5, 8.8, 6, 1.0,
         'Pacientes con múltiples estancias UCI\n'
         '→ Compactación requerida: 1 registro por paciente',
         color='#FFECB3', edgecolor='#FF9800')

# Etiqueta de exclusión lateral
draw_lateral_box(ax, 8.8, 8.3, 2.5, 0.7,
                 'Excluídos:\nestancias sin datos\nen ventana 72h',
                 fontsize=8)
ax.plot([6.5, 7.5], [8.8, 8.3], color='#C62828', lw=1.5, linestyle='--')

draw_arrow(ax, 5, 8.8 - 1.0/2, 5, 7.2 + 0.9/2)

# ── Compactación ─────────────────────────────────────────────────────────────
draw_box(ax, 5, 7.2, 5.5, 0.9,
         'Compactación: 1 registro por paciente\nPor criterio de selección de estancia',
         color='#E1BEE7', edgecolor='#7B1FA2')

# Bifurcación hacia Opciones B y C
draw_arrow(ax, 5, 7.2 - 0.9/2, 3, 5.7 + 1.0/2)
draw_arrow(ax, 5, 7.2 - 0.9/2, 7, 5.7 + 1.0/2)

# ── Opción B ─────────────────────────────────────────────────────────────────
draw_box(ax, 3, 5.7, 3.8, 1.0,
         'Opción B:\nEstancia más grave\n(SOFA/20 + APACHE II/51)\nSin severity_score',
         color='#D1C4E9', edgecolor='#512DA8', fontsize=9)

draw_arrow(ax, 3, 5.7 - 1.0/2, 3, 4.2 + 0.9/2)

draw_box(ax, 3, 4.2, 3.2, 0.9,
         f'Dataset v7 — Opción B\nn = {n_final_optionB:,} pacientes',
         color='#B39DDB', edgecolor='#4527A0')

# ── Opción C ─────────────────────────────────────────────────────────────────
draw_box(ax, 7, 5.7, 3.8, 1.0,
         'Opción C:\nÚltima estancia UCI\n(más reciente por admittime)\nCon severity_score',
         color='#B2DFDB', edgecolor='#00796B', fontsize=9)

draw_arrow(ax, 7, 5.7 - 1.0/2, 7, 4.2 + 0.9/2)

draw_box(ax, 7, 4.2, 3.2, 0.9,
         f'Dataset v7 — Opción C\nn = {n_final_optionC:,} pacientes',
         color='#80CBC4', edgecolor='#004D40')

# ── Nota de features ─────────────────────────────────────────────────────────
ax.text(5, 3.2,
        'Features extraídas en ventana 72h · 5 agregaciones por variable continua:\n'
        'val_first (baseline) · val_last (72h) · val_min · val_max · val_delta (last−first)\n'
        '+ 22 variables PMHX binarias · Variables binarias clínicas (bin_sofa_ge2, bin_apache2_ge10, …)',
        ha='center', va='center', fontsize=8.5, color='#37474F',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', edgecolor='#90A4AE', linewidth=1))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'flow_inclusion_criteria.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'flow_inclusion_criteria.pdf'), bbox_inches='tight')
print(f"✓ Diagrama de flujo guardado en {OUTPUT_DIR}/flow_inclusion_criteria.png")

# ============================================================================
# 2. TABLE-1: Características demográficas y clínicas (usando tableone)
# ============================================================================
print("\nGenerando Table-1 con tableone...")

from tableone import TableOne

df = df_v7_optionB.copy()
df['mortality_30d'] = df['dod_within_30_days'].notna().astype(int)
df['mortality_status'] = df['mortality_30d'].map({1: 'Fallecidos', 0: 'Supervivientes'})

# Rangos clínicos razonables (nueva nomenclatura _val_max / _val_min)
clinical_ranges = {
    'lab_glucose_val_max':      (0, 1000),
    'lab_creatinine_val_max':   (0, 30),
    'lab_bilirubin_val_max':    (0, 50),
    'lab_platelets_val_min':    (0, 1000),
    'lab_bun_val_max':          (0, 200),
    'lab_pao2_val_min':         (0, 700),
    'vital_sbp_val_min':        (40, 250),
    'vital_dbp_val_min':        (20, 150),
    'vital_hr_val_max':         (30, 250),
    'vital_temp_val_max':       (86, 113),
    'vital_rr_val_max':         (4, 70),
    'vital_spo2_val_min':       (50, 100),
    'vital_bmi_val_max':        (10, 80),
}

print("Filtrando outliers fuera de rangos clínicos...")
for col, (lo, hi) in clinical_ranges.items():
    if col in df.columns:
        n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} outliers reemplazados por NaN")
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

# Renombrar columnas para presentación
rename_map = {
    'anchor_age':               'Edad, años (μ ± σ)',
    'vital_bmi_val_max':        'IMC, kg/m² (μ ± σ)',
    'vital_sbp_val_min':        'Presión sistólica mín, mmHg (μ ± σ)',
    'vital_dbp_val_min':        'Presión diastólica mín, mmHg (μ ± σ)',
    'vital_spo2_val_min':       'SpO2 mín, % (μ ± σ)',
    'vital_hr_val_max':         'Frecuencia cardíaca máx, lpm (μ ± σ)',
    'vital_temp_val_max':       'Temperatura máx, °F (μ ± σ)',
    'vital_rr_val_max':         'Frecuencia resp máx, rpm (μ ± σ)',
    'lab_glucose_val_max':      'Glucosa máx, mg/dL (μ ± σ)',
    'lab_creatinine_val_max':   'Creatinina máx, mg/dL (μ ± σ)',
    'lab_bilirubin_val_max':    'Bilirrubina máx, mg/dL (μ ± σ)',
    'lab_platelets_val_min':    'Plaquetas mín, ×10³/µL (μ ± σ)',
    'lab_bun_val_max':          'BUN máx, mg/dL (μ ± σ)',
    'lab_pao2_val_min':         'PaO2 mín, mmHg (μ ± σ)',
    'lab_lactate_val_max':      'Lactato máx, mmol/L (μ ± σ)',
    'sofa_partial':             'SOFA parcial (μ ± σ)',
    'apache2_partial':          'APACHE II parcial (μ ± σ)',
    'gender':                   'Sexo, n (%)',
    'is_mechanical_ventilation': 'Ventilación mecánica, n (%)',
    'is_intubated':             'Intubación, n (%)',
    'is_crrt':                  'CRRT, n (%)',
    'has_hemodialysis':         'Hemodiálisis, n (%)',
    'is_ecmo':                  'ECMO, n (%)',
    'is_prone_position':        'Posición prona, n (%)',
    'has_arterial_line':        'Línea arterial, n (%)',
    'has_central_line':         'Línea central, n (%)',
    'has_niv':                  'Ventilación no invasiva, n (%)',
    # PMHX
    'pmhx_diabetes':            'Diabetes mellitus, n (%)',
    'pmhx_hypertension':        'Hipertensión arterial, n (%)',
    'pmhx_heart_failure':       'Insuficiencia cardíaca, n (%)',
    'pmhx_ckd':                 'Enfermedad renal crónica, n (%)',
    'pmhx_copd':                'EPOC, n (%)',
    'pmhx_cancer':              'Neoplasia maligna, n (%)',
    'pmhx_obesity':             'Obesidad, n (%)',
}

df_renamed = df.rename(columns=rename_map)

# Variables continuas
continuous_vars = [
    'Edad, años (μ ± σ)',
    'IMC, kg/m² (μ ± σ)',
    'Presión sistólica mín, mmHg (μ ± σ)',
    'Presión diastólica mín, mmHg (μ ± σ)',
    'SpO2 mín, % (μ ± σ)',
    'Frecuencia cardíaca máx, lpm (μ ± σ)',
    'Temperatura máx, °F (μ ± σ)',
    'Frecuencia resp máx, rpm (μ ± σ)',
    'Glucosa máx, mg/dL (μ ± σ)',
    'Creatinina máx, mg/dL (μ ± σ)',
    'Bilirrubina máx, mg/dL (μ ± σ)',
    'Plaquetas mín, ×10³/µL (μ ± σ)',
    'BUN máx, mg/dL (μ ± σ)',
    'PaO2 mín, mmHg (μ ± σ)',
    'Lactato máx, mmol/L (μ ± σ)',
    'SOFA parcial (μ ± σ)',
    'APACHE II parcial (μ ± σ)',
]

# Variables categóricas
categorical_vars = [
    'Sexo, n (%)',
    'Ventilación mecánica, n (%)',
    'Intubación, n (%)',
    'CRRT, n (%)',
    'Hemodiálisis, n (%)',
    'ECMO, n (%)',
    'Posición prona, n (%)',
    'Línea arterial, n (%)',
    'Línea central, n (%)',
    'Ventilación no invasiva, n (%)',
    'Diabetes mellitus, n (%)',
    'Hipertensión arterial, n (%)',
    'Insuficiencia cardíaca, n (%)',
    'Enfermedad renal crónica, n (%)',
    'EPOC, n (%)',
    'Neoplasia maligna, n (%)',
    'Obesidad, n (%)',
]

# Filtrar solo variables que existen en el dataframe
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

print("\n" + "="*80)
print("TABLE ONE - COVID-19 ICU COHORT (Dataset v7 — Opción B)")
print("Valores continuos: Media ± Desviación Típica")
print("Valores categóricos: n (%)")
print("="*80)
print(mytable.tabulate(tablefmt='grid'))

mytable.to_csv(os.path.join(OUTPUT_DIR, 'table_1_demographics.csv'))
mytable.to_html(os.path.join(OUTPUT_DIR, 'table_1_demographics.html'))
print(f"✓ Table-1 CSV guardada en {OUTPUT_DIR}/table_1_demographics.csv")
print(f"✓ Table-1 HTML guardada en {OUTPUT_DIR}/table_1_demographics.html")

# Visualización matplotlib
table_df = mytable.tableone

fig, ax = plt.subplots(figsize=(14, 18))
ax.axis('off')

title_text = ('Table 1: Características de la Cohorte COVID-19 UCI (Dataset v7 — Opción B)\n'
              'Estratificada por mortalidad a 30 días · Ventana de extracción: 72 horas\n'
              'Variables continuas: (μ ± σ) | Variables categóricas: n (%)')
ax.set_title(title_text, fontsize=13, fontweight='bold', pad=20)

table_display = table_df.reset_index()
table_display.columns = ['Variable', 'Categoría', 'Missing', 'Overall', 'Fallecidos', 'Supervivientes', 'P-Value']

table = ax.table(cellText=table_display.values,
                 colLabels=table_display.columns.tolist(),
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.22, 0.08, 0.08, 0.18, 0.18, 0.18, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 1.5)

n_cols = len(table_display.columns)
for i in range(n_cols):
    table[(0, i)].set_facecolor('#1976D2')
    table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=9)

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

# En v7 la info de estancias por paciente viene del parquet final (OptionC tiene más estancias
# que OptionB por definición, pero ambas ya están compactadas → usamos OptionC como referencia
# para distribución ya que conserva la misma lógica de agrupación)
# Si se dispone del parquet "all_stays" de v7 se puede cargar aquí:
all_stays_path = os.path.join(BASE_DIR, 'dataset_v7_all_stays.parquet')
if os.path.exists(all_stays_path):
    df_all_stays = pd.read_parquet(all_stays_path)
    n_total_stays  = len(df_all_stays)
    n_unique_patients = df_all_stays['subject_id'].nunique()

    stays_per_patient = df_all_stays.groupby('subject_id').agg(
        num_stays=('stay_id', 'count'),
        first_admit=('admittime', 'min'),
        last_admit=('admittime', 'max')
    ).reset_index()

    n_patients_multiple_stays = (stays_per_patient['num_stays'] > 1).sum()
    n_patients_single_stay    = n_unique_patients - n_patients_multiple_stays

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
        'Mediana estancias por paciente': stays_per_patient['num_stays'].median(),
    }

    stays_distribution = stays_per_patient['num_stays'].value_counts().sort_index()
    total = stays_distribution.sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

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

    ax2 = axes[1]
    colors = sns.color_palette("viridis", len(stays_distribution))
    bars = ax2.bar(stays_distribution.index.astype(str), stays_distribution.values,
                   color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Número de Estancias UCI', fontsize=11)
    ax2.set_ylabel('Número de Pacientes', fontsize=11)
    ax2.set_title('Distribución de Estancias por Paciente', fontsize=12, fontweight='bold')

    for bar, val in zip(bars, stays_distribution.values):
        height = bar.get_height()
        ax2.annotate(f'{val:,}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars, stays_distribution.values):
        height = bar.get_height()
        pct = (val / total) * 100
        ax2.annotate(f'({pct:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 15), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, color='gray')

    ax2.set_ylim(0, max(stays_distribution.values) * 1.2)
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'table_multiple_stays.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'table_multiple_stays.pdf'), bbox_inches='tight')
    print(f"✓ Tabla de múltiples estancias guardada en {OUTPUT_DIR}/table_multiple_stays.png")

    distribution_df = pd.DataFrame({
        'Número de Estancias': stays_distribution.index,
        'Número de Pacientes': stays_distribution.values,
        'Porcentaje': [f"{(v/total)*100:.2f}%" for v in stays_distribution.values],
        'Total Estancias Acumuladas': stays_distribution.index * stays_distribution.values
    })
    distribution_df.to_csv(os.path.join(OUTPUT_DIR, 'stays_distribution.csv'), index=False)
    print(f"✓ Distribución de estancias CSV guardada en {OUTPUT_DIR}/stays_distribution.csv")
else:
    print(f"  ⚠️  No se encontró {all_stays_path} — omitiendo tabla de múltiples estancias.")
    print("     Genera el parquet 'dataset_v7_all_stays.parquet' desde dataset_v7.ipynb para habilitar esta sección.")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*60)
print("RESUMEN DE GENERACIÓN — Dataset v7")
print("="*60)
print(f"Archivos generados en: {OUTPUT_DIR}/")
print(f"  1. flow_inclusion_criteria.png/pdf  — Diagrama de flujo")
print(f"  2. table_1_demographics.png/pdf/csv — Table-1")
if os.path.exists(all_stays_path):
    print(f"  3. table_multiple_stays.png/pdf     — Tabla múltiples estancias")
    print(f"  4. stays_distribution.csv           — Distribución de estancias")
print("="*60)

df_b = df_v7_optionB.copy()
df_b['mortality_30d'] = df_b['dod_within_30_days'].notna().astype(int)

print(f"\nESTADÍSTICAS CLAVE DEL DATASET V7:")
print(f"  • Pacientes únicos (Opción B): {n_final_optionB:,}")
print(f"  • Pacientes únicos (Opción C): {n_final_optionC:,}")
print(f"  • Mortalidad a 30 días (B):    {df_b['mortality_30d'].mean()*100:.1f}%")
print(f"  • Variables en v7 (OptionB):   {len(df_v7_optionB.columns)} columnas")
print(f"  • Ventana de extracción:        72 horas")
print(f"  • Opción B: compactación por   SOFA/20 + APACHE II/51 (sin severity_score)")
print(f"  • Opción C: compactación por   última estancia (con severity_score)")
print("="*60)
