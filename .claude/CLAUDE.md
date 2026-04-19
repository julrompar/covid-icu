# COVID ICU — Predicción de Mortalidad UCI COVID-19

## Contexto
TFG (Trabajo de Fin de Grado). Modelo de predicción de mortalidad a 30 días en pacientes UCI COVID-19 usando datos MIMIC-IV. Ejecución 100% local, sin infraestructura cloud.

- **Python**: 3.13.7
- **Repositorio**: github.com/julrompar/covid-icu (rama única: `main`)

## Objetivo del modelo
Clasificación binaria: predecir si un paciente UCI COVID-19 fallece dentro de los 30 días desde el ingreso. Variable target: `dod_within_30_days` no nulo.

## Stack principal
- **Modelo**: XGBoost v7 con optimización de hiperparámetros vía Optuna (100 trials, MedianPruner)
- **Métricas objetivo**: F1 y F2 (beta=2). AUROC de referencia: ~0.785
- **Pipeline de datos**: Spark + pandas
- **Interpretabilidad**: SHAP beeswarm (top 20 features)
- **Calibración**: Platt scaling (CalibratedClassifierCV, cv='prefit')
- **Baselines**: SOFA parcial, APACHE II parcial, Regresión Logística

## Dataset
- **Versión activa**: v7 (con variantes v7 y v7.1)
- **Variantes**: OptionB y OptionC (parquets en `datasets/v7/`)
  - **OptionB**: compacta stays por SOFA/20 + APACHE II/51 (sin `severity_score`)
  - **OptionC**: compacta stays seleccionando la última stay (con `severity_score`)
- **v7 vs v7.1**:
  - **v7**: Imputación completa (mediana para vitales/labs/scores, 0 para procedimientos/PMHX/indicadores)
  - **v7.1**: Sin imputación (XGBoost maneja nativamente NaN; solo procedimientos/PMHX/indicadores → 0)
- **Mejoras en v7 respecto a v6**:
  - Criterio de inclusión revisado: U07.1 en posición 1 siempre; posición 2 solo si diagnóstico primario es comorbilidad COVID
  - 22 variables PMHX binarias (antecedentes patológicos)
  - Ventana de extracción ampliada a 72 horas
  - 5 agregaciones por variable continua: `val_first`, `val_last`, `val_min`, `val_max`, `val_delta`
  - Variables clínicas binarias adicionales: `bin_sofa_ge2`, `bin_apache2_ge10`, etc.
- **Features**: 100+ variables en 9 grupos:
  - A: Vitales (5 agregaciones × n vitales)
  - B: Labs alta cobertura (5 agregaciones × n labs)
  - C: Labs cobertura media (5 agregaciones × n labs)
  - D: Labs baja cobertura — solo indicadores `_measured`, sin agregaciones numéricas
  - E: Procedimientos (prefijos `is_` y `has_`)
  - F: Demográficas (`anchor_age`, `gender`, `marital_status`, `race`)
  - G: Scores clínicos (`sofa_partial`, `apache2_partial`; `severity_score` solo en OptionC)
  - H: PMHX binarios (22 variables `pmhx_*`)
  - I: Variables binarias clínicas (`bin_sofa_ge2`, `bin_apache2_ge10`, etc.)
- **Columnas excluidas siempre**: `subject_id`, `admission_id`, `stay_id`, `admittime`, `dod_within_30_days`, `target`
- **Labs de baja cobertura** (usar solo indicador `_measured`, sin agregaciones): `pt`, `crp`, `ferritin`, `dimer`, `lymphocytes`, `neutrophils`, `troponin`

## Preprocesamiento

### v7 (con imputación completa)
- Split: 70/15/15 (train/val/test), estratificado, `random_state=42`
- Imputación diferenciada — fit solo en train, nunca en val/test:
  - Procedimientos (`is_*`, `has_*`): fillna(0)
  - Indicadores (`_measured`): fillna(0)
  - PMHX binarios (`pmhx_*`): fillna(0)
  - Resto (vitales, labs, scores, demográficas): mediana del train
- Encoding one-hot de categóricas: `gender`, `marital_status`, `race`
- `scale_pos_weight` = ratio negativo/positivo del train

### v7.1 (sin imputación en vitales/labs/scores)
- Split: idéntico a v7 (70/15/15, estratificado, `random_state=42`)
- **Solo imputa**: procedimientos, PMHX e indicadores → 0
- **Deja NaN**: vitales, labs, scores, demográficas (XGBoost maneja nativamente)
- Encoding one-hot de categóricas: `gender`, `marital_status`, `race`
- `scale_pos_weight` = ratio negativo/positivo del train

## Scripts principales

### v7 (con imputación)
`models/xgboost_v7_opt_comparator.py`

```bash
# Uso básico
python3 models/xgboost_v7_opt_comparator.py --option B --metric F1 --trials 100

# Argumentos
--option  B | C | all     # variante del dataset (default: all)
--metric  F1 | F2 | both  # métrica de optimización (default: both)
--trials  int             # número de trials Optuna (default: 100)
```

### v7.1 (sin imputación en numéricas)
`models/xgboost_v7_1_opt_comparator.py`

```bash
# Uso básico (igual interfaz que v7)
python3 models/xgboost_v7_1_opt_comparator.py --option C --metric F2 --trials 50

# Argumentos: idénticos a v7
--option  B | C | all     # variante del dataset (default: all)
--metric  F1 | F2 | both  # métrica de optimización (default: both)
--trials  int             # número de trials Optuna (default: 100)
```

### Salida de resultados
- **v7**: `v7-models/{option}/{metric}/`
- **v7.1**: también en `v7-models/{option}/{metric}/` (distinguible por timestamps)

Archivos generados en cada ejecución:
- `metrics_report.txt` — métricas completas (AUROC, F1, F2, etc.)
- `confusion_matrix.png` — matriz de confusión
- `calibration_curve.png` — curva de calibración (Platt)
- `shap_beeswarm.png` — top 20 features por SHAP
- `fn_analysis.txt` — análisis de falsos negativos

## Estructura del proyecto
```
covid-icu/
├── data/           ← DATOS RAW MIMIC-IV — NO TOCAR, NO LEER
├── datasets/       ← DATOS PROCESADOS — NO TOCAR, NO LEER
│   ├── v1-v6/      ← Versiones anteriores (referencia)
│   └── v7/         ← Dataset v7 (OptionB, OptionC) — versión activa
├── markdowns/      ← Documentación y notas
├── models/         ← Scripts de entrenamiento y comparación
│   ├── xgboost_v7_opt_comparator.py        ← v7 (con imputación)
│   └── xgboost_v7_1_opt_comparator.py      ← v7.1 (sin imputación en numéricas)
├── notebooks/      ← Exploración y validación
├── queries/        ← SQL/PySpark de extracción de features
└── utils/          ← Inicialización de Spark únicamente
```

## Restricciones importantes
- **No acceder ni modificar `data/` ni `datasets/`** bajo ningún concepto
- No aplicar convenciones de estilo (PEP8, type hints, etc.) salvo que se pida explícitamente
- No proponer llevar el proyecto a cloud ni añadir infraestructura
- Al versionar modelos, seguir el patrón: `xgboost_v{N}_opt_comparator.py` o `xgboost_v{N}_{N2}_opt_comparator.py` para variantes menores