# Comparativa de Modelos: v1 XGBoost vs v3 Stacking Ensemble

## 1. Resumen Ejecutivo

| Aspecto | v1 XGBoost F1 / F2 | v3 Stacking Ensemble |
|---|---|---|
| **Modelo** | XGBoost individual + calibración isotónica | Stacking: XGBoost + LightGBM + Logistic Regression con meta-learner LR |
| **Dataset** | `dataset_v1_clean.csv` (v1) | `dataset_v2_engineered.csv` (v2) |
| **Imputación** | MICE (IterativeImputer) para TODO | Solo para LR (mediana); XGB/LGB manejan NaN nativamente |
| **Optimización** | Optuna 50 trials, optimiza AUPRC | Optuna 75 trials (XGB) + 50 trials (LGB), optimiza F2 directamente |
| **Calibración** | Calibración isotónica post-hoc | Sin calibración post-hoc (stacking actúa como calibrador implícito) |
| **Selección de umbral** | Sobre validation set (15% del train) | Sobre predicciones OOF 5-fold (sin holdout separado) |

---

## 2. Diferencias en el Dataset

### v1: `dataset_v1_clean.csv`
- Derivado del dataset original con limpieza manual (`dataset_cleaner.py`)
- **Eliminó**: `length_of_stay`, `min_bp_systolic_line`, `min_bp_diastolic_line`, `max_lactate`
- **Feature engineering inline** en el script: `mean_arterial_pressure`, `shock_index`, `severe_hypoxemia`, `fever`, `hypothermia`, `tachycardia_severe`, `hypotension`, `organ_support_count`, interacciones edad×procedimiento, `sofa_partial`
- **Incluye** `max_glucose` como feature
- **Drop manual**: `is_dnr`, `has_chest_tube` (eliminados como features, no como parte del pipeline de datos)

### v3: `dataset_v2_engineered.csv`
- Derivado de `generate_v2.py` con pipeline de datos estandarizado
- **Eliminó desde la base**: `length_of_stay`, `max_lactate`, `max_glucose` (antes del engineering)
- **Eliminó en clean**: `min_bp_systolic_line`, `min_bp_diastolic_line` (81% NaN)
- **Feature engineering centralizado** en `generate_v2.py`: `severity_score`, `bp_ratio`, `oxygenation_index`, `age_severity`, `severe_hypotension`, `multi_organ_failure`
- **Indicadores de missing** agregados en el script de entrenamiento: `{col}_is_missing` para vitales con NaN
- **Mantiene** `is_dnr` y `has_chest_tube` como features (no los elimina)

### Diferencias clave en features

| Feature | v1 | v3 | Nota |
|---|---|---|---|
| `max_glucose` | ✅ | ❌ | Eliminado en v2 por ruido |
| `max_lactate` | ❌ (eliminado pre-clean) | ❌ | Ausente en ambos |
| `mean_arterial_pressure` | ✅ (calculado inline) | ❌ | Solo v1 |
| `shock_index` | ✅ | ❌ | Solo v1 |
| `severe_hypoxemia` | ✅ | ❌ | Solo v1 |
| `fever` / `hypothermia` | ✅ | ❌ | Solo v1 |
| `tachycardia_severe` | ✅ | ❌ | Solo v1 |
| `hypotension` | ✅ | `severe_hypotension` | Mismo concepto, nombre diferente |
| `organ_support_count` | ✅ | ❌ (tiene `multi_organ_failure`) | v1 suma 9 procedimientos; v3 suma solo 3 |
| `sofa_partial` | ✅ | ❌ | Solo v1 |
| `age_x_ventilation/hypoxemia/hypotension` | ✅ | ❌ (tiene `age_severity`) | Diferentes interacciones edad |
| `severity_score` | ❌ | ✅ | Solo v3 |
| `bp_ratio` | ❌ | ✅ | Solo v3 |
| `oxygenation_index` | ❌ | ✅ | Solo v3 |
| `{col}_is_missing` flags | ❌ | ✅ | v3 agrega 5 indicadores de missing |
| `is_dnr` | ❌ (dropped) | ✅ | v1 lo elimina, v3 lo mantiene |
| `has_chest_tube` | ❌ (dropped) | ✅ | v1 lo elimina, v3 lo mantiene |

---

## 3. Arquitectura del Modelo

### v1: XGBoost individual con calibración

```
Datos → OneHotEncoder → MICE Imputer → XGBoost → Calibración Isotónica → Predicción
                                          ↑
                                    Optuna (50 trials)
                                    CV scoring: AUPRC
```

**Pipeline detallado:**
1. OneHotEncoder para `gender`, `marital_status`, `race` (fit en train_fit)
2. MICE IterativeImputer (`max_iter=10`) para **todas** las features (fit en train_fit)
3. XGBoost con `scale_pos_weight = ratio` (fijo, no optimizado)
4. Calibración isotónica (`CalibratedClassifierCV(cv="prefit")`) sobre validation set
5. Búsqueda de umbral sobre validation set

### v3: Stacking ensemble con meta-learner

```
Datos → pd.get_dummies → [Sin imputación global]
  ↓
  ├─ XGBoost (NaN nativo) ─→ OOF predictions ─┐
  ├─ LightGBM (NaN nativo) → OOF predictions ─├→ Stack → Meta-learner LR → Predicción
  └─ LR Pipeline ──────────→ OOF predictions ─┘
       (Imputer+Scaler+LR)       ↑
                            Selección automática:
                            Meta-learner vs Media Ponderada
                            (criterio: AUPRC en OOF)
```

**Pipeline detallado:**
1. `pd.get_dummies` para categóricas (más simple que OneHotEncoder explícito)
2. Sin imputación global — XGBoost y LightGBM manejan NaN nativamente
3. Solo LR usa Pipeline con `SimpleImputer(strategy='median')` + `StandardScaler`
4. 5-fold OOF: cada modelo genera predicciones para el fold no visto
5. Test predictions: promedio de 5 modelos de cada fold
6. Meta-learner LR aprende combinación óptima de los 3 modelos
7. Alternativa automática: media ponderada con pesos optimizados por grid search
8. Selección entre meta-learner y media ponderada basada en AUPRC sobre OOF
9. Umbral buscado sobre predicciones OOF (no sobre validation set ni test)

---

## 4. Estrategia de Split y Evaluación

### v1: Triple split + validación en holdout

```
Datos (100%)
  ├─ Train-fit: 64%  → Entrenamiento modelo base
  ├─ Validation: 16% → Calibración + búsqueda de umbral
  └─ Test: 20%       → Evaluación final
```

- `train_test_split(test_size=0.2)` → 80% train, 20% test
- Segundo split: `train_test_split(X_train, test_size=0.2)` → 64% train_fit, 16% val
- **Problema**: El modelo se entrena con solo 64% de los datos
- **Umbral**: Buscado en validation set (16% de los datos → alta varianza)
- **Calibración**: Isotónica sobre validation set (mismos datos que el umbral)

### v3: Split simple con OOF

```
Datos (100%)
  ├─ Train: 80%  → Entrenamiento + OOF 5-fold (todo el train se usa)
  └─ Test: 20%   → Evaluación final SOLO con umbral determinado en OOF
```

- `train_test_split(test_size=0.2)` → 80% train, 20% test
- **Sin validation set separado** — no se necesita porque:
  - El umbral se busca en predicciones OOF (cada ejemplo del train fue predicho por un modelo que NO lo vio)
  - No hay calibración post-hoc (el meta-learner/stacking calibra implícitamente)
- **Ventaja**: 100% del train se usa para entrenar (vs 64% en v1)
- **Métricas honestas**: El umbral OOF no tiene leakage sobre test

---

## 5. Optimización de Hiperparámetros

### v1: Optuna optimizando AUPRC

| Aspecto | Valor |
|---|---|
| Trials | 50 |
| Métrica CV | `average_precision` (AUPRC) |
| `scale_pos_weight` | Fijo en `ratio` (ratio negatives/positives) |
| `n_estimators` | 200–1500 |
| `max_depth` | 3–8 |
| `learning_rate` | 0.001–0.3 |
| `min_child_weight` | 1–10 (entero) |
| `gamma` | 0.0–5.0 |
| `reg_alpha` | 0.0001–10.0 |
| `reg_lambda` | 0.0001–10.0 |
| CV interno | `cross_val_score` 5-fold |

- Optimiza AUPRC (no F1/F2 directamente)
- `scale_pos_weight` fijo — no se explora como hiperparámetro
- v1 F1 y v1 F2 usan la **misma** función objetivo (AUPRC), solo difieren en cómo buscan el umbral después

### v3: Optuna optimizando F-beta directamente

| Aspecto | XGBoost | LightGBM |
|---|---|---|
| Trials | 75 (ensemble) / 50 (param) | 50 (ensemble) / 30 (param) |
| Métrica CV | F2 (o F1) con threshold search por fold | F2 (o F1) con threshold search por fold |
| `scale_pos_weight` | **Optimizado** (ratio×0.3 a ratio×3.0) | N/A (usa `class_weight='balanced'`) |
| `n_estimators` | 100–1500 | 100–1500 |
| `max_depth` | 2–8 | 2–10 |
| `learning_rate` | 0.003–0.15 | 0.003–0.15 |
| `min_child_weight` | 1–30 (float) | N/A |
| `min_child_samples` | N/A | 3–100 |
| `num_leaves` | N/A | 7–255 |
| `gamma` | 0.0–8.0 | N/A |
| `reg_alpha` | 0.001–15.0 | 0.001–15.0 |
| `reg_lambda` | 0.001–15.0 | 0.001–15.0 |

**Diferencias clave:**
1. v3 optimiza la métrica objetivo DIRECTAMENTE (F2 o F1) en lugar de AUPRC como proxy
2. v3 trata `scale_pos_weight` como hiperparámetro optimizable (rango amplio alrededor del ratio natural)
3. v3 tiene espacio de búsqueda más amplio: `min_child_weight` hasta 30 (vs 10), `gamma` hasta 8 (vs 5), `reg_alpha/lambda` hasta 15 (vs 10)
4. v3 optimiza LightGBM también (v1 no tiene LightGBM)
5. v3 usa `logloss` como `eval_metric` (vs `aucpr` en v1) — la métrica de Optuna es la verdadera función objetivo

---

## 6. Manejo de Datos Faltantes

### v1: MICE (IterativeImputer)
```python
imputer = IterativeImputer(max_iter=10, random_state=42)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_enc), columns=X_train_enc.columns)
```
- Imputación multivariada iterativa para **todas** las features
- Fit solo en train_fit, transform en val y test
- **Problema**: XGBoost maneja NaN nativamente MEJOR que valores imputados, porque sabe distinguir "dato faltante" de "valor real". MICE introduce ruido al generar valores sintéticos.

### v3: NaN nativo + indicadores de missing
```python
# XGB y LGB reciben NaN directamente (aprenden splits óptimos para missing)
# Solo LR necesita imputación:
lr_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(...))
])
```
- XGBoost y LightGBM reciben NaN directamente → aprenden la dirección óptima del split para cada feature con missing
- Solo Logistic Regression requiere imputación (SimpleImputer mediana, más simple y robusto que MICE para este caso)
- Además, v3 agrega **indicadores binarios** `{col}_is_missing` para las 5 columnas de vitales, permitiendo al modelo usar el hecho de que un dato está ausente como información predictiva

---

## 7. Calibración y Combinación de Modelos

### v1: Calibración isotónica post-hoc
```python
calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
calibrated_model.fit(X_val_imp, y_val)
```
- Un solo modelo XGBoost
- Calibración isotónica sobre validation set (16% de los datos)
- **Riesgo**: La calibración isotónica sobre pocos datos (16%) puede sobreajustar, especialmente la clase minoritaria

### v3: Stacking ensemble como calibrador implícito

El stacking en v3 cumple tres funciones simultáneas:

1. **Combinación de modelos**: Aprovecha las fortalezas de XGB (interacciones complejas), LGB (eficiencia, regularización diferente), y LR (relaciones lineales, calibración natural)

2. **Calibración implícita**: Las predicciones OOF alimentan un meta-learner LR, que al operar en el espacio [0,1]×3 actúa como calibrador natural. Los coeficientes aprendidos (XGB=2.167, LGB=0.247, LR=2.046) muestran que da más peso a XGB y LR.

3. **Selección automática de método**: Compara meta-learner vs media ponderada optimizada por AUPRC en OOF, eligiendo la mejor combinación sin intervención manual.

---

## 8. Estrategia de Selección de Umbral

### v1: Umbral en validation set
```python
thresholds_search = np.linspace(0.01, 0.99, 200)
f1_scores_val = [f1_score(y_val, y_probs_val > thr, ...) for thr in thresholds_search]  # F1
f2_scores_val = [fbeta_score(y_val, y_probs_val > thr, beta=2.0, ...) for thr in thresholds_search]  # F2
```
- Busca en 200 umbrales sobre **validation set** (16% datos, ~130 ejemplos de test)
- La decisión del umbral depende de un subconjunto pequeño → alta varianza
- El mismo validation set se usa para calibración Y para umbral → posible sobreajuste compuesto

### v3: Umbral en predicciones OOF
```python
optimal_threshold, oof_f2 = find_best_threshold_f2(y_train, final_oof_probs)
# final_oof_probs son predicciones OOF: cada ejemplo fue predicho por un modelo que NO lo vio
```
- Busca en 200 umbrales sobre **predicciones OOF** del conjunto completo de train (80% datos, ~1653 ejemplos)
- Cada predicción OOF fue generada por un modelo que NO vio ese ejemplo → sin leakage
- ~12× más datos para estimar el umbral → estimación mucho más estable
- Adicionalmente, v3 reporta `test_f2_leaky` (umbral buscado en test) para comparación justa con v1/v2

---

## 9. Métricas Comparativas

### Resultados en Test Set

| Métrica | v1 XGBoost F1 | v1 XGBoost F2 | v3 Ensemble | v3 F1-opt | v3 F2-opt |
|---|---|---|---|---|---|
| **AUROC** | 0.7792 | 0.7628 | **0.7942** | 0.7911 | **0.7944** |
| **AUPRC** | 0.4747 | 0.3122 | **0.5507** | 0.5449 | **0.5516** |
| **Brier Score** | 0.1547 | 0.0890 | — | — | — |
| **F1 (test)** | — | — | 0.4667 | **0.5372** | — |
| **F2 (test, OOF thresh)** | — | — | **0.6711** | — | 0.6627 |
| **F2 (test, leaky)** | — | — | 0.6759 | — | 0.6759 |
| **CV AUROC** | 0.7574 | 0.7658 | — | — | — |
| **CV AUPRC** | 0.4797 | 0.3214 | — | — | — |
| **Umbral óptimo (F1)** | 0.1676 | — | — | 0.2858 | — |
| **Umbral óptimo (F2)** | — | 0.0937 | 0.1085 | — | 0.1085 |

### Mejoras absolutas v3 vs v1

| Métrica | v1 mejor | v3 mejor | Δ absoluto | Δ relativo |
|---|---|---|---|---|
| **AUROC** | 0.7792 (F1) | 0.7944 (F2-opt) | **+0.0152** | +2.0% |
| **AUPRC** | 0.4747 (F1) | 0.5516 (F2-opt) | **+0.0769** | +16.2% |

La mejora más significativa es en **AUPRC (+16.2%)**, que es la métrica más informativa para datasets desbalanceados ya que no se beneficia del azar como AUROC.

---

## 10. Diferencias Adicionales Notables

### Encoding categórico
- **v1**: `OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")` — fit explícito en train_fit
- **v3**: `pd.get_dummies(drop_first=True)` — aplicado a todo X antes del split (técnicamente correcto porque solo crea columnas binarias, no aprende nada de los datos)

### Manejo de clases desbalanceadas
- **v1**: Solo `scale_pos_weight` fijo (ratio exacto negatives/positives)
- **v3**: 
  - XGB: `scale_pos_weight` optimizado por Optuna en rango [ratio×0.3, ratio×3.0]
  - LGB: `class_weight='balanced'` (equivalente automático)
  - LR: `class_weight='balanced'`

### SHAP / Explicabilidad
- **v1**: SHAP sobre el modelo calibrado (`calibrated_model.estimator`)
- **v3**: SHAP sobre el XGBoost base del ensemble (el modelo más potente del stacking)

### Reproducibilidad
- **v1**: Guarda `best_params.json` con hiperparámetros y métricas incrementalmente
- **v3**: Guarda `ensemble_metrics.json` completo + `model.joblib` con meta-model, params, umbral, y lista de features

---

## 11. Diagrama de Evolución

```
v1 (XGBoost solo)                    v3 (Stacking ensemble)
─────────────────                    ──────────────────────
dataset_v1_clean.csv                 dataset_v2_engineered.csv
  └─ max_glucose, features inline      └─ Sin glucose/lactate, features centralizados

1 modelo (XGBoost)                   3 modelos (XGB + LGB + LR)
MICE para todo                       NaN nativo (XGB/LGB) + mediana (LR)
Optuna → AUPRC                       Optuna → F-beta directo
scale_pos_weight fijo                scale_pos_weight optimizado
Calibración isotónica                Stacking como calibrador implícito
Umbral en val set (16%)              Umbral en OOF (80%)
Triple split 64/16/20               Split 80/20 + 5-fold OOF
```

---

## 12. Conclusión

El modelo v3 representa una evolución significativa sobre v1 en múltiples dimensiones:

1. **Capacidad predictiva**: AUPRC mejoró un 16.2%, indicando que las probabilidades del modelo v3 son sustancialmente más informativas para separar las clases.

2. **Robustez metodológica**: La evaluación con umbrales OOF elimina el data leakage presente en v1 (umbral en validation set pequeño), produciendo estimaciones más confiables del rendimiento real.

3. **Eficiencia de datos**: v3 usa el 100% del training set para entrenar (vs 64% en v1), lo cual es crítico con solo ~2067 muestras totales.

4. **Manejo de missing values**: Aprovechar la capacidad nativa de XGB/LGB para NaN es superior a la imputación MICE, que introduce valores sintéticos que el modelo no puede distinguir de mediciones reales.

5. **Diversidad de modelos**: El stacking con XGB+LGB+LR captura diferentes patrones en los datos (interacciones no lineales, regularización diferente, relaciones lineales) y el meta-learner aprende la combinación óptima.
