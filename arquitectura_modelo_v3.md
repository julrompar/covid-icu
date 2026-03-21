# Explicación en Detalle: Arquitectura del Modelo v3

> **Fecha de redacción:** marzo 2026  
> **Script de referencia:** `models/v3-models/train_v3_ensemble.py`  
> **Dataset:** `datasets/v2/dataset_v2_engineered.csv`  
> **Resultados:** `models/v3-models/ensemble_metrics.json`

---

## Índice

1. [Contexto y motivación](#1-contexto-y-motivación)
2. [Dataset y pipeline de datos](#2-dataset-y-pipeline-de-datos)
3. [Visión general de la arquitectura](#3-visión-general-de-la-arquitectura)
4. [Nivel 0 — Modelos base](#4-nivel-0--modelos-base)
   - 4.1 XGBoost
   - 4.2 LightGBM
   - 4.3 Regresión Logística (Pipeline)
5. [Predicciones OOF (Out-of-Fold)](#5-predicciones-oof-out-of-fold)
6. [Nivel 1 — Combinación de modelos](#6-nivel-1--combinación-de-modelos)
7. [Optimización de hiperparámetros con Optuna](#7-optimización-de-hiperparámetros-con-optuna)
8. [Manejo del desbalanceo de clases](#8-manejo-del-desbalanceo-de-clases)
9. [Manejo de valores missing](#9-manejo-de-valores-missing)
10. [Selección del umbral de clasificación](#10-selección-del-umbral-de-clasificación)
11. [Métricas de evaluación y resultados](#11-métricas-de-evaluación-y-resultados)
12. [Decisiones de diseño clave](#12-decisiones-de-diseño-clave)
13. [Diagrama de flujo completo](#13-diagrama-de-flujo-completo)

---

## 1. Contexto y motivación

El modelo v3 es un **ensemble de stacking de dos niveles** desarrollado para predecir la mortalidad a 30 días de pacientes con COVID-19 ingresados en UCI. El contexto clínico impone restricciones importantes al diseño:

- **Objetivo clínico:** maximizar la sensibilidad (recall) de muertes correctamente identificadas, ya que un falso negativo (no detectar un paciente que va a morir) tiene mayor coste que un falso positivo.
- **Métrica de optimización principal:** F2-score (ponderación 2:1 de recall sobre precisión, por definición de la métrica $F_\beta$ con $\beta = 2$).
- **Desbalanceo estructural:** ~23% de positivos (fallecidos), ratio negativo/positivo ≈ 3.3:1.

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}, \quad \beta = 2$$

El modelo v3 surge de un proceso evolutivo desde modelos más simples (v1: XGBoost único con SMOTE, v2: stacking básico) e incorpora siete mejoras fundamentales respecto a versiones anteriores:

| # | Mejora | Versiones anteriores | v3 |
|---|--------|---------------------|-----|
| 1 | Manejo de desbalanceo | SMOTE (sobremuestreo sintético) | `scale_pos_weight` / `class_weight` |
| 2 | Features de línea arterial | Incluidas (81% NaN) | Eliminadas (solo indicador binario) |
| 3 | Estrategia de umbral | Val set separado (leakage) | OOF sin leakage |
| 4 | Optimización LightGBM | Parámetros fijos | Optuna (50 trials) |
| 5 | Combinación ensemble | Pesos fijos | Selección automática meta-learner vs. media ponderada |
| 6 | Imputación en árbol | SimpleImputer para XGB/LGB | NaN nativo (sin imputación) |
| 7 | Espacio de búsqueda | Reducido | Ampliado |

---

## 2. Dataset y pipeline de datos

### 2.1 Origen de los datos

Los datos provienen de **MIMIC-IV**, un dataset clínico de UCI del Hospital Beth Israel Deaconess (Boston). Las tablas utilizadas incluyen `chartevents`, `labevents`, `procedureevents`, `admissions`, `icustays`, y `patients`.

El dataset final (`dataset_v2_engineered.csv`) contiene **2067 pacientes** con estancias en UCI relacionadas con COVID-19.

### 2.2 Variable objetivo

```python
df['target'] = df['dod_within_30_days'].notnull().astype(int)
```

Binaria: `1` si el paciente falleció dentro de los 30 días desde el ingreso a UCI, `0` en caso contrario.

### 2.3 Features disponibles

El dataset v2 incorpora feature engineering centralizado en `datasets/v2/generate_v2.py` que construye variables compuestas de relevancia clínica:

| Feature compuesta | Descripción |
|---|---|
| `severity_score` | Score de severidad basado en signos vitales y laboratorio |
| `bp_ratio` | Cociente PAS/PAD (refleja estado hemodinámico) |
| `oxygenation_index` | Estimación del índice de oxigenación |
| `age_severity_interaction` | Interacción edad × severidad |
| `is_dnr` | Indicador de órdenes de "no reanimar" |
| `has_chest_tube` | Indicador de drenaje torácico |
| `{col}_is_missing` | Indicadores binarios de ausencia de datos (5 columnas) |

### 2.4 Missing indicators

Para 5 columnas de signos vitales frecuentemente ausentes en UCIs, se crean indicadores binarios explícitos **antes** del split:

```python
cols_with_nans = ['min_bp_systolic', 'min_bp_diastolic',
                  'min_oxygen_saturation', 'max_pulse', 'max_temp']
for col in cols_with_nans:
    df[f'{col}_is_missing'] = df[col].isna().astype(int)
```

Estos indicadores permiten al modelo aprender que la **ausencia misma** de un dato (p. ej., no tener registrada la saturación de oxígeno) puede ser informativa.

### 2.5 Split train/test

```
Split estratificado: 80% train (1653 pacientes) / 20% test (414 pacientes)
Positivos train: ~382  |  Positivos test: ~96
Ratio negativo/positivo: ~3.3:1
```

Se usa `stratify=y` para preservar la proporción de positivos en ambas particiones. El conjunto de test **nunca se toca** durante el entrenamiento ni la selección de umbral.

---

## 3. Visión general de la arquitectura

El modelo v3 implementa un **stacking de dos niveles**:

```
                    ┌─────────────────────────────────────┐
                    │         NIVEL 0 — Modelos Base       │
                    │                                     │
              ┌─────┤ XGBoost  (Optuna, 75 trials)        │
  X_train ───►├─────┤ LightGBM (Optuna, 50 trials)        │──► OOF predictions (1653×3)
              └─────┤ Logistic Regression (Pipeline)      │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       NIVEL 1 — Meta-combinación     │
                    │                                     │
                    │  Opción A: Meta-learner LR           │
                    │  Opción B: Media ponderada (5³=125)  │
                    │                                     │
                    │  Selección automática por AUPRC OOF  │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       UMBRAL DE CLASIFICACIÓN        │
                    │  Búsqueda OOF — maximizar F2         │
                    │  Umbral óptimo: 0.1085               │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                              Predicción final
```

La clave de este diseño es que **el test set permanece completamente ajeno** durante la fase de entrenamiento de los modelos base, del meta-learner y de la selección del umbral. La estimación insesgada del rendimiento se obtiene siempre sobre OOF (para entrenamiento interno) y sobre el test set reservado (para el reporte final).

---

## 4. Nivel 0 — Modelos base

Se eligieron tres modelos con características complementarias deliberadas:

### 4.1 XGBoost

**¿Por qué XGBoost?**

XGBoost (eXtreme Gradient Boosting) es un clasificador basado en boosting por gradiente que construye árboles de decisión de forma secuencial, donde cada árbol nuevo corrige los errores del anterior.

**Ventajas específicas para este problema:**
1. **Soporte nativo de NaN:** XGBoost aprende la dirección óptima para vals missing directamente durante el entrenamiento (aprende si un NaN debe ir a la rama izquierda o derecha del árbol). Esto evita imputación artificial.
2. **`scale_pos_weight`:** Parámetro que multiplica el gradiente de los positivos por un escalar, compensando el desbalanceo sin generar muestras sintéticas.
3. **Alta capacidad de modelado no lineal:** Captura interacciones complejas entre features clínicas.
4. **Regularización integrada:** Parámetros `reg_alpha` (L1) y `reg_lambda` (L2) previenen sobreajuste.

**Hiperparámetros óptimos encontrados:**

| Parámetro | Valor | Significado |
|---|---|---|
| `n_estimators` | 1201 | Número de árboles |
| `max_depth` | 4 | Profundidad máxima por árbol |
| `learning_rate` | 0.0250 | Shrinkage (step size) |
| `subsample` | 0.487 | Fracción de filas por árbol |
| `colsample_bytree` | 0.331 | Fracción de columnas por árbol |
| `min_child_weight` | 2.75 | Peso mínimo en hoja |
| `gamma` | 7.94 | Umbral de ganancia para split |
| `reg_alpha` | 10.75 | Regularización L1 |
| `reg_lambda` | 2.37 | Regularización L2 |
| `scale_pos_weight` | **8.11** | Peso de positivos (ratio ≈ 3.3, amplificado) |

El valor de `scale_pos_weight = 8.11` (mayor que el ratio 3.3:1) fue optimizado por Optuna, indicando que el modelo requiere sobreponderar bastante más los positivos para maximizar F2 en este dataset.

---

### 4.2 LightGBM

**¿Por qué LightGBM?**

LightGBM es otro clasificador basado en boosting, pero con una implementación diferente a XGBoost:
- Crece los árboles **hoja por hoja** (_leaf-wise_) en lugar de nivel por nivel, logrando mayor precisión con los mismos árboles.
- Usa **Histogram-based splitting**, que agrupa los valores de cada feature en cubos discretos, acelerando el entrenamiento y reduciendo el uso de memoria.

**Complementariedad con XGBoost:**

Aunque ambos son GBM, sus mecanismos internos difieren suficientemente para generar **predicciones correlacionadas pero no idénticas**. En un ensemble, la baja correlación entre predicciones base es la clave para reducir la varianza del modelo combinado.

**Parámetro clave: `num_leaves`**

```python
'num_leaves': trial.suggest_int('num_leaves', 7, 255)
```

`num_leaves` controla la complejidad del árbol en LightGBM de forma más directa que `max_depth`. El valor óptimo encontrado fue **45 hojas**, lo que corresponde a un árbol de complejidad moderada.

**Manejo de desbalanceo:** `class_weight='balanced'` (equivalente a ponderar cada clase por el inverso de su frecuencia), en lugar de `scale_pos_weight`.

**Hiperparámetros óptimos encontrados:**

| Parámetro | Valor |
|---|---|
| `n_estimators` | 1337 |
| `max_depth` | 5 |
| `num_leaves` | **45** |
| `learning_rate` | 0.00545 |
| `subsample` | 0.591 |
| `colsample_bytree` | 0.822 |
| `min_child_samples` | 85 |
| `reg_alpha` | 2.80 |
| `reg_lambda` | 6.04 |

---

### 4.3 Regresión Logística (Pipeline)

**¿Por qué Regresión Logística?**

La inclusión de un modelo lineal en el ensemble cumple un papel estratégico bien definido:

1. **Sesgo diferente:** La LR aprende combinaciones lineales de features, mientras que XGBoost y LightGBM aprenden interacciones no lineales. Diferentes sesgos generan diferentes patrones de error, que el meta-learner puede combinar.

2. **Regularización fuerte:** `C=0.1` (equivalente a alta regularización L2) evita sobreajuste y produce predicciones de probabilidad bien calibradas.

3. **Ancla de estabilidad:** En folds donde los modelos de árbol sobreajustan, la LR actúa como regularizador del ensemble.

**Pipeline de preprocesamiento:**

A diferencia de XGBoost y LightGBM, la Regresión Logística **no puede manejar NaN** ni features en escalas muy distintas. Por ello se usa un `Pipeline` de scikit-learn:

```python
lr_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # NaN → mediana de entrenamiento
    ('scaler', StandardScaler()),                    # Normalización Z
    ('lr', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        C=0.1                                        # Alta regularización L2
    ))
])
```

**Importante:** El `Pipeline` se ajusta **solo sobre los datos de entrenamiento de cada fold**, previniendo que información del fold de validación contamine la imputación de medianas o los parámetros del escalador. Esto es correcto metodológicamente.

---

## 5. Predicciones OOF (Out-of-Fold)

### 5.1 ¿Qué son las predicciones OOF?

Las predicciones Out-of-Fold son el mecanismo central del stacking. El problema a resolver es: **¿cómo generamos features para el meta-learner sin que estén contaminadas por sobreajuste?**

Si entrenamos XGBoost con todos los datos de train y luego usamos sus predicciones sobre esos mismos datos como feature para el meta-learner, el meta-learner aprenderá a confiar en un clasificador que ha memorizado el conjunto. El resultado sería un modelo que funciona bien en train pero mal en test.

**La solución: validación cruzada k-fold para generar predicciones "honestas".**

### 5.2 Mecanismo de validación cruzada estratificada (k=5)

El conjunto de entrenamiento (1653 muestras) se divide en **5 folds estratificados** (preservando la proporción de positivos en cada fold).

```
Train set (1653 muestras)
├── Fold 1: val={0..330}    train={331..1652}
├── Fold 2: val={330..661}  train={0..329, 662..1652}
├── Fold 3: val={661..991}  train={0..660, 992..1652}
├── Fold 4: val={991..1322} train={0..990, 1323..1652}
└── Fold 5: val={1322..1652} train={0..1321}
```

Para cada fold $k$ y cada modelo base $m$:
1. Entrenar $m$ sobre las 4-5 partes restantes.
2. Predecir probabilidades sobre la parte $k$ (que el modelo **nunca ha visto**).
3. Almacenar esas predicciones en `oof_m[val_idx]`.

Al completar los 5 folds, se obtiene un vector de predicciones `oof_m` de longitud 1653 donde **cada muestra fue predicha por un modelo que no la entrenó**.

```python
oof_xgb = np.zeros(len(X_train))  # 1653 valores
oof_lgb = np.zeros(len(X_train))
oof_lr  = np.zeros(len(X_train))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr  = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr  = y_train.iloc[train_idx]

    # Entrenar XGB sobre train_idx, predecir sobre val_idx
    xgb_model = xgb.XGBClassifier(**xgb_best_params)
    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]  # ← OOF prediction

    # Simultaneamente, acumular predicciones de test (promedio entre folds)
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
```

### 5.3 Predicciones sobre test: promedio entre folds

Para el test set, se acumulan las predicciones de **cada uno de los 5 modelos** entrenados en los distintos folds y se promedian:

$$\hat{p}_{test}^{(m)} = \frac{1}{K} \sum_{k=1}^{K} \hat{p}_{test, k}^{(m)}$$

Este promedio reduce la varianza de la predicción final respecto a usar un solo modelo.

### 5.4 ¿Por qué k=5?

- **k=5** es un compromiso estándar entre:
  - **Varianza:** con k pequeño (e.g., k=3), cada fold usa menos datos de entrenamiento → más varianza en los OOF.
  - **Sesgo:** con k grande (e.g., k=10), el coste computacional se multiplica y el solapamiento entre folds aumenta.
- Con 1653 muestras y k=5, cada fold de validación tiene ~330 muestras (~76 positivos), suficiente para estimar probabilidades de forma estable.

### 5.5 Resultado: la matriz de stacking

Tras completar los 5 folds, se dispone de:

```python
oof_stack  = np.column_stack([oof_xgb, oof_lgb, oof_lr])   # (1653, 3)
test_stack = np.column_stack([test_xgb, test_lgb, test_lr]) # (414, 3)
```

Cada fila de `oof_stack` es un vector de 3 probabilidades para una misma muestra, generadas por modelos que no la vieron. Esta es la feature matrix que alimenta al meta-learner.

---

## 6. Nivel 1 — Combinación de modelos

El meta-learner recibe la **matrix de stacking OOF** (1653 × 3) y aprende a combinar las predicciones de los tres modelos base.

### 6.1 Opción A — Meta-learner Logístico

```python
meta_model = LogisticRegression(random_state=42, max_iter=1000)
meta_model.fit(oof_stack, y_train)
```

El meta-learner aprende **coeficientes** para cada modelo base:

| Modelo | Coeficiente |
|--------|------------|
| XGBoost | **2.167** |
| LightGBM | **0.247** |
| Logistic Reg. | **2.046** |

La predicción final es:

$$\hat{p} = \sigma(2.167 \cdot p_{XGB} + 0.247 \cdot p_{LGB} + 2.046 \cdot p_{LR} + b)$$

Donde $\sigma$ es la función sigmoide. El meta-learner aprende que XGBoost y LR aportan más señal que LightGBM en este dataset. El bajo coeficiente de LightGBM no significa que sea un mal modelo, sino que su contribución marginal una vez que XGB y LR ya están presentes es menor.

### 6.2 Opción B — Media ponderada optimizada

En paralelo, se realiza una búsqueda exhaustiva sobre todas las combinaciones de pesos enteros del 1 al 5 para cada modelo:

```python
for w1, w2, w3 in itertools.product(range(1, 6), repeat=3):
    oof_avg = (w1 * oof_xgb + w2 * oof_lgb + w3 * oof_lr) / (w1 + w2 + w3)
    _, f2_w = find_best_threshold_f2(y_train, oof_avg)
    # keep best combination
```

Esto evalúa **5³ = 125 combinaciones** diferentes de pesos, optimizando el F2-score sobre las predicciones OOF. La combinación óptima encontrada fue **(2, 5, 1)** (XGB=2, LGB=5, LR=1), que maximiza el F2 OOF.

### 6.3 Selección automática por AUPRC

En lugar de elegir el método por F2 (que requeriría optimizar un umbral sobre OOF y podría introducir sesgo), se usa el **AUPRC** (Area Under the Precision-Recall Curve) como criterio de selección:

```python
oof_auprc_meta = average_precision_score(y_train, oof_meta_probs)
oof_auprc_wavg = average_precision_score(y_train, oof_weighted_avg)

if oof_auprc_wavg > oof_auprc_meta:
    # Usar media ponderada
else:
    # Usar meta-learner
```

**¿Por qué AUPRC?**

- Es **independiente del umbral** (mide la calidad de las probabilidades en todo el rango).
- Especialmente informativa en datasets desbalanceados (penaliza más los errores en la clase positiva que AUROC).
- Evita el sesgo de selección de umbral que introduciría usar F2 directamente.

**Resultado en la última ejecución:** El meta-learner fue seleccionado (AUPRC=0.4634 > 0.4437 de la media ponderada), a pesar de que la media ponderada tenía mayor F2-OOF. Esto indica que las probabilidades del meta-learner son de mejor calidad, aunque por umbral óptimo el F2 sea similar.

---

## 7. Optimización de hiperparámetros con Optuna

### 7.1 ¿Qué es Optuna?

Optuna es un framework de optimización bayesiana de hiperparámetros. A diferencia de una búsqueda en rejilla (Grid Search) que prueba todas las combinaciones posibles, o una búsqueda aleatoria (Random Search) que muestrea sin estrategia, Optuna usa el algoritmo **TPE (Tree-structured Parzen Estimator)**:

1. Inicia con muestras aleatorias para explorar el espacio.
2. Modela la función objetivo con dos distribuciones: una para configuraciones "buenas" (aquellas que superan un umbral de calidad) y otra para las "malas".
3. En cada trial siguiente, elige el punto que maximiza la razón $p(\text{buena}) / p(\text{mala})$.

Este enfoque converge hacia configuraciones óptimas mucho más eficientemente que Grid Search.

### 7.2 Función objetivo

La función objetivo es el **F2-score promedio en 5-fold CV** sobre el conjunto de entrenamiento:

```python
def xgb_objective(trial):
    params = { ... }  # Hiperparámetros sugeridos por Optuna
    f2_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds_proba = model.predict_proba(X_val)[:, 1]
        _, best_f2 = find_best_threshold_f2(y_val, preds_proba)
        f2_scores.append(best_f2)
    return np.mean(f2_scores)
```

Para cada trial, se entrena el modelo 5 veces (una por fold) y se devuelve el F2 promedio. La búsqueda del umbral óptimo se hace dentro de la función objetivo sobre cada fold de validación.

### 7.3 Presupuesto de trials

| Modelo | Trials | Razón |
|--------|--------|-------|
| XGBoost | 75 | Espacio más complejo (10 hiperparámetros, incluyendo `scale_pos_weight`) |
| LightGBM | 50 | Espacio ligeramente menor |

Cada trial implica 5 entrenamientos completos del modelo, por lo que **XGBoost se entrena 375 veces** durante la optimización y **LightGBM 250 veces**.

---

## 8. Manejo del desbalanceo de clases

El dataset tiene un ratio negativo/positivo de aproximadamente **3.3:1**. En versiones anteriores se usó **SMOTE** (Synthetic Minority Over-sampling Technique), que genera muestras sintéticas de la clase minoritaria interpolando entre muestras existentes.

**¿Por qué se abandona SMOTE en v3?**

1. **Ruido sintético:** Las muestras generadas por SMOTE no corresponden a pacientes reales. En un dominio médico con relaciones complejas entre features, las interpolaciones pueden generar combinaciones clínicamente imposibles.
2. **Sesgo en OOF:** Si se aplica SMOTE antes del split OOF, las muestras sintéticas pueden filtrarse entre folds (leakage). Si se aplica dentro de cada fold, el coste computacional aumenta y las predicciones OOF ya no son directamente comparables con el test set (que no tiene SMOTE).
3. **El desbalanceo 3.3:1 es manejable:** No es un desbalanceo extremo. Los modificadores de peso internos de los modelos son suficientes.

**Estrategias usadas en v3:**

| Modelo | Estrategia |
|--------|-----------|
| XGBoost | `scale_pos_weight = 8.11` (amplificado más allá del ratio, optimizado por Optuna) |
| LightGBM | `class_weight = 'balanced'` (pondera por inverso de frecuencia = `scale_pos_weight` ≈ 3.3) |
| Logistic Reg. | `class_weight = 'balanced'` |

La diferencia entre `scale_pos_weight = 8.11` (XGB) y el valor equivalente a `balanced` (≈ 3.3) refleja que Optuna descubrió que XGBoost necesita sobreponderar aún más los positivos para maximizar F2, posiblemente debido a su mayor tendencia a sobreajustar en la clase mayoritaria.

---

## 9. Manejo de valores missing

Un desafío central en datos de UCI es la alta proporción de valores faltantes. Los datos clínicos son missing completamente al azar (MCAR), al azar condicionalmente (MAR) o no al azar (MNAR). En este contexto, ciertas variables como la saturación de oxígeno pueden no estar registradas precisamente porque el paciente estaba estable (MNAR negativo) o porque el monitor falló (MCAR).

El modelo v3 usa un enfoque de **tres capas**:

### Capa 1: Eliminación de features con > 70% missing

Las variables de línea arterial (presión arterial invasiva, etc.) tenían un 81% de valores faltantes. Con tan pocos valores reales, cualquier estrategia de imputación introduce más ruido que señal. Se eliminan del dataset y se sustituyen por un único indicador binario `has_arterial_line`.

### Capa 2: Indicadores de ausencia (Missing Indicators)

Para las 5 columnas con NaN moderado (`min_bp_systolic`, `min_bp_diastolic`, `min_oxygen_saturation`, `max_pulse`, `max_temp`), se crean columnas binarias `{col}_is_missing`. Esto transforma la ausencia de un dato en una feature explícita, capturando información potencialmente predictiva sobre el proceso de generación de datos.

### Capa 3: Manejo nativo vs. imputación

- **XGBoost y LightGBM:** Los árboles manejan NaN de forma nativa. Durante el entrenamiento, aprenden si un NaN debe clasificarse junto a los valores altos o bajos de esa feature. No se realiza ninguna imputación previa.
- **Logistic Regression:** No puede manejar NaN. El `Pipeline` dentro de cada fold realiza `SimpleImputer(strategy='median')` sobre los datos de entrenamiento de ese fold, y aplica la mediana aprendida (no la del fold completo) al fold de validación.

---

## 10. Selección del umbral de clasificación

Un clasificador binario basado en probabilidades requiere un **umbral** para convertir probabilidades en clases:

$$\hat{y} = \begin{cases} 1 & \text{si } \hat{p} \geq \tau \\ 0 & \text{si } \hat{p} < \tau \end{cases}$$

Por defecto, $\tau = 0.5$, pero en datasets desbalanceados con métricas asimétricas, el umbral óptimo rara vez es 0.5.

### Búsqueda del umbral

Se implementa una búsqueda en rejilla fina sobre 200 puntos entre 0.01 y 0.99:

```python
def find_best_threshold_f2(y_true, y_probs, n=200):
    thresholds = np.linspace(0.01, 0.99, n)
    f2_scores = [fbeta_score(y_true, (y_probs >= t).astype(int), beta=2, zero_division=0)
                 for t in thresholds]
    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx], f2_scores[best_idx]
```

### ¿Sobre qué datos se busca el umbral?

El umbral se busca sobre las **predicciones OOF del conjunto de entrenamiento**:

```python
optimal_threshold, oof_f2 = find_best_threshold_f2(y_train, final_oof_probs)
```

**¿Por qué OOF y no test?** Buscar el umbral sobre el test set introduciría una forma de leakage: estaríamos "mirando" el test set para tomar una decisión de modelado. Las predicciones OOF son una estimación honesta del comportamiento del modelo sobre datos no vistos, por lo que el umbral seleccionado sobre OOF es un umbral válido para aplicar al test.

**Umbral óptimo encontrado:** $\tau = 0.1085$

Este valor tan bajo (< 0.5) refleja directamente el objetivo de maximizar la sensibilidad: el modelo prefiere clasificar como positivo (muerte) a muchos pacientes, aceptando más falsos positivos a cambio de no perder verdaderos positivos.

---

## 11. Métricas de evaluación y resultados

### 11.1 Métricas utilizadas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **AUROC** | $\int_0^1 TPR \, d(FPR)$ | Capacidad discriminativa general |
| **AUPRC** | $\int_0^1 P \, d(R)$ | Discriminación en clase positiva (más informativa que AUROC en desbalanceo) |
| **F2** (OOF thresh) | $F_\beta$ con $\beta=2$, $\tau=0.1085$ | Evaluación honesta sin leakage |
| **F2** (test thresh) | $F_\beta$ con $\tau$ óptimo sobre test | Referencia comparable con v2 (informado) |
| **F1** | $F_\beta$ con $\beta=1$ | Balance recall-precisión |

### 11.2 Resultados del modelo v3 ensemble

| Métrica | Modelo ensemble | XGBoost solo | LightGBM solo | LR solo |
|---------|----------------|--------------|---------------|---------|
| **AUROC** | **0.7942** | — | — | — |
| **AUPRC** | **0.5507** | — | — | — |
| **F2 (OOF thresh)** | **0.6711** | — | — | — |
| **F2 (leaky)** | **0.6759** | — | — | — |
| **F1** | **0.4667** | — | — | — |

### 11.3 Comparativa con versiones anteriores

| Versión | Arquitectura | AUROC | AUPRC | F2 |
|---------|-------------|-------|-------|-----|
| v1 XGBoost F1 | XGB + SMOTE + calibración isotónica | 0.7792 | 0.4747 | ~0.61 |
| v1 XGBoost F2 | XGB + SMOTE + umbral agresivo | 0.7628 | 0.3122 | ~0.67 |
| **v3 Ensemble** | Stacking XGB+LGB+LR | **0.7942** | **0.5507** | **0.6711** |

El modelo v3 supera al v1 en las tres métricas principales:
- **AUROC +1.9%** respecto al mejor v1 (F1-optimized)
- **AUPRC +16.0%** respecto al mejor v1 — la mejora más significativa
- **F2 comparable** al v1 F2 optimizado, pero sin leakage de umbral

La mejora en AUPRC (+16%) es especialmente relevante: indica que las probabilidades calibradas del ensemble son sustancialmente mejores que las del modelo v1, lo que a su vez mejora la validez de cualquier umbral seleccionado.

---

## 12. Decisiones de diseño clave

### 12.1 ¿Por qué tres modelos y no más?

Añadir más modelos base (e.g., Random Forest, SVM, redes neuronales) tiene rendimientos decrecientes. Los tres modelos seleccionados cubren el espacio de sesgos más relevante:
- **XGBoost / LightGBM:** Alta capacidad no lineal, complementarios en mecanismo interno.
- **Logistic Regression:** Lineal, bien calibrada, ancla el ensemble.

Modelos adicionales añaden coste computacional sin garantizar mejora, y aumentan el riesgo de que el meta-learner sobreajuste a los OOF (cuanto más columnas tiene `oof_stack`, más parámetros aprende el meta-learner).

### 12.2 ¿Por qué Regresión Logística como meta-learner?

El meta-learner es intencionalmente simple. Si fuera complejo (e.g., XGBoost como meta-learner), podría sobreajustar a los OOF, especialmente con solo 3 features. La LR sin regularización adicional (solo L2 por defecto leve) es suficiente para aprender la combinación óptima de tres probabilidades calibradas.

### 12.3 ¿Por qué AUPRC para seleccionar el método de ensemble?

Usar F2 para seleccionar entre meta-learner y media ponderada crearía un ciclo indeseable: se optimizaría F2 para encontrar los mejores pesos, y luego se elegiría el método con mayor F2. El resultado podría ser que siempre se elija media ponderada (porque fue diseñada para maximizar F2 directamente), aunque el meta-learner produzca mejores probabilidades.

AUPRC mide la calidad de las probabilidades en todo el espectro de umbrales, lo que es una señal más limpia de qué método produce mejor ranking de riesgo.

### 12.4 ¿Por qué 80/20 en lugar de 70/30?

Con 2067 muestras y 23% de positivos, el dataset no es grande. Un split 80/20 deja 1653 muestras para entrenamiento (~382 positivos), lo que es suficiente para entrenar los modelos con Optuna pero no tan generoso que se empiece a perder potencia estadística en el test. Con 70/30, el train solo tendría 283 positivos y la búsqueda de Optuna se volvería más ruidosa.

### 12.5 Umbral 0.1085 y su interpretación clínica

Un umbral de 0.11 significa que el modelo clasifica como "muerte" a cualquier paciente al que asigna al menos un 11% de probabilidad de morir en 30 días. Esto es coherente con el objetivo clínico: en UCI, incluso un 11% de riesgo de mortalidad es clínicamente relevante y justifica intervención intensificada.

---

## 13. Diagrama de flujo completo

```
datos MIMIC-IV
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  PREPROCESAMIENTO (dataset_v2_engineered.csv)                │
│  • Feature engineering: severity_score, bp_ratio, etc.      │
│  • Eliminación de arterial line features (81% NaN)           │
│  • Missing indicators: {col}_is_missing (5 cols)            │
│  • One-hot: gender, marital_status, race                    │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  SPLIT ESTRATIFICADO 80/20                                   │
│  X_train (1653) ────────────────► X_test (414) [reservado]  │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  OPTIMIZACIÓN DE HIPERPARÁMETROS (Optuna TPE)                │
│  XGBoost: 75 trials × 5-fold CV → best_xgb_params           │
│  LightGBM: 50 trials × 5-fold CV → best_lgb_params          │
│  Métrica optimizada: F2-score (CV)                           │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  STACKING — 5-FOLD OOF                                       │
│                                                              │
│  Para cada fold k = 1..5:                                    │
│    XGB.fit(train_k) → oof_xgb[val_k], test_xgb += pred/5   │
│    LGB.fit(train_k) → oof_lgb[val_k], test_lgb += pred/5   │
│    LR_pipe.fit(train_k) → oof_lr[val_k], test_lr += pred/5 │
│                                                              │
│  oof_stack  (1653×3): predicciones OOF sin leakage          │
│  test_stack (414×3):  promedio de 5 modelos por fold        │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  META-COMBINACIÓN                                            │
│  A) Meta-learner LR.fit(oof_stack, y_train)                  │
│     → coefs: XGB=2.17, LGB=0.25, LR=2.05                   │
│                                                              │
│  B) Grid search 125 pesos (1..5)³ sobre OOF                 │
│     → pesos óptimos: (XGB=2, LGB=5, LR=1)                  │
│                                                              │
│  Selección por AUPRC OOF:                                   │
│     Meta-learner: 0.4634 ≥ Media ponderada: 0.4437          │
│     → Seleccionado: Meta-learner LR                         │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  SELECCIÓN DE UMBRAL                                         │
│  Búsqueda sobre final_oof_probs (sin leakage de test)       │
│  Rango: [0.01, 0.99], n=200 puntos                          │
│  Criterio: maximizar F2 OOF                                 │
│  → Umbral óptimo: τ = 0.1085                               │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  EVALUACIÓN FINAL SOBRE TEST SET (nunca visto)               │
│  AUROC = 0.7942                                              │
│  AUPRC = 0.5507                                              │
│  F2 (τ=0.1085, honesto) = 0.6711                            │
│  F2 (τ óptimo test, leaky) = 0.6759                         │
│  F1 = 0.4667                                                 │
└──────────────────────────────────────────────────────────────┘
```

---

*Documento generado para la memoria técnica del proyecto. Para reproducir los resultados: `python models/v3-models/train_v3_ensemble.py` desde la raíz del proyecto.*
