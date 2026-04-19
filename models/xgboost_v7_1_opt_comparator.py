"""
XGBoost v7.1 — Predicción mortalidad UCI COVID-19 a 30 días

Mejoras vs v7:
- SIN imputación: solo procedimientos, PMHX e indicadores → 0
- Resto (vitales, labs, scores, demográficas) se dejan con NaN
- XGBoost maneja nativamente los NaN en los features
- Nota: target y features sin NaN en X_train/X_val/X_test tras split

Criterios de inclusión, ventana 72h, 5 agregaciones y compactación: idénticos a v7
"""

import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import optuna
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    f1_score, fbeta_score, confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE FEATURES
# ============================================================================

DROP_COLS = {"subject_id", "admission_id", "stay_id", "admittime", "dod_within_30_days", "target"}

# Labs de baja cobertura en v7: excluir sus agregaciones numéricas (val_*),
# pero mantener los indicadores _measured
LOW_COVERAGE_LABS_V7 = {'pt', 'crp', 'ferritin', 'dimer', 'lymphocytes', 'neutrophils', 'troponin'}


def _is_low_coverage_numeric(col):
    """
    Devuelve True si la columna es una agregación numérica de un lab de baja cobertura.
    Ejemplos excluidos: lab_pt_val_max, lab_crp_val_first, lab_troponin_val_delta
    Ejemplos conservados: lab_pt_measured, lab_crp_measured
    """
    if not col.startswith('lab_'):
        return False
    parts = col.split('_val_')
    if len(parts) != 2:
        return False
    lab_name = parts[0][4:]
    return lab_name in LOW_COVERAGE_LABS_V7


def get_feature_cols(df):
    """
    Extrae columnas de feature del dataset v7.

    Grupos de features:
    - A: Vitales (5 agg × n vitales)
    - B: Labs alta cobertura (5 agg × n labs)
    - C: Labs cobertura media (5 agg × n labs)
    - D: Labs baja cobertura (solo indicador _measured, sin valores numéricos)
    - E: Procedimientos (is_*, has_*)
    - F: Demográficas (anchor_age, gender, marital_status, race)
    - G: Scores clínicos (sofa_partial, apache2_partial; severity_score solo en OptionC)
    - H: PMHX binarios (pmhx_*)
    - I: Variables binarias clínicas (bin_*)
    """
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLS and not _is_low_coverage_numeric(c)
    ]
    return feature_cols


# ============================================================================
# PREPROCESAMIENTO
# ============================================================================

def preprocess_data(df):
    """
    Preprocesamiento:
    1. Target = dod_within_30_days no-nulo
    2. Encoding one-hot de categóricas
    3. Conversión de booleanos a int
    """
    df = df.copy()
    df['target'] = df['dod_within_30_days'].notnull().astype(int)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    y = df['target']

    # Encoding de categóricas
    cat_cols = [c for c in ['gender', 'marital_status', 'race'] if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Conversión bool → int
    for col in X.select_dtypes(include='bool').columns:
        X[col] = X[col].astype(int)

    return X, y


def minimal_impute_data(X_train, X_val, X_test):
    """
    Imputación MÍNIMA — solo para procedimientos, PMHX e indicadores.
    El resto se deja con NaN para que XGBoost lo maneje.

    - Procedimientos (is_*, has_*): fillna(0) — ausencia = no realizado
    - PMHX (pmhx_*): fillna(0) — antecedente no registrado = ausente
    - Variables binarias clínicas (bin_*): fillna(0)
    - Indicadores _measured: fillna(0)
    - REST (vitales, labs, scores, demográficas): SE DEJAN CON NaN
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    # Columnas que SÍ se imputan a 0 (semántica: "no ocurrió" o "no presente")
    zero_impute_cols = [
        c for c in X_train.columns
        if (c.startswith('is_') or c.startswith('has_')
            or c.startswith('pmhx_') or c.startswith('bin_')
            or c.endswith('_measured'))
    ]

    # Imputar solo esas columnas
    for col in zero_impute_cols:
        X_train[col] = X_train[col].fillna(0)
        X_val[col]   = X_val[col].fillna(0)
        X_test[col]  = X_test[col].fillna(0)

    # Resto: se dejan con NaN (vitales, labs, scores continuos, demográficas)
    # XGBoost lo maneja nativamente

    return X_train, X_val, X_test


# ============================================================================
# OPTUNA CON PRUNING
# ============================================================================

def optimize_hyperparameters(X_train, y_train, ratio, optimize_for='F1', beta=2.0, n_trials=100):
    """
    Búsqueda de hiperparámetros con Optuna.

    Métrica: F1 si optimize_for=='F1', F2 si optimize_for=='F2'
    Pruning: MedianPruner para acelerar búsqueda

    PARÁMETRO BETA (para F2):
    - En esta función: parámetro 'beta=2.0' (línea ~51)
    - Se usa en fbeta_score() para calcular F2 durante CV
    """

    ZERO_IMPUTE = [
        c for c in X_train.columns
        if (c.startswith('is_') or c.startswith('has_')
            or c.startswith('pmhx_') or c.startswith('bin_')
            or c.endswith('_measured'))
    ]

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  ratio,
            "random_state":      42,
            "eval_metric":       "aucpr",
            "early_stopping_rounds": 50,
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_cv_train = X_train.iloc[train_idx].copy()
            X_cv_val   = X_train.iloc[val_idx].copy()
            y_cv_train = y_train.iloc[train_idx]
            y_cv_val   = y_train.iloc[val_idx]

            # Solo imputar columnas de procedimientos/PMHX/bin en el fold
            for col in ZERO_IMPUTE:
                X_cv_train[col] = X_cv_train[col].fillna(0)
                X_cv_val[col]   = X_cv_val[col].fillna(0)

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_val, y_cv_val)],
                verbose=False
            )

            probs = model.predict_proba(X_cv_val)[:, 1]

            if optimize_for == 'F1':
                score = f1_score(y_cv_val, probs >= 0.5, zero_division=0)
            else:  # F2
                # ← PARÁMETRO BETA: cambiar aquí para optimizar con beta diferente
                score = fbeta_score(y_cv_val, probs >= 0.5, beta=beta, zero_division=0)

            cv_scores.append(score)

            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


# ============================================================================
# BÚSQUEDA DE UMBRAL ÓPTIMO
# ============================================================================

def find_optimal_threshold(y_val, y_probs_val, optimize_for='F1', beta=2.0):
    """
    Busca el umbral que maximiza F1 o F2 en el validation set.

    PARÁMETRO BETA (para F2):
    - En esta función: parámetro 'beta=2.0' (línea ~128)
    - Se usa en fbeta_score() para evaluar cada threshold
    """
    best_score = -1
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.95, 0.01):
        y_pred = (y_probs_val >= threshold).astype(int)

        if optimize_for == 'F1':
            score = f1_score(y_val, y_pred, zero_division=0)
        else:
            # ← PARÁMETRO BETA: cambiar aquí para búsqueda de umbral con beta diferente
            score = fbeta_score(y_val, y_pred, beta=beta, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# ============================================================================
# BASELINES
# ============================================================================

def compute_baselines(X_train, y_train, X_test, y_test):
    """
    Baselines:
    1. SOFA parcial como predictor único (AUROC)
    2. APACHE II parcial como predictor único (AUROC)
    3. Regresión Logística con todas las features (excepto scores clínicos)
    """
    results = {}

    if 'sofa_partial' in X_test.columns:
        sofa_vals = X_test['sofa_partial'].fillna(X_test['sofa_partial'].median())
        try:
            results['SOFA parcial'] = roc_auc_score(y_test, sofa_vals)
        except Exception:
            results['SOFA parcial'] = np.nan

    if 'apache2_partial' in X_test.columns:
        apache_vals = X_test['apache2_partial'].fillna(X_test['apache2_partial'].median())
        try:
            results['APACHE II parcial'] = roc_auc_score(y_test, apache_vals)
        except Exception:
            results['APACHE II parcial'] = np.nan

    try:
        exclude_scores = {'sofa_partial', 'apache2_partial', 'severity_score'}
        feature_cols_lr = [c for c in X_train.columns if c not in exclude_scores]

        # Impute solo para LR (necesita datos completos)
        X_train_lr = X_train[feature_cols_lr].fillna(0)
        X_test_lr  = X_test[feature_cols_lr].fillna(0)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_lr)
        X_test_sc  = scaler.transform(X_test_lr)

        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_train_sc, y_train)

        probs_lr = lr.predict_proba(X_test_sc)[:, 1]
        results['Regresión Logística'] = roc_auc_score(y_test, probs_lr)
    except Exception as e:
        print(f"  ⚠ Regresión Logística falló: {e}")
        results['Regresión Logística'] = np.nan

    return results


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Matriz de confusión con valores absolutos y porcentajes por fila."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['No muere', 'Muere'],
                yticklabels=['No muere', 'Muere'])
    plt.ylabel('Verdadero')
    plt.xlabel('Predicción')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration(model, X_val, y_val, X_test, y_test, output_path):
    """Curva de calibración: modelo raw vs Platt scaling."""
    from sklearn.calibration import calibration_curve

    y_probs_test = model.predict_proba(X_test)[:, 1]

    calibrator = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
    calibrator.fit(X_val, y_val)
    y_probs_test_cal = calibrator.predict_proba(X_test)[:, 1]

    prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_probs_test, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_probs_test_cal, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
    plt.plot(prob_pred_raw, prob_true_raw, 'o-', label='Raw', linewidth=2)
    plt.plot(prob_pred_cal, prob_true_cal, 's-', label='Calibrado (Platt)', linewidth=2)
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Probabilidad verdadera')
    plt.title('Curva de Calibración')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return calibrator


def plot_shap_beeswarm(model, X_test, option_name, optimize_for, auroc, output_path):
    """SHAP beeswarm plot (top 20 features)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure(figsize=(12, 9))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Beeswarm — {option_name} {optimize_for} | AUROC: {auroc:.4f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_examples(model, X_test, y_test, y_pred_test, y_probs_test,
                       output_dir, option_name, optimize_for):
    """
    Genera SHAP waterfall plots para ejemplos de TP, FP, TN, FN.

    TP: Verdadero Positivo   (y_true=1, y_pred=1) — correctamente predicho como muerte
    FP: Falso Positivo       (y_true=0, y_pred=1) — incorrectamente predicho como muerte
    TN: Verdadero Negativo   (y_true=0, y_pred=0) — correctamente predicho como supervivencia
    FN: Falso Negativo       (y_true=1, y_pred=0) — incorrectamente predicho como supervivencia
    """
    # Resetear índices para poder acceder por posición
    if hasattr(y_test, 'reset_index'):
        y_test = y_test.reset_index(drop=True)
    if hasattr(y_pred_test, 'reset_index'):
        y_pred_test = y_pred_test.reset_index(drop=True)
    if hasattr(y_probs_test, 'reset_index'):
        y_probs_test = y_probs_test.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Definir máscaras para cada tipo
    tp_mask = (y_test == 1) & (y_pred_test == 1)
    fp_mask = (y_test == 0) & (y_pred_test == 1)
    tn_mask = (y_test == 0) & (y_pred_test == 0)
    fn_mask = (y_test == 1) & (y_pred_test == 0)

    examples = {
        'TP': {
            'mask': tp_mask,
            'label': 'Verdadero Positivo (muerte correctamente predicha)',
            'color': '#2E7D32'
        },
        'FP': {
            'mask': fp_mask,
            'label': 'Falso Positivo (supervivencia predicha como muerte)',
            'color': '#F57C00'
        },
        'TN': {
            'mask': tn_mask,
            'label': 'Verdadero Negativo (supervivencia correctamente predicha)',
            'color': '#1565C0'
        },
        'FN': {
            'mask': fn_mask,
            'label': 'Falso Negativo (muerte predicha como supervivencia)',
            'color': '#C62828'
        }
    }

    for ex_type, ex_info in examples.items():
        mask = ex_info['mask']
        n_examples = mask.sum()

        if n_examples == 0:
            print(f"  ⚠️  No hay ejemplos de {ex_type}")
            continue

        # Seleccionar un ejemplo (el del medio si hay varios)
        indices = np.where(mask)[0]
        idx = indices[len(indices) // 2]

        # Crear waterfall plot
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[idx], show=False)

        prob = y_probs_test[idx]
        pred_label = 'MUERTE' if y_pred_test[idx] == 1 else 'SUPERVIVENCIA'
        true_label = 'MUERTE' if y_test[idx] == 1 else 'SUPERVIVENCIA'

        title = (f"{ex_type}: {ex_info['label']}\n"
                f"Predicción: {pred_label} (prob={prob:.3f}) | "
                f"Real: {true_label}\n"
                f"(ejemplo {indices.tolist().index(idx)+1} de {n_examples})")

        plt.title(title, fontsize=12, fontweight='bold', color=ex_info['color'])
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'shap_example_{ex_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ {ex_type}: {output_path}")


# ============================================================================
# ANÁLISIS DE FALSOS NEGATIVOS
# ============================================================================

def analyze_false_negatives(X_test, y_test, y_pred, output_path):
    """Compara scores clínicos entre falsos negativos y verdaderos positivos."""
    fn_mask = (y_test == 1) & (y_pred == 0)
    tp_mask = (y_test == 1) & (y_pred == 1)

    fn_count = fn_mask.sum()
    tp_count = tp_mask.sum()
    total_deaths = y_test.sum()
    recall = tp_count / total_deaths if total_deaths > 0 else 0

    report = [
        "=" * 70 + "\n",
        "ANÁLISIS DE FALSOS NEGATIVOS\n",
        "=" * 70 + "\n\n",
        f"Total de muertes en test: {total_deaths}\n",
        f"Verdaderos Positivos (TP): {tp_count}\n",
        f"Falsos Negativos (FN):     {fn_count}\n",
        f"Recall: {recall:.1%} ({tp_count}/{total_deaths})\n\n",
        "-" * 70 + "\n",
        "Comparativa de variables entre FN y TP:\n",
        "-" * 70 + "\n\n",
    ]

    compare_cols = [
        'sofa_partial', 'apache2_partial', 'severity_score',
        'vital_spo2_val_min', 'vital_spo2_val_last',
        'lab_creatinine_val_max', 'lab_creatinine_val_last',
        'lab_lactate_val_max', 'lab_lactate_val_last',
        'lab_platelets_val_min',
        'lab_bilirubin_val_max',
        'lab_pao2_val_min',
        'anchor_age',
    ]

    for col in compare_cols:
        if col in X_test.columns:
            fn_vals = X_test.loc[fn_mask, col].dropna()
            tp_vals = X_test.loc[tp_mask, col].dropna()

            if len(fn_vals) > 0 and len(tp_vals) > 0:
                fn_mean    = fn_vals.mean()
                tp_mean    = tp_vals.mean()
                fn_missing = fn_mask.sum() - len(fn_vals)
                tp_missing = tp_mask.sum() - len(tp_vals)

                report.append(f"{col:<35}\n")
                report.append(f"  FN: {fn_mean:8.2f} (n={len(fn_vals)}, missing={fn_missing})\n")
                report.append(f"  TP: {tp_mean:8.2f} (n={len(tp_vals)}, missing={tp_missing})\n")
                report.append(f"  Δ:  {fn_mean - tp_mean:8.2f}\n\n")

    with open(output_path, 'w') as f:
        f.writelines(report)


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(y_true, y_prob, n_iterations=1000, ci=95, random_state=42):
    """
    Calcula intervalos de confianza bootstrap para métricas de evaluación.

    Remuestrea (y_true, y_prob) CON reemplazamiento, sin reentrenar el modelo.
    Calcula AUROC, AUPRC, F1, F2 para cada iteración bootstrap.

    Args:
        y_true: Array de labels verdaderos (0/1)
        y_prob: Array de probabilidades predichas
        n_iterations: Número de iteraciones bootstrap (default: 1000)
        ci: Nivel de confianza (default: 95 para IC 95%)
        random_state: Seed para reproducibilidad

    Returns:
        Dict con estructura:
        {
            'AUROC': {'point': 0.7947, 'lower': 0.7800, 'upper': 0.8100},
            'AUPRC': {'point': 0.7043, 'lower': 0.6800, 'upper': 0.7300},
            'F1':    {'point': 0.6269, 'lower': 0.5800, 'upper': 0.6700},
            'F2':    {'point': 0.6213, 'lower': 0.5700, 'upper': 0.6600}
        }
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    # Convertir a numpy arrays si son Series
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Almacenar scores de cada iteración
    auroc_scores = []
    auprc_scores = []
    f1_scores = []
    f2_scores = []

    for iteration in range(n_iterations):
        # Remuestrear índices con reemplazamiento
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[boot_idx]
        prob_boot = y_prob[boot_idx]

        # Skip si no hay ambas clases en el bootstrap sample
        if len(np.unique(y_boot)) < 2:
            continue

        # Calcular métricas
        auroc = roc_auc_score(y_boot, prob_boot)
        auprc = average_precision_score(y_boot, prob_boot)

        # Para F1/F2, usar threshold 0.5
        pred_boot = (prob_boot >= 0.5).astype(int)
        f1 = f1_score(y_boot, pred_boot, zero_division=0)
        f2 = fbeta_score(y_boot, pred_boot, beta=2, zero_division=0)

        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        f1_scores.append(f1)
        f2_scores.append(f2)

    # Calcular percentiles
    alpha = 100 - ci
    lower_percentile = alpha / 2
    upper_percentile = 100 - alpha / 2

    def percentile_ci(scores, point_estimate):
        lower = np.percentile(scores, lower_percentile)
        upper = np.percentile(scores, upper_percentile)
        return {
            'point': point_estimate,
            'lower': lower,
            'upper': upper
        }

    # Estimaciones puntuales (en dataset original de test)
    auroc_point = roc_auc_score(y_true, y_prob)
    auprc_point = average_precision_score(y_true, y_prob)
    pred_point = (y_prob >= 0.5).astype(int)
    f1_point = f1_score(y_true, pred_point, zero_division=0)
    f2_point = fbeta_score(y_true, pred_point, beta=2, zero_division=0)

    return {
        'AUROC': percentile_ci(auroc_scores, auroc_point),
        'AUPRC': percentile_ci(auprc_scores, auprc_point),
        'F1':    percentile_ci(f1_scores, f1_point),
        'F2':    percentile_ci(f2_scores, f2_point),
        'n_iterations': len(auroc_scores)  # Número de iteraciones válidas
    }


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def optimize_and_train(dataset_path, option_name, optimize_for='F1', beta=2.0, n_trials=100):
    """
    Pipeline completo: carga → preproceso → Optuna → entrenamiento → evaluación.

    PARÁMETRO BETA:
    - En esta función (línea ~400): parámetro 'beta=2.0' en firma
    - Se pasa a optimize_hyperparameters() y find_optimal_threshold()
    - Se usa para calcular F2 durante búsqueda y evaluación
    """
    print(f"\n{'='*70}")
    print(f"  XGBOOST V7.1 — {option_name} ({optimize_for})")
    print(f"  [SIN imputación de datos — solo procedimientos/PMHX/indicadores]")
    print(f"{'='*70}\n")

    output_dir = f"v7_1-models/{option_name}/{optimize_for}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carga del dataset
    print("[1/9] Cargando dataset...")
    df = pd.read_parquet(dataset_path)
    print(f"      Shape: {df.shape}")

    # 2. Preprocesamiento
    print("[2/9] Preprocesamiento...")
    X, y = preprocess_data(df)
    print(f"      Features: {X.shape[1]}")
    print(f"      Clase 0: {(y==0).sum()} | Clase 1: {(y==1).sum()} (prevalencia: {y.mean():.1%})")

    # 3. Split 70/15/15
    print("[3/9] Split 70/15/15...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"      Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # 4. Imputación MÍNIMA (solo procedimientos, PMHX, indicadores)
    print("[4/9] Imputación mínima (solo procedimientos/PMHX/indicadores)...")
    X_train, X_val, X_test = minimal_impute_data(X_train, X_val, X_test)
    print(f"      NaN en X_train: {X_train.isna().sum().sum()}")
    print(f"      (XGBoost manejará los NaN de vitales/labs/scores nativamente)")

    # 5. Optuna
    print(f"[5/9] Optimizando hiperparámetros con Optuna ({n_trials} trials, métrica: {optimize_for}, beta: {beta})...")
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"      Scale_pos_weight: {ratio:.4f}")

    best_params = optimize_hyperparameters(X_train, y_train, ratio, optimize_for, beta, n_trials)
    print(f"      Best params: {best_params}")

    best_params['scale_pos_weight']      = ratio
    best_params['random_state']          = 42
    best_params['eval_metric']           = 'aucpr'
    best_params['early_stopping_rounds'] = 50

    # 6. Entrenamiento del modelo final
    print("[6/9] Entrenamiento modelo final...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 7. Búsqueda de umbral óptimo
    print("[7/9] Búsqueda de umbral óptimo...")
    y_probs_val = model.predict_proba(X_val)[:, 1]
    threshold_opt, threshold_score = find_optimal_threshold(y_val, y_probs_val, optimize_for, beta)
    print(f"      Umbral óptimo: {threshold_opt:.3f} ({optimize_for}-score en val: {threshold_score:.4f})")

    # 8. Evaluación en test
    print("[8/9] Evaluación en test...")
    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_test  = (y_probs_test >= threshold_opt).astype(int)

    auroc    = roc_auc_score(y_test, y_probs_test)
    auprc    = average_precision_score(y_test, y_probs_test)
    f1_test  = f1_score(y_test, y_pred_test, zero_division=0)
    f2_test  = fbeta_score(y_test, y_pred_test, beta=2, zero_division=0)
    brier_raw = brier_score_loss(y_test, y_probs_test)

    print(f"      AUROC: {auroc:.4f}")
    print(f"      AUPRC: {auprc:.4f}")
    print(f"      F1:    {f1_test:.4f}")
    print(f"      F2:    {f2_test:.4f}")
    print(f"      Brier (raw): {brier_raw:.4f}")

    # Calcular intervalos de confianza bootstrap 95%
    print("\n      Calculando intervalos de confianza bootstrap (1000 iteraciones)...")
    ci_results = bootstrap_ci(y_test, y_probs_test, n_iterations=1000, ci=95, random_state=42)
    print(f"\n      📊 Intervalos de Confianza Bootstrap (95%):")
    print(f"         AUROC = {ci_results['AUROC']['point']:.4f} (IC: {ci_results['AUROC']['lower']:.4f} – {ci_results['AUROC']['upper']:.4f})")
    print(f"         AUPRC = {ci_results['AUPRC']['point']:.4f} (IC: {ci_results['AUPRC']['lower']:.4f} – {ci_results['AUPRC']['upper']:.4f})")
    print(f"         F1    = {ci_results['F1']['point']:.4f} (IC: {ci_results['F1']['lower']:.4f} – {ci_results['F1']['upper']:.4f})")
    print(f"         F2    = {ci_results['F2']['point']:.4f} (IC: {ci_results['F2']['lower']:.4f} – {ci_results['F2']['upper']:.4f})")
    print(f"         (n_iter válidas: {ci_results['n_iterations']}/1000)\n")

    # 9. Calibración, baselines y visualizaciones
    print("[9/9] Calibración, baselines y visualizaciones...")

    calibrator = plot_calibration(model, X_val, y_val, X_test, y_test,
                                  f"{output_dir}/calibration_curve.png")
    y_probs_test_cal = calibrator.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, y_probs_test_cal)

    baselines = compute_baselines(X_train, y_train, X_test, y_test)

    plot_confusion_matrix(y_test, y_pred_test, f"{output_dir}/confusion_matrix.png")
    plot_shap_beeswarm(model, X_test, option_name, optimize_for, auroc,
                       f"{output_dir}/shap_beeswarm.png")
    print("\nGenerando SHAP examples (TP, FP, TN, FN)...")
    plot_shap_examples(model, X_test, y_test, y_pred_test, y_probs_test,
                       output_dir, option_name, optimize_for)
    analyze_false_negatives(X_test, y_test, y_pred_test, f"{output_dir}/fn_analysis.txt")

    # Reporte final
    report_lines = []
    report_lines.append(f"{'='*70}\n")
    report_lines.append(f"REPORTE V7.1 — {option_name} ({optimize_for})\n")
    report_lines.append(f"{'='*70}\n\n")

    report_lines.append("--- CONFIGURACIÓN ---\n")
    report_lines.append(f"Dataset: {dataset_path}\n")
    report_lines.append(f"Imputación: MÍNIMA (solo procedimientos/PMHX/indicadores → 0)\n")
    report_lines.append(f"            Vitales/labs/scores: NaN (XGBoost maneja nativamente)\n")
    report_lines.append(f"Ventana de extracción: 72 horas\n")
    report_lines.append(f"Trials Optuna: {n_trials}\n")
    report_lines.append(f"Beta (para F2): {beta}\n\n")

    report_lines.append("--- MÉTRICAS PRINCIPALES ---\n")
    report_lines.append(f"AUROC:  {auroc:.4f}\n")
    report_lines.append(f"AUPRC:  {auprc:.4f}\n")
    report_lines.append(f"Umbral Óptimo ({optimize_for}): {threshold_opt:.3f}\n")
    report_lines.append(f"{optimize_for}-Score (Test): {(f1_test if optimize_for == 'F1' else f2_test):.4f}  (Val: {threshold_score:.4f})\n")
    report_lines.append(f"Brier Score (raw):      {brier_raw:.4f}\n")
    report_lines.append(f"Brier Score (calibrado): {brier_cal:.4f}\n\n")

    report_lines.append("--- INTERVALOS DE CONFIANZA BOOTSTRAP (95%) ---\n")
    report_lines.append(f"AUROC = {ci_results['AUROC']['point']:.4f} (IC: {ci_results['AUROC']['lower']:.4f} – {ci_results['AUROC']['upper']:.4f})\n")
    report_lines.append(f"AUPRC = {ci_results['AUPRC']['point']:.4f} (IC: {ci_results['AUPRC']['lower']:.4f} – {ci_results['AUPRC']['upper']:.4f})\n")
    report_lines.append(f"F1    = {ci_results['F1']['point']:.4f} (IC: {ci_results['F1']['lower']:.4f} – {ci_results['F1']['upper']:.4f})\n")
    report_lines.append(f"F2    = {ci_results['F2']['point']:.4f} (IC: {ci_results['F2']['lower']:.4f} – {ci_results['F2']['upper']:.4f})\n")
    report_lines.append(f"(Basado en {ci_results['n_iterations']} iteraciones bootstrap)\n\n")

    report_lines.append("--- CLASSIFICATION REPORT ---\n")
    report_lines.append(classification_report(y_test, y_pred_test,
                                              target_names=['No muere', 'Muere']) + "\n")

    report_lines.append("--- COMPARATIVA BASELINES (AUROC) ---\n")
    for name, auroc_bl in baselines.items():
        if not np.isnan(auroc_bl):
            report_lines.append(f"  {name:<30}: {auroc_bl:.4f}\n")
    report_lines.append(f"  {'XGBoost v7.1':<30}: {auroc:.4f}\n")

    report_lines.append("\n--- HIPERPARÁMETROS ÓPTIMOS ---\n")
    for k, v in best_params.items():
        report_lines.append(f"  {k}: {v}\n")

    with open(f"{output_dir}/metrics_report.txt", 'w') as f:
        f.writelines(report_lines)

    print(f"\n✅ Completado. Resultados en: {output_dir}/")
    print(f"   - metrics_report.txt")
    print(f"   - confusion_matrix.png")
    print(f"   - calibration_curve.png")
    print(f"   - shap_beeswarm.png (importancia global de features)")
    print(f"   - shap_example_TP.png (Verdadero Positivo)")
    print(f"   - shap_example_FP.png (Falso Positivo)")
    print(f"   - shap_example_TN.png (Verdadero Negativo)")
    print(f"   - shap_example_FN.png (Falso Negativo)")
    print(f"   - fn_analysis.txt")

    return {
        'auroc':     auroc,
        'auprc':     auprc,
        'f1':        f1_test,
        'f2':        f2_test,
        'brier_raw': brier_raw,
        'brier_cal': brier_cal,
    }


# ============================================================================
# INTERFAZ CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost v7.1 con Optuna — Sin imputación de datos (solo procedimientos/PMHX)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--option',
        choices=['B', 'C', 'all'],
        default='all',
        help="OptionB, OptionC, o all (default: all)"
    )
    parser.add_argument(
        '--metric',
        choices=['F1', 'F2', 'both'],
        default='both',
        help="Optimizar para F1, F2, o ambos (default: both)"
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help="Número de trials en Optuna (default: 100)"
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=2.0,
        help="Valor de beta para F2 (default: 2.0) — CAMBIAR AQUÍ para otros valores"
    )

    args = parser.parse_args()

    options = ['B', 'C'] if args.option == 'all' else [args.option]
    metrics = ['F1', 'F2'] if args.metric == 'both' else [args.metric]

    print("\n" + "="*70)
    print("XGBOOST V7.1 — Configuración:")
    print(f"  Options: {', '.join(options)}")
    print(f"  Metrics: {', '.join(metrics)}")
    print(f"  Trials:  {args.trials}")
    print(f"  Beta (para F2): {args.beta}  ← CAMBIAR AQUÍ (ej: --beta 1.0)")
    print("="*70)

    results_summary = {}

    for option in options:
        for metric in metrics:
            script_dir   = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)

            possible_paths = [
                os.path.join(project_root, "notebooks", "datasets", "v7",
                             f"dataset_v7_option{option}.parquet"),
                os.path.join(project_root, "datasets", "v7",
                             f"dataset_v7_option{option}.parquet"),
                f"../notebooks/datasets/v7/dataset_v7_option{option}.parquet",
                f"notebooks/datasets/v7/dataset_v7_option{option}.parquet",
            ]

            dataset_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break

            if dataset_path is None:
                print(f"\n❌ Dataset no encontrado para Option{option}")
                print(f"   Intenté en: {possible_paths[0]}")
                continue

            option_label = f"Option{option}"
            result = optimize_and_train(dataset_path, option_label, metric,
                                        beta=args.beta, n_trials=args.trials)
            results_summary[f"{option}-{metric}"] = result

    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS:")
    print("="*70)
    for key, result in results_summary.items():
        print(f"\n{key}:")
        for metric_name, value in result.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
