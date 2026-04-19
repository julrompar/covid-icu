"""
XGBoost v6 — Predicción mortalidad UCI COVID-19 a 30 días

Mejoras vs v5:
- 50+ features incluyendo scores clínicos (SOFA + APACHE II)
- Imputación diferenciada por grupo fisiológico (fit solo en train)
- 100 trials Optuna con pruning
- Métrica de optimización alineada con F1/F2
- Calibración de probabilidades
- Baselines: SOFA solo, APACHE II solo, Regresión Logística
- SHAP beeswarm + análisis de falsos negativos
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE FEATURES
# ============================================================================

DROP_COLS = {"subject_id", "admission_id", "stay_id", "admittime", "dod_within_30_days", "target"}

# Labs de baja cobertura: solo usar indicador _measured, no el valor
LOW_COVERAGE_NUMERIC = {
    "max_pt", "max_crp", "max_ferritin", "max_dimer",
    "min_lymphocytes", "max_neutrophils", "max_troponin"
}

def get_feature_cols(df):
    """
    Extrae columnas de feature del dataset v6.

    Estructura de grupos:
    - A: Vitales (7)
    - B: Labs alta cobertura (12)
    - C: Labs cobertura media (7)
    - D: Labs baja cobertura (solo indicadores)
    - E: Procedimientos (11)
    - F: Demográficas (4)
    - G: Scores clínicos (3)
    """
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLS and c not in LOW_COVERAGE_NUMERIC
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


def impute_data(X_train, X_val, X_test):
    """
    Imputación diferenciada por grupo.

    Estrategia:
    - Procedimientos (is_*, has_*): 0 (ausencia = no realizado)
    - Indicadores _measured: 0 (no deberían tener NaN)
    - Resto (vitales, labs, scores): mediana del train (fit solo en train)
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Identificar columnas por tipo
    procedure_cols = [c for c in X_train.columns if c.startswith('is_') or c.startswith('has_')]
    measured_cols = [c for c in X_train.columns if c.endswith('_measured')]

    # impute_cols: todas las demás columnas
    impute_cols = [c for c in X_train.columns if c not in procedure_cols and c not in measured_cols]

    # Asegurar que todas las columnas existen
    impute_cols = [c for c in impute_cols if c in X_train.columns]
    procedure_cols = [c for c in procedure_cols if c in X_train.columns]
    measured_cols = [c for c in measured_cols if c in X_train.columns]

    # Imputar procedimientos a 0
    for col in procedure_cols:
        X_train[col] = X_train[col].fillna(0)
        X_val[col] = X_val[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    # Imputar indicadores a 0
    for col in measured_cols:
        X_train[col] = X_train[col].fillna(0)
        X_val[col] = X_val[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    # Imputar resto con mediana (fit solo en train)
    if impute_cols:
        # Calcular mediana por columna en train
        for col in impute_cols:
            col_median = X_train[col].median()

            # Rellenar con la mediana
            X_train[col] = X_train[col].fillna(col_median)
            X_val[col] = X_val[col].fillna(col_median)
            X_test[col] = X_test[col].fillna(col_median)

    return X_train, X_val, X_test


# ============================================================================
# OPTUNA CON PRUNING
# ============================================================================

def optimize_hyperparameters(X_train, y_train, ratio, optimize_for='F1', beta=2.0, n_trials=100):
    """
    Búsqueda de hiperparámetros con Optuna.

    Métrica: F1 si optimize_for=='F1', F2 si optimize_for=='F2'
    Pruning: MedianPruner para acelerar búsqueda
    """

    PROCEDURE_COLS = [c for c in X_train.columns if c.startswith('is_') or c.startswith('has_')]
    MEASURED_COLS = [c for c in X_train.columns if c.endswith('_measured')]
    IMPUTE_COLS = [c for c in X_train.columns if c not in PROCEDURE_COLS and c not in MEASURED_COLS]

    def objective(trial):
        # Sugerir hiperparámetros
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": ratio,
            "random_state": 42,
            "eval_metric": "aucpr",
            "early_stopping_rounds": 50,
        }

        # CV con imputación dentro del fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_cv_train = X_train.iloc[train_idx].copy()
            X_cv_val = X_train.iloc[val_idx].copy()
            y_cv_train = y_train.iloc[train_idx]
            y_cv_val = y_train.iloc[val_idx]

            # Imputar dentro del fold con mediana de la partición de train del fold
            for col in IMPUTE_COLS:
                col_median = X_cv_train[col].median()
                X_cv_train[col] = X_cv_train[col].fillna(col_median)
                X_cv_val[col] = X_cv_val[col].fillna(col_median)

            # Entrenar modelo
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_val, y_cv_val)],
                verbose=False
            )

            # Evaluar con métrica objetivo
            probs = model.predict_proba(X_cv_val)[:, 1]

            if optimize_for == 'F1':
                score = f1_score(y_cv_val, probs >= 0.5, zero_division=0)
            else:  # F2
                score = fbeta_score(y_cv_val, probs >= 0.5, beta=beta, zero_division=0)

            cv_scores.append(score)

            # Pruning: detener si el score es muy bajo
            trial.report(np.mean(cv_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    # Crear estudio con pruning
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

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
    """
    best_score = -1
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.95, 0.01):
        y_pred = (y_probs_val >= threshold).astype(int)

        if optimize_for == 'F1':
            score = f1_score(y_val, y_pred, zero_division=0)
        else:
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
    Calcula baselines:
    1. SOFA parcial como predictor único
    2. APACHE II parcial como predictor único
    3. Regresión Logística con todas las features
    """
    results = {}

    # SOFA como predictor único
    if 'sofa_partial' in X_test.columns:
        sofa_vals = X_test['sofa_partial'].fillna(X_test['sofa_partial'].median())
        try:
            auroc = roc_auc_score(y_test, sofa_vals)
            results['SOFA parcial'] = auroc
        except:
            results['SOFA parcial'] = np.nan

    # APACHE II como predictor único
    if 'apache2_partial' in X_test.columns:
        apache_vals = X_test['apache2_partial'].fillna(X_test['apache2_partial'].median())
        try:
            auroc = roc_auc_score(y_test, apache_vals)
            results['APACHE II parcial'] = auroc
        except:
            results['APACHE II parcial'] = np.nan

    # Regresión Logística
    try:
        feature_cols_lr = [c for c in X_train.columns if c not in ['sofa_partial', 'apache2_partial', 'severity_score']]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train[feature_cols_lr].fillna(0))
        X_test_sc = scaler.transform(X_test[feature_cols_lr].fillna(0))

        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_train_sc, y_train)

        probs_lr = lr.predict_proba(X_test_sc)[:, 1]
        auroc = roc_auc_score(y_test, probs_lr)
        results['Regresión Logística'] = auroc
    except Exception as e:
        print(f"  ⚠ Regresión Logística falló: {e}")
        results['Regresión Logística'] = np.nan

    return results


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Matriz de confusión con números absolutos + porcentajes.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calcular porcentajes por fila
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Crear etiquetas: valor absoluto + porcentaje
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
    """
    Curva de calibración: modelo original vs calibrado.
    """
    from sklearn.calibration import calibration_curve

    # Probabilidades raw
    y_probs_val = model.predict_proba(X_val)[:, 1]
    y_probs_test = model.predict_proba(X_test)[:, 1]

    # Calibrar con Platt scaling
    calibrator = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
    calibrator.fit(X_val, y_val)
    y_probs_test_cal = calibrator.predict_proba(X_test)[:, 1]

    # Calcular curvas de calibración
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_probs_test, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_probs_test_cal, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
    plt.plot(prob_pred_raw, prob_true_raw, 'o-', label='Raw', linewidth=2)
    plt.plot(prob_pred_cal, prob_true_cal, 's-', label='Calibrado', linewidth=2)
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
    """
    SHAP beeswarm plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure(figsize=(12, 9))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Beeswarm — {option_name} {optimize_for} | AUROC: {auroc:.4f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# ANÁLISIS DE FALSOS NEGATIVOS
# ============================================================================

def analyze_false_negatives(X_test, y_test, y_pred, output_path):
    """
    Análisis de falsos negativos: comparar scores clínicos entre FN y TP.
    """
    fn_mask = (y_test == 1) & (y_pred == 0)
    tp_mask = (y_test == 1) & (y_pred == 1)

    fn_count = fn_mask.sum()
    tp_count = tp_mask.sum()
    total_deaths = y_test.sum()
    recall = tp_count / total_deaths if total_deaths > 0 else 0

    report = [
        "=" * 70 + "\n",
        f"ANÁLISIS DE FALSOS NEGATIVOS\n",
        "=" * 70 + "\n\n",
        f"Total de muertes en test: {total_deaths}\n",
        f"Verdaderos Positivos (TP): {tp_count}\n",
        f"Falsos Negativos (FN):     {fn_count}\n",
        f"Recall: {recall:.1%} ({tp_count}/{total_deaths})\n\n",
        "-" * 70 + "\n",
        "Comparativa de variables entre FN y TP:\n",
        "-" * 70 + "\n\n",
    ]

    # Columnas a comparar
    compare_cols = [
        'sofa_partial', 'apache2_partial', 'severity_score',
        'min_oxygen_saturation', 'max_creatinine', 'anchor_age',
        'max_lactate', 'min_platelets', 'max_bilirubin'
    ]

    for col in compare_cols:
        if col in X_test.columns:
            fn_vals = X_test.loc[fn_mask, col].dropna()
            tp_vals = X_test.loc[tp_mask, col].dropna()

            if len(fn_vals) > 0 and len(tp_vals) > 0:
                fn_mean = fn_vals.mean()
                tp_mean = tp_vals.mean()
                fn_missing = fn_mask.sum() - len(fn_vals)
                tp_missing = tp_mask.sum() - len(tp_vals)

                report.append(f"{col:<30}\n")
                report.append(f"  FN: {fn_mean:8.2f} (n={len(fn_vals)}, missing={fn_missing})\n")
                report.append(f"  TP: {tp_mean:8.2f} (n={len(tp_vals)}, missing={tp_missing})\n")
                report.append(f"  Δ:  {fn_mean - tp_mean:8.2f}\n\n")

    with open(output_path, 'w') as f:
        f.writelines(report)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def optimize_and_train(dataset_path, option_name, optimize_for='F1', beta=2.0):
    """
    Pipeline completo: carga → preproceso → Optuna → entrenamiento → evaluación.
    """
    print(f"\n{'='*70}")
    print(f"  XGBOOST V6 — {option_name} ({optimize_for})")
    print(f"{'='*70}\n")

    output_dir = f"v6-models/{option_name}/{optimize_for}"
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

    # 4. Imputación
    print("[4/9] Imputación...")
    X_train, X_val, X_test = impute_data(X_train, X_val, X_test)
    print(f"      NaN train: {X_train.isna().sum().sum()}")

    # 5. Optuna
    print(f"[5/9] Optimizando hiperparámetros con Optuna (100 trials, métrica: {optimize_for})...")
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"      Scale_pos_weight: {ratio:.4f}")

    best_params = optimize_hyperparameters(X_train, y_train, ratio, optimize_for, beta, n_trials=100)
    print(f"      Best params: {best_params}")

    best_params['scale_pos_weight'] = ratio
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'aucpr'
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
    print(f"      Umbral óptimo: {threshold_opt:.3f} ({optimize_for}-score: {threshold_score:.4f})")

    # 8. Evaluación en test
    print("[8/9] Evaluación en test...")
    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_probs_test >= threshold_opt).astype(int)

    auroc = roc_auc_score(y_test, y_probs_test)
    auprc = average_precision_score(y_test, y_probs_test)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    f2_test = fbeta_score(y_test, y_pred_test, beta=2, zero_division=0)
    brier_raw = brier_score_loss(y_test, y_probs_test)

    print(f"      AUROC: {auroc:.4f}")
    print(f"      AUPRC: {auprc:.4f}")
    print(f"      F1: {f1_test:.4f}")
    print(f"      F2: {f2_test:.4f}")
    print(f"      Brier (raw): {brier_raw:.4f}")

    # 9. Calibración y análisis
    print("[9/9] Calibración, baselines y visualizaciones...")

    # Calibración
    calibrator = plot_calibration(model, X_val, y_val, X_test, y_test,
                                   f"{output_dir}/calibration_curve.png")
    y_probs_test_cal = calibrator.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, y_probs_test_cal)

    # Baselines
    baselines = compute_baselines(X_train, y_train, X_test, y_test)

    # Visualizaciones
    plot_confusion_matrix(y_test, y_pred_test, f"{output_dir}/confusion_matrix.png")
    plot_shap_beeswarm(model, X_test, option_name, optimize_for, auroc,
                       f"{output_dir}/shap_beeswarm.png")
    analyze_false_negatives(X_test, y_test, y_pred_test, f"{output_dir}/fn_analysis.txt")

    # Reporte final
    report_lines = []
    report_lines.append(f"{'='*70}\n")
    report_lines.append(f"REPORTE V6 — {option_name} ({optimize_for})\n")
    report_lines.append(f"{'='*70}\n\n")

    report_lines.append("--- MÉTRICAS PRINCIPALES ---\n")
    report_lines.append(f"AUROC:  {auroc:.4f}\n")
    report_lines.append(f"AUPRC:  {auprc:.4f}\n")
    report_lines.append(f"Umbral Óptimo ({optimize_for}): {threshold_opt:.3f}\n")
    report_lines.append(f"{optimize_for}-Score (Test): {(f1_test if optimize_for == 'F1' else f2_test):.4f} (Val: {threshold_score:.4f})\n")
    report_lines.append(f"Brier Score (raw): {brier_raw:.4f}\n")
    report_lines.append(f"Brier Score (calibrado): {brier_cal:.4f}\n\n")

    report_lines.append("--- CLASSIFICATION REPORT ---\n")
    report_lines.append(classification_report(y_test, y_pred_test,
                                             target_names=['No muere', 'Muere']) + "\n")

    report_lines.append("--- COMPARATIVA BASELINES ---\n")
    for name, auroc_bl in baselines.items():
        if not np.isnan(auroc_bl):
            report_lines.append(f"{name:<30}: AUROC = {auroc_bl:.4f}\n")
    report_lines.append(f"{'XGBoost v6':<30}: AUROC = {auroc:.4f}\n")

    with open(f"{output_dir}/metrics_report.txt", 'w') as f:
        f.writelines(report_lines)

    print(f"\n✅ Completado. Resultados en: {output_dir}/")
    print(f"   - metrics_report.txt")
    print(f"   - confusion_matrix.png")
    print(f"   - calibration_curve.png")
    print(f"   - shap_beeswarm.png")
    print(f"   - fn_analysis.txt")

    return {
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1_test if optimize_for == 'F1' else None,
        'f2': f2_test if optimize_for == 'F2' else None,
        'brier_raw': brier_raw,
        'brier_cal': brier_cal,
    }


# ============================================================================
# INTERFAZ CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost v6 con Optuna — Modular (OptionB/C × F1/F2)",
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

    args = parser.parse_args()

    # Determinar qué combinaciones ejecutar
    options = ['B', 'C'] if args.option == 'all' else [args.option]
    metrics = ['F1', 'F2'] if args.metric == 'both' else [args.metric]

    print("\n" + "="*70)
    print(f"XGBOOST V6 — Configuración:")
    print(f"  Options: {', '.join(options)}")
    print(f"  Metrics: {', '.join(metrics)}")
    print(f"  Trials: {args.trials}")
    print("="*70)

    results_summary = {}

    for option in options:
        for metric in metrics:
            # Buscar dataset en múltiples ubicaciones posibles
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)

            possible_paths = [
                os.path.join(project_root, "notebooks/datasets/v6", f"dataset_v6_option{option}.parquet"),
                f"../notebooks/datasets/v6/dataset_v6_option{option}.parquet",
                f"notebooks/datasets/v6/dataset_v6_option{option}.parquet",
            ]

            dataset_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break

            if dataset_path is None:
                print(f"\n❌ Dataset no encontrado para option{option}")
                print(f"   Intenté en: {possible_paths[0]}")
                continue

            result = optimize_and_train(dataset_path, f"OptionC" if option == 'C' else f"OptionB", metric)
            results_summary[f"{option}-{metric}"] = result

    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS:")
    print("="*70)
    for key, result in results_summary.items():
        print(f"\n{key}:")
        for metric_name, value in result.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
