import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import optuna
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             classification_report, f1_score, fbeta_score,
                             confusion_matrix)

def optimize_and_train(dataset_path, option_name, optimize_for='F1', beta=2.0):
    print(f"\n{'='*50}\nOPTIMIZANDO {option_name} - {optimize_for}\n{'='*50}")
    
    output_dir = f"v3-models/{option_name}/{optimize_for}"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_parquet(dataset_path)
    
    # 1. Variable Objetivo
    df['target'] = df['dod_within_30_days'].notnull().astype(int)
    
    # 2. Drop columns
    drop_cols = ["target", "dod_within_30_days", "subject_id", "admission_id", "stay_id", "admittime"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df['target']
    
    # 3. Preprocesamiento de categóricas
    cat_cols = [c for c in ['gender', 'marital_status', 'race'] if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
            
    # 4. Triple split: 70% train, 15% val (threshold/early stopping), 15% test (evaluación final)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # 5. Optuna
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": ratio,
            "random_state":     42,
            "eval_metric":      "aucpr",
            "early_stopping_rounds": 50,
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train = X_train.iloc[train_idx]
            X_cv_val   = X_train.iloc[val_idx]
            y_cv_train = y_train.iloc[train_idx]
            y_cv_val   = y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_val, y_cv_val)],
                verbose=False
            )
            preds = model.predict_proba(X_cv_val)[:, 1]
            cv_scores.append(average_precision_score(y_cv_val, preds))
            
        return np.mean(cv_scores)

    print("--- Iniciando búsqueda de hiperparámetros con Optuna ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    best_params = study.best_params
    best_params["scale_pos_weight"] = ratio
    best_params["random_state"] = 42
    best_params["eval_metric"] = "aucpr"
    best_params["early_stopping_rounds"] = 50
    
    # 6. Train final model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 7. Threshold optimization - buscado sobre X_val
    y_probs_val = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    
    if optimize_for == 'F1':
        scores_val = [f1_score(y_val, y_probs_val >= t, zero_division=0) for t in thresholds]
    else: # F2
        scores_val = [fbeta_score(y_val, y_probs_val >= t, beta=beta, zero_division=0) for t in thresholds]
        
    opt_threshold = thresholds[np.argmax(scores_val)]
    best_score_val = max(scores_val)
    
    # 8. Metrics - evaluado sobre X_test
    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_final = (y_probs_test >= opt_threshold).astype(int)
    
    auroc = roc_auc_score(y_test, y_probs_test)
    auprc = average_precision_score(y_test, y_probs_test)
    
    # Evaluate score on test set
    if optimize_for == 'F1':
        final_test_score = f1_score(y_test, y_pred_final, zero_division=0)
    else:
        final_test_score = fbeta_score(y_test, y_pred_final, beta=beta, zero_division=0)
    
    # Generate the formatted text report
    report_text = f"=== REPORTE V3 {option_name} ({optimize_for}) ===\n"
    report_text += f"AUROC: {auroc:.4f}\n"
    report_text += f"AUPRC: {auprc:.4f}\n"
    report_text += f"Umbral Optimo ({optimize_for}): {opt_threshold:.2f}\n"
    report_text += f"{optimize_for}-Score (Test): {final_test_score:.4f} (Val: {best_score_val:.4f})\n\n"
    report_text += classification_report(y_test, y_pred_final)
    
    print(report_text)
    
    with open(f"{output_dir}/metrics_report.txt", "w") as f:
        f.write(report_text)
        
    # 9. Confusion Matrix (Single, clean matrix like v3 but styled like v2)
    plt.figure(figsize=(6, 5))
    cm_opt = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Pred Vivo", "Pred Muerte"], 
                yticklabels=["Real Vivo", "Real Muerte"])
    plt.title(f"Matriz de Confusión — Umbral {optimize_for}-óptimo ({opt_threshold:.2f})")
    plt.ylabel("Valor Real")
    plt.xlabel("Valor Predicho")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 10. SHAP (on test set)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"Importancia Clínica de las Variables (SHAP Values) - {option_name} {optimize_for}\nAUROC: {auroc:.4f}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    datasets = {
        "OptionB": "../datasets/v3/dataset_v3_optionB.parquet",
        "OptionC": "../datasets/v3/dataset_v3_optionC.parquet",
    }
    metrics = [("F1", 1.0), ("F2", 2.0)]

    for option_name, dataset_path in datasets.items():
        for metric, beta in metrics:
            optimize_and_train(dataset_path, option_name, metric, beta)
