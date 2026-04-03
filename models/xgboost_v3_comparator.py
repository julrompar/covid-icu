import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             classification_report, f1_score, confusion_matrix, 
                             ConfusionMatrixDisplay)

os.makedirs("v3-models/OptionB", exist_ok=True)
os.makedirs("v3-models/OptionC", exist_ok=True)

def train_and_evaluate(dataset_path, option_name):
    print(f"\n{'='*50}\nEVALUANDO {option_name}\n{'='*50}")
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
    
    # Asegurar que todas las features de X sean int, float o bool (boolean cast to int for xgboost)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
            
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # 5. Model
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.01,
        scale_pos_weight=ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='aucpr',
        early_stopping_rounds=100
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 6. Umbral óptimo F1
    y_probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_test, y_probs >= t, zero_division=0) for t in thresholds]
    opt_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_opt = (y_probs >= opt_threshold).astype(int)
    
    # 7. Métricas
    auroc = roc_auc_score(y_test, y_probs)
    auprc = average_precision_score(y_test, y_probs)
    
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Umbral óptimo F1: {opt_threshold:.2f}")
    print(classification_report(y_test, y_pred_opt))
    
    # 8. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_opt)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Supervive", "Fallece"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {option_name}")
    plt.savefig(f"v3-models/{option_name}/confusion_matrix.png")
    plt.close()
    
    # 9. SHAP Summary Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary - {option_name}\nAUROC: {auroc:.4f}")
    plt.tight_layout()
    plt.savefig(f"v3-models/{option_name}/shap_summary.png")
    plt.close()

if __name__ == "__main__":
    train_and_evaluate("../datasets/v3/dataset_v3_optionB.parquet", "OptionB")
    train_and_evaluate("../datasets/v3/dataset_v3_optionC.parquet", "OptionC")
