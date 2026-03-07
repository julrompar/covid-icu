import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    brier_score_loss, f1_score, fbeta_score
)

# ============================================================================
# CARGAR DATOS Y PREPARAR TEST SET
# ============================================================================
df = pd.read_csv('../datasets/v1/dataset_v1_clean.csv')
df['target'] = df['dod_within_30_days'].notnull().astype(int)

# Split básico (mismo que usas en tu código)
_, _, _, y_test = train_test_split(
    df.drop(['target', 'dod_within_30_days', 'subject_id', 'admission_id', 'stay_id'], axis=1),
    df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)

n_test = len(y_test)
prevalence = y_test.mean()

print("="*70)
print("MODELO ALEATORIO - COMPARACIÓN")
print("="*70)
print(f"Tamaño test set: {n_test}")
print(f"Prevalencia real de muertes: {prevalence:.4f} ({prevalence*100:.2f}%)")
print("\n")

# ============================================================================
# MODELO 1: Probabilidades completamente al azar (Uniform [0,1])
# ============================================================================
print("-"*70)
print("MODELO 1: Probabilidades UNIFORMES al azar [0, 1]")
print("-"*70)

np.random.seed(42)
y_probs_uniform = np.random.uniform(0, 1, n_test)

# Métricas de probabilidad
auroc_uniform = roc_auc_score(y_test, y_probs_uniform)
auprc_uniform = average_precision_score(y_test, y_probs_uniform)
brier_uniform = brier_score_loss(y_test, y_probs_uniform)

print(f"AUROC: {auroc_uniform:.4f}")
print(f"AUPRC: {auprc_uniform:.4f}")
print(f"Brier Score: {brier_uniform:.4f}")

# Con umbral 0.5
y_pred_uniform_05 = (y_probs_uniform >= 0.5).astype(int)
print(f"\nCon umbral 0.5:")
print(classification_report(y_test, y_pred_uniform_05, zero_division=0))

# Optimizar F1
f1_scores_uniform = []
thresholds = np.linspace(0.01, 0.99, 100)
for thr in thresholds:
    y_pred_temp = (y_probs_uniform >= thr).astype(int)
    f1_scores_uniform.append(f1_score(y_test, y_pred_temp, zero_division=0))

best_f1_uniform = max(f1_scores_uniform)
best_thr_f1_uniform = thresholds[np.argmax(f1_scores_uniform)]
print(f"\nF1-Score máximo: {best_f1_uniform:.4f} (umbral: {best_thr_f1_uniform:.4f})")

# Optimizar F2
f2_scores_uniform = []
for thr in thresholds:
    y_pred_temp = (y_probs_uniform >= thr).astype(int)
    f2_scores_uniform.append(fbeta_score(y_test, y_pred_temp, beta=2, zero_division=0))

best_f2_uniform = max(f2_scores_uniform)
best_thr_f2_uniform = thresholds[np.argmax(f2_scores_uniform)]
print(f"F2-Score máximo: {best_f2_uniform:.4f} (umbral: {best_thr_f2_uniform:.4f})")

print("\n")

# ============================================================================
# MODELO 2: Predicción estratificada (probabilidad fija = prevalencia)
# ============================================================================
print("-"*70)
print("MODELO 2: Probabilidad fija igual a la prevalencia")
print("-"*70)

y_probs_stratified = np.full(n_test, prevalence)

auroc_stratified = roc_auc_score(y_test, y_probs_stratified)
auprc_stratified = average_precision_score(y_test, y_probs_stratified)
brier_stratified = brier_score_loss(y_test, y_probs_stratified)

print(f"Probabilidad asignada a todos: {prevalence:.4f}")
print(f"AUROC: {auroc_stratified:.4f}")
print(f"AUPRC: {auprc_stratified:.4f}")
print(f"Brier Score: {brier_stratified:.4f}")

print("\n")

# ============================================================================
# MODELO 3: Predicción al azar con probabilidad = prevalencia
# ============================================================================
print("-"*70)
print("MODELO 3: Clasificación binaria al azar (prob = prevalencia)")
print("-"*70)

np.random.seed(42)
y_pred_random_binary = np.random.binomial(1, prevalence, n_test)

print(f"Predicciones positivas: {y_pred_random_binary.sum()} de {n_test} ({y_pred_random_binary.mean()*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_random_binary, zero_division=0))

print("\n")

# ============================================================================
# COMPARACIÓN CON TU MODELO REAL
# ============================================================================
print("="*70)
print("RESUMEN COMPARATIVO")
print("="*70)
print(f"{'Modelo':<40} {'AUROC':<10} {'AUPRC':<10} {'Brier':<10}")
print("-"*70)
print(f"{'Tu modelo (XGBoost + isotonic)':<40} {0.810:<10.4f} {0.384:<10.4f} {0.084:<10.4f}")
print(f"{'Azar uniforme [0,1]':<40} {auroc_uniform:<10.4f} {auprc_uniform:<10.4f} {brier_uniform:<10.4f}")
print(f"{'Probabilidad fija (prevalencia)':<40} {auroc_stratified:<10.4f} {auprc_stratified:<10.4f} {brier_stratified:<10.4f}")
print("="*70)

print("\n")
print("="*70)
print("COMPARACIÓN DE F-SCORES ÓPTIMOS")
print("="*70)
print(f"{'Modelo':<40} {'F1 máx':<10} {'F2 máx':<10}")
print("-"*70)
print(f"{'Tu modelo F1-optimizado':<40} {0.43:<10.2f} {'N/A':<10}")
print(f"{'Tu modelo F2-optimizado':<40} {'N/A':<10} {0.55:<10.2f}")
print(f"{'Azar uniforme [0,1]':<40} {best_f1_uniform:<10.2f} {best_f2_uniform:<10.2f}")
print("="*70)

print("\n")
print("INTERPRETACIÓN:")
print("-"*70)
print(f"• AUROC de azar uniforme ≈ 0.5 (sin discriminación)")
print(f"  Tu modelo: 0.810 → {(0.810-0.5)/0.5*100:.0f}% mejor que azar")
print(f"\n• AUPRC de baseline ≈ {prevalence:.3f} (prevalencia)")
print(f"  Azar uniforme: {auprc_uniform:.3f}")
print(f"  Tu modelo: 0.384 → {0.384/prevalence:.1f}x mejor que baseline")
print(f"\n• Brier Score (más bajo = mejor):")
print(f"  Azar: {brier_uniform:.3f}")
print(f"  Tu modelo: 0.084 → {(brier_uniform-0.084)/brier_uniform*100:.0f}% mejor")
print(f"\n• F1/F2 óptimos:")
print(f"  Tu modelo F1: 0.43 vs Azar: {best_f1_uniform:.2f}")
print(f"  Tu modelo F2: 0.55 vs Azar: {best_f2_uniform:.2f}")
print("="*70)
