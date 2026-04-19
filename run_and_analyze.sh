#!/bin/bash

# Script para ejecutar modelo y detectar overfitting automáticamente
# Uso:
#   bash run_and_analyze.sh v6 B F1
#   bash run_and_analyze.sh v6_1 B F1
#   bash run_and_analyze.sh v7 B F1
#   bash run_and_analyze.sh v7_1 C F2

set -e

if [ $# -lt 3 ]; then
    echo "Uso: bash run_and_analyze.sh <version> <option> <metric>"
    echo ""
    echo "Ejemplos:"
    echo "  bash run_and_analyze.sh v6 B F1"
    echo "  bash run_and_analyze.sh v6_1 B F1"
    echo "  bash run_and_analyze.sh v7 B F1"
    echo "  bash run_and_analyze.sh v7 all both"
    echo "  bash run_and_analyze.sh v7_1 C F2"
    exit 1
fi

VERSION=$1
OPTION=$2
METRIC=$3

echo "=================================================="
echo "Ejecutando XGBoost $VERSION"
echo "Option: $OPTION | Metric: $METRIC"
echo "=================================================="

if [ "$VERSION" = "v6" ]; then
    python3 models/xgboost_v6_opt_comparator.py --option $OPTION --metric $METRIC --trials 100
    MODEL_DIR="v6-models"
elif [ "$VERSION" = "v6_1" ]; then
    python3 models/xgboost_v6_1_opt_comparator.py --option $OPTION --metric $METRIC --trials 100
    MODEL_DIR="v6_1-models"
elif [ "$VERSION" = "v7" ]; then
    python3 models/xgboost_v7_opt_comparator.py --option $OPTION --metric $METRIC --trials 100
    MODEL_DIR="v7-models"
elif [ "$VERSION" = "v7_1" ]; then
    python3 models/xgboost_v7_1_opt_comparator.py --option $OPTION --metric $METRIC --trials 100
    MODEL_DIR="v7_1-models"
else
    echo "❌ Versión desconocida: $VERSION (opciones: v6, v6_1, v7, v7_1)"
    exit 1
fi

echo ""
echo "=================================================="
echo "Analizando resultados en busca de overfitting..."
echo "=================================================="
echo ""

# Encontrar directorios de salida
if [ "$OPTION" = "all" ]; then
    for opt in B C; do
        for met in $METRIC; do
            if [ -d "${MODEL_DIR}/Option${opt}/${met}/" ]; then
                echo ""
                echo "📊 Analizando Option${opt} ${met}..."
                python3 models/detect_overfitting.py "${MODEL_DIR}/Option${opt}/${met}/"
            fi
        done
    done
else
    if [ -d "${MODEL_DIR}/Option${OPTION}/${METRIC}/" ]; then
        echo ""
        echo "📊 Analizando Option${OPTION} ${METRIC}..."
        python3 models/detect_overfitting.py "${MODEL_DIR}/Option${OPTION}/${METRIC}/"
    fi
fi

echo ""
echo "✅ Análisis completado"
