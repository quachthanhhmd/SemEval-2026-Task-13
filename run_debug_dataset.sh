#!/bin/bash
# run_debug_dataset.sh
# Helper script to run dataset diagnostics

# Configuration
TRAIN_DATA="data/train.parquet"
VAL_DATA="data/validation.parquet"
MODEL="microsoft/graphcodebert-base"
MAX_LEN=512
K_LANGS=6
M_PER_LANG=16

echo "==========================================================="
# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "⚠ WARNING: Train data not found at $TRAIN_DATA"
    echo "Please update TRAIN_DATA in this script."
fi

echo "🚀 Starting Dataset Diagnostics..."
echo "-----------------------------------------------------------"

python debug_dataset.py \
    --train_csv "$TRAIN_DATA" \
    --val_csv "$VAL_DATA" \
    --model_name "$MODEL" \
    --max_len $MAX_LEN \
    --k_langs $K_LANGS \
    --m_per_lang $M_PER_LANG

echo "-----------------------------------------------------------"
echo "✅ Diagnostics complete."
echo "==========================================================="
