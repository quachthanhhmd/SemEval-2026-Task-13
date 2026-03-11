#!/bin/bash
# run_train.sh
# Script to easily run the Meta-Training Pipeline for SemEval 2026 Task 13

# Exit on error
set -e

# ==========================================
# 1. Configuration
# ==========================================
# Paths (Update these to point to your actual .parquet/.csv files)
TRAIN_FILE="data/train.parquet"
VAL_FILE="data/validation.parquet"
CHECKPOINT_DIR="checkpoints/meta_training"

# To resume training from a checkpoint, set this to the checkpoint dir
# RESUME_FROM="checkpoints/meta_training/best_model"
RESUME_FROM=""

# Sampler settings (batch_size = k_langs * m_per_lang)
K_LANGS=6
M_PER_LANG=16
BATCH_SIZE=$((K_LANGS * M_PER_LANG)) # 96

# Hyperparameters (Optimized for stability)
EPOCHS=3
MAX_LEN=512
LR=2e-5
LR_INNER=5e-5

# Loss Weights
ALPHA=0.2        # SupCon L_train weight
ALPHA_META=0.05  # SupCon L_meta weight
BETA=0.05        # Generator Adv weight
GAMMA=0.05       # Language Adv weight
GRL_SCALE=5.0    # Slower adversarial ramp

# ==========================================
# 2. Execution
# ==========================================
echo "==========================================================="
echo " Starting Meta-Training Pipeline"
echo "   Train: $TRAIN_FILE"
echo "   Val:   $VAL_FILE"
echo "   Batch: $BATCH_SIZE (k=$K_LANGS, m=$M_PER_LANG)"
echo "==========================================================="

python meta_train.py \
    --train_csv "$TRAIN_FILE" \
    --val_csv "$VAL_FILE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --batch_size $BATCH_SIZE \
    --k_langs $K_LANGS \
    --m_per_lang $M_PER_LANG \
    --max_len $MAX_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --lr_inner $LR_INNER \
    --alpha $ALPHA \
    --alpha_meta $ALPHA_META \
    --beta $BETA \
    --gamma $GAMMA \
    --grl_scale $GRL_SCALE \
    --fp16

if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume_from $RESUME_FROM"
    echo "   Resuming from: $RESUME_FROM"
    python meta_train.py \
        --train_csv "$TRAIN_FILE" \
        --val_csv "$VAL_FILE" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --batch_size $BATCH_SIZE \
        --k_langs $K_LANGS \
        --m_per_lang $M_PER_LANG \
        --max_len $MAX_LEN \
        --epochs $EPOCHS \
        --lr $LR \
        --lr_inner $LR_INNER \
        --alpha $ALPHA \
        --alpha_meta $ALPHA_META \
        --beta $BETA \
        --gamma $GAMMA \
        --grl_scale $GRL_SCALE \
        --resume_from "$RESUME_FROM" \
        --fp16
fi

echo "==========================================================="
echo " Training Completed! Weights saved to $CHECKPOINT_DIR"
echo "==========================================================="
