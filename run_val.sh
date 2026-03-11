#!/bin/bash
# run_val.sh
# Script to run Test-Time Augmentation (TTA) Validation for SemEval Task 13

# Exit on error
set -e

# ==========================================
# 1. Configuration
# ==========================================
# Paths (Update these to point to your actual .parquet/.csv files)
VAL_FILE="data/validation.parquet"
CHECKPOINT_DIR="checkpoints/meta_training/best_model" # Point to the saved model folder

# Inference settings
BATCH_SIZE=32
MAX_LEN=512
TTA_VIEWS=5  # Number of Test-Time Augmentations (spans) per sequence

# Model Architecture Configuration (Should match training)
NUM_GENERATORS=10
NUM_LANGUAGES=10
MODEL_NAME="microsoft/graphcodebert-base"

# ==========================================
# 2. Execution
# ==========================================
echo "==========================================================="
echo " Starting TTA Validation Pipeline"
echo "   Validation File:  $VAL_FILE"
echo "   Checkpoint:       $CHECKPOINT_DIR"
echo "   TTA Views:        $TTA_VIEWS"
echo "==========================================================="

python meta_inference.py \
    --test_file "$VAL_FILE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN \
    --tta_views $TTA_VIEWS \
    --num_generators $NUM_GENERATORS \
    --num_languages $NUM_LANGUAGES \
    --model_name "$MODEL_NAME"

echo "==========================================================="
echo " Validation Completed! Metrics and predictions are generated."
echo "==========================================================="
