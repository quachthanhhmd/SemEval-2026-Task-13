#!/bin/bash
# run_meta_inference.sh
# Optimized Runner for Meta-Learning Inference with TTA

# Configuration
TEST_FILE="data/Task_A/test.parquet"
CHECKPOINT_DIR="checkpoints/meta_training/best_model"
MODEL_NAME="microsoft/graphcodebert-base"

# Must match your training configuration
NUM_GEN=10
NUM_LANG=10

# Inference Params
BATCH_SIZE=32
TTA_VIEWS=5
NUM_WORKERS=4

echo "==========================================================="
echo "🚀 Starting Optimized Inference..."
echo "File: $TEST_FILE"
echo "TTA Views: $TTA_VIEWS"
echo "==========================================================="

python meta_inference.py \
    --test_file "$TEST_FILE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_generators $NUM_GEN \
    --num_languages $NUM_LANG \
    --batch_size $BATCH_SIZE \
    --tta_views $TTA_VIEWS \
    --num_workers $NUM_WORKERS

echo "==========================================================="
echo "✅ Inference complete."
