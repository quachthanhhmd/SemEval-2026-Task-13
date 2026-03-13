#!/bin/bash
# run_threshold_pipeline.sh
# Automated threshold optimization pipeline.
#
# Usage:
#   bash run_threshold_pipeline.sh DATA_PATH MODEL_PATH
#
# Example:
#   bash run_threshold_pipeline.sh data/val.parquet checkpoints/meta_training/best_model

DATA_PATH="${1:-data/val.parquet}"
MODEL_PATH="${2:-checkpoints/meta_training/best_model}"

echo "=============================================="
echo "  Threshold Optimization Pipeline"
echo "  Data   : $DATA_PATH"
echo "  Model  : $MODEL_PATH"
echo "=============================================="

python run_inference_and_find_threshold.py \
    --data_file      "$DATA_PATH" \
    --checkpoint_dir "$MODEL_PATH" \
    --model_name     microsoft/graphcodebert-base \
    --num_generators 35 \
    --num_languages  3 \
    --tta_views      5 \
    --batch_size     16 \
    --num_workers    4 \
    --max_len        512 \
    --plot_dir       threshold_analysis
