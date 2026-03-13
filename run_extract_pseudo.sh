#!/bin/bash
# run_extract_pseudo.sh
# Extracts pseudo-labels from unlabeled test data for domain adaptation.
# Usage: bash run_extract_pseudo.sh

python extract_pseudo_samples.py \
    --test_file data/test.parquet \
    --checkpoint_dir checkpoints/meta_training/best_model \
    --output_dir pseudo_data/ \
    --model_name microsoft/graphcodebert-base \
    --tta_views 5 \
    --batch_size 16 \
    --num_workers 4 \
    --max_len 512
    # Optionally add --no_code to exclude the source code column in outputs
