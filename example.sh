#!/bin/bash

# ==============================================================================
# Sample Commands for Debugging and Experimentation
# ==============================================================================
# IMPORTANT: Update the data paths (--train_csv, --val_csv) to match your setup.

TRAIN_DATA="data/train.parquet"
VAL_DATA="data/val.parquet"
REGISTRY="checkpoints/registry.json"
MODEL="microsoft/graphcodebert-base"

# 1. ULTRÁ-FAST DEBUG (5,000 samples)
# Use this to verify that the pipeline runs without crashes.
echo "Running DEBUG mode (5k samples)..."
python meta_train.py \
    --train_csv $TRAIN_DATA \
    --val_csv $VAL_DATA \
    --model_name $MODEL \
    --registry_path $REGISTRY \
    --experiment_mode debug \
    --adv_warmup_epochs 0 \
    --batch_size 16 \
    --max_len 256 \
    --log_every 10 \
    --checkpoint_dir checkpoints/debug_run

# 2. FAST DEVELOPMENT (50,000 samples)
# Use this for quick ablation studies or parameter tuning.
echo "Running FAST_DEV mode (50k samples)..."
python meta_train.py \
    --train_csv $TRAIN_DATA \
    --val_csv $VAL_DATA \
    --model_name $MODEL \
    --registry_path $REGISTRY \
    --experiment_mode fast_dev \
    --adv_warmup_epochs 1 \
    --batch_size 32 \
    --log_every 100 \
    --checkpoint_dir checkpoints/fast_dev_run

# 3. LEAVE-ONE-LANGUAGE-OUT (OOD Evaluation)
# Train on all languages EXCEPT Python; validate ONLY on Python.
echo "Running Leave-Language-Out (Python)..."
python meta_train.py \
    --train_csv $TRAIN_DATA \
    --val_csv $VAL_DATA \
    --model_name $MODEL \
    --registry_path $REGISTRY \
    --experiment_mode strong_dev \
    --val_strategy leave_language_out \
    --holdout_value python \
    --adv_warmup_epochs 2 \
    --checkpoint_dir checkpoints/ood_python

# 4. LEAVE-ONE-GENERATOR-OUT (OOD Evaluation)
# Train on all generators EXCEPT GPT-4; validate ONLY on GPT-4.
echo "Running Leave-Generator-Out (GPT-4)..."
python meta_train.py \
    --train_csv $TRAIN_DATA \
    --val_csv $VAL_DATA \
    --model_name $MODEL \
    --registry_path $REGISTRY \
    --experiment_mode strong_dev \
    --val_strategy leave_generator_out \
    --holdout_value gpt-4 \
    --adv_warmup_epochs 2 \
    --checkpoint_dir checkpoints/ood_gpt4

# 5. STEP-BASED TRAINING (Flexible Duration)
# Train for exactly 1000 steps with 200 steps of warmup.
echo "Running Step-based training (1000 steps)..."
python meta_train.py \
    --train_csv $TRAIN_DATA \
    --val_csv $VAL_DATA \
    --model_name $MODEL \
    --registry_path $REGISTRY \
    --epochs 0 \
    --num_steps 1000 \
    --adv_warmup_epochs 0 \
    --adv_warmup_steps 200 \
    --checkpoint_dir checkpoints/step_based_run
