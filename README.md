# Sem Eval 2026 Task 13

## For training this project


1. Install the dependancy
```py
  pip install -r requirements.txt
```

2. Training parse

```bash
cd linevul
```
```
python3 linevul_main.py \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=datasets/train.parquet \
  --eval_data_file=datasets/val.parquet \
  --test_data_file=datasets/test.parquet\
  --epochs 5 \
  --block_size 256 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456  2>&1 | tee train.log
```


3. Inference
```
python linevul_main.py \
  --model_name=model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=/kaggle/input/datasets/quachthanhhmd/semval-processed-data/train_processed.parquet \
  --eval_data_file=/kaggle/input/datasets/quachthanhhmd/semval-processed-data/val_processed.parquet \
  --test_data_file=/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A/test.parquet\
  --block_size 512 \
  --write_raw_preds\
  --eval_batch_size 512
```