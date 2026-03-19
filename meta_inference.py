"""
meta_inference.py
-----------------
Optimized Inference script for the OOD Meta-Training Pipeline.
Supports:
1. Lazy Parquet/CSV loading (Memory efficient)
2. Precomputed entropy (Speed)
3. Token-level TTA (Avoids BPE boundary artifacts + Fast)
4. Batch flattening (High GPU utilization)
5. Deterministic TTA spans
"""

import os
import sys
import argparse
import logging
import random
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Ensure project root is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import identifier_entropy, multi_span, random_span
from src.model   import GraphCodeBERTDomainModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTA Dataset (Optimized)
# ---------------------------------------------------------------------------

class TTACodeDataset(Dataset):
    """
    Optimized Dataset:
    - Lazy loading from DataFrame (iloc)
    - Tokenizes once per sample
    - Performs span sampling on token IDs
    - Deterministic seeding per index
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=512, tta_views=5):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.tta_views = tta_views
        
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        code = str(row["code"])
        
        # 1. Label safety
        raw_label = row.get("label", 0)
        label = 1 if str(raw_label).lower() in ["1", "ai", "true"] else 0
        
        # 2. Precomputed entropy
        entropy = float(row.get("entropy", 0.0))

        # 3. Tokenize ONCE (with a buffer for spanning)
        # We grab up to 2048 tokens to allow diverse spans
        enc = self.tokenizer(
            code, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=2048, 
            return_tensors=None
        )
        base_tokens = enc["input_ids"]

        # 4. Generate Views (Deterministic)
        rng = random.Random(42 + idx)
        views_ids = []
        views_masks = []

        for v in range(self.tta_views):
            if v == 0:
                # View 0: Direct prefix
                ids = base_tokens[:self.max_len - 2]
            elif v % 2 == 1:
                # Odd: Multi span
                ids = multi_span(base_tokens, self.max_len)
            else:
                # Even: Random span
                ids = random_span(base_tokens, self.max_len)

            # Wrap with special tokens
            if self.cls_id is not None and self.sep_id is not None:
                ids = [self.cls_id] + ids + [self.sep_id]

            # Post-process (Truncate/Pad)
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
                if self.sep_id is not None: ids[-1] = self.sep_id
                mask = [1] * self.max_len
            else:
                pad_len = self.max_len - len(ids)
                mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [self.pad_id] * pad_len
            
            views_ids.append(torch.tensor(ids, dtype=torch.long))
            views_masks.append(torch.tensor(mask, dtype=torch.long))

        return {
            "input_ids":      torch.stack(views_ids),   # [V, L]
            "attention_mask": torch.stack(views_masks), # [V, L]
            "label":          torch.tensor(label,   dtype=torch.long),
            "entropy":        torch.tensor(entropy, dtype=torch.float),
        }

def tta_collate_fn(batch):
    """
    Modified to support B*V flattening:
    Returns flattened tensors [B*V, L]
    """
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)      # [B*V, L]
    masks     = torch.cat([item["attention_mask"] for item in batch], dim=0) # [B*V, L]
    
    # Entropy needs to be repeated V times for the flattened batch
    tta_views = batch[0]["input_ids"].size(0)
    entropy = torch.cat([item["entropy"].repeat(tta_views) for item in batch]) # [B*V]
    
    return {
        "input_ids":      input_ids,
        "attention_mask": masks,
        "entropy":        entropy,
        "labels":         torch.stack([item["label"] for item in batch]), # [B]
    }

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference Device: {device}")

    # 1. Load Data
    logger.info(f"Loading test data: {args.test_file}")
    if args.test_file.endswith(".parquet"):
        test_df = pd.read_parquet(args.test_file)
    else:
        test_df = pd.read_csv(args.test_file)

    # 2. Precompute Entropy (CPU efficient)
    if "entropy" not in test_df.columns:
        logger.info("Precomputing identifier entropy...")
        test_df["entropy"] = test_df["code"].apply(identifier_entropy)
    
    # 3. Setup Model
    state_dict_path = os.path.join(args.checkpoint_dir, "model.pt")
    if not os.path.exists(state_dict_path):
        logger.error(f"Cannot find model.pt in {args.checkpoint_dir}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = GraphCodeBERTDomainModel(
        num_generators = args.num_generators, 
        num_languages  = args.num_languages,
        num_domains    = args.num_domains,
        model_name     = args.model_name,
    )
    
    logger.info(f"Loading weights from {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    
    # Filter out domain heads as they might mismatch in size and aren't used for inference
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if "generator_head" not in k and "language_head" not in k and "domain_head" not in k
    }
    
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        logger.info(f"Missing keys (expected for domain heads): {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
    model.to(device)
    model.eval()

    # 4. DataLoader
    dataset = TTACodeDataset(test_df, tokenizer, max_length=args.max_len, tta_views=args.tta_views)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=tta_collate_fn,
        pin_memory=True
    )

    # 5. Pipeline
    logger.info(f"Running TTA Inference (Views={args.tta_views}, B*V Flattening=True)")
    all_final_probs = []
    all_labels = []
    has_labels = 'label' in test_df.columns

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            masks     = batch["attention_mask"].to(device, non_blocking=True)
            entropy   = batch["entropy"].to(device, non_blocking=True)
            
            # Forward pass [B*V, 2]
            out = model(input_ids, masks, entropy)
            logits = out["label_logits"] # [B*V, 2]
            
            # Reshape to [B, V, 2]
            batch_size = batch["labels"].size(0)
            logits = logits.view(batch_size, args.tta_views, 2)
            
            # Ensemble: softmax then mean (Better calibration)
            view_probs = F.softmax(logits, dim=-1)[:, :, 1] # [B, V]
            ai_probs = view_probs.mean(dim=1).cpu().numpy()  # [B]
            
            all_final_probs.extend(ai_probs)
            
            if has_labels:
                all_labels.extend(batch["labels"].numpy())

    # 6. Post-process
    all_final_probs = np.array(all_final_probs)
    
    logger.info(f"Decision threshold: {args.decision_threshold}")
    all_final_preds = (all_final_probs >= args.decision_threshold).astype(int)
    
    if has_labels:
        all_labels = np.array(all_labels)
        logger.info(f"\nEvaluation Results @ threshold {args.decision_threshold}:")
        acc = accuracy_score(all_labels, all_final_preds)
        auc = roc_auc_score(all_labels, all_final_probs)
        logger.info(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
        print("\n" + classification_report(all_labels, all_final_preds, target_names=["Human", "AI"], digits=4))

    # 7. Save output (Exclude 'code' if large)
    test_df['prediction'] = all_final_preds
    test_df['probability_ai'] = all_final_probs
    
    if args.output_file:
        out_path = args.output_file
    else:
        out_basename = os.path.basename(args.test_file).replace(".parquet", "_tta_preds.csv").replace(".csv", "_tta_preds.csv")
        out_path = out_basename
    
    if 'ID' in test_df.columns:
        logger.info(f"Formatting for Kaggle submission (id, label)...")
        submission_df = test_df[['ID', 'prediction']].rename(columns={'prediction': 'label'})
        submission_df.to_csv(out_path, index=False)

    else:
        # Default detailed format
        logger.info(f"Saving detailed results...")
        save_cols = [c for c in ['ID', 'label', 'prediction', 'probability_ai'] if c in test_df.columns]
        if not save_cols: save_cols = ['prediction', 'probability_ai']
        test_df[save_cols].to_csv(out_path, index=False)
    
    logger.info(f"Results saved to {out_path}")
    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--num_generators", type=int, default=10)
    parser.add_argument("--num_languages", type=int, default=10)
    parser.add_argument("--num_domains", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta_views", type=int, default=5)
    parser.add_argument("--decision_threshold", type=float, default=0.5,
                        help="Probability threshold used to classify AI vs Human")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the output CSV (defaults to current directory)")
    
    args = parser.parse_args()
    run_inference(args)
