"""
meta_inference.py
-----------------
Inference script for the OOD Meta-Training Pipeline (GraphCodeBERTDomainModel)
with Test-Time Augmentation (TTA) support via multiple span sampling.

Usage:
    python meta_inference.py \
        --test_parquet data/Task_A/test.parquet \
        --checkpoint_dir checkpoints/meta_training/best_model \
        --batch_size  32 \
        --tta_views   5
"""

import os
import sys
import argparse
import logging

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Ensure project root is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import build_datasets, CodeDataset, identifier_entropy, multi_span, random_span
from src.model   import GraphCodeBERTDomainModel
from src.features import AgnosticFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTA Dataset Wrapper
# ---------------------------------------------------------------------------

class TTACodeDataset(torch.utils.data.Dataset):
    """
    Dataset that applies span sampling at test time (TTA).
    Returns `tta_views` number of differently sampled views for the same code.
    """
    def __init__(self, df, tokenizer, extractor=None, max_length=512, tta_views=5):
        self.dataset   = df.to_dict('records') if isinstance(df, pd.DataFrame) else df
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.max_len   = max_length
        self.tta_views = tta_views

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        code = str(row["code"])
        label = int(row.get("label", 0))

        # 1. Stylometric Features (10 features)
        if self.extractor is not None:
            raw_features = self.extractor.extract_all(code)
            extra_features = torch.tensor(raw_features, dtype=torch.float)
            # Log normalization for unbounded features (0: avg_id_len, 6: line_std)
            for i in [0, 6]:
                extra_features[i] = torch.log1p(extra_features[i])
            extra_features = torch.clamp(extra_features, min=0.0, max=100.0)
        else:
            entropy = identifier_entropy(code)
            extra_features = torch.zeros(10, dtype=torch.float)
            extra_features[1] = entropy 

        # Baseline tokenize (no truncation yet)
        enc = self.tokenizer(code, add_special_tokens=True, truncation=False, return_tensors=None)
        original_ids = enc["input_ids"]

        views = []
        for v in range(self.tta_views):
            if v == 0:
                # View 0: Strict prefix (deterministic anchor)
                ids = original_ids[:self.max_len]
            elif v % 2 == 1:
                # Odd views: multi_span (middle/ends fusion)
                ids = multi_span(original_ids, self.max_len)
            else:
                # Even views: random continuous span
                ids = random_span(original_ids, self.max_len)

            # Pad / Truncate
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
                mask = [1] * self.max_len
            else:
                pad_len = self.max_len - len(ids)
                mask = [1] * len(ids) + [0] * pad_len
                ids  = ids + [self.tokenizer.pad_token_id] * pad_len
            
            views.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            })

        return {
            "views": views,  # List of dicts, length = tta_views
            "label": torch.tensor(label, dtype=torch.long),
            "extra_features": extra_features,
            "idx": idx,
        }

def tta_collate_fn(batch):
    # batch is list of item dicts
    views_collated = []
    tta_views = len(batch[0]["views"])
    
    for v in range(tta_views):
        input_ids = torch.stack([item["views"][v]["input_ids"] for item in batch])
        attention_mask = torch.stack([item["views"][v]["attention_mask"] for item in batch])
        views_collated.append({"input_ids": input_ids, "attention_mask": attention_mask})

    return {
        "views": views_collated, # List of Batched Inputs
        "label": torch.stack([item["label"] for item in batch]),
        "extra_features": torch.stack([item["extra_features"] for item in batch]),
        "idx": [item["idx"] for item in batch],
    }

# ---------------------------------------------------------------------------
# Inference Logic
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

    has_labels = 'label' in test_df.columns
    if not has_labels:
        test_df['label'] = 0

    # 2. Extract model config info from checkpoint/name
    state_dict_path = os.path.join(args.checkpoint_dir, "model.pt")
    if not os.path.exists(state_dict_path):
        logger.error(f"Cannot find model.pt in {args.checkpoint_dir}")
        sys.exit(1)

    # 3. Load Tokenizer & Model
    # Note: We hardcode model arguments assuming they match the meta-training config
    # In a fully integrated system, load these from config.yaml
    logger.info("Initializing Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = GraphCodeBERTDomainModel(
        num_generators = args.num_generators, 
        num_languages  = args.num_languages,
        num_style      = 10,
        model_name     = args.model_name,
    )
    
    logger.info(f"Loading weights from {state_dict_path}")
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    model.to(device)
    model.eval()

    # 4. DataLoader
    extractor = AgnosticFeatureExtractor()
    dataset = TTACodeDataset(test_df, tokenizer, extractor=extractor, max_length=args.max_len, tta_views=args.tta_views)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=tta_collate_fn)

    # 5. Prediction Loop with TTA
    logger.info(f"Running TTA Inference ({args.tta_views} views)...")
    all_final_probs = []
    all_labels = []

    with torch.no_grad():
        saved_debug_batches = 0
        for step, batch in enumerate(tqdm(dataloader, desc="Inferencing")):
            views = batch["views"]
            extra_features = batch["extra_features"].to(device)

            # --- Debug Batch Saving ---
            if saved_debug_batches < 3:
                try:
                    # Just save the first view for debugging the tokens
                    first_view = views[0]
                    b_dict = {
                        "input_ids": first_view["input_ids"].cpu().numpy().tolist(),
                        "label": batch["label"].cpu().numpy().tolist(),
                        "entropy": batch["extra_features"][:, 1].cpu().numpy().tolist(), # entropy is at index 1
                        "idx": batch["idx"] # idx is already a list
                    }
                    df_debug = pd.DataFrame(b_dict)
                    debug_path = os.path.join(args.checkpoint_dir, f"debug_inference_batch_{saved_debug_batches}.csv")
                    df_debug.to_csv(debug_path, index=False)
                    logger.info(f"Saved debug inference batch to {debug_path}")
                    saved_debug_batches += 1
                except Exception as e:
                    logger.warning(f"Failed to save debug inference batch: {e}")
            # --------------------------
            
            # shape will be [tta_views, batch_size, 2]
            view_probs = []
            for v_data in views:
                input_ids = v_data["input_ids"].to(device)
                mask = v_data["attention_mask"].to(device)
                
                out = model(input_ids, mask, extra_features)
                # Softmax to get probabilities
                probs = F.softmax(out["label_logits"], dim=-1)
                view_probs.append(probs)
                
            # Stack all views: [tta_views, batch_size, 2]
            stacked_probs = torch.stack(view_probs, dim=0)
            
            # Average probabilities across views (Soft Voting)
            mean_probs = torch.mean(stacked_probs, dim=0) # [batch_size, 2]
            
            # AI class probability is index 1
            ai_probs = mean_probs[:, 1].cpu().numpy()
            all_final_probs.extend(ai_probs)
            
            if has_labels:
                all_labels.extend(batch["label"].numpy())

    # 6. Report & Save
    all_final_probs = np.array(all_final_probs)
    all_final_preds = (all_final_probs >= 0.5).astype(int)
    
    if has_labels:
        print("\n" + "="*60)
        print("TTA TEST SET EVALUATION REPORT".center(60))
        print("="*60)
        
        acc = accuracy_score(all_labels, all_final_preds)
        f1  = classification_report(all_labels, all_final_preds, target_names=["Human", "AI"], output_dict=True)['AI']['f1-score']
        auc = roc_auc_score(all_labels, all_final_probs)
        
        print(f"\nAccuracy: {acc:.4f} | F1 (AI): {f1:.4f} | ROC-AUC: {auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_final_preds, target_names=["Human", "AI"], digits=4))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_final_preds))

    print("\nSaving predictions...")
    test_df['prediction'] = all_final_preds
    test_df['probability_ai'] = all_final_probs
    
    out_basename = os.path.basename(args.test_file).replace(".parquet", "_tta_predictions.csv").replace(".csv", "_tta_predictions.csv")
    out_path = os.path.join(os.path.dirname(args.test_file), out_basename)
    test_df[['code', 'prediction', 'probability_ai']].to_csv(out_path, index=False)
    logger.info(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_file", type=str, required=True, help="Path to test .parquet or .csv file")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the saved model folder containing model.pt")
    
    parser.add_argument("--model_name", type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--num_generators", type=int, default=35, help="Must match training setup")
    parser.add_argument("--num_languages", type=int, default=3, help="Must match training setup")
    
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta_views", type=int, default=5, help="Number of views for Test Time Augmentation inference")
    
    args = parser.parse_args()
    run_inference(args)
