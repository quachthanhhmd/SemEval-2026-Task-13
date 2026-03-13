"""
extract_pseudo_samples.py
--------------------------
Extracts high-quality pseudo-labels from an unlabeled test dataset by
leveraging the existing TTA inference pipeline from `meta_inference.py`.

Pipeline:
1. Run TTA inference (reusing TTACodeDataset, tta_collate_fn, and model loading).
2. Collect per-view softmax probabilities to compute rich uncertainty statistics.
3. Split samples into 3 groups based on confidence thresholds:
   - pseudo_easy.csv   : HIGH confidence labels  (mean > 0.95 AND var < 0.02)
   - pseudo_medium.csv : MEDIUM confidence labels (0.60 < mean <= 0.95)
   - pseudo_hard.csv   : LOW confidence / domain shift  (var >= 0.05 OR disagree >= 0.40)
4. Print summary statistics.

Usage:
    python extract_pseudo_samples.py \\
        --test_file test.parquet \\
        --checkpoint_dir checkpoints/model \\
        --output_dir pseudo_data/ \\
        --tta_views 5
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure project root is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reuse everything from the existing inference pipeline
from meta_inference import TTACodeDataset, tta_collate_fn
from src.dataset import identifier_entropy
from src.model import GraphCodeBERTDomainModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence Thresholds
# ---------------------------------------------------------------------------
HIGH_CONF_MEAN_MIN   = 0.95   # mean_prob > this for easy AI
HIGH_CONF_MEAN_MAX   = 0.05   # mean_prob < this for easy Human (1 - 0.95)
HIGH_CONF_VAR_MAX    = 0.02   # variance < this for easy
MEDIUM_CONF_MEAN_MIN = 0.60   # 0.60 < mean <= 0.95 for medium
MEDIUM_CONF_MEAN_MAX = 0.40   # < 0.40 for medium Human (symmetric)
HARD_VAR_MIN         = 0.05   # variance >= this for hard
HARD_DISAGREE_MIN    = 0.40   # disagreement >= this for hard


# ---------------------------------------------------------------------------
# Load Model with Robust Weight Handling
# ---------------------------------------------------------------------------
def load_model(checkpoint_dir: str, model_name: str, num_generators: int, num_languages: int, device: torch.device):
    state_dict_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(state_dict_path):
        logger.error(f"Cannot find model.pt in {checkpoint_dir}")
        sys.exit(1)

    model = GraphCodeBERTDomainModel(
        num_generators=num_generators,
        num_languages=num_languages,
        model_name=model_name,
    )

    logger.info(f"Loading weights from {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu")

    # Domain heads may have mismatched sizes (trained on different registry) — filter them
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if "generator_head" not in k and "language_head" not in k
    }
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        logger.info(f"Missing keys (expected for domain heads): {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core: Run TTA Inference and Collect Full View Statistics
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_tta_and_collect_stats(model, dataloader, device: torch.device, tta_views: int):
    """
    Run TTA inference and collect per-view probabilities.

    Returns:
        mean_probs  : np.ndarray [N]  — mean of TTA view probs
        var_probs   : np.ndarray [N]  — variance across views
        disagree    : np.ndarray [N]  — max - min across views
    """
    all_mean_probs  = []
    all_var_probs   = []
    all_disagree    = []
    all_gt_labels   = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)       # [B*V, L]
        masks     = batch["attention_mask"].to(device, non_blocking=True)  # [B*V, L]
        entropy   = batch["entropy"].to(device, non_blocking=True)         # [B*V]

        out    = model(input_ids, masks, entropy)
        logits = out["label_logits"]                                        # [B*V, 2]

        batch_size = batch["labels"].size(0)
        logits = logits.view(batch_size, tta_views, 2)                     # [B, V, 2]

        view_probs = F.softmax(logits, dim=-1)[:, :, 1]                    # [B, V] AI probabilities

        mean_prob = view_probs.mean(dim=1).cpu().numpy()                   # [B]
        var_prob  = view_probs.var(dim=1).cpu().numpy()                    # [B]
        disagree  = (view_probs.max(dim=1).values - view_probs.min(dim=1).values).cpu().numpy()  # [B]

        all_mean_probs.extend(mean_prob)
        all_var_probs.extend(var_prob)
        all_disagree.extend(disagree)
        all_gt_labels.extend(batch["labels"].numpy())

    return (
        np.array(all_mean_probs),
        np.array(all_var_probs),
        np.array(all_disagree),
        np.array(all_gt_labels),
    )


# ---------------------------------------------------------------------------
# Categorize by Confidence
# ---------------------------------------------------------------------------
def categorize_samples(mean_probs, var_probs, disagree):
    """
    Returns boolean masks for easy, medium, and hard sets.

    - easy   : (mean > 0.95 OR mean < 0.05) AND var < 0.02
    - hard   : var >= 0.05 OR disagree >= 0.40 (dominant condition)
    - medium : everything else in 0.40 < mean <= 0.95
    """
    # Hard condition takes priority
    is_hard = (var_probs >= HARD_VAR_MIN) | (disagree >= HARD_DISAGREE_MIN)

    # Easy: very high or very low confidence, low variance
    is_easy_ai     = (mean_probs > HIGH_CONF_MEAN_MIN)  & (var_probs < HIGH_CONF_VAR_MAX)
    is_easy_human  = (mean_probs < HIGH_CONF_MEAN_MAX)  & (var_probs < HIGH_CONF_VAR_MAX)
    is_easy = (is_easy_ai | is_easy_human) & (~is_hard)

    # Medium: uncertain but not hard
    is_medium_ai    = (mean_probs > MEDIUM_CONF_MEAN_MAX) & (mean_probs <= HIGH_CONF_MEAN_MIN)
    is_medium_human = (mean_probs < (1 - MEDIUM_CONF_MEAN_MAX)) & (mean_probs >= HIGH_CONF_MEAN_MAX)
    is_medium = (is_medium_ai | is_medium_human) & (~is_hard) & (~is_easy)

    return is_easy, is_medium, is_hard


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_extraction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Load Data
    logger.info(f"Loading test data: {args.test_file}")
    if args.test_file.endswith(".parquet"):
        test_df = pd.read_parquet(args.test_file)
    else:
        test_df = pd.read_csv(args.test_file)

    # 2. Precompute Entropy
    if "entropy" not in test_df.columns:
        logger.info("Precomputing identifier entropy...")
        test_df["entropy"] = test_df["code"].apply(identifier_entropy)

    # 3. Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_model(args.checkpoint_dir, args.model_name, args.num_generators, args.num_languages, device)

    # 4. DataLoader (same as meta_inference)
    dataset = TTACodeDataset(test_df, tokenizer, max_length=args.max_len, tta_views=args.tta_views)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=tta_collate_fn,
        pin_memory=True,
    )

    # 5. TTA Inference — Collect View Statistics
    logger.info(f"Running TTA Inference (Views={args.tta_views}) for pseudo-label extraction...")
    mean_probs, var_probs, disagree, gt_labels = run_tta_and_collect_stats(
        model, dataloader, device, args.tta_views
    )

    # 6. Categorize
    is_easy, is_medium, is_hard = categorize_samples(mean_probs, var_probs, disagree)
    pseudo_labels = (mean_probs >= 0.5).astype(int)

    # 7. Build output base dataframe
    stats_df = test_df.copy().reset_index(drop=True)
    stats_df["mean_probability"]    = mean_probs
    stats_df["variance_probability"] = var_probs
    stats_df["disagreement"]        = disagree
    stats_df["pseudo_label"]        = pseudo_labels

    # Determine base columns to save
    id_col   = "id"    if "id"   in stats_df.columns else None
    code_col = "code"  if "code" in stats_df.columns and not args.no_code else None
    base_cols = [c for c in [id_col, code_col, "entropy"] if c is not None]
    stat_cols = ["mean_probability", "variance_probability", "disagreement"]

    # 8. Save datasets
    os.makedirs(args.output_dir, exist_ok=True)

    easy_df   = stats_df[is_easy][base_cols + stat_cols + ["pseudo_label"]].reset_index(drop=True)
    medium_df = stats_df[is_medium][base_cols + stat_cols].reset_index(drop=True)
    hard_df   = stats_df[is_hard][base_cols + stat_cols].reset_index(drop=True)

    easy_path   = os.path.join(args.output_dir, "pseudo_easy.csv")
    medium_path = os.path.join(args.output_dir, "pseudo_medium.csv")
    hard_path   = os.path.join(args.output_dir, "pseudo_hard.csv")

    easy_df.to_csv(easy_path,   index=False)
    medium_df.to_csv(medium_path, index=False)
    hard_df.to_csv(hard_path,   index=False)

    # 9. Summary Statistics
    n_total  = len(test_df)
    n_easy   = is_easy.sum()
    n_medium = is_medium.sum()
    n_hard   = is_hard.sum()
    n_uncat  = n_total - n_easy - n_medium - n_hard  # Should be ~0

    logger.info("=" * 60)
    logger.info("  PSEUDO-LABEL EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total samples   : {n_total}")
    logger.info(f"  Pseudo-labels   : {n_easy}  ({n_easy/n_total*100:.1f}%) → {easy_path}")
    logger.info(f"  Medium samples  : {n_medium} ({n_medium/n_total*100:.1f}%) → {medium_path}")
    logger.info(f"  Hard samples    : {n_hard}  ({n_hard/n_total*100:.1f}%) → {hard_path}")
    if n_uncat > 0:
        logger.info(f"  Uncategorized   : {n_uncat}")
    logger.info("=" * 60)

    if n_easy > 0:
        easy_ai    = easy_df["pseudo_label"].sum()
        easy_human = len(easy_df) - easy_ai
        logger.info(f"  Easy class dist : AI={easy_ai} ({easy_ai/n_easy*100:.1f}%) | Human={easy_human} ({easy_human/n_easy*100:.1f}%)")
        logger.info("=" * 60)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pseudo-labels from unlabeled test data for domain adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_file",       type=str, required=True,
                        help="Path to test data (.parquet or .csv)")
    parser.add_argument("--checkpoint_dir",  type=str, required=True,
                        help="Path to directory containing model.pt")
    parser.add_argument("--output_dir",      type=str, default="pseudo_data",
                        help="Directory to save pseudo_easy/medium/hard CSV files")
    parser.add_argument("--model_name",      type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--num_generators",  type=int, default=10)
    parser.add_argument("--num_languages",   type=int, default=10)
    parser.add_argument("--max_len",         type=int, default=512)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--tta_views",       type=int, default=5,
                        help="Number of TTA augmentation views per sample")
    parser.add_argument("--no_code",         action="store_true",
                        help="Exclude code column in output (useful for large datasets)")

    args = parser.parse_args()
    run_extraction(args)
