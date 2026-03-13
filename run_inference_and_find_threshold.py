"""
run_inference_and_find_threshold.py
-------------------------------------
Automated threshold optimization pipeline for the OOD Meta-Training classifier.

Pipeline:
1. Run TTA inference (reusing meta_inference.py components — no code duplication).
2. Collect probabilities and ground-truth labels in memory.
3. Sweep 1000 threshold candidates over [0.0, 1.0].
4. Report optimal thresholds for F1 and Balanced Accuracy.
5. Save ROC curve, Precision-Recall curve, and threshold analysis plots.

Usage:
    python run_inference_and_find_threshold.py \\
        --data_file dataset.parquet \\
        --checkpoint_dir checkpoints/model_dir \\
        --tta_views 5

Requirements:
    Ground-truth "label" column must exist in the dataset.
    If not, the script exits with a clear error.
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure project root is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reuse ALL inference components from meta_inference — no duplication
from meta_inference import TTACodeDataset, tta_collate_fn
from extract_pseudo_samples import load_model  # Reuse the robust model loader
from src.dataset import identifier_entropy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Run TTA Inference (collect labels + probs in memory)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_tta_inference(model, dataloader, device: torch.device, tta_views: int):
    """
    Run TTA inference.

    Returns:
        all_probs  : np.ndarray [N]  — mean probability across TTA views
        all_labels : np.ndarray [N]  — ground-truth labels
    """
    all_probs  = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        masks     = batch["attention_mask"].to(device, non_blocking=True)
        entropy   = batch["entropy"].to(device, non_blocking=True)

        out    = model(input_ids, masks, entropy)
        logits = out["label_logits"]                         # [B*V, 2]

        batch_size = batch["labels"].size(0)
        logits = logits.view(batch_size, tta_views, 2)       # [B, V, 2]

        view_probs = F.softmax(logits, dim=-1)[:, :, 1]          # [B, V] AI probabilities
        probs = view_probs.mean(dim=1)                           # [B]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

    return np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Step 2 — Threshold Sweep
# ---------------------------------------------------------------------------
def sweep_thresholds(probs, labels, n_steps: int = 1000):
    """
    Sweep thresholds from 0.0 to 1.0.

    Returns a dict of arrays indexed by threshold.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, balanced_accuracy_score,
    )

    thresholds = np.linspace(0.001, 0.999, n_steps)
    results = {
        "threshold":         thresholds,
        "f1":                np.zeros(n_steps),
        "accuracy":          np.zeros(n_steps),
        "precision":         np.zeros(n_steps),
        "recall":            np.zeros(n_steps),
        "balanced_accuracy": np.zeros(n_steps),
    }

    for i, thr in enumerate(thresholds):
        preds = (probs >= thr).astype(int)
        results["f1"][i]                = f1_score(labels, preds, zero_division=0)
        results["accuracy"][i]          = accuracy_score(labels, preds)
        results["precision"][i]         = precision_score(labels, preds, zero_division=0)
        results["recall"][i]            = recall_score(labels, preds, zero_division=0)
        results["balanced_accuracy"][i] = balanced_accuracy_score(labels, preds)

    return results


# ---------------------------------------------------------------------------
# Step 3 — Print Optimal Thresholds
# ---------------------------------------------------------------------------
def report_results(sweep, probs, labels):
    from sklearn.metrics import roc_auc_score

    best_f1_idx  = np.argmax(sweep["f1"])
    best_acc_idx = np.argmax(sweep["balanced_accuracy"])
    roc_auc      = roc_auc_score(labels, probs)

    thr_f1  = sweep["threshold"][best_f1_idx]
    thr_acc = sweep["threshold"][best_acc_idx]

    print("\n" + "=" * 50)
    print("  Threshold Optimization Results")
    print("=" * 50)

    print(f"\n  Best threshold (F1)         : {thr_f1:.4f}")
    print(f"  F1 score                    : {sweep['f1'][best_f1_idx]:.4f}")
    print(f"  Accuracy                    : {sweep['accuracy'][best_f1_idx]:.4f}")
    print(f"  Precision                   : {sweep['precision'][best_f1_idx]:.4f}")
    print(f"  Recall                      : {sweep['recall'][best_f1_idx]:.4f}")
    print(f"  Balanced Accuracy           : {sweep['balanced_accuracy'][best_f1_idx]:.4f}")

    print(f"\n  Best threshold (Bal.Acc)    : {thr_acc:.4f}")
    print(f"  Balanced Accuracy           : {sweep['balanced_accuracy'][best_acc_idx]:.4f}")
    print(f"  Accuracy                    : {sweep['accuracy'][best_acc_idx]:.4f}")
    print(f"  F1 score                    : {sweep['f1'][best_acc_idx]:.4f}")
    print(f"  Precision                   : {sweep['precision'][best_acc_idx]:.4f}")
    print(f"  Recall                      : {sweep['recall'][best_acc_idx]:.4f}")

    print(f"\n  ROC-AUC                     : {roc_auc:.4f}")
    print("=" * 50 + "\n")

    return thr_f1, thr_acc, roc_auc


# ---------------------------------------------------------------------------
# Step 4 — Visualization
# ---------------------------------------------------------------------------
def save_plots(sweep, probs, labels, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    thresholds = sweep["threshold"]

    # --- Plot 1: Threshold vs F1 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sweep["f1"], color="#2563eb", lw=2, label="F1 Score")
    best_f1_idx = np.argmax(sweep["f1"])
    ax.axvline(thresholds[best_f1_idx], color="#ef4444", linestyle="--",
               label=f"Best thr={thresholds[best_f1_idx]:.3f} (F1={sweep['f1'][best_f1_idx]:.3f})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Threshold vs F1 Score", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_vs_f1.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {os.path.join(output_dir, 'threshold_vs_f1.png')}")

    # --- Plot 2: Threshold vs Balanced Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sweep["balanced_accuracy"], color="#7c3aed", lw=2, label="Balanced Accuracy")
    best_acc_idx = np.argmax(sweep["balanced_accuracy"])
    ax.axvline(thresholds[best_acc_idx], color="#ef4444", linestyle="--",
               label=f"Best thr={thresholds[best_acc_idx]:.3f} (BA={sweep['balanced_accuracy'][best_acc_idx]:.3f})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Threshold vs Balanced Accuracy", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_vs_balanced_accuracy.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {os.path.join(output_dir, 'threshold_vs_balanced_accuracy.png')}")

    # --- Plot 3: ROC Curve ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, lw=2, color="#2563eb", label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {os.path.join(output_dir, 'roc_curve.png')}")

    # --- Plot 4: Precision-Recall Curve ---
    prec, rec, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(rec, prec, lw=2, color="#059669", label=f"PR AUC = {pr_auc:.4f}")
    baseline = labels.mean()
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline (P={baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {os.path.join(output_dir, 'precision_recall_curve.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Load data
    logger.info(f"Loading data: {args.data_file}")
    if args.data_file.endswith(".parquet"):
        df = pd.read_parquet(args.data_file)
    else:
        df = pd.read_csv(args.data_file)

    # Validate that labels are available
    if "label" not in df.columns:
        logger.error(
            "Ground truth labels required for threshold optimization. "
            "Column 'label' not found in dataset. Exiting."
        )
        sys.exit(1)

    logger.info(f"Dataset loaded: {len(df)} samples | label dist: {df['label'].value_counts().to_dict()}")

    # 2. Precompute entropy
    if "entropy" not in df.columns:
        logger.info("Precomputing identifier entropy...")
        df["entropy"] = df["code"].apply(identifier_entropy)

    # 3. Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_model(args.checkpoint_dir, args.model_name, args.num_generators, args.num_languages, device)

    # 4. DataLoader (B*V flattening, same as meta_inference)
    dataset = TTACodeDataset(df, tokenizer, max_length=args.max_len, tta_views=args.tta_views)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tta_collate_fn,
        pin_memory=True,
    )

    # 5. Run inference (keep in memory — no CSV write)
    logger.info(f"Running TTA Inference (views={args.tta_views})...")
    probs, labels = run_tta_inference(model, dataloader, device, args.tta_views)

    # 6. Threshold sweep (1000 steps)
    logger.info("Sweeping 1000 thresholds...")
    sweep = sweep_thresholds(probs, labels, n_steps=1000)

    # 7. Report results
    thr_f1, thr_acc, _ = report_results(sweep, probs, labels)

    # 8. Save plots
    logger.info(f"Saving analysis plots to ./{args.plot_dir}/...")
    save_plots(sweep, probs, labels, output_dir=args.plot_dir)

    logger.info("Threshold optimization complete.")
    return thr_f1, thr_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated threshold optimization for AI-code detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_file",       type=str, required=True,
                        help="Dataset path (.parquet or .csv); must contain 'label' column.")
    parser.add_argument("--checkpoint_dir",  type=str, required=True,
                        help="Directory containing model.pt")
    parser.add_argument("--model_name",      type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--num_generators",  type=int, default=35)
    parser.add_argument("--num_languages",   type=int, default=3)
    parser.add_argument("--tta_views",       type=int, default=5)
    parser.add_argument("--max_len",         type=int, default=512)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--plot_dir",        type=str, default="threshold_analysis",
                        help="Directory to save analysis plots")

    args = parser.parse_args()
    run(args)
