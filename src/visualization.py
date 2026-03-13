"""
visualization.py
----------------
Utilities for plotting training history from CSV.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def save_training_plots(csv_path: str, output_dir: str) -> None:
    """
    Reads history.csv and generates a summary plot.
    """
    if not os.path.exists(csv_path):
        logger.warning("History CSV not found at %s. Skipping plot.", csv_path)
        return

    try:
        df = pd.read_csv(csv_path)
        if len(df) < 1:
            return

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Training Summary", fontsize=16)

        # 1. Losses
        loss_cols = [c for c in df.columns if "L_" in c and "val" not in c]
        if loss_cols:
            df.plot(x="epoch", y=loss_cols, ax=axes[0, 0], marker='o')
            axes[0, 0].set_title("Training Losses")
            axes[0, 0].set_ylabel("Loss")

        # 2. Validation Metrics
        metric_cols = ["accuracy", "f1", "roc_auc"]
        metric_cols = [c for c in metric_cols if c in df.columns]
        if metric_cols:
            df.plot(x="epoch", y=metric_cols, ax=axes[0, 1], marker='s')
            axes[0, 1].set_title("Validation Metrics")
            axes[0, 1].set_ylabel("Score")
            axes[0, 1].set_ylim(0, 1.05)

        # 3. Learning Rate
        if "lr" in df.columns:
            df.plot(x="epoch", y="lr", ax=axes[1, 0], color='orange', marker='^')
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_ylabel("LR")

        # 4. GRL Lambda
        if "lam" in df.columns:
            df.plot(x="epoch", y="lam", ax=axes[1, 1], color='green', marker='v')
            axes[1, 1].set_title("GRL Lambda (λ)")
            axes[1, 1].set_ylabel("λ")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_path = os.path.join(output_dir, "training_summary.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info("Training plots saved → %s", output_path)

    except Exception as e:
        logger.error("Failed to generate training plots: %s", e)
