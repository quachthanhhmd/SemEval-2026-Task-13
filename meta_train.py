"""
train.py
--------
Entry point for the OOD Meta-Training Pipeline.

Usage:
    python train.py \\
        --train_csv train.csv \\
        --val_csv   val.csv   \\
        --batch_size 128      \\
        --max_len    512
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure project root is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import build_datasets
from src.model   import GraphCodeBERTDomainModel
from src.sampler  import LanguageBalancedSampler, domain_collate_fn
from src.trainer  import MetaTrainer
from src.features import AgnosticFeatureExtractor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOD Meta-Training — AI-generated code detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--train_csv",   type=str,   required=True,
                   help="Path to training CSV (code, label, language, generator)")
    p.add_argument("--val_csv",     type=str,   required=True,
                   help="Path to validation CSV")
    p.add_argument("--max_len",     type=int,   default=512,
                   help="Max token length")
    p.add_argument("--registry_path", type=str, default=None,
                   help="Where to save/load domain registry JSON")
    p.add_argument("--char_crop_limit", type=int, default=None,
                   help="Randomly crop code to this many characters")

    # ---- Model ----
    p.add_argument("--model_name",  type=str,
                   default="microsoft/graphcodebert-base",
                   help="HuggingFace model identifier")
    p.add_argument("--dropout",     type=float, default=0.1,
                   help="Dropout for shared feature layer")
    p.add_argument("--freeze_layers", type=int, default=0,
                   help="Number of encoder layers to freeze (0-12)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to save VRAM")

    # ---- Sampler ----
    p.add_argument("--batch_size",  type=int,   default=96,
                   help="Total batch size (= k * m)")
    p.add_argument("--k_langs",     type=int,   default=6,
                   help="Languages per batch (last one = meta_test)")
    p.add_argument("--m_per_lang",  type=int,   default=16,
                   help="Samples per language per batch")

    # ---- Training ----
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--accumulate_steps", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--lr_inner",    type=float, default=5e-5,
                   help="Inner-loop learning rate for MAML θ' update")
    p.add_argument("--weight_decay",type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="Fraction of steps used for linear warmup")
    p.add_argument("--fp16",        action="store_true",
                   help="Enable automatic mixed precision")

    # ---- Loss weights ----
    p.add_argument("--alpha",      type=float, default=0.2,
                   help="SupCon loss weight in L_train")
    p.add_argument("--alpha_meta", type=float, default=0.05,
                   help="SupCon loss weight in L_meta")
    p.add_argument("--beta",       type=float, default=0.05,

                   help="Generator adversarial loss weight")
    p.add_argument("--gamma",      type=float, default=0.05,
                   help="Language adversarial loss weight")
    p.add_argument("--delta",      type=float, default=1.0,
                   help="Meta-test loss weight")
    p.add_argument("--grl_scale",  type=float, default=5.0,
                   help="GRL λ schedule scale (lower = slower ramp)")

    # ---- Misc ----
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/meta_training")
    p.add_argument("--resume_from",  type=str, default=None,
                   help="Path to checkpoint directory to resume training from")
    p.add_argument("--log_every",    type=int, default=20,
                   help="Log every N global steps")
    p.add_argument("--use_wandb",    action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project",type=str, default="semeval-task13-ood",
                   help="W&B project name")
    p.add_argument("--wandb_run_name",type=str, default=None,
                   help="W&B run name")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Validate k * m == batch_size
    if args.k_langs * args.m_per_lang != args.batch_size:
        logger.warning(
            "k_langs(%d) * m_per_lang(%d) = %d ≠ batch_size(%d). "
            "Adjusting batch_size to k*m.",
            args.k_langs, args.m_per_lang,
            args.k_langs * args.m_per_lang, args.batch_size,
        )
        args.batch_size = args.k_langs * args.m_per_lang

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ----------------------------------------------------------------
    # 1. Tokenizer + Datasets
    # ----------------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Initializing stylometric feature extractor …")
    extractor = AgnosticFeatureExtractor()

    logger.info("Building datasets …")
    train_ds, val_ds, registry = build_datasets(
        train_file         = args.train_csv,
        val_file           = args.val_csv,
        tokenizer          = tokenizer,
        extractor          = extractor,
        max_length         = args.max_len,
        char_crop_limit    = args.char_crop_limit,
        registry_save_path = args.registry_path,
    )

    # ----------------------------------------------------------------
    # 2. DataLoaders
    # ----------------------------------------------------------------
    train_sampler = LanguageBalancedSampler(
        language_ids  = train_ds.language_ids_list,
        labels        = train_ds.labels_list,
        k             = args.k_langs,
        m             = args.m_per_lang,
        shuffle_langs = True,
        drop_last     = True,
        seed          = args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler = train_sampler,
        num_workers   = args.num_workers,
        pin_memory    = True,
        collate_fn    = domain_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    logger.info(
        "Train batches: %d | Val batches: %d",
        len(train_loader), len(val_loader),
    )

    # ----------------------------------------------------------------
    # 3. Model
    # ----------------------------------------------------------------
    logger.info("Initialising model …")
    model = GraphCodeBERTDomainModel(
        num_generators = registry.num_generators,
        num_languages  = registry.num_languages,
        num_style      = 10,
        model_name     = args.model_name,
        dropout        = args.dropout,
        gradient_checkpointing = args.gradient_checkpointing,
        freeze_layers  = args.freeze_layers,
    ).to(device)

    # ----------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # ----------------------------------------------------------------
    # Separate LR for backbone vs new heads
    backbone_params = list(model.encoder.parameters())
    new_params      = [
        p for n, p in model.named_parameters()
        if not n.startswith("encoder.") and p.requires_grad
    ]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": new_params,      "lr": args.lr},
        ],
        weight_decay = args.weight_decay,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    logger.info(
        "Total steps: %d | Warmup: %d (%.0f%%)",
        total_steps, warmup_steps, args.warmup_ratio * 100,
    )

    # ----------------------------------------------------------------
    # 5. W&B init (optional)
    # ----------------------------------------------------------------
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project = args.wandb_project,
                name    = args.wandb_run_name,
                config  = vars(args),
            )
            logger.info("W&B run: %s/%s", args.wandb_project, wandb.run.name)
        except Exception as e:
            logger.warning("W&B init failed: %s. Continuing without it.", e)
            args.use_wandb = False

    # ----------------------------------------------------------------
    # 6. Trainer & train
    # ----------------------------------------------------------------
    trainer = MetaTrainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        optimizer      = optimizer,
        scheduler      = scheduler,
        device         = device,
        alpha          = args.alpha,
        alpha_meta     = args.alpha_meta,
        beta           = args.beta,
        gamma          = args.gamma,
        delta          = args.delta,
        lr_inner       = args.lr_inner,
        grl_scale      = args.grl_scale,
        fp16           = args.fp16,
        accumulate_steps = args.accumulate_steps,
        log_every      = args.log_every,
        checkpoint_dir = args.checkpoint_dir,
        use_wandb      = args.use_wandb,
    )

    # ----------------------------------------------------------------
    # 7. Resume & Train
    # ----------------------------------------------------------------
    start_epoch = 1
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)

    logger.info("Starting training loop ...")
    trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)

    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    logger.info("Done. Best val F1 = %.4f", trainer.best_f1)


if __name__ == "__main__":
    main()
