"""
trainer.py
----------
MetaTrainer: Leave-One-Language Meta Training (First-Order MAML).

Fixes applied vs. original design
-----------------------------------
1. Leave-One-Language split instead of leave-one-domain:
   - meta_train = k-1 languages in the batch
   - meta_test  = 1 held-out language
   This directly simulates test-time distribution shift.

2. FOMAML (create_graph=False) instead of second-order MAML:
   - ≈same empirical performance; 2-3× faster; half VRAM.

3. L_meta now includes contrastive term:
   L_meta = L_label + 0.3 * L_contrastive
   So representation learning is also meta-optimised.

4. GRL λ schedule uses slower ramp (divisor 5 instead of 10):
   λ = 2 / (1 + exp(-5p)) - 1
   Prevents adversarial training from being too aggressive early on.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import math
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.losses import SupConLoss, compute_total_loss
from src.model import GraphCodeBERTDomainModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GRL λ schedule  (fix #6: slower ramp, divisor 5 instead of 10)
# ---------------------------------------------------------------------------

def grl_lambda_schedule(current_step: int, total_steps: int, scale: float = 5.0) -> float:
    """λ = 2/(1+exp(-scale*p)) − 1,  p = current_step/total_steps."""
    p = current_step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-scale * p)) - 1.0


# ---------------------------------------------------------------------------
# Leave-One-Language batch splitter  (fix #1 & #7)
# ---------------------------------------------------------------------------

def split_meta_language(
    batch: Dict[str, torch.Tensor],
) -> Tuple[Optional[Dict], Optional[Dict], int]:
    """
    Split a language-balanced batch into meta_train and meta_test by language.

    The LanguageBalancedSampler guarantees k distinct languages per batch
    with m consecutive samples each. We pick the last unique language as
    meta_test (deterministic — no random call needed since the sampler
    already shuffled language order each epoch).

    Returns (meta_train_batch, meta_test_batch, meta_test_lang_id)
    or (None, None, -1) if only one language is present.
    """
    lang_ids = batch["language_id"].tolist()
    unique_langs = list(dict.fromkeys(lang_ids))  # preserves order, unique

    if len(unique_langs) < 2:
        return None, None, -1

    # Last language in the batch is the meta_test language
    meta_test_lang = unique_langs[-1]

    device = batch["language_id"].device
    test_mask  = torch.tensor([l == meta_test_lang for l in lang_ids], device=device)
    train_mask = ~test_mask

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None, None, -1

    def _select(m: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v[m] for k, v in batch.items()}

    return _select(train_mask), _select(test_mask), int(meta_test_lang)


# ---------------------------------------------------------------------------
# Functional θ' update for FOMAML  (fix #3: create_graph=False)
# ---------------------------------------------------------------------------

def functional_params_update(
    named_params: Dict[str, torch.Tensor],
    grads:        Tuple[torch.Tensor, ...],
    param_names:  List[str],
    lr_inner:     float,
) -> Dict[str, torch.Tensor]:
    """θ' = θ − lr_inner * grad (FOMAML — gradients are detached)."""
    grad_map = dict(zip(param_names, grads))
    updated = {}
    for name, param in named_params.items():
        g = grad_map.get(name)
        if g is not None:
            updated[name] = param - lr_inner * g.detach()  # detach = FOMAML
        else:
            updated[name] = param
    return updated


# ---------------------------------------------------------------------------
# MetaTrainer
# ---------------------------------------------------------------------------

class MetaTrainer:
    """
    Parameters
    ----------
    model           : GraphCodeBERTDomainModel
    train_loader    : DataLoader using LanguageBalancedSampler
    val_loader      : DataLoader (standard)
    optimizer       : AdamW
    scheduler       : LR scheduler
    device          : torch device
    alpha           : SupCon weight in L_train
    alpha_meta      : SupCon weight in L_meta  (default 0.3)
    beta            : generator adversarial weight
    gamma           : language adversarial weight
    delta           : meta-test loss multiplier
    lr_inner        : FOMAML inner-loop LR
    grl_scale       : scale factor in GRL λ schedule (default 5.0)
    fp16            : AMP
    log_every       : log every N global steps
    checkpoint_dir  : where to save best model
    use_wandb       : attempt wandb logging
    """

    def __init__(
        self,
        model:          GraphCodeBERTDomainModel,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        optimizer:      torch.optim.Optimizer,
        scheduler,
        device:         torch.device,
        alpha:          float = 0.5,
        alpha_meta:     float = 0.3,
        beta:           float = 0.1,
        gamma:          float = 0.1,
        delta:          float = 1.0,
        lr_inner:       float = 1e-4,
        grl_scale:      float = 5.0,
        fp16:           bool  = True,
        log_every:      int   = 20,
        checkpoint_dir: str   = "checkpoints",
        use_wandb:      bool  = False,
        adv_warmup_epochs: int = 1,
        adv_warmup_steps: int = 0,
    ) -> None:
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.alpha          = alpha
        self.alpha_meta     = alpha_meta
        self.beta           = beta
        self.gamma          = gamma
        self.delta          = delta
        self.lr_inner       = lr_inner
        self.grl_scale      = grl_scale
        self.fp16           = fp16 and torch.cuda.is_available()
        self.log_every      = log_every
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb      = use_wandb
        self.adv_warmup_epochs = adv_warmup_epochs
        self.adv_warmup_steps  = adv_warmup_steps
        self.current_epoch  = 0
        self.eff_warmup_steps = 0  # Unified threshold
        self.current_epoch  = 0

        self.scaler    = GradScaler("cuda", enabled=self.fp16)
        self.supcon_fn = SupConLoss(temperature=0.07).to(device)

        os.makedirs(checkpoint_dir, exist_ok=True)

        tb_logdir = os.path.join(checkpoint_dir, "tb_logs")
        self.writer = SummaryWriter(log_dir=tb_logdir)
        logger.info("TensorBoard logs → %s", tb_logdir)

        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed; using TensorBoard only.")

        self.global_step = 0
        self.total_steps = 0  # To be set in train()
        self.best_f1     = 0.0
        self.history_path = os.path.join(self.checkpoint_dir, "history.csv")
        self._init_history()

    def get_current_lambda(self) -> float:
        if self.global_step < self.eff_warmup_steps:
            return 0.0
        return grl_lambda_schedule(
            self.global_step - self.eff_warmup_steps,
            max(1, self.total_steps - self.eff_warmup_steps),
            self.grl_scale
        )

    def _init_history(self) -> None:
        """Initialize history CSV if it doesn't exist."""
        if not os.path.exists(self.history_path):
            with open(self.history_path, "w", encoding="utf-8") as f:
                # Basic columns
                cols = ["epoch", "step", "lr", "lam"]
                # Placeholders for metrics (will be added dynamically)
                f.write(",".join(cols) + "\n")

    # ------------------------------------------------------------------
    def _to(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------
    def _forward(
        self,
        batch:  Dict[str, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
        buffers: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass — uses functional_call if params (θ') are given."""
        if params is not None:
            # Use current model buffers if none provided
            b = buffers if buffers is not None else dict(self.model.named_buffers())
            return self.model.forward_with_params(
                batch["input_ids"], batch["attention_mask"], batch["entropy"], params, buffers=b
            )
        return self.model(batch["input_ids"], batch["attention_mask"], batch["entropy"])

    # ------------------------------------------------------------------
    def _compute_train_loss(
        self,
        batch:  Dict[str, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
        buffers: Optional[Dict[str, torch.Tensor]] = None,
        beta:   Optional[float] = None,
        gamma:  Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full L_train = L_label + α·L_con + β·L_gen + γ·L_lang."""
        out = self._forward(batch, params, buffers)
        
        total, breakdown = compute_total_loss(
            label_logits     = out["label_logits"],
            generator_logits = out["generator_logits"],
            language_logits  = out["language_logits"],
            projection       = out["projection"],
            labels           = batch["label"],
            generator_ids    = batch["generator_id"],
            language_ids     = batch["language_id"],
            supcon_fn        = self.supcon_fn,
            gamma            = gamma if gamma is not None else self.gamma
        )
        return total, breakdown

    # ------------------------------------------------------------------
    def _compute_meta_loss(
        self,
        batch:  Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        L_meta = L_label + alpha_meta * L_contrastive
        No adversarial terms in meta-test step.
        """
        out = self._forward(batch, params, buffers)
        L_label = F.cross_entropy(out["label_logits"], batch["label"])
        
        # Use pure binary labels for meta-test contrastive: AI vs Human invariant
        L_con   = self.supcon_fn(out["projection"], batch["label"])
        return L_label + self.alpha_meta * L_con

    # ------------------------------------------------------------------
    def _train_step(
        self,
        batch:       Dict[str, torch.Tensor],
        total_steps: int,
    ) -> Dict[str, float]:
        """One FOMAML meta-training step."""

        # ---- Smooth Adversarial Warmup: scale weights by lam ----
        eff_lam = self.get_current_lambda()
        eff_beta  = self.beta * eff_lam
        eff_gamma = self.gamma * eff_lam
        self.model.set_lambda(eff_lam)

        # ---- Leave-One-Language split ----
        meta_train_batch, meta_test_batch, _ = split_meta_language(batch)
        # Skip meta step if meta-test is too small for stable contrastive
        use_meta = (meta_train_batch is not None and meta_test_batch["label"].shape[0] >= 4)

        self.optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------------------
        # Case A: meta split succeeded — FOMAML inner/outer loop
        # ----------------------------------------------------------------
        if use_meta:
            meta_train_batch = self._to(meta_train_batch)
            meta_test_batch  = self._to(meta_test_batch)

            # Params + Buffers for functional call
            param_names  = [n for n, p in self.model.named_parameters() if p.requires_grad]
            named_params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            named_buffers = dict(self.model.named_buffers())

            # Step 1: Inner Adaptation Pass (Classification only)
            # We run a dedicated forward pass for θ' and use autocast=False for grad stability
            with autocast(device_type=self.device.type, enabled=False):
                out_inner = self.model(
                    meta_train_batch["input_ids"], 
                    meta_train_batch["attention_mask"], 
                    meta_train_batch["entropy"]
                )
                
                L_label_inner = F.cross_entropy(out_inner["label_logits"], meta_train_batch["label"])
                # Inner Adaptation: Optimize for pure AI vs Human invariant
                L_con_inner = self.supcon_fn(out_inner["projection"], meta_train_batch["label"])
                L_inner = L_label_inner + self.alpha * L_con_inner

            # FOMAML Inner Step: Derive θ' from classification task only
            # create_graph=False as we don't do second-order derivs in FO-MAML
            grads = torch.autograd.grad(L_inner, list(named_params.values()), create_graph=False, allow_unused=True)
            grads = tuple(g if g is not None else torch.zeros_like(p) for g, p in zip(grads, named_params.values()))
            theta_prime = functional_params_update(named_params, grads, param_names, self.lr_inner)

            # Step 2: Outer/Final Pass (Full Regularization + Meta-Test)
            # This is the "real" update pass using θ and θ'. 
            # We run it under GradScaler/AMP if enabled.
            with autocast(device_type=self.device.type, enabled=self.fp16):
                # A) Loss on Meta-Train using base θ (Full Adversarial Regularization)
                out_outer = self._forward(meta_train_batch)
                L_train, breakdown = compute_total_loss(
                    label_logits     = out_outer["label_logits"],
                    generator_logits = out_outer["generator_logits"],
                    language_logits  = out_outer["language_logits"],
                    projection       = out_outer["projection"],
                    labels           = meta_train_batch["label"],
                    generator_ids    = meta_train_batch["generator_id"],
                    language_ids     = meta_train_batch["language_id"],
                    supcon_fn        = self.supcon_fn,
                    alpha            = self.alpha,
                    beta             = eff_beta,
                    gamma            = eff_gamma
                )

            # Meta step
            with autocast(device_type=self.device.type, enabled=self.fp16):
                L_meta = self._compute_meta_loss(meta_test_batch, theta_prime, named_buffers)

            L_final = L_train + self.delta * L_meta

        # ----------------------------------------------------------------
        # Case B: meta split failed — Standard training (No inner loop)
        # ----------------------------------------------------------------
        else:
            batch = self._to(batch)
            with autocast(device_type=self.device.type, enabled=self.fp16):
                # Forward pass on full batch
                out = self._forward(batch)
                L_final, breakdown = compute_total_loss(
                    label_logits     = out["label_logits"],
                    generator_logits = out["generator_logits"],
                    language_logits  = out["language_logits"],
                    projection       = out["projection"],
                    labels           = batch["label"],
                    generator_ids    = batch["generator_id"],
                    language_ids     = batch["language_id"],
                    supcon_fn        = self.supcon_fn,
                    alpha            = self.alpha,
                    beta             = eff_beta,
                    gamma            = eff_gamma
                )
            L_meta = torch.tensor(0.0, device=self.device)

        # ---- Backward & update ----
        self.scaler.scale(L_final).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        log = {k: v.item() for k, v in breakdown.items()}
        log["L_meta"]  = L_meta.item()
        log["L_final"] = L_final.item()
        log["grl_lam"] = eff_lam
        return log

    # ------------------------------------------------------------------
    def train(self, num_epochs: int = 0, num_steps: int = 0, start_epoch: int = 1) -> None:
        self.model.train()
        # Selective Unfreezing: Keep layers 8-11 trainable for meta-adaptation
        # GraphCodeBERT has 12 layers (0-11). Freezing the first 8 layers preserves 
        # base features while allowing the top 4 layers to adapt.
        for name, param in self.model.encoder.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in [8, 9, 10, 11]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        steps_per_epoch = len(self.train_loader)
        
        if num_epochs > 0:
            total_steps = num_epochs * steps_per_epoch
            # Warmup: use epochs if specified, else steps
            if self.adv_warmup_epochs > 0:
                self.eff_warmup_steps = self.adv_warmup_epochs * steps_per_epoch
            else:
                self.eff_warmup_steps = self.adv_warmup_steps
        else:
            total_steps = num_steps
            self.eff_warmup_steps = self.adv_warmup_steps

        self.total_steps = total_steps

        # Adjust start if needed
        actual_start_epoch = start_epoch - 1
        
        logger.info(
            "Training setup: total_steps=%d | warmup_steps=%d",
            total_steps, self.eff_warmup_steps
        )

        saved_debug_batches = 0

        self.train_start_time = time.time()

        for epoch in range(actual_start_epoch, num_epochs if num_epochs > 0 else 10000):
            if num_steps > 0 and self.global_step >= num_steps:
                break
                
            self.current_epoch = epoch
            epoch_logs: Dict[str, float] = defaultdict(float)

            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            for step, batch in enumerate(self.train_loader):
                # --- Debug Batch Saving ---
                if epoch == 1 and saved_debug_batches < 3:
                    try:
                        import pandas as pd
                        b_dict = {
                            "input_ids": batch["input_ids"].cpu().numpy().tolist(),
                            "label": batch["label"].cpu().numpy().tolist(),
                            "language_id": batch.get("language_id", torch.zeros_like(batch["label"])).cpu().numpy().tolist(),
                            "generator_id": batch.get("generator_id", torch.zeros_like(batch["label"])).cpu().numpy().tolist(),
                            "entropy": batch.get("entropy", torch.zeros_like(batch["label"])).cpu().numpy().tolist()
                        }
                        df_debug = pd.DataFrame(b_dict)
                        debug_path = os.path.join(self.checkpoint_dir, f"debug_train_batch_{saved_debug_batches}.csv")
                        df_debug.to_csv(debug_path, index=False)
                        logger.info(f"Saved debug batch to {debug_path}")
                        saved_debug_batches += 1
                    except Exception as e:
                        logger.warning(f"Failed to save debug batch: {e}")
                # --------------------------

                step_log = self._train_step(batch, total_steps)
                self.global_step += 1

                for k, v in step_log.items():
                    epoch_logs[k] += v

                if self.global_step % self.log_every == 0:
                    self._log(step_log, prefix="train/step")
                    
                    # Compute ETA
                    elapsed = time.time() - self.train_start_time
                    steps_done = self.global_step
                    steps_left = self.total_steps - steps_done
                    sec_per_step = elapsed / max(steps_done, 1)
                    eta_sec = steps_left * sec_per_step
                    
                    # Format ETA
                    if eta_sec > 3600:
                        eta_str = f"{eta_sec // 3600:.0f}h { (eta_sec % 3600) // 60:.0f}m"
                    else:
                        eta_str = f"{eta_sec // 60:.0f}m {eta_sec % 60:.0f}s"
                    
                    print(
                        f"Epoch {epoch+1} | "
                        f"step {step+1}/{len(self.train_loader)} "
                        f"(glob {self.global_step}/{self.total_steps}) | "
                        f"L={step_log['L_final']:.4f} | "
                        f"label={step_log.get('L_label', 0):.3f} | "
                        f"con={step_log.get('L_contrastive', 0):.3f} | "
                        f"gen={step_log.get('L_generator', 0):.3f} | "
                        f"lang={step_log.get('L_language', 0):.3f} | "
                        f"meta={step_log['L_meta']:.3f} | "
                        f"lam={step_log['grl_lam']:.3f} | "
                        f"ETA {eta_str}"
                    )

            n_steps = len(self.train_loader)
            avg_log = {k: v / n_steps for k, v in epoch_logs.items()}
            self._log(avg_log, prefix="train/epoch", step=epoch)
            logger.info(
                "[Epoch %d] L_total=%.4f  L_label=%.4f  L_con=%.4f  "
                "L_gen=%.4f  L_lang=%.4f  L_meta=%.4f",
                epoch,
                avg_log.get("L_total", 0), avg_log.get("L_label", 0),
                avg_log.get("L_contrastive", 0), avg_log.get("L_generator", 0),
                avg_log.get("L_language", 0), avg_log.get("L_meta", 0),
            )

            val_metrics = self.evaluate()
            self._save_history(epoch, val_metrics)
            self._log(val_metrics, prefix="val/epoch", step=epoch)
            logger.info(
                "[Val   %d] acc=%.4f  f1=%.4f  roc_auc=%.4f",
                epoch,
                val_metrics["accuracy"], val_metrics["f1"],
                val_metrics.get("roc_auc", -1.0),
            )

            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self._save_checkpoint(epoch, val_metrics)

        self.writer.close()
        logger.info("Training complete. Best val F1 = %.4f", self.best_f1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for batch in self.val_loader:
            batch  = self._to(batch)
            out    = self.model(batch["input_ids"], batch["attention_mask"], batch["entropy"])
            logits = out["label_logits"]
            probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()
            labs   = batch["label"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)
            all_probs.extend(probs)

        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }
        try:
            # Handle both binary and multi-class AUC
            if len(np.unique(all_labels)) > 2:
                # Need one-hot probs for multi-class AUC
                # For now assume mostly binary code detection
                pass
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
        except Exception:
            metrics["roc_auc"] = 0.0

        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    def _log(self, metrics: Dict[str, float], prefix: str, step: Optional[int] = None) -> None:
        s = step if step is not None else self.global_step
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, s)
        if self._wandb is not None:
            self._wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=s)

    # ------------------------------------------------------------------
    def _save_history(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Append epoch metrics to CSV."""
        import pandas as pd
        lr = self.optimizer.param_groups[0]["lr"]
        # Use stored total_steps for consistent lam in history
        lam = self.get_current_lambda()
        
        row = {"epoch": epoch, "step": self.global_step, "lr": lr, "lam": lam}
        row.update(metrics)
        
        df = pd.DataFrame([row])
        # Reorder to keep epoch/step first
        cols = ["epoch", "step", "lr", "lam"]
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]
        
        header = not os.path.exists(self.history_path) or os.path.getsize(self.history_path) == 0
        df.to_csv(self.history_path, mode="a", index=False, header=header)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        path = os.path.join(self.checkpoint_dir, "best_model")
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch":     epoch,
                "step":      self.global_step,
                "metrics":   metrics,
                "best_f1":   self.best_f1,
            },
            os.path.join(path, "training_state.pt"),
        )
        logger.info(
            "Checkpoint saved → %s  (epoch=%d, f1=%.4f)",
            path, epoch + 1, metrics["f1"],
        )

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str) -> int:
        """Loads model and training state from directory. Returns start_epoch (1-indexed)."""
        model_path = os.path.join(path, "model.pt")
        state_path = os.path.join(path, "training_state.pt")
        
        if not os.path.exists(model_path) or not os.path.exists(state_path):
            logger.warning("Checkpoint %s not found. Training from scratch.", path)
            return 1
            
        logger.info("Resuming from checkpoint: %s", path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        state = torch.load(state_path, map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
            
        self.global_step = state.get("step", 0)
        self.best_f1     = state.get("best_f1", 0.0)
        start_epoch      = state.get("epoch", 0) + 1  # `epoch` in state is the one that just finished
        
        logger.info("Resumed at epoch %d, global_step %d, best_f1 %.4f", start_epoch, self.global_step, self.best_f1)
        return start_epoch
