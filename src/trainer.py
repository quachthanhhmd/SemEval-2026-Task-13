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

from __future__ import annotations

import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

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

    test_mask  = torch.tensor([l == meta_test_lang for l in lang_ids])
    train_mask = ~test_mask

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None, None, -1

    def _select(m: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: v[m] for k, v in batch.items()}

    return _select(train_mask), _select(test_mask), meta_test_lang


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

        self.scaler    = GradScaler(enabled=self.fp16)
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
        self.best_f1     = 0.0

    # ------------------------------------------------------------------
    def _to(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------
    def _forward(
        self,
        batch:  Dict[str, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass — uses functional_call if params (θ') are given."""
        if params is not None:
            return self.model.forward_with_params(
                batch["input_ids"], batch["attention_mask"], batch["entropy"], params
            )
        return self.model(batch["input_ids"], batch["attention_mask"], batch["entropy"])

    # ------------------------------------------------------------------
    def _compute_train_loss(
        self,
        batch:  Dict[str, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full L_train = L_label + α·L_con + β·L_gen + γ·L_lang."""
        out = self._forward(batch, params)
        total, breakdown = compute_total_loss(
            label_logits     = out["label_logits"],
            generator_logits = out["generator_logits"],
            language_logits  = out["language_logits"],
            projection       = out["projection"],
            labels           = batch["label"],
            generator_ids    = batch["generator_id"],
            language_ids     = batch["language_id"],
            supcon_fn        = self.supcon_fn,
            alpha            = self.alpha,
            beta             = self.beta,
            gamma            = self.gamma,
        )
        return total, breakdown

    # ------------------------------------------------------------------
    def _compute_meta_loss(
        self,
        batch:  Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        L_meta = L_label + alpha_meta * L_contrastive   (fix #2)
        No adversarial terms in meta-test step.
        """
        out = self._forward(batch, params)
        L_label = F.cross_entropy(out["label_logits"], batch["label"])
        # Contrastive on the projection using label as positive signal
        L_con   = self.supcon_fn(out["projection"], batch["label"])
        return L_label + self.alpha_meta * L_con

    # ------------------------------------------------------------------
    def _train_step(
        self,
        batch:       Dict[str, torch.Tensor],
        total_steps: int,
    ) -> Dict[str, float]:
        """One FOMAML meta-training step."""

        # ---- GRL λ (fix #6: scale=5.0) ----
        lam = grl_lambda_schedule(self.global_step, total_steps, self.grl_scale)
        self.model.set_lambda(lam)

        # ---- Leave-One-Language split (fix #1 & #7) ----
        meta_train_batch, meta_test_batch, _ = split_meta_language(batch)
        use_meta = (meta_train_batch is not None)

        self.optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------------------
        # Case A: meta split succeeded — FOMAML inner/outer loop
        # ----------------------------------------------------------------
        if use_meta:
            meta_train_batch = self._to(meta_train_batch)
            meta_test_batch  = self._to(meta_test_batch)

            # ---- Inner step ----
            # Freeze encoder in FOMAML inner loop for meta stability
            param_names  = [
                n for n, p in self.model.named_parameters() 
                if p.requires_grad and not n.startswith("encoder")
            ]
            named_params = {
                n: p for n, p in self.model.named_parameters() 
                if p.requires_grad and not n.startswith("encoder")
            }

            with autocast(device_type=self.device.type, enabled=self.fp16):
                L_train, breakdown = self._compute_train_loss(meta_train_batch)

            # FOMAML: create_graph=False  (fix #3)
            grads = torch.autograd.grad(
                L_train,
                [named_params[n] for n in param_names],
                create_graph=False,   # << FOMAML key change
                allow_unused=True,
            )
            grads = tuple(
                g if g is not None else torch.zeros_like(named_params[n])
                for g, n in zip(grads, param_names)
            )

            theta_prime = functional_params_update(
                named_params, grads, param_names, self.lr_inner
            )

            # ---- Meta step: L_label + 0.3 * L_contrastive (fix #2) ----
            with autocast(device_type=self.device.type, enabled=self.fp16):
                L_meta = self._compute_meta_loss(meta_test_batch, theta_prime)

            L_final = L_train + self.delta * L_meta

        # ----------------------------------------------------------------
        # Case B: only one language in batch — standard training step
        # ----------------------------------------------------------------
        else:
            batch = self._to(batch)
            with autocast(device_type=self.device.type, enabled=self.fp16):
                L_final, breakdown = self._compute_train_loss(batch)
            L_meta = torch.tensor(0.0, device=self.device)

        # ---- Backward & update ----
        self.scaler.scale(L_final).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        log = {k: v.item() for k, v in breakdown.items()}
        log["L_meta"]  = L_meta.item()
        log["L_final"] = L_final.item()
        log["grl_lam"] = lam
        return log

    # ------------------------------------------------------------------
    def train(self, num_epochs: int, start_epoch: int = 1) -> None:
        # ==============================================================
        self.model.train()
        epoch_logs: Dict[str, float] = defaultdict(float) # Changed from epoch_losses to epoch_logs and defaultdict
        total_steps = len(self.train_loader) * num_epochs # Using num_epochs as parameter

        # For debugging: we want to inspect the first 3 batches
        saved_debug_batches = 0

        for epoch in range(start_epoch, num_epochs + 1): 
            logger.info("Epoch %d/%d", epoch, num_epochs)
            if hasattr(self.train_loader.batch_sampler, "set_epoch"): # Changed to batch_sampler
                self.train_loader.batch_sampler.set_epoch(epoch)

            pbar = tqdm(self.train_loader, desc=f"Train E{epoch}") # Different desc
            for step, batch in enumerate(pbar): # Added enumerate and step
                # --- Debug Batch Saving ---
                if epoch == 1 and saved_debug_batches < 3:
                    try:
                        import pandas as pd
                        # Tokenizer is needed to decode, but trainer itself doesn't hold it.
                        # We just save the raw tensors/ids to a dict and then DF.
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

                step_log = self._train_step(batch, total_steps) # Corrected variable name and removed trailing 'r,'
                self.global_step += 1

                for k, v in step_log.items():
                    epoch_logs[k] += v

                if self.global_step % self.log_every == 0:
                    self._log(step_log, prefix="train/step")

                pbar.set_postfix({
                    "L_total": f"{step_log['L_total']:.3f}",
                    "L_label": f"{step_log['L_label']:.3f}",
                    "L_meta":  f"{step_log['L_meta']:.3f}",
                    "λ":       f"{step_log['grl_lam']:.3f}",
                })

            n_steps = len(self.train_loader)
            avg_log = {k: v / n_steps for k, v in epoch_logs.items()}
            self._log(avg_log, prefix="train/epoch", step=epoch)
            logger.info(
                "[Epoch %d] L_total=%.4f  L_label=%.4f  L_con=%.4f  "
                "L_gen=%.4f  L_lang=%.4f  L_meta=%.4f",
                epoch + 1,
                avg_log.get("L_total", 0), avg_log.get("L_label", 0),
                avg_log.get("L_contrastive", 0), avg_log.get("L_generator", 0),
                avg_log.get("L_language", 0), avg_log.get("L_meta", 0),
            )

            val_metrics = self.evaluate()
            self._log(val_metrics, prefix="val/epoch", step=epoch)
            logger.info(
                "[Val   %d] acc=%.4f  f1=%.4f  roc_auc=%.4f",
                epoch + 1,
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

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False, dynamic_ncols=True):
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
            "f1":       f1_score(all_labels, all_preds, average="binary", zero_division=0),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
        except ValueError:
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
