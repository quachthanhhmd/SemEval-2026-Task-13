"""
losses.py
---------
Custom loss functions for the OOD meta-training pipeline:

  1. SupConLoss  – Supervised Contrastive Loss (temperature=0.07)
  2. GRL         – Gradient Reversal Layer  (used inside model.py)
  3. combined_loss – computes L_total = L_label + α·L_con + β·L_gen + γ·L_lang
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# 1. Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradientReversalFunction(Function):
    """Custom autograd Function that negates gradients with scale λ."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (lam,) = ctx.saved_tensors
        return grad_output.neg() * lam, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).

    forward : identity
    backward: multiply gradient by −λ

    λ can be updated dynamically via ``set_lambda``.
    """

    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def set_lambda(self, lam: float) -> None:
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lam)  # type: ignore[no-untyped-call]

    def extra_repr(self) -> str:
        return f"λ={self.lam:.4f}"


# ---------------------------------------------------------------------------
# 2. Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Reference:
        Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020.

    Parameters
    ----------
    temperature : float
        Cosine similarity temperature τ.  Default: 0.07.
    contrast_mode : str
        'all'  – every sample as anchor.
        'one'  – only the first view as anchor.
    base_temperature : float
        Base temperature for scaling.

    Inputs
    ------
    features : [B, D]   (L2-normalised projection-head output)
    labels   : [B]      (class labels 0/1)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature      = temperature
        self.contrast_mode    = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        language_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features     : [B, D]   already L2-normalised
        labels       : [B]      long tensor of class ids
        language_ids : [B]      long tensor of language ids (optional)
        """
        device = features.device
        B = features.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure contiguous & re-normalised (defensive)
        features = F.normalize(features, dim=1)

        # Similarity matrix [B, B]
        sim_matrix = torch.matmul(features, features.T)
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0) # Numerical stability
        sim_matrix = sim_matrix / self.temperature

        # Mask of positive pairs
        labels_col = labels.unsqueeze(1)  # [B, 1]
        labels_row = labels.unsqueeze(0)  # [1, B]
        pos_mask = (labels_col == labels_row).float()

        # Remove diagonal (self similarity)
        diag_mask = torch.eye(B, device=device)
        pos_mask  = pos_mask * (1 - diag_mask)  # positive pairs excl. self

        # For numerical stability: subtract row max for softmax logic
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: all pairs except self
        exp_logits = torch.exp(logits) * (1 - diag_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        # Fix SupCon bias by normalizing per-sample according to pos_count
        pos_count = pos_mask.sum(dim=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-9)
        loss = loss.mean()
        
        return loss


# ---------------------------------------------------------------------------
# 3. Combined loss helper
# ---------------------------------------------------------------------------

def compute_total_loss(
    label_logits:     torch.Tensor,
    generator_logits: torch.Tensor,
    language_logits:  torch.Tensor,
    domain_logits:    torch.Tensor,
    projection:       torch.Tensor,
    labels:           torch.Tensor,
    generator_ids:    torch.Tensor,
    language_ids:     torch.Tensor,
    domain_ids:       torch.Tensor,
    supcon_fn:        SupConLoss,
    is_mixed:         Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    beta:  float = 0.1,
    gamma: float = 0.1,
    eta:   float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute:
        L_total = L_label
                + α * L_contrastive
                + β * L_generator_adv
                + γ * L_language_adv
    Returns
    -------
    (total_loss, breakdown_dict)
    """
    ce = nn.CrossEntropyLoss()

    L_label = ce(label_logits, labels)

    # 2. Contrastive Loss (AI vs Human Invariant)
    L_con = supcon_fn(projection, labels)

    # Adversarial losses (GRL already reversed the gradient inside the model)
    # We use nn.CrossEntropyLoss(reduction='none') to selectively mask mixed samples
    ce_none = nn.CrossEntropyLoss(reduction='none')
    
    if generator_ids.min() >= 0:
        L_gen_all = ce_none(generator_logits, generator_ids)
        if is_mixed is not None:
             L_gen_all = L_gen_all * (1.0 - is_mixed)
        L_gen = L_gen_all.mean()
    else:
        L_gen = torch.tensor(0.0, device=labels.device)
        
    if language_ids.min() >= 0:
        L_lang_all = ce_none(language_logits, language_ids)
        if is_mixed is not None:
             L_lang_all = L_lang_all * (1.0 - is_mixed)
        L_lang = L_lang_all.mean()
    else:
        L_lang = torch.tensor(0.0, device=labels.device)
        
    if domain_ids.min() >= 0:
        # We do NOT mask L_domain since the domain ID actively reflects if a sample is mixed
        L_domain = ce_none(domain_logits, domain_ids).mean()
    else:
        L_domain = torch.tensor(0.0, device=labels.device)

    # Decouple the adversarial loss magnitude from the scale, while keeping the gradient.
    # The trick: add `(adv_loss - adv_loss.detach())`, which equals 0 value but passes gradients.
    adv_loss = beta * L_gen + gamma * L_lang + eta * L_domain
    total = L_label + alpha * L_con + (adv_loss - adv_loss.detach())

    # For logging, we report the conceptual loss values
    breakdown = {
        "L_label":       L_label,
        "L_contrastive": L_con,
        "L_generator":   L_gen,
        "L_language":    L_lang,
        "L_domain":      L_domain,
        "L_total":       L_label + alpha * L_con + beta * L_gen.detach() + gamma * L_lang.detach() + eta * L_domain.detach(),
    }
    return total, breakdown
