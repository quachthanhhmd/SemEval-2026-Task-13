"""
model.py
--------
GraphCodeBERTDomainModel — full OOD detection model.

Architecture:
    GraphCodeBERT encoder
    ├── shared_feature : Linear(768→512) + GELU + Dropout(0.1)
    ├── label_head     : Linear(512→2)               [CrossEntropy]
    ├── generator_head : GRL → Linear(512→N_gen)     [CrossEntropy adv]
    ├── language_head  : GRL → Linear(512→N_lang)    [CrossEntropy adv]
    └── projection_head: Linear(512→256) → GELU
                         → Linear(256→128) → L2-norm [SupCon]
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.losses import GradientReversalLayer

logger = logging.getLogger(__name__)


class GraphCodeBERTDomainModel(nn.Module):
    """
    Parameters
    ----------
    num_generators : int
        Number of unique code generators (e.g. GPT4, Codex, Human …).
    num_languages  : int
        Number of unique programming languages.
    model_name     : str
        HuggingFace model identifier.
    dropout        : float
        Dropout rate for shared feature layer.
    lam            : float
        Initial GRL lambda value (updated externally during training).
    """

    MODEL_NAME = "microsoft/graphcodebert-base"
    HIDDEN    = 768
    SHARED    = 512
    PROJ_MID  = 256
    PROJ_OUT  = 128

    def __init__(
        self,
        num_generators: int,
        num_languages:  int,
        num_domains:    int,
        model_name:     str  = MODEL_NAME,
        dropout:        float = 0.1,
        lam:            float = 1.0,
    ) -> None:
        super().__init__()

        logger.info("Loading backbone: %s", model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=cfg)

        # ----------------------------------------------------------------
        # Shared feature extractor
        # ----------------------------------------------------------------
        self.shared_feature = nn.Sequential(
            nn.Linear(self.HIDDEN, self.SHARED),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------------------
        # Heads: Use MLP for entropy fusion (1 -> 32 dims)
        # ----------------------------------------------------------------
        self.entropy_proj = nn.Linear(1, 32)
        # HEAD_IN is not shared anymore; entropy is label-head only.

        # ----------------------------------------------------------------
        # Decouple Task vs Domain representations (Orthogonal Decomposition)
        # ----------------------------------------------------------------
        self.domain_adapter = nn.Sequential(
            nn.Linear(self.SHARED, self.SHARED),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 1: Label classification (AI vs Human) - Uses task_feat + entropy
        self.label_head = nn.Linear(self.SHARED + 32, 2)

        # Head 2: Generator adv head (with GRL) - Uses domain_feat
        self.grl_generator = GradientReversalLayer(lam=lam)
        self.generator_head = nn.Linear(self.SHARED, num_generators)

        # Head 3: Language adv head (with GRL) - Uses domain_feat
        self.grl_language = GradientReversalLayer(lam=lam)
        self.language_head = nn.Linear(self.SHARED, num_languages)

        # Head 4: Comprehensive Domain adv head (with GRL) - Uses domain_feat
        self.grl_domain = GradientReversalLayer(lam=lam)
        self.domain_head = nn.Linear(self.SHARED, num_domains)

        # Head 5: Projection head for SupCon - Uses task_feat
        self.projection_head = nn.Sequential(
            nn.Linear(self.SHARED, self.PROJ_MID),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout to prevent representation overfit
            nn.Linear(self.PROJ_MID, self.PROJ_OUT),
        )

        self._init_heads()
        logger.info(
            "GraphCodeBERTDomainModel ready | generators=%d | languages=%d | domains=%d | entropy=True",
            num_generators, num_languages, num_domains
        )

    # ------------------------------------------------------------------
    def _init_heads(self) -> None:
        for module in [
            self.label_head,
            self.generator_head,
            self.language_head,
            self.domain_head,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for module_seq in [self.shared_feature, self.domain_adapter]:
            for m in module_seq.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        nn.init.xavier_uniform_(self.entropy_proj.weight)
        nn.init.zeros_(self.entropy_proj.bias)

    # ------------------------------------------------------------------
    def set_lambda(self, lam: float) -> None:
        """Update GRL lambda (called by trainer according to schedule)."""
        self.grl_generator.set_lambda(lam)
        self.grl_language.set_lambda(lam)
        self.grl_domain.set_lambda(lam)

    # ------------------------------------------------------------------
    def encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return Mean-Pooled embedding [B, 768] with Token Distribution Normalization (TDN)."""
        # 1. Get word/position/token_type embeddings from RoBERTa backbone
        # We intercept here to implement TDN (Token Distribution Normalization)
        # This removes the sequence-level mean to eliminate language-specific token bias.
        inputs_embeds = self.encoder.embeddings(input_ids) # [B, L, 768]
        
        # 2. Apply TDN: subtract mean per feature dimension (protects sequence position)
        mean = inputs_embeds.mean(dim=-1, keepdim=True)
        inputs_embeds = inputs_embeds - mean
        
        # 3. Pass normalized embeddings to transformer encoder
        # We use inputs_embeds instead of input_ids
        out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state # [B, L, 768]
        
        # Expand mask to [B, L, 768]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        
        # Masked sum (pooling)
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        entropy:        torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
            label_logits          : [B, 2]
            generator_logits      : [B, num_generators]
            language_logits       : [B, num_languages]
            projection            : [B, 128]  (L2-normalised)
            features_with_entropy : [B, 512+32]
        """
        # 1. Base features via Mean Pooling
        pooled_feat = self.encode(input_ids, attention_mask) # [B, 768]
        feat = self.shared_feature(pooled_feat)               # [B, 512]
        
        # 2. Orthogonal Domain Decomposition (Pure Code Semantics)
        # We explicitly subtract the domain_feat from the shared feat.
        # Since domain_feat is regularized by GRL to contain domain info,
        # task_feat is forced to represent everything EXCEPT domain info.
        domain_feat = self.domain_adapter(feat)   # [B, 512]
        task_feat   = feat - domain_feat          # [B, 512]

        # 3. Label-Specific Fusion: Fuse entropy ONLY for the final label head
        # This prevents the model from using entropy as a shortcut for domain headers or clustering
        entropy_embed = torch.tanh(self.entropy_proj(entropy.unsqueeze(1))) # [B, 32]
        label_input = torch.cat([task_feat, entropy_embed], dim=1)           # [B, 512+32]
        label_logits = self.label_head(label_input)                          # [B, 2]

        # GRADIENT FIX: Remove .detach() so encoder learns from SupCon
        proj            = self.projection_head(task_feat)    # [B, 128]
        projection      = F.normalize(proj, dim=1)            # L2-norm

        # Domain/Adversarial Heads
        gen_feat         = self.grl_generator(domain_feat)
        generator_logits = self.generator_head(gen_feat)     # [B, N_gen]

        lang_feat        = self.grl_language(domain_feat)
        language_logits  = self.language_head(lang_feat)     # [B, N_lang]
        
        dom_feat         = self.grl_domain(domain_feat)
        domain_logits    = self.domain_head(dom_feat)        # [B, N_domain]

        return {
            "label_logits":     label_logits,
            "generator_logits": generator_logits,
            "language_logits":  language_logits,
            "domain_logits":    domain_logits,
            "projection":       projection,
            "features_with_entropy": label_input,
        }

    # ------------------------------------------------------------------
    def forward_with_params(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        entropy:        torch.Tensor,
        params:         Dict[str, torch.Tensor],
        buffers:        Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Functional forward using temporary parameter dict θ' and buffers.
        """
        try:
            from torch.func import functional_call
        except ImportError:
            from torch._functorch.eager_transforms import functional_call

        # combine params and buffers into a single state dict for functional_call
        state = params.copy()
        if buffers is not None:
            state.update(buffers)
            
        out = functional_call(self, state, (input_ids, attention_mask, entropy))
        return out
