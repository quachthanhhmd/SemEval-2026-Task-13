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


class FeatureGatingNetwork(nn.Module):
    """
    Processa le features stilometriche (Agnostic Features) e crea un embedding
    che verrà usato per modulare il segnale semantico.
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return self.net(x)


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
    STYLE     = 128
    PROJ_MID  = 256
    PROJ_OUT  = 128

    def __init__(
        self,
        num_generators: int,
        num_languages:  int,
        num_style:      int   = 10,
        model_name:     str   = MODEL_NAME,
        dropout:        float = 0.1,
        lam:            float = 1.0,
        gradient_checkpointing: bool = False,
        freeze_layers:   int = 0,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        # 1. Text Backbone
        # ----------------------------------------------------------------
        logger.info("Loading backbone: %s", model_name)
        cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=cfg)

        # ----------------------------------------------------------------
        # 2. Shared feature extractor (Semantic)
        # ----------------------------------------------------------------
        self.shared_feature = nn.Sequential(
            nn.Linear(self.HIDDEN, self.SHARED),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------------------
        # 3. Stylometric Feature Encoder
        # ----------------------------------------------------------------
        self.style_encoder = FeatureGatingNetwork(
            input_dim=num_style,
            output_dim=self.STYLE,
            dropout_rate=dropout
        )

        # ----------------------------------------------------------------
        # 4. Fusion & Adapters
        # ----------------------------------------------------------------
        FUSION_IN = self.SHARED + self.STYLE

        # ----------------------------------------------------------------
        # Decouple Task vs Domain representations (fix gradient conflict)
        # ----------------------------------------------------------------
        self.task_adapter = nn.Sequential(
            nn.Linear(FUSION_IN, self.SHARED),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.domain_adapter = nn.Sequential(
            nn.Linear(FUSION_IN, self.SHARED),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 1: Label classification (AI vs Human) - Uses task_feat
        self.label_head = nn.Linear(self.SHARED, 2)

        # Head 2: Generator adv head (with GRL) - Uses domain_feat
        self.grl_generator = GradientReversalLayer(lam=lam)
        self.generator_head = nn.Linear(self.SHARED, num_generators)

        # Head 3: Language adv head (with GRL) - Uses domain_feat
        self.grl_language = GradientReversalLayer(lam=lam)
        self.language_head = nn.Linear(self.SHARED, num_languages)

        # Head 4: Projection head for SupCon - Uses task_feat
        self.projection_head = nn.Sequential(
            nn.Linear(self.SHARED, self.PROJ_MID),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout to prevent representation overfit
            nn.Linear(self.PROJ_MID, self.PROJ_OUT),
        )

        if gradient_checkpointing:
            logger.info("Enabling gradient checkpointing...")
            self.encoder.gradient_checkpointing_enable()

        if freeze_layers > 0:
            logger.info("Freezing first %d layers of the encoder...", freeze_layers)
            # GraphCodeBERT (Roberta) has embeddings at index 0, then layers in encoder.layer
            # Freeze embeddings
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            # Freeze layers
            for i in range(min(freeze_layers, 12)):
                for p in self.encoder.encoder.layer[i].parameters():
                    p.requires_grad = False

        self._init_heads()
        logger.info(
            "GraphCodeBERTDomainModel ready | generators=%d | languages=%d | style=%d",
            num_generators, num_languages, num_style
        )

    # ------------------------------------------------------------------
    def _init_heads(self) -> None:
        for module in [
            self.label_head,
            self.generator_head,
            self.language_head,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        for module_seq in [self.shared_feature, self.task_adapter, self.domain_adapter, self.style_encoder]:
            for m in module_seq.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def set_lambda(self, lam: float) -> None:
        """Update GRL lambda (called by trainer according to schedule)."""
        self.grl_generator.set_lambda(lam)
        self.grl_language.set_lambda(lam)

    # ------------------------------------------------------------------
    def encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return CLS-token embedding [B, 768]."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS token

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        extra_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
            label_logits     : [B, 2]
            generator_logits : [B, num_generators]
            language_logits  : [B, num_languages]
            projection       : [B, 128]  (L2-normalised)
            shared_features  : [B, FUSION_IN]
        """
        cls = self.encode(input_ids, attention_mask)         # [B, 768]
        sem_feat = self.shared_feature(cls)                   # [B, 512]
        
        # Stylometric encoding
        style_feat = self.style_encoder(extra_features)      # [B, 128]

        # Fusion
        feat = torch.cat([sem_feat, style_feat], dim=1)      # [B, 640]

        # Decouple features
        task_feat   = self.task_adapter(feat)     # [B, 512]
        domain_feat = self.domain_adapter(feat)   # [B, 512]

        # Task Heads
        label_logits    = self.label_head(task_feat)         # [B, 2]
        proj            = self.projection_head(task_feat.detach()) # [B, 128]
        projection      = F.normalize(proj, dim=1)           # L2-norm

        # Domain/Adversarial Heads
        gen_feat         = self.grl_generator(domain_feat)
        generator_logits = self.generator_head(gen_feat)     # [B, N_gen]

        lang_feat        = self.grl_language(domain_feat)
        language_logits  = self.language_head(lang_feat)     # [B, N_lang]

        return {
            "label_logits":     label_logits,
            "generator_logits": generator_logits,
            "language_logits":  language_logits,
            "projection":       projection,
            "shared_features":  feat,
        }

    # ------------------------------------------------------------------
    def forward_with_params(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        extra_features: torch.Tensor,
        params:         Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Functional forward using temporary parameter dict θ'.
        """
        try:
            from torch.func import functional_call
        except ImportError:
            from torch._functorch.eager_transforms import functional_call

        out = functional_call(self, params, (input_ids, attention_mask, extra_features))
        return out
