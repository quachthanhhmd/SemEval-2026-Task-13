"""
sampler.py
----------
Two samplers for domain-balanced meta-learning:

  1. LanguageBalancedSampler — k languages × m samples/language.
     (With Label Balance: m/2 AI and m/2 Human per language).
     Uses sampling with replacement to ensure infinite batches per epoch,
     meaning it always works even if num_languages < k.

  2. DomainBalancedSampler   — fallback/legacy

batch_size = k * m  (default: 6 * 16 = 96)
"""

from __future__ import annotations

import random
import logging
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_group_index(ids: List[int]) -> Dict[int, List[int]]:
    """Map group_id → list of sample indices."""
    group: Dict[int, List[int]] = defaultdict(list)
    for sample_idx, gid in enumerate(ids):
        if gid >= 0:
            group[gid].append(sample_idx)
    return dict(group)


def _build_group_label_index(group_ids: List[int], labels: List[int]) -> Dict[int, Dict[int, List[int]]]:
    """Map group_id → label → list of sample indices."""
    group: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: {0: [], 1: []})
    for sample_idx, (gid, lbl) in enumerate(zip(group_ids, labels)):
        if gid >= 0:
            group[gid][lbl].append(sample_idx)
    return dict(group)


def _sample_group(pool: List[int], m: int, rng: random.Random) -> List[int]:
    """Draw m indices from pool; sample with replacement if pool < m."""
    if len(pool) == 0:
        return []
    if len(pool) >= m:
        return rng.sample(pool, m)
    return rng.choices(pool, k=m)


# ---------------------------------------------------------------------------
# 1. LanguageBalancedSampler
# ---------------------------------------------------------------------------

class LanguageBalancedSampler(Sampler):
    """
    Yields batches with EXACTLY k languages, m samples per language.
    Guarantees label balance (m//2 positive, m - m//2 negative) per language.
    
    Draws 'k' languages with replacement from valid languages, guaranteeing
    it always works even if dataset has fewer than k languages total.
    """

    def __init__(
        self,
        language_ids:  List[int],
        labels:        List[int],
        k:             int  = 6,
        m:             int  = 16,
        shuffle_langs: bool = True,
        drop_last:     bool = True,
        seed:          Optional[int] = None,
    ) -> None:
        if k < 2:
            raise ValueError("k must be >= 2 (need at least 1 meta_train + 1 meta_test language)")

        self.k             = k
        self.m             = m
        self.drop_last     = drop_last
        self.shuffle_langs = shuffle_langs
        self._rng          = random.Random(seed)
        self._epoch        = 0

        self._lang_label_idx = _build_group_label_index(language_ids, labels)
        self._valid_langs    = sorted(self._lang_label_idx.keys())
        self._dataset_size   = len(language_ids)

        if len(self._valid_langs) < 2:
            raise ValueError(f"Dataset must have at least 2 distinct languages, found {len(self._valid_langs)}")

        logger.info(
            "LanguageBalancedSampler: %d valid languages | k=%d | m=%d | batch_size=%d",
            len(self._valid_langs), self.k, self.m, self.k * self.m,
        )

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self)
        
        for _ in range(num_batches):
            # Sample k languages with replacement so it never fails even if len(valid_langs) < k
            # But we want to ensure the batch has *distinct* languages if possible, 
            # so we sample without replacement if we have enough, otherwise with replacement.
            if len(self._valid_langs) >= self.k:
                langs = self._rng.sample(self._valid_langs, self.k)
            else:
                langs = self._rng.choices(self._valid_langs, k=self.k)

            batch: List[int] = []
            for lang in langs:
                pool_0 = self._lang_label_idx[lang][0]
                pool_1 = self._lang_label_idx[lang][1]

                m_0 = self.m // 2
                m_1 = self.m - m_0

                # If a language is missing one of the labels entirely, 
                # fallback to taking all m samples from the available label
                if len(pool_0) == 0 or len(pool_1) == 0:
                    combined = pool_0 + pool_1
                    batch.extend(_sample_group(combined, self.m, self._rng))
                else:
                    batch.extend(_sample_group(pool_0, m_0, self._rng))
                    batch.extend(_sample_group(pool_1, m_1, self._rng))

            # Preserve per-language grouping in the output list
            yield batch

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        batch_size = self.k * self.m
        n = self._dataset_size // batch_size
        if not self.drop_last and self._dataset_size % batch_size != 0:
            n += 1
        # Guarantee at least 1 round if dataset is very small
        return max(1, n)


# ---------------------------------------------------------------------------
# 2. DomainBalancedSampler  (legacy / fallback)
# ---------------------------------------------------------------------------

class DomainBalancedSampler(Sampler):
    """
    Yields batches where every batch contains exactly k domains,
    each represented by m samples.
    """

    def __init__(
        self,
        domain_ids:     List[int],
        k:              int  = 8,
        m:              int  = 12,
        shuffle_domains: bool = True,
        drop_last:      bool = True,
        seed:           Optional[int] = None,
    ) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")

        self.k              = k
        self.m              = m
        self.drop_last      = drop_last
        self.shuffle_domains = shuffle_domains
        self._rng           = random.Random(seed)
        self._epoch         = 0

        self._domain_to_indices = _build_group_index(domain_ids)
        self._valid_domains     = sorted(self._domain_to_indices.keys())
        self._dataset_size      = len(domain_ids)

        if len(self._valid_domains) < 2:
             raise ValueError(f"Dataset must have at least 2 distinct domains.")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self)
        for _ in range(num_batches):
            if len(self._valid_domains) >= self.k:
                domains = self._rng.sample(self._valid_domains, self.k)
            else:
                domains = self._rng.choices(self._valid_domains, k=self.k)
            
            batch: List[int] = []
            for d in domains:
                batch.extend(
                    _sample_group(self._domain_to_indices[d], self.m, self._rng)
                )
            yield batch

    def __len__(self) -> int:
        batch_size = self.k * self.m
        n = self._dataset_size // batch_size
        if not self.drop_last and self._dataset_size % batch_size != 0:
            n += 1
        return max(1, n)


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def domain_collate_fn(samples: list) -> Dict:
    """Stack per-sample dicts into a batched dict of tensors."""
    batch: Dict = {}
    for key in samples[0].keys():
        vals = [s[key] for s in samples]
        batch[key] = torch.stack(vals)
    return batch
