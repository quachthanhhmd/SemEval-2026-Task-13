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

def _build_hierarchy_index(
    lang_ids: List[int], 
    label_ids: List[int], 
    gen_ids:  List[int]
) -> Dict[int, Dict[int, Dict[int, List[int]]]]:
    """Map lang_id → label → generator_id → list of sample indices."""
    # hierarchy[lang][label][gen] = [idx1, idx2, ...]
    hierarchy: Dict[int, Dict[int, Dict[int, List[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for idx, (lid, lbl, gid) in enumerate(zip(lang_ids, label_ids, gen_ids)):
        if lid >= 0:
            hierarchy[lid][lbl][gid].append(idx)
    return hierarchy


def _sample_diverse(gen_dict: Dict[int, List[int]], count: int, rng: random.Random) -> List[int]:
    """Sample 'count' indices from gen_dict {gen_id: [indices]} with diversity."""
    if not gen_dict: return []
    
    # 1. Round-robin draw from available generators
    gens = sorted(gen_dict.keys())
    rng.shuffle(gens)
    
    # Copy and shuffle per-generator pools
    pools = {g: list(indices) for g, indices in gen_dict.items()}
    for g in pools: rng.shuffle(pools[g])
    
    selected = []
    while len(selected) < count:
        found_any = False
        for g in gens:
            if pools[g]:
                selected.append(pools[g].pop())
                found_any = True
                if len(selected) == count: break
        if not found_any: break
    
    # 2. If pool was smaller than count (rare), sample with replacement
    if len(selected) < count:
        all_indices = []
        for indices in gen_dict.values(): all_indices.extend(indices)
        if all_indices:
            selected.extend(rng.choices(all_indices, k=count - len(selected)))
            
    return selected


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
        generator_ids: List[int],
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
        self._base_seed    = seed if seed is not None else random.randint(0, 10000)
        self._rng          = random.Random(self._base_seed)
        self._epoch        = 0

        # Build hierarchy: lang_id -> label -> generator_id -> [indices]
        self._hierarchy    = _build_hierarchy_index(language_ids, labels, generator_ids)
        self._valid_langs  = sorted(self._hierarchy.keys())
        self._dataset_size = len(language_ids)

        if len(self._valid_langs) < 2:
            raise ValueError(f"Dataset must have at least 2 distinct languages, found {len(self._valid_langs)}")

        logger.info(
            "LanguageBalancedSampler: %d valid languages | k=%d | m=%d | batch_size=%d",
            len(self._valid_langs), self.k, self.m, self.k * self.m,
        )

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._rng.seed(self._base_seed + epoch)

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self)
        
        for _ in range(num_batches):
            # 1. Sample k UNIQUE Languages (Tasks) 
            # This ensures no language leakage between meta-train and meta-test blocks.
            n_langs = len(self._valid_langs)
            if n_langs >= self.k:
                langs = self._rng.sample(self._valid_langs, self.k)
            else:
                langs = list(self._valid_langs)
                while len(langs) < self.k:
                    langs.append(self._rng.choice(self._valid_langs))

            batch: List[int] = []
            for lang in langs:
                # 2. Build Language Block (m/2 AI, m/2 Human)
                m_pos = self.m // 2
                m_neg = self.m - m_pos

                # Specific generators within this language for diversity
                group_1 = self._hierarchy[lang][1] # dict: gen -> [indices]
                group_0 = self._hierarchy[lang][0] # dict: gen -> [indices]

                indices_pos = _sample_diverse(group_1, m_pos, self._rng)
                indices_neg = _sample_diverse(group_0, m_neg, self._rng)
                
                block = indices_pos + indices_neg
                self._rng.shuffle(block)
                batch.extend(block)

            # CRITICAL: Keep language blocks contiguous for Trainer.split_meta_language
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
        self._base_seed     = seed if seed is not None else random.randint(0, 10000)
        self._rng           = random.Random(self._base_seed)
        self._epoch         = 0

        self._domain_to_indices = _build_group_index(domain_ids)
        self._valid_domains     = sorted(self._domain_to_indices.keys())
        self._dataset_size      = len(domain_ids)

        if len(self._valid_domains) < 2:
             raise ValueError(f"Dataset must have at least 2 distinct domains.")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._rng.seed(self._base_seed + epoch)

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self)
        for _ in range(num_batches):
            n_doms = len(self._valid_domains)
            if n_doms >= self.k:
                domains = self._rng.sample(self._valid_domains, self.k)
            else:
                domains = list(self._valid_domains)
                while len(domains) < self.k:
                    domains.append(self._rng.choice(self._valid_domains))
            
            batch: List[int] = []
            for d in domains:
                batch.extend(
                    _sample_group(self._domain_to_indices[d], self.m, self._rng)
                )
            self._rng.shuffle(batch)
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
