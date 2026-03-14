"""
dataset.py
----------
Enhanced Dataset for AI-generated code detection with domain-aware labeling,
advanced source code augmentations, and backend-optimized entropy processing.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and Regex
# ---------------------------------------------------------------------------

IDENTIFIER_REGEX = r"[A-Za-z_][A-Za-z0-9_]*"

HEADER_PATTERNS = [
    re.compile(r"^\s*import\s+"),
    re.compile(r"^\s*from\s+.*\s+import\s+"),
    re.compile(r"^\s*#include\s+"),
    re.compile(r"^\s*package\s+"),
    re.compile(r"^\s*using\s+"),
    re.compile(r"^\s*require\s*\("),
]

LANG_PATTERNS = [
    re.compile(r"#include\s*<.*?>"),
    re.compile(r"public\s+class"),
    re.compile(r"System\.out\.println"),
    re.compile(r"console\.log"),
    re.compile(r"\bpackage\s+\w+"),
    re.compile(r"\busing\s+\w+"),
    re.compile(r"<\?php"),
]

PYTHON_KW = {
    "False","await","else","import","pass","None","break","except","in","raise",
    "True","class","finally","is","return","and","continue","for","lambda","try",
    "as","def","from","nonlocal","while","assert","del","global","not","with",
    "async","elif","if","or","yield"
}

C_LIKE_KW = {
    "int","float","double","char","void","long","short","unsigned","signed",
    "if","else","for","while","do","switch","case","default","break","continue",
    "return","struct","class","public","private","protected","static","const",
    "new","delete","this","namespace","using","try","catch","throw",
    "import","package","interface","extends","implements","final"
}

GO_KW = {
    "break","default","func","interface","select",
    "case","defer","go","map","struct",
    "chan","else","goto","package","switch",
    "const","fallthrough","if","range","type",
    "continue","for","import","return","var"
}

PHP_KW = {
    "echo","print","array","function","class","public",
    "private","protected","static","var","new","if",
    "else","elseif","while","for","foreach","return"
}

KEYWORDS = PYTHON_KW | C_LIKE_KW | GO_KW | PHP_KW

DOUBLE_COLON_PATTERN = re.compile(r"::")
ARROW_PATTERN = re.compile(r"->")
PHP_TAG_PATTERN = re.compile(r"<\?php|\?>")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def extract_identifiers(code: str) -> List[str]:
    ids = re.findall(IDENTIFIER_REGEX, code)
    return [i for i in ids if i not in KEYWORDS]

def identifier_entropy(code: str) -> float:
    ids = extract_identifiers(str(code))
    if len(ids) == 0:
        return 0.0
    counter = Counter(ids)
    total = len(ids)
    entropy = 0.0
    for c in counter.values():
        p = c / total
        entropy -= p * math.log2(p)
    entropy = entropy / math.log2(len(counter) + 1)
    return float(entropy)

# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def mask_identifiers(code: str, p: float = 0.3) -> str:
    identifiers = list(set(extract_identifiers(code)))
    identifiers = sorted(identifiers, key=len, reverse=True)
    for name in identifiers:
        if random.random() < p:
            pattern = rf"\b{re.escape(name)}\b"
            code = re.sub(pattern, "VAR", code)
    return code

def shuffle_identifiers(code: str, p: float = 0.3) -> str:
    ids = list(set(extract_identifiers(code)))
    random.shuffle(ids)
    mapping = {k: f"VAR{i}" for i, k in enumerate(ids)}
    for k in sorted(mapping.keys(), key=len, reverse=True):
        if random.random() < p:
            pattern = rf"\b{re.escape(k)}\b"
            code = re.sub(pattern, mapping[k], code)
    return code

def mask_comments(code: str, p: float = 0.5) -> str:
    if random.random() < p:
        code = re.sub(r"#.*", "# COMMENT", code)
        code = re.sub(r"//.*", "// COMMENT", code)
        code = re.sub(r"/\*.*?\*/", "/* COMMENT */", code, flags=re.DOTALL)
    return code

def remove_headers(code: str, p: float = 0.3) -> str:
    if random.random() >= p:
        return code
    lines = code.split("\n")
    new_lines = []
    for line in lines:
        if any(pattern.match(line) for pattern in HEADER_PATTERNS):
            continue
        new_lines.append(line)
    return "\n".join(new_lines)

def language_dropout(code: str, p: float = 0.3) -> str:
    if random.random() >= p:
        return code
    for pat in LANG_PATTERNS:
        code = pat.sub("", code)
    return code

def normalize_indent(code: str) -> str:
    lines = code.split("\n")
    lines = [line.lstrip() for line in lines]
    return "\n".join(lines)

def reduce_language_specific_tokens(code: str, p: float = 0.5) -> str:
    if random.random() >= p:
        return code
    code = DOUBLE_COLON_PATTERN.sub(" ", code)
    code = ARROW_PATTERN.sub(" ", code)
    code = PHP_TAG_PATTERN.sub(" ", code)
    code = re.sub(r"\s+", " ", code)
    return code.strip()

# ---------------------------------------------------------------------------
# Spanning Logic
# ---------------------------------------------------------------------------

def random_span(ids: List[int], max_len: int) -> List[int]:
    target_max = max_len - 2
    target_len = random.randint(int(target_max * 0.7), target_max)
    if len(ids) > target_len:
        start = random.randint(0, len(ids) - target_len)
        ids = ids[start:start + target_len]
    return ids

def multi_span(ids: List[int], max_len: int) -> List[int]:
    target_max = max_len - 2
    span = target_max // 2
    if len(ids) <= target_max:
        return ids[:target_max]
    start1 = random.randint(0, len(ids) - span)
    start2 = random.randint(0, len(ids) - span)
    span1 = ids[start1:start1 + span]
    span2 = ids[start2:start2 + span]
    return (span1 + span2)[:target_max]

def char_span(code: str, max_chars: int = 2000) -> str:
    if len(code) <= max_chars:
        return code
    lines = code.split("\n")
    if not lines:
        return code
    start_line_idx = random.randint(0, len(lines) - 1)
    out = []
    total_len = 0
    for line in lines[start_line_idx:]:
        if total_len + len(line) + 1 > max_chars and out:
            break
        out.append(line)
        total_len += len(line) + 1
    if not out:
        return code[:max_chars]
    return "\n".join(out)

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

class DomainRegistry:
    def __init__(self) -> None:
        self.language2id: Dict[str, int] = {}
        self.generator2id: Dict[str, int] = {}
        self.domain2id: Dict[Tuple[str, str], int] = {}

    def fit(self, languages: pd.Series, generators: pd.Series) -> "DomainRegistry":
        unique_langs = sorted(languages.dropna().unique().tolist())
        unique_gens  = sorted(generators.dropna().unique().tolist())
        self.language2id  = {lang: i for i, lang in enumerate(unique_langs)}
        self.generator2id = {gen:  i for i, gen  in enumerate(unique_gens)}
        idx = 0
        for lang in unique_langs:
            for gen in unique_gens:
                self.domain2id[(lang, gen)] = idx
                idx += 1
        return self

    def get_domain_id(self, language: str, generator: str) -> int:
        return self.domain2id.get((language, generator), -1)

    @property
    def num_domains(self) -> int: return len(self.domain2id)
    @property
    def num_languages(self) -> int: return len(self.language2id)
    @property
    def num_generators(self) -> int: return len(self.generator2id)

    def save(self, path: str | Path) -> None:
        payload = {
            "language2id":  self.language2id,
            "generator2id": self.generator2id,
            "domain2id": {f"{lang}|{gen}": idx for (lang, gen), idx in self.domain2id.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DomainRegistry":
        payload = json.loads(Path(path).read_text())
        reg = cls()
        reg.language2id  = payload["language2id"]
        reg.generator2id = payload["generator2id"]
        reg.domain2id    = {tuple(k.split("|", 1)): v for k, v in payload["domain2id"].items()}
        return reg

# ---------------------------------------------------------------------------
# CodeDataset
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    def __init__(
        self,
        df: Any,
        tokenizer: PreTrainedTokenizerBase,
        registry: DomainRegistry,
        max_length: int = 512,
        augment: bool = False,
    ) -> None:
        self.dataset    = df
        self.tokenizer  = tokenizer
        self.registry   = registry
        self.max_length = max_length
        self.augment    = augment

        # 1. Backend Detection
        if isinstance(df, datasets.Dataset):
            self.backend = "hf"
            self.dataset.set_format(type="python")
        elif isinstance(df, pd.DataFrame):
            self.backend = "pandas"
        elif isinstance(df, list):
            self.backend = "list"
        else:
            self.backend = "unknown"

        # 2. Setup entropy strategy
        self._entropy = self._setup_entropy()

        # 3. Cache labels and IDs for sampling speed
        self._labels        = self._build_ids("label")
        self._domain_ids    = self._build_ids("domain")
        self._language_ids  = self._build_ids("language")
        self._generator_ids = self._build_ids("generator")

    def _setup_entropy(self) -> List[float]:
        # A) Check if column "entropy" exists
        has_col = False
        if self.backend == "hf":
            has_col = "entropy" in self.dataset.column_names
        elif self.backend == "pandas":
            has_col = "entropy" in self.dataset.columns
        elif self.backend == "list" and len(self.dataset) > 0:
            has_col = "entropy" in self.dataset[0]

        if has_col:
            logger.info("Using pre-existing entropy column.")
            if self.backend == "hf": return self.dataset["entropy"]
            if self.backend == "pandas": return self.dataset["entropy"].tolist()
            return [row["entropy"] for row in self.dataset]

        # B) Compute once based on backend
        logger.info("Computing identifier entropy for backend: %s", self.backend)
        if self.backend == "pandas":
            self.dataset["entropy"] = self.dataset["code"].apply(identifier_entropy)
            return self.dataset["entropy"].tolist()
        
        if self.backend == "hf":
            # Optimized column access for HF
            codes = self.dataset["code"]
            return [identifier_entropy(c) for c in codes]

        # List or mixed backend
        return [identifier_entropy(row["code"]) for row in self.dataset]

    def _build_ids(self, kind: str) -> List[int]:
        # Optimization: Row-wise access avoidance for HF and Pandas
        if self.backend == "hf":
            if kind == "label": return [int(x) for x in self.dataset["label"]]
            langs, gens = self.dataset["language"], self.dataset["generator"]
        elif self.backend == "pandas":
            if kind == "label": return self.dataset["label"].astype(int).tolist()
            langs, gens = self.dataset["language"].tolist(), self.dataset["generator"].tolist()
        else:
            if kind == "label": return [int(r["label"]) for r in self.dataset]
            langs = [str(r["language"]) for r in self.dataset]
            gens  = [str(r["generator"]) for r in self.dataset]

        if kind == "language": return [self.registry.language2id.get(l, -1) for l in langs]
        if kind == "generator": return [self.registry.generator2id.get(g, -1) for g in gens]
        # Domain
        return [self.registry.get_domain_id(l, g) for l, g in zip(langs, gens)]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Detection-based indexing
        if self.backend == "pandas":
            row = self.dataset.iloc[idx]
        else:
            row = self.dataset[idx]

        code = str(row["code"])
        label = int(row["label"])
        entropy = self._entropy[idx]

        # 2. Augmentations
        if self.augment:
            code = normalize_indent(code)
            code = remove_headers(code)
            code = language_dropout(code)
            code = reduce_language_specific_tokens(code)
            code = mask_comments(code)
            code = mask_identifiers(code)
            if random.random() < 0.3:
                code = shuffle_identifiers(code)
            code = char_span(code, max_chars=2000)

        # 3. Tokenize
        enc = self.tokenizer(
            code,
            add_special_tokens=False,
            truncation=True,
            max_length=100000,
            return_tensors=None,
        )
        ids = enc["input_ids"]

        if self.augment:
            if random.random() < 0.5:
                ids = random_span(ids, self.max_length)
            else:
                ids = multi_span(ids, self.max_length)
        
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        
        if cls_id is not None and sep_id is not None:
            ids = [cls_id] + ids + [sep_id]
        
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
            if sep_id is not None and ids[-1] != sep_id:
                ids[-1] = sep_id
            attention_mask = [1] * self.max_length
        else:
            pad_len = self.max_length - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
            ids = ids + [pad_id] * pad_len

        return {
            "input_ids":      torch.tensor(ids,            dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label":          torch.tensor(label,          dtype=torch.long),
            "entropy":        torch.tensor(entropy,        dtype=torch.float),
            "domain_id":      torch.tensor(self._domain_ids[idx],    dtype=torch.long),
            "language_id":    torch.tensor(self._language_ids[idx],  dtype=torch.long),
            "generator_id":   torch.tensor(self._generator_ids[idx], dtype=torch.long),
        }

    # API compat for samplers
    @property
    def domain_ids_list(self) -> List[int]: return self._domain_ids
    @property
    def language_ids_list(self) -> List[int]: return self._language_ids
    @property
    def generator_ids_list(self) -> List[int]: return self._generator_ids
    @property
    def labels_list(self) -> List[int]: return self._labels

# ---------------------------------------------------------------------------
# Experiment Mode Configuration
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ExperimentConfig:
    """
    Controls dataset size and validation strategy for fast experimentation.

    Parameters
    ----------
    mode : str
        "debug"       → ~5,000 samples
        "fast_dev"    → ~50,000 samples
        "strong_dev"  → ~100,000 samples
        "full"        → all samples (default)
    val_strategy : str
        "standard_random"   → normal train/val split
        "leave_language_out" → hold out one language for validation
        "leave_generator_out"→ hold out one generator for validation
    holdout_value : str
        The language or generator to hold out (only used with OOD strategies).
    val_ratio : float
        Fraction of training data used for standard_random val split.
    seed : int
        Random seed for reproducibility.
    """
    mode:          Literal["debug", "fast_dev", "strong_dev", "full"] = "full"
    val_strategy:  Literal["standard_random", "leave_language_out", "leave_generator_out"] = "standard_random"
    val_ratio:     float         = 0.1
    seed:          int           = 42

    SIZES = {
        "debug":      5_000,
        "fast_dev":   50_000,
        "strong_dev": 100_000,
        "full":       None,  # No limit
    }

    @property
    def target_size(self) -> Optional[int]:
        return self.SIZES[self.mode]


def stratified_sample(
    df:    pd.DataFrame,
    n:     int,
    keys:  List[str] = ("label", "language", "generator"),
    seed:  int = 42,
) -> pd.DataFrame:
    """
    Draw `n` rows from `df` using proportional stratified sampling
    across all combinations of `keys`. Works for 1M+ samples efficiently.
    """
    if n >= len(df):
        return df.reset_index(drop=True)

    groups = df.groupby(list(keys), group_keys=False)
    n_total = len(df)

    def proportional_draw(group: pd.DataFrame) -> pd.DataFrame:
        quota = max(1, round(len(group) / n_total * n))
        quota = min(quota, len(group))
        return group.sample(quota, random_state=seed)

    sampled = groups.apply(proportional_draw).reset_index(drop=True)
    # Due to rounding, trim or top up to exactly n
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed).reset_index(drop=True)
    return sampled


def split_ood_validation(
    df:            pd.DataFrame,
    strategy:      str,
    val_ratio:     float = 0.1,
    seed:          int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df) based on the specified validation strategy.
    """
    if strategy == "leave_language_out":
        langs = sorted(df["language"].unique().tolist())
        target = langs[0]
        logger.info("Leave-Language-Out val strategy: holding out '%s'", target)
        val_df   = df[df["language"] == target].reset_index(drop=True)
        train_df = df[df["language"] != target].reset_index(drop=True)

    elif strategy == "leave_generator_out":
        gens = sorted(df["generator"].unique().tolist())
        target = gens[0]
        logger.info("Leave-Generator-Out val strategy: holding out '%s'", target)
        val_df   = df[df["generator"] == target].reset_index(drop=True)
        train_df = df[df["generator"] != target].reset_index(drop=True)

    else:  # standard_random
        val_df   = df.sample(frac=val_ratio, random_state=seed).reset_index(drop=True)
        train_df = df.drop(val_df.index).reset_index(drop=True)

    return train_df, val_df


def print_dataset_stats(name: str, df: pd.DataFrame) -> None:
    """Print useful stats about a DataFrame-based dataset split."""
    label_counts = df["label"].value_counts().to_dict()
    n_langs = df["language"].nunique() if "language" in df.columns else "?"
    n_gens  = df["generator"].nunique() if "generator" in df.columns else "?"

    logger.info("━" * 60)
    logger.info("  %-20s  %d samples", f"[{name}]", len(df))
    logger.info("  Languages:  %d | Generators: %d", n_langs, n_gens)
    logger.info("  Label dist: %s", {k: f"{v} ({v/len(df)*100:.1f}%)" for k, v in sorted(label_counts.items())})
    if "language" in df.columns:
        lang_dist = df["language"].value_counts().head(10).to_dict()
        logger.info("  Languages:  %s", lang_dist)
    logger.info("━" * 60)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_datasets(
    train_file:         str,
    val_file:           str,
    tokenizer:          PreTrainedTokenizerBase,
    max_length:         int = 512,
    registry_save_path: Optional[str] = None,
    # Experiment parameters (new, all optional for backward compat)
    experiment_mode:    str = "full",
    val_strategy:       str = "standard_random",
    val_ratio:          float = 0.1,
    seed:               int = 42,
) -> Tuple[CodeDataset, CodeDataset, DomainRegistry]:
    """
    Load data, apply experiment sizing + validation strategy, build datasets.

    Parameters
    ----------
    experiment_mode : {"debug", "fast_dev", "strong_dev", "full"}
        Automatically subsets training data for faster experiments.
    val_strategy : {"standard_random", "leave_language_out", "leave_generator_out"}
        Validation data selection strategy.
    val_ratio : float
        Fraction used for standard_random val split.
    seed : int
        Random seed for reproducibility.
    """
    cfg = ExperimentConfig(mode=experiment_mode, val_strategy=val_strategy,
                           val_ratio=val_ratio, seed=seed)

    logger.info("Loading datasets | mode=%s | val_strategy=%s", cfg.mode, cfg.val_strategy)

    # ---- Load to DataFrame for preprocessing --------------------------------
    if train_file.endswith(".parquet"):
        import pyarrow.parquet as pq
        train_df = pq.read_table(train_file).to_pandas()
        val_df_raw = pq.read_table(val_file).to_pandas() if cfg.val_strategy == "standard_random" and val_file else None
    else:
        train_df = pd.read_csv(train_file)
        val_df_raw = pd.read_csv(val_file) if cfg.val_strategy == "standard_random" else None

    for col in ["language", "generator"]:
        train_df[col] = train_df[col].fillna("unknown")

    # ---- Stratified subsetting -----------------------------------------------
    if cfg.target_size is not None:
        logger.info("Subsampling to ~%d samples (stratified)...", cfg.target_size)
        train_df = stratified_sample(train_df, cfg.target_size,
                                     keys=["label", "language", "generator"], seed=cfg.seed)

    # ---- OOD Validation split ------------------------------------------------
    if cfg.val_strategy != "standard_random":
        train_df, val_df = split_ood_validation(
            train_df, cfg.val_strategy, cfg.val_ratio, cfg.seed)
    elif val_df_raw is not None:
        val_df = val_df_raw
        for col in ["language", "generator"]:
            val_df[col] = val_df[col].fillna("unknown")
        if cfg.target_size is not None:
            val_size = max(1000, cfg.target_size // 5)
            val_df = stratified_sample(val_df, val_size,
                                       keys=["label", "language", "generator"], seed=cfg.seed)
    else:
        # No external val file — carve out from train
        train_df, val_df = split_ood_validation(
            train_df, "standard_random", cfg.val_ratio, cfg.seed)

    # ---- Stats ---------------------------------------------------------------
    print_dataset_stats("Train", train_df)
    print_dataset_stats("Val",   val_df)

    # ---- Fit Registry --------------------------------------------------------
    registry = DomainRegistry().fit(train_df["language"], train_df["generator"])
    if registry_save_path:
        registry.save(registry_save_path)

    # ---- Use HF Dataset for large files, Pandas DataFrame otherwise ----------
    if train_file.endswith(".parquet") and experiment_mode == "full":
        # For full Parquet mode, re-load as HF Dataset for memory efficiency
        train_ds_hf = hf_load_dataset("parquet", data_files={"train": train_file})["train"]
        val_ds_hf   = hf_load_dataset("parquet", data_files={"val": val_file})["val"]
        train_dataset = CodeDataset(train_ds_hf, tokenizer, registry, max_length, augment=True)
        val_dataset   = CodeDataset(val_ds_hf,   tokenizer, registry, max_length, augment=False)
    else:
        # Experiment mode or CSV: use processed DataFrames
        train_dataset = CodeDataset(train_df, tokenizer, registry, max_length, augment=True)
        val_dataset   = CodeDataset(val_df,   tokenizer, registry, max_length, augment=False)

    logger.info("Train size: %d | Val size: %d", len(train_dataset), len(val_dataset))
    return train_dataset, val_dataset, registry
