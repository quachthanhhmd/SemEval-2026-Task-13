"""
dataset.py
----------
Enhanced Dataset for AI-generated code detection with domain-aware labeling,
advanced source code augmentations, and Parquet support.

Incorporates logic from reference.txt:
  - Identifier extraction and entropy calculation.
  - Augmentations: masking, shuffling, header removal, language dropout, etc.
  - Parquet loading via HF datasets.
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
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset as hf_load_dataset
from src.features import AgnosticFeatureExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and Regex from reference.txt
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
# Helper Functions from reference.txt
# ---------------------------------------------------------------------------

def extract_identifiers(code: str) -> List[str]:
    ids = re.findall(IDENTIFIER_REGEX, code)
    return [i for i in ids if i not in KEYWORDS]

def identifier_entropy(code: str) -> float:
    ids = extract_identifiers(code)
    if len(ids) == 0:
        return 0.0
    counter = Counter(ids)
    total = len(ids)
    entropy = 0.0
    for c in counter.values():
        p = c / total
        entropy -= p * math.log2(p)
    # Normalize by max possible entropy for this unique set of identifiers
    entropy = entropy / math.log2(len(counter) + 1)
    return float(entropy)

# ---------------------------------------------------------------------------
# Augmentations from reference.txt
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
# Span Sampling from reference.txt
# ---------------------------------------------------------------------------

def random_span(ids: List[int], max_len: int) -> List[int]:
    target_len = random.randint(int(max_len * 0.7), max_len)
    if len(ids) > target_len:
        start = random.randint(0, len(ids) - target_len)
        ids = ids[start:start + target_len]
    return ids

def multi_span(ids: List[int], max_len: int) -> List[int]:
    span = max_len // 2
    if len(ids) <= max_len:
        return ids
    start1 = random.randint(0, len(ids) - span)
    start2 = random.randint(0, len(ids) - span)
    span1 = ids[start1:start1 + span]
    span2 = ids[start2:start2 + span]
    return (span1 + span2)[:max_len]

# ---------------------------------------------------------------------------
# Domain registry helpers (unchanged)
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
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):
    REQUIRED_COLS = {"code", "label", "language", "generator"}

    def __init__(
        self,
        df: Any,  # Can be pd.DataFrame or HF Dataset
        tokenizer: PreTrainedTokenizerBase,
        registry: DomainRegistry,
        extractor: Optional[AgnosticFeatureExtractor] = None,
        max_length: int = 512,
        augment: bool = False,
        char_crop_limit: Optional[int] = None,
    ) -> None:
        self.dataset    = df
        self.tokenizer  = tokenizer
        self.registry   = registry
        self.extractor  = extractor
        self.max_length = max_length
        self.augment    = augment
        self.char_crop_limit = char_crop_limit

        # Cache meta-ids for sampling speed
        self._domain_ids    = self._build_ids("domain")
        self._language_ids  = self._build_ids("language")
        self._generator_ids = self._build_ids("generator")
        self._labels        = self._build_ids("label")

    def _build_ids(self, kind: str) -> List[int]:
        ids: List[int] = []
        for i in range(len(self.dataset)):
            row = self.dataset[i]
            if kind == "label":
                ids.append(int(row["label"]))
                continue
            
            lang = str(row["language"])
            gen  = str(row["generator"])
            if kind == "domain":
                ids.append(self.registry.get_domain_id(lang, gen))
            elif kind == "language":
                ids.append(self.registry.language2id.get(lang, -1))
            else:
                ids.append(self.registry.generator2id.get(gen, -1))
        return ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.dataset[idx]
        code = str(row["code"])
        label = int(row["label"])

        # 0. Random char crop (VRAM Optimization)
        if self.char_crop_limit and len(code) > self.char_crop_limit:
            start_idx = random.randint(0, len(code) - self.char_crop_limit)
            code = code[start_idx : start_idx + self.char_crop_limit]

        # 1. Stylometric Features (10 features)
        if self.extractor is not None:
            raw_features = self.extractor.extract_all(code)
            extra_features = torch.tensor(raw_features, dtype=torch.float)
            # Log normalization for unbounded features (0: avg_id_len, 6: line_std)
            for i in [0, 6]:
                extra_features[i] = torch.log1p(extra_features[i])
            extra_features = torch.clamp(extra_features, min=0.0, max=100.0)
        else:
            # Fallback to just identifier entropy if extractor is not provided
            entropy = identifier_entropy(code)
            extra_features = torch.zeros(10, dtype=torch.float)
            extra_features[1] = entropy # idx 1 is id_entropy

        # 2. Augmentations (if enabled)
        if self.augment:
            code = normalize_indent(code)
            code = remove_headers(code)
            code = language_dropout(code)
            code = reduce_language_specific_tokens(code)
            code = mask_comments(code)
            code = mask_identifiers(code)
            if random.random() < 0.3:
                code = shuffle_identifiers(code)

        # 3. Tokenize
        enc = self.tokenizer(
            code,
            add_special_tokens=True,
            truncation=False,  # We handle truncation manually via spans if augmenting
            return_tensors=None,
        )
        ids = enc["input_ids"]

        # 4. Span sampling or standard padding
        if self.augment:
            if random.random() < 0.5:
                ids = random_span(ids, self.max_length)
            else:
                ids = multi_span(ids, self.max_length)
        
        # Manual pad/trunc to max_length
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            pad_len = self.max_length - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids":      torch.tensor(ids,            dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label":          torch.tensor(label,          dtype=torch.long),
            "extra_features": extra_features,
            "domain_id":      torch.tensor(self._domain_ids[idx],    dtype=torch.long),
            "language_id":    torch.tensor(self._language_ids[idx],  dtype=torch.long),
            "generator_id":   torch.tensor(self._generator_ids[idx], dtype=torch.long),
        }

    @property
    def domain_ids_list(self) -> List[int]:
        return self._domain_ids

    @property
    def language_ids_list(self) -> List[int]:
        return self._language_ids

    @property
    def generator_ids_list(self) -> List[int]:
        return self._generator_ids

    @property
    def labels_list(self) -> List[int]:
        return self._labels

# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_datasets(
    train_file: str,
    val_file: str,
    tokenizer: PreTrainedTokenizerBase,
    extractor: Optional[AgnosticFeatureExtractor] = None,
    max_length: int = 512,
    char_crop_limit: Optional[int] = None,
    registry_save_path: Optional[str] = None,
) -> Tuple[CodeDataset, CodeDataset, DomainRegistry]:
    """
    Load data (CSV or Parquet), fit DomainRegistry, build datasets.
    """
    logger.info("Loading datasets from %s and %s", train_file, val_file)
    
    if train_file.endswith(".parquet"):
        train_ds_hf = hf_load_dataset("parquet", data_files={"train": train_file})["train"]
        val_ds_hf   = hf_load_dataset("parquet", data_files={"val": val_file})["val"]
        
        # Build registry from training data
        train_langs = pd.Series(train_ds_hf["language"])
        train_gens  = pd.Series(train_ds_hf["generator"])
        
        train_ds_src = train_ds_hf
        val_ds_src   = val_ds_hf
    else:
        # Fallback to CSV
        train_df = pd.read_csv(train_file)
        val_df   = pd.read_csv(val_file)
        for col in ["language", "generator"]:
            train_df[col] = train_df[col].fillna("unknown")
            val_df[col]   = val_df[col].fillna("unknown")
        
        train_langs = train_df["language"]
        train_gens  = train_df["generator"]
        
        train_ds_src = train_df.to_dict('records') # Convert to list of dicts for uniformity
        val_ds_src   = val_df.to_dict('records')

    registry = DomainRegistry().fit(train_langs, train_gens)
    if registry_save_path:
        registry.save(registry_save_path)

    train_dataset = CodeDataset(train_ds_src, tokenizer, registry, extractor, max_length, augment=True, char_crop_limit=char_crop_limit)
    val_dataset   = CodeDataset(val_ds_src,   tokenizer, registry, extractor, max_length, augment=False, char_crop_limit=char_crop_limit)

    logger.info("Train size: %d | Val size: %d", len(train_dataset), len(val_dataset))
    return train_dataset, val_dataset, registry
