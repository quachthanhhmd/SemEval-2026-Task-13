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

from datasets import load_dataset as hf_load_dataset

try:
    from tree_sitter_languages import get_parser
    TREE_SITTER_PARSERS = {
        "python":     get_parser("python"),
        "cpp":        get_parser("cpp"),
        "java":       get_parser("java"),
        "go":         get_parser("go"),
        "javascript": get_parser("javascript"),
        "php":        get_parser("php"),
        "c":          get_parser("c"),
        "c_sharp":    get_parser("c_sharp"),
    }
except ImportError:
    logger.warning("tree_sitter_languages not found. AST augmentations will be disabled.")
    TREE_SITTER_PARSERS = {}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and Regex
# ---------------------------------------------------------------------------

IDENTIFIER_REGEX = r"\$?[A-Za-z_][A-Za-z0-9_]*"

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
    re.compile(r"std::"),
    re.compile(r"System\."),
    re.compile(r"print\("),
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

COMMON_STD = {
    "main", "printf", "cout", "endl", "len", "size", 
    "map", "vector", "string", "list", "dict", "set", 
    "String", "Object", "Math", "std"
}

KEYWORDS = PYTHON_KW | C_LIKE_KW | GO_KW | PHP_KW | COMMON_STD

DOUBLE_COLON_PATTERN = re.compile(r"::")
ARROW_PATTERN = re.compile(r"->")
PHP_TAG_PATTERN = re.compile(r"<\?php|\?>")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def ast_collect_identifiers(node, code_bytes, ids):
    # Tree-sitter specific identifier collection
    if "identifier" in node.type:
        try:
            name = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
            ids.append(name)
        except Exception:
            pass
    for child in node.children:
        ast_collect_identifiers(child, code_bytes, ids)

def ast_extract_identifiers(code: str, language: str) -> List[str]:
    # Map high-level language names to tree-sitter keys
    lang_map = {
        "Python": "python", "C++": "cpp", "Java": "java", "Go": "go",
        "JavaScript": "javascript", "PHP": "php", "C": "c", "C#": "c_sharp"
    }
    ts_lang = lang_map.get(language)
    if not ts_lang or ts_lang not in TREE_SITTER_PARSERS:
        # Fallback to regex if parser is missing
        return extract_identifiers(code)
    
    try:
        parser = TREE_SITTER_PARSERS[ts_lang]
        tree = parser.parse(bytes(code, "utf8"))
        ids = []
        ast_collect_identifiers(tree.root_node, bytes(code, "utf8"), ids)
        return [i for i in ids if i not in KEYWORDS]
    except Exception:
        return extract_identifiers(code)

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

def rename_identifiers(code: str, p: float = 0.5) -> str:
    ids = list(set(extract_identifiers(code)))
    # Use randomized names to avoid uniform patterns that lead to artifact overfitting
    mapping = {k: f"var_{random.randint(0, 9999)}" for k in ids}
    # Ensure longest keys are replaced first to prevent partial replacement of longer identifiers
    for k in sorted(mapping.keys(), key=len, reverse=True):
        if random.random() < p:
            code = re.sub(rf"(?<!\.)(?<!::)(?<!->)\b{re.escape(k)}\b", mapping[k], code)
    return code

def ast_rename_identifiers(code: str, language: str, p: float = 0.5) -> str:
    lang_map = {
        "Python": "python", "C++": "cpp", "Java": "java", "Go": "go",
        "JavaScript": "javascript", "PHP": "php", "C": "c", "C#": "c_sharp"
    }
    ts_lang = lang_map.get(language)
    if not ts_lang or ts_lang not in TREE_SITTER_PARSERS:
        return rename_identifiers(code, p)
        
    try:
        parser = TREE_SITTER_PARSERS[ts_lang]
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        
        nodes = []
        def collect(node):
            if "identifier" in node.type:
                nodes.append(node)
            for child in node.children:
                collect(child)
                
        collect(tree.root_node)
        
        # Filter keywords and build mapping
        mapping = {}
        for n in nodes:
            name = code_bytes[n.start_byte:n.end_byte].decode("utf-8")
            if name not in KEYWORDS and name not in mapping:
                mapping[name] = f"var_{random.randint(0, 9999)}".encode("utf-8")
                
        # Replace from bottom up to preserve earlier indices
        code_bytearray = bytearray(code_bytes)
        for node in reversed(nodes):
            name = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
            if name in mapping and random.random() < p:
                code_bytearray[node.start_byte:node.end_byte] = mapping[name]
                
        return code_bytearray.decode("utf-8")
    except Exception:
        return rename_identifiers(code, p)

COMMENT_NOISE = [
    "// TODO",
    "// fix later",
    "// temporary",
    "# TODO",
]

def inject_comments(code: str, p: float = 0.3) -> str:
    if random.random() >= p:
        return code
    lines = code.split("\n")
    if not lines: return code
    pos = random.randint(0, len(lines))
    comment = random.choice(COMMENT_NOISE)
    lines.insert(pos, comment)
    return "\n".join(lines)

def perturb_constants(code: str, p: float = 0.3) -> str:
    if random.random() >= p:
        return code
    def repl(match):
        return str(random.randint(0, 20))
    return re.sub(r"\b\d+\b", repl, code)

def whitespace_noise(code: str, p: float = 0.3) -> str:
    if random.random() >= p:
        return code
    return re.sub(r"\s+", " ", code)

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

def language_dropout(code: str, p: float = 0.1) -> str:
    if random.random() >= p:
        return code
    for pat in LANG_PATTERNS:
        code = pat.sub("", code)
    return code

# ------------------------------------------------
# Canonicalization
# ------------------------------------------------

def canonicalize_assignment_ops(code: str) -> str:
    patterns = [
        (r"\b(\w+)\s*\+=\s*(\w+)", r"\1 = \1 + \2"),
        (r"\b(\w+)\s*-\=\s*(\w+)", r"\1 = \1 - \2"),
        (r"\b(\w+)\s*\*=\s*(\w+)", r"\1 = \1 * \2"),
        (r"\b(\w+)\s*/=\s*(\w+)", r"\1 = \1 / \2"),
        (r"\b(\w+)\s*%\=\s*(\w+)", r"\1 = \1 % \2"),
    ]
    for pattern, repl in patterns:
        code = re.sub(pattern, repl, code)
    return code

def canonicalize_increment(code: str) -> str:
    # Restrict to standalone statements to avoid breaking expressions like arr[i++]
    patterns = [
        (r"(?m)^\s*(\w+)\s*\+\+\s*(;)?$", r"\1 = \1 + 1\2"),
        (r"(?m)^\s*\+\+\s*(\w+)\s*(;)?$", r"\1 = \1 + 1\2"),
        (r"(?m)^\s*(\w+)\s*--\s*(;)?$", r"\1 = \1 - 1\2"),
        (r"(?m)^\s*--\s*(\w+)\s*(;)?$", r"\1 = \1 - 1\2"),
    ]
    for pattern, repl in patterns:
        code = re.sub(pattern, repl, code)
    return code

def canonicalize_logical_ops(code: str) -> str:
    # We only normalize spacing around operators, no symbol replacement to avoid pseudo-Python artifacts.
    replacements = {
        "&&": " && ",
        "||": " || ",
        "!": " ! ",
    }
    for k, v in replacements.items():
        code = code.replace(k, v)
    return code

def canonicalize_print(code: str) -> str:
    # Full expression capture for C++ to avoid trailing garbage
    code = re.sub(r"std::cout\s*<<\s*(.*?);", r"print(\1);", code)
    
    patterns = [
        r"System\.out\.println",
        r"System\.out\.print",
        r"console\.log",
        r"printf",
    ]
    for p in patterns:
        code = re.sub(p, "print", code)
    return code

def canonicalize_boolean_checks(code: str, language: str) -> str:
    # Only safe for C-like syntaxes. Python relies on 'if flag:'
    if language in ["C++", "Java", "C", "C#", "Go", "PHP", "JavaScript"]:
        patterns = [
            (r"\bif\s*\(\s*(\w+)\s*\)", r"if (\1 != 0)"),      # if(flag)
            (r"\bwhile\s*\(\s*(\w+)\s*\)", r"while (\1 != 0)"), # while(flag)
        ]
        for pattern, repl in patterns:
            code = re.sub(pattern, repl, code)
    return code

def canonicalize_code(code: str, language: str, p: float = 0.15) -> str:
    if random.random() > p:
        return code
    code = canonicalize_assignment_ops(code)
    code = canonicalize_increment(code)
    code = canonicalize_logical_ops(code)
    code = canonicalize_print(code)
    code = canonicalize_boolean_checks(code, language)
    return code

def normalize_indent(code: str) -> str:
    lines = code.split("\n")
    lines = [line.lstrip() for line in lines]
    return "\n".join(lines)

def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(name: str) -> str:
    components = name.split('_')
    if len(components) <= 1: return name
    return components[0].lower() + ''.join(x.title() for x in components[1:])

def perturb_identifier_styles(code: str, p: float = 0.3) -> str:
    if random.random() >= p:
        return code
    ids = list(set(extract_identifiers(code)))
    mapping = {}
    for k in ids:
        if '_' in k:
            if random.random() < 0.5:
                mapping[k] = snake_to_camel(k)
        elif k != k.lower() and k != k.upper():
            if random.random() < 0.5:
                mapping[k] = camel_to_snake(k)
                
    for k in sorted(mapping.keys(), key=len, reverse=True):
        code = re.sub(rf"(?<!\.)(?<!::)(?<!->)\b{re.escape(k)}\b", mapping[k], code)
    return code

def brace_style_transfer(code: str) -> str:
    # Convert compact brace style to production/expanded style
    return re.sub(r"\)\s*{", ")\n{", code)

def indent_transfer(code: str) -> str:
    # Randomly shuffle indentation styles to prevent the model from overfitting on whitespace
    lines = code.split("\n")
    new_lines = []
    indent_choices = ["  ", "    ", "\t"]
    new_indent = random.choice(indent_choices)
    
    for line in lines:
        new_lines.append(new_indent + line.lstrip())
    return "\n".join(new_lines)

def style_transfer(code: str, language: str, p: float = 0.3) -> str:
    if random.random() > p:
        return code
    # Composite augmentation: changes names, braces, and indentation
    code = perturb_identifier_styles(code, p=1.0) # Always try styles if p is triggered
    if language in ["C++", "Java", "JavaScript", "PHP", "C", "C#"]:
        code = brace_style_transfer(code)
    code = indent_transfer(code)
    return code

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

def span_mixup(ids1: List[int], ids2: List[int], max_len: int) -> List[int]:
    """
    Mixup two token sequences by splitting at 30%-70% of max_len.
    Takes first part from ids1, remaining from end of ids2.
    Resulting sequence length is exactly the target split sum.
    """
    target_max = max_len - 2  # Leave space for [CLS] and [SEP]
    
    # Trim to avoid index out of bounds if they are short
    if len(ids1) > target_max: ids1 = ids1[:target_max]
    if len(ids2) > target_max: ids2 = ids2[:target_max]
    
    # We need to pick a valid split point based on available tokens
    min_cut = max(1, int(target_max * 0.3))
    max_cut = min(len(ids1) - 1, int(target_max * 0.7))
    
    # If sequences are too short to safely mixup at proportion, fallback to simple concat or return ids1
    if min_cut >= max_cut or len(ids2) == 0:
        return ids1

    cut = random.randint(min_cut, max_cut)
    needed_from_2 = target_max - cut
    
    part1 = ids1[:cut]
    part2 = ids2[-needed_from_2:] if needed_from_2 <= len(ids2) else ids2
    
    return (part1 + part2)[:target_max]

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
        self.domain2id: Dict[Tuple[str, str, str], int] = {}
        self.aug_types = ["orig", "canon", "scramble", "mix"]
        self.id2language: Dict[int, str] = {}
        self.id2generator: Dict[int, str] = {}

    def fit(self, languages: pd.Series, generators: pd.Series) -> "DomainRegistry":
        unique_langs = sorted(languages.dropna().unique().tolist())
        unique_gens  = sorted(generators.dropna().unique().tolist())
        self.language2id  = {lang: i for i, lang in enumerate(unique_langs)}
        self.generator2id = {gen:  i for i, gen  in enumerate(unique_gens)}
        
        self.id2language  = {v: k for k, v in self.language2id.items()}
        self.id2generator = {v: k for k, v in self.generator2id.items()}
        
        idx = 0
        for lang in unique_langs:
            for gen in unique_gens:
                for aug in self.aug_types:
                    self.domain2id[(lang, gen, aug)] = idx
                    idx += 1
        return self

    def get_domain_id(self, language: str, generator: str, aug_type: str = "orig") -> int:
        return self.domain2id.get((language, generator, aug_type), -1)

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
            "domain2id": {f"{lang}|{gen}|{aug}": idx for (lang, gen, aug), idx in self.domain2id.items()},
            "aug_types": self.aug_types,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DomainRegistry":
        payload = json.loads(Path(path).read_text())
        reg = cls()
        reg.language2id  = payload["language2id"]
        reg.generator2id = payload["generator2id"]
        reg.aug_types    = payload.get("aug_types", ["orig", "canon", "scramble", "mix"])
        reg.domain2id    = {tuple(k.split("|")): v for k, v in payload["domain2id"].items()}
        reg.id2language  = {v: k for k, v in reg.language2id.items()}
        reg.id2generator = {v: k for k, v in reg.generator2id.items()}
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
        
        # 4. Token Cache
        self._cached_ids    = self._build_cache()

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

    def _build_cache(self) -> List[List[int]]:
        """Pre-tokenize all codes in the dataset down to integer lists to save overhead."""
        logger.info("Pre-tokenizing entire dataset to memory...")
        codes = []
        if self.backend == "pandas":
            codes = self.dataset["code"].tolist()
        elif self.backend == "hf":
            codes = self.dataset["code"]
        else:
            codes = [row["code"] for row in self.dataset]
            
        cached_ids = []
        for code in codes:
            # We don't apply augmentations during caching as those are dynamic
            enc = self.tokenizer(str(code), add_special_tokens=False, truncation=True, max_length=100000, return_tensors=None)
            cached_ids.append(enc["input_ids"])
            
        logger.info("Cached %d tokenized sequences.", len(cached_ids))
        return cached_ids

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
        # Domain (orig as fallback base)
        return [self.registry.get_domain_id(l, g, "orig") for l, g in zip(langs, gens)]

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
        lang_str = str(row.get("language", "unknown"))
        entropy = self._entropy[idx]
        is_mixed = 0
        aug_type = "orig"

        # 2. Augmentations (Textual)
        if self.augment:
            code_before = code
            code = canonicalize_code(code, lang_str, p=0.15)
            if code != code_before:
                aug_type = "canon"
                
            code = normalize_indent(code)
            code = remove_headers(code)
            code = language_dropout(code, p=0.1)
            code = reduce_language_specific_tokens(code)
            code = mask_comments(code)
            
            code_style_before = code
            code = style_transfer(code, lang_str, p=0.3)
            # Use AST-level renaming if available
            code = ast_rename_identifiers(code, lang_str, p=0.3)
            if code != code_style_before:
                aug_type = "scramble"
                
            code = inject_comments(code, p=0.3)
            code = perturb_constants(code, p=0.3)
            code = whitespace_noise(code, p=0.3)
            code = char_span(code, max_chars=2000)

            # Re-tokenize augmented code dynamically
            enc = self.tokenizer(
                code,
                add_special_tokens=False,
                truncation=True,
                max_length=100000,
                return_tensors=None,
            )
            ids = enc["input_ids"]
        else:
            # Drop-in un-augmented cached tokens
            ids = self._cached_ids[idx].copy()

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
            
        # 4. Domain Mixup Augmentation (AI ONLY)
        mixup_prob = 0.15 if label == 1 else 0.0
        if self.augment and random.random() < mixup_prob:
            idx2 = self.sample_different_domain(idx, target_label=label)
            if idx2 != idx:
                aug_type = "mix"
                # Use exactly the same token caching strategy for ids2 directly!
                ids2 = self._cached_ids[idx2].copy()
                
                # Make sure the incoming mix sample is correctly spanned prior to mixup
                if random.random() < 0.5:
                    ids2 = random_span(ids2, self.max_length)
                else:
                    ids2 = multi_span(ids2, self.max_length)
                
                # Perform Span Mixup
                mixed_ids = span_mixup(ids[1:-1] if cls_id is not None else ids, ids2, self.max_length)
                
                # Re-wrap with special tokens
                if cls_id is not None and sep_id is not None:
                    mixed_ids = [cls_id] + mixed_ids + [sep_id]
                
                # Re-calculate padding and masking exactly for max_length constraint
                if len(mixed_ids) > self.max_length:
                    mixed_ids = mixed_ids[:self.max_length]
                    if sep_id is not None and mixed_ids[-1] != sep_id:
                        mixed_ids[-1] = sep_id
                    attention_mask = [1] * self.max_length
                else:
                    pad_len = self.max_length - len(mixed_ids)
                    attention_mask = [1] * len(mixed_ids) + [0] * pad_len
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
                    mixed_ids = mixed_ids + [pad_id] * pad_len
                
                ids = mixed_ids
                is_mixed = 1
                
        # Resolve names and compute updated domain_id
        lang_name = self.registry.id2language.get(self._language_ids[idx], "unknown")
        gen_name  = self.registry.id2generator.get(self._generator_ids[idx], "unknown")
        domain_id = self.registry.get_domain_id(lang_name, gen_name, aug_type)

        return {
            "input_ids":      torch.tensor(ids,            dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label":          torch.tensor(label,          dtype=torch.long),
            "entropy":        torch.tensor(entropy,        dtype=torch.float),
            "domain_id":      torch.tensor(domain_id,      dtype=torch.long),
            "language_id":    torch.tensor(self._language_ids[idx],  dtype=torch.long),
            "generator_id":   torch.tensor(self._generator_ids[idx], dtype=torch.long),
            "is_mixed":       torch.tensor(is_mixed,       dtype=torch.float),
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

    def sample_different_domain(self, current_idx: int, target_label: int) -> int:
        """
        Efficiently find a random sample that differs in language or generator.
        Optimized by checking random indices rather than scanning the entire huge dataset.
        Limit to 10 attempts to guarantee fast runtime.
        """
        curr_lang = self._language_ids[current_idx]
        curr_gen = self._generator_ids[current_idx]
        dataset_len = len(self)
        
        for _ in range(10):
            rand_idx = random.randint(0, dataset_len - 1)
            # Both AI and human check is optional but good practice to mix AI with AI
            # The prompt only requires mixup when the main sample is AI, but ideally the incoming mix is also AI.
            # We strictly enforce different domain criteria:
            if self._labels[rand_idx] == target_label and (self._language_ids[rand_idx] != curr_lang or self._generator_ids[rand_idx] != curr_gen):
                return rand_idx
                
        # Fallback to pure random AI sample if we can't find domain different one quickly
        for _ in range(10):
            rand_idx = random.randint(0, dataset_len - 1)
            if self._labels[rand_idx] == target_label and rand_idx != current_idx:
                return rand_idx
                
        return current_idx

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
