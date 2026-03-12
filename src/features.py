import re
import math
import torch
import logging
import numpy as np
from collections import Counter
from typing import List, Dict

logger = logging.getLogger(__name__)

MULTI_LANG_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return',
    'int', 'float', 'double', 'char', 'void', 'bool', 'boolean', 'string', 'var', 'let', 'const',
    'def', 'function', 'class', 'struct', 'interface', 'package', 'import', 'using', 'namespace',
    'public', 'private', 'protected', 'static', 'final', 'const', 'try', 'catch', 'finally',
    'throw', 'throws', 'new', 'delete', 'true', 'false', 'null', 'nil', 'None', 'self', 'this',
    'func', 'defer', 'go', 'map', 'chan', 'type', 'range'
}

class AgnosticFeatureExtractor:
    """
    Estrae feature stilometriche e language-agnostic dal codice sorgente.
    Usa metriche statistiche (Consistenza, Entropia, Complessità) senza caricare model Neurali Pesanti.
    """
    def __init__(self):
        # B. Compile Regex for Performance
        self.re_words = re.compile(r'\w+')
        self.re_camel = re.compile(r'[a-z][A-Z]')
        self.re_snake = re.compile(r'_')
        self.re_digits = re.compile(r'\d')
        self.re_eq_spaced = re.compile(r' = ')
        self.re_eq_nospaced = re.compile(r'(?<=[^\s])=(?=[^\s])')

    def get_feature_names(self) -> List[str]:
        return [
            # "perplexity",          # RIMOSSO per risparmiare RAM
            "id_len_avg",          # 0
            "id_entropy",          # 1
            "id_short_ratio",      # 2
            "id_num_ratio",        # 3
            "style_consistency",   # 4
            "spacing_ratio",       # 5
            "line_len_std",        # 6
            "ttr",                 # 7
            "comment_ratio",       # 8
            "human_markers"        # 9
        ]

    def _analyze_identifiers(self, words: List[str]) -> List[float]:
        identifiers = [w for w in words if w not in MULTI_LANG_KEYWORDS and not w.isdigit()]
        if not identifiers:
            return [0.0, 0.0, 0.0, 0.0]

        lens = [len(w) for w in identifiers]
        avg_len = np.mean(lens)

        all_chars = "".join(identifiers)
        if not all_chars:
            entropy = 0.0
        else:
            char_counts = Counter(all_chars)
            total_chars = sum(char_counts.values())
            entropy = -sum((c / total_chars) * math.log2(c / total_chars) for c in char_counts.values())

        short_ids = sum(1 for w in identifiers if len(w) <= 2)
        short_ratio = short_ids / len(identifiers)

        num_ids = sum(1 for w in identifiers if self.re_digits.search(w))
        num_ratio = num_ids / len(identifiers)

        return [avg_len, entropy, short_ratio, num_ratio]

    def _analyze_consistency(self, code: str, words: List[str]) -> List[float]:
        identifiers = [w for w in words if w not in MULTI_LANG_KEYWORDS]
        snake_count = sum(1 for w in identifiers if '_' in w)
        camel_count = sum(1 for w in identifiers if self.re_camel.search(w))
        total_style = snake_count + camel_count
        
        consistency = abs(snake_count - camel_count) / total_style if total_style > 0 else 0.0
        
        spaced = len(self.re_eq_spaced.findall(code))
        nospaced = len(self.re_eq_nospaced.findall(code))
        total_eq = spaced + nospaced
        spacing_ratio = nospaced / total_eq if total_eq > 0 else 0.0

        return [consistency, spacing_ratio]

    def _analyze_structure(self, code: str, words: List[str]) -> List[float]:
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if non_empty_lines:
            line_lens = [len(l) for l in non_empty_lines]
            line_std = np.std(line_lens)
        else:
            line_std = 0.0
            
        ttr = len(set(words)) / len(words) if words else 0.0

        comment_lines = sum(1 for l in non_empty_lines if l.strip().startswith(('#', '//', '/*')))
        comment_ratio = comment_lines / (len(non_empty_lines) + 1)
        
        markers = len(re.findall(r'\b(TODO|FIXME|XXX|HACK|DEBUG)\b', code, re.IGNORECASE))
        marker_score = 1.0 if markers > 0 else 0.0

        return [line_std, ttr, comment_ratio, marker_score]

    def extract_all(self, code: str) -> List[float]:
        if not isinstance(code, str):
            code = str(code)

        words = self.re_words.findall(code)
        f_ids = self._analyze_identifiers(words)
        f_const = self._analyze_consistency(code, words)
        f_struct = self._analyze_structure(code, words)
        
        return f_ids + f_const + f_struct
