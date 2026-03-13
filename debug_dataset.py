"""
debug_dataset.py
----------------
Standalone diagnostic tool for checking dataset integrity, tokenization,
and meta-learning batch structures.
"""

import argparse
import logging
import os
import sys
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure project root is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import build_datasets
from src.sampler import LanguageBalancedSampler, domain_collate_fn
from src.trainer import split_meta_language

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def analyze_dataset(ds, name: str = "Dataset"):
    logger.info(f"--- ANALYZING {name.upper()} ---")
    df = ds.dataset
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        df_pd = df
    else:
        df_pd = pd.DataFrame(df)

    stats = {
        "Total Samples": len(df_pd),
        "Columns": list(df_pd.columns),
        "Languages": df_pd["language"].nunique() if "language" in df_pd.columns else "N/A",
        "Generators": df_pd["generator"].nunique() if "generator" in df_pd.columns else "N/A",
        "Labels (0/1)": dict(df_pd["label"].value_counts()) if "label" in df_pd.columns else "N/A",
    }

    for k, v in stats.items():
        logger.info(f"{k}: {v}")

    # Column distributions
    if "language" in df_pd.columns:
        lang_dist = df_pd["language"].value_counts().head(10).to_dict()
        logger.info(f"Top 10 Languages: {lang_dist}")
    
    if "generator" in df_pd.columns:
        gen_dist = df_pd["generator"].value_counts().head(10).to_dict()
        logger.info(f"Top 10 Generators: {gen_dist}")

    # Missing values
    missing = df_pd.isnull().sum()
    if missing.any():
        logger.warning(f"Found missing values:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found.")

    return df_pd

def analyze_tokenization(df_pd: pd.DataFrame, model_name: str, max_len: int, sample_size: int = 5000):
    logger.info(f"--- TOKENIZATION STATS (Sample size={sample_size}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Sample subset for speed
    if len(df_pd) > sample_size:
        sample_df = df_pd.sample(n=sample_size, random_state=42)
    else:
        sample_df = df_pd

    codes = sample_df["code"].astype(str).tolist()
    lengths = []
    
    for c in tqdm(codes, desc="Tokenizing"):
        tokens = tokenizer.encode(c, add_special_tokens=True)
        lengths.append(len(tokens))

    lengths = np.array(lengths)
    truncated = (lengths >= max_len).sum()
    perc_truncated = (truncated / len(lengths)) * 100

    logger.info(f"Avg tokens: {lengths.mean():.1f}")
    logger.info(f"Max tokens: {lengths.max()}")
    logger.info(f"Min tokens: {lengths.min()}")
    logger.info(f"p50: {np.percentile(lengths, 50):.1f} | p95: {np.percentile(lengths, 95):.1f}")
    
    if perc_truncated > 10:
        logger.warning(f"⚠ HIGH TRUNCATION: {perc_truncated:.1f}% samples exceed max_len={max_len}")
    else:
        logger.info(f"Truncated samples: {perc_truncated:.1f}%")

def inspect_batches(loader: DataLoader, num_batches: int = 3):
    logger.info(f"--- BATCH STRUCTURE INSPECTION (First {num_batches} batches) ---")
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        lang_ids = batch["language_id"].tolist()
        gen_ids = batch["generator_id"].tolist()
        labels = batch["label"].tolist()
        
        # 1. Distribution
        lang_counts = Counter(lang_ids)
        logger.info(f"Batch {i} Language counts: {dict(lang_counts)}")
        
        # 2. Block view (detect grouping vs mixing)
        # We look for runs of the same language
        runs = []
        if lang_ids:
            current_run = [lang_ids[0]]
            for l in lang_ids[1:]:
                if l == current_run[-1]:
                    current_run.append(l)
                else:
                    runs.append((current_run[0], len(current_run)))
                    current_run = [l]
            runs.append((current_run[0], len(current_run)))
        
        logger.info(f"Batch {i} Structure (Lang, Count): {runs}")
        
        # 3. Meta-learning split validation
        mt, mtest, test_lang = split_meta_language(batch)
        if mt is not None:
            mt_langs = Counter(mt["language_id"].tolist())
            mtest_langs = Counter(mtest["language_id"].tolist())
            logger.info(f"Batch {i} Meta-Split: SUCCESS")
            logger.info(f"  -> Meta-Train langs: {dict(mt_langs)}")
            logger.info(f"  -> Meta-Test lang ({test_lang}): {dict(mtest_langs)}")
            
            # Sanity check: meta-test should be exactly one language
            if len(mtest_langs) > 1:
                logger.error(f"  ⚠ BUG: Meta-test batch contains multiple languages! {dict(mtest_langs)}")
            
            # Check overlap
            overlap = set(mt_langs.keys()) & set(mtest_langs.keys())
            if overlap:
                logger.warning(f"  ⚠ OOD VIOLATION: Meta-test language {overlap} also present in Meta-train!")
        else:
            logger.warning(f"Batch {i} Meta-Split: FAILED (Less than 2 languages)")

        # 4. Label Balance check (m_per_lang balance)
        # Assuming k*m structure. Since shuffle is true, we check total balance.
        pos = sum(labels)
        neg = len(labels) - pos
        logger.info(f"  -> Labels: pos={pos}, neg={neg} (ratio={pos/len(labels):.2%})")

def main():
    parser = argparse.ArgumentParser(description="Dataset Debugging Tool")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train data")
    parser.add_argument("--val_csv",   type=str, default=None, help="Path to val data")
    parser.add_argument("--model_name", type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--max_len",   type=int, default=512)
    parser.add_argument("--k_langs",   type=int, default=6)
    parser.add_argument("--m_per_lang", type=int, default=16)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    logger.info("Initializing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds, val_ds, _ = build_datasets(
        train_file=args.train_csv,
        val_file=args.val_csv,
        tokenizer=tokenizer,
        max_length=args.max_len,
        seed=args.seed
    )

    # 1. Dataset Analysis
    df_train = analyze_dataset(train_ds, "Train")
    
    # 2. Tokenization Analysis
    analyze_tokenization(df_train, args.model_name, args.max_len)

    # 3. DataLoader behavior
    batch_size = args.k_langs * args.m_per_lang
    sampler = LanguageBalancedSampler(
        language_ids  = train_ds.language_ids_list,
        labels        = train_ds.labels_list,
        generator_ids = train_ds.generator_ids_list,
        k             = args.k_langs,
        m             = args.m_per_lang,
        seed          = args.seed
    )

    loader = DataLoader(
        train_ds,
        batch_sampler = sampler,
        collate_fn    = domain_collate_fn,
        num_workers   = 0  # 0 for debugging stability
    )

    inspect_batches(loader)

    logger.info("--- DEBUGGING COMPLETE ---")

if __name__ == "__main__":
    main()
