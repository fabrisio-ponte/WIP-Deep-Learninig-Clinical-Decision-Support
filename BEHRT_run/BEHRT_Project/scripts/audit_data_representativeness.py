#!/usr/bin/env python3
"""Audit whether the cleaned MIMIC-IV / CCSR splits are representative enough
to justify running the model and reporting metrics from it.

Reports (per split and cross-split):
  1. Row / patient counts and patient overlap across train/val/test (leakage check)
  2. Sequence-length distribution (visits per patient sample)
  3. Age distribution (min / median / max, plus histogram bucketed by decade)
  4. Label prevalence: top-K labels, tail behavior, % of labels with support < K
  5. Label coverage across splits: labels in train that never appear in val/test and vice versa
  6. Per-split label cardinality (avg number of positive labels per sample)

Prints a compact summary at the end tagging each check as OK / WARN / FAIL so it
is trivial to decide whether the current data is publication-quality.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = {
    "train": DATA_DIR / "train_nextvisit_ccsr_clean.parquet",
    "val": DATA_DIR / "val_nextvisit_ccsr_clean.parquet",
    "test": DATA_DIR / "test_nextvisit_ccsr_clean.parquet",
}
VOCAB_PATH = DATA_DIR / "vocab_ccsr_clean.pkl"


def as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def summarize_ages(ages_series: pd.Series):
    flat = []
    for row in ages_series:
        flat.extend([a for a in as_list(row) if a is not None])
    if not flat:
        return None
    arr = np.asarray(flat, dtype=float)
    return {
        "n_tokens": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def summarize_sequence_lengths(code_series: pd.Series):
    lengths = np.array([len(as_list(x)) for x in code_series])
    return {
        "min": int(lengths.min()),
        "median": int(np.median(lengths)),
        "mean": float(lengths.mean()),
        "p95": int(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }


def summarize_label_cardinality(label_series: pd.Series):
    per_row = np.array([len(as_list(x)) for x in label_series])
    return {
        "min_labels_per_sample": int(per_row.min()),
        "median_labels_per_sample": float(np.median(per_row)),
        "mean_labels_per_sample": float(per_row.mean()),
        "max_labels_per_sample": int(per_row.max()),
    }


def label_counts(label_series: pd.Series) -> Counter:
    c = Counter()
    for row in label_series:
        c.update(as_list(row))
    return c


def main():
    print("=" * 78)
    print("BEHRT CLEAN-DATA REPRESENTATIVENESS AUDIT (MIMIC-IV v3.1, CCSR)")
    print("=" * 78)

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    special = {"PAD", "SEP", "CLS", "MASK", "UNK"}
    label_vocab = [t for t in vocab["token2idx"].keys() if t not in special]
    print(f"\nVocab labels (excluding special tokens): {len(label_vocab)}")

    dfs = {}
    for name, path in SPLITS.items():
        dfs[name] = load_split(path)

    # --------------------- per-split summaries ---------------------
    warnings = []
    label_supports = {}

    for name, df in dfs.items():
        print(f"\n--- {name.upper()} ---")
        print(f"rows: {len(df):,}")
        has_patient = "patient" in df.columns or "patid" in df.columns
        pid_col = "patient" if "patient" in df.columns else ("patid" if "patid" in df.columns else None)
        if pid_col:
            n_patients = df[pid_col].nunique()
            print(f"unique patients ({pid_col}): {n_patients:,}")
            print(f"rows per patient: mean={len(df)/n_patients:.2f}")

        seq = summarize_sequence_lengths(df["code"])
        print(f"history length (tokens/sample): min={seq['min']} median={seq['median']} mean={seq['mean']:.1f} p95={seq['p95']} max={seq['max']}")
        if seq["min"] < 2:
            warnings.append(f"{name}: has samples with <2 history tokens (min={seq['min']})")

        card = summarize_label_cardinality(df["label"])
        print(f"labels per sample: min={card['min_labels_per_sample']} median={card['median_labels_per_sample']:.1f} mean={card['mean_labels_per_sample']:.2f} max={card['max_labels_per_sample']}")
        if card["min_labels_per_sample"] < 1:
            warnings.append(f"{name}: has samples with 0 labels")

        age = summarize_ages(df["age"])
        if age:
            print(f"age tokens (units as in vocab): n={age['n_tokens']:,} min={age['min']:.0f} p05={age['p05']:.0f} median={age['median']:.0f} p95={age['p95']:.0f} max={age['max']:.0f}")

        counts = label_counts(df["label"])
        label_supports[name] = counts
        n_unique = len(counts)
        top5 = counts.most_common(5)
        print(f"unique labels observed: {n_unique}")
        print(f"top-5 labels: {[(l, c) for l, c in top5]}")

        # tail behavior
        support_arr = np.array(sorted(counts.values(), reverse=True))
        n_lt_10 = int((support_arr < 10).sum())
        n_lt_50 = int((support_arr < 50).sum())
        n_lt_100 = int((support_arr < 100).sum())
        print(f"labels with support <10: {n_lt_10} | <50: {n_lt_50} | <100: {n_lt_100}")

        # Concentration
        total_label_tokens = int(support_arr.sum())
        top10_share = float(support_arr[:10].sum()) / total_label_tokens * 100
        print(f"top-10 labels account for {top10_share:.1f}% of all label tokens in {name}")
        if top10_share > 60:
            warnings.append(f"{name}: extreme label concentration — top-10 = {top10_share:.1f}% of positives")

    # --------------------- cross-split checks ---------------------
    print("\n--- CROSS-SPLIT PATIENT LEAKAGE ---")
    pid_col = "patient" if "patient" in dfs["train"].columns else ("patid" if "patid" in dfs["train"].columns else None)
    if pid_col:
        train_p = set(dfs["train"][pid_col].unique())
        val_p = set(dfs["val"][pid_col].unique())
        test_p = set(dfs["test"][pid_col].unique())
        print(f"train ∩ val patients: {len(train_p & val_p)}")
        print(f"train ∩ test patients: {len(train_p & test_p)}")
        print(f"val ∩ test patients: {len(val_p & test_p)}")
        if (train_p & val_p) or (train_p & test_p) or (val_p & test_p):
            warnings.append("Patient-level leakage detected across splits")
    else:
        print("(no patient column present — cannot check patient leakage)")
        warnings.append("No patient identifier column in splits (cannot verify patient-level split integrity)")

    print("\n--- LABEL COVERAGE ACROSS SPLITS ---")
    train_labels = set(label_supports["train"].keys())
    val_labels = set(label_supports["val"].keys())
    test_labels = set(label_supports["test"].keys())
    val_missing = val_labels - train_labels
    test_missing = test_labels - train_labels
    print(f"labels in val not in train: {len(val_missing)} {sorted(val_missing)[:10]}")
    print(f"labels in test not in train: {len(test_missing)} {sorted(test_missing)[:10]}")
    if test_missing:
        warnings.append(f"{len(test_missing)} labels appear in TEST but never in TRAIN (will contribute F1=0 mechanically)")

    # Distinguish tokens that also appear in HISTORY (legitimate input embeddings)
    # from tokens that appear NOWHERE in the data (dead vocab entries).
    all_history_tokens = set()
    for df in dfs.values():
        for row in df["code"]:
            all_history_tokens.update(as_list(row))

    label_unused = set(label_vocab) - (train_labels | val_labels | test_labels)
    history_only = label_unused & all_history_tokens
    dead_vocab = label_unused - all_history_tokens
    print(f"vocab tokens never used as a label: {len(label_unused)} (of {len(label_vocab)})")
    print(f"  - of these, appear only in history (input-only tokens, fine): {len(history_only)}")
    print(f"  - of these, appear nowhere at all (dead vocab entries): {len(dead_vocab)}")
    if len(dead_vocab) > 0:
        warnings.append(f"{len(dead_vocab)} vocab entries have zero support in labels AND history — classifier outputs those columns are dead weight")

    # --------------------- global support histogram ---------------------
    print("\n--- GLOBAL LABEL SUPPORT DISTRIBUTION (TRAIN) ---")
    train_supports = np.array(sorted(label_supports["train"].values(), reverse=True))
    buckets = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 100000)]
    for lo, hi in buckets:
        n = int(((train_supports >= lo) & (train_supports < hi)).sum())
        print(f"  support [{lo:>5}, {hi:>6}): {n:>4} labels")

    # --------------------- verdict ---------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if warnings:
        print("WARNINGS / RISKS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No structural warnings raised.")

    print("\nWhat metrics a training/eval run against these splits will actually measure:")
    print("  - Sample-wise APS  (multi-label ranking; robust to rare-class prevalence)")
    print("  - Sample-wise ROC-AUC (multi-label; less trustworthy with severe imbalance)")
    print("  - Per-disease F1 at 0.5 threshold (will still collapse to 0 for tail classes)")
    print("  - Macro/micro precision/recall (macro will be dominated by rare-class failures)")
    print("  - Per-class support (from this audit) tells you which per-disease numbers are trustable")


if __name__ == "__main__":
    main()
