#!/usr/bin/env python3
"""Finalize the BEHRT clean splits so the training data is publication-ready.

Applies three guards on top of the existing `*_ccsr_clean.parquet` files and
`vocab_ccsr_clean.pkl`, then overwrites them in place:

    Guard 1: drop rows with fewer than MIN_HISTORY_TOKENS history tokens
             (a sequence of length 1 cannot form a next-visit context).

    Guard 2: strip any label from val/test that never appears in the TRAIN label
             set. Rows whose labels collapse to zero after this step are dropped.
             Rationale: labels with 0 train support mechanically contribute F1=0
             at eval time and inflate the "hard" label count in a misleading way.

    Guard 3: prune vocab entries that are never observed anywhere across
             (train + val + test) label + history token space. Those slots are
             dead classifier outputs and only waste capacity.

The raw uncleaned parquets (`*_nextvisit_ccsr.parquet`) remain untouched, so
this operation is fully re-derivable by rerunning the cleaner + this script.
"""

from __future__ import annotations

import pickle
import shutil
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
BACKUP_DIR = DATA_DIR / "pre_finalize_backup"

MIN_HISTORY_TOKENS = 2
SPECIAL_TOKENS = ("PAD", "CLS", "SEP", "MASK", "UNK")


def as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def backup_originals():
    BACKUP_DIR.mkdir(exist_ok=True)
    for path in list(SPLITS.values()) + [VOCAB_PATH]:
        dest = BACKUP_DIR / path.name
        if not dest.exists():
            shutil.copy2(path, dest)
    print(f"Originals backed up to: {BACKUP_DIR}")


def load_vocab() -> dict:
    with open(VOCAB_PATH, "rb") as f:
        return pickle.load(f)


def save_vocab(vocab: dict):
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)


def apply_guards():
    print("=" * 78)
    print("FINALIZE SPLITS — apply publication guards")
    print("=" * 78)

    backup_originals()

    # Load all splits once
    dfs = {name: pd.read_parquet(path) for name, path in SPLITS.items()}

    original_rows = {name: len(df) for name, df in dfs.items()}
    stats = {name: {"original_rows": original_rows[name]} for name in dfs}

    # --------------------- Guard 1: drop short-history rows ---------------------
    print(f"\n[Guard 1] Drop rows with <{MIN_HISTORY_TOKENS} history tokens")
    for name, df in dfs.items():
        seq_lens = df["code"].apply(lambda x: len(as_list(x)))
        mask = seq_lens >= MIN_HISTORY_TOKENS
        dropped = int((~mask).sum())
        dfs[name] = df[mask].reset_index(drop=True)
        stats[name]["dropped_short_history"] = dropped
        print(f"  {name}: dropped {dropped} rows (kept {len(dfs[name]):,})")

    # --------------------- Guard 2: strip val/test labels not in train ---------------------
    print("\n[Guard 2] Strip labels from val/test that never appear in TRAIN labels")
    train_label_set = set()
    for row in dfs["train"]["label"]:
        train_label_set.update(as_list(row))
    print(f"  train label vocabulary (observed): {len(train_label_set)} unique labels")

    for name in ("val", "test"):
        df = dfs[name]
        stripped_count = 0

        def _strip(labels):
            nonlocal stripped_count
            keep = [lab for lab in as_list(labels) if lab in train_label_set]
            stripped_count += len(as_list(labels)) - len(keep)
            return keep

        df = df.copy()
        df["label"] = df["label"].apply(_strip)
        before = len(df)
        df = df[df["label"].apply(len) > 0].reset_index(drop=True)
        after = len(df)
        dfs[name] = df
        stats[name]["stripped_label_tokens"] = stripped_count
        stats[name]["dropped_after_strip"] = before - after
        print(f"  {name}: stripped {stripped_count} label tokens, dropped {before - after} newly-empty rows")

    # --------------------- Guard 3: prune unused vocab entries ---------------------
    print("\n[Guard 3] Prune vocab entries never observed in any split")
    vocab = load_vocab()
    old_token2idx = vocab["token2idx"]
    old_size = len(old_token2idx)

    observed = set()
    for df in dfs.values():
        for row in df["code"]:
            observed.update(as_list(row))
        for row in df["label"]:
            observed.update(as_list(row))

    # Preserve special tokens with indices 0..N-1 in original order
    new_tokens = []
    seen = set()
    for tok in SPECIAL_TOKENS:
        if tok in old_token2idx and tok not in seen:
            new_tokens.append(tok)
            seen.add(tok)

    # Then append every non-special token that is actually observed, in the
    # original vocab order (stable, reproducible).
    ordered_original = sorted(old_token2idx.items(), key=lambda kv: kv[1])
    for tok, _idx in ordered_original:
        if tok in SPECIAL_TOKENS or tok in seen:
            continue
        if tok in observed:
            new_tokens.append(tok)
            seen.add(tok)

    new_token2idx = {tok: i for i, tok in enumerate(new_tokens)}
    new_idx2token = {i: tok for tok, i in new_token2idx.items()}
    new_vocab = {"token2idx": new_token2idx, "idx2token": new_idx2token}

    pruned = old_size - len(new_token2idx)
    print(f"  vocab: {old_size} -> {len(new_token2idx)} (pruned {pruned} unused tokens)")

    # Sanity: every observed token must be in the new vocab
    missing_history = observed - set(new_token2idx.keys())
    if missing_history:
        raise SystemExit(
            f"FATAL: {len(missing_history)} tokens observed in data are missing "
            f"from the pruned vocab (sample: {list(missing_history)[:5]})"
        )

    save_vocab(new_vocab)

    # --------------------- write back splits ---------------------
    print("\nWriting cleaned splits back to disk")
    for name, path in SPLITS.items():
        dfs[name].to_parquet(path)
        stats[name]["final_rows"] = len(dfs[name])
        change = stats[name]["final_rows"] - stats[name]["original_rows"]
        print(f"  {name}: {stats[name]['original_rows']:,} -> {stats[name]['final_rows']:,} ({change:+,})")

    # --------------------- summary ---------------------
    print("\n" + "=" * 78)
    print("FINALIZE SUMMARY")
    print("=" * 78)
    print(f"vocab size: {old_size} -> {len(new_token2idx)}")
    print(f"train label vocabulary (used in Guard 2): {len(train_label_set)}")
    for name in ("train", "val", "test"):
        s = stats[name]
        print(
            f"  {name}: rows {s['original_rows']:,} -> {s['final_rows']:,} | "
            f"short-history dropped: {s.get('dropped_short_history', 0)} | "
            f"labels stripped: {s.get('stripped_label_tokens', 0)} | "
            f"empty-after-strip dropped: {s.get('dropped_after_strip', 0)}"
        )

    print("\nBackups preserved under: data/processed/pre_finalize_backup/")
    print("Re-run scripts/verify_pipeline_contract.py and scripts/audit_data_representativeness.py to confirm.")


if __name__ == "__main__":
    apply_guards()
