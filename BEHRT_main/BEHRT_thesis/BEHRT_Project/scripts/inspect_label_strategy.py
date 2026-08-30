#!/usr/bin/env python3
"""Inspect the cleaned NextVisit label space and candidate subset strategies.

This script summarizes the current train/validation/test label structure and
prints a few label-reduction candidates that are useful for the next experiment:

- top-K labels by training prevalence
- labels with minimum training prevalence thresholds

It also reports how much label mass and how many samples would remain after
each cutoff, which helps decide whether the label space is too sparse.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def as_label_list(value) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    return [str(value)]


def count_labels(series: pd.Series) -> Counter:
    counts: Counter = Counter()
    for labels in series:
        counts.update(as_label_list(labels))
    return counts


def row_label_counts(series: pd.Series) -> pd.Series:
    return series.map(lambda value: len(as_label_list(value)))


def summarize_split(name: str, df: pd.DataFrame, pid_col: str) -> None:
    patient_count = df[pid_col].nunique() if pid_col in df.columns else 0
    label_counts = row_label_counts(df["label"])
    code_counts = row_label_counts(df["code"])

    print(f"\n{name.upper()} SPLIT")
    print(f"  rows: {len(df):,}")
    if pid_col in df.columns:
        print(f"  patients: {patient_count:,}")
    print(f"  code length mean/p95/max: {code_counts.mean():.2f} / {int(code_counts.quantile(0.95))} / {int(code_counts.max())}")
    print(f"  label count mean/p95/max: {label_counts.mean():.2f} / {int(label_counts.quantile(0.95))} / {int(label_counts.max())}")


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def describe_candidate(name: str, keep_labels: set[str], train_df: pd.DataFrame) -> None:
    train_total_labels = 0
    kept_label_tokens = 0
    kept_rows = 0
    kept_per_row = []

    for labels in train_df["label"]:
        label_list = as_label_list(labels)
        train_total_labels += len(label_list)
        retained = [label for label in label_list if label in keep_labels]
        kept_label_tokens += len(retained)
        if retained:
            kept_rows += 1
            kept_per_row.append(len(retained))

    label_fraction = kept_label_tokens / max(1, train_total_labels)
    row_fraction = kept_rows / max(1, len(train_df))
    mean_kept = float(np.mean(kept_per_row)) if kept_per_row else 0.0

    print(f"\n{name}")
    print(f"  labels kept: {len(keep_labels):,}")
    print(f"  train label-token coverage: {format_pct(label_fraction)}")
    print(f"  train row coverage: {format_pct(row_fraction)}")
    print(f"  mean retained labels per kept row: {mean_kept:.2f}")


def main() -> None:
    train_path = DATA_DIR / "train_nextvisit_ccsr_clean.parquet"
    val_path = DATA_DIR / "val_nextvisit_ccsr_clean.parquet"
    test_path = DATA_DIR / "test_nextvisit_ccsr_clean.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    pid_col = "patid" if "patid" in train_df.columns else train_df.columns[0]

    print("CLEAN NEXTVISIT LABEL INSPECTION")
    print(f"train/val/test rows: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    if pid_col in train_df.columns:
        print(
            "patient overlaps:",
            len(set(train_df[pid_col]) & set(val_df[pid_col])),
            len(set(train_df[pid_col]) & set(test_df[pid_col])),
            len(set(val_df[pid_col]) & set(test_df[pid_col])),
        )

    summarize_split("train", train_df, pid_col)
    summarize_split("val", val_df, pid_col)
    summarize_split("test", test_df, pid_col)

    train_counts = count_labels(train_df["label"])
    val_counts = count_labels(val_df["label"])
    test_counts = count_labels(test_df["label"])

    total_train_tokens = sum(train_counts.values())
    total_val_tokens = sum(val_counts.values())
    total_test_tokens = sum(test_counts.values())
    labels = sorted(train_counts.items(), key=lambda item: (-item[1], item[0]))

    print("\nTRAIN LABEL DISTRIBUTION")
    print(f"  unique labels: {len(labels):,}")
    prevalence_values = [count / len(train_df) for count in train_counts.values()]
    print(
        "  prevalence min/median/max:",
        format_pct(min(prevalence_values)),
        format_pct(float(np.median(prevalence_values))),
        format_pct(max(prevalence_values)),
    )
    print(f"  labels below 1% prevalence: {sum(value < 0.01 for value in prevalence_values):,}")
    print(f"  labels below 0.1% prevalence: {sum(value < 0.001 for value in prevalence_values):,}")

    print("\nTOP 20 TRAIN LABELS")
    for label, count in labels[:20]:
        train_prev = count / len(train_df)
        val_prev = val_counts.get(label, 0) / len(val_df)
        test_prev = test_counts.get(label, 0) / len(test_df)
        print(f"  {label:<16} train={format_pct(train_prev):>8} val={format_pct(val_prev):>8} test={format_pct(test_prev):>8}")

    print("\nCANDIDATE TOP-K SUBSETS")
    for k in [25, 50, 100, 150, 200]:
        keep = {label for label, _ in labels[: min(k, len(labels))]}
        describe_candidate(f"  top-{k}", keep, train_df)

    print("\nCANDIDATE MIN-PREVALENCE SUBSETS")
    for threshold in [0.001, 0.005, 0.01, 0.02]:
        keep = {
            label
            for label, count in train_counts.items()
            if (count / len(train_df)) >= threshold
        }
        describe_candidate(f"  prevalence >= {threshold * 100:.1f}%", keep, train_df)

    print("\nLABEL MASS SUMMARY")
    print(f"  train tokens: {total_train_tokens:,}")
    print(f"  val tokens:   {total_val_tokens:,}")
    print(f"  test tokens:  {total_test_tokens:,}")


if __name__ == "__main__":
    main()