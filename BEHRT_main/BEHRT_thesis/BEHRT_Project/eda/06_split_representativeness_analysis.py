#!/usr/bin/env python3
"""
EDA Step 6: Split Representativeness Analysis
Checks whether train/val/test splits are comparable in patient mix, age distribution,
sequence lengths, and label support.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "eda" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "06_split_representativeness_analysis.json"

DATASETS = {
    "train": DATA_DIR / "train_nextvisit_ccsr_clean.parquet",
    "val": DATA_DIR / "val_nextvisit_ccsr_clean.parquet",
    "test": DATA_DIR / "test_nextvisit_ccsr_clean.parquet",
}


def summarize_numeric(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def summary_for_split(split_name: str, df: pd.DataFrame) -> dict:
    history_lengths = df["code"].apply(lambda x: len(x) if x is not None else 0).to_numpy(dtype=float)
    label_lengths = df["label"].apply(lambda x: len(x) if x is not None else 0).to_numpy(dtype=float)
    age_last = df["age"].apply(lambda x: x[-1] / 12.0 if len(x) > 0 else 0.0).to_numpy(dtype=float)

    label_counter = Counter()
    for labels in df["label"]:
        for label in labels:
            label_counter[label] += 1

    top_labels = label_counter.most_common(20)
    top_label_summary = [
        {
            "label": label,
            "count": int(count),
            "fraction_of_samples": float(count / len(df)),
        }
        for label, count in top_labels
    ]

    return {
        "samples": int(len(df)),
        "unique_patients": int(df["patid"].nunique()),
        "history_length": summarize_numeric(history_lengths),
        "label_length": summarize_numeric(label_lengths),
        "age_years": summarize_numeric(age_last),
        "top_labels": top_label_summary,
        "label_support": {
            "unique_labels": int(len(label_counter)),
            "max_label_count": int(max(label_counter.values())) if label_counter else 0,
            "min_label_count": int(min(label_counter.values())) if label_counter else 0,
            "mean_label_count": float(np.mean(list(label_counter.values()))) if label_counter else 0.0,
        },
    }


def main() -> None:
    print("=" * 80)
    print("SPLIT REPRESENTATIVENESS ANALYSIS")
    print("=" * 80)

    split_summaries = {}
    for split_name, path in DATASETS.items():
        df = pd.read_parquet(path)
        split_summaries[split_name] = summary_for_split(split_name, df)
        print(f"\n--- {split_name.upper()} ---")
        print(f"Samples: {split_summaries[split_name]['samples']:,}")
        print(f"Patients: {split_summaries[split_name]['unique_patients']:,}")
        print(f"History length median: {split_summaries[split_name]['history_length']['median']:.1f}")
        print(f"Label length median: {split_summaries[split_name]['label_length']['median']:.1f}")
        print(f"Age median (years): {split_summaries[split_name]['age_years']['median']:.1f}")
        print("Top labels:")
        for item in split_summaries[split_name]["top_labels"][:5]:
            print(f"  {item['label']}: count={item['count']:,}, sample_frac={item['fraction_of_samples']*100:.2f}%")

    # Cross-split comparison on the most important distributional quantities
    label_names = sorted({label for summary in split_summaries.values() for label in [x["label"] for x in summary["top_labels"]]})
    top_label_rows = {}
    for label in label_names:
        vals = {}
        for split_name, summary in split_summaries.items():
            match = next((item for item in summary["top_labels"] if item["label"] == label), None)
            vals[split_name] = match["fraction_of_samples"] if match is not None else 0.0
        top_label_rows[label] = vals

    top_diff_by_label = []
    for label, vals in top_label_rows.items():
        values = list(vals.values())
        if len(values) < 2:
            continue
        max_val = max(values)
        min_val = min(values)
        top_diff_by_label.append({
            "label": label,
            "max_minus_min_fraction": float(max_val - min_val),
            "values": vals,
        })
    top_diff_by_label.sort(key=lambda x: x["max_minus_min_fraction"], reverse=True)

    # Compare age and history statistics across splits
    age_medians = {split: summary["age_years"]["median"] for split, summary in split_summaries.items()}
    history_medians = {split: summary["history_length"]["median"] for split, summary in split_summaries.items()}
    label_medians = {split: summary["label_length"]["median"] for split, summary in split_summaries.items()}

    results = {
        "step": 6,
        "title": "Split Representativeness Analysis",
        "split_summaries": split_summaries,
        "cross_split_differences": {
            "age_median_by_split": age_medians,
            "history_length_median_by_split": history_medians,
            "label_length_median_by_split": label_medians,
            "largest_top_label_drift": top_diff_by_label[:20],
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Split analysis saved to: {OUTPUT_PATH}")
    print(f"{'=' * 80}")

    # Human-readable assessment
    print("\nAssessment:")
    age_gap = max(age_medians.values()) - min(age_medians.values())
    history_gap = max(history_medians.values()) - min(history_medians.values())
    label_gap = max(label_medians.values()) - min(label_medians.values())
    print(f"  Age median difference: {age_gap:.2f} years")
    print(f"  History median difference: {history_gap:.2f} codes")
    print(f"  Label count median difference: {label_gap:.2f} codes")

    if age_gap < 2 and history_gap < 5 and label_gap < 1:
        print("  Interpretation: splits are broadly comparable on core distributional features.")
    else:
        print("  Interpretation: some distributional drift exists across splits; review before interpreting performance comparisons.")


if __name__ == "__main__":
    main()
