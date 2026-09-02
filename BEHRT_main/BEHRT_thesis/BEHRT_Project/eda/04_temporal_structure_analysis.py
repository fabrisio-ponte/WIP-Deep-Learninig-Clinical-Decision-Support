#!/usr/bin/env python3
"""
EDA Step 4: Temporal Structure Analysis
Analyzes temporal spacing, sequence evolution, and code addition/removal patterns.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "eda" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "04_temporal_structure_analysis.json"

datasets = {
    "train": DATA_DIR / "train_nextvisit_ccsr_clean.parquet",
    "val": DATA_DIR / "val_nextvisit_ccsr_clean.parquet",
    "test": DATA_DIR / "test_nextvisit_ccsr_clean.parquet",
}


def quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
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


def to_years(month_value: float) -> float:
    return month_value / 12.0


def analyze_split(split_name: str, df: pd.DataFrame) -> dict:
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {split_name.upper()} TEMPORAL STRUCTURE")
    print(f"{'=' * 80}")

    # Per-sample derived arrays
    history_lengths = df["code"].apply(lambda x: len(x) if x is not None else 0).to_numpy(dtype=np.int64)
    label_lengths = df["label"].apply(lambda x: len(x) if x is not None else 0).to_numpy(dtype=np.int64)

    age_first = df["age"].apply(lambda x: int(x[0]) if len(x) > 0 else 0).to_numpy(dtype=np.int64)
    age_last = df["age"].apply(lambda x: int(x[-1]) if len(x) > 0 else 0).to_numpy(dtype=np.int64)
    history_span_months = age_last - age_first

    # Monotonicity and inter-event gaps from age sequences
    non_monotonic_rows = 0
    zero_gap_count = 0
    positive_gaps_all = []

    for age_seq in df["age"]:
        if len(age_seq) <= 1:
            continue
        diffs = np.diff(age_seq)
        if np.any(diffs < 0):
            non_monotonic_rows += 1
        zero_gap_count += int(np.sum(diffs == 0))
        positive_gaps = diffs[diffs > 0]
        if positive_gaps.size > 0:
            positive_gaps_all.extend(positive_gaps.tolist())

    positive_gaps_all = np.array(positive_gaps_all, dtype=np.int64)

    # Unique age points per sample as a proxy for distinct visit-time bins
    unique_age_counts = df["age"].apply(lambda x: len(set(x.tolist())) if len(x) > 0 else 0).to_numpy(dtype=np.int64)

    # Overlap between history and target labels: recurrence vs new codes
    recurrent_counts = []
    new_counts = []

    for _, row in df.iterrows():
        history_set = set(row["code"].tolist())
        next_set = set(row["label"].tolist())
        recurrent = len(history_set & next_set)
        new_only = len(next_set - history_set)
        recurrent_counts.append(recurrent)
        new_counts.append(new_only)

    recurrent_counts = np.array(recurrent_counts, dtype=np.int64)
    new_counts = np.array(new_counts, dtype=np.int64)
    total_target_codes = int(np.sum(label_lengths))
    total_recurrent = int(np.sum(recurrent_counts))
    total_new = int(np.sum(new_counts))

    recurrent_pct = 100.0 * total_recurrent / total_target_codes if total_target_codes > 0 else 0.0
    new_pct = 100.0 * total_new / total_target_codes if total_target_codes > 0 else 0.0

    # Longitudinal snapshot progression within patient
    progression_patient_count = 0
    progression_steps = 0
    progression_age_deltas = []
    progression_history_growth = []
    progression_added_code_counts = []
    progression_removed_code_counts = []

    for patid, group in df.groupby("patid"):
        if len(group) < 2:
            continue
        progression_patient_count += 1

        g = group.copy()
        g["end_age"] = g["age"].apply(lambda x: int(x[-1]) if len(x) > 0 else 0)
        g["history_len"] = g["code"].apply(lambda x: len(x) if x is not None else 0)
        g = g.sort_values(["end_age", "history_len"])  # deterministic progression order

        prev_age = None
        prev_codes = None
        prev_len = None

        for _, row in g.iterrows():
            current_age = int(row["end_age"])
            current_codes = set(row["code"].tolist())
            current_len = int(row["history_len"])

            if prev_age is not None:
                progression_steps += 1
                progression_age_deltas.append(current_age - prev_age)
                progression_history_growth.append(current_len - prev_len)
                progression_added_code_counts.append(len(current_codes - prev_codes))
                progression_removed_code_counts.append(len(prev_codes - current_codes))

            prev_age = current_age
            prev_codes = current_codes
            prev_len = current_len

    progression_age_deltas = np.array(progression_age_deltas, dtype=np.int64)
    progression_history_growth = np.array(progression_history_growth, dtype=np.int64)
    progression_added_code_counts = np.array(progression_added_code_counts, dtype=np.int64)
    progression_removed_code_counts = np.array(progression_removed_code_counts, dtype=np.int64)

    # Human-readable report
    print("\nSample-level temporal stats:")
    print(f"  Samples: {len(df):,}")
    print(f"  History span (months), median: {np.median(history_span_months):.1f}")
    print(f"  History span (years), median: {to_years(float(np.median(history_span_months))):.2f}")
    print(f"  Unique age points/sample, median: {np.median(unique_age_counts):.1f}")
    print(f"  Non-monotonic age sequences: {non_monotonic_rows:,}")

    if positive_gaps_all.size > 0:
        print(f"  Positive inter-event gap (months), median: {np.median(positive_gaps_all):.1f}")
    print(f"  Zero-month consecutive gaps: {zero_gap_count:,}")

    print("\nRecurrence vs new target codes:")
    print(f"  Recurrent targets: {total_recurrent:,} ({recurrent_pct:.2f}%)")
    print(f"  New targets: {total_new:,} ({new_pct:.2f}%)")

    print("\nLongitudinal progression stats:")
    print(f"  Patients with >=2 snapshots: {progression_patient_count:,}")
    print(f"  Snapshot-to-snapshot transitions: {progression_steps:,}")
    if progression_steps > 0:
        print(f"  End-age delta (months), median: {np.median(progression_age_deltas):.1f}")
        print(f"  History growth per step, median: {np.median(progression_history_growth):.1f}")
        print(f"  Added codes per step, median: {np.median(progression_added_code_counts):.1f}")
        print(f"  Removed codes per step, median: {np.median(progression_removed_code_counts):.1f}")

    return {
        "samples": int(len(df)),
        "history_length": quantiles(history_lengths.astype(float)),
        "label_length": quantiles(label_lengths.astype(float)),
        "history_span_months": quantiles(history_span_months.astype(float)),
        "history_span_years": {
            k: to_years(v) for k, v in quantiles(history_span_months.astype(float)).items()
        },
        "unique_age_points_per_sample": quantiles(unique_age_counts.astype(float)),
        "age_sequence_quality": {
            "non_monotonic_rows": int(non_monotonic_rows),
            "non_monotonic_pct": float(100.0 * non_monotonic_rows / len(df)),
            "zero_month_consecutive_gaps": int(zero_gap_count),
        },
        "inter_event_positive_gaps_months": quantiles(positive_gaps_all.astype(float)),
        "recurrence_vs_new_targets": {
            "total_target_codes": total_target_codes,
            "recurrent_count": total_recurrent,
            "new_count": total_new,
            "recurrent_pct": float(recurrent_pct),
            "new_pct": float(new_pct),
            "recurrent_per_sample": quantiles(recurrent_counts.astype(float)),
            "new_per_sample": quantiles(new_counts.astype(float)),
        },
        "longitudinal_progression": {
            "patients_with_multiple_snapshots": int(progression_patient_count),
            "snapshot_transitions": int(progression_steps),
            "end_age_delta_months": quantiles(progression_age_deltas.astype(float)),
            "end_age_delta_years": {
                k: to_years(v) for k, v in quantiles(progression_age_deltas.astype(float)).items()
            },
            "history_growth_per_transition": quantiles(progression_history_growth.astype(float)),
            "codes_added_per_transition": quantiles(progression_added_code_counts.astype(float)),
            "codes_removed_per_transition": quantiles(progression_removed_code_counts.astype(float)),
            "transitions_with_removals": int(np.sum(progression_removed_code_counts > 0)) if progression_removed_code_counts.size > 0 else 0,
            "transitions_with_removals_pct": float(
                100.0 * np.mean(progression_removed_code_counts > 0)
            ) if progression_removed_code_counts.size > 0 else 0.0,
        },
    }


def main() -> None:
    print("=" * 80)
    print("TEMPORAL STRUCTURE ANALYSIS")
    print("=" * 80)

    all_results = {
        "step": 4,
        "title": "Temporal Structure Analysis",
        "datasets": {},
    }

    for split_name, path in datasets.items():
        df = pd.read_parquet(path)
        all_results["datasets"][split_name] = analyze_split(split_name, df)

    # Cross-split rollup
    recurrences = {
        split: all_results["datasets"][split]["recurrence_vs_new_targets"]["recurrent_pct"]
        for split in datasets.keys()
    }
    new_rates = {
        split: all_results["datasets"][split]["recurrence_vs_new_targets"]["new_pct"]
        for split in datasets.keys()
    }
    all_results["cross_split_summary"] = {
        "recurrent_pct_by_split": recurrences,
        "new_pct_by_split": new_rates,
        "max_recurrent_pct_gap": float(max(recurrences.values()) - min(recurrences.values())),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Temporal structure analysis saved to: {OUTPUT_PATH}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
