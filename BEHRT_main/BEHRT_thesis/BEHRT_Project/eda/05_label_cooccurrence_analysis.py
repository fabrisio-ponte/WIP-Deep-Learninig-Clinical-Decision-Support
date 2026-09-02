#!/usr/bin/env python3
"""
EDA Step 5: Label Co-occurrence Analysis
Examines which diagnoses tend to occur together in the same next-visit target set.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "eda" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "05_label_cooccurrence_analysis.json"

DATASETS = {
    "train": DATA_DIR / "train_nextvisit_ccsr_clean.parquet",
    "val": DATA_DIR / "val_nextvisit_ccsr_clean.parquet",
    "test": DATA_DIR / "test_nextvisit_ccsr_clean.parquet",
}

TOP_K = 30


def analyze_split(split_name: str, df: pd.DataFrame) -> dict:
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {split_name.upper()} CO-OCCURRENCE PATTERNS")
    print(f"{'=' * 80}")

    label_counts = Counter()
    pair_counts = Counter()
    sample_count = 0

    for _, row in df.iterrows():
        labels = set(row["label"].tolist())
        if not labels:
            continue
        sample_count += 1
        for label in labels:
            label_counts[label] += 1
        label_list = sorted(labels)
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                a, b = label_list[i], label_list[j]
                pair_counts[(a, b)] += 1

    top_labels = [label for label, _ in label_counts.most_common(TOP_K)]
    top_label_set = set(top_labels)

    top_label_stats = [
        {
            "label": label,
            "count": int(label_counts[label]),
            "sample_fraction": float(label_counts[label] / sample_count) if sample_count > 0 else 0.0,
        }
        for label in top_labels
    ]

    pair_records = []
    for (a, b), count in pair_counts.most_common(200):
        if a not in top_label_set or b not in top_label_set:
            continue
        pair_records.append(
            {
                "label_a": a,
                "label_b": b,
                "count": int(count),
                "sample_fraction": float(count / sample_count) if sample_count > 0 else 0.0,
                "p_a_given_b": float(count / label_counts[a]) if label_counts[a] > 0 else 0.0,
                "p_b_given_a": float(count / label_counts[b]) if label_counts[b] > 0 else 0.0,
                "jaccard": float(count / (label_counts[a] + label_counts[b] - count)) if (label_counts[a] + label_counts[b] - count) > 0 else 0.0,
            }
        )

    print(f"\nSamples analyzed: {sample_count:,}")
    print(f"Unique labels observed: {len(label_counts):,}")
    print(f"Top {TOP_K} labels by frequency:")
    for item in top_label_stats[:10]:
        print(f"  {item['label']}: count={item['count']:,}, fraction={item['sample_fraction'] * 100:.2f}%")

    print(f"\nTop co-occurrence pairs:")
    for item in pair_records[:10]:
        print(
            f"  {item['label_a']} + {item['label_b']} | count={item['count']:,}, "
            f"frac={item['sample_fraction'] * 100:.3f}%, p(a|b)={item['p_a_given_b']:.3f}"
        )

    return {
        "samples_analyzed": int(sample_count),
        "unique_labels_observed": int(len(label_counts)),
        "top_labels": top_label_stats,
        "top_cooccurrence_pairs": pair_records[:50],
        "pair_count_total": int(len(pair_counts)),
    }


def main() -> None:
    print("=" * 80)
    print("LABEL CO-OCCURRENCE ANALYSIS")
    print("=" * 80)

    results = {"step": 5, "title": "Label Co-occurrence Analysis", "datasets": {}}

    for split_name, path in DATASETS.items():
        df = pd.read_parquet(path)
        results["datasets"][split_name] = analyze_split(split_name, df)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Co-occurrence analysis saved to: {OUTPUT_PATH}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
