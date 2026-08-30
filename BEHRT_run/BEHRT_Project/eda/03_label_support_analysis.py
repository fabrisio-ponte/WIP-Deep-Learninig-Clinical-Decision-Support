#!/usr/bin/env python3
"""
EDA Step 3: Label Support & Class Imbalance Analysis
Analyzes disease frequency, class imbalance, and label distribution.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path("data/processed")
VOCAB_PATH = DATA_DIR / "vocab_ccsr_clean.pkl"
TRAIN_PATH = DATA_DIR / "train_nextvisit_ccsr_clean.parquet"
VAL_PATH = DATA_DIR / "val_nextvisit_ccsr_clean.parquet"
TEST_PATH = DATA_DIR / "test_nextvisit_ccsr_clean.parquet"
OUTPUT_DIR = Path("eda/results")
OUTPUT_PATH = OUTPUT_DIR / "03_label_support_analysis.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LABEL SUPPORT & CLASS IMBALANCE ANALYSIS")
print("=" * 80)

# Load vocabulary
print("\nLoading vocabulary...")
import pickle
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

# idx2token contains: {0: 'PAD', 1: 'CLS', ..., 5+: disease codes}
idx2token = vocab['idx2token']
# Filter to get only disease codes (skip special tokens 0-4)
label_vocab = {k: v for k, v in idx2token.items() if k >= 5}
num_labels_vocab = len(label_vocab)
print(f"Total disease labels in vocabulary: {num_labels_vocab}")

# Load datasets
print("\nLoading datasets...")
train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"Train samples: {len(train_df):,}")
print(f"Val samples: {len(val_df):,}")
print(f"Test samples: {len(test_df):,}")

results = {
    "total_labels_vocab": num_labels_vocab,
    "datasets": {}
}

# Analyze each split
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {split_name.upper()} SPLIT")
    print(f"{'=' * 80}")
    
    # Count occurrences of each label (labels are stored as CCSR code strings)
    label_counts = defaultdict(int)
    total_samples = len(df)
    
    for idx, row in df.iterrows():
        labels = row['label']  # array of CCSR code strings
        for label in labels:
            label_counts[label] += 1
    
    # Convert to sorted list
    label_stats = []
    for label_code, count in label_counts.items():
        frequency = count / total_samples
        label_stats.append({
            "label_code": label_code,
            "count": int(count),
            "frequency": float(frequency)
        })
    
    # Sort by frequency (most common first)
    label_stats.sort(key=lambda x: x['count'], reverse=True)
    
    # Overall statistics
    counts = [s['count'] for s in label_stats]
    frequencies = [s['frequency'] for s in label_stats]
    
    # Get actual number of unique labels in this split
    num_unique_labels = len(label_stats)
    
    print(f"\nLabel Support Statistics:")
    print(f"  Total unique labels appearing: {num_unique_labels}")
    print(f"  Labels in vocabulary: {num_labels_vocab}")
    print(f"  Labels never appearing: {num_labels_vocab - num_unique_labels}")
    
    print(f"\nLabel frequency (% of samples):")
    print(f"  Min: {min(frequencies)*100:.2f}%")
    print(f"  25th percentile: {np.percentile(frequencies, 25)*100:.2f}%")
    print(f"  Median: {np.median(frequencies)*100:.2f}%")
    print(f"  75th percentile: {np.percentile(frequencies, 75)*100:.2f}%")
    print(f"  Max: {max(frequencies)*100:.2f}%")
    print(f"  Mean: {np.mean(frequencies)*100:.2f}%")
    
    print(f"\nLabel count distribution:")
    print(f"  Min: {min(counts):,}")
    print(f"  25th percentile: {int(np.percentile(counts, 25)):,}")
    print(f"  Median: {int(np.median(counts)):,}")
    print(f"  75th percentile: {int(np.percentile(counts, 75)):,}")
    print(f"  Max: {max(counts):,}")
    print(f"  Mean: {np.mean(counts):.1f}")
    
    # Categorize by support level
    ultra_rare = [s for s in label_stats if s['count'] < 10]
    very_rare = [s for s in label_stats if 10 <= s['count'] < 100]
    rare = [s for s in label_stats if 100 <= s['count'] < 1000]
    common = [s for s in label_stats if 1000 <= s['count'] < 10000]
    very_common = [s for s in label_stats if s['count'] >= 10000]
    
    print(f"\nSupport level distribution:")
    print(f"  Ultra-rare (<10 samples):      {len(ultra_rare):4d} ({len(ultra_rare)/len(label_stats)*100:5.1f}%)")
    print(f"  Very rare (10-99 samples):     {len(very_rare):4d} ({len(very_rare)/len(label_stats)*100:5.1f}%)")
    print(f"  Rare (100-999 samples):        {len(rare):4d} ({len(rare)/len(label_stats)*100:5.1f}%)")
    print(f"  Common (1,000-9,999 samples):  {len(common):4d} ({len(common)/len(label_stats)*100:5.1f}%)")
    print(f"  Very common (≥10,000 samples): {len(very_common):4d} ({len(very_common)/len(label_stats)*100:5.1f}%)")
    
    # Top 10 most common diseases
    print(f"\n{'Top 10 Most Common Diseases:':^80}")
    print(f"{'Rank':<6} {'Count':>8} {'Freq':>8} {'Disease Code':<55}")
    print("-" * 80)
    for i, stat in enumerate(label_stats[:10], 1):
        print(f"{i:<6} {stat['count']:8,} {stat['frequency']*100:7.2f}% {stat['label_code'][:53]}")
    
    # Bottom 10 rarest diseases
    print(f"\n{'Bottom 10 Rarest Diseases:':^80}")
    print(f"{'Rank':<6} {'Count':>8} {'Freq':>8} {'Disease Code':<55}")
    print("-" * 80)
    for i, stat in enumerate(label_stats[-10:], len(label_stats)-9):
        print(f"{i:<6} {stat['count']:8,} {stat['frequency']*100:7.2f}% {stat['label_code'][:53]}")
    
    # Class imbalance ratio
    max_freq = max(frequencies)
    min_freq = min(frequencies)
    imbalance_ratio = max_freq / min_freq
    
    print(f"\nClass Imbalance:")
    print(f"  Most common disease frequency: {max_freq*100:.2f}%")
    print(f"  Rarest disease frequency: {min_freq*100:.2f}%")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Positive/negative ratio
    total_possible_labels = total_samples * num_unique_labels
    total_positive_labels = sum(counts)
    total_negative_labels = total_possible_labels - total_positive_labels
    positive_ratio = total_positive_labels / total_possible_labels
    
    print(f"\nOverall Label Density:")
    print(f"  Total possible label slots: {total_possible_labels:,}")
    print(f"  Positive labels: {total_positive_labels:,}")
    print(f"  Negative labels: {total_negative_labels:,}")
    print(f"  Positive ratio: {positive_ratio*100:.2f}%")
    print(f"  Negative ratio: {(1-positive_ratio)*100:.2f}%")
    print(f"  Sparsity (negative:positive): {(1-positive_ratio)/positive_ratio:.1f}:1")
    
    # Store results
    results["datasets"][split_name] = {
        "total_samples": int(total_samples),
        "unique_labels_appearing": num_unique_labels,
        "labels_never_appearing": num_labels_vocab - num_unique_labels,
        "frequency_stats": {
            "min": float(min(frequencies)),
            "q25": float(np.percentile(frequencies, 25)),
            "median": float(np.median(frequencies)),
            "q75": float(np.percentile(frequencies, 75)),
            "max": float(max(frequencies)),
            "mean": float(np.mean(frequencies))
        },
        "count_stats": {
            "min": int(min(counts)),
            "q25": int(np.percentile(counts, 25)),
            "median": int(np.median(counts)),
            "q75": int(np.percentile(counts, 75)),
            "max": int(max(counts)),
            "mean": float(np.mean(counts))
        },
        "support_distribution": {
            "ultra_rare_lt10": len(ultra_rare),
            "very_rare_10_99": len(very_rare),
            "rare_100_999": len(rare),
            "common_1k_9k": len(common),
            "very_common_gte10k": len(very_common)
        },
        "class_imbalance_ratio": float(imbalance_ratio),
        "label_density": {
            "positive_ratio": float(positive_ratio),
            "negative_ratio": float(1 - positive_ratio),
            "sparsity_ratio": float((1-positive_ratio)/positive_ratio)
        },
        "top_10_diseases": label_stats[:10],
        "bottom_10_diseases": label_stats[-10:],
        "all_label_stats": label_stats
    }

# Save results
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 80}")
print(f"Label support analysis saved to: {OUTPUT_PATH}")
print(f"{'=' * 80}")

# Cross-split comparison
print(f"\n{'=' * 80}")
print("CROSS-SPLIT LABEL COMPARISON")
print(f"{'=' * 80}")

train_labels = set(s['label_code'] for s in results['datasets']['train']['all_label_stats'])
val_labels = set(s['label_code'] for s in results['datasets']['val']['all_label_stats'])
test_labels = set(s['label_code'] for s in results['datasets']['test']['all_label_stats'])

print(f"\nLabel coverage:")
print(f"  Labels in train: {len(train_labels)}")
print(f"  Labels in val: {len(val_labels)}")
print(f"  Labels in test: {len(test_labels)}")
print(f"  Labels in val but not train: {len(val_labels - train_labels)}")
print(f"  Labels in test but not train: {len(test_labels - train_labels)}")

if len(val_labels - train_labels) > 0:
    print(f"  ⚠ Val contains unseen labels (will always predict as negative)")
if len(test_labels - train_labels) > 0:
    print(f"  ⚠ Test contains unseen labels (will always predict as negative)")
if len(val_labels - train_labels) == 0 and len(test_labels - train_labels) == 0:
    print(f"  ✓ All val/test labels appear in training set")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
