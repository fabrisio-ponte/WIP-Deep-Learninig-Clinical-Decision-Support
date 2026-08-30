"""
EDA Step 1: Dataset Structure Analysis
Inspect the schema, data types, and basic structure of the cleaned CCSR datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "eda" / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("DATASET STRUCTURE ANALYSIS")
print("=" * 80)

# Dataset paths
datasets = {
    "train": DATA_DIR / "train_nextvisit_ccsr_clean.parquet",
    "val": DATA_DIR / "val_nextvisit_ccsr_clean.parquet",
    "test": DATA_DIR / "test_nextvisit_ccsr_clean.parquet"
}

structure_summary = {}

for split_name, path in datasets.items():
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {split_name.upper()} DATASET")
    print(f"{'=' * 80}")
    
    df = pd.read_parquet(path)
    
    print(f"\nPath: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    print("\nColumn Schema:")
    print("-" * 80)
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        
        # For array/object columns, get more info
        if dtype == 'object':
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                avg_len = df[col].apply(lambda x: len(x) if x is not None else 0).mean()
                max_len = df[col].apply(lambda x: len(x) if x is not None else 0).max()
                print(f"  {col:20s} | {str(dtype):15s} | Non-null: {non_null:>7,} ({100-null_pct:>5.1f}%) | Avg len: {avg_len:>6.1f} | Max len: {max_len:>4}")
            else:
                print(f"  {col:20s} | {str(dtype):15s} | Non-null: {non_null:>7,} ({100-null_pct:>5.1f}%)")
        else:
            print(f"  {col:20s} | {str(dtype):15s} | Non-null: {non_null:>7,} ({100-null_pct:>5.1f}%)")
    
    # Sample first row
    print("\nSample First Row:")
    print("-" * 80)
    first_row = df.iloc[0]
    for col in df.columns:
        val = first_row[col]
        if isinstance(val, (list, np.ndarray)):
            display_val = f"Array with {len(val)} elements"
            if len(val) > 0:
                display_val += f", first: {val[0]}"
        else:
            display_val = str(val)[:60]
        print(f"  {col:20s}: {display_val}")
    
    # Collect structure info
    structure_summary[split_name] = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
    }
    
    # For array columns, get length stats
    array_stats = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                lengths = df[col].apply(lambda x: len(x) if x is not None else 0)
                array_stats[col] = {
                    "mean_length": float(lengths.mean()),
                    "median_length": float(lengths.median()),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "std_length": float(lengths.std())
                }
    
    structure_summary[split_name]["array_stats"] = array_stats

# Save structure summary
output_path = OUTPUT_DIR / "01_dataset_structure.json"
with open(output_path, 'w') as f:
    json.dump(structure_summary, f, indent=2)

print(f"\n{'=' * 80}")
print(f"Structure summary saved to: {output_path}")
print(f"{'=' * 80}")

# Quick cross-split comparison
print("\n" + "=" * 80)
print("CROSS-SPLIT COMPARISON")
print("=" * 80)

print("\nRow counts:")
for split_name in ["train", "val", "test"]:
    rows = structure_summary[split_name]["rows"]
    pct = (rows / sum(s["rows"] for s in structure_summary.values())) * 100
    print(f"  {split_name:10s}: {rows:>8,} ({pct:>5.1f}%)")

print("\nTotal samples:", sum(s["rows"] for s in structure_summary.values()))

print("\nColumn consistency:")
train_cols = set(structure_summary["train"]["columns"])
val_cols = set(structure_summary["val"]["columns"])
test_cols = set(structure_summary["test"]["columns"])

if train_cols == val_cols == test_cols:
    print("  ✓ All splits have identical columns")
else:
    print("  ✗ Column mismatch detected:")
    print(f"    Train only: {train_cols - val_cols - test_cols}")
    print(f"    Val only: {val_cols - train_cols - test_cols}")
    print(f"    Test only: {test_cols - train_cols - val_cols}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
