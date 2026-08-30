"""
EDA Step 2: Patient-Level Analysis
Analyze patient distribution, visit patterns, and sequence characteristics
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
print("PATIENT-LEVEL ANALYSIS")
print("=" * 80)

# Dataset paths
datasets = {
    "train": DATA_DIR / "train_nextvisit_ccsr_clean.parquet",
    "val": DATA_DIR / "val_nextvisit_ccsr_clean.parquet",
    "test": DATA_DIR / "test_nextvisit_ccsr_clean.parquet"
}

patient_analysis = {}

for split_name, path in datasets.items():
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {split_name.upper()} PATIENTS")
    print(f"{'=' * 80}")
    
    df = pd.read_parquet(path)
    
    # Patient statistics
    unique_patients = df['patid'].nunique()
    total_rows = len(df)
    samples_per_patient = df.groupby('patid').size()
    
    print(f"\nPatient Statistics:")
    print(f"  Total rows (samples): {total_rows:,}")
    print(f"  Unique patients: {unique_patients:,}")
    print(f"  Samples per patient: {total_rows / unique_patients:.2f}")
    
    print(f"\nSamples per patient distribution:")
    print(f"  Min: {samples_per_patient.min()}")
    print(f"  Max: {samples_per_patient.max()}")
    print(f"  Mean: {samples_per_patient.mean():.2f}")
    print(f"  Median: {samples_per_patient.median():.0f}")
    
    # Check if one sample per patient (expected for next-visit task)
    if samples_per_patient.min() == 1 and samples_per_patient.max() == 1:
        print(f"  ✓ One sample per patient (standard next-visit format)")
    else:
        print(f"  ⚠ Multiple samples per patient detected")
        multi_sample_patients = samples_per_patient[samples_per_patient > 1]
        print(f"    Patients with >1 sample: {len(multi_sample_patients)}")
    
    # Sequence length statistics (code history)
    code_lengths = df['code'].apply(lambda x: len(x) if x is not None else 0)
    
    print(f"\nCode History Length (diagnosis count):")
    print(f"  Min: {code_lengths.min()}")
    print(f"  25th percentile: {code_lengths.quantile(0.25):.0f}")
    print(f"  Median: {code_lengths.median():.0f}")
    print(f"  75th percentile: {code_lengths.quantile(0.75):.0f}")
    print(f"  95th percentile: {code_lengths.quantile(0.95):.0f}")
    print(f"  99th percentile: {code_lengths.quantile(0.99):.0f}")
    print(f"  Max: {code_lengths.max()}")
    print(f"  Mean: {code_lengths.mean():.1f}")
    print(f"  Std: {code_lengths.std():.1f}")
    
    # Check for very short vs very long histories
    very_short = (code_lengths <= 10).sum()
    short = ((code_lengths > 10) & (code_lengths <= 50)).sum()
    medium = ((code_lengths > 50) & (code_lengths <= 100)).sum()
    long = ((code_lengths > 100) & (code_lengths <= 200)).sum()
    very_long = (code_lengths > 200).sum()
    
    print(f"\nHistory length distribution:")
    print(f"  Very short (≤10 codes): {very_short:>6,} ({very_short/len(df)*100:>5.1f}%)")
    print(f"  Short (11-50 codes):     {short:>6,} ({short/len(df)*100:>5.1f}%)")
    print(f"  Medium (51-100 codes):   {medium:>6,} ({medium/len(df)*100:>5.1f}%)")
    print(f"  Long (101-200 codes):    {long:>6,} ({long/len(df)*100:>5.1f}%)")
    print(f"  Very long (>200 codes):  {very_long:>6,} ({very_long/len(df)*100:>5.1f}%)")
    
    # Label count statistics (next-visit predictions)
    label_lengths = df['label'].apply(lambda x: len(x) if x is not None else 0)
    
    print(f"\nNext-Visit Label Count (target diagnoses):")
    print(f"  Min: {label_lengths.min()}")
    print(f"  25th percentile: {label_lengths.quantile(0.25):.0f}")
    print(f"  Median: {label_lengths.median():.0f}")
    print(f"  75th percentile: {label_lengths.quantile(0.75):.0f}")
    print(f"  95th percentile: {label_lengths.quantile(0.95):.0f}")
    print(f"  Max: {label_lengths.max()}")
    print(f"  Mean: {label_lengths.mean():.1f}")
    print(f"  Std: {label_lengths.std():.1f}")
    
    # Label distribution
    single_label = (label_lengths == 1).sum()
    few_labels = ((label_lengths > 1) & (label_lengths <= 5)).sum()
    many_labels = ((label_lengths > 5) & (label_lengths <= 15)).sum()
    very_many = (label_lengths > 15).sum()
    
    print(f"\nLabel count distribution:")
    print(f"  Single label (1):        {single_label:>6,} ({single_label/len(df)*100:>5.1f}%)")
    print(f"  Few labels (2-5):        {few_labels:>6,} ({few_labels/len(df)*100:>5.1f}%)")
    print(f"  Many labels (6-15):      {many_labels:>6,} ({many_labels/len(df)*100:>5.1f}%)")
    print(f"  Very many labels (>15):  {very_many:>6,} ({very_many/len(df)*100:>5.1f}%)")
    
    # Age statistics (convert from months to years for readability)
    # Get the last age from each patient's age array (their current age)
    current_ages_months = df['age'].apply(lambda x: x[-1] if len(x) > 0 else 0)
    current_ages_years = current_ages_months / 12
    
    print(f"\nPatient Age Distribution (in years):")
    print(f"  Min: {current_ages_years.min():.1f}")
    print(f"  25th percentile: {current_ages_years.quantile(0.25):.1f}")
    print(f"  Median: {current_ages_years.median():.1f}")
    print(f"  75th percentile: {current_ages_years.quantile(0.75):.1f}")
    print(f"  Max: {current_ages_years.max():.1f}")
    print(f"  Mean: {current_ages_years.mean():.1f}")
    
    # Age groups
    age_0_18 = (current_ages_years < 18).sum()
    age_18_40 = ((current_ages_years >= 18) & (current_ages_years < 40)).sum()
    age_40_65 = ((current_ages_years >= 40) & (current_ages_years < 65)).sum()
    age_65_plus = (current_ages_years >= 65).sum()
    
    print(f"\nAge group distribution:")
    print(f"  <18 years:     {age_0_18:>6,} ({age_0_18/len(df)*100:>5.1f}%)")
    print(f"  18-40 years:   {age_18_40:>6,} ({age_18_40/len(df)*100:>5.1f}%)")
    print(f"  40-65 years:   {age_40_65:>6,} ({age_40_65/len(df)*100:>5.1f}%)")
    print(f"  65+ years:     {age_65_plus:>6,} ({age_65_plus/len(df)*100:>5.1f}%)")
    
    # Save statistics
    patient_analysis[split_name] = {
        "total_rows": int(total_rows),
        "unique_patients": int(unique_patients),
        "samples_per_patient": {
            "min": int(samples_per_patient.min()),
            "max": int(samples_per_patient.max()),
            "mean": float(samples_per_patient.mean()),
            "median": float(samples_per_patient.median())
        },
        "code_history_length": {
            "min": int(code_lengths.min()),
            "p25": float(code_lengths.quantile(0.25)),
            "median": float(code_lengths.median()),
            "p75": float(code_lengths.quantile(0.75)),
            "p95": float(code_lengths.quantile(0.95)),
            "p99": float(code_lengths.quantile(0.99)),
            "max": int(code_lengths.max()),
            "mean": float(code_lengths.mean()),
            "std": float(code_lengths.std())
        },
        "label_count": {
            "min": int(label_lengths.min()),
            "p25": float(label_lengths.quantile(0.25)),
            "median": float(label_lengths.median()),
            "p75": float(label_lengths.quantile(0.75)),
            "p95": float(label_lengths.quantile(0.95)),
            "max": int(label_lengths.max()),
            "mean": float(label_lengths.mean()),
            "std": float(label_lengths.std())
        },
        "age_years": {
            "min": float(current_ages_years.min()),
            "p25": float(current_ages_years.quantile(0.25)),
            "median": float(current_ages_years.median()),
            "p75": float(current_ages_years.quantile(0.75)),
            "max": float(current_ages_years.max()),
            "mean": float(current_ages_years.mean())
        }
    }

# Save analysis
output_path = OUTPUT_DIR / "02_patient_level_analysis.json"
with open(output_path, 'w') as f:
    json.dump(patient_analysis, f, indent=2)

print(f"\n{'=' * 80}")
print(f"Patient-level analysis saved to: {output_path}")
print(f"{'=' * 80}")

# Cross-split patient overlap check
print(f"\n{'=' * 80}")
print("PATIENT OVERLAP CHECK")
print(f"{'=' * 80}")

train_df = pd.read_parquet(datasets["train"])
val_df = pd.read_parquet(datasets["val"])
test_df = pd.read_parquet(datasets["test"])

train_patients = set(train_df['patid'].unique())
val_patients = set(val_df['patid'].unique())
test_patients = set(test_df['patid'].unique())

train_val_overlap = len(train_patients & val_patients)
train_test_overlap = len(train_patients & test_patients)
val_test_overlap = len(val_patients & test_patients)

print(f"\nPatient overlap between splits:")
print(f"  Train ∩ Val:  {train_val_overlap} patients")
print(f"  Train ∩ Test: {train_test_overlap} patients")
print(f"  Val ∩ Test:   {val_test_overlap} patients")

if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
    print(f"\n  ✓ No patient overlap detected (proper split)")
else:
    print(f"\n  ⚠ WARNING: Patient overlap detected - potential data leakage!")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
