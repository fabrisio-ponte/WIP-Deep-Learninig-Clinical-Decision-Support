"""
MIMIC-IV to BEHRT Format Converter
This script processes MIMIC-IV data and converts it to BEHRT format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import sys
from pathlib import Path
import gzip

# Configuration
# Update this path to your MIMIC-IV data location 
MIMIC_DATA_PATH = "../../physionet.org/files/mimiciv/3.1/hosp"
OUTPUT_PATH = "../data/processed"  # Relative to scripts/ directory
MIN_VISITS = 5  # Minimum number of visits to include a patient
MAX_AGE_YEARS = 110

# You can also pass the path as a command line argument:
# python preprocess_mimic.py /path/to/mimic-iv/hosp
if len(sys.argv) > 1:
    MIMIC_DATA_PATH = sys.argv[1]
    print(f"Using MIMIC data path from command line: {MIMIC_DATA_PATH}")

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_mimic_data():
    """Load relevant MIMIC-IV tables"""
    print("Loading MIMIC-IV data...")
    
    # Load patients (for demographics)
    print("  - Loading patients...")
    patients = pd.read_csv(f"{MIMIC_DATA_PATH}/patients.csv.gz", compression='gzip')
    patients = patients[['subject_id', 'anchor_age', 'anchor_year']]
    
    # Load admissions (for dates)
    print("  - Loading admissions...")
    admissions = pd.read_csv(f"{MIMIC_DATA_PATH}/admissions.csv.gz", compression='gzip')
    admissions = admissions[['subject_id', 'hadm_id', 'admittime']]
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    
    # Load diagnoses
    print("  - Loading diagnoses...")
    diagnoses = pd.read_csv(f"{MIMIC_DATA_PATH}/diagnoses_icd.csv.gz", compression='gzip')
    diagnoses = diagnoses[['subject_id', 'hadm_id', 'icd_code', 'icd_version']]
    
    print(f"Loaded {len(patients)} patients, {len(admissions)} admissions, {len(diagnoses)} diagnoses")
    return patients, admissions, diagnoses

def calculate_age(patients, admissions):
    """Calculate age at each admission"""
    print("Calculating ages...")
    
    # Merge patients with admissions
    df = admissions.merge(patients, on='subject_id', how='left')
    
    # Calculate year of admission
    df['admit_year'] = df['admittime'].dt.year
    
    # Calculate age at admission (in years)
    df['age_at_admission'] = df['anchor_age'] + (df['admit_year'] - df['anchor_year'])
    
    # Cap age at MAX_AGE_YEARS (for privacy in MIMIC)
    df['age_at_admission'] = df['age_at_admission'].clip(upper=MAX_AGE_YEARS)
    
    # Convert to months for finer granularity
    df['age_months'] = (df['age_at_admission'] * 12).astype(int)
    
    return df[['subject_id', 'hadm_id', 'admittime', 'age_months']]

def merge_and_format(admissions_with_age, diagnoses):
    """Merge diagnoses with admissions and format"""
    print("Merging diagnoses with admissions...")
    
    # Merge diagnoses with admissions
    df = diagnoses.merge(admissions_with_age, on=['subject_id', 'hadm_id'], how='inner')
    
    # Create code with version prefix
    df['code'] = df['icd_version'].astype(str) + '_' + df['icd_code'].astype(str)
    
    # Keep only necessary columns
    df = df[['subject_id', 'hadm_id', 'admittime', 'code', 'age_months']]
    
    # Sort by patient and admission time
    df = df.sort_values(['subject_id', 'admittime', 'code'])
    
    print(f"Total records: {len(df)}")
    return df

def create_visit_sequences(df):
    """Create sequences with SEP tokens between visits"""
    print("Creating visit sequences...")
    
    sequences = []
    
    for subject_id, group in df.groupby('subject_id'):
        codes = []
        ages = []
        visit_count = 0
        
        for hadm_id, visit in group.groupby('hadm_id'):
            visit_count += 1
            
            # Get codes and ages for this visit
            visit_codes = visit['code'].tolist()
            visit_ages = visit['age_months'].tolist()
            
            # Add codes and ages
            codes.extend(visit_codes)
            ages.extend(visit_ages)
            
            # Add SEP token between visits (except after last visit)
            codes.append('SEP')
            ages.append(visit_ages[0] if visit_ages else 0)
        
        # Only include patients with minimum number of visits
        if visit_count >= MIN_VISITS:
            sequences.append({
                'patid': subject_id,
                'code': codes,
                'age': ages,
                'num_visits': visit_count
            })
    
    print(f"Created sequences for {len(sequences)} patients with >={MIN_VISITS} visits")
    return pd.DataFrame(sequences)

def create_vocabulary(sequences_df):
    """Create vocabulary from all codes"""
    print("Creating vocabulary...")
    
    # Get all unique codes
    all_codes = set()
    for codes in sequences_df['code']:
        all_codes.update(codes)
    
    # Remove SEP as it will be added as special token
    all_codes.discard('SEP')
    
    # Sort codes for consistent ordering
    sorted_codes = sorted(list(all_codes))
    
    # Create token2idx with special tokens
    special_tokens = ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']
    token2idx = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Add regular codes
    for idx, code in enumerate(sorted_codes):
        token2idx[code] = idx + len(special_tokens)
    
    # Create idx2token
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    vocab = {
        'token2idx': token2idx,
        'idx2token': idx2token
    }
    
    print(f"Vocabulary size: {len(token2idx)} (including {len(special_tokens)} special tokens)")
    return vocab

def create_age_vocabulary(max_age=110, granularity='month'):
    """Create age vocabulary"""
    print("Creating age vocabulary...")
    
    age2idx = {}
    idx2age = {}
    
    # Special tokens
    age2idx['PAD'] = 0
    age2idx['UNK'] = 1
    idx2age[0] = 'PAD'
    idx2age[1] = 'UNK'
    
    # Age values (in months)
    if granularity == 'month':
        max_age_months = max_age * 12
        for i in range(max_age_months):
            age2idx[str(i)] = i + 2
            idx2age[i + 2] = str(i)
    
    print(f"Age vocabulary size: {len(age2idx)}")
    return age2idx, idx2age

def split_data(sequences_df, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation, and test sets"""
    print("Splitting data...")
    
    # Shuffle
    sequences_df = sequences_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(sequences_df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_df = sequences_df[:train_size]
    val_df = sequences_df[train_size:train_size+val_size]
    test_df = sequences_df[train_size+val_size:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def create_label_sequences(sequences_df):
    """Create label sequences for next visit prediction"""
    print("Creating label sequences...")
    
    labeled_sequences = []
    
    for idx, row in sequences_df.iterrows():
        codes = row['code']
        ages = row['age']
        
        # Find all SEP positions
        sep_positions = [i for i, code in enumerate(codes) if code == 'SEP']
        
        if len(sep_positions) < 2:
            continue  # Need at least 2 visits
        
        # For each visit except the last, create a training example
        for i in range(len(sep_positions) - 1):
            # Input: everything up to and including current visit
            if i == 0:
                input_codes = codes[:sep_positions[i]+1]
                input_ages = ages[:sep_positions[i]+1]
            else:
                input_codes = codes[:sep_positions[i]+1]
                input_ages = ages[:sep_positions[i]+1]
            
            # Label: codes in next visit (between current SEP and next SEP)
            label_start = sep_positions[i] + 1
            label_end = sep_positions[i + 1]
            label_codes = [c for c in codes[label_start:label_end] if c != 'SEP']
            
            if label_codes:  # Only add if there are labels
                labeled_sequences.append({
                    'patid': row['patid'],
                    'code': input_codes,
                    'age': input_ages,
                    'label': label_codes
                })
    
    print(f"Created {len(labeled_sequences)} labeled sequences")
    return pd.DataFrame(labeled_sequences)

def save_data(train_df, val_df, test_df, vocab, age_vocab):
    """Save processed data and vocabularies"""
    print("Saving data...")
    
    # Save dataframes
    train_df.to_parquet(f"{OUTPUT_PATH}/train_mlm.parquet", index=False)
    val_df.to_parquet(f"{OUTPUT_PATH}/val_mlm.parquet", index=False)
    test_df.to_parquet(f"{OUTPUT_PATH}/test_mlm.parquet", index=False)
    
    # Save vocabularies
    with open(f"{OUTPUT_PATH}/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    
    with open(f"{OUTPUT_PATH}/age_vocab.pkl", 'wb') as f:
        pickle.dump(age_vocab, f)
    
    print(f"Data saved to {OUTPUT_PATH}")
    print("\nFiles created:")
    print("  - train_mlm.parquet: Training data for MLM pre-training")
    print("  - val_mlm.parquet: Validation data for MLM")
    print("  - test_mlm.parquet: Test data for MLM")
    print("  - vocab.pkl: Code vocabulary (token2idx, idx2token)")
    print("  - age_vocab.pkl: Age vocabulary (age2idx, idx2age)")

def main():
    """Main processing pipeline"""
    print("="*60)
    print("MIMIC-IV to BEHRT Format Converter")
    print("="*60)
    
    # Step 1: Load data
    patients, admissions, diagnoses = load_mimic_data()
    
    # Step 2: Calculate ages
    admissions_with_age = calculate_age(patients, admissions)
    
    # Step 3: Merge and format
    formatted_df = merge_and_format(admissions_with_age, diagnoses)
    
    # Step 4: Create sequences
    sequences_df = create_visit_sequences(formatted_df)
    
    # Step 5: Create vocabularies
    vocab = create_vocabulary(sequences_df)
    age2idx, idx2age = create_age_vocabulary()
    age_vocab = {'age2idx': age2idx, 'idx2age': idx2age}
    
    # Step 6: Split data
    train_df, val_df, test_df = split_data(sequences_df)
    
    # Step 7: Create labeled sequences for next visit prediction
    print("\nCreating labeled data for next visit prediction...")
    train_labeled = create_label_sequences(train_df)
    val_labeled = create_label_sequences(val_df)
    test_labeled = create_label_sequences(test_df)
    
    # Save labeled data
    train_labeled.to_parquet(f"{OUTPUT_PATH}/train_nextvisit.parquet", index=False)
    val_labeled.to_parquet(f"{OUTPUT_PATH}/val_nextvisit.parquet", index=False)
    test_labeled.to_parquet(f"{OUTPUT_PATH}/test_nextvisit.parquet", index=False)
    
    # Step 8: Save everything
    save_data(train_df, val_df, test_df, vocab, age_vocab)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print("\nData statistics:")
    print(f"  Total patients: {len(sequences_df)}")
    print(f"  Vocabulary size: {len(vocab['token2idx'])}")
    print(f"  Age vocabulary size: {len(age_vocab['age2idx'])}")
    print(f"  Train samples (MLM): {len(train_df)}")
    print(f"  Train samples (NextVisit): {len(train_labeled)}")
    
    print("\nNext steps:")
    print("  1. Run MLM pre-training using MLM.ipynb")
    print("  2. Fine-tune for next visit prediction using NextXVisit.ipynb")

if __name__ == "__main__":
    main()
