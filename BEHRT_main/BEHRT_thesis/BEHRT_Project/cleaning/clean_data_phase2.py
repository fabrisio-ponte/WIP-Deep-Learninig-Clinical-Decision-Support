#!/usr/bin/env python3
"""
Advanced BEHRT Data Cleaning Phase 2
====================================

Additional cleaning beyond XXX000:
1. Remove ultra-rare diseases (<10 samples)
2. Filter insufficient context patients  
3. Remove duplicate sequences
4. Handle extreme sequence lengths
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
from pathlib import Path

# Add project to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.common import save_obj, load_obj

class AdvancedDataCleaner:
    def __init__(self, data_dir="../data/processed"):
        self.data_dir = Path(data_dir)
        self.removal_stats = {}
        
    def clean_phase2(self):
        """Advanced cleaning phase 2"""
        print("🧹 ADVANCED BEHRT DATA CLEANING - PHASE 2")
        print("=" * 60)
        print("Beyond XXX000: Rare diseases, duplicates, context issues")
        
        # Load base cleaned data (from phase 1)
        datasets = {
            "train": pd.read_parquet(self.data_dir / "train_nextvisit_ccsr_clean.parquet"),
            "val": pd.read_parquet(self.data_dir / "val_nextvisit_ccsr_clean.parquet"), 
            "test": pd.read_parquet(self.data_dir / "test_nextvisit_ccsr_clean.parquet")
        }
        vocab = load_obj(str(self.data_dir / "vocab_ccsr_clean"))
        
        print(f"📊 STARTING STATE:")
        total_patients = sum(len(df) for df in datasets.values())
        print(f"  Total patients: {total_patients:,}")
        print(f"  Vocabulary: {len(vocab['token2idx'])} codes")
        
        # Step 1: Identify rare diseases across all datasets
        rare_diseases = self._identify_rare_diseases(datasets)
        
        # Step 2: Clean each dataset  
        cleaned_datasets = {}
        for name, df in datasets.items():
            print(f"\n🔧 CLEANING {name.upper()} DATASET:")
            cleaned_df = self._clean_dataset(df, rare_diseases, name)
            cleaned_datasets[name] = cleaned_df
        
        # Step 3: Update vocabulary
        clean_vocab = self._update_vocabulary(vocab, rare_diseases)
        
        # Step 4: Save cleaned datasets
        self._save_cleaned_data(cleaned_datasets, clean_vocab)
        
        # Step 5: Generate report
        self._generate_phase2_report(datasets, cleaned_datasets)
        
    def _identify_rare_diseases(self, datasets):
        """Identify diseases with <10 total samples"""
        print(f"\n1️⃣ IDENTIFYING RARE DISEASES")
        print("-" * 40)
        
        # Count across all datasets
        all_labels = []
        for df in datasets.values():
            for labels in df['label']:
                all_labels.extend(labels)
                
        label_counts = Counter(all_labels)
        
        # Find diseases with <10 samples
        rare_threshold = 10
        rare_diseases = [
            disease for disease, count in label_counts.items()
            if count < rare_threshold and 'CCSR_' in disease
        ]
        
        print(f"📊 Rare Disease Candidates (< {rare_threshold} samples):")
        print(f"  Found: {len(rare_diseases)} ultra-rare diseases")
        
        # Show some examples
        rare_with_counts = [(d, label_counts[d]) for d in rare_diseases]
        rare_with_counts.sort(key=lambda x: x[1])  # Sort by count
        
        print(f"  Examples (rarest first):")
        for disease, count in rare_with_counts[:8]:
            print(f"    {disease}: {count} samples")
        if len(rare_with_counts) > 8:
            print(f"    ... and {len(rare_with_counts) - 8} more")
            
        total_rare_instances = sum(label_counts[d] for d in rare_diseases)
        total_instances = len(all_labels)
        impact_pct = (total_rare_instances / total_instances) * 100
        
        print(f"  Impact: {total_rare_instances:,} label instances ({impact_pct:.2f}%)")
        
        return rare_diseases
    
    def _clean_dataset(self, df, rare_diseases, dataset_name):
        """Clean individual dataset"""
        original_size = len(df)
        
        print(f"  Original size: {original_size:,} patients")
        
        # Step 1: Remove rare diseases from labels
        cleaned_labels = []
        rare_labels_removed = 0
        
        for labels in df['label']:
            clean_label_list = [
                label for label in labels 
                if label not in rare_diseases
            ]
            cleaned_labels.append(clean_label_list)
            rare_labels_removed += len(labels) - len(clean_label_list)
            
        df_clean = df.copy()
        df_clean['label'] = cleaned_labels
        
        # Step 2: Remove patients with insufficient labels (<2)
        sufficient_labels = df_clean['label'].apply(len) >= 2
        df_clean = df_clean[sufficient_labels].reset_index(drop=True)
        insufficient_removed = original_size - len(df_clean)
        
        # Step 3: Remove duplicate sequences  
        before_dedup = len(df_clean)
        code_strings = df_clean['code'].apply(str)
        duplicated = code_strings.duplicated()
        df_clean = df_clean[~duplicated].reset_index(drop=True)
        duplicates_removed = before_dedup - len(df_clean)
        
        # Step 4: Handle extreme sequence lengths
        seq_lengths = df_clean['code'].apply(len)
        
        # Remove very short sequences (<3 events)  
        before_short = len(df_clean)
        sufficient_length = seq_lengths >= 3
        df_clean = df_clean[sufficient_length].reset_index(drop=True)
        short_removed = before_short - len(df_clean)
        
        # Truncate very long sequences (>100 events, keep last 100)
        long_sequences = df_clean['code'].apply(len) > 100
        if long_sequences.sum() > 0:
            print(f"    Truncating {long_sequences.sum()} long sequences to 100 events")
            df_clean.loc[long_sequences, 'code'] = df_clean.loc[long_sequences, 'code'].apply(lambda x: x[-100:])
            df_clean.loc[long_sequences, 'age'] = df_clean.loc[long_sequences, 'age'].apply(lambda x: x[-100:])
        
        final_size = len(df_clean)
        retention = (final_size / original_size) * 100
        
        print(f"  Rare labels removed: {rare_labels_removed:,}")
        print(f"  Insufficient context removed: {insufficient_removed:,}")
        print(f"  Duplicates removed: {duplicates_removed:,}")  
        print(f"  Short sequences removed: {short_removed:,}")
        print(f"  Final size: {final_size:,} ({retention:.1f}% retained)")
        
        # Store stats
        self.removal_stats[dataset_name] = {
            'original': original_size,
            'final': final_size, 
            'rare_labels_removed': rare_labels_removed,
            'insufficient_removed': insufficient_removed,
            'duplicates_removed': duplicates_removed,
            'short_removed': short_removed,
            'retention_pct': retention
        }
        
        return df_clean
    
    def _update_vocabulary(self, vocab, rare_diseases):
        """Update vocabulary removing rare diseases"""
        print(f"\n2️⃣ UPDATING VOCABULARY")
        print("-" * 40)
        
        original_size = len(vocab['token2idx'])
        
        # Remove rare diseases from vocabulary
        clean_token2idx = {}
        
        # Keep special tokens first
        special_tokens = ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']
        for token in special_tokens:
            if token in vocab['token2idx']:
                clean_token2idx[token] = len(clean_token2idx)
        
        # Add non-rare medical codes
        for token in vocab['token2idx']:
            if token not in special_tokens and token not in rare_diseases:
                clean_token2idx[token] = len(clean_token2idx)
                
        clean_idx2token = {idx: token for token, idx in clean_token2idx.items()}
        
        clean_vocab = {
            'token2idx': clean_token2idx,
            'idx2token': clean_idx2token
        }
        
        final_size = len(clean_token2idx)
        removed = original_size - final_size
        
        print(f"  Original vocabulary: {original_size}")
        print(f"  Removed rare diseases: {removed}")
        print(f"  Final vocabulary: {final_size}")
        print(f"  Retention: {(final_size/original_size)*100:.1f}%")
        
        return clean_vocab
    
    def _save_cleaned_data(self, datasets, vocab):
        """Save phase 2 cleaned data"""
        print(f"\n3️⃣ SAVING PHASE 2 CLEANED DATA")
        print("-" * 40)
        
        for name, df in datasets.items():
            filename = f"{name}_nextvisit_ccsr_ultraclean.parquet"
            filepath = self.data_dir / filename
            df.to_parquet(filepath)
            print(f"  ✓ {filename}")
            
        vocab_path = self.data_dir / "vocab_ccsr_ultraclean.pkl"
        save_obj(vocab, str(vocab_path).replace('.pkl', ''))
        print(f"  ✓ vocab_ccsr_ultraclean.pkl")
        
    def _generate_phase2_report(self, original_datasets, cleaned_datasets):
        """Generate comprehensive phase 2 report"""
        print(f"\n📊 PHASE 2 CLEANING REPORT")
        print("=" * 60)
        
        # Overall statistics
        orig_total = sum(len(df) for df in original_datasets.values())
        clean_total = sum(len(df) for df in cleaned_datasets.values())
        overall_retention = (clean_total / orig_total) * 100
        
        print(f"OVERALL IMPACT:")
        print(f"  Original patients: {orig_total:,}")
        print(f"  Clean patients: {clean_total:,}")
        print(f"  Overall retention: {overall_retention:.1f}%")
        print(f"  Patients removed: {orig_total - clean_total:,}")
        
        print(f"\nPER-DATASET BREAKDOWN:")
        for name in ['train', 'val', 'test']:
            stats = self.removal_stats[name]
            print(f"  {name.capitalize()}:")
            print(f"    {stats['original']:,} → {stats['final']:,} ({stats['retention_pct']:.1f}%)")
            print(f"    Rare labels: -{stats['rare_labels_removed']:,}")
            print(f"    Insufficient context: -{stats['insufficient_removed']:,}")  
            print(f"    Duplicates: -{stats['duplicates_removed']:,}")
            print(f"    Short sequences: -{stats['short_removed']:,}")
        
        print(f"\n✨ ULTRA-CLEAN DATASETS READY:")
        print(f"  train_nextvisit_ccsr_ultraclean.parquet")
        print(f"  val_nextvisit_ccsr_ultraclean.parquet")
        print(f"  test_nextvisit_ccsr_ultraclean.parquet") 
        print(f"  vocab_ccsr_ultraclean.pkl")
        
        print(f"\n🎯 BENEFITS:")
        print(f"  ✓ No ultra-rare diseases (<10 samples)")
        print(f"  ✓ All patients have sufficient context (≥2 diagnoses)")
        print(f"  ✓ No duplicate sequences (prevents overfitting)")
        print(f"  ✓ Consistent sequence lengths (3-100 events)")
        print(f"  ✓ Higher data quality for robust model training")

def main():
    cleaner = AdvancedDataCleaner()
    cleaner.clean_phase2()
    
    print(f"\n🚀 RECOMMENDATION:")
    print("Use ultra-clean datasets for final full model training.")
    print("Expected: Even better performance and interpretability!")

if __name__ == "__main__":
    main()