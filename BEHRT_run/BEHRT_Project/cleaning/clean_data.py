#!/usr/bin/env python3
"""
BEHRT Data Cleaning Script
=========================

Clean the BEHRT dataset to remove problematic codes and improve data quality
before training the full model.

Issues identified:
1. CCSR_XXX000 - Generic catch-all code (8.08% frequency) 
2. Other placeholder/generic codes that skew analysis
3. Need to focus on clinically meaningful CCSR codes

This script will:
- Identify problematic codes
- Create clean version of datasets
- Update vocabularies
- Generate data quality report
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

class BEHRTDataCleaner:
    def __init__(self, data_dir="../data/processed"):
        self.data_dir = Path(data_dir)
        self.original_vocab = None
        self.clean_vocab = None
        self.codes_to_remove = set()
        self.cleaning_stats = {}
        
    def analyze_data_quality(self):
        """Analyze current data quality and identify problematic codes"""
        print("🔍 ANALYZING DATA QUALITY")
        print("=" * 50)
        
        # Load vocabulary
        self.original_vocab = load_obj(str(self.data_dir / "vocab_ccsr"))
        print(f"Original vocabulary size: {len(self.original_vocab['token2idx'])}")
        
        # Load test data for frequency analysis
        test_data = pd.read_parquet(self.data_dir / "test_nextvisit_ccsr.parquet")
        train_data = pd.read_parquet(self.data_dir / "train_nextvisit_ccsr.parquet")
        val_data = pd.read_parquet(self.data_dir / "val_nextvisit_ccsr.parquet")
        
        # Count all label frequencies
        all_labels = []
        for df in [train_data, val_data, test_data]:
            for labels in df['label']:
                all_labels.extend(labels)
        
        label_counts = Counter(all_labels)
        total_labels = len(all_labels)
        
        print(f"\nTotal label instances: {total_labels:,}")
        print(f"Unique labels: {len(label_counts)}")
        
        # Identify problematic codes
        problematic_codes = self._identify_problematic_codes(label_counts, total_labels)
        
        return problematic_codes, label_counts
    
    def _identify_problematic_codes(self, label_counts, total_labels):
        """Identify codes that should be removed"""
        problematic = {}
        
        # Check for XXX codes (generic/unspecified)
        xxx_codes = [code for code in label_counts.keys() if 'XXX' in code]
        if xxx_codes:
            problematic['generic_xxx'] = xxx_codes
            print(f"\n📍 Generic XXX codes found: {len(xxx_codes)}")
            for code in xxx_codes:
                pct = (label_counts[code] / total_labels) * 100
                print(f"  {code}: {label_counts[code]:,} ({pct:.2f}%)")
        
        # Check for very high frequency codes (potential defaults)
        high_freq_threshold = 0.05  # 5%
        high_freq_codes = [
            code for code, count in label_counts.items() 
            if (count / total_labels) > high_freq_threshold and 'XXX' in code
        ]
        if high_freq_codes:
            problematic['high_frequency_generic'] = high_freq_codes
            print(f"\n📍 High-frequency generic codes (>{high_freq_threshold*100}%): {len(high_freq_codes)}")
            for code in high_freq_codes:
                pct = (label_counts[code] / total_labels) * 100
                print(f"  {code}: {label_counts[code]:,} ({pct:.2f}%)")
        
        # Check for codes with suspicious patterns
        pattern_codes = [
            code for code in label_counts.keys() 
            if any(pattern in code for pattern in ['000', '999', 'UNK', 'OTHER'])
        ]
        if pattern_codes:
            problematic['pattern_codes'] = pattern_codes
            print(f"\n📍 Suspicious pattern codes: {len(pattern_codes)}")
            for code in pattern_codes[:10]:  # Show first 10
                pct = (label_counts[code] / total_labels) * 100
                print(f"  {code}: {label_counts[code]:,} ({pct:.2f}%)")
        
        return problematic
    
    def create_cleaning_strategy(self, problematic_codes, label_counts):
        """Create data cleaning strategy"""
        print(f"\n📋 CLEANING STRATEGY")
        print("=" * 50)
        
        codes_to_remove = set()
        
        # Strategy 1: Remove all XXX codes (generic/unspecified)
        if 'generic_xxx' in problematic_codes:
            codes_to_remove.update(problematic_codes['generic_xxx'])
            print(f"✓ Remove all XXX generic codes: {len(problematic_codes['generic_xxx'])}")
        
        # Strategy 2: Remove high-frequency generic codes
        if 'high_frequency_generic' in problematic_codes:
            codes_to_remove.update(problematic_codes['high_frequency_generic'])
            print(f"✓ Remove high-frequency generic codes: {len(problematic_codes['high_frequency_generic'])}")
        
        # Strategy 3: Manual review of pattern codes
        if 'pattern_codes' in problematic_codes:
            # Be more conservative with pattern codes
            suspicious_patterns = ['000', '999', 'UNK']
            pattern_to_remove = [
                code for code in problematic_codes['pattern_codes']
                if any(pattern in code for pattern in suspicious_patterns)
            ]
            codes_to_remove.update(pattern_to_remove)
            print(f"✓ Remove suspicious pattern codes: {len(pattern_to_remove)}")
        
        self.codes_to_remove = codes_to_remove
        
        # Calculate impact
        total_instances = sum(label_counts[code] for code in codes_to_remove)
        total_labels = sum(label_counts.values())
        impact_pct = (total_instances / total_labels) * 100
        
        print(f"\nIMPACT ASSESSMENT:")
        print(f"  Codes to remove: {len(codes_to_remove)}")
        print(f"  Label instances removed: {total_instances:,} ({impact_pct:.2f}%)")
        print(f"  Remaining codes: {len(label_counts) - len(codes_to_remove)}")
        
        # Show codes being removed
        print(f"\nCODES TO BE REMOVED:")
        removed_codes_sorted = sorted(
            [(code, label_counts[code]) for code in codes_to_remove],
            key=lambda x: x[1], reverse=True
        )
        for code, count in removed_codes_sorted[:10]:  # Show top 10
            pct = (count / total_labels) * 100
            print(f"  {code}: {count:,} ({pct:.2f}%)")
        if len(removed_codes_sorted) > 10:
            print(f"  ... and {len(removed_codes_sorted) - 10} more")
        
        return codes_to_remove
    
    def clean_datasets(self):
        """Clean all datasets by removing problematic codes"""
        print(f"\n🧹 CLEANING DATASETS")
        print("=" * 50)
        
        datasets = [
            "train_nextvisit_ccsr.parquet",
            "val_nextvisit_ccsr.parquet", 
            "test_nextvisit_ccsr.parquet"
        ]
        
        cleaning_stats = {}
        
        for dataset_file in datasets:
            print(f"\nProcessing {dataset_file}...")
            
            # Load dataset
            df = pd.read_parquet(self.data_dir / dataset_file)
            original_size = len(df)
            
            # Clean labels
            df['label_original'] = df['label'].copy()  # Keep original for reference
            
            cleaned_labels = []
            removed_count = 0
            
            for labels in df['label']:
                clean_label_list = [
                    label for label in labels 
                    if label not in self.codes_to_remove
                ]
                cleaned_labels.append(clean_label_list)
                removed_count += len(labels) - len(clean_label_list)
            
            df['label'] = cleaned_labels
            
            # Remove rows with empty labels after cleaning
            df = df[df['label'].apply(len) > 0].reset_index(drop=True)
            final_size = len(df)
            
            # Save cleaned dataset
            clean_filename = dataset_file.replace('.parquet', '_clean.parquet')
            df.to_parquet(self.data_dir / clean_filename)
            
            # Track statistics
            cleaning_stats[dataset_file] = {
                'original_rows': original_size,
                'cleaned_rows': final_size,
                'removed_rows': original_size - final_size,
                'removed_labels': removed_count,
                'retention_rate': (final_size / original_size) * 100
            }
            
            print(f"  Original rows: {original_size:,}")
            print(f"  Cleaned rows: {final_size:,}")
            print(f"  Removed rows: {original_size - final_size:,}")
            print(f"  Removed labels: {removed_count:,}")
            print(f"  Retention rate: {(final_size/original_size)*100:.1f}%")
        
        self.cleaning_stats = cleaning_stats
        return cleaning_stats
    
    def create_clean_vocabulary(self):
        """Create cleaned vocabulary without problematic codes"""
        print(f"\n📚 CREATING CLEAN VOCABULARY")
        print("=" * 50)
        
        # Create new vocabulary excluding problematic codes 
        original_token2idx = self.original_vocab['token2idx']
        
        # Keep special tokens
        clean_token2idx = {}
        special_tokens = ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']
        
        for token in special_tokens:
            if token in original_token2idx:
                clean_token2idx[token] = len(clean_token2idx)
        
        # Add clean medical codes
        for token in original_token2idx:
            if token not in special_tokens and token not in self.codes_to_remove:
                clean_token2idx[token] = len(clean_token2idx)
        
        # Create reverse mapping
        clean_idx2token = {idx: token for token, idx in clean_token2idx.items()}
        
        # Create clean vocabulary
        clean_vocab = {
            'token2idx': clean_token2idx,
            'idx2token': clean_idx2token
        }
        
        print(f"Original vocabulary size: {len(original_token2idx)}")
        print(f"Clean vocabulary size: {len(clean_token2idx)}")
        print(f"Removed codes: {len(original_token2idx) - len(clean_token2idx)}")
        print(f"Retention rate: {(len(clean_token2idx)/len(original_token2idx))*100:.1f}%")
        
        # Save clean vocabulary
        save_obj(clean_vocab, str(self.data_dir / "vocab_ccsr_clean"))
        print(f"✓ Saved clean vocabulary to: {self.data_dir / 'vocab_ccsr_clean.pkl'}")
        
        self.clean_vocab = clean_vocab
        return clean_vocab
    
    def generate_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        print(f"\n📊 DATA CLEANING REPORT")
        print("=" * 50)
        
        total_original_rows = sum(stats['original_rows'] for stats in self.cleaning_stats.values())
        total_cleaned_rows = sum(stats['cleaned_rows'] for stats in self.cleaning_stats.values())
        total_removed_labels = sum(stats['removed_labels'] for stats in self.cleaning_stats.values())
        
        print(f"OVERALL IMPACT:")
        print(f"  Total original rows: {total_original_rows:,}")
        print(f"  Total cleaned rows: {total_cleaned_rows:,}")
        print(f"  Retention rate: {(total_cleaned_rows/total_original_rows)*100:.1f}%")
        print(f"  Total removed label instances: {total_removed_labels:,}")
        
        print(f"\nPER-DATASET BREAKDOWN:")
        for dataset, stats in self.cleaning_stats.items():
            print(f"  {dataset}:")
            print(f"    Rows: {stats['original_rows']:,} → {stats['cleaned_rows']:,} ({stats['retention_rate']:.1f}%)")
            print(f"    Removed labels: {stats['removed_labels']:,}")
        
        print(f"\nVOCABULARY CHANGES:")
        if self.clean_vocab and self.original_vocab:
            original_size = len(self.original_vocab['token2idx'])
            clean_size = len(self.clean_vocab['token2idx'])
            print(f"  Original size: {original_size}")
            print(f"  Clean size: {clean_size}")
            print(f"  Reduction: {original_size - clean_size} codes ({((original_size-clean_size)/original_size)*100:.1f}%)")
        
        print(f"\nRECOMMENDATIONS:")
        print(f"  ✓ Use cleaned datasets (*_clean.parquet) for training")
        print(f"  ✓ Use clean vocabulary (vocab_ccsr_clean.pkl)")
        print(f"  ✓ Expected improvement in model interpretability")
        print(f"  ✓ More meaningful disease predictions")
    
    def run_full_cleaning(self):
        """Run complete data cleaning process"""
        print("🧹 BEHRT DATA CLEANING PROCESS")
        print("=" * 60)
        
        # Step 1: Analyze data quality
        problematic_codes, label_counts = self.analyze_data_quality()
        
        # Step 2: Create cleaning strategy  
        codes_to_remove = self.create_cleaning_strategy(problematic_codes, label_counts)
        
        # Step 3: Clean datasets
        cleaning_stats = self.clean_datasets()
        
        # Step 4: Create clean vocabulary
        clean_vocab = self.create_clean_vocabulary()
        
        # Step 5: Generate report
        self.generate_cleaning_report()
        
        print(f"\n✅ DATA CLEANING COMPLETE!")
        print(f"Clean files created:")
        print(f"  - train_nextvisit_ccsr_clean.parquet")
        print(f"  - val_nextvisit_ccsr_clean.parquet") 
        print(f"  - test_nextvisit_ccsr_clean.parquet")
        print(f"  - vocab_ccsr_clean.pkl")

def main():
    cleaner = BEHRTDataCleaner()
    cleaner.run_full_cleaning()

if __name__ == "__main__":
    main()