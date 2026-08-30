#!/usr/bin/env python3
"""
BEHRT Data Quality Investigation
===============================

Identify additional data quality issues beyond XXX000:
1. Rare disease codes with insufficient samples  
2. Generic/placeholder codes we missed
3. Temporal sequence issues
4. Patient data anomalies
5. Code consistency problems
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.common import load_obj

class DataQualityInvestigator:
    def __init__(self):
        # Get project root and data directory
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "processed"
        
    def investigate_all_issues(self):
        """Run comprehensive data quality investigation"""
        print("🔍 COMPREHENSIVE DATA QUALITY INVESTIGATION")
        print("=" * 60)
        
        # Load data
        train_data = pd.read_parquet(self.data_dir / "train_nextvisit_ccsr.parquet")
        val_data = pd.read_parquet(self.data_dir / "val_nextvisit_ccsr.parquet")  
        test_data = pd.read_parquet(self.data_dir / "test_nextvisit_ccsr.parquet")
        vocab = load_obj(str(self.data_dir / "vocab_ccsr"))
        
        all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
        
        print(f"📊 Dataset Overview:")
        print(f"  Total patients: {len(all_data):,}")
        print(f"  Vocabulary size: {len(vocab['token2idx'])}")
        
        # Issue 1: Rare disease analysis
        self.analyze_rare_diseases(all_data)
        
        # Issue 2: Generic/placeholder codes
        self.find_suspicious_codes(all_data, vocab)
        
        # Issue 3: Temporal sequence issues
        self.analyze_sequence_quality(all_data)
        
        # Issue 4: Patient data anomalies 
        self.analyze_patient_anomalies(all_data)
        
        # Issue 5: Label distribution issues
        self.analyze_label_patterns(all_data)
        
        return self.generate_cleaning_recommendations()
    
    def analyze_rare_diseases(self, data):
        """Find diseases with insufficient training samples"""
        print(f"\n1️⃣ RARE DISEASE ANALYSIS")
        print("=" * 50)
        
        # Count all label frequencies
        all_labels = []
        for labels in data['label']:
            all_labels.extend(labels)
        
        label_counts = Counter(all_labels)
        total_labels = len(all_labels)
        
        # Define thresholds
        very_rare_threshold = 10      # < 10 samples
        rare_threshold = 50          # 10-50 samples
        uncommon_threshold = 200     # 50-200 samples
        
        very_rare = [code for code, count in label_counts.items() if count < very_rare_threshold]
        rare = [code for code, count in label_counts.items() if very_rare_threshold <= count < rare_threshold]
        uncommon = [code for code, count in label_counts.items() if rare_threshold <= count < uncommon_threshold]
        
        print(f"📈 Disease Frequency Distribution:")
        print(f"  Very rare (< {very_rare_threshold} samples): {len(very_rare)} diseases")
        print(f"  Rare ({very_rare_threshold}-{rare_threshold} samples): {len(rare)} diseases") 
        print(f"  Uncommon ({rare_threshold}-{uncommon_threshold} samples): {len(uncommon)} diseases")
        
        if very_rare:
            print(f"\n🚨 Very Rare Diseases (may need removal):")
            for code in very_rare[:10]:  # Show first 10
                print(f"    {code}: {label_counts[code]} samples")
            if len(very_rare) > 10:
                print(f"    ... and {len(very_rare) - 10} more")
        
        return {'very_rare': very_rare, 'rare': rare, 'uncommon': uncommon}
    
    def find_suspicious_codes(self, data, vocab):
        """Find other generic/placeholder codes"""
        print(f"\n2️⃣ SUSPICIOUS CODE PATTERNS")
        print("=" * 50)
        
        all_codes = list(vocab['token2idx'].keys())
        
        # Pattern matching for suspicious codes
        suspicious_patterns = {
            'generic': ['XXX', '000', '999', 'UNK', 'OTHER', 'MISC'],
            'placeholders': ['TMP', 'TEMP', 'NULL', 'NA', 'DEFAULT'],
            'catch_all': ['GENERAL', 'UNSPECIFIED', 'OTHER', 'MISCELLANEOUS']
        }
        
        found_suspicious = {}
        for category, patterns in suspicious_patterns.items():
            suspicious_codes = [
                code for code in all_codes 
                if any(pattern in code.upper() for pattern in patterns) and 'CCSR_' in code
            ]
            if suspicious_codes:
                found_suspicious[category] = suspicious_codes
        
        # Count frequencies
        all_labels = []
        for labels in data['label']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        for category, codes in found_suspicious.items():
            print(f"\n🔍 {category.upper()} codes:")
            total_impact = 0
            for code in codes:
                count = label_counts.get(code, 0)
                pct = (count / len(all_labels)) * 100 if count > 0 else 0
                print(f"    {code}: {count:,} ({pct:.2f}%)")
                total_impact += count
            print(f"  Total impact: {total_impact:,} labels ({(total_impact/len(all_labels))*100:.2f}%)")
        
        return found_suspicious
    
    def analyze_sequence_quality(self, data):
        """Analyze sequence length and temporal patterns"""
        print(f"\n3️⃣ SEQUENCE QUALITY ANALYSIS")
        print("=" * 50)
        
        # Calculate sequence lengths (code column contains the sequences)
        seq_lengths = data['code'].apply(len)
        label_counts_per_patient = data['label'].apply(len)
        
        print(f"📏 Sequence Length Statistics:")
        print(f"  Mean length: {seq_lengths.mean():.1f}")
        print(f"  Median length: {seq_lengths.median():.1f}")
        print(f"  Min/Max: {seq_lengths.min()}/{seq_lengths.max()}")
        print(f"  Std dev: {seq_lengths.std():.1f}")
        
        # Find problematic sequences
        very_short = seq_lengths < 5
        very_long = seq_lengths > 95  # Assuming max_len = 100
        
        print(f"\n⚠️ Problematic Sequences:")
        print(f"  Very short (< 5 tokens): {very_short.sum()} patients ({(very_short.sum()/len(data))*100:.1f}%)")
        print(f"  Very long (> 95 tokens): {very_long.sum()} patients ({(very_long.sum()/len(data))*100:.1f}%)")
        
        # Label count analysis
        print(f"\n🏷️ Labels per Patient:")
        print(f"  Mean labels: {label_counts_per_patient.mean():.1f}")
        print(f"  Median labels: {label_counts_per_patient.median():.1f}")
        print(f"  Min/Max labels: {label_counts_per_patient.min()}/{label_counts_per_patient.max()}")
        
        # Find patients with extreme label counts
        too_few_labels = label_counts_per_patient < 2
        too_many_labels = label_counts_per_patient > 20
        
        print(f"  Too few labels (< 2): {too_few_labels.sum()} patients")
        print(f"  Too many labels (> 20): {too_many_labels.sum()} patients")
        
        return {
            'very_short_sequences': data[very_short],
            'very_long_sequences': data[very_long], 
            'too_few_labels': data[too_few_labels],
            'too_many_labels': data[too_many_labels]
        }
    
    def analyze_patient_anomalies(self, data):
        """Find anomalous patient data"""
        print(f"\n4️⃣ PATIENT DATA ANOMALIES")
        print("=" * 50)
        
        # Age analysis - age contains arrays, get first/last age per patient
        ages_first = data['age'].apply(lambda x: x[0] if len(x) > 0 else None)
        ages_last = data['age'].apply(lambda x: x[-1] if len(x) > 0 else None)
        
        print(f"👥 Age Distribution (first visit):")
        print(f"  Mean age: {ages_first.mean():.1f}")
        print(f"  Age range: {ages_first.min()}-{ages_first.max()}")
        
        unusual_ages = (ages_first < 0) | (ages_first > 120)
        print(f"  Unusual ages (< 0 or > 120): {unusual_ages.sum()} patients")
        
        # Check for potential duplicates based on sequence similarity
        print(f"\n🔄 Duplicate Detection:")
        code_strings = data['code'].apply(lambda x: str(x))
        duplicates = code_strings.duplicated()
        print(f"  Potential duplicate sequences: {duplicates.sum()} patients")
        
        # Check for patients with identical label sets
        label_strings = data['label'].apply(lambda x: str(sorted(x)))
        identical_labels = label_strings.duplicated() 
        print(f"  Patients with identical label sets: {identical_labels.sum()} patients")
        
        return {
            'unusual_ages': data[unusual_ages],
            'duplicate_sequences': data[duplicates],
            'identical_labels': data[identical_labels]
        }
    
    def analyze_label_patterns(self, data):
        """Analyze suspicious label patterns"""
        print(f"\n5️⃣ LABEL PATTERN ANALYSIS") 
        print("=" * 50)
        
        # Find patients with single vs multiple diagnoses
        single_label = data['label'].apply(len) == 1
        multiple_labels = data['label'].apply(len) > 1
        
        print(f"📊 Label Distribution Patterns:")
        print(f"  Single diagnosis: {single_label.sum():,} patients ({(single_label.sum()/len(data))*100:.1f}%)")
        print(f"  Multiple diagnoses: {multiple_labels.sum():,} patients ({(multiple_labels.sum()/len(data))*100:.1f}%)")
        
        # Check for common label combinations that might be artificial
        label_combinations = data['label'].apply(lambda x: tuple(sorted(x)) if len(x) > 1 else tuple(x))
        combo_counts = Counter(label_combinations)
        
        print(f"\n🔗 Most Common Label Combinations:")
        for combo, count in combo_counts.most_common(5):
            pct = (count / len(data)) * 100
            if len(combo) > 1:
                combo_str = " + ".join(c.replace('CCSR_', '') for c in combo[:3])
                if len(combo) > 3:
                    combo_str += f" + {len(combo)-3} more"
                print(f"  {combo_str}: {count} patients ({pct:.2f}%)")
        
        return {
            'single_diagnosis_patients': data[single_label],
            'common_combinations': combo_counts.most_common(20)
        }
    
    def generate_cleaning_recommendations(self):
        """Generate cleaning recommendations"""
        print(f"\n💡 DATA CLEANING RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        print("🎯 HIGH PRIORITY:")
        print("✓ Remove diseases with < 10 training samples (too rare to learn)")
        print("✓ Remove additional generic/placeholder codes found") 
        print("✓ Filter out patients with < 2 labels (insufficient context)")
        recommendations.extend(['rare_diseases', 'generic_codes', 'insufficient_labels'])
        
        print(f"\n🔍 MEDIUM PRIORITY:")
        print("✓ Investigate very short sequences (< 5 tokens)")
        print("✓ Review patients with >20 diagnoses (potential data errors)")
        print("✓ Check duplicate sequences for data leakage")
        recommendations.extend(['short_sequences', 'excessive_labels', 'duplicates'])
        
        print(f"\n📊 LOW PRIORITY (Monitor):")
        print("✓ Track rare diseases (10-50 samples) - may improve with more data")
        print("✓ Analyze common label combinations for clinical validity")
        print("✓ Review age distribution anomalies")
        recommendations.extend(['monitor_rare', 'label_combinations', 'age_anomalies'])
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Run this analysis to identify specific issues")
        print("2. Create targeted cleaning scripts for high priority items")
        print("3. Update clean_data.py to include additional filters")
        print("4. Validate cleaned data maintains clinical meaningfulness")
        
        return recommendations

def main():
    investigator = DataQualityInvestigator()
    recommendations = investigator.investigate_all_issues()
    
    print(f"\n✅ INVESTIGATION COMPLETE!")
    print("Run this script to identify specific data quality issues")
    print("beyond XXX000 and create targeted cleaning strategies.")

if __name__ == "__main__":
    main()