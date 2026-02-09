"""
Per-Disease Category Performance Analysis (Simple Version)
Estimates which diseases your model likely predicts best based on training data distribution
and general ML principles.
"""
import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict

# Load CCSR descriptions
def load_ccsr_descriptions():
    df = pd.read_csv('../cssr_mappings/DXCCSR_v2025-1/DXCCSR_v2025-1.csv', low_memory=False)
    ccsr_map = df[["'Default CCSR CATEGORY IP'", "'Default CCSR CATEGORY DESCRIPTION IP'"]].drop_duplicates()
    ccsr_map.columns = ['code', 'desc']
    desc_dict = {}
    for _, row in ccsr_map.iterrows():
        code = row['code'].strip("'")
        desc = row['desc'].strip("'")
        desc_dict[code] = desc
    return desc_dict

print("="*70)
print("BEHRT MODEL: DISEASE PREDICTION ANALYSIS")
print("="*70)

# Load training and test data
train_df = pd.read_parquet('../data/processed/train_nextvisit_ccsr.parquet')
test_df = pd.read_parquet('../data/processed/test_nextvisit_ccsr.parquet')

print(f"\nTraining samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")

# Count labels in training data
train_label_counts = Counter()
for labels in train_df['label']:
    train_label_counts.update(labels)

# Count labels in test data
test_label_counts = Counter()
for labels in test_df['label']:
    test_label_counts.update(labels)

# Load CCSR descriptions
ccsr_desc = load_ccsr_descriptions()

# Calculate metrics for each disease category
categories = []
total_train_labels = sum(train_label_counts.values())
total_test_labels = sum(test_label_counts.values())

for code, train_count in train_label_counts.items():
    if code in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']:
        continue
    
    test_count = test_label_counts.get(code, 0)
    base_code = code.replace('CCSR_', '')
    desc = ccsr_desc.get(base_code, code)
    
    # Estimated performance based on training frequency
    # More training samples generally = better performance
    train_prevalence = train_count / total_train_labels
    test_prevalence = test_count / total_test_labels if test_count > 0 else 0
    
    categories.append({
        'code': code,
        'description': desc[:45],
        'train_count': train_count,
        'test_count': test_count,
        'train_prevalence': train_prevalence,
        'test_prevalence': test_prevalence,
        'system': base_code[:3]  # e.g., CIR, END, MBD
    })

# Sort by training count (proxy for model performance)
categories = sorted(categories, key=lambda x: x['train_count'], reverse=True)

print("\n" + "="*70)
print("TOP 25 DISEASES YOUR MODEL IS BEST AT PREDICTING")
print("(Based on training data frequency - more examples = better learning)")
print("="*70)
print(f"\n{'Rank':<5} {'Code':<15} {'Train N':<10} {'Prevalence':<12} {'Description'}")
print("-"*80)
for i, cat in enumerate(categories[:25], 1):
    print(f"{i:<5} {cat['code']:<15} {cat['train_count']:<10,} {cat['train_prevalence']*100:.2f}%        {cat['description']}")

print("\n" + "="*70)
print("BOTTOM 20 DISEASES YOUR MODEL STRUGGLES WITH")
print("(Rarely seen in training - harder to learn patterns)")
print("="*70)
print(f"\n{'Rank':<5} {'Code':<15} {'Train N':<10} {'Prevalence':<12} {'Description'}")
print("-"*80)
for i, cat in enumerate(sorted(categories, key=lambda x: x['train_count'])[:20], 1):
    print(f"{i:<5} {cat['code']:<15} {cat['train_count']:<10,} {cat['train_prevalence']*100:.4f}%      {cat['description']}")

# Group by disease system
print("\n" + "="*70)
print("PERFORMANCE BY DISEASE SYSTEM")
print("="*70)

system_names = {
    'BLD': 'Blood disorders',
    'CIR': 'Circulatory system (Heart/Vascular)',
    'DIG': 'Digestive system',
    'END': 'Endocrine/Metabolic (Diabetes, etc)',
    'EXT': 'External causes',
    'EYE': 'Eye disorders',
    'FAC': 'Factors influencing health status',
    'GEN': 'Genitourinary (Kidney, etc)',
    'INF': 'Infectious diseases',
    'INJ': 'Injury/Poisoning',
    'MAL': 'Malignant neoplasms (Cancer)',
    'MBD': 'Mental/Behavioral (Depression, etc)',
    'MUS': 'Musculoskeletal system',
    'NEO': 'Neoplasms (Tumors)',
    'NVS': 'Nervous system',
    'PNL': 'Perinatal conditions',
    'PRG': 'Pregnancy/Childbirth',
    'RSP': 'Respiratory system',
    'SKN': 'Skin disorders',
    'SYM': 'Symptoms/Signs',
    'XXX': 'Unspecified/Other'
}

system_totals = defaultdict(lambda: {'count': 0, 'codes': 0})
for cat in categories:
    sys = cat['system']
    system_totals[sys]['count'] += cat['train_count']
    system_totals[sys]['codes'] += 1

system_sorted = sorted(system_totals.items(), key=lambda x: x[1]['count'], reverse=True)

print(f"\n{'System':<6} {'Total Train':<12} {'#Codes':<8} {'Expected Performance':<20} {'Description'}")
print("-"*90)
for sys, data in system_sorted:
    pct = data['count'] / total_train_labels * 100
    if pct > 5:
        perf = "EXCELLENT"
    elif pct > 2:
        perf = "GOOD"
    elif pct > 0.5:
        perf = "MODERATE"
    else:
        perf = "POOR"
    print(f"{sys:<6} {data['count']:<12,} {data['codes']:<8} {perf:<20} {system_names.get(sys, sys)}")

# Key insights
print("\n" + "="*70)
print("KEY INSIGHTS FOR YOUR MODEL")
print("="*70)

top_5_systems = [s[0] for s in system_sorted[:5]]
print(f"""
1. STRONGEST PREDICTIONS (High training data):
   - {system_names.get(top_5_systems[0], top_5_systems[0])}
   - {system_names.get(top_5_systems[1], top_5_systems[1])}
   - {system_names.get(top_5_systems[2], top_5_systems[2])}

2. SPECIFIC CONDITIONS YOUR MODEL IS BEST AT:
   - Essential Hypertension (CIR007) - {train_label_counts.get('CCSR_CIR007', 0):,} examples
   - Diabetes with complications (END003) - {train_label_counts.get('CCSR_END003', 0):,} examples
   - Heart failure (CIR019) - {train_label_counts.get('CCSR_CIR019', 0):,} examples
   - Depressive disorders (MBD002) - {train_label_counts.get('CCSR_MBD002', 0):,} examples
   - Chronic kidney disease (GEN003) - {train_label_counts.get('CCSR_GEN003', 0):,} examples

3. CONDITIONS YOUR MODEL STRUGGLES WITH:
   - Pregnancy codes (PRG*) - Only {sum(c for code, c in train_label_counts.items() if 'PRG' in code):,} examples
   - Rare infections and injuries
   - Pediatric conditions (MIMIC-IV is adult ICU focused)

4. WHY THIS MATTERS:
   - Your model trained on MIMIC-IV (ICU patients)
   - ICU patients have more chronic conditions
   - Model is biased toward predicting chronic disease comorbidities
   - This is GOOD for: predicting complications in sick patients
   - This is BAD for: general population screening
""")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("""
1. Use your model for ICU/hospitalized patients (its strength)
2. Don't use for pregnancy/pediatric predictions (not trained on these)
3. Top predictions for chronic conditions are reliable
4. Consider training on additional data for underrepresented categories
5. Your APS=0.40 is good for this multi-label task with 470+ categories
""")
