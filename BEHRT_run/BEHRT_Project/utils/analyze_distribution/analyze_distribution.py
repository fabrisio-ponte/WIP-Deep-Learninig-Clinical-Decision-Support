import pandas as pd
import pickle
from collections import Counter

print("="*60)
print("BEHRT DATA DISTRIBUTION ANALYSIS")
print("="*60)

# Load train data
train = pd.read_parquet('../data/processed/train_nextvisit_ccsr.parquet')
print(f"\nDATASET SIZE")
print(f"Training samples: {len(train):,}")

test = pd.read_parquet('../data/processed/test_nextvisit_ccsr.parquet')
print(f"Test samples: {len(test):,}")

# Load vocabulary
with open('../data/processed/vocab_ccsr.pkl', 'rb') as f:
    vocab = pickle.load(f)
print(f"Vocabulary size: {len(vocab['token2idx'])} tokens")

# Analyze code distribution
print("\n" + "="*60)
print("DIAGNOSIS CODE DISTRIBUTION (Top 25)")
print("="*60)

all_codes = []
for codes in train['code']:
    for code in codes:
        if code not in ['CLS', 'SEP', 'PAD', 'MASK']:
            all_codes.append(code)

code_counts = Counter(all_codes)
total_codes = len(all_codes)

print(f"\nTotal diagnosis occurrences: {total_codes:,}")
print(f"Unique codes: {len(code_counts)}")
print("\nTop 25 most frequent codes:")
for code, count in code_counts.most_common(25):
    pct = count / total_codes * 100
    print(f"  {code:20s}: {count:8,} ({pct:5.2f}%)")

# Check for pregnancy codes
print("\n" + "="*60)
print("PREGNANCY-RELATED CODES")
print("="*60)
preg_codes = [c for c in code_counts.keys() if c.startswith('PRG')]
if preg_codes:
    for code in preg_codes:
        pct = code_counts[code] / total_codes * 100
        print(f"  {code:20s}: {code_counts[code]:8,} ({pct:5.2f}%)")
else:
    print("  No pregnancy codes found")

# Age distribution
print("\n" + "="*60)
print("AGE DISTRIBUTION")
print("="*60)
all_ages = []
for ages in train['age']:
    all_ages.extend(ages)
# Convert ages from months to years for proper analysis
age_years_list = []
for age_months in all_ages:
    try:
        age_years = int(age_months) // 12  # Convert months to years
        age_years_list.append(age_years)
    except:
        pass

age_counts_years = Counter(age_years_list)
# Show age ranges in years (not months)
age_ranges = {'0-18': 0, '19-30': 0, '31-50': 0, '51-65': 0, '66-80': 0, '81+': 0}
for age_year, count in age_counts_years.items():
    if age_year <= 18: age_ranges['0-18'] += count
    elif age_year <= 30: age_ranges['19-30'] += count  
    elif age_year <= 50: age_ranges['31-50'] += count
    elif age_year <= 65: age_ranges['51-65'] += count
    elif age_year <= 80: age_ranges['66-80'] += count
    else: age_ranges['81+'] += count

total_ages = sum(age_ranges.values())
print("Age ranges (converted from months to years):")
for r, c in age_ranges.items():
    pct = c / total_ages * 100 if total_ages > 0 else 0
    print(f"  {r:10s}: {c:8,} ({pct:5.1f}%)")

print(f"\nAge statistics (in years):")
if age_years_list:
    print(f"  Min age: {min(age_years_list)} years")
    print(f"  Max age: {max(age_years_list)} years") 
    print(f"  Avg age: {sum(age_years_list)/len(age_years_list):.1f} years")
    sorted_ages = sorted(age_years_list)
    print(f"  Median age: {sorted_ages[len(sorted_ages)//2]} years")
print(f"  Note: Ages stored in months ({min(all_ages[:100])}-{max(all_ages[:100])}) for finer granularity")

# Visits per patient distribution
print("\n" + "="*60)
print("VISITS PER PATIENT (Sequence Length)")
print("="*60)
visit_lens = [len(codes) for codes in train['code']]
print(f"  Min visits: {min(visit_lens)}")
print(f"  Max visits: {max(visit_lens)}")
print(f"  Avg visits: {sum(visit_lens)/len(visit_lens):.1f}")
print(f"  Median visits: {sorted(visit_lens)[len(visit_lens)//2]}")

# Labels distribution (what we're predicting)
print("\n" + "="*60)
print("LABEL DISTRIBUTION (Next Visit Diagnoses)")
print("="*60)
label_codes = []
for labels in train['label']:
    label_codes.extend(labels)
label_counts = Counter(label_codes)
print(f"\nTotal label occurrences: {len(label_codes):,}")
print(f"Unique labels: {len(label_counts)}")
print("\nTop 20 most predicted conditions:")
for code, count in label_counts.most_common(20):
    pct = count / len(label_codes) * 100
    print(f"  {code:20s}: {count:8,} ({pct:5.2f}%)")

# Check label sparsity
print("\n" + "="*60)
print("LABEL SPARSITY")
print("="*60)
labels_per_sample = [len(l) for l in train['label']]
print(f"  Avg labels per sample: {sum(labels_per_sample)/len(labels_per_sample):.2f}")
print(f"  Max labels per sample: {max(labels_per_sample)}")
print(f"  Min labels per sample: {min(labels_per_sample)}")
