"""
Complete ICD-9 to ICD-10 to CCSR Mapping Pipeline
Uses official GEMs crosswalk
"""
import pandas as pd
import pickle
import os
from collections import defaultdict

print("="*80)
print("Creating Complete Unified Mapping: ICD-9 → ICD-10 → CCSR")
print("="*80)

# ============================================================
# STEP 1: Load GEMs (ICD-9 to ICD-10 Crosswalk) - FIXED VERSION
# ============================================================
print("\n" + "="*80)
print("STEP 1: Loading ICD-9 to ICD-10 GEMs - FIXED FORMAT")
print("="*80)

gems_file = "/Users/telaclaimstech/Desktop/BEHRT_run/BEHRT_Project/cssr_mappings/DiagnosisGEMs_2015/2015_I9gem.txt"

icd9_to_icd10 = defaultdict(list)

try:
    print(f"\nReading: {gems_file}")
    
    # FIXED: Use fixed-width format per documentation
    with open(gems_file, 'r', encoding='latin1') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    
    # Parse using fixed-width positions
    for line_num, line in enumerate(lines):
        line = line.rstrip()  # Remove trailing newline but keep leading spaces
        
        # Skip empty lines
        if not line.strip():
            continue
            
        # Extract fields using fixed positions
        # Positions 1-5: ICD-9 code
        icd9 = line[0:5].strip()
        
        # Position 6: Filler (skip)
        
        # Positions 7-13: ICD-10 code  
        icd10 = line[6:13].strip()
        
        # Position 14: Filler (skip)
        
        # Positions 15-19: Flags (we'll use these later)
        flags = line[14:19].strip() if len(line) >= 19 else "00000"
        
        # Only process if we have valid codes
        if icd9 and icd10:
            # Remove dots if present (though format says no decimals)
            icd9_clean = icd9.replace('.', '')
            icd10_clean = icd10.replace('.', '')
            
            # Check "No Map" flag (position 16)
            no_map_flag = flags[1] if len(flags) > 1 else '0'
            
            if no_map_flag == '0':  # Only add if not a "no map" entry
                icd9_to_icd10[icd9_clean].append(icd10_clean)
    
    print(f"\n✓ Loaded {len(icd9_to_icd10)} ICD-9 codes")
    print(f"  Mapped to ICD-10 equivalents")
    
    # Show some examples
    print("\nSample ICD-9 → ICD-10 mappings:")
    for i, (icd9, icd10_list) in enumerate(list(icd9_to_icd10.items())[:10]):
        if len(icd10_list) == 1:
            print(f"  {icd9:10s} → {icd10_list[0]}")
        else:
            print(f"  {icd9:10s} → {icd10_list} (multiple)")
    
    # Statistics
    one_to_one = sum(1 for v in icd9_to_icd10.values() if len(v) == 1)
    one_to_many = sum(1 for v in icd9_to_icd10.values() if len(v) > 1)
    
    print(f"\nMapping statistics:")
    print(f"  One-to-one mappings:  {one_to_one:,}")
    print(f"  One-to-many mappings: {one_to_many:,}")
    
    # Debug: Show first few raw lines to verify parsing
    print(f"\nFirst 5 raw lines for verification:")
    for i in range(min(5, len(lines))):
        line = lines[i].rstrip()
        print(f"  Line {i}: '{line}'")
        if len(line) >= 13:
            print(f"    ICD-9: '{line[0:5]}' → ICD-10: '{line[6:13]}'")
    
except FileNotFoundError:
    print(f" File not found: {gems_file}")
    print("Please check the path")
    exit(1)
except Exception as e:
    print(f" Error reading GEMs file: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# STEP 2: Loading ICD-10 to CCSR Mapping - FIXED VERSION
# ============================================================
print("\n" + "="*80)
print("STEP 2: Loading ICD-10 to CCSR Mapping - FIXED VERSION")
print("="*80)

ccsr_file = "/Users/telaclaimstech/Desktop/BEHRT_run/BEHRT_Project/cssr_mappings/DXCCSR_v2025-1/DXCCSR_v2025-1.csv"

icd10_to_ccsr = {}

try:
    print(f"\nReading: {ccsr_file}")
    
    # Use the exact column names as they appear in the file
    df_ccsr = pd.read_csv(
        ccsr_file,
        encoding='latin1',
        dtype=str,
        on_bad_lines='skip'
    )
    
    print(f"\n✓ Loaded {len(df_ccsr)} ICD-10 codes")
    print(f"  Columns: {df_ccsr.columns.tolist()}")
    
    # Use the exact column names with single quotes
    code_col = "'ICD-10-CM CODE'"
    ccsr_col = "'Default CCSR CATEGORY IP'"
    ccsr_desc_col = "'Default CCSR CATEGORY DESCRIPTION IP'"
    
    print(f"\nUsing columns:")
    print(f"  ICD-10 Code: '{code_col}'")
    print(f"  CCSR Category: '{ccsr_col}'")
    print(f"  CCSR Description: '{ccsr_desc_col}'")  # FIXED: Changed cccsr_desc_col to ccsr_desc_col
    
    # Show first few rows to understand structure
    print("\nFirst 3 rows (showing key columns):")
    print(df_ccsr[[code_col, ccsr_col, ccsr_desc_col]].head(3))
    
    # Create mapping - Handle quoted codes properly
    ccsr_descriptions = {}
    unmapped_count = 0
    
    for idx, row in df_ccsr.iterrows():
        # Get and clean the ICD-10 code
        icd10_raw = str(row[code_col]).strip()
        
        # Remove any quotes and dots, convert to uppercase
        icd10 = icd10_raw.replace("'", "").replace('"', '').replace('.', '').strip().upper()
        
        # Get CCSR category
        ccsr_raw = str(row[ccsr_col]).strip()
        ccsr = ccsr_raw.replace("'", "").replace('"', '').strip().upper()
        
        # Get CCSR description
        ccsr_desc_raw = str(row[ccsr_desc_col]).strip()
        ccsr_desc = ccsr_desc_raw.replace("'", "").replace('"', '').strip()
        
        if icd10 and ccsr and ccsr not in ['', 'NAN', 'NONE', 'NULL', 'N/A']:
            icd10_to_ccsr[icd10] = ccsr
            if ccsr and ccsr not in ccsr_descriptions:
                ccsr_descriptions[ccsr] = ccsr_desc
        else:
            unmapped_count += 1
    
    print(f"\n✓ Mapped {len(icd10_to_ccsr)} ICD-10 codes to CCSR")
    print(f"  Could not map {unmapped_count} codes")
    print(f"  Found {len(set(icd10_to_ccsr.values()))} unique CCSR categories")
    
    # Show samples - test with codes we know should work from GEMs
    print("\nSample ICD-10 → CCSR mappings:")
    test_codes = ['A000', 'A001', 'A009', 'A0100', 'A011', 'A012', 'A013', 'A014', 'A020', 'A021']
    found_any = False
    for test_code in test_codes:
        if test_code in icd10_to_ccsr:
            ccsr = icd10_to_ccsr[test_code]
            desc = ccsr_descriptions.get(ccsr, '')
            print(f"  {test_code:10s} → {ccsr:15s} ({desc[:40]}...)")
            found_any = True
        else:
            print(f"  {test_code:10s} → NOT FOUND IN CCSR")
    
    if not found_any:
        print("\nShowing first 5 actual mappings from file:")
        for i, (icd10, ccsr) in enumerate(list(icd10_to_ccsr.items())[:5]):
            desc = ccsr_descriptions.get(ccsr, '')
            print(f"  {icd10:10s} → {ccsr:15s} ({desc[:40]}...)")
    
    # Debug: Check if our GEMs test cases exist
    print(f"\nDebug - Checking GEMs test case codes in CCSR:")
    # Get first 10 ICD-10 codes from our GEMs mapping to test
    gems_test_codes = []
    for icd9, icd10_list in list(icd9_to_icd10.items())[:10]:
        gems_test_codes.extend(icd10_list)
    
    found_count = 0
    for code in gems_test_codes:
        if code in icd10_to_ccsr:
            found_count += 1
            if found_count <= 5:  # Show first 5 found
                ccsr = icd10_to_ccsr[code]
                desc = ccsr_descriptions.get(ccsr, '')
                print(f"  {code:10s} → {ccsr:15s} ({desc[:30]}...)")
    
    print(f"  Found {found_count}/{len(gems_test_codes)} GEMs ICD-10 codes in CCSR mapping")
    
    # Show some statistics about the mapping
    print(f"\nCCSR Mapping Statistics:")
    print(f"  Total rows in CSV: {len(df_ccsr)}")
    print(f"  Successfully mapped: {len(icd10_to_ccsr)}")
    print(f"  Unmapped: {unmapped_count}")
    print(f"  Unique CCSR categories: {len(set(icd10_to_ccsr.values()))}")
    
except FileNotFoundError:
    print(f" File not found: {ccsr_file}")
    print("Please check the path")
    exit(1)
except Exception as e:
    print(f" Error reading CCSR file: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# STEP 3: Create Unified ICD-9 → CCSR Mapping - WITH DEBUG
# ============================================================
print("\n" + "="*80)
print("STEP 3: Creating Unified ICD-9 → ICD-10 → CCSR Mapping")
print("="*80)

# DEBUG: Check why mappings are failing
print("\nDebug - Checking first 10 ICD-9 mappings:")
for i, (icd9, icd10_list) in enumerate(list(icd9_to_icd10.items())[:10]):
    found_ccsr = []
    for icd10 in icd10_list:
        if icd10 in icd10_to_ccsr:
            found_ccsr.append((icd10, icd10_to_ccsr[icd10]))
    
    if found_ccsr:
        print(f"  {icd9} → {found_ccsr}")
    else:
        print(f"  {icd9} → NO CCSR FOUND for ICD-10: {icd10_list}")

# Continue with your existing STEP 3 code...
icd9_to_ccsr = {}
icd9_mapping_stats = {
    'direct': 0,
    'multiple_same': 0,
    'multiple_different': 0,
    'unmapped': 0
}

for icd9, icd10_list in icd9_to_icd10.items():
    # Get CCSR for each possible ICD-10 code
    ccsr_options = []
    
    for icd10 in icd10_list:
        if icd10 in icd10_to_ccsr:
            ccsr_options.append(icd10_to_ccsr[icd10])
    
    if len(ccsr_options) == 0:
        # No mapping found
        icd9_mapping_stats['unmapped'] += 1
    elif len(set(ccsr_options)) == 1:
        # All ICD-10 codes map to same CCSR (good!)
        icd9_to_ccsr[icd9] = ccsr_options[0]
        if len(ccsr_options) == 1:
            icd9_mapping_stats['direct'] += 1
        else:
            icd9_mapping_stats['multiple_same'] += 1
    else:
        # Multiple ICD-10 codes map to different CCSR (take first)
        icd9_to_ccsr[icd9] = ccsr_options[0]
        icd9_mapping_stats['multiple_different'] += 1

print(f"\nMapping Statistics:")
print(f"  Direct mappings (1-to-1):           {icd9_mapping_stats['direct']:,}")
print(f"  Multiple ICD-10, same CCSR:         {icd9_mapping_stats['multiple_same']:,}")
print(f"  Multiple ICD-10, different CCSR:    {icd9_mapping_stats['multiple_different']:,}")
print(f"  Unmapped (no ICD-10 → CCSR):        {icd9_mapping_stats['unmapped']:,}")
print(f"  Total ICD-9 codes mapped:           {len(icd9_to_ccsr):,}")

# ============================================================
# STEP 4: Combine ICD-9 and ICD-10 Mappings
# ============================================================
print("\n" + "="*80)
print("STEP 4: Creating Final Unified Mapping")
print("="*80)

# Final unified mapping: All codes → CCSR
unified_mapping = {}

# Add ICD-9 mappings
for icd9, ccsr in icd9_to_ccsr.items():
    unified_mapping[icd9] = ccsr

# Add ICD-10 mappings
for icd10, ccsr in icd10_to_ccsr.items():
    unified_mapping[icd10] = ccsr

print(f"\nFinal Unified Mapping:")
print(f"  ICD-9 codes:  {len(icd9_to_ccsr):,}")
print(f"  ICD-10 codes: {len(icd10_to_ccsr):,}")
print(f"  Total codes:  {len(unified_mapping):,}")

# Count unique CCSR categories
unique_ccsr = set(unified_mapping.values())
print(f"  Unique CCSR categories: {len(unique_ccsr)}")

# ============================================================
# STEP 5: Verify Consistency
# ============================================================
print("\n" + "="*80)
print("STEP 5: Verifying Consistency")
print("="*80)

# Test common conditions (codes WITHOUT dots)
test_cases = [
    ('Acute Myocardial Infarction', ['41001', '41011', '41021'], ['I2102', 'I2109', 'I2111']),
    ('Diabetes Mellitus', ['25000', '25001'], ['E119', 'E1100', 'E1101']),
    ('Essential Hypertension', ['4019'], ['I10']),  # Note: '4019' not '401.9'
    ('Congestive Heart Failure', ['4280', '4281'], ['I500', 'I501']),  # Updated codes
    ('Chronic Kidney Disease', ['585', '5851', '5859'], ['N18', 'N181', 'N189']),
    ('COPD', ['4910', '4911', '496'], ['J440', 'J441', 'J449']),
]

print("\nConsistency Check (Same Condition, Different Versions):")
print("-" * 80)

for condition, icd9_codes, icd10_codes in test_cases:
    print(f"\n{condition}:")
    
    # ICD-9 mappings
    icd9_ccsr = []
    for code in icd9_codes:
        if code in unified_mapping:
            ccsr = unified_mapping[code]
            icd9_ccsr.append(ccsr)
            print(f"  ICD-9:  {code:8s} → {ccsr}")
        else:
            print(f"  ICD-9:  {code:8s} → NOT MAPPED")
    
    # ICD-10 mappings
    icd10_ccsr = []
    for code in icd10_codes:
        if code in unified_mapping:
            ccsr = unified_mapping[code]
            icd10_ccsr.append(ccsr)
            print(f"  ICD-10: {code:8s} → {ccsr}")
        else:
            print(f"  ICD-10: {code:8s} → NOT MAPPED")
    
    # Check consistency
    all_ccsr = icd9_ccsr + icd10_ccsr
    if len(set(all_ccsr)) == 1:
        print(f"  ✓ Consistent! All map to {all_ccsr[0]}")
    elif len(set(all_ccsr)) > 1:
        print(f"  ⚠️  Inconsistent: {set(all_ccsr)}")
    else:
        print(f"  ❌ No mappings found")

# ============================================================
# STEP 6: Save Mappings
# ============================================================
print("\n" + "="*80)
print("STEP 6: Saving Mappings")
print("="*80)

# Save unified mapping
output_file = 'ccs_mapping_unified_official.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(unified_mapping, f)

print(f"\n✓ Saved unified mapping to: {output_file}")

# Also save intermediate mappings for reference
with open('icd9_to_icd10_gems.pkl', 'wb') as f:
    pickle.dump(dict(icd9_to_icd10), f)

with open('icd10_to_ccsr.pkl', 'wb') as f:
    pickle.dump(icd10_to_ccsr, f)

print(f"✓ Saved intermediate mappings:")
print(f"  - icd9_to_icd10_gems.pkl")
print(f"  - icd10_to_ccsr.pkl")

# Save human-readable CSV
print("\n✓ Creating human-readable CSV...")

# Sample of unified mapping
sample_size = min(1000, len(unified_mapping))
sample_mapping = dict(list(unified_mapping.items())[:sample_size])

df_sample = pd.DataFrame([
    {
        'ICD_Code': code,
        'CCSR_Category': ccsr,
        'Version': 'ICD-9' if code.isdigit() or (code and code[0].isdigit()) else 'ICD-10',
        'CCSR_Description': ccsr_descriptions.get(ccsr, '')
    }
    for code, ccsr in sample_mapping.items()
])

df_sample.to_csv('unified_mapping_sample.csv', index=False)
print(f"✓ Saved sample to: unified_mapping_sample.csv")

# ============================================================
# STEP 7: Summary Statistics
# ============================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

# Count CCSR category usage
from collections import Counter
ccsr_counts = Counter(unified_mapping.values())

print(f"\nTotal Statistics:")
print(f"  Total codes mapped:     {len(unified_mapping):,}")
print(f"  Unique CCSR categories: {len(unique_ccsr)}")
print(f"  ICD-9 coverage:         {len(icd9_to_ccsr):,} codes")
print(f"  ICD-10 coverage:        {len(icd10_to_ccsr):,} codes")

print(f"\nTop 10 Most Common CCSR Categories:")
for ccsr, count in ccsr_counts.most_common(10):
    desc = ccsr_descriptions.get(ccsr, 'No description')
    print(f"  {ccsr:15s} : {count:6,} codes - {desc[:50]}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)

print("\nFiles created:")
print("  1. ccs_mapping_unified_official.pkl  ← Use this in preprocessing!")
print("  2. icd9_to_icd10_gems.pkl           (reference)")
print("  3. icd10_to_ccsr.pkl                (reference)")
print("  4. unified_mapping_sample.csv       (human-readable)")

print("\nNext steps:")
print("  1. Update preprocess_mimic_ccs.py:")
print("     CCS_MAPPING_PATH = '../data/mappings/ccs_mapping_unified_official.pkl'")
print("  2. Run: python scripts/preprocess_mimic_ccs.py")
print("="*80)