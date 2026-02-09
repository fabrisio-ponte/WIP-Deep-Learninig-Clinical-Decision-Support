#!/usr/bin/env python3
"""
BEHRT Quick Start Script
Run this after setting up your environment to process data and start training.
"""

import os
import sys

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_mimic_data(path):
    """Check if MIMIC-IV data exists"""
    required_files = [
        'patients.csv.gz',
        'admissions.csv.gz',
        'diagnoses_icd.csv.gz'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            missing.append(file)
    
    if missing:
        print(f"❌ Missing required files in {path}:")
        for file in missing:
            print(f"   - {file}")
        return False
    else:
        print(f"✅ Found all required MIMIC-IV files in {path}")
        return True

def main():
    print_header("BEHRT Project - Quick Start")
    
    print("This script will guide you through setting up and running BEHRT.")
    print("\nSteps:")
    print("  1. Verify MIMIC-IV data location")
    print("  2. Install required packages")
    print("  3. Preprocess data")
    print("  4. Run MLM pre-training")
    print("  5. Run next visit prediction")
    
    # Step 1: Check MIMIC data
    print_header("Step 1: Verify MIMIC-IV Data")
    mimic_path = "/Users/telaclaimstech/Desktop/BERHT_gh/physionet.org/files/mimiciv/3.1/hosp"
    print(f"Checking for MIMIC-IV data at: {mimic_path}")
    
    if not os.path.exists(mimic_path):
        print(f"❌ Directory not found: {mimic_path}")
        print("\nPlease update the MIMIC_DATA_PATH in scripts/preprocess_mimic.py")
        print("to point to your actual MIMIC-IV data location.")
        return
    
    data_ok = check_mimic_data(mimic_path)
    
    if not data_ok:
        print("\n⚠️  Please ensure all MIMIC-IV files are present before continuing.")
        return
    
    # Step 2: Installation check
    print_header("Step 2: Check Required Packages")
    print("Required packages:")
    print("  - torch")
    print("  - pytorch-pretrained-bert")
    print("  - pandas, numpy, scikit-learn")
    print("  - pyarrow (for parquet files)")
    
    print("\nTo install, run:")
    print("  pip install -r requirements.txt --break-system-packages")
    
    response = input("\nHave you installed the requirements? (y/n): ")
    if response.lower() != 'y':
        print("Please install requirements first and run this script again.")
        return
    
    # Step 3: Data preprocessing
    print_header("Step 3: Data Preprocessing")
    print("The preprocessing script will:")
    print("  - Load MIMIC-IV data (patients, admissions, diagnoses)")
    print("  - Calculate patient ages at each visit")
    print("  - Create sequential visit data with SEP tokens")
    print("  - Build vocabularies (medical codes and ages)")
    print("  - Split into train/validation/test sets")
    print("  - Save processed data to data/processed/")
    
    print("\n⚠️  This may take 10-30 minutes depending on your system.")
    
    response = input("\nRun preprocessing now? (y/n): ")
    if response.lower() == 'y':
        print("\nRunning preprocessing...")
        print("Command: python scripts/preprocess_mimic.py")
        print("\nYou can also run this manually to see detailed progress.")
        
        # Don't actually run it here - user should run manually to see progress
        print("\nPlease run: python scripts/preprocess_mimic.py")
        print("in the BEHRT_Project directory.")
    
    # Step 4 & 5: Training
    print_header("Step 4 & 5: Model Training")
    print("After preprocessing completes, you can train the models:")
    
    print("\n📊 Phase 1: MLM Pre-training")
    print("   Open notebooks/MLM.ipynb in Jupyter")
    print("   This trains the model to learn EHR representations")
    print("   Expected time: Several hours (depends on data size and GPU)")
    
    print("\n🎯 Phase 2: Next Visit Prediction")
    print("   Open notebooks/NextXVisit.ipynb in Jupyter")
    print("   This fine-tunes for predicting next visit diagnoses")
    print("   Expected time: 1-2 hours")
    
    print("\nTo start Jupyter:")
    print("   jupyter notebook")
    
    # Summary
    print_header("Summary")
    print("Project structure:")
    print("  📁 data/")
    print("     ├── raw/            (your MIMIC-IV data)")
    print("     ├── processed/      (processed parquet files + vocabularies)")
    print("     └── models/         (saved model checkpoints)")
    print("\n  📁 notebooks/")
    print("     ├── MLM.ipynb       (pre-training notebook)")
    print("     └── NextXVisit.ipynb (fine-tuning notebook)")
    print("\n  📁 scripts/")
    print("     └── preprocess_mimic.py (data preprocessing)")
    
    print("\n📚 For detailed information, see:")
    print("   - SETUP_GUIDE.md: Comprehensive setup guide")
    print("   - README.md: Project overview and instructions")
    
    print("\n✨ Next steps:")
    print("   1. Run: python scripts/preprocess_mimic.py")
    print("   2. Wait for processing to complete")
    print("   3. Open Jupyter and run MLM.ipynb")
    print("   4. After MLM training, run NextXVisit.ipynb")

if __name__ == "__main__":
    main()
