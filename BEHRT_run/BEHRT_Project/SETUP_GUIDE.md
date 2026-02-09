# BEHRT Project Setup Guide for MIMIC-IV

This guide will help you run the BEHRT (BERT for Electronic Health Records) project using MIMIC-IV data.

## Project Overview

BEHRT is a deep learning model that uses BERT architecture to learn representations from electronic health records. The project has two main phases:

1. **Masked Language Model (MLM) Pre-training**: Learn general EHR representations
2. **Next Visit Prediction**: Fine-tune for predicting diagnoses in the next visit

## Prerequisites

### Required Python Packages
```bash
pip install torch pandas numpy scikit-learn h5py --break-system-packages
pip install pytorch-pretrained-bert --break-system-packages
pip install pyarrow fastparquet --break-system-packages
```

### Optional (for big data processing)
```bash
# Only if you want to use PySpark for data preprocessing
pip install pyspark --break-system-packages
```

## Project Structure

```
BEHRT/
├── common/
│   ├── __init__.py
│   ├── common.py       # Utility functions
│   ├── pytorch.py      # Model save/load functions
│   └── spark.py        # PySpark utilities (optional)
├── dataLoader/
│   ├── __init__.py
│   ├── MLM.py          # Data loader for MLM pre-training
│   ├── NextXVisit.py   # Data loader for next visit prediction
│   └── utils.py        # Data processing utilities
├── model/
│   ├── __init__.py
│   ├── MLM.py          # BERT model for MLM
│   ├── NextXVisit.py   # BERT model for next visit prediction
│   ├── optimiser.py    # Optimizer configuration
│   └── utils.py        # Model utilities
├── preprocessing/
│   └── behrtFormat.py  # Data formatting (PySpark-based)
├── notebooks/
│   ├── MLM.ipynb       # Pre-training notebook
│   └── NextXVisit.ipynb # Fine-tuning notebook
└── data/              # Your data directory
    ├── raw/           # MIMIC-IV raw data
    ├── processed/     # Processed data
    └── models/        # Saved models
```

## Data Processing Steps

### Step 1: Understand MIMIC-IV Structure

Your MIMIC-IV data is located at: `/Users/telaclaimstech/Desktop/BERHT_gh/physionet.org/files/mimiciv/3.1/hosp`

Key files:
- `diagnoses_icd.csv.gz` - Diagnosis codes (ICD-9/ICD-10)
- `procedures_icd.csv.gz` - Procedure codes
- `prescriptions.csv.gz` - Medication data
- `patients.csv.gz` - Patient demographics (including year of birth)
- `admissions.csv.gz` - Hospital admission records

### Step 2: Data Preprocessing

The BEHRT format requires:
1. Patient ID (patid)
2. Event date (eventdate)
3. Medical codes (diagnosis/procedure codes)
4. Age at each event
5. Sequential organization with 'SEP' tokens between visits

### Step 3: Vocabulary Creation

You need to create:
1. **token2idx**: Maps medical codes to indices
2. **idx2token**: Reverse mapping
3. **age2idx**: Maps ages to indices (monthly granularity: 0-1320 for 110 years)

## Key Configuration Parameters

### Model Configuration
- `vocab_size`: Number of unique medical codes + 4 special tokens (PAD, SEP, CLS, MASK)
- `hidden_size`: 288 (embedding dimension)
- `num_hidden_layers`: 6 (transformer layers)
- `num_attention_heads`: 12
- `max_position_embeddings`: 64 or 100 (max sequence length)
- `intermediate_size`: 512

### Training Configuration
- `batch_size`: 256
- `learning_rate`: 3e-5
- `max_len_seq`: 64 (for MLM) or 100 (for NextVisit)
- `min_visit`: 5 (minimum number of visits per patient)
- `max_age`: 110 (years)

## Common Issues and Solutions

### 1. Memory Issues
- Reduce `batch_size` to 64 or 32
- Reduce `max_len_seq` to 32 or 64
- Use gradient accumulation

### 2. CUDA/GPU Issues
- Set `device: 'cpu'` if no GPU available
- Reduce model size (hidden_size, num_layers)

### 3. Data Format Issues
- Ensure dates are in datetime format
- Medical codes should be strings
- Each patient's sequence should have 'SEP' tokens between visits

## Next Steps

1. Run the data preprocessing script to convert MIMIC-IV to BEHRT format
2. Create vocabulary from processed data
3. Run MLM pre-training (can take hours/days depending on data size)
4. Fine-tune for next visit prediction
5. Evaluate model performance

## Expected Data Format

### Input Format (before processing):
```
patid | eventdate | icd_code | age
------|-----------|----------|----
1001  | 2020-01-15| I50.9    | 65
1001  | 2020-01-15| E11.9    | 65
1001  | 2020-02-10| I50.9    | 65
```

### BEHRT Format (after processing):
```
patid | code                              | age
------|-----------------------------------|--------------------
1001  | [CLS, I50.9, E11.9, SEP, I50.9, SEP] | [65, 65, 65, 65, 65, 65]
```

## Performance Expectations

- **MLM Pre-training**: 
  - Loss should decrease from ~8-10 to ~2-4
  - Precision should increase to 0.3-0.5
  
- **Next Visit Prediction**:
  - Average Precision Score (APS): 0.15-0.30
  - ROC-AUC: 0.65-0.75

## References

- Original BEHRT paper: https://arxiv.org/abs/1907.09538
- MIMIC-IV documentation: https://mimic.mit.edu/docs/iv/
