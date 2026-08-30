# BEHRT Main - Thesis Work with MIMIC-IV Data

This directory contains the main thesis work using real MIMIC-IV data for next-visit clinical code prediction.

---

## Directory Structure

```
BEHRT_main/
├── BEHRT_core/          # Original BEHRT implementation
│   ├── common/          # Utility functions
│   ├── dataLoader/      # Data loading
│   ├── model/           # Model definitions
│   ├── preprocess/      # Preprocessing
│   └── task/            # Task-specific notebooks
│
├── BEHRT_thesis/        # ACTIVE THESIS WORK
│   ├── data/
│   │   ├── processed/   # Cleaned CCSR parquet files
│   │   └── models/      # Saved model checkpoints
│   ├── scripts/
│   │   └── train_nextvisit_clean.py  # Main training script
│   ├── eda/             # Exploratory data analysis (3/7 complete)
│   │   ├── 01_dataset_structure.py
│   │   ├── 02_patient_level_analysis.py
│   │   ├── 03_label_support_analysis.py
│   │   └── results/     # EDA outputs (JSON)
│   ├── model/           # BEHRT model implementation
│   ├── dataLoader/      # Data loaders for thesis tasks
│   ├── common/          # Shared utilities
│   └── README.md        # Detailed thesis documentation
│
└── physionet.org/       # MIMIC-IV data reference/documentation
```

---

## Purpose

### BEHRT_core/
- **Original BEHRT implementation** from the paper
- Base model architecture
- MLM pre-training utilities
- Reference implementation

**Use when:** Need base BEHRT model or utilities

### BEHRT_thesis/
- **Primary thesis work** - next-visit clinical code prediction
- Cleaned CCSR-coded MIMIC-IV sequences
- Positive-class weighting experiments
- Comprehensive EDA analysis
- Production model training

**Use when:** Working on thesis experiments (most of the time)

### physionet.org/
- MIMIC-IV data access documentation
- Dataset structure reference

---

## Workflow

### 1. Data Flow
```
MIMIC-IV raw data 
  ↓ (preprocessing)
BEHRT_thesis/data/processed/*.parquet
  ↓ (training)
BEHRT_thesis/data/models/clean_runs/
```

### 2. Typical Development
```bash
# Navigate to thesis work
cd BEHRT_thesis/

# Run EDA
python3.12 eda/04_temporal_structure.py

# Train model
MLFLOW_ENABLE=0 USE_POS_WEIGHT=1 MAX_POS_WEIGHT=30.0 \
  python scripts/train_nextvisit_clean.py

# Evaluate results
ls data/models/clean_runs/  # Check latest run
```

---

## Current Status

**Completed:**
- ✅ Data preprocessing (CCSR-coded, cleaned)
- ✅ Baseline training (AUC 0.891, APS 0.262)
- ✅ Pos_weight training (AUC 0.901, APS 0.274)
- ✅ EDA Steps 1-3 (dataset structure, patient-level, class imbalance)

**In Progress:**
- EDA Steps 4-7 (temporal, co-occurrence, representativeness, cleaning impact)

**Next:**
- Comparison table creation
- Disease-level performance analysis
- NEW vs RECURRING stratified evaluation

---

## Key Finding

This is **NOT** disease onset prediction — it's **clinical code recurrence prediction** for multimorbid adults:
- 70% of predictions are recurrent (already in patient history)
- 30% are new diagnoses
- Median patient: 62 years, 44 prior diagnoses, 9 next-visit codes

See `../../docs/publication_readiness_summary.md` for complete thesis strategy.

---

## Documentation

**Detailed README:** `BEHRT_thesis/README.md`  
**Publication Strategy:** `../../docs/publication_readiness_summary.md`  
**EDA Results:** `BEHRT_thesis/eda/results/`

---

**Last Updated:** August 29, 2026  
**Primary Work Location:** `BEHRT_thesis/`
