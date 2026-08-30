# BEHRT Thesis - Next-Visit Clinical Code Prediction [WIP]

**Location:** `BEHRT_main/BEHRT_thesis/`  
**Status:** Work in Progress - Thesis Project  
**Last Updated:** August 29, 2026

This is the **primary thesis work** for next-visit clinical code prediction using BEHRT-style transformers on MIMIC-IV data.

> **📖 Navigation:**  
> - **Workspace overview:** `../../README.md`  
> - **BEHRT_main structure:** `../README.md`  
> - **Publication strategy:** `../../docs/publication_readiness_summary.md` **← READ THIS**

---

## 🎯 Project Overview

A BEHRT-style transformer model for **next-visit clinical code prediction** in multimorbid adult patients. The model learns temporal patterns from longitudinal EHR sequences to predict which clinical codes (diagnoses, symptoms, encounters) will appear at a patient's next visit.

### ⚠️ Important Reframing (August 2026)

**Original concept:** Disease onset prediction  
**Actual task (discovered through EDA):** Clinical code recurrence prediction

- **70% of predictions are RECURRENCES** (chronic diseases already in patient history)
- **30% are new diagnoses**
- **14.4% of targets are non-disease codes** (symptoms, administrative, injuries)

This is a **care coordination and chronic disease management** tool, NOT a general disease screening system.

---

## 📊 Dataset Characteristics

**Source:** MIMIC-IV → CCSR-coded diagnoses → Cleaned multilabel sequences

### Key Statistics
- **Total samples:** 179,713 (143,677 train / 17,799 val / 18,237 test)
- **Unique patients:** ~25,000
- **Samples per patient:** ~7 (longitudinal snapshots at different time points)
- **Unique CCSR codes:** 463 (prediction targets)

### Patient Demographics
- **Median age:** 62 years
- **Age distribution:** 88% adults 40+, 46% age 65+, 0% under 18
- **Median history length:** 44 prior diagnoses
- **Median next-visit codes:** 9 simultaneous conditions

### Code Type Distribution
| Category | Percentage | Description |
|----------|------------|-------------|
| Disease diagnoses | 85.6% | True pathological conditions |
| Symptoms | 7.0% | Signs/symptoms (chest pain, abnormal labs) |
| Administrative | 4.0% | Encounter codes (routine checkup, preventive care) |
| Injuries | 2.7% | Acute trauma/falls (unpredictable) |
| Other | 0.7% | Pregnancy, congenital conditions |

### Class Imbalance (Extreme)
- **Most common code:** CCSR_END010 (diabetes) - 52,559 samples (36.6%)
- **Rarest codes:** 7 diseases with only 1 sample each
- **Imbalance ratio:** 52,559:1
- **Rare diseases (<100 samples):** 133 codes (29% of all disease types)
- **Overall sparsity:** 48:1 (negative:positive), only 2% of labels are positive

### Top 5 Most Common Codes
1. **CCSR_END010** (36.6%) - Diabetes without complication
2. **CCSR_CIR007** (35.3%) - Hypertension
3. **CCSR_FAC025** (28.0%) - Encounter for general exam/checkup ⚠️ (admin code)
4. **CCSR_DIG004** (26.4%) - Esophageal disorders
5. **CCSR_CIR011** (23.5%) - Cardiac dysrhythmias

---

## 🏗️ Model Architecture

**Type:** BEHRT-style transformer for longitudinal EHR  
**Framework:** PyTorch

### Configuration
- **Hidden size:** 288
- **Attention layers:** 6
- **Attention heads:** 12
- **Training epochs:** 3
- **Random seed:** 42 (for reproducibility)
- **Loss function:** BCEWithLogitsLoss (with optional positive-class weighting)

### Input Format
Each sample contains:
- `code`: Array of CCSR diagnosis codes (patient history, variable length 2-1636, median 44)
- `age`: Array of ages in months (matching timeline for codes)
- `label`: Array of next-visit target codes (variable length 1-34, median 9)

---

## 🧪 Experimental Results

### Baseline vs Positive-Class Weighting Comparison

| Metric | Baseline (No Weighting) | Pos_Weight (MAX=30) | Change |
|--------|-------------------------|---------------------|--------|
| **Sample-wise APS** | 0.2625 | 0.2735 | **+4.2%** ✓ |
| **Sample-wise AUC** | 0.8914 | 0.9006 | **+1.0%** ✓ |
| **Test Micro F1** | ~0.20-0.23 | 0.2381 | Improved |
| **Tuned Threshold** | 0.15 | 0.600 | Higher (expected) |
| **Configuration** | USE_POS_WEIGHT=0 | USE_POS_WEIGHT=1, MAX_POS_WEIGHT=30.0 | |

### Key Findings
- ✅ **Positive-class weighting improves rare code detection** (4.2% APS gain)
- ✅ **Maintains strong ranking performance** (AUC 0.90)
- ⚠️ **Performance dominated by common diseases** (top 5 codes drive most signal)
- ⚠️ **Rare diseases (<100 samples) remain difficult** despite weighting

### Why pos_weight helps
Without weighting, model optimally predicts "always absent" for rare codes (99.9% accuracy).  
With `pos_weight = min(neg_count/pos_count, 30)`, rare disease errors get 30x more weight → model pays attention.

---

## 📈 Exploratory Data Analysis (Completed)

### ✅ Step 1: Dataset Structure Analysis
**File:** `eda/01_dataset_structure.py`

**Findings:**
- Confirmed 5 columns: patid, code, age, label, label_original
- Arrays in code/age/label columns (numpy arrays stored in parquet cells)
- No missing values across all splits
- Proper 80/10/10 split ratio maintained

### ✅ Step 2: Patient-Level Analysis
**File:** `eda/02_patient_level_analysis.py`

**Findings:**
- **Multiple samples per patient:** Each patient appears ~7 times (different temporal snapshots)
- **Zero patient overlap** between train/val/test splits ✓
- **History length:** 45% have 11-50 codes (short), 9.4% have >200 codes (ultra-complex)
- **Label count:** 64% predict 6-15 simultaneous conditions (true multilabel challenge)
- **Age distribution:** Heavily skewed to older adults (median 62), zero pediatric patients

### ✅ Step 3: Label Support & Class Imbalance Analysis
**File:** `eda/03_label_support_analysis.py`

**Findings:**
- **Extreme imbalance:** 52,559:1 ratio (most common to rarest)
- **Support distribution:** 8% ultra-rare (<10 samples), 21% very rare (10-99), 33% rare (100-999)
- **Code type heterogeneity:** 14.4% of targets are non-disease codes
- **Cross-split validation:** All val/test labels appear in training set ✓
- **Recurrence pattern:** 70% of target codes already in patient history (discovered)

### 🔄 Step 4: Temporal Structure Analysis [PENDING]
- Time gaps between visits
- Sequence length evolution over time
- Code addition/removal patterns

### 🔄 Step 5: Label Co-occurrence Patterns [PENDING]
- Which diseases co-occur frequently
- Comorbidity networks
- Conditional probabilities

### 🔄 Step 6: Split Representativeness [PENDING]
- Distribution comparisons across splits
- Statistical tests for similarity

### 🔄 Step 7: Cleaning Impact Analysis [PENDING]
- Compare label vs label_original
- Assess information loss from cleaning

---

## 🎓 Publication Strategy

See `../../docs/publication_readiness_summary.md` for comprehensive strategy.

### What We CAN Claim
✅ Next-visit clinical code prediction for multimorbid adults  
✅ Learns temporal recurrence patterns (70%) with some new diagnosis detection (30%)  
✅ Demonstrates transformer effectiveness on longitudinal EHR with extreme class imbalance  
✅ Positive-class weighting improves rare code detection by 4.2% APS  
✅ Supports care coordination and resource allocation for complex patients  

### What We CANNOT Claim
❌ General disease prediction model  
❌ Predicts new disease onset (it's primarily recurrence)  
❌ Learns disease mechanisms or causal pathways  
❌ Generalizes to pediatric, young adult, or healthy populations  
❌ Solves rare disease prediction (fundamental data limitation)  

### Key Limitations to State
1. **Task is recurrence prediction (70%), not onset prediction**
2. **Population-specific:** Multimorbid adults only (median age 62, 44 prior diagnoses)
3. **Code type mixing:** 14.4% non-disease codes in targets
4. **Extreme imbalance:** 133 rare diseases have <100 samples (insufficient signal)
5. **Performance dominated by common chronic diseases**
6. **No causal interpretation** - statistical associations only

### Target Venues
- Applied ML in healthcare: CHIL, ML4H
- Clinical informatics: JAMIA, JBI
- Medical AI with honest limitations: Nature Digital Medicine, npj Digital Medicine

---

## 🗂️ Project Structure

```
BEHRT_thesis/  (current directory)
├── data/
│   ├── processed/
│   │   ├── train_nextvisit_ccsr_clean.parquet
│   │   ├── val_nextvisit_ccsr_clean.parquet
│   │   ├── test_nextvisit_ccsr_clean.parquet
│   │   └── vocab_ccsr_clean.pkl
│   └── models/
│       └── clean_runs/
│           ├── clean_run_20260826_133741/  # Baseline
│           └── clean_run_20260827_130400/  # Pos_weight
├── scripts/
│   └── train_nextvisit_clean.py           # Main training script
├── eda/
│   ├── 01_dataset_structure.py            # ✅ Completed
│   ├── 02_patient_level_analysis.py       # ✅ Completed
│   ├── 03_label_support_analysis.py       # ✅ Completed
│   └── results/
│       ├── 01_dataset_structure.json
│       ├── 02_patient_level_analysis.json
│       └── 03_label_support_analysis.json
├── model/                                  # BEHRT model implementation
├── dataLoader/                             # Data loading utilities
├── common/                                 # Common utilities
└── README.md                               # This file
```

---

## 🚀 Quick Start

### Training with Positive-Class Weighting
```bash
# With pos_weight (recommended for rare code detection)
MLFLOW_ENABLE=0 USE_POS_WEIGHT=1 MAX_POS_WEIGHT=30.0 SEED=42 EPOCHS=3 \
  python scripts/train_nextvisit_clean.py

# Baseline (no weighting)
MLFLOW_ENABLE=0 USE_POS_WEIGHT=0 SEED=42 EPOCHS=3 \
  python scripts/train_nextvisit_clean.py
```

### Running EDA
```bash
# Dataset structure analysis
python3.12 eda/01_dataset_structure.py

# Patient-level analysis
python3.12 eda/02_patient_level_analysis.py

# Label support & class imbalance analysis
python3.12 eda/03_label_support_analysis.py
```

---

## 📋 Next Steps (Priority Order)

### Immediate Priorities
1. **[ ] Complete remaining EDA** (steps 4-7)
2. **[ ] Create comparison table** - Baseline vs pos_weight (clean side-by-side metrics)
3. **[ ] Disease-level performance** - Break down by common vs rare, new vs recurrent
4. **[ ] Code-type stratified analysis** - Separate diseases vs symptoms vs admin codes
5. **[ ] NEW vs RECURRING performance** - Novel analysis comparing 70% recurrent vs 30% new

### Analysis Opportunities (Novel Contributions)
- **Recurrence vs onset stratified evaluation** - Show model excels at recurrence, weaker at onset
- **Pos_weight effectiveness by code type** - Does weighting help new codes more than recurrent?
- **Temporal persistence modeling** - Analyze which diseases recur consistently vs sporadically

### Documentation
1. **[ ] Write methods section** with dataset characterization
2. **[ ] Write limitations section** FIRST (before adding experiments)
3. **[ ] Frame results** around care coordination utility
4. **[ ] Calibration analysis** - Threshold sensitivity, per-class calibration

### What NOT to Do
- ❌ Deep mechanistic interpretability (low ROI for thesis timeline)
- ❌ Architecture experiments without clear hypothesis
- ❌ Trying to fundamentally solve 52,559:1 imbalance (it's a data limit)
- ❌ Overclaiming about rare disease prediction

---

## 🔬 Technical Details

### Environment
- **Python:** 3.12
- **Platform:** macOS
- **Framework:** PyTorch
- **Data format:** Parquet (with numpy arrays in cells)

### Key Dependencies
- PyTorch (transformer model)
- pandas (parquet data loading)
- numpy (array operations)
- scikit-learn (metrics: APS, AUC, F1)
- pickle (vocabulary loading)

### Reproducibility
- All experiments use `SEED=42`
- Same data splits across all runs
- Deterministic training when possible

---

## 📚 References

**Similar published work (honest recurrence/code prediction):**
- **Clinical BERT** - Predicts hospital readmission (recurrence task)
- **BEHRT** - Next diagnosis code prediction (mixed new + recurrent)
- **Hi-BEHRT** - Hierarchical disease prediction (includes recurrence patterns)

None of these are "pure onset prediction" — all work with existing patient populations!

---

## 📝 Notes & Insights

### Why This Is Still Valuable Research
Even though it's primarily recurrence prediction (not onset), this has **real clinical utility**:
- **Resource allocation:** Which patients need intensive follow-up?
- **Care coordination:** Schedule specialists for predicted active conditions
- **Preventive intervention:** Predict COPD exacerbation → proactive pulmonology consult
- **Clinical decision support:** "These 9 conditions will likely be active next visit"

### Honest Framing Makes It Stronger
Reviewers will appreciate:
- Careful dataset analysis and transparent limitations
- Clear distinction between what model does vs doesn't do
- Focus on practical clinical utility, not overclaimed capabilities
- Evidence-based understanding of 70/30 recurrence/onset split

---

## 🤝 Contributing

This is a thesis project. Analysis and experiments follow the publication readiness strategy outlined in `../../docs/publication_readiness_summary.md`.

---

**Last Updated:** August 29, 2026  
**Status:** EDA in progress (3/7 complete), experimental runs complete, publication strategy defined  
**See also:** `../../docs/publication_readiness_summary.md` for complete thesis strategy
