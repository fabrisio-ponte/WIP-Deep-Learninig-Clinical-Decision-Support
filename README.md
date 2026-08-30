# BEHRT Workspace - Thesis Project [WIP]

**Project:** Next-Visit Clinical Code Prediction using BEHRT-style Transformers  
**Status:** Work in Progress  
**Last Updated:** August 29, 2026

---

## Workspace Organization

This workspace contains multiple BEHRT-related projects organized by purpose:

```
BEHRT_WIP/
├── BEHRT_main/                         # Main thesis work (MIMIC-IV data)
│   ├── BEHRT_core/                     # Original BEHRT implementation
│   ├── BEHRT_thesis/                   # Thesis-specific code & experiments
│   └── physionet.org/                  # MIMIC-IV data reference
│
├── BEHRT_controlled_experiments/       # Synthetic data experiments
│                                       # (reverse engineering, mechanistic testing)
│
├── BEHRT_mechanism_interpretability/   # Future: Model interpretability work
│                                       # (attention analysis, feature attribution)
│
├── docs/                               # All documentation
│   ├── publication_readiness_summary.md
│   ├── Week1_Architectural_Review.md
│   └── setup_reverse_engineering.py
│
└── LICENSE
```

---

## Project Directories

### 1. BEHRT_main/
**Purpose:** Primary thesis work with real MIMIC-IV data

**Subdirectories:**
- **`BEHRT_core/`** - Original BEHRT implementation (base model, utilities)
- **`BEHRT_thesis/`** - **ACTIVE WORK HERE**
  - Next-visit clinical code prediction
  - Cleaned CCSR-coded sequences
  - Positive-class weighting experiments
  - EDA analysis (3/7 complete)
  - Model training and evaluation

**Key Focus:** Clinical code recurrence prediction for multimorbid adults (70% recurrence, 30% new diagnoses)

**See:** `BEHRT_main/BEHRT_thesis/README.md` for detailed thesis work

---

### 2. BEHRT_controlled_experiments/
**Purpose:** Synthetic data experiments for understanding model behavior

**Contents:**
- Reverse engineering experiments with controlled data
- Mechanistic hypothesis testing
- Simplified scenarios to isolate model capabilities
- Performance analysis on synthetic progressions

**Status:** Completed exploratory experiments

**See:** `BEHRT_controlled_experiments/REVERSE_ENGINEERING_RESULTS.md`

---

### 3. BEHRT_mechanism_interpretability/
**Purpose:** Future work on model interpretability (not started)

**Planned Work:**
- Attention pattern analysis
- Feature attribution methods
- Mechanism discovery
- Clinical decision justification

**Status:** Not started (deprioritized per publication strategy)

**Note:** Per `docs/publication_readiness_summary.md`, deep mechanistic interpretability has lower ROI for thesis timeline. Focus remains on rigorous evaluation and honest limitations.

---

### 4. docs/
**Purpose:** All project documentation

**Key Documents:**
- **`publication_readiness_summary.md`** - **READ THIS FIRST**
  - Comprehensive thesis strategy after EDA findings
  - What we can/cannot claim
  - Limitations to state
  - Publication framing (recurrence vs onset)
  
- **`Week1_Architectural_Review.md`** - Initial architectural analysis

- **`setup_reverse_engineering.py`** - Controlled experiment setup script

---

## Current Work Status

### Completed
- [x] Baseline training run (no pos_weight, AUC 0.891, APS 0.262)
- [x] Pos_weight training run (MAX_POS_WEIGHT=30, AUC 0.901, APS 0.274)
- [x] EDA Step 1: Dataset structure analysis
- [x] EDA Step 2: Patient-level analysis (discovered 70/30 recurrence/new split)
- [x] EDA Step 3: Label support & class imbalance (found 52,559:1 ratio)
- [x] Publication readiness strategy document

### In Progress
- [ ] EDA Step 4: Temporal structure analysis
- [ ] EDA Step 5: Label co-occurrence patterns
- [ ] EDA Step 6: Split representativeness
- [ ] EDA Step 7: Cleaning impact analysis

### Next Priorities
1. Complete remaining EDA (steps 4-7)
2. Create comparison table: Baseline vs pos_weight
3. Disease-level performance breakdown
4. NEW vs RECURRING stratified analysis (novel contribution)
5. Code-type stratified analysis (diseases vs symptoms vs admin)

---

## Key Findings (August 2026)

### Critical Dataset Insights
- **179,713 samples** from **~25,000 patients** (~7 snapshots each)
- **Median patient:** 62 years, 44 prior diagnoses, 9 next-visit codes
- **70% recurrence, 30% new** - This is NOT disease onset prediction!
- **14.4% non-disease codes** (symptoms 7%, admin 4%, injuries 2.7%)
- **Extreme imbalance:** 52,559:1 ratio (diabetes to rarest disease)

### Reframed Thesis
**Original:** Disease onset prediction  
**Actual:** Next-visit clinical code recurrence prediction for multimorbid adults

**Clinical utility:** Care coordination, resource allocation, specialist scheduling

**Honest limitations:** 
- Adult-only (median age 62, 0% under 18)
- Multimorbid population (median 44 diagnoses)
- Cannot generalize to healthy/young populations
- Performance dominated by common chronic diseases

---

## Quick Start

### Work on Thesis Project
```bash
cd BEHRT_main/BEHRT_thesis

# Run training with pos_weight
MLFLOW_ENABLE=0 USE_POS_WEIGHT=1 MAX_POS_WEIGHT=30.0 SEED=42 EPOCHS=3 \
  python scripts/train_nextvisit_clean.py

# Run EDA
python3.12 eda/01_dataset_structure.py
python3.12 eda/02_patient_level_analysis.py
python3.12 eda/03_label_support_analysis.py
```

### Read Documentation
```bash
# Start here for thesis strategy
open docs/publication_readiness_summary.md

# Or view in terminal
cat docs/publication_readiness_summary.md | less
```

---

## References

**Key Insight:** Similar published papers (Clinical BERT, BEHRT, Hi-BEHRT) all predict code recurrence/readmission, NOT pure disease onset. Our honest framing aligns with established literature.

---

## Development Guidelines

### Where to Work
- **Thesis experiments** → `BEHRT_main/BEHRT_thesis/`
- **Documentation** → `docs/`
- **Controlled tests** → `BEHRT_controlled_experiments/`
- **Interpretability** → `BEHRT_mechanism_interpretability/` (future)

### Commit Convention
Using conventional commits (see `.gitmemory`):
- `feat:` - New features
- `docs:` - Documentation
- `fix:` - Bug fixes
- `refactor:` - Code restructuring
- `WIP:` - Work in progress

---

## Project Context

This workspace supports a thesis on transformer-based clinical code prediction. The work has been reframed (August 2026) from "disease onset prediction" to the more accurate and defensible "next-visit clinical code recurrence prediction for multimorbid adults" based on comprehensive EDA findings.

**Core contribution:** Demonstrates transformer effectiveness on longitudinal EHR with extreme class imbalance (52,559:1), using positive-class weighting to improve rare code detection (+4.2% APS).

---

**Last Updated:** August 29, 2026  
**Primary Focus:** `BEHRT_main/BEHRT_thesis/` (thesis work with MIMIC-IV data)  
**See:** `docs/publication_readiness_summary.md` for complete strategy
