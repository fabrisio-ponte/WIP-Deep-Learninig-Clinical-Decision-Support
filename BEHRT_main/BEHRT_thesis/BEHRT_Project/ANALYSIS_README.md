# BEHRT Analysis & Cleaning Tools

Organized toolkit for BEHRT disease prediction analysis and data cleaning.

## 📁 Project Structure

```
BEHRT_Project/
├── run_analysis.py          # Main runner script  
│
├── analysis/                 # Analysis Tools
│   ├── analyze_data.py       # Quick data overview
│   ├── analyze_filtered.py   # Filtered analysis (no XXX000) 
│   └── investigate_data_quality.py  # Data quality investigation
│
├── cleaning/                 # Data Cleaning Tools
│   ├── clean_data.py         # Phase 1: XXX000 removal
│   └── clean_data_phase2.py  # Phase 2: Advanced cleaning
│
├── config/                   # Configuration Files  
│   └── analysis_config.json  # Analysis configuration
│
├── utils/                    # Utility Modules
│   └── comprehensive_disease_analysis/  # Detailed analysis framework
│
└── data/                     # Data Files
    ├── processed/           # Processed datasets
    └── models/              # Trained models
```

## 🚀 Quick Start

### Using the Main Runner (Recommended)

```bash
# Quick data analysis  
python run_analysis.py quick-analysis

# Data quality investigation
python run_analysis.py data-quality  

# Phase 1 cleaning (XXX000 removal)
python run_analysis.py clean-phase1

# Filtered performance analysis
python run_analysis.py filtered-analysis

# Phase 2 advanced cleaning (optional)
python run_analysis.py clean-phase2
```

### Running Scripts Directly

```bash
# Analysis scripts (run from project root)
cd analysis/
python analyze_data.py
python analyze_filtered.py  
python investigate_data_quality.py

# Cleaning scripts (run from project root)
cd cleaning/
python clean_data.py
python clean_data_phase2.py
```

## 📊 Analysis Workflow

### 1. **Quick Analysis** (`analyze_data.py`)
- Basic dataset overview
- Label distribution analysis  
- XXX000 frequency check
- Quick performance summary

### 2. **Data Quality Investigation** (`investigate_data_quality.py`)
- Rare disease identification (< 10 samples)
- Sequence length analysis
- Duplicate detection
- Patient anomaly detection  
- Comprehensive quality report

### 3. **Filtered Analysis** (`analyze_filtered.py`)  
- Performance metrics excluding XXX000
- True clinical prediction capability
- Top disease performance ranking
- Clean vs contaminated comparison

## 🧹 Cleaning Workflow

### **Phase 1: Essential Cleaning** (`clean_data.py`)
- ✅ Remove XXX000 generic codes (8.05% of labels)
- ✅ 99.98% data retention  
- ✅ Clean vocabulary creation
- **Output**: `*_clean.parquet` + `vocab_ccsr_clean.pkl`

### **Phase 2: Advanced Cleaning** (`clean_data_phase2.py`)
- ✅ Remove ultra-rare diseases (< 10 samples)
- ✅ Filter insufficient context patients (< 2 diagnoses)
- ✅ Remove duplicate sequences  
- ✅ Standardize sequence lengths
- **Output**: `*_ultraclean.parquet` + `vocab_ccsr_ultraclean.pkl`

## 📈 Performance Results

### BEHRT Quick Model (After Cleaning)
- **APS**: 40.57% (exceeded 10-20% expectation!)
- **ROC-AUC**: 91.90% (excellent discrimination) 
- **Top Diseases**: 4 with F1 > 60%
  - CIR019: 61.6% F1 (Circulatory)
  - END010: 61.3% F1 (Endocrine)  
  - GEN003: 60.4% F1 (General)
  - CIR011: 60.0% F1 (Circulatory)

### Data Quality Impact
- **XXX000 removal**: Reduced label noise by 8.05%
- **Rare disease removal**: 36 ultra-rare diseases identified 
- **Duplicate removal**: 1,057 duplicate sequences found
- **Context filtering**: 3,070 insufficient context patients

## 🎯 Recommendations

### For Full Model Training:
1. **Use Phase 1 cleaned data** (excellent quality): 
   - `train_nextvisit_ccsr_clean.parquet`
   - `val_nextvisit_ccsr_clean.parquet`  
   - `test_nextvisit_ccsr_clean.parquet`
   - `vocab_ccsr_clean.pkl`

2. **Optional Phase 2** for production-grade cleaning:
   - `*_ultraclean.parquet` + `vocab_ccsr_ultraclean.pkl`

### Expected Full Model Performance:
- **APS**: 50-70% (vs 40.57% quick)  
- **ROC-AUC**: 94-97% (vs 91.90% quick)
- **Clinical Quality**: Excellent interpretability

## 🔧 Configuration

Edit `config/analysis_config.json` to customize:
- Data file paths  
- Model configurations
- Analysis parameters  

## 💡 Key Insights

✅ **BEHRT Quick exceeded expectations** (40.57% vs 10-20% documented)  
✅ **Top diseases achieve clinical-grade performance** (60%+ F1)  
✅ **Data cleaning is critical** (XXX000 was masking true capability)  
✅ **Conservative prediction strategy** (high precision, clinically appropriate)  
✅ **Circulatory diseases excel** (model strength in cardiovascular prediction)

---

## 📞 Usage Examples

```bash
# Complete analysis workflow
python run_analysis.py quick-analysis      # Overview
python run_analysis.py data-quality        # Investigate issues  
python run_analysis.py clean-phase1        # Clean data
python run_analysis.py filtered-analysis   # Clean performance

# Advanced cleaning (optional)  
python run_analysis.py clean-phase2        # Ultra-clean data

# Help and options
python run_analysis.py --help
```