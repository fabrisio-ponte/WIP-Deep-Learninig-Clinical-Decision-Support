# BEHRT Project TODO

## ✅ COMPLETED (Data Cleaning & Analysis)

### Disease Performance Analysis
- [x] Set up comprehensive disease analysis framework  
- [x] Analyzed BEHRT Quick model (40.57% APS, 91.90% ROC-AUC)
- [x] Identified top diseases: CIR019 (61.6% F1), END010 (61.3% F1), GEN003 (60.4% F1)
- [x] Organized analysis utilities in structured directories

### Data Quality Issues  
- [x] Discovered XXX000 generic catch-all code (8.05% frequency, 148K instances)
- [x] Created data cleaning script (`clean_data.py`)
- [x] Generated cleaned datasets: `*_clean.parquet` + `vocab_ccsr_clean.pkl`
- [x] Verified 99.98% data retention, removed problematic codes
- [x] Filtered analysis shows true performance (4 diseases with F1 > 60%)

### Repository Management
- [x] Committed pre-cleaning checkpoint (commit: `aa22a7f`)
- [x] Structured codebase with analysis configurations

---

## 🚀 NEXT STEPS

### 1. Train Full BEHRT Model (HIGH PRIORITY)
- [ ] **Use cleaned data**: `train_nextvisit_ccsr_clean.parquet` + `vocab_ccsr_clean.pkl`  
- [ ] **Configuration**: 6 layers, 288 hidden size, 12 heads, 100% data
- [ ] **Expected time**: 4-12 hours on GPU
- [ ] **Expected performance**: 50-70% APS, 94-97% ROC-AUC

### 2. Model Comparison & Evaluation  
- [ ] Run comprehensive analysis on full model
- [ ] Compare Quick vs Full model results side-by-side
- [ ] Document performance scaling (30% → 100% data impact)  
- [ ] Validate that cleaned data improves interpretability

### 3. Advanced Analysis (OPTIONAL)
- [ ] Disease category deep-dive (Circulatory, Endocrine focus)
- [ ] Prediction confidence analysis  
- [ ] Rare disease handling strategies
- [ ] Clinical validation with domain experts

### 4. Production Considerations (FUTURE)
- [ ] Model deployment pipeline
- [ ] Real-time prediction API
- [ ] Performance monitoring dashboard
- [ ] Clinical decision support integration

---

## 📁 KEY FILES TO USE

**For Full Model Training:**
```
data/processed/
├── train_nextvisit_ccsr_clean.parquet  # Use this
├── val_nextvisit_ccsr_clean.parquet    # Use this  
├── test_nextvisit_ccsr_clean.parquet   # Use this
└── vocab_ccsr_clean.pkl                # Use this
```

**Analysis Scripts:**
- `analyze_filtered.py` - Clean performance metrics
- `clean_data.py` - Data quality pipeline  
- `utils/comprehensive_disease_analysis/` - Full analysis framework

---

## 💡 KEY INSIGHTS DISCOVERED

✅ **BEHRT Quick exceeded expectations**: 40.57% APS vs 10-20% documented  
✅ **Top diseases achieve 60%+ F1**: Excellent for medical AI  
✅ **Data cleaning is critical**: XXX000 was masking true performance  
✅ **Model is conservative**: High precision, clinically appropriate approach  
✅ **Circulatory diseases**: Best performing category (heart health focus)

---

## 🎯 IMMEDIATE NEXT ACTION

**Run full model training on cleaned data** - this will take several hours but should provide production-ready performance with excellent interpretability.