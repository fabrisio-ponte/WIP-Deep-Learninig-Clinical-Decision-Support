# 🚀 Quick vs Full Training Guide

## Two Training Options Available

I've created **TWO complete versions** of the BEHRT training pipeline:

---

## 📊 Comparison Table

| Feature | QUICK Version ⚡ | FULL Version 🎯 |
|---------|-----------------|----------------|
| **Total Time** | 1-1.5 hours | 4-12 hours |
| **Model Size** | ~15 MB | ~50 MB |
| **Hidden Size** | 144 | 288 |
| **Layers** | 3 | 6 |
| **Attention Heads** | 6 | 12 |
| **Batch Size** | 128 | 256 |
| **Data Used** | 30% | 100% |
| **Epochs (MLM)** | 10 | 50 |
| **Epochs (NextVisit)** | 10 | 50 |
| **Expected APS** | 0.10-0.20 | 0.15-0.30 |
| **Use Case** | Testing, learning, prototyping | Production, research, best results |

---

## ⚡ QUICK VERSION (Recommended to Start)

### Files to Use:
```
notebooks/MLM_QUICK.ipynb           → Pre-training (30-60 min)
notebooks/NextXVisit_QUICK.ipynb    → Fine-tuning (20-30 min)
```

### When to Use:
- ✅ First time using BEHRT
- ✅ Testing if everything works
- ✅ Learning how the model works
- ✅ Limited GPU/time resources
- ✅ Prototyping before full training
- ✅ Quick experiments

### What's Different:
- **Smaller model:** 3 layers instead of 6
- **Less data:** Uses 30% of processed data
- **Fewer epochs:** 10 instead of 50
- **Faster batches:** Batch size 128 vs 256

### Expected Results:
- MLM Loss: ~6 → ~3
- MLM Precision: ~0.2-0.4
- NextVisit APS: ~0.10-0.20
- NextVisit ROC-AUC: ~0.60-0.70

### Time Breakdown:
```
Preprocessing:     10-30 min  (same for both)
MLM Pre-training:  30-60 min  ⚡
NextVisit:         20-30 min  ⚡
──────────────────────────────
TOTAL:             1-2 hours  ⚡
```

---

## 🎯 FULL VERSION (Best Results)

### Files to Use:
```
notebooks/MLM.ipynb           → Pre-training (2-8 hours)
notebooks/NextXVisit.ipynb    → Fine-tuning (1-2 hours)
```

### When to Use:
- ✅ After successful quick test
- ✅ For research papers
- ✅ Production deployment
- ✅ Best possible performance
- ✅ Have GPU available
- ✅ Can wait for training

### What's Different:
- **Full model:** 6 layers, 288 hidden size
- **All data:** Uses 100% of processed data
- **More epochs:** 50 for better convergence
- **Larger batches:** Batch size 256

### Expected Results:
- MLM Loss: ~8 → ~2-4
- MLM Precision: ~0.3-0.5
- NextVisit APS: ~0.15-0.30
- NextVisit ROC-AUC: ~0.65-0.75

### Time Breakdown:
```
Preprocessing:     10-30 min  (same for both)
MLM Pre-training:  2-8 hours  🐌
NextVisit:         1-2 hours  🐌
──────────────────────────────
TOTAL:             4-12 hours 🐌
```

---

## 🎯 Recommended Workflow

### Step 1: Start with QUICK Version
```bash
# 1. Preprocess data (once, same for both)
python scripts/preprocess_mimic.py

# 2. Open Jupyter
jupyter notebook

# 3. Run QUICK notebooks
# - Open notebooks/MLM_QUICK.ipynb
# - Run all cells (30-60 min)
# - Open notebooks/NextXVisit_QUICK.ipynb  
# - Run all cells (20-30 min)
```

**Result:** Working model in 1-2 hours ✅

### Step 2: If Results Look Good, Run FULL Version
```bash
# Keep Jupyter open

# 4. Run FULL notebooks
# - Open notebooks/MLM.ipynb
# - Run all cells (2-8 hours)
# - Open notebooks/NextXVisit.ipynb
# - Run all cells (1-2 hours)
```

**Result:** Best possible model ✅

---

## 📁 Where Models Are Saved

### QUICK Version Models:
```
data/models/quick/
├── behrt_mlm_quick.pt           (Pretrained model)
├── behrt_nextvisit_quick.pt     (Final model)
└── mlm_training_quick.log       (Training log)
```

### FULL Version Models:
```
data/models/
├── behrt_mlm.pt                 (Pretrained model)
├── behrt_nextvisit.pt           (Final model)
└── mlm_training.log             (Training log)
```

**They don't overwrite each other!** You can have both.

---

## 🔄 Can I Switch Between Versions?

### ⚠️ Important: Models are NOT Compatible

You **cannot** mix quick and full versions:
- ❌ Don't use `behrt_mlm_quick.pt` with `NextXVisit.ipynb` (full)
- ❌ Don't use `behrt_mlm.pt` with `NextXVisit_QUICK.ipynb` (quick)

**Why?** Different model architectures (different hidden sizes, layers)

### ✅ Correct Combinations:

**QUICK Pipeline:**
```
MLM_QUICK.ipynb 
    ↓ produces behrt_mlm_quick.pt
    ↓
NextXVisit_QUICK.ipynb (loads behrt_mlm_quick.pt)
```

**FULL Pipeline:**
```
MLM.ipynb 
    ↓ produces behrt_mlm.pt
    ↓
NextXVisit.ipynb (loads behrt_mlm.pt)
```

---

## 💡 Which Should I Choose?

### Choose QUICK if:
- ⏰ You want results in 1-2 hours
- 🆕 First time using BEHRT
- 🔬 Just testing/learning
- 💻 CPU-only or limited GPU
- 🧪 Prototyping

### Choose FULL if:
- 📊 Need best performance
- 📝 Writing research paper
- 🏭 Production deployment
- 💪 Have good GPU
- ⏳ Can wait 4-12 hours

### My Recommendation:
```
Day 1: Run QUICK version (1-2 hours)
       ↓
       Verify everything works
       Check if results make sense
       ↓
Day 2: If satisfied, run FULL version (4-12 hours)
       ↓
       Get best results for your application
```

---

## 📊 Performance Expectations

### What "Good Results" Look Like:

**QUICK Version:**
- ✅ MLM Precision reaches 0.2-0.4
- ✅ NextVisit APS > 0.10
- ✅ Training completes without errors
- ✅ Loss decreases over epochs

**FULL Version:**
- ✅ MLM Precision reaches 0.3-0.5
- ✅ NextVisit APS > 0.15
- ✅ Better than random (APS > 0.05)
- ✅ Competitive with published results

---

## 🛠️ Troubleshooting

### QUICK Version Still Too Slow?

Further reduce in notebook:
```python
global_params = {
    'data_fraction': 0.1,  # Use only 10% of data
    # ...
}

# Run even fewer epochs
for e in range(5):  # Change from 10 to 5
```

### QUICK Version Runs Fine, FULL Crashes?

In FULL notebooks, reduce:
```python
train_params = {
    'batch_size': 128,  # Reduce from 256
    # ...
}
```

### Want Intermediate Option?

Modify QUICK version:
```python
model_config = {
    'hidden_size': 200,  # Between 144 and 288
    'num_hidden_layers': 4,  # Between 3 and 6
    # ...
}

global_params = {
    'data_fraction': 0.5,  # Use 50% of data
    # ...
}
```

---

## 📝 Summary Checklist

**Getting Started:**
- [ ] Downloaded BEHRT_Project.tar.gz
- [ ] Extracted project
- [ ] Ran preprocessing (once)

**QUICK Version (Start Here):**
- [ ] Open `notebooks/MLM_QUICK.ipynb`
- [ ] Run all cells (30-60 min)
- [ ] Model saved to `data/models/quick/behrt_mlm_quick.pt`
- [ ] Open `notebooks/NextXVisit_QUICK.ipynb`
- [ ] Run all cells (20-30 min)
- [ ] Results look reasonable (APS > 0.10)

**FULL Version (Optional, Better Results):**
- [ ] Open `notebooks/MLM.ipynb`
- [ ] Run all cells (2-8 hours)
- [ ] Model saved to `data/models/behrt_mlm.pt`
- [ ] Open `notebooks/NextXVisit.ipynb`
- [ ] Run all cells (1-2 hours)
- [ ] Results better than QUICK (APS > 0.15)

---

## 🎓 Key Takeaway

**You have flexibility!**

- Start with **QUICK** (1-2 hours) to verify everything works
- Move to **FULL** (4-12 hours) when you need best results
- Both are complete, independent pipelines
- Both save models in different locations (no conflicts)

**The QUICK version is perfect for learning and testing.**
**The FULL version is what you use for real work.**

Choose based on your needs and available time! 🚀
