# Running BEHRT with MIMIC-IV Data - Complete Guide

## 🚀 Quick Start (5 steps)

### 1. Install Dependencies
```bash
cd BEHRT_Project
pip install torch pandas numpy scikit-learn h5py pyarrow --break-system-packages
pip install pytorch-pretrained-bert --break-system-packages
pip install jupyter --break-system-packages
```

### 2. Verify Your MIMIC-IV Data Location
Make sure you can access these files:
```
/Users/telaclaimstech/Desktop/BERHT_gh/physionet.org/files/mimiciv/3.1/hosp/
├── patients.csv.gz
├── admissions.csv.gz
└── diagnoses_icd.csv.gz
```

### 3. Run Data Preprocessing
```bash
cd BEHRT_Project
python scripts/preprocess_mimic.py
```

This will create:
- `data/processed/train_mlm.parquet` - Training data for pre-training
- `data/processed/val_mlm.parquet` - Validation data
- `data/processed/test_mlm.parquet` - Test data
- `data/processed/vocab.pkl` - Medical code vocabulary
- `data/processed/age_vocab.pkl` - Age vocabulary
- `data/processed/train_nextvisit.parquet` - Training data for next visit prediction
- `data/processed/val_nextvisit.parquet` - Validation data for next visit
- `data/processed/test_nextvisit.parquet` - Test data for next visit

**Expected time:** 10-30 minutes depending on your system

### 4. Run MLM Pre-training
```bash
jupyter notebook notebooks/MLM.ipynb
```

**What to update in the notebook:**
- Cell 4: Update `file_config` paths:
  ```python
  file_config = {
      'vocab': '../data/processed/vocab.pkl',
      'data': '../data/processed/train_mlm.parquet',
      'model_path': '../data/models/',
      'model_name': 'behrt_mlm.pt',
      'file_name': 'mlm_training.log',
  }
  ```

**Expected time:** Several hours (depends on data size and GPU availability)

**Expected results:**
- Training loss: ~8-10 → ~2-4
- Precision: Should reach 0.3-0.5
- Model saved to: `data/models/behrt_mlm.pt`

### 5. Run Next Visit Prediction Fine-tuning
```bash
jupyter notebook notebooks/NextXVisit.ipynb
```

**What to update in the notebook:**
- Cell 3: Update `file_config` paths:
  ```python
  file_config = {
      'vocab': '../data/processed/vocab.pkl',
      'train': '../data/processed/train_nextvisit.parquet',
      'test': '../data/processed/test_nextvisit.parquet',
  }
  
  global_params = {
      'output_dir': '../data/models/',
      'best_name': 'behrt_nextvisit.pt',
      # ... rest of params
  }
  
  pretrain_model_path = '../data/models/behrt_mlm.pt'
  ```

**Expected time:** 1-2 hours

**Expected results:**
- Average Precision Score (APS): 0.15-0.30
- ROC-AUC: 0.65-0.75

---

## 📊 Understanding the Data Flow

### MIMIC-IV → BEHRT Format

**Input (MIMIC-IV):**
```
Patient 1001:
  Visit 1 (2020-01-15): I50.9, E11.9  (age 65)
  Visit 2 (2020-02-10): I50.9         (age 65)
  Visit 3 (2020-03-05): J44.0, E11.9  (age 65)
```

**Output (BEHRT Format):**
```python
{
    'patid': 1001,
    'code': ['CLS', 'I50.9', 'E11.9', 'SEP', 'I50.9', 'SEP', 'J44.0', 'E11.9', 'SEP'],
    'age':  [65,    65,      65,      65,    65,      65,    65,     65,      65]
}
```

### Special Tokens
- `CLS`: Classification token (start of sequence)
- `SEP`: Separator between visits
- `MASK`: Used during MLM pre-training (randomly masks tokens)
- `PAD`: Padding token for shorter sequences
- `UNK`: Unknown token for rare codes

---

## 🔧 Common Issues and Solutions

### Issue 1: Out of Memory Error
**Symptoms:** `RuntimeError: CUDA out of memory` or system freezing

**Solutions:**
```python
# In the notebook, reduce batch_size:
train_params = {
    'batch_size': 64,  # or even 32
    # ...
}

# Or reduce model size:
model_config = {
    'hidden_size': 144,  # instead of 288
    'num_hidden_layers': 4,  # instead of 6
    # ...
}
```

### Issue 2: No GPU Available
**Symptoms:** `CUDA not available`

**Solution:**
```python
# Use CPU instead:
train_params = {
    'use_cuda': False,
    'device': 'cpu',
    # ...
}
```

**Note:** Training on CPU will be significantly slower (10-20x)

### Issue 3: File Not Found Errors
**Symptoms:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:** Make sure you're running notebooks from the `notebooks/` directory and paths use `../` to go up one level:
```python
# Correct:
'vocab': '../data/processed/vocab.pkl'

# Wrong:
'vocab': 'data/processed/vocab.pkl'
```

### Issue 4: Module Not Found
**Symptoms:** `ModuleNotFoundError: No module named 'dataLoader'`

**Solution:** Add parent directory to path at the start of notebook:
```python
import sys
sys.path.insert(0, '../')
```

### Issue 5: Preprocessing Takes Too Long
**Solution:** Reduce the data size by:
```python
# In preprocess_mimic.py, after loading data:
diagnoses = diagnoses.sample(frac=0.1, random_state=42)  # Use 10% of data
```

---

## 📈 Monitoring Training

### During MLM Pre-training:
Watch for:
- Loss should decrease steadily
- Precision should increase to ~0.3-0.5
- If loss plateaus early (<20 epochs), might be underfitting

### During Next Visit Prediction:
Watch for:
- Loss should decrease in early epochs
- APS and ROC-AUC should improve
- Save the model when APS is highest

---

## 🎯 Expected File Sizes (approximate)

After preprocessing ~100K patients:
- `train_mlm.parquet`: 200-500 MB
- `vocab.pkl`: 1-5 MB
- `behrt_mlm.pt`: 50-150 MB (depends on model size)

---

## 💡 Tips for Better Results

1. **More data = better results**
   - Try to include as many patients as possible
   - Minimum 10,000 patients recommended

2. **Pre-training is crucial**
   - Don't skip the MLM pre-training step
   - Train for at least 20-30 epochs

3. **Hyperparameter tuning**
   - Try different learning rates (1e-5 to 5e-5)
   - Experiment with model sizes
   - Adjust max_len_seq based on your data

4. **Validation monitoring**
   - Always check validation metrics
   - Stop training if validation loss increases

---

## 📚 Additional Resources

- **BEHRT Paper:** https://arxiv.org/abs/1907.09538
- **MIMIC-IV Documentation:** https://mimic.mit.edu/docs/iv/
- **BERT Paper:** https://arxiv.org/abs/1810.04805

---

## 🐛 Debugging Checklist

Before asking for help, verify:
- [ ] All dependencies installed correctly
- [ ] MIMIC-IV data path is correct
- [ ] Preprocessing completed without errors
- [ ] File paths in notebooks are correct (using `../`)
- [ ] Sufficient disk space (>5 GB recommended)
- [ ] Python 3.7+ is being used

---

## 🎓 Understanding the Output

### Vocabulary Files (vocab.pkl)
```python
{
    'token2idx': {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK': 3, 'UNK': 4, 
                  'I50.9': 5, 'E11.9': 6, ...},
    'idx2token': {0: 'PAD', 1: 'CLS', 2: 'SEP', 3: 'MASK', 4: 'UNK',
                  5: 'I50.9', 6: 'E11.9', ...}
}
```

### Model Output (Next Visit Prediction)
For each patient, the model outputs probabilities for each possible diagnosis code:
```python
# Input: Patient history up to visit N
# Output: Probability for each code appearing in visit N+1
[0.01, 0.23, 0.45, 0.02, ...]  # Higher = more likely
```

---

## 🔄 Re-running After Changes

If you need to re-process data or retrain:

1. **Delete processed data:**
   ```bash
   rm -rf data/processed/*
   ```

2. **Delete saved models:**
   ```bash
   rm -rf data/models/*
   ```

3. **Re-run preprocessing:**
   ```bash
   python scripts/preprocess_mimic.py
   ```

4. **Restart notebooks:**
   - Kernel → Restart & Clear Output
   - Run all cells again

---

## ✅ Success Criteria

You'll know everything is working when:
- [ ] Preprocessing completes with ~1,000+ patients in training set
- [ ] Vocabulary size is >100 unique codes
- [ ] MLM training loss decreases to <4
- [ ] NextVisit training completes with APS >0.10
- [ ] Model files are saved in `data/models/`

Good luck! 🚀
