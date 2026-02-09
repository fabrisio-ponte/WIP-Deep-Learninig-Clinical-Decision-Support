# BEHRT - BERT for Electronic Health Records

This project implements BEHRT (BERT for Electronic Health Records) for learning representations from EHR data.

## Quick Start

### 1. Installation

```bash
cd BEHRT_Project
pip install -r requirements.txt
```

### 2. Data Preprocessing

Process your MIMIC-IV data:

```bash
python scripts/preprocess_mimic.py
```

This will:
- Load MIMIC-IV data from the specified path
- Convert to BEHRT format
- Create vocabularies
- Split into train/val/test sets
- Save processed data to `data/processed/`

### 3. Model Training

#### Phase 1: Pre-training with Masked Language Model (MLM)

```bash
cd notebooks
jupyter notebook MLM.ipynb
```

This trains the model to predict masked medical codes, learning general EHR representations.

#### Phase 2: Fine-tuning for Next Visit Prediction

```bash
cd notebooks
jupyter notebook NextXVisit.ipynb
```

This fine-tunes the pre-trained model to predict diagnoses in the next hospital visit.

## Project Structure

```
BEHRT_Project/
├── common/              # Utility functions
├── dataLoader/          # Data loading and processing
├── model/              # Model definitions
├── preprocessing/      # Data preprocessing scripts
├── notebooks/          # Jupyter notebooks for training
├── scripts/           # Helper scripts
└── data/              # Data directory
    ├── raw/           # Raw MIMIC-IV data
    ├── processed/     # Processed data
    └── models/        # Saved models
```

## Configuration

Key parameters to adjust in the notebooks:

- `batch_size`: Default 256 (reduce if out of memory)
- `max_len_seq`: Maximum sequence length (64 for MLM, 100 for NextVisit)
- `hidden_size`: Model dimension (288)
- `num_hidden_layers`: Number of transformer layers (6)
- `device`: 'cuda:0' for GPU, 'cpu' for CPU

## Expected Results

### MLM Pre-training
- Training loss: Should decrease from ~8-10 to ~2-4
- Precision: Should reach 0.3-0.5

### Next Visit Prediction
- Average Precision Score: 0.15-0.30
- ROC-AUC: 0.65-0.75

## Citation

```
@article{li2020behrt,
  title={BEHRT: Transformer for Electronic Health Records},
  author={Li, Yikuan and Rao, Shishir and Solares, José Roberto Ayala and others},
  journal={Scientific Reports},
  year={2020}
}
```

## License

Please ensure you have proper access to MIMIC-IV data through PhysioNet.
