"""
Per-Disease Category Performance Analysis
Analyzes which disease categories your BEHRT model predicts best/worst
"""
import pandas as pd
import numpy as np
import pickle
import torch
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import sys
sys.path.append('..')

import pytorch_pretrained_bert as Bert
from model.NextXVisit import BertForMultiLabelPrediction
from dataLoader.NextXVisit import NextVisit
from torch.utils.data import DataLoader

# Load CCSR descriptions
def load_ccsr_descriptions():
    df = pd.read_csv('../cssr_mappings/DXCCSR_v2025-1/DXCCSR_v2025-1.csv', low_memory=False)
    ccsr_map = df[["'Default CCSR CATEGORY IP'", "'Default CCSR CATEGORY DESCRIPTION IP'"]].drop_duplicates()
    ccsr_map.columns = ['code', 'desc']
    desc_dict = {}
    for _, row in ccsr_map.iterrows():
        code = row['code'].strip("'")
        desc = row['desc'].strip("'")
        desc_dict[code] = desc
    return desc_dict

def format_label_vocab(token2idx):
    """Create label vocabulary by removing special tokens"""
    token2idx = token2idx.copy()
    del token2idx['PAD']
    del token2idx['SEP']
    del token2idx['CLS']
    del token2idx['MASK']
    token = list(token2idx.keys())
    labelVocab = {}
    for i,x in enumerate(token):
        labelVocab[x] = i
    return labelVocab

class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')

print("="*70)
print("BEHRT MODEL: PER-DISEASE CATEGORY PERFORMANCE ANALYSIS")
print("="*70)

# Load vocabulary
with open('../data/processed/vocab_ccsr.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load age vocabulary
with open('../data/processed/age_vocab.pkl', 'rb') as f:
    age_vocab = pickle.load(f)

BertVocab = vocab
ageVocab = age_vocab['age2idx']
labelVocab = format_label_vocab(BertVocab['token2idx'])

print(f"Main vocabulary size: {len(BertVocab['token2idx'])}")
print(f"Label vocabulary size: {len(labelVocab)}")
print(f"Age vocabulary size: {len(ageVocab)}")

# Load test data
test_df = pd.read_parquet('../data/processed/test_nextvisit_ccsr.parquet')
print(f"Test samples: {len(test_df):,}")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Model config matching the QUICK training setup
model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()),
    'hidden_size': 144,  # QUICK model size
    'seg_vocab_size': 2,
    'age_vocab_size': len(ageVocab.keys()),
    'max_position_embedding': 64,
    'hidden_dropout_prob': 0.1,
    'num_hidden_layers': 3,  # QUICK model layers
    'num_attention_heads': 6,  # QUICK model heads
    'attention_probs_dropout_prob': 0.1,
    'intermediate_size': 256,  # QUICK model size
    'hidden_act': 'gelu',
    'initializer_range': 0.02,
}

feature_dict = {
    'word': True,
    'seg': True,
    'age': True,
    'position': True
}

# Create and load model
conf = BertConfig(model_config)
model = BertForMultiLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict)

# Load pretrained weights
def load_model(path, model):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

model_path = '../data/models/quick/behrt_nextvisit_ccsr_quick.pt'
print(f"Loading model from: {model_path}")
model = load_model(model_path, model)
model.to(device)
model.eval()
print("Model loaded successfully!")

# Create test dataset
test_dataset = NextVisit(token2idx=BertVocab['token2idx'], label2idx=labelVocab, age2idx=ageVocab, 
                        dataframe=test_df, max_len=64)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Setup MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=list(labelVocab.values()))
mlb.fit([[each] for each in list(labelVocab.values())])

print(f"Model output dimensions: {len(labelVocab.keys())}")
print(f"MLB classes: {len(mlb.classes_)}")

# Collect predictions
all_preds = []
all_labels = []

print("\nRunning predictions on test set...")
with torch.no_grad():
    for step, batch in enumerate(test_loader):
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch
        
        # Transform targets using MLB
        targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32)
        
        # Move to device
        input_ids = input_ids.to(device)
        age_ids = age_ids.to(device)
        segment_ids = segment_ids.to(device)
        posi_ids = posi_ids.to(device)
        attMask = attMask.to(device)
        
        # Get predictions - handle both training and inference mode
        outputs = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask, labels=None)
        
        # Handle different return formats
        if isinstance(outputs, tuple):
            _, logits = outputs  # (loss, logits) format when labels provided
        else:
            logits = outputs  # Just logits when labels=None
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_preds.append(probs)
        all_labels.append(targets.numpy())
        
        if step % 100 == 0:
            print(f"Processed batch {step}/{len(test_loader)}")

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

print(f"Predictions shape: {all_preds.shape}")
print(f"Labels shape: {all_labels.shape}")

# Calculate per-category metrics
print("\n" + "="*70)
print("PER-DISEASE CATEGORY ANALYSIS")
print("="*70)

ccsr_descriptions = load_ccsr_descriptions()

# Create mapping between labelVocab indices and tokens
idx2label = {v: k for k, v in labelVocab.items()}

# For each disease category (column), calculate AP
category_metrics = []

for col_idx in range(all_labels.shape[1]):
    token = idx2label.get(col_idx, f"UNK_{col_idx}")
    
    # Skip if unknown token
    if token.startswith('UNK_'):
        continue
    
    y_true = all_labels[:, col_idx]
    y_pred = all_preds[:, col_idx]
    
    # Only calculate if there are positive samples
    n_positive = y_true.sum()
    if n_positive < 10:  # Need at least 10 positive samples for reliable metrics
        continue
    
    try:
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        # Extract base CCSR code for description lookup
        base_code = token.replace('CCSR_', '')
        description = ccsr_descriptions.get(base_code, token)
        
        category_metrics.append({
            'code': token,
            'description': description[:50],  # Truncate long descriptions
            'n_positive': int(n_positive),
            'prevalence': n_positive / len(y_true),
            'AP': ap,
            'AUC': auc
        })
    except Exception as e:
        print(f"Error processing {token}: {e}")
        continue

print(f"Successfully analyzed {len(category_metrics)} disease categories")

# Sort by AP
category_metrics = sorted(category_metrics, key=lambda x: x['AP'], reverse=True)

# Top 20 best predicted
print("\n" + "="*70)
print("TOP 20 BEST PREDICTED DISEASES (Highest Average Precision)")
print("="*70)
print(f"{'Rank':<5} {'CCSR Code':<15} {'AP':<8} {'AUC':<8} {'N':<8} {'Description'}")
print("-"*70)
for i, m in enumerate(category_metrics[:20], 1):
    print(f"{i:<5} {m['code']:<15} {m['AP']:.4f}   {m['AUC']:.4f}   {m['n_positive']:<8} {m['description']}")

# Bottom 20 worst predicted
print("\n" + "="*70)
print("BOTTOM 20 WORST PREDICTED DISEASES (Lowest Average Precision)")
print("="*70)
print(f"{'Rank':<5} {'CCSR Code':<15} {'AP':<8} {'AUC':<8} {'N':<8} {'Description'}")
print("-"*70)
for i, m in enumerate(category_metrics[-20:][::-1], 1):
    print(f"{i:<5} {m['code']:<15} {m['AP']:.4f}   {m['AUC']:.4f}   {m['n_positive']:<8} {m['description']}")

# Group by disease system (first 3 letters of CCSR)
print("\n" + "="*70)
print("PERFORMANCE BY DISEASE SYSTEM (CCSR Category)")
print("="*70)

system_names = {
    'BLD': 'Blood disorders',
    'CIR': 'Circulatory system',
    'DIG': 'Digestive system',
    'END': 'Endocrine/Metabolic',
    'EXT': 'External causes',
    'EYE': 'Eye disorders',
    'FAC': 'Factors influencing health',
    'GEN': 'Genitourinary system',
    'INF': 'Infectious diseases',
    'INJ': 'Injury/Poisoning',
    'MAL': 'Malignant neoplasms',
    'MBD': 'Mental/Behavioral disorders',
    'MUS': 'Musculoskeletal system',
    'NEO': 'Neoplasms',
    'NVS': 'Nervous system',
    'PNL': 'Perinatal conditions',
    'PRG': 'Pregnancy/Childbirth',
    'RSP': 'Respiratory system',
    'SKN': 'Skin disorders',
    'SYM': 'Symptoms/Signs',
    'XXX': 'Unspecified/Other'
}

system_metrics = defaultdict(list)
for m in category_metrics:
    # Extract system from code like CCSR_CIR007
    code = m['code'].replace('CCSR_', '')
    system = code[:3]
    system_metrics[system].append(m['AP'])

system_avg = []
for system, aps in system_metrics.items():
    system_avg.append({
        'system': system,
        'name': system_names.get(system, system),
        'mean_AP': np.mean(aps),
        'n_categories': len(aps)
    })

system_avg = sorted(system_avg, key=lambda x: x['mean_AP'], reverse=True)

print(f"\n{'System':<6} {'Mean AP':<10} {'#Codes':<8} {'Description'}")
print("-"*60)
for s in system_avg:
    print(f"{s['system']:<6} {s['mean_AP']:.4f}     {s['n_categories']:<8} {s['name']}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
aps = [m['AP'] for m in category_metrics]
print(f"Total disease categories analyzed: {len(category_metrics)}")
print(f"Mean AP across all categories: {np.mean(aps):.4f}")
print(f"Median AP: {np.median(aps):.4f}")
print(f"Std AP: {np.std(aps):.4f}")
print(f"Best category AP: {max(aps):.4f}")
print(f"Worst category AP: {min(aps):.4f}")

# Prevalence vs Performance analysis
print("\n" + "="*70)
print("PREVALENCE VS PERFORMANCE (Does more data help?)")
print("="*70)

# Bin by prevalence
low_prev = [m for m in category_metrics if m['prevalence'] < 0.01]
mid_prev = [m for m in category_metrics if 0.01 <= m['prevalence'] < 0.05]
high_prev = [m for m in category_metrics if m['prevalence'] >= 0.05]

if low_prev:
    print(f"\nLow prevalence (<1%): {len(low_prev)} categories, Mean AP: {np.mean([m['AP'] for m in low_prev]):.4f}")
if mid_prev:
    print(f"Medium prevalence (1-5%): {len(mid_prev)} categories, Mean AP: {np.mean([m['AP'] for m in mid_prev]):.4f}")
if high_prev:
    print(f"High prevalence (>5%): {len(high_prev)} categories, Mean AP: {np.mean([m['AP'] for m in high_prev]):.4f}")

print("\nInsight: Higher prevalence diseases tend to have better prediction performance")
print("because the model has more examples to learn from.")

print("\n" + "="*70)
print("MODEL PERFORMANCE ANALYSIS COMPLETE")
print("="*70)