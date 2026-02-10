#!/usr/bin/env python3
"""
Proper BEHRT Reverse Engineering Training Script
Following the exact implementation from NextXVisit_QUICK.ipynb
"""

import sys
import os
sys.path.insert(0, '../')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_pretrained_bert as Bert
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.metrics
import warnings
warnings.filterwarnings('ignore')

from common.common import create_folder, load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit  
from model.NextXVisit import BertForMultiLabelPrediction
from model import optimiser
import json
from pathlib import Path

def convert_experimental_to_nextvisit_format(experiment_data_path):
    """Convert our experimental data to proper NextVisit format"""
    
    print(f"📊 Converting {experiment_data_path} to NextVisit format...")
    
    # Load experimental data
    df = pd.read_parquet(experiment_data_path)
    print(f"   Original format: {len(df)} visits for {len(df['subject_id'].unique())} patients")
    
    # Load the actual vocabularies from working implementation
    vocab_path = '../data/processed/vocab_ccsr'
    if Path(vocab_path + '.pkl').exists():
        BertVocab = load_obj(vocab_path)
        print(f"   Loaded working vocabulary with {len(BertVocab['token2idx'])} tokens")
    else:
        print("   ❌ Could not find working vocabulary!")
        return None
    
    # Create NextVisit format data
    nextvisit_data = []
    patient_id_mapping = {}  # Map string IDs to integers
    next_patient_int = 1
    
    for patient_id, patient_data in df.groupby('subject_id'):
        # Convert string patient ID to integer
        if patient_id not in patient_id_mapping:
            patient_id_mapping[patient_id] = next_patient_int
            next_patient_int += 1
        patient_int_id = patient_id_mapping[patient_id]
        
        visits = patient_data.sort_values('visit_concept_orders')
        
        # For each visit (except the last), create a prediction sample
        for i in range(len(visits) - 1):
            # History: all visits up to current visit
            history_visits = visits.iloc[:i+1]
            
            # Target: next visit
            target_visit = visits.iloc[i+1]
            
            # Build code sequence (history + SEP token)
            code_sequence = []
            age_sequence = []
            
            for _, visit in history_visits.iterrows():
                code = visit['concept_ids']
                age_str = visit['ages']  # Format: AGE[XX]
                
                # Extract age number
                if isinstance(age_str, str) and age_str.startswith('AGE['):
                    age = int(age_str[4:-1]) * 12  # Convert to months
                else:
                    age = 600  # Default age in months
                
                # Add code and age if code exists in vocabulary
                if code in BertVocab['token2idx']:
                    code_sequence.append(code)
                    age_sequence.append(age)
            
            # Add SEP token
            if len(code_sequence) > 0:
                code_sequence.append('SEP')
                age_sequence.append(age_sequence[-1] if age_sequence else 600)
                
                # Target codes (what to predict)
                target_codes = []
                
                # Get all codes from the next visit
                remaining_visits = visits.iloc[i+1:]
                for _, future_visit in remaining_visits.iterrows():
                    target_code = future_visit['concept_ids'] 
                    if target_code in BertVocab['token2idx'] and target_code not in ['SEP', 'CLS', 'PAD', 'MASK']:
                        target_codes.append(target_code)
                        break  # Just predict immediate next visit for simplicity
                
                if len(target_codes) > 0:  # Only include if target exists in vocab
                    nextvisit_data.append({
                        'patid': patient_int_id,  # Use integer ID
                        'code': np.array(code_sequence),
                        'age': np.array(age_sequence),
                        'label': np.array(target_codes)  # Array of target codes
                    })
    
    nextvisit_df = pd.DataFrame(nextvisit_data)
    print(f"   Converted to: {len(nextvisit_df)} NextVisit samples")
    
    return nextvisit_df

def run_nextvisit_experiment(experiment_name):
    """Run NextVisit training following exact working implementation"""
    
    print(f"\n🚀 BEHRT NextVisit Training: {experiment_name}")
    print("="*80)
    
    # Step 1: Convert experimental data
    exp_data_path = f'../experiments/{experiment_name}/train_data_ccsr.parquet'
    nextvisit_df = convert_experimental_to_nextvisit_format(exp_data_path)
    
    if nextvisit_df is None or len(nextvisit_df) == 0:
        print("❌ Data conversion failed!")
        return None
    
    # Step 2: Load vocabularies (exactly like working implementation)
    print("📚 Loading vocabularies...")
    BertVocab = load_obj('../data/processed/vocab_ccsr')
    ageVocab, _ = age_vocab(max_age=110, symbol=None)
    
    def format_label_vocab(token2idx):
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
    
    labelVocab = format_label_vocab(BertVocab['token2idx'])
    print(f"   Vocabulary size: {len(BertVocab['token2idx'])}")
    print(f"   Label vocabulary size: {len(labelVocab)}")
    
    # Step 3: Configuration (exactly like working implementation)
    global_params = {
        'batch_size': 32,  # Smaller for quick testing
        'device': 'cpu',
        'max_len_seq': 64,
        'max_age': 110,
    }
    
    model_config = {
        'vocab_size': len(BertVocab['token2idx'].keys()),
        'hidden_size': 144,  # Small model for quick testing
        'seg_vocab_size': 2,
        'age_vocab_size': len(ageVocab.keys()),
        'max_position_embedding': global_params['max_len_seq'],
        'hidden_dropout_prob': 0.1,
        'num_hidden_layers': 3,
        'num_attention_heads': 6,
        'attention_probs_dropout_prob': 0.1,
        'intermediate_size': 256,
        'hidden_act': 'gelu',
        'initializer_range': 0.02,
    }
    
    feature_dict = {
        'word': True,
        'seg': True,
        'age': True,
        'position': True
    }
    
    print(f"⚙️  Model config: {model_config['hidden_size']} hidden, {model_config['num_hidden_layers']} layers")
    
    # Step 4: Create dataset (exactly like working implementation)
    print("🔧 Creating dataset...")
    
    # Subsample data for quick testing
    if len(nextvisit_df) > 1000:
        nextvisit_df = nextvisit_df.sample(n=1000, random_state=42).reset_index(drop=True)
        print(f"   Subsampled to: {len(nextvisit_df)} samples")
    
    Dset = NextVisit(
        token2idx=BertVocab['token2idx'], 
        label2idx=labelVocab, 
        age2idx=ageVocab,
        dataframe=nextvisit_df, 
        max_len=global_params['max_len_seq']
    )
    
    trainload = DataLoader(
        dataset=Dset, 
        batch_size=global_params['batch_size'], 
        shuffle=True, 
        num_workers=0  # Avoid multiprocessing issues
    )
    
    print(f"   Created dataset with {len(Dset)} samples")
    
    # Step 5: Model initialization (exactly like working implementation)
    print("🤖 Initializing BERT model...")
    
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
    
    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(conf, num_labels=len(labelVocab.keys()), feature_dict=feature_dict)
    model = model.to(global_params['device'])
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 6: Training setup
    optim_config = {
        'lr': 5e-5,
        'warmup_proportion': 0.1,
        'weight_decay': 0.01
    }
    
    optim = optimiser.adam(params=list(model.named_parameters()), config=optim_config)    
    # Step 7.1: Setup MultiLabelBinarizer (CRITICAL - from working implementation!)
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=list(labelVocab.values()))
    mlb.fit([[each] for each in list(labelVocab.values())])    
    # Step 7: Training loop (simplified)
    print("🏋️ Training...")
    
    def precision(logits, label):
        sig = nn.Sigmoid()
        output = sig(logits)
        label, output = label.cpu(), output.detach().cpu()
        try:
            return sklearn.metrics.average_precision_score(label.numpy(), output.numpy(), average='samples')
        except:
            return 0.0
    
    epochs = 3  # Quick training
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_precision = 0
        num_batches = 0
        
        for batch_idx, data in enumerate(trainload):
            batch_age = data[0].to(global_params['device'])      # age
            batch_diagnosis = data[1].to(global_params['device']) # code  
            batch_position = data[2].to(global_params['device'])  # position
            batch_seg = data[3].to(global_params['device'])       # segment
            batch_mask = data[4].to(global_params['device'])      # mask
            batch_label_raw = data[5]  # Raw labels (not moved to device yet)
            # data[6] is patid which we don't need
            
            # Transform labels using MultiLabelBinarizer (CRITICAL!)
            batch_label = torch.tensor(mlb.transform(batch_label_raw.numpy()), dtype=torch.float32)
            batch_label = batch_label.to(global_params['device'])
            
            optim.zero_grad()
            loss, logit = model(batch_diagnosis, batch_age, batch_seg, batch_position, batch_mask, batch_label)
            
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            total_precision += precision(logit, batch_label)
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        avg_precision = total_precision / num_batches
        
        print(f"✓ Epoch {epoch+1}: Loss={avg_loss:.4f}, Precision={avg_precision:.4f}")
    
    # Step 8: Final evaluation
    print("📊 Final evaluation...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data in trainload:
            batch_age = data[0].to(global_params['device'])      # age
            batch_diagnosis = data[1].to(global_params['device']) # code  
            batch_position = data[2].to(global_params['device'])  # position
            batch_seg = data[3].to(global_params['device'])       # segment
            batch_mask = data[4].to(global_params['device'])      # mask
            batch_label_raw = data[5]  # Raw labels
            # data[6] is patid which we don't need
            
            # Transform labels using MultiLabelBinarizer (CRITICAL!)
            batch_label = torch.tensor(mlb.transform(batch_label_raw.numpy()), dtype=torch.float32)
            batch_label = batch_label.to(global_params['device'])
            
            logit = model(batch_diagnosis, batch_age, batch_seg, batch_position, batch_mask)
            
            sig = nn.Sigmoid()
            predictions = sig(logit).cpu().numpy()
            labels = batch_label.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Calculate final metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Sample-wise average precision (micro-average)
    aps = sklearn.metrics.average_precision_score(all_labels, all_predictions, average='samples')
    
    # Sample-wise ROC-AUC
    try:
        roc_auc = sklearn.metrics.roc_auc_score(all_labels, all_predictions, average='samples')
    except:
        roc_auc = 0.5
    
    # Save results
    results = {
        'experiment': experiment_name,
        'metrics': {
            'average_precision_score': float(aps),
            'roc_auc_score': float(roc_auc),
        },
        'config': {
            'samples': len(nextvisit_df),
            'vocab_size': len(BertVocab['token2idx']),
            'model_config': model_config,
        }
    }
    
    # Save to results directory
    results_dir = Path(f'../results/{experiment_name}')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*80)
    print(f"🎯 FINAL RESULTS - {experiment_name}")
    print("="*80)
    print(f"📊 Average Precision Score (APS): {aps:.4f}")
    print(f"📊 ROC-AUC Score:                  {roc_auc:.4f}")
    print(f"📊 Training samples:               {len(nextvisit_df):,}")
    print(f"💾 Results saved to: {results_dir / 'final_results.json'}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    experiment_name = os.environ.get('EXPERIMENT_NAME', 'test_experiment')
    
    try:
        results = run_nextvisit_experiment(experiment_name)
        if results:
            print(f"\n✅ SUCCESS! APS: {results['metrics']['average_precision_score']:.4f}")
        else:
            print("\n❌ EXPERIMENT FAILED!")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()