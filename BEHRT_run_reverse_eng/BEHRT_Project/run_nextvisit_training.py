#!/usr/bin/env python3
"""
Automated BEHRT NextVisit Training Script
Converts experimental datasets to actual performance metrics
"""

import sys
import os
sys.path.insert(0, '../')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pytorch_pretrained_bert as Bert
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import BEHRT modules
from common.common import create_folder, load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit  
from model.NextXVisit import BertForMultiLabelPrediction

def run_nextvisit_training(experiment_name=None):
    """Run NextVisit training and return performance metrics"""
    
    print(f"🚀 Starting NextVisit training for {experiment_name or 'experiment'}")
    
    # Configuration
    config = {
        'vocab': '../data/processed/vocab_ccsr',  
        'train': '../data/processed/train_data_ccsr.parquet',
        'model_path': f'../data/models/quick/',
        'output_dir': f'../results/{experiment_name}/model_outputs/' if experiment_name else '../results/',
        
        # Quick training config
        'max_seq_len': 128,
        'max_age': 110,
        'batch_size': 32,  # Small for quick training
        'epochs': 5,      # Few epochs for quick testing
        'hidden_size': 144,
        'learning_rate': 5e-5,
        
        # Random seed
        'seed': 42
    }
    
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Config: {config['epochs']} epochs, batch_size={config['batch_size']}")
    
    try:
        # Load vocabularies
        print("📚 Loading vocabularies...")
        if Path(config['vocab'] + '.pkl').exists():
            code2idx = load_obj(config['vocab'])
            print(f"✓ Loaded vocabulary with {len(code2idx)} codes")
        else:
            print("❌ Vocabulary not found. Creating dummy vocab...")
            # Create dummy vocabulary for our experimental data
            code2idx = {f'CCSR_{i:03d}': i for i in range(500)}
            
        # Create age vocabulary 
        age2idx, idx2age = age_vocab(max_age=config['max_age'], mon=12)  # Use years
        print(f"✓ Created age vocabulary: 0-{config['max_age']} years")
        
        # Load training data
        print(f"📋 Loading training data from {config['train']}...")
        if Path(config['train']).exists():
            df = pd.read_parquet(config['train'])
            print(f"✓ Loaded {len(df)} records for {len(df['subject_id'].unique())} patients")
            
            # Convert our experimental format to NextVisit format
            print("🔄 Converting data format...")
            nextvisit_df = convert_to_nextvisit_format(df)
            print(f"✓ Converted to {len(nextvisit_df)} next-visit prediction samples")
            
        else:
            print("❌ Training data not found!")
            return None
        
        # Create dataset and dataloader
        print("🔧 Creating dataset...")
        train_dataset = NextVisit(
            token2idx=code2idx,
            label2idx=code2idx,  # Using same vocab for labels
            age2idx=age2idx, 
            dataframe=nextvisit_df,
            max_len=config['max_seq_len'],
            code='history_codes',
            age='history_ages', 
            label='target_codes'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        print(f"✓ Created dataset with {len(train_dataset)} samples")
        
        # Initialize model
        print("🤖 Initializing BEHRT model...")
        
        # Create simple config object
        class SimpleConfig:
            def __init__(self):
                self.vocab_size = len(code2idx)
                self.hidden_size = config['hidden_size']
                self.num_hidden_layers = 2
                self.num_attention_heads = 2
                self.intermediate_size = config['hidden_size'] * 4
                self.hidden_dropout_prob = 0.1
                self.attention_probs_dropout_prob = 0.1
                self.max_position_embeddings = 512
                self.type_vocab_size = 2
                self.seg_vocab_size = 2
                self.age_vocab_size = len(age2idx)
        
        model_config = SimpleConfig()
        
        # Feature dictionary for embeddings
        feature_dict = {
            'word': True,
            'seg': True, 
            'age': True,
            'position': True
        }
        
        model = BertForMultiLabelPrediction(
            config=model_config,
            num_labels=len(code2idx),
            feature_dict=feature_dict
        )
        
        # Use GPU if available  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✓ Model initialized on {device}")
        
        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        print(f"🏋️ Starting training for {config['epochs']} epochs...")
        
        train_losses = []
        
        for epoch in range(config['epochs']):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch[0].to(device)
                age_ids = batch[1].to(device)  
                labels = batch[2].to(device).float()
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids, age_ids)
                loss = criterion(outputs.logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            print(f"✓ Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        print("📊 Evaluating model performance...")
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in train_loader:  # Using train data as test for quick evaluation
                input_ids = batch[0].to(device)
                age_ids = batch[1].to(device)
                labels = batch[2].to(device).float()
                
                outputs = model(input_ids, age_ids)
                predictions = torch.sigmoid(outputs.logits)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate per-sample metrics (micro-average)
        y_true_flat = all_labels.flatten()
        y_pred_flat = all_predictions.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[valid_mask]
        y_pred_clean = y_pred_flat[valid_mask]
        
        if len(y_true_clean) > 0 and len(np.unique(y_true_clean)) > 1:
            roc_auc = roc_auc_score(y_true_clean, y_pred_clean)
            aps = average_precision_score(y_true_clean, y_pred_clean)
        else:
            print("⚠️  Cannot calculate metrics - insufficient data or no positive labels")
            roc_auc = 0.5
            aps = 0.0
        
        # Save results
        results = {
            'experiment_name': experiment_name,
            'config': config,
            'training_losses': train_losses,
            'final_metrics': {
                'roc_auc': float(roc_auc),
                'average_precision_score': float(aps),
                'num_samples': len(all_predictions),
                'num_patients': len(nextvisit_df['subject_id'].unique()) if 'subject_id' in nextvisit_df.columns else 'unknown'
            },
            'data_statistics': {
                'total_visits': len(df),
                'total_patients': len(df['subject_id'].unique()),
                'prediction_samples': len(nextvisit_df)
            }
        }
        
        # Save results
        results_file = Path(config['output_dir']) / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("=" * 80)
        print(f"🎯 TRAINING COMPLETE - {experiment_name}")
        print("=" * 80)
        print(f"📊 ROC-AUC:  {roc_auc:.4f}")
        print(f"📊 APS:      {aps:.4f}")
        print(f"📊 Samples:  {len(all_predictions):,}")
        print(f"📊 Patients: {len(df['subject_id'].unique()):,}")
        print(f"💾 Results saved: {results_file}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_to_nextvisit_format(df):
    """Convert our experimental data to NextVisit format"""
    
    # Group by patient and create next-visit prediction samples
    result_rows = []
    
    for patient_id, patient_data in df.groupby('subject_id'):
        visits = patient_data.sort_values('visit_concept_orders')
        
        # Create next-visit prediction samples
        for i in range(len(visits) - 1):
            current_visits = visits.iloc[:i+1]
            next_visit = visits.iloc[i+1]
            
            # Current sequence (history)
            history_codes = current_visits['concept_ids'].tolist()
            history_ages = current_visits['ages'].tolist()
            
            # Next visit target
            target_codes = [next_visit['concept_ids']]
            
            result_rows.append({
                'subject_id': patient_id,
                'patid': patient_id,  # Required by NextVisit class
                'history_codes': history_codes,
                'history_ages': history_ages, 
                'target_codes': target_codes,
                'sequence_length': len(history_codes)
            })
    
    return pd.DataFrame(result_rows)

if __name__ == "__main__":
    # Get experiment name from environment or use default
    experiment_name = os.environ.get('EXPERIMENT_NAME', 'current_experiment')
    
    # Run for the specified experiment
    results = run_nextvisit_training(experiment_name)
    
    if results:
        print(f"\n✅ Training completed successfully!")
        print(f"APS: {results['final_metrics']['average_precision_score']:.4f}")
        print(f"ROC-AUC: {results['final_metrics']['roc_auc']:.4f}")
    else:
        print("❌ Training failed!")