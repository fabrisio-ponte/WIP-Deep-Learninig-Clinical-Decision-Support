#!/usr/bin/env python3
"""
Comprehensive BEHRT Disease Performance Analysis
==============================================

This module provides comprehensive analysis of BEHRT model performance at both the overall
and individual disease level, with structured reporting and clinical interpretability focus.

Generates detailed metrics including:
- Overall weighted/macro performance metrics
- Disease category analysis
- Individual disease performance evaluation
- Structured reporting for clinical evaluation

Author: BEHRT Analysis Team
Date: 2024
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, hamming_loss
)

# Add BEHRT_Project to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if project_root.exists():
    sys.path.insert(0, str(project_root))

# BEHRT imports
try:
    from common.common import load_obj
    from model.utils import age_vocab
    from dataLoader.NextXVisit import NextVisit
    import pytorch_pretrained_bert as Bert
    from model.NextXVisit import BertForMultiLabelPrediction
except ImportError as e:
    print(f"Error importing BEHRT modules: {e}")
    print("Please ensure you're running from the BEHRT_Project directory or subdirectory")
    print(f"Project root: {project_root}")
    sys.exit(1)


class ComprehensiveDiseaseAnalysis:
    """
    Comprehensive analysis of BEHRT model performance at disease level.
    
    Provides detailed metrics including overall performance, category-level analysis,
    and individual disease predictions with structured clinical reporting.
    """
    
    def __init__(self, config_path: str = "analysis_config.json"):
        """
        Initialize comprehensive disease analysis.
        
        Args:
            config_path: Path to configuration file (creates default if not found)
        """
        self.config = self._load_config(config_path)
        
        # Disease category mappings (CCSR to human-readable categories)
        self.disease_categories = {
            'NEO': 'Neoplasms',
            'BLD': 'Blood and immune disorders',
            'END': 'Endocrine and metabolic',
            'MBD': 'Mental and behavioral disorders', 
            'NVS': 'Nervous system',
            'EYE': 'Eye and adnexa',
            'EAR': 'Ear and mastoid process',
            'CIR': 'Circulatory system',
            'RES': 'Respiratory system',
            'DIG': 'Digestive system',
            'SKN': 'Skin and subcutaneous tissue',
            'MUS': 'Musculoskeletal and connective tissue',
            'GU': 'Genitourinary system',
            'PRG': 'Pregnancy and childbirth',
            'PNL': 'Perinatal conditions',
            'CON': 'Congenital anomalies',
            'SYM': 'Symptoms and signs',
            'INJ': 'Injury and poisoning',
            'FAC': 'Factors influencing health status',
            'UTL': 'Utilization'
        }
        
        # Initialize placeholders
        self.vocab = None
        self.age_vocab = None
        self.label_vocab = None
        self.model = None
        self.test_loader = None
        self.predictions = None
        self.true_labels = None
        self.probabilities = None
        self.overall_metrics = None
        self.category_metrics = None
        self.individual_disease_metrics = None
        self.top_performing_diseases = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load analysis configuration"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration - corrected for CCSR NextVisit QUICK model
            return {
                'model_path': '../../data/models/quick/behrt_nextvisit_ccsr_quick.pt',
                'vocab_path': '../../data/processed/vocab_ccsr',
                'test_data_path': '../../data/processed/test_nextvisit_ccsr.parquet',
                'batch_size': 16,  # Smaller batch for quick evaluation
                'device': 'cpu',  # Use CPU for compatibility
                'max_len_seq': 64,  # Quick model uses 64 max length
                'max_age': 110
            }
    
    def load_model_and_data(self):
        """Load trained BEHRT model and test data"""
        print("Loading model and data...")
        
        # Normalize pickle path to avoid double-appending .pkl when config already includes the suffix.
        vocab_path = Path(self.config['vocab_path'])
        if vocab_path.suffix == '.pkl':
            vocab_path = vocab_path.with_suffix('')
        self.vocab = load_obj(str(vocab_path))
        self.age_vocab, _ = age_vocab(max_age=self.config['max_age'], mon=1)
        
        # Format label vocabulary
        self.label_vocab = self._format_label_vocab(self.vocab['token2idx'])
        
        # Use the same architecture as the trained clean model instead of the stale quick-model defaults.
        self.model_config = {
            'vocab_size': len(self.vocab['token2idx'].keys()),
            'hidden_size': self.config.get('hidden_size', 144),
            'seg_vocab_size': 2,
            'age_vocab_size': len(self.age_vocab.keys()),
            'max_position_embedding': self.config.get('max_len_seq', 64),
            'hidden_dropout_prob': self.config.get('hidden_dropout_prob', 0.1),
            'num_hidden_layers': self.config.get('num_hidden_layers', 3),
            'num_attention_heads': self.config.get('num_attention_heads', 6),
            'attention_probs_dropout_prob': self.config.get('attention_probs_dropout_prob', 0.1),
            'intermediate_size': self.config.get('intermediate_size', 256),
            'hidden_act': 'gelu',
            'initializer_range': 0.02,
        }
        
        # Load model
        feature_dict = {'word': True, 'seg': True, 'age': True, 'position': True}
        conf = self._create_bert_config(self.model_config)
        self.model = BertForMultiLabelPrediction(conf, len(self.label_vocab.keys()), feature_dict)
        
        # Load trained weights
        model_path = Path(self.config['model_path'])
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.config['device']))
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"⚠️  Model not found at {model_path}, using random weights")
        
        self.model.to(self.config['device'])
        self.model.eval()
        
        # Load test data
        test_data = pd.read_parquet(self.config['test_data_path']).reset_index(drop=True)
        test_data['label'] = test_data.label.apply(lambda x: list(set(x)))
        
        # Create dataset and dataloader
        test_dataset = NextVisit(
            token2idx=self.vocab['token2idx'],
            label2idx=self.label_vocab,  # Corrected parameter name
            age2idx=self.age_vocab,
            dataframe=test_data,
            max_len=self.config['max_len_seq']
        )
        
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        print(f"✓ Loaded {len(test_data)} test samples")
        
    def _format_label_vocab(self, token2idx: Dict) -> Dict:
        """Format vocabulary for labels"""
        token2idx = token2idx.copy()
        for special_token in ['PAD', 'SEP', 'CLS', 'MASK']:
            if special_token in token2idx:
                del token2idx[special_token]
        
        label_vocab = {}
        for i, token in enumerate(token2idx.keys()):
            label_vocab[token] = i
        return label_vocab
    
    def _create_bert_config(self, config: Dict):
        """Create BERT configuration"""
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
                    max_position_embeddings=config.get('max_position_embedding'),
                    initializer_range=config.get('initializer_range'),
                )
                self.seg_vocab_size = config.get('seg_vocab_size')
                self.age_vocab_size = config.get('age_vocab_size')
                
        return BertConfig(config)
    
    def generate_predictions(self):
        """Generate predictions on test set"""
        print("Generating predictions...")
        
        # Setup MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(classes=list(self.label_vocab.values()))
        self.mlb.fit([[each] for each in list(self.label_vocab.values())])
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                age_ids, input_ids, posi_ids, segment_ids, att_mask, targets, _ = batch
                
                # Strip padded -1 labels before multilabel binarization. The dataset pads labels with -1,
                # which should not be treated as a real class during evaluation.
                cleaned_targets = []
                for row in targets.numpy():
                    cleaned = [int(x) for x in row if int(x) >= 0]
                    cleaned_targets.append(cleaned)

                targets_binary = torch.tensor(
                    self.mlb.transform(cleaned_targets),
                    dtype=torch.float32
                ).to(self.config['device'])
                
                # Move to device
                age_ids = age_ids.to(self.config['device'])
                input_ids = input_ids.to(self.config['device'])
                posi_ids = posi_ids.to(self.config['device'])
                segment_ids = segment_ids.to(self.config['device'])
                att_mask = att_mask.to(self.config['device'])
                
                # Get model predictions
                logits = self.model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=att_mask)
                
                # Apply sigmoid to get probabilities
                sig = nn.Sigmoid()
                probabilities = sig(logits).cpu().numpy()
                
                # Convert to binary predictions (threshold = 0.5)
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_labels.extend(targets_binary.cpu().numpy())
                all_probabilities.extend(probabilities)
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx * self.config['batch_size']} samples...")
        
        self.predictions = np.array(all_predictions)
        self.true_labels = np.array(all_labels)
        self.probabilities = np.array(all_probabilities)
        
        print(f"✓ Generated predictions for {len(self.predictions)} samples")
    
    def calculate_overall_metrics(self):
        """Calculate overall performance metrics"""
        print("Calculating overall metrics...")
        
        # Exact-match subset accuracy is strict in multilabel settings but still useful to report explicitly.
        subset_accuracy = accuracy_score(self.true_labels, self.predictions)

        # Multilabel-friendly aggregate metrics.
        micro_precision = precision_score(self.true_labels, self.predictions, average='micro', zero_division=0)
        micro_recall = recall_score(self.true_labels, self.predictions, average='micro', zero_division=0)
        micro_f1 = f1_score(self.true_labels, self.predictions, average='micro', zero_division=0)

        macro_precision = precision_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        macro_recall = recall_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(self.true_labels, self.predictions, average='macro', zero_division=0)

        samples_precision = precision_score(self.true_labels, self.predictions, average='samples', zero_division=0)
        samples_recall = recall_score(self.true_labels, self.predictions, average='samples', zero_division=0)
        samples_f1 = f1_score(self.true_labels, self.predictions, average='samples', zero_division=0)

        hamming_acc = 1.0 - hamming_loss(self.true_labels, self.predictions)
        
        # Per-class F1 scores
        per_class_f1 = f1_score(self.true_labels, self.predictions, average=None, zero_division=0)
        mean_per_class_f1 = np.mean(per_class_f1)
        
        # Sample-wise metrics (for comparison with previous results)
        sample_wise_aps = average_precision_score(self.true_labels, self.probabilities, average='samples')
        sample_wise_auc = roc_auc_score(self.true_labels, self.probabilities, average='samples')
        
        # Token-level accuracy
        total_predictions = self.predictions.size
        correct_predictions = np.sum(self.predictions == self.true_labels)
        token_accuracy = correct_predictions / total_predictions
        
        self.overall_metrics = {
            'subset_accuracy': subset_accuracy,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'samples_precision': samples_precision,
            'samples_recall': samples_recall,
            'samples_f1': samples_f1,
            'hamming_accuracy': hamming_acc,
            'mean_per_class_f1': mean_per_class_f1,
            'sample_wise_aps': sample_wise_aps,
            'sample_wise_auc': sample_wise_auc,
            'token_accuracy': token_accuracy,
            'correct_predictions': int(correct_predictions),
            'total_predictions': int(total_predictions),
            'per_class_f1': per_class_f1.tolist()
        }
    
    def analyze_disease_categories(self):
        """Analyze performance by disease categories"""
        print("Analyzing disease categories...")
        
        # Map diseases to categories
        disease_to_category = {}
        category_diseases = defaultdict(list)
        
        for disease_code, disease_idx in self.label_vocab.items():
            # Extract category from CCSR code (e.g., CCSR_CIR007 -> CIR)
            if disease_code.startswith('CCSR_'):
                category_code = disease_code.split('_')[1][:3]
                category_name = self.disease_categories.get(category_code, f"Unknown_{category_code}")
            else:
                category_name = "Other"
            
            disease_to_category[disease_idx] = category_name
            category_diseases[category_name].append(disease_idx)
        
        # Calculate metrics per category
        category_metrics = {}
        for category_name, disease_indices in category_diseases.items():
            if len(disease_indices) == 0:
                continue
                
            # Extract predictions and labels for this category
            category_predictions = self.predictions[:, disease_indices]
            category_labels = self.true_labels[:, disease_indices]
            
            # Calculate metrics
            if category_predictions.size > 0:
                category_f1 = f1_score(category_labels, category_predictions, average='macro', zero_division=0)
                category_recall = recall_score(category_labels, category_predictions, average='macro', zero_division=0)
                category_precision = precision_score(category_labels, category_predictions, average='macro', zero_division=0)
                
                category_metrics[category_name] = {
                    'f1_score': category_f1,
                    'recall': category_recall,
                    'precision': category_precision,
                    'num_diseases': len(disease_indices)
                }
        
        self.category_metrics = category_metrics
    
    def analyze_individual_diseases(self, top_n: int = 10):
        """Analyze performance of individual diseases"""
        print("Analyzing individual diseases...")
        
        # Create reverse mapping from index to disease code
        idx_to_disease = {idx: code for code, idx in self.label_vocab.items()}
        
        individual_results = []
        
        for disease_idx in range(len(self.label_vocab)):
            disease_code = idx_to_disease[disease_idx]
            
            # Get predictions and labels for this disease
            disease_predictions = self.predictions[:, disease_idx]
            disease_labels = self.true_labels[:, disease_idx]
            
            # Calculate metrics
            precision = precision_score(disease_labels, disease_predictions, zero_division=0)
            recall = recall_score(disease_labels, disease_predictions, zero_division=0)
            f1 = f1_score(disease_labels, disease_predictions, zero_division=0)
            
            # Support (number of true positives)
            support = np.sum(disease_labels)
            
            individual_results.append({
                'disease_code': disease_code,
                'disease_idx': disease_idx,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': int(support)
            })
        
        # Sort by F1 score and get top performers
        individual_results.sort(key=lambda x: x['f1_score'], reverse=True)
        
        self.individual_disease_metrics = individual_results
        self.top_performing_diseases = individual_results[:top_n]
    
    def generate_formatted_report(self) -> str:
        """Generate formatted report matching the desired structure"""
        
        report = []
        report.append("4. PERFORMANCE METRICS")
        report.append("=" * 84)
        
        # Overall Performance
        report.append("OVERALL PERFORMANCE:")
        report.append(f"├─ Subset Accuracy:         {self.overall_metrics['subset_accuracy']:.2%}")
        report.append(f"├─ Micro Precision:         {self.overall_metrics['micro_precision']:.2%}")
        report.append(f"├─ Micro Recall:            {self.overall_metrics['micro_recall']:.2%}")
        report.append(f"├─ Micro F1 Score:          {self.overall_metrics['micro_f1']:.2%}")
        report.append("│")
        report.append(f"├─ Macro Precision:         {self.overall_metrics['macro_precision']:.2%}")
        report.append(f"├─ Macro Recall:            {self.overall_metrics['macro_recall']:.2%}")
        report.append(f"├─ Macro F1 Score:          {self.overall_metrics['macro_f1']:.2%}")
        report.append("│")
        report.append(f"├─ Samples Precision:       {self.overall_metrics['samples_precision']:.2%}")
        report.append(f"├─ Samples Recall:          {self.overall_metrics['samples_recall']:.2%}")
        report.append(f"├─ Samples F1 Score:        {self.overall_metrics['samples_f1']:.2%}")
        report.append("│")
        report.append(f"├─ Hamming Accuracy:        {self.overall_metrics['hamming_accuracy']:.2%}")
        report.append(f"└─ Mean Per-Class F1:       {self.overall_metrics['mean_per_class_f1']:.2%}")
        report.append("")
        
        # Best performing categories
        sorted_categories = sorted(
            self.category_metrics.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )[:3]
        
        report.append("BEST PERFORMING DISEASE CATEGORIES:")
        for i, (category, metrics) in enumerate(sorted_categories):
            symbol = "├─" if i < len(sorted_categories) - 1 else "└─"
            report.append(f"{symbol} {category} ({metrics['num_diseases']} classes): "
                        f"F1={metrics['f1_score']:.4f} | Recall={metrics['recall']:.4f}")
        report.append("")
        
        # Top performing individual diseases
        report.append("TOP PERFORMING DISEASE PREDICTIONS:")
        for i, disease in enumerate(self.top_performing_diseases[:5]):
            symbol = "├─" if i < 4 else "└─"
            disease_code = disease['disease_code'].replace('CCSR_', '')
            report.append(f"{symbol} CCS Code {disease_code}: "
                        f"P={disease['precision']:.4f} | "
                        f"R={disease['recall']:.4f} | "
                        f"F1={disease['f1_score']:.4f}")
        report.append("")
        
        # Token accuracy
        report.append(f"CORRECTLY PREDICTED TOKENS: {self.overall_metrics['correct_predictions']:,} "
                     f"out of {self.overall_metrics['total_predictions']:,} "
                     f"({self.overall_metrics['token_accuracy']:.2%})")
        
        # Sample-wise metrics for comparison
        report.append("")
        report.append("SAMPLE-WISE METRICS (for comparison with previous results):")
        report.append(f"├─ Average Precision Score: {self.overall_metrics['sample_wise_aps']:.2%}")
        report.append(f"└─ ROC-AUC Score:          {self.overall_metrics['sample_wise_auc']:.2%}")
        
        return "\n".join(report)
    
    def save_results(self, output_file: str = "comprehensive_disease_analysis_results.json"):
        """Save detailed results to JSON file"""
        
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'overall_metrics': self.overall_metrics,
            'category_metrics': self.category_metrics,
            'top_performing_diseases': self.top_performing_diseases,
            'all_individual_diseases': self.individual_disease_metrics,
            'config': self.config
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Detailed results saved to {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🔬 Starting Comprehensive BEHRT Disease-Level Analysis")
        print("=" * 60)
        
        try:
            # Load model and data
            self.load_model_and_data()
            
            # Generate predictions
            self.generate_predictions()
            
            # Calculate metrics
            self.calculate_overall_metrics()
            self.analyze_disease_categories()
            self.analyze_individual_diseases()
            
            # Generate and display report
            report = self.generate_formatted_report()
            print("\n" + report)
            
            # Save results
            self.save_results()
            
            print("\n✅ Analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run comprehensive disease analysis"""
    parser = argparse.ArgumentParser(description="Run BEHRT disease-level evaluation.")
    parser.add_argument("--config", default="config/analysis_config.json", help="Path to the analysis configuration JSON file.")
    args = parser.parse_args()

    analyzer = ComprehensiveDiseaseAnalysis(config_path=args.config)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()