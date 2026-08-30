#!/usr/bin/env python3
"""
Simple test to understand BEHRT data format issues
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../')

from common.common import load_obj
from model.utils import age_vocab

def test_data_format():
    print("Testing BEHRT data format...")
    
    # Load vocabulary
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
    
    print(f"Total vocab size: {len(BertVocab['token2idx'])}")
    print(f"Label vocab size: {len(labelVocab)}")
    
    # Create simple test data like our experiments
    test_data = pd.DataFrame([
        {'patid': 1, 'code': np.array(['CCSR_CIR007', 'SEP']), 'age': np.array([600, 600]), 'label': np.array(['CCSR_DIG004'])},
        {'patid': 1, 'code': np.array(['CCSR_CIR007', 'CCSR_DIG004', 'SEP']), 'age': np.array([600, 600, 600]), 'label': np.array(['CCSR_DIG010'])},
    ])
    
    print(f"Test data shape: {test_data.shape}")
    print("Sample data:")
    print(test_data)
    
    # Test NextVisit dataloader
    from dataLoader.NextXVisit import NextVisit  
    
    Dset = NextVisit(
        token2idx=BertVocab['token2idx'], 
        label2idx=labelVocab, 
        age2idx=ageVocab,
        dataframe=test_data, 
        max_len=64
    )
    
    print(f"Dataset length: {len(Dset)}")
    
    # Get first sample
    sample = Dset[0]
    print("Sample tensor shapes:")
    print(f"  age: {sample[0].shape}")
    print(f"  code: {sample[1].shape}")
    print(f"  position: {sample[2].shape}")
    print(f"  segment: {sample[3].shape}")
    print(f"  mask: {sample[4].shape}")
    print(f"  label: {sample[5].shape}")
    print(f"  patid: {sample[6].shape}")
    
    # Check label content
    print(f"Label tensor: {sample[5]}")
    print(f"Non-zero labels: {torch.where(sample[5] >= 0)}")
    
    return True

if __name__ == "__main__":
    import torch
    test_data_format()