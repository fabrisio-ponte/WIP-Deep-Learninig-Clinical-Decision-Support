# """
# FastAPI Backend for BEHRT Diagnosis Prediction Demo - Fixed Vocabulary Size
# """
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict
# import torch
# import torch.nn as nn
# import pickle
# import sys
# import os
# import numpy as np

# # Add paths to import the actual BEHRT model
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))

# app = FastAPI(title="BEHRT Diagnosis Predictor")

# # CORS settings
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables
# model = None
# vocab = None
# age_vocab = None
# device = None

# # ============================================================
# # Fixed BertEmbeddings class (with the dtype fix)
# # ============================================================

# class FixedBertEmbeddings(nn.Module):
#     """Construct the embeddings from word, segment, age - FIXED VERSION"""

#     def __init__(self, config):
#         super(FixedBertEmbeddings, self).__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
#         self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
#         self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        
#         # Fixed position embeddings with explicit dtype
#         posi_embedding = self._init_posi_embedding(config.max_position_embeddings, config.hidden_size)
#         self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.posi_embeddings.weight = nn.Parameter(posi_embedding)
#         self.posi_embeddings.weight.requires_grad = False  # Freeze position embeddings

#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def _init_posi_embedding(self, max_position_embedding, hidden_size):
#         def even_code(pos, idx):
#             return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

#         def odd_code(pos, idx):
#             return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

#         # initialize position embedding table
#         lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

#         # reset table parameters with hard encoding
#         # set even dimension
#         for pos in range(max_position_embedding):
#             for idx in np.arange(0, hidden_size, step=2):
#                 lookup_table[pos, idx] = even_code(pos, idx)
#         # set odd dimension
#         for pos in range(max_position_embedding):
#             for idx in np.arange(1, hidden_size, step=2):
#                 lookup_table[pos, idx] = odd_code(pos, idx)

#         # FIX: Explicitly specify dtype
#         return torch.tensor(lookup_table, dtype=torch.float32)

#     def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
#         if seg_ids is None:
#             seg_ids = torch.zeros_like(word_ids)
#         if age_ids is None:
#             age_ids = torch.zeros_like(word_ids)
#         if posi_ids is None:
#             posi_ids = torch.zeros_like(word_ids)

#         word_embed = self.word_embeddings(word_ids)
#         segment_embed = self.segment_embeddings(seg_ids)
#         age_embed = self.age_embeddings(age_ids)
#         posi_embeddings = self.posi_embeddings(posi_ids)

#         if age:
#             embeddings = word_embed + segment_embed + age_embed + posi_embeddings
#         else:
#             embeddings = word_embed + segment_embed + posi_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings

# # ============================================================
# # Fixed BertModel class
# # ============================================================

# class FixedBertModel(nn.Module):
#     def __init__(self, config):
#         super(FixedBertModel, self).__init__()
#         self.embeddings = FixedBertEmbeddings(config=config)
        
#         # Use standard Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=config.hidden_size,
#             nhead=config.num_attention_heads,
#             dim_feedforward=config.intermediate_size,
#             dropout=config.hidden_dropout_prob,
#             activation='gelu',
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        
#         # Pooler
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.pooler_activation = nn.Tanh()

#     def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None,
#                 output_all_encoded_layers=True):
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)
#         if age_ids is None:
#             age_ids = torch.zeros_like(input_ids)
#         if seg_ids is None:
#             seg_ids = torch.zeros_like(input_ids)
#         if posi_ids is None:
#             posi_ids = torch.zeros_like(input_ids)

#         # Create attention mask for transformer
#         if attention_mask is not None:
#             # Convert to transformer format: (batch_size, seq_len) -> (batch_size, seq_len)
#             # Transformer expects mask where True means ignore
#             src_key_padding_mask = (attention_mask == 0)
#         else:
#             src_key_padding_mask = None

#         embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        
#         # Pass through transformer encoder
#         encoded = self.encoder(embedding_output, src_key_padding_mask=src_key_padding_mask)
        
#         # Get pooled output from CLS token
#         cls_output = encoded[:, 0, :]
#         pooled_output = self.pooler(cls_output)
#         pooled_output = self.pooler_activation(pooled_output)
        
#         return encoded, pooled_output

# # ============================================================
# # BEHRT Model for Prediction - WITH VOCAB SIZE FIX
# # ============================================================

# class BEHRTForPrediction(nn.Module):
#     def __init__(self, config, actual_vocab_size=None):
#         super(BEHRTForPrediction, self).__init__()
#         self.bert = FixedBertModel(config)
        
#         # Use actual vocab size from checkpoint if provided, otherwise use config
#         # This fixes the size mismatch issue
#         classifier_vocab_size = actual_vocab_size if actual_vocab_size else config.vocab_size
        
#         # Prediction head for next visit
#         self.classifier = nn.Linear(config.hidden_size, classifier_vocab_size)
        
#         # Store for reference
#         self.actual_vocab_size = classifier_vocab_size
#         self.config_vocab_size = config.vocab_size

#     def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None):
#         sequence_output, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
#                                        output_all_encoded_layers=False)
        
#         # Use CLS token for prediction
#         cls_output = sequence_output[:, 0, :]
#         logits = self.classifier(cls_output)
#         return logits

# # ============================================================
# # Data Models
# # ============================================================

# class Visit(BaseModel):
#     codes: List[str]
#     age_months: int

# class PredictionRequest(BaseModel):
#     patient_history: List[Visit]
#     top_k: int = 10

# class PredictionResponse(BaseModel):
#     predictions: List[Dict]
#     patient_summary: Dict

# # ============================================================
# # Configuration Class - EXACT MATCH TO YOUR TRAINING CONFIG
# # ============================================================

# class BertConfig:
#     def __init__(self, config_dict):
#         # EXACT CONFIG FROM YOUR TRAINING SCRIPT
#         self.vocab_size = config_dict.get('vocab_size', 478)
#         self.hidden_size = config_dict.get('hidden_size', 144)  # ← MUST MATCH: 144
#         self.num_hidden_layers = config_dict.get('num_hidden_layers', 3)  # ← MUST MATCH: 3
#         self.num_attention_heads = config_dict.get('num_attention_heads', 6)  # ← MUST MATCH: 6
#         self.intermediate_size = config_dict.get('intermediate_size', 256)  # ← MUST MATCH: 256
#         self.hidden_dropout_prob = config_dict.get('hidden_dropout_prob', 0.1)
#         self.attention_probs_dropout_prob = config_dict.get('attention_probs_dropout_prob', 0.1)
#         self.max_position_embeddings = config_dict.get('max_position_embeddings', 64)  # ← MUST MATCH: 64
#         self.type_vocab_size = config_dict.get('type_vocab_size', 2)
#         self.seg_vocab_size = config_dict.get('seg_vocab_size', 2)
#         self.age_vocab_size = config_dict.get('age_vocab_size', 1322)
#         self.hidden_act = config_dict.get('hidden_act', 'gelu')
#         self.initializer_range = config_dict.get('initializer_range', 0.02)

# # ============================================================
# # Startup: Load Model - WITH VOCAB SIZE FIX
# # ============================================================

# @app.on_event("startup")
# async def load_model_and_vocab():
#     global model, vocab, age_vocab, device
    
#     print("Loading model and vocabularies...")
    
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Load vocabularies
#     vocab_path = '../data/processed/vocab_ccsr.pkl'
#     age_vocab_path = '../data/processed/age_vocab.pkl'
    
#     try:
#         with open(vocab_path, 'rb') as f:
#             vocab = pickle.load(f)
#         print(f"✓ Loaded vocabulary: {len(vocab['token2idx'])} tokens")
        
#         with open(age_vocab_path, 'rb') as f:
#             age_vocab = pickle.load(f)
#         print(f"✓ Loaded age vocabulary: {len(age_vocab['age2idx'])} tokens")
        
#     except FileNotFoundError as e:
#         print(f"⚠️  Vocabulary files not found: {e}")
#         print("Using dummy vocab for demo")
#         vocab = {
#             'token2idx': {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK': 3, 'UNK': 4},
#             'idx2token': {0: 'PAD', 1: 'CLS', 2: 'SEP', 3: 'MASK', 4: 'UNK'}
#         }
#         age_vocab = {'age2idx': {'PAD': 0, 'UNK': 1}, 'idx2age': {0: 'PAD', 1: 'UNK'}}
    
#     # Load model
#     model_path = '../data/models/quick/behrt_nextvisit_ccsr_quick.pt'
    
#     try:
#         checkpoint = torch.load(model_path, map_location=device)
#         print(f"✓ Loaded checkpoint from: {model_path}")
        
#         # Extract ACTUAL vocabulary size from the checkpoint
#         if 'classifier.weight' in checkpoint:
#             actual_vocab_size = checkpoint['classifier.weight'].shape[0]
#         elif 'model_state_dict' in checkpoint and 'classifier.weight' in checkpoint['model_state_dict']:
#             actual_vocab_size = checkpoint['model_state_dict']['classifier.weight'].shape[0]
#         else:
#             # Try to find classifier weight in state dict
#             state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
#             classifier_key = None
#             for key in state_dict.keys():
#                 if 'classifier' in key and 'weight' in key:
#                     classifier_key = key
#                     break
#             if classifier_key:
#                 actual_vocab_size = state_dict[classifier_key].shape[0]
#             else:
#                 actual_vocab_size = 474  # Default from your error message
        
#         print(f"✓ Detected actual vocabulary size from checkpoint: {actual_vocab_size}")
#         print(f"✓ Current vocabulary size: {len(vocab['token2idx'])}")
        
#         # EXACT CONFIGURATION FROM YOUR TRAINING SCRIPT
#         config_dict = {
#             'vocab_size': len(vocab['token2idx']),  # Current vocab size
#             'hidden_size': 144,  # ← EXACT from your config
#             'num_hidden_layers': 3,  # ← EXACT from your config
#             'num_attention_heads': 6,  # ← EXACT from your config
#             'intermediate_size': 256,  # ← EXACT from your config
#             'hidden_dropout_prob': 0.1,
#             'attention_probs_dropout_prob': 0.1,
#             'max_position_embeddings': 64,  # ← EXACT from global_params['max_len_seq']
#             'type_vocab_size': 2,
#             'seg_vocab_size': 2,
#             'age_vocab_size': len(age_vocab['age2idx']),
#             'hidden_act': 'gelu',
#             'initializer_range': 0.02,
#         }
        
#         config = BertConfig(config_dict)
        
#         print("✓ Using EXACT model configuration from training:")
#         print(f"  - Hidden size: {config.hidden_size}")
#         print(f"  - Layers: {config.num_hidden_layers}")
#         print(f"  - Attention heads: {config.num_attention_heads}")
#         print(f"  - Intermediate size: {config.intermediate_size}")
#         print(f"  - Max position: {config.max_position_embeddings}")
#         print(f"  - Config vocabulary size: {config.vocab_size}")
#         print(f"  - Actual vocabulary size (from checkpoint): {actual_vocab_size}")
        
#         # Create model with EXACT configuration and ACTUAL vocab size
#         model = BEHRTForPrediction(config, actual_vocab_size=actual_vocab_size)
#         print("✓ Created BEHRTForPrediction model with exact training configuration")
        
#         # Get state dict
#         if 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#         else:
#             state_dict = checkpoint
        
#         # Load state dict
#         model.load_state_dict(state_dict, strict=False)
#         model.to(device)
#         model.eval()
        
#         print(f"✓ Successfully loaded BEHRT model")
#         print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters())}")
        
#         # Print which parameters were loaded successfully
#         model_state_dict = model.state_dict()
#         loaded_count = 0
#         total_count = len(model_state_dict)
        
#         for name, param in model_state_dict.items():
#             if name in state_dict and param.shape == state_dict[name].shape:
#                 loaded_count += 1
#             elif name in state_dict:
#                 print(f"  ⚠️  Shape mismatch: {name} - expected {param.shape}, got {state_dict[name].shape}")
#             else:
#                 print(f"  ⚠️  Missing: {name}")
        
#         print(f"✓ Loaded {loaded_count}/{total_count} parameters successfully")
        
#         # Create mapping between vocab indices if sizes are different
#         if actual_vocab_size != len(vocab['token2idx']):
#             print(f"⚠️  Vocabulary size mismatch: model trained with {actual_vocab_size}, current vocab has {len(vocab['token2idx'])}")
#             print(f"⚠️  Some predictions may not work correctly")
        
#     except FileNotFoundError:
#         print(f"⚠️  Model not found: {model_path}")
#         print("Running in demo mode without actual predictions")
#         model = None
#     except Exception as e:
#         print(f"⚠️  Could not load model: {e}")
#         import traceback
#         traceback.print_exc()
#         print("Running in demo mode without actual predictions")
#         model = None

# # ============================================================
# # Helper Functions - UPDATED FOR VOCAB SIZE HANDLING
# # ============================================================

# def encode_sequence(visits: List[Visit]) -> tuple:
#     """Convert patient history to model input"""
    
#     codes = ['CLS']
#     ages = [visits[0].age_months if visits else 0]
#     segments = [0]  # CLS segment
    
#     for i, visit in enumerate(visits):
#         # Add visit codes
#         for code in visit.codes:
#             token = code if code in vocab['token2idx'] else 'UNK'
#             codes.append(token)
#             ages.append(visit.age_months)
#             segments.append(0)  # Same segment for all codes in visit
        
#         # Add separator between visits (but not after last visit)
#         if i < len(visits) - 1:
#             codes.append('SEP')
#             ages.append(visit.age_months)
#             segments.append(1)  # SEP has different segment
    
#     # Truncate if sequence exceeds max length (64)
#     if len(codes) > 64:
#         print(f"⚠️  Truncating sequence from {len(codes)} to 64 tokens")
#         codes = codes[:64]
#         ages = ages[:64]
#         segments = segments[:64]
    
#     # Convert to indices - handle vocabulary mapping
#     code_indices = []
#     for c in codes:
#         if c in vocab['token2idx']:
#             idx = vocab['token2idx'][c]
#             # If model was trained with smaller vocab, remap indices if necessary
#             if model and idx >= model.actual_vocab_size:
#                 # Map to UNK token if index is out of bounds for the model
#                 code_indices.append(vocab['token2idx']['UNK'])
#             else:
#                 code_indices.append(idx)
#         else:
#             code_indices.append(vocab['token2idx']['UNK'])
    
#     # Age indices
#     age_indices = []
#     for age in ages:
#         age_str = str(age)
#         age_idx = age_vocab['age2idx'].get(age_str, age_vocab['age2idx']['UNK'])
#         age_indices.append(age_idx)
    
#     # Position indices
#     position_indices = list(range(len(codes)))
    
#     return code_indices, age_indices, segments, position_indices, codes

# def decode_predictions(logits: torch.Tensor, top_k: int = 10) -> List[Dict]:
#     """Convert model output to readable predictions"""
    
#     # Use softmax for multi-class prediction
#     probs = torch.softmax(logits[0], dim=0)
#     top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
    
#     predictions = []
#     for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
#         # Handle vocabulary size mismatch
#         if model and idx >= model.actual_vocab_size:
#             # This index is from the model's vocabulary, which might be smaller
#             # We need to map it to our current vocabulary if possible
#             if idx < len(vocab['idx2token']):
#                 code = vocab['idx2token'][idx]
#             else:
#                 continue  # Skip if index is out of bounds
#         else:
#             code = vocab['idx2token'].get(idx, 'UNKNOWN')
        
#         # Skip special tokens
#         if code in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']:
#             continue
        
#         predictions.append({
#             'code': code,
#             'probability': round(prob, 4),
#             'description': get_code_description(code)
#         })
    
#     return predictions[:top_k]

# def get_code_description(code: str) -> str:
#     """Get human-readable description for code"""
    
#     descriptions = {
#         'CIR008': 'Acute myocardial infarction',
#         'CIR003': 'Essential hypertension',
#         'END004': 'Diabetes mellitus',
#         'RSP006': 'COPD and bronchiectasis',
#         'DIG011': 'Liver disease',
#         'GEN003': 'Chronic kidney disease',
#         'NEO014': 'Cancer of breast',
#         'MBD008': 'Mood disorders',
#         'NVS008': 'Epilepsy and convulsions',
#         'INF002': 'Septicemia',
#         'CIR007': 'Heart failure',
#         'RSP004': 'Pneumonia',
#         'END003': 'Thyroid disorders',
#         'CIR009': 'Ischemic heart disease',
#         'DIG010': 'Gallbladder disease',
#         'MSS003': 'Osteoarthritis',
#     }
    
#     return descriptions.get(code, f'Diagnosis: {code}')

# def get_demo_predictions(patient_history: List[Visit], top_k: int = 10) -> List[Dict]:
#     """Generate realistic demo predictions based on patient history"""
    
#     common_progressions = {
#         'CIR003': ['END004', 'CIR008', 'GEN003'],
#         'END004': ['CIR008', 'GEN003', 'NVS008'],
#         'RSP006': ['CIR008', 'INF002', 'RSP004'],
#         'CIR008': ['CIR007', 'GEN003', 'END004'],
#         'GEN003': ['CIR007', 'END004', 'INF002'],
#     }
    
#     all_codes = []
#     for visit in patient_history:
#         all_codes.extend(visit.codes)
    
#     seen = set()
#     unique_codes = []
#     for code in all_codes:
#         if code not in seen:
#             seen.add(code)
#             unique_codes.append(code)
    
#     predictions = []
#     base_prob = 0.8
    
#     for code in unique_codes[-3:]:
#         if code in common_progressions:
#             for next_code in common_progressions[code][:2]:
#                 if next_code not in unique_codes:
#                     predictions.append({
#                         'code': next_code,
#                         'probability': round(base_prob, 4),
#                         'description': get_code_description(next_code)
#                     })
#                     base_prob *= 0.7
    
#     if not predictions:
#         common_predictions = [
#             {"code": "CIR003", "probability": 0.65, "description": "Essential hypertension"},
#             {"code": "END004", "probability": 0.55, "description": "Diabetes mellitus"},
#             {"code": "MBD008", "probability": 0.45, "description": "Mood disorders"},
#         ]
#         predictions = common_predictions
    
#     return predictions[:top_k]

# # ============================================================
# # API Endpoints
# # ============================================================

# @app.get("/")
# async def root():
#     vocab_info = {
#         "current_vocab_size": len(vocab['token2idx']) if vocab else 0,
#         "model_vocab_size": model.actual_vocab_size if model else 0,
#         "vocab_mismatch": model and model.actual_vocab_size != len(vocab['token2idx']) if model else False
#     }
    
#     return {
#         "message": "BEHRT Diagnosis Prediction API",
#         "status": "running",
#         "model_loaded": model is not None,
#         "vocab_info": vocab_info,
#         "max_sequence_length": 64
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "model": "loaded" if model is not None else "not loaded", 
#         "device": str(device) if device else "unknown",
#         "config": "quick (144 hidden, 3 layers, max_seq=64)" if model is not None else "demo",
#         "vocab_size": f"{model.actual_vocab_size} (model) vs {len(vocab['token2idx'])} (current)" if model else "unknown"
#     }

# @app.post("/predict", response_model=PredictionResponse)
# async def predict_next_diagnoses(request: PredictionRequest):
#     """
#     Predict next visit diagnoses based on patient history
#     """
    
#     if not request.patient_history:
#         raise HTTPException(status_code=400, detail="Patient history cannot be empty")
    
#     if model is None:
#         from .demo_predictions import get_demo_predictions
#         demo_predictions = get_demo_predictions(request.patient_history, request.top_k)
#         return PredictionResponse(
#             predictions=demo_predictions,
#             patient_summary={
#                 "total_visits": len(request.patient_history),
#                 "total_diagnoses": sum(len(v.codes) for v in request.patient_history),
#                 "age_years": request.patient_history[-1].age_months // 12
#             }
#         )
    
#     try:
#         code_indices, age_indices, seg_indices, pos_indices, codes = encode_sequence(request.patient_history)
        
#         print(f"Encoded sequence ({len(codes)} tokens): {codes}")
        
#         code_tensor = torch.tensor([code_indices]).to(device)
#         age_tensor = torch.tensor([age_indices]).to(device)
#         seg_tensor = torch.tensor([seg_indices]).to(device)
#         pos_tensor = torch.tensor([pos_indices]).to(device)
        
#         attention_mask = torch.ones_like(code_tensor)
        
#         with torch.no_grad():
#             logits = model(
#                 input_ids=code_tensor,
#                 age_ids=age_tensor,
#                 seg_ids=seg_tensor,
#                 posi_ids=pos_tensor,
#                 attention_mask=attention_mask
#             )
        
#         predictions = decode_predictions(logits, top_k=request.top_k)
        
#         patient_summary = {
#             "total_visits": len(request.patient_history),
#             "total_diagnoses": sum(len(v.codes) for v in request.patient_history),
#             "age_years": request.patient_history[-1].age_months // 12,
#             "sequence_length": len(codes),
#             "max_sequence_allowed": 64,
#             "vocab_mismatch": model.actual_vocab_size != len(vocab['token2idx'])
#         }
        
#         return PredictionResponse(
#             predictions=predictions,
#             patient_summary=patient_summary
#         )
        
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# @app.get("/vocab/search")
# async def search_vocabulary(query: str, limit: int = 20):
#     if not vocab:
#         return {"results": []}
    
#     query_upper = query.upper()
#     matches = []
    
#     for code in vocab['token2idx'].keys():
#         if code in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']:
#             continue
        
#         if query_upper in code:
#             matches.append({
#                 'code': code,
#                 'description': get_code_description(code)
#             })
        
#         if len(matches) >= limit:
#             break
    
#     return {"results": matches}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


"""
FastAPI Backend for BEHRT Diagnosis Prediction Demo - Fixed Forward Pass
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch
import torch.nn as nn
import pickle
import sys
import os
import numpy as np
import math

# Add paths to import the actual BEHRT model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))

app = FastAPI(title="BEHRT Diagnosis Predictor")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
vocab = None
age_vocab = None
device = None

# ============================================================
# Fixed BertEmbeddings class (with the dtype fix)
# ============================================================

class FixedBertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age - FIXED VERSION"""

    def __init__(self, config):
        super(FixedBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        
        # Fixed position embeddings with explicit dtype
        posi_embedding = self._init_posi_embedding(config.max_position_embeddings, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.posi_embeddings.weight = nn.Parameter(posi_embedding)
        self.posi_embeddings.weight.requires_grad = False  # Freeze position embeddings

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        # FIX: Explicitly specify dtype
        return torch.tensor(lookup_table, dtype=torch.float32)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# ============================================================
# Fixed BertModel class - COMPATIBLE WITH CHECKPOINT
# ============================================================

class FixedBertModel(nn.Module):
    def __init__(self, config):
        super(FixedBertModel, self).__init__()
        self.embeddings = FixedBertEmbeddings(config=config)
        
        # Use the SAME structure as the original BEHRT model
        self.encoder = BertEncoder(config)
        
        # Pooler - match the original structure
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None,
                output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        
        # Pass through encoder - FIX: Get the last layer only
        encoded_layers = self.encoder(embedding_output, attention_mask, 
                                     output_all_encoded_layers=output_all_encoded_layers)
        
        # FIX: Extract the last layer from the list
        if output_all_encoded_layers:
            sequence_output = encoded_layers[-1]  # Get last layer from list
        else:
            sequence_output = encoded_layers[0]   # Get the single layer
        
        # Get pooled output from CLS token
        pooled_output = self.pooler(sequence_output[:, 0, :])
        
        return sequence_output, pooled_output

# ============================================================
# BertEncoder class - COMPATIBLE WITH ORIGINAL BEHRT
# ============================================================

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers

# ============================================================
# BertLayer class - COMPATIBLE WITH ORIGINAL BEHRT
# ============================================================

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

# ============================================================
# BertAttention class - COMPATIBLE WITH ORIGINAL BEHRT
# ============================================================

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

# ============================================================
# BertSelfAttention class - COMPATIBLE WITH ORIGINAL BEHRT
# ============================================================

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

# ============================================================
# BertSelfOutput class - COMPATIBLE WITH ORIGINAL BEHRT
# ============================================================

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ============================================================
# BertIntermediate class - COMPATIBLE WITH ORIGINAL BEHRT
# ============================================================

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# ============================================================
# BertOutput class - COMPATIBLE WITH ORIGINAL BEHRT
# ============================================================

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ============================================================
# BEHRT Model for Prediction - WITH COMPATIBLE STRUCTURE
# ============================================================

class BEHRTForPrediction(nn.Module):
    def __init__(self, config, actual_vocab_size=None):
        super(BEHRTForPrediction, self).__init__()
        self.bert = FixedBertModel(config)
        
        # Use actual vocab size from checkpoint if provided, otherwise use config
        classifier_vocab_size = actual_vocab_size if actual_vocab_size else config.vocab_size
        
        # Prediction head for next visit
        self.classifier = nn.Linear(config.hidden_size, classifier_vocab_size)
        
        # Store for reference
        self.actual_vocab_size = classifier_vocab_size
        self.config_vocab_size = config.vocab_size

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
                                       output_all_encoded_layers=False)
        
        # Use CLS token for prediction
        cls_output = sequence_output[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# ============================================================
# Data Models
# ============================================================

class Visit(BaseModel):
    codes: List[str]
    age_months: int

class PredictionRequest(BaseModel):
    patient_history: List[Visit]
    top_k: int = 10

class PredictionResponse(BaseModel):
    predictions: List[Dict]
    patient_summary: Dict

# ============================================================
# Configuration Class - EXACT MATCH TO YOUR TRAINING CONFIG
# ============================================================

class BertConfig:
    def __init__(self, config_dict):
        # EXACT CONFIG FROM YOUR TRAINING SCRIPT
        self.vocab_size = config_dict.get('vocab_size', 478)
        self.hidden_size = config_dict.get('hidden_size', 144)
        self.num_hidden_layers = config_dict.get('num_hidden_layers', 3)
        self.num_attention_heads = config_dict.get('num_attention_heads', 6)
        self.intermediate_size = config_dict.get('intermediate_size', 256)
        self.hidden_dropout_prob = config_dict.get('hidden_dropout_prob', 0.1)
        self.attention_probs_dropout_prob = config_dict.get('attention_probs_dropout_prob', 0.1)
        self.max_position_embeddings = config_dict.get('max_position_embeddings', 64)
        self.type_vocab_size = config_dict.get('type_vocab_size', 2)
        self.seg_vocab_size = config_dict.get('seg_vocab_size', 2)
        self.age_vocab_size = config_dict.get('age_vocab_size', 1322)
        self.hidden_act = config_dict.get('hidden_act', 'gelu')
        self.initializer_range = config_dict.get('initializer_range', 0.02)

# ============================================================
# Parameter Mapping Function
# ============================================================

def map_state_dict(state_dict):
    """Map the original BEHRT parameter names to our compatible structure"""
    new_state_dict = {}
    
    # Mapping rules
    mapping_rules = {
        # Embeddings
        'bert.embeddings.word_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
        'bert.embeddings.segment_embeddings.weight': 'bert.embeddings.segment_embeddings.weight', 
        'bert.embeddings.age_embeddings.weight': 'bert.embeddings.age_embeddings.weight',
        'bert.embeddings.posi_embeddings.weight': 'bert.embeddings.posi_embeddings.weight',
        'bert.embeddings.LayerNorm.weight': 'bert.embeddings.LayerNorm.weight',
        'bert.embeddings.LayerNorm.bias': 'bert.embeddings.LayerNorm.bias',
        
        # Classifier
        'classifier.weight': 'classifier.weight',
        'classifier.bias': 'classifier.bias',
    }
    
    # Add encoder layer mappings
    for i in range(3):  # 3 layers based on your config
        # Attention
        mapping_rules.update({
            f'bert.encoder.layer.{i}.attention.self.query.weight': f'bert.encoder.layer.{i}.attention.self.query.weight',
            f'bert.encoder.layer.{i}.attention.self.query.bias': f'bert.encoder.layer.{i}.attention.self.query.bias',
            f'bert.encoder.layer.{i}.attention.self.key.weight': f'bert.encoder.layer.{i}.attention.self.key.weight',
            f'bert.encoder.layer.{i}.attention.self.key.bias': f'bert.encoder.layer.{i}.attention.self.key.bias',
            f'bert.encoder.layer.{i}.attention.self.value.weight': f'bert.encoder.layer.{i}.attention.self.value.weight',
            f'bert.encoder.layer.{i}.attention.self.value.bias': f'bert.encoder.layer.{i}.attention.self.value.bias',
            f'bert.encoder.layer.{i}.attention.output.dense.weight': f'bert.encoder.layer.{i}.attention.output.dense.weight',
            f'bert.encoder.layer.{i}.attention.output.dense.bias': f'bert.encoder.layer.{i}.attention.output.dense.bias',
            f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight': f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight',
            f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias': f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias',
            
            # Intermediate
            f'bert.encoder.layer.{i}.intermediate.dense.weight': f'bert.encoder.layer.{i}.intermediate.dense.weight',
            f'bert.encoder.layer.{i}.intermediate.dense.bias': f'bert.encoder.layer.{i}.intermediate.dense.bias',
            
            # Output
            f'bert.encoder.layer.{i}.output.dense.weight': f'bert.encoder.layer.{i}.output.dense.weight',
            f'bert.encoder.layer.{i}.output.dense.bias': f'bert.encoder.layer.{i}.output.dense.bias',
            f'bert.encoder.layer.{i}.output.LayerNorm.weight': f'bert.encoder.layer.{i}.output.LayerNorm.weight',
            f'bert.encoder.layer.{i}.output.LayerNorm.bias': f'bert.encoder.layer.{i}.output.LayerNorm.bias',
        })
    
    # Pooler
    mapping_rules.update({
        'bert.pooler.dense.weight': 'bert.pooler.0.weight',
        'bert.pooler.dense.bias': 'bert.pooler.0.bias',
    })
    
    # Apply mapping
    for old_key, new_key in mapping_rules.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict[old_key]
    
    return new_state_dict

# ============================================================
# Startup: Load Model - WITH PARAMETER MAPPING
# ============================================================

@app.on_event("startup")
async def load_model_and_vocab():
    global model, vocab, age_vocab, device
    
    print("Loading model and vocabularies...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabularies
    vocab_path = '../data/processed/vocab_ccsr.pkl'
    age_vocab_path = '../data/processed/age_vocab.pkl'
    
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"✓ Loaded vocabulary: {len(vocab['token2idx'])} tokens")
        
        with open(age_vocab_path, 'rb') as f:
            age_vocab = pickle.load(f)
        print(f"✓ Loaded age vocabulary: {len(age_vocab['age2idx'])} tokens")
        
    except FileNotFoundError as e:
        print(f"⚠️  Vocabulary files not found: {e}")
        print("Using dummy vocab for demo")
        vocab = {
            'token2idx': {'PAD': 0, 'CLS': 1, 'SEP': 2, 'MASK': 3, 'UNK': 4},
            'idx2token': {0: 'PAD', 1: 'CLS', 2: 'SEP', 3: 'MASK', 4: 'UNK'}
        }
        age_vocab = {'age2idx': {'PAD': 0, 'UNK': 1}, 'idx2age': {0: 'PAD', 1: 'UNK'}}
    
    # Load model
    model_path = '../data/models/quick/behrt_nextvisit_ccsr_quick.pt'
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✓ Loaded checkpoint from: {model_path}")
        
        # Extract ACTUAL vocabulary size from the checkpoint
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        actual_vocab_size = state_dict['classifier.weight'].shape[0]
        
        print(f"✓ Detected actual vocabulary size from checkpoint: {actual_vocab_size}")
        print(f"✓ Current vocabulary size: {len(vocab['token2idx'])}")
        
        # EXACT CONFIGURATION
        config_dict = {
            'vocab_size': len(vocab['token2idx']),
            'hidden_size': 144,
            'num_hidden_layers': 3,
            'num_attention_heads': 6,
            'intermediate_size': 256,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 64,
            'type_vocab_size': 2,
            'seg_vocab_size': 2,
            'age_vocab_size': len(age_vocab['age2idx']),
            'hidden_act': 'gelu',
            'initializer_range': 0.02,
        }
        
        config = BertConfig(config_dict)
        
        print("✓ Using EXACT model configuration from training")
        
        # Create model with EXACT configuration and ACTUAL vocab size
        model = BEHRTForPrediction(config, actual_vocab_size=actual_vocab_size)
        print("✓ Created BEHRTForPrediction model with compatible structure")
        
        # Map state dict to our model structure
        mapped_state_dict = map_state_dict(state_dict)
        
        # Load state dict
        model.load_state_dict(mapped_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"✓ Successfully loaded BEHRT model")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Check which parameters were loaded
        model_state_dict = model.state_dict()
        loaded_count = 0
        total_count = len(model_state_dict)
        
        for name, param in model_state_dict.items():
            if name in mapped_state_dict and param.shape == mapped_state_dict[name].shape:
                loaded_count += 1
        
        print(f"✓ Loaded {loaded_count}/{total_count} parameters successfully")
        
        if actual_vocab_size != len(vocab['token2idx']):
            print(f"⚠️  Vocabulary size mismatch: model trained with {actual_vocab_size}, current vocab has {len(vocab['token2idx'])}")
        
    except FileNotFoundError:
        print(f"⚠️  Model not found: {model_path}")
        print("Running in demo mode without actual predictions")
        model = None
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        import traceback
        traceback.print_exc()
        print("Running in demo mode without actual predictions")
        model = None

# ============================================================
# Helper Functions - FIXED FOR VOCAB MISMATCH
# ============================================================

def encode_sequence(visits: List[Visit]) -> tuple:
    """Convert patient history to model input"""
    
    codes = ['CLS']
    ages = [visits[0].age_months if visits else 0]
    segments = [0]  # CLS segment
    
    for i, visit in enumerate(visits):
        # Add visit codes
        for code in visit.codes:
            # Check if code exists in vocabulary, otherwise use UNK
            if code in vocab['token2idx']:
                token = code
            else:
                token = 'UNK'
                print(f"⚠️  Code '{code}' not in vocabulary, using 'UNK'")
            codes.append(token)
            ages.append(visit.age_months)
            segments.append(0)  # Same segment for all codes in visit
        
        # Add separator between visits (but not after last visit)
        if i < len(visits) - 1:
            codes.append('SEP')
            ages.append(visit.age_months)
            segments.append(1)  # SEP has different segment
    
    # Truncate if sequence exceeds max length (64)
    if len(codes) > 64:
        print(f"⚠️  Truncating sequence from {len(codes)} to 64 tokens")
        codes = codes[:64]
        ages = ages[:64]
        segments = segments[:64]
    
    # Convert to indices - handle vocabulary mapping
    code_indices = []
    for c in codes:
        if c in vocab['token2idx']:
            idx = vocab['token2idx'][c]
            # If model was trained with smaller vocab, remap indices if necessary
            if model and idx >= model.actual_vocab_size:
                # Map to UNK token if index is out of bounds for the model
                code_indices.append(vocab['token2idx']['UNK'])
                print(f"⚠️  Code '{c}' index {idx} exceeds model vocab size {model.actual_vocab_size}, using UNK")
            else:
                code_indices.append(idx)
        else:
            code_indices.append(vocab['token2idx']['UNK'])
    
    # Age indices
    age_indices = []
    for age in ages:
        age_str = str(age)
        age_idx = age_vocab['age2idx'].get(age_str, age_vocab['age2idx']['UNK'])
        age_indices.append(age_idx)
    
    # Position indices
    position_indices = list(range(len(codes)))
    
    print(f"✓ Encoded sequence: {codes}")
    print(f"✓ Code indices: {code_indices}")
    
    return code_indices, age_indices, segments, position_indices, codes

def decode_predictions(logits: torch.Tensor, top_k: int = 10) -> List[Dict]:
    """Convert model output to readable predictions"""
    
    # Use softmax for multi-class prediction
    probs = torch.softmax(logits[0], dim=0)
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
    
    predictions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        # Handle vocabulary size mismatch
        if model and idx >= model.actual_vocab_size:
            # This index is from the model's vocabulary, which might be smaller
            # We need to map it to our current vocabulary if possible
            if idx < len(vocab['idx2token']):
                code = vocab['idx2token'][idx]
            else:
                continue  # Skip if index is out of bounds
        else:
            code = vocab['idx2token'].get(idx, 'UNKNOWN')
        
        # Skip special tokens
        if code in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']:
            continue
        
        predictions.append({
            'code': code,
            'probability': round(prob, 4),
            'description': get_code_description(code)
        })
    
    return predictions[:top_k]

def get_code_description(code: str) -> str:
    """Get human-readable description for code"""
    
    descriptions = {
        'CIR008': 'Acute myocardial infarction',
        'CIR003': 'Essential hypertension',
        'END004': 'Diabetes mellitus',
        'RSP006': 'COPD and bronchiectasis',
        'DIG011': 'Liver disease',
        'GEN003': 'Chronic kidney disease',
        'NEO014': 'Cancer of breast',
        'MBD008': 'Mood disorders',
        'NVS008': 'Epilepsy and convulsions',
        'INF002': 'Septicemia',
        'CIR007': 'Heart failure',
        'RSP004': 'Pneumonia',
        'END003': 'Thyroid disorders',
        'CIR009': 'Ischemic heart disease',
        'DIG010': 'Gallbladder disease',
        'MSS003': 'Osteoarthritis',
    }
    
    return descriptions.get(code, f'Diagnosis: {code}')

def get_demo_predictions(patient_history: List[Visit], top_k: int = 10) -> List[Dict]:
    """Generate realistic demo predictions based on patient history"""
    
    common_progressions = {
        'CIR003': ['END004', 'CIR008', 'GEN003'],
        'END004': ['CIR008', 'GEN003', 'NVS008'],
        'RSP006': ['CIR008', 'INF002', 'RSP004'],
        'CIR008': ['CIR007', 'GEN003', 'END004'],
        'GEN003': ['CIR007', 'END004', 'INF002'],
    }
    
    all_codes = []
    for visit in patient_history:
        all_codes.extend(visit.codes)
    
    seen = set()
    unique_codes = []
    for code in all_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    
    predictions = []
    base_prob = 0.8
    
    for code in unique_codes[-3:]:
        if code in common_progressions:
            for next_code in common_progressions[code][:2]:
                if next_code not in unique_codes:
                    predictions.append({
                        'code': next_code,
                        'probability': round(base_prob, 4),
                        'description': get_code_description(next_code)
                    })
                    base_prob *= 0.7
    
    if not predictions:
        common_predictions = [
            {"code": "CIR003", "probability": 0.65, "description": "Essential hypertension"},
            {"code": "END004", "probability": 0.55, "description": "Diabetes mellitus"},
            {"code": "MBD008", "probability": 0.45, "description": "Mood disorders"},
        ]
        predictions = common_predictions
    
    return predictions[:top_k]

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    vocab_info = {
        "current_vocab_size": len(vocab['token2idx']) if vocab else 0,
        "model_vocab_size": model.actual_vocab_size if model else 0,
        "vocab_mismatch": model and model.actual_vocab_size != len(vocab['token2idx']) if model else False
    }
    
    return {
        "message": "BEHRT Diagnosis Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "vocab_info": vocab_info,
        "max_sequence_length": 64
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "loaded" if model is not None else "not loaded", 
        "device": str(device) if device else "unknown",
        "config": "quick (144 hidden, 3 layers, max_seq=64)" if model is not None else "demo",
        "vocab_size": f"{model.actual_vocab_size} (model) vs {len(vocab['token2idx'])} (current)" if model else "unknown"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_diagnoses(request: PredictionRequest):
    """
    Predict next visit diagnoses based on patient history
    """
    
    if not request.patient_history:
        raise HTTPException(status_code=400, detail="Patient history cannot be empty")
    
    if model is None:
        demo_predictions = get_demo_predictions(request.patient_history, request.top_k)
        return PredictionResponse(
            predictions=demo_predictions,
            patient_summary={
                "total_visits": len(request.patient_history),
                "total_diagnoses": sum(len(v.codes) for v in request.patient_history),
                "age_years": request.patient_history[-1].age_months // 12
            }
        )
    
    try:
        code_indices, age_indices, seg_indices, pos_indices, codes = encode_sequence(request.patient_history)
        
        code_tensor = torch.tensor([code_indices]).to(device)
        age_tensor = torch.tensor([age_indices]).to(device)
        seg_tensor = torch.tensor([seg_indices]).to(device)
        pos_tensor = torch.tensor([pos_indices]).to(device)
        
        attention_mask = torch.ones_like(code_tensor)
        
        print(f"✓ Running prediction with sequence length: {len(code_indices)}")
        
        with torch.no_grad():
            logits = model(
                input_ids=code_tensor,
                age_ids=age_tensor,
                seg_ids=seg_tensor,
                posi_ids=pos_tensor,
                attention_mask=attention_mask
            )
        
        print(f"✓ Model output shape: {logits.shape}")
        
        predictions = decode_predictions(logits, top_k=request.top_k)
        
        patient_summary = {
            "total_visits": len(request.patient_history),
            "total_diagnoses": sum(len(v.codes) for v in request.patient_history),
            "age_years": request.patient_history[-1].age_months // 12,
            "sequence_length": len(codes),
            "max_sequence_allowed": 64,
            "vocab_mismatch": model.actual_vocab_size != len(vocab['token2idx'])
        }
        
        return PredictionResponse(
            predictions=predictions,
            patient_summary=patient_summary
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/vocab/search")
async def search_vocabulary(query: str, limit: int = 20):
    if not vocab:
        return {"results": []}
    
    query_upper = query.upper()
    matches = []
    
    for code in vocab['token2idx'].keys():
        if code in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']:
            continue
        
        if query_upper in code:
            matches.append({
                'code': code,
                'description': get_code_description(code)
            })
        
        if len(matches) >= limit:
            break
    
    return {"results": matches}

# Add this debug endpoint
@app.get("/model-debug")
async def model_debug():
    """Debug model and vocabulary issues"""
    if model is None:
        return {"error": "Model not loaded"}
    
    # Test with a simple known sequence
    test_codes = ["CCSR_CIR003", "CCSR_END003", "CCSR_CIR008"]
    
    # Check if these exist
    code_check = {}
    for code in test_codes:
        code_check[code] = {
            "in_vocab": code in vocab['token2idx'],
            "index": vocab['token2idx'].get(code, "MISSING"),
            "in_model_range": code in vocab['token2idx'] and vocab['token2idx'][code] < model.actual_vocab_size
        }
    
    return {
        "model_actual_vocab_size": model.actual_vocab_size,
        "current_vocab_size": len(vocab['token2idx']),
        "code_check": code_check,
        "issue": "Any 'in_model_range: false' means the code exists in vocab but model wasn't trained on it"
    }

@app.get("/quick-check")
async def quick_check():
    """Quick vocabulary check"""
    if model is None:
        return {"error": "Model not loaded"}
    
    test_codes = ["CCSR_CIR003", "CCSR_END003", "CCSR_CIR008", "CCSR_END004"]
    
    results = {}
    for code in test_codes:
        in_vocab = code in vocab['token2idx']
        if in_vocab:
            idx = vocab['token2idx'][code]
            in_model_range = idx < model.actual_vocab_size
        else:
            idx = "NOT_FOUND"
            in_model_range = False
        
        results[code] = {
            "in_vocab": in_vocab,
            "index": idx,
            "in_model_range": in_model_range
        }
    
    return {
        "model_vocab_size": model.actual_vocab_size,
        "current_vocab_size": len(vocab['token2idx']),
        "code_check": results
    }

@app.get("/list-available-codes")
async def list_available_codes(limit: int = 10):
    """List codes that are actually in model range"""
    available = []
    for code, idx in vocab['token2idx'].items():
        if idx < model.actual_vocab_size and code not in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']:
            available.append(f"{code} (index: {idx})")
        if len(available) >= limit:
            break
    
    return {
        "available_codes_sample": available,
        "total_available": sum(1 for code, idx in vocab['token2idx'].items() 
                              if idx < model.actual_vocab_size and code not in ['PAD', 'CLS', 'SEP', 'MASK', 'UNK'])
    }

@app.get("/parameter-check")
async def parameter_check():
    """Check if model parameters loaded correctly"""
    if model is None:
        return {"error": "Model not loaded"}
    
    # Reload checkpoint to check
    model_path = '../data/models/quick/behrt_nextvisit_ccsr_quick.pt'
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    model_state_dict = model.state_dict()
    loaded_count = 0
    total_count = len(model_state_dict)
    mismatch_details = []
    
    for name, param in model_state_dict.items():
        if name in state_dict:
            if param.shape == state_dict[name].shape:
                loaded_count += 1
            else:
                mismatch_details.append(f"Shape mismatch: {name} - expected {param.shape}, got {state_dict[name].shape}")
        else:
            mismatch_details.append(f"Missing: {name}")
    
    return {
        "total_parameters": total_count,
        "loaded_parameters": loaded_count,
        "loading_rate": f"{(loaded_count/total_count)*100:.1f}%",
        "status": "GOOD" if (loaded_count/total_count) > 0.9 else "POOR",
        "mismatches": mismatch_details[:5]  # First 5 issues
    }

@app.get("/model-output-test")
async def model_output_test():
    """Test raw model outputs to see if they make sense"""
    if model is None:
        return {"error": "Model not loaded"}
    
    # Test with simple input
    test_input = torch.tensor([[vocab['token2idx']['CLS']]]).to(device)
    test_age = torch.tensor([[age_vocab['age2idx']['360']]]).to(device)
    
    with torch.no_grad():
        logits = model(input_ids=test_input, age_ids=test_age)
    
    # Analyze the output
    probs = torch.softmax(logits[0], dim=0)
    
    # Check if outputs are reasonable
    max_prob = torch.max(probs).item()
    min_prob = torch.min(probs).item()
    mean_prob = torch.mean(probs).item()
    std_prob = torch.std(probs).item()
    
    # Count how many predictions have reasonable probability
    reasonable_predictions = torch.sum(probs > 0.01).item()
    
    return {
        "max_probability": max_prob,
        "min_probability": min_prob, 
        "mean_probability": mean_prob,
        "std_probability": std_prob,
        "reasonable_predictions_count": reasonable_predictions,
        "total_predictions": len(probs),
        "diagnosis": "RANDOM" if max_prob < 0.01 else "TRAINED",
        "expected_mean": f"~{1/len(probs):.6f} (random)",
        "actual_mean": f"{mean_prob:.6f}"
    }

@app.get("/find-models")
async def find_models():
    """Look for other model files"""
    model_dir = '../data/models/'
    models_found = []
    
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                full_path = os.path.join(root, file)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                models_found.append({
                    'path': full_path,
                    'size_mb': round(size_mb, 2),
                    'name': file
                })
    
    return {
        "models_found": models_found,
        "recommendation": "Try loading different .pt files if available"
    } 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)