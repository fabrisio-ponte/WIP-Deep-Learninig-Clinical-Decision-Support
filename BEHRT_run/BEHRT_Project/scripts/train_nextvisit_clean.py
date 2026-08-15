#!/usr/bin/env python3
"""
Train BEHRT NextVisit on cleaned CCSR data without overwriting prior artifacts.

Outputs are written to a timestamped run directory under data/models/clean_runs/.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
import pytorch_pretrained_bert as Bert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj, create_folder
from common.pytorch import save_model
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForMultiLabelPrediction
from model import optimiser


class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get("vocab_size"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config.get("num_hidden_layers"),
            num_attention_heads=config.get("num_attention_heads"),
            intermediate_size=config.get("intermediate_size"),
            hidden_act=config.get("hidden_act"),
            hidden_dropout_prob=config.get("hidden_dropout_prob"),
            attention_probs_dropout_prob=config.get("attention_probs_dropout_prob"),
            max_position_embeddings=config.get("max_position_embedding"),
            initializer_range=config.get("initializer_range"),
        )
        self.seg_vocab_size = config.get("seg_vocab_size")
        self.age_vocab_size = config.get("age_vocab_size")


def format_label_vocab(token2idx):
    token2idx = token2idx.copy()
    for tok in ["PAD", "SEP", "CLS", "MASK"]:
        if tok in token2idx:
            del token2idx[tok]
    label_vocab = {}
    for i, token in enumerate(token2idx.keys()):
        label_vocab[token] = i
    return label_vocab


def targets_to_multihot(targets, mlb, device=None):
    """Convert padded label tensors into clean multi-hot vectors for loss/metrics."""
    target_rows = targets.detach().cpu().numpy()
    cleaned_targets = []
    for row in target_rows:
        cleaned = [int(x) for x in row if int(x) >= 0]
        cleaned_targets.append(cleaned)

    target_ml = torch.tensor(mlb.transform(cleaned_targets), dtype=torch.float32)
    if device is not None:
        target_ml = target_ml.to(device)
    return target_ml


def evaluate(model, loader, mlb, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            age_ids, input_ids, posi_ids, segment_ids, att_mask, targets, _ = batch
            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            posi_ids = posi_ids.to(device)
            segment_ids = segment_ids.to(device)
            att_mask = att_mask.to(device)

            target_ml = targets_to_multihot(targets, mlb)
            logits = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=att_mask)
            probs = sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(target_ml.numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_targets)

    aps = average_precision_score(y_true, y_prob, average="samples")
    try:
        auc = roc_auc_score(y_true, y_prob, average="samples")
    except ValueError:
        auc = float("nan")

    return {"sample_wise_aps": float(aps), "sample_wise_auc": float(auc)}


def compute_pos_weight_from_train(train_df, label_vocab, max_pos_weight=30.0):
    """Compute stable per-class positive weights for BCEWithLogitsLoss.

    Weight formula: neg_count / pos_count, clipped to avoid extreme gradients.
    """
    num_classes = len(label_vocab)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for labels in train_df["label"]:
        for code in set(labels):
            idx = label_vocab.get(code)
            if idx is not None:
                class_counts[idx] += 1

    n_samples = len(train_df)
    neg_counts = n_samples - class_counts

    safe_pos = np.maximum(class_counts, 1)
    pos_weight = neg_counts / safe_pos
    pos_weight = np.clip(pos_weight, 1.0, float(max_pos_weight))

    return torch.tensor(pos_weight, dtype=torch.float32)


def main():
    seed = int(os.getenv("SEED", "42"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    run_id = datetime.now().strftime("clean_run_%Y%m%d_%H%M%S")
    run_dir = project_root / "data" / "models" / "clean_runs" / run_id
    create_folder(str(run_dir))

    train_path = data_dir / "train_nextvisit_ccsr_clean.parquet"
    val_path = data_dir / "val_nextvisit_ccsr_clean.parquet"
    test_path = data_dir / "test_nextvisit_ccsr_clean.parquet"
    vocab_path = data_dir / "vocab_ccsr_clean"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Optional quick-run controls for faster validation cycles.
    sample_limit = int(os.getenv("SAMPLE_LIMIT", "0"))
    if sample_limit > 0:
        train_df = train_df.sample(n=min(sample_limit, len(train_df)), random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(max(1, sample_limit // 5), len(val_df)), random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(n=min(max(1, sample_limit // 5), len(test_df)), random_state=seed).reset_index(drop=True)

    bert_vocab = load_obj(str(vocab_path))
    age_vocab_dict, _ = age_vocab(max_age=110, symbol=None)
    label_vocab = format_label_vocab(bert_vocab["token2idx"])

    global_params = {
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_len_seq": 64,
    }

    model_config = {
        "vocab_size": len(bert_vocab["token2idx"]),
        "hidden_size": 288,
        "seg_vocab_size": 2,
        "age_vocab_size": len(age_vocab_dict),
        "max_position_embedding": global_params["max_len_seq"],
        "hidden_dropout_prob": 0.1,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.1,
        "intermediate_size": 256,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
    }

    feature_dict = {"word": True, "seg": True, "age": True, "position": True}

    train_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=train_df,
        max_len=global_params["max_len_seq"],
    )
    val_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=val_df,
        max_len=global_params["max_len_seq"],
    )
    test_set = NextVisit(
        token2idx=bert_vocab["token2idx"],
        label2idx=label_vocab,
        age2idx=age_vocab_dict,
        dataframe=test_df,
        max_len=global_params["max_len_seq"],
    )

    train_loader = DataLoader(train_set, batch_size=global_params["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=global_params["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=global_params["batch_size"], shuffle=False, num_workers=0)

    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[x] for x in list(label_vocab.values())])

    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(conf, num_labels=len(label_vocab), feature_dict=feature_dict)
    model = model.to(global_params["device"])

    optim = optimiser.adam(
        params=list(model.named_parameters()),
        config={"lr": 5e-5, "warmup_proportion": 0.1, "weight_decay": 0.01},
    )

    best_val_aps = -1.0
    best_model_path = run_dir / "behrt_nextvisit_ccsr_clean_best.pt"
    num_epochs = int(os.getenv("EPOCHS", "3"))
    log_every = int(os.getenv("LOG_EVERY", "100"))
    use_pos_weight = os.getenv("USE_POS_WEIGHT", "0") == "1"
    max_pos_weight = float(os.getenv("MAX_POS_WEIGHT", "30.0"))

    if use_pos_weight:
        pos_weight = compute_pos_weight_from_train(train_df, label_vocab, max_pos_weight=max_pos_weight)
        pos_weight = pos_weight.to(global_params["device"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight (max clip={max_pos_weight:.1f})")
    else:
        loss_fn = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            age_ids, input_ids, posi_ids, segment_ids, att_mask, targets, _ = batch
            input_ids = input_ids.to(global_params["device"])
            age_ids = age_ids.to(global_params["device"])
            posi_ids = posi_ids.to(global_params["device"])
            segment_ids = segment_ids.to(global_params["device"])
            att_mask = att_mask.to(global_params["device"])
            target_ml = targets_to_multihot(targets, mlb, device=global_params["device"])

            optim.zero_grad()
            if loss_fn is None:
                loss, _ = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=att_mask, labels=target_ml)
            else:
                logits = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=att_mask)
                loss = loss_fn(logits, target_ml)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

            if step % max(1, log_every) == 0:
                print(f"Epoch {epoch+1}/{num_epochs} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, mlb, global_params["device"])
        avg_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{num_epochs} - loss={avg_loss:.4f} val_APS={val_metrics['sample_wise_aps']:.4f} val_AUC={val_metrics['sample_wise_auc']:.4f}")

        if val_metrics["sample_wise_aps"] > best_val_aps:
            best_val_aps = val_metrics["sample_wise_aps"]
            save_model(str(best_model_path), model)

    # Load best model weights for final test metrics
    model.load_state_dict(torch.load(best_model_path, map_location=global_params["device"]))
    test_metrics = evaluate(model, test_loader, mlb, global_params["device"])

    result = {
        "run_id": run_id,
        "data": {
            "train": str(train_path.name),
            "val": str(val_path.name),
            "test": str(test_path.name),
            "vocab": str(vocab_path.name) + ".pkl",
        },
        "model_config": model_config,
        "training": {"epochs": num_epochs, "batch_size": global_params["batch_size"]},
        "run_controls": {
            "sample_limit": sample_limit,
            "log_every": log_every,
            "seed": seed,
            "use_pos_weight": use_pos_weight,
            "max_pos_weight": max_pos_weight,
        },
        "metrics": test_metrics,
        "artifacts": {"best_model": str(best_model_path.relative_to(project_root))},
    }

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 72)
    print(f"Saved run to: {run_dir}")
    print(f"Seed: {seed}")
    print(f"Test APS: {test_metrics['sample_wise_aps']:.4f}")
    print(f"Test AUC: {test_metrics['sample_wise_auc']:.4f}")
    print(f"Metrics file: {metrics_path}")


if __name__ == "__main__":
    main()
