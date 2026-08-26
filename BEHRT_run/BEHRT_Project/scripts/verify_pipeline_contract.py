#!/usr/bin/env python3
"""Validate the BEHRT clean-data training pipeline contract.

Checks that the data, vocabulary, and checkpoint all agree on the same label space
before any model-quality interpretation is trusted.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_label_vocab(token2idx):
    token2idx = token2idx.copy()
    for tok in ["PAD", "SEP", "CLS", "MASK"]:
        if tok in token2idx:
            del token2idx[tok]
    return {label: idx for idx, label in enumerate(token2idx.keys())}


def main():
    project_root = PROJECT_ROOT
    data_dir = project_root / "data" / "processed"
    config_path = project_root / "config" / "analysis_config.json"
    config = load_json(config_path)

    checks = []

    data_files = {
        "train": data_dir / "train_nextvisit_ccsr_clean.parquet",
        "val": data_dir / "val_nextvisit_ccsr_clean.parquet",
        "test": data_dir / "test_nextvisit_ccsr_clean.parquet",
    }

    all_dataset_labels = set()
    for name, path in data_files.items():
        exists = path.exists()
        checks.append((f"{name}_exists", exists, str(path)))
        if exists:
            df = pd.read_parquet(path)
            checks.append((f"{name}_rows", len(df), f"{len(df)} rows"))
            labels = []
            code_history_tokens = []
            for value in df["label"]:
                if isinstance(value, str):
                    labels.append(value)
                else:
                    try:
                        labels.extend([str(x) for x in list(value)])
                    except TypeError:
                        pass

            for code_seq in df["code"]:
                if isinstance(code_seq, str):
                    code_history_tokens.append(code_seq)
                else:
                    try:
                        code_history_tokens.extend([str(x) for x in list(code_seq)])
                    except TypeError:
                        pass

            unique = sorted(set(labels))
            all_dataset_labels.update(unique)
            checks.append((f"{name}_unique_labels", len(unique), f"{len(unique)} unique labels"))
            checks.append((f"{name}_no_xxx000", "CCSR_XXX000" not in unique, str("CCSR_XXX000" not in unique)))
            checks.append((f"{name}_history_no_xxx000", "CCSR_XXX000" not in set(code_history_tokens), str("CCSR_XXX000" not in set(code_history_tokens))))

    vocab_path = data_dir / "vocab_ccsr_clean.pkl"
    vocab_exists = vocab_path.exists()
    checks.append(("vocab_exists", vocab_exists, str(vocab_path)))

    if vocab_exists:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        vocab_label_map = format_label_vocab(vocab["token2idx"])
        checks.append(("vocab_size", len(vocab_label_map), len(vocab_label_map)))
        checks.append(("vocab_no_xxx000", "CCSR_XXX000" not in vocab_label_map, str("CCSR_XXX000" not in vocab_label_map)))
        checks.append(("dataset_labels_subset_of_vocab", all_dataset_labels.issubset(set(vocab_label_map.keys())), f"{len(all_dataset_labels)} dataset labels / {len(vocab_label_map)} vocab labels"))

    checkpoint_path = project_root / config["model_path"]
    ckpt_exists = checkpoint_path.exists()
    checks.append(("checkpoint_exists", ckpt_exists, str(checkpoint_path)))

    if ckpt_exists:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            state_dict = {}

        classifier_weight = None
        for key, value in state_dict.items():
            if "classifier.weight" in key:
                classifier_weight = value
                break

        if classifier_weight is not None:
            out_features = int(classifier_weight.shape[0])
            checks.append(("checkpoint_classifier_output_dim", out_features, out_features))
            if vocab_exists:
                expected = len(format_label_vocab(vocab["token2idx"]))
                checks.append(("checkpoint_matches_vocab", out_features == expected, f"model={out_features}, vocab={expected}"))

    print("PIPELINE CONTRACT CHECK")
    print("=" * 72)
    for name, result, detail in checks:
        status = "PASS" if bool(result) else "FAIL"
        print(f"{status:4} | {name:<35} | {detail}")

    failing = [name for name, result, _ in checks if not bool(result)]
    if failing:
        print("\nFAILING CHECKS:")
        for name in failing:
            print(f" - {name}")
        raise SystemExit(1)

    print("\nAll required pipeline checks passed.")


if __name__ == "__main__":
    main()
