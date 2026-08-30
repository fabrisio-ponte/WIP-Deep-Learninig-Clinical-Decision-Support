#!/usr/bin/env python3
"""Summarize clean training runs and compute seed-wise mean/std metrics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


def load_metrics(run_dir: Path):
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    project_root = Path(__file__).resolve().parents[1]
    clean_runs_dir = project_root / "data" / "models" / "clean_runs"

    runs = []
    for run_dir in sorted(clean_runs_dir.glob("clean_run_*")):
        data = load_metrics(run_dir)
        if not data:
            continue
        run_controls = data.get("run_controls", {})
        metrics = data.get("metrics", {})
        runs.append(
            {
                "run_id": data.get("run_id", run_dir.name),
                "seed": run_controls.get("seed"),
                "aps": metrics.get("sample_wise_aps"),
                "auc": metrics.get("sample_wise_auc"),
                "epochs": data.get("training", {}).get("epochs"),
                "sample_limit": run_controls.get("sample_limit"),
            }
        )

    # Keep only full runs with explicit seed metadata.
    full_seeded = [
        r
        for r in runs
        if r["seed"] is not None and r["epochs"] == 3 and (r["sample_limit"] in (0, None))
    ]

    print("Full seeded clean runs:")
    for r in full_seeded:
        print(
            f"  {r['run_id']} seed={r['seed']} APS={r['aps']:.6f} AUC={r['auc']:.6f}"
        )

    if len(full_seeded) >= 2:
        aps_values = [r["aps"] for r in full_seeded]
        auc_values = [r["auc"] for r in full_seeded]
        print("\nAggregate statistics:")
        print(f"  APS mean={mean(aps_values):.6f} std={stdev(aps_values):.6f}")
        print(f"  AUC mean={mean(auc_values):.6f} std={stdev(auc_values):.6f}")
    else:
        print("\nNeed at least two full seeded runs to compute standard deviation.")


if __name__ == "__main__":
    main()
