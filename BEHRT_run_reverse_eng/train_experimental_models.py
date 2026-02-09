#!/usr/bin/env python3
"""
Train BEHRT models on experimental datasets
"""

import subprocess
import sys
from pathlib import Path
import json

def train_experiment(experiment_name):
    """Train BEHRT on specific experimental dataset"""
    
    print(f"Training BEHRT on {experiment_name} dataset...")
    
    # Copy experiment data to data/processed for training
    exp_data = f"experiments/{experiment_name}/train_data_ccsr.parquet"
    target_data = "data/processed/train_data_ccsr.parquet"
    
    if Path(exp_data).exists():
        import shutil
        shutil.copy2(exp_data, target_data)
        print(f"✓ Copied {experiment_name} data for training")
    else:
        print(f"❌ Data file not found: {exp_data}")
        return False
    
    # Run quickstart training
    try:
        result = subprocess.run([
            sys.executable, "quickstart.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} training completed successfully")
            
            # Save results
            results_dir = Path(f"results/{experiment_name}")
            results_dir.mkdir(exist_ok=True)
            
            # Copy training logs
            if Path("data/models/quick").exists():
                import shutil
                shutil.copytree("data/models/quick", results_dir / "model_outputs", dirs_exist_ok=True)
            
            return True
        else:
            print(f"❌ {experiment_name} training failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ {experiment_name} training error: {e}")
        return False

def main():
    experiments = [
        "perfect_progressions",
        "optimal_temporal", 
        "amplified_comorbidity",
        "minimal_complexity"
    ]
    
    results = {}
    
    for experiment in experiments:
        success = train_experiment(experiment)
        results[experiment] = success
    
    # Save training summary
    with open("results/training_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY:")
    for exp, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {exp}: {status}")
    print("="*60)

if __name__ == "__main__":
    main()
