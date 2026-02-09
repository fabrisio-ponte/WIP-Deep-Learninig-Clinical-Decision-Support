"""
BEHRT Reverse Engineering Setup
===============================

This script sets up the reverse engineering environment by copying necessary files
from the working BEHRT_run implementation to BEHRT_run_reverse_eng for experimentation.

IMPORTANT: Run this AFTER committing your current work to git!

Usage:
    cd BEHRT_run_reverse_eng
    python setup_reverse_engineering.py
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
import os

class ReverseEngineeringSetup:
    def __init__(self):
        self.source_path = Path("../BEHRT_run/BEHRT_Project")
        self.target_path = Path(".")
        self.backup_info = {
            'setup_date': datetime.now().isoformat(),
            'source_path': str(self.source_path.absolute()),
            'copied_files': [],
            'git_commit': self._get_git_commit(),
            'baseline_metrics': None
        }
    
    def _get_git_commit(self):
        """Get current git commit for tracking"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.source_path)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def copy_essential_files(self):
        """Copy essential files needed for experiments"""
        
        print("=" * 70)
        print("SETTING UP BEHRT REVERSE ENGINEERING ENVIRONMENT")
        print("=" * 70)
        
        # Essential directories and files to copy
        copy_plan = {
            # Core model implementation
            'model/': ['*.py'],
            'dataLoader/': ['*.py'], 
            'common/': ['*.py'],
            
            # Training scripts and notebooks
            'notebooks/': ['NextXVisit_QUICK.ipynb', 'NextXVisit.ipynb'],
            'quickstart.py': None,
            
            # Data processing
            'preprocessing/': ['*.py'],  
            'cssr_mappings/': ['**/*'],  # Keep all mapping files
            
            # Configuration files
            'requirements.txt': None,
            'README.md': None,
            
            # Analysis utilities (for baseline measurement)
            'utils/': ['analyze_*.py']
        }
        
        for source_item, patterns in copy_plan.items():
            source_full = self.source_path / source_item
            
            if source_full.is_file():
                # Single file
                target_full = self.target_path / source_item
                target_full.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_full, target_full)
                self.backup_info['copied_files'].append(str(source_item))
                print(f"✓ Copied file: {source_item}")
                
            elif source_full.is_dir():
                # Directory with patterns
                target_dir = self.target_path / source_item
                target_dir.mkdir(parents=True, exist_ok=True)
                
                if patterns is None or '**/*' in patterns:
                    # Copy entire directory
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(source_full, target_dir)
                    self.backup_info['copied_files'].append(f"{source_item} (full directory)")
                    print(f"✓ Copied directory: {source_item}")
                else:
                    # Copy specific patterns
                    for pattern in patterns:
                        for file_path in source_full.glob(pattern):
                            if file_path.is_file():
                                relative_path = file_path.relative_to(self.source_path)
                                target_file = self.target_path / relative_path
                                target_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(file_path, target_file)
                                self.backup_info['copied_files'].append(str(relative_path))
                    print(f"✓ Copied patterns from: {source_item}")
    
    def setup_data_links(self):
        """Set up data directory structure (links to avoid duplication)"""
        
        print("\nSetting up data directory links...")
        
        # Create data directory structure
        data_dir = self.target_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Link to processed data (don't duplicate large files)
        source_processed = self.source_path / "data" / "processed"
        target_processed = data_dir / "processed"
        
        if source_processed.exists() and not target_processed.exists():
            try:
                # Try to create symlink (works on Unix systems)
                target_processed.symlink_to(source_processed.absolute())
                print("✓ Created symlink to processed data")
            except (OSError, NotImplementedError):
                # Fall back to copying key files
                target_processed.mkdir(exist_ok=True)
                for file in source_processed.glob("*.parquet"):
                    if file.stat().st_size < 100 * 1024 * 1024:  # Only copy files < 100MB
                        shutil.copy2(file, target_processed)
                        print(f"✓ Copied {file.name}")
                    else:
                        print(f"⚠ Skipped large file: {file.name} (create symlink manually)")
        
        # Create subdirectories for experiments
        (data_dir / "experiments").mkdir(exist_ok=True)
        (data_dir / "controlled_data").mkdir(exist_ok=True)
        (data_dir / "results").mkdir(exist_ok=True)
    
    def create_experiment_structure(self):
        """Create directory structure for experiments"""
        
        print("\nCreating experiment structure...")
        
        experiment_dirs = [
            "experiments/perfect_progressions",
            "experiments/optimal_temporal", 
            "experiments/amplified_comorbidity",
            "experiments/minimal_complexity",
            "experiments/baseline_comparison",
            "results/training_logs",
            "results/metrics",
            "results/analysis"
        ]
        
        for dir_path in experiment_dirs:
            (self.target_path / dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {dir_path}/")
    
    def copy_baseline_metrics(self):
        """Copy current performance metrics as baseline"""
        
        print("\nCopying baseline metrics...")
        
        # Copy training logs
        source_logs = self.source_path / "data" / "models" / "quick"
        target_logs = self.target_path / "results" / "baseline"
        
        if source_logs.exists():
            target_logs.mkdir(parents=True, exist_ok=True)
            for log_file in source_logs.glob("*.txt"):
                shutil.copy2(log_file, target_logs)
                print(f"✓ Copied baseline log: {log_file.name}")
        
        # Extract latest metrics from logs
        metrics = self._extract_latest_metrics()
        if metrics:
            self.backup_info['baseline_metrics'] = metrics
            
            with open(self.target_path / "results" / "baseline_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"✓ Baseline metrics: APS={metrics.get('aps', 'N/A'):.4f}, ROC-AUC={metrics.get('roc_auc', 'N/A'):.4f}")
    
    def _extract_latest_metrics(self):
        """Extract latest performance metrics from training logs"""
        log_file = self.source_path / "data" / "models" / "quick" / "nextvisit_training_log.txt"
        
        if not log_file.exists():
            return None
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find last epoch results
            latest_metrics = {}
            for line in reversed(lines):
                if "APS (Average Precision Score):" in line:
                    latest_metrics['aps'] = float(line.split(":")[1].strip())
                elif "ROC-AUC:" in line and 'aps' in latest_metrics:
                    latest_metrics['roc_auc'] = float(line.split(":")[1].strip())
                    break
            
            return latest_metrics if latest_metrics else None
        except:
            return None
    
    def create_environment_info(self):
        """Create environment documentation"""
        
        print("\nCreating environment documentation...")
        
        # Save setup info
        with open(self.target_path / "setup_info.json", 'w') as f:
            json.dump(self.backup_info, f, indent=2)
        
        # Create README
        readme_content = f"""# BEHRT Reverse Engineering Environment

## Setup Information
- **Setup Date**: {self.backup_info['setup_date']}
- **Source**: {self.backup_info['source_path']}
- **Git Commit**: {self.backup_info['git_commit']}
- **Baseline APS**: {self.backup_info.get('baseline_metrics', {}).get('aps', 'N/A')}
- **Baseline ROC-AUC**: {self.backup_info.get('baseline_metrics', {}).get('roc_auc', 'N/A')}

## Objective
Reverse engineer BEHRT to understand what data characteristics can increase:
- Current APS: ~0.40
- Target APS: 0.85-0.90 (85-90% accuracy)

## Experimental Design
1. **Perfect Disease Progressions**: Create deterministic disease sequences
2. **Optimal Temporal Patterns**: Perfect age-disease correlations
3. **Amplified Comorbidity Signals**: Strengthen disease associations  
4. **Minimal Complexity**: Find minimal dataset for maximum performance

## Original BEHRT Paper Performance
- Next Visit: 0.462 APS | 0.954 ROC-AUC
- Next 6M: 0.525 APS | 0.958 ROC-AUC  
- Next 12M: 0.506 APS | 0.955 ROC-AUC

## Directory Structure
```
experiments/          # Experimental datasets
├── perfect_progressions/
├── optimal_temporal/
├── amplified_comorbidity/
└── minimal_complexity/

results/               # Training results and analysis
├── baseline/          # Original performance metrics
├── training_logs/     # Experimental training logs
├── metrics/           # Performance comparisons
└── analysis/          # What drives performance insights

data/
├── processed/         # Link to original processed data
├── controlled_data/   # Manipulated experimental datasets
└── experiments/       # Data for each experiment
```

## Usage
1. Run experiments: `python run_reverse_engineering_experiments.py`
2. Train models: `python train_experimental_models.py`
3. Analyze results: `python analyze_experimental_results.py`

## Safety
- Original BEHRT_run implementation is preserved
- All changes are in this isolated environment
- Git commit {self.backup_info['git_commit']} was the baseline
"""
        
        with open(self.target_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("✓ Created environment documentation")
    
    def run_setup(self):
        """Run complete setup process"""
        
        # Verify we're in the right directory
        if not Path("../BEHRT_run/BEHRT_Project").exists():
            print("❌ Error: Run this script from BEHRT_run_reverse_eng directory")
            print("❌ BEHRT_run/BEHRT_Project not found")
            return False
        
        # Check if setup already exists
        if (self.target_path / "setup_info.json").exists():
            print("⚠ Setup already exists. Remove setup_info.json to re-run setup.")
            return False
        
        try:
            self.copy_essential_files()
            self.setup_data_links()
            self.create_experiment_structure()
            self.copy_baseline_metrics()
            self.create_environment_info()
            
            print("\n" + "=" * 70)
            print("✅ REVERSE ENGINEERING ENVIRONMENT SETUP COMPLETE!")
            print("=" * 70)
            print("Next steps:")
            print("1. Review the created structure")
            print("2. Run: python run_reverse_engineering_experiments.py")
            print("3. Compare results to find what drives 85-90% performance")
            print("4. Analyze successful patterns for implementation insights")
            print("=" * 70)
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False

if __name__ == "__main__":
    setup = ReverseEngineeringSetup()
    setup.run_setup()