"""
BEHRT Reverse Engineering Environment Setup
===========================================

This script sets up a complete isolated environment for reverse engineering 
BEHRT performance through controlled data manipulation experiments.

Creates: BEHRT_run_reverse_eng/ with all necessary files to run experiments
without modifying the working BEHRT_run implementation.
"""

import shutil
import json
from pathlib import Path
from datetime import datetime

class ReverseEngineeringSetup:
    """Setup isolated reverse engineering environment"""
    
    def __init__(self, source_dir: str = "BEHRT_run", target_dir: str = "BEHRT_run_reverse_eng"):
        self.source_path = Path(source_dir)
        self.target_path = Path(target_dir)
        
        if not self.source_path.exists():
            raise ValueError(f"Source directory {source_dir} does not exist!")
    
    def setup_environment(self):
        """Set up complete reverse engineering environment"""
        
        print("🔬 Setting up BEHRT Reverse Engineering Environment")
        print("="*60)
        
        # Create target directory
        if self.target_path.exists():
            print(f"⚠️  {self.target_path} already exists - updating...")
        else:
            self.target_path.mkdir()
            print(f"📁 Created {self.target_path}")
        
        # Copy essential files and directories
        self._copy_essential_files()
        
        # Setup data links (without copying large files)
        self._setup_data_links()
        
        # Create experiment structure
        self._create_experiment_structure()
        
        # Create setup info
        self._create_setup_info()
        
        print("\n✅ REVERSE ENGINEERING ENVIRONMENT READY!")
        print(f"📍 Location: {self.target_path.absolute()}")
        print("📝 Next: cd into the directory and run python run_reverse_engineering_experiments.py")
    
    def _copy_essential_files(self):
        """Copy essential files for BEHRT operation"""
        
        essential_files = [
            # Core BEHRT code
            "BEHRT",
            "BEHRT_Project/common",
            "BEHRT_Project/dataLoader", 
            "BEHRT_Project/model",
            "BEHRT_Project/preprocessing",
            
            # Configuration files
            "BEHRT_Project/quickstart.py",
            "BEHRT_Project/requirements.txt",
            "BEHRT_Project/README.md",
            
            # CCSR mappings (essential for experiments)
            "BEHRT_Project/cssr_mappings"
        ]
        
        for item in essential_files:
            source_item = self.source_path / item
            if source_item.exists():
                if source_item.is_file():
                    target_item = self.target_path / item
                    target_item.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_item, target_item)
                    print(f"📄 Copied {item}")
                else:
                    # Directory
                    target_item = self.target_path / item
                    if target_item.exists():
                        shutil.rmtree(target_item)
                    shutil.copytree(source_item, target_item)
                    print(f"📁 Copied {item}/")
            else:
                print(f"⚠️  Missing: {item}")
    
    def _setup_data_links(self):
        """Setup data directory structure without copying large files"""
        
        # Create data directory structure
        data_dirs = [
            "data/processed",
            "data/models/quick", 
            "data/raw"
        ]
        
        for dir_path in data_dirs:
            full_path = self.target_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks to processed data instead of copying
        source_processed = self.source_path / "BEHRT_Project/data/processed"
        target_processed = self.target_path / "data/processed"
        
        if source_processed.exists():
            # Try to create symlinks for large data files
            processed_files = [
                "train_data_ccsr.parquet",
                "test_data_ccsr.parquet", 
                "val_data_ccsr.parquet",
                "age_vocab.txt",
                "ccsr_vocab.txt",
                "diag_vocab.txt"
            ]
            
            for file_name in processed_files:
                source_file = source_processed / file_name
                target_file = target_processed / file_name
                
                if source_file.exists():
                    if target_file.exists():
                        target_file.unlink()  # Remove existing file/link
                    
                    try:
                        # Create symlink (preferred for large files)
                        target_file.symlink_to(source_file.absolute())
                        print(f"🔗 Linked {file_name}")
                    except OSError:
                        # Fallback to copying if symlinks don't work
                        shutil.copy2(source_file, target_file)
                        print(f"📄 Copied {file_name}")
        
        print(f"📊 Setup data directories")
    
    def _create_experiment_structure(self):
        """Create directory structure for experiments"""
        
        experiment_dirs = [
            "experiments",
            "results",
            "logs"
        ]
        
        for dir_name in experiment_dirs:
            dir_path = self.target_path / dir_name
            dir_path.mkdir(exist_ok=True)
        
        print(f"🧪 Created experiment directories")
    
    def _create_setup_info(self):
        """Create setup information file"""
        
        setup_info = {
            "environment": "BEHRT_run_reverse_eng",
            "description": "Isolated environment for BEHRT reverse engineering experiments",
            "created": datetime.now().isoformat(),
            "source": str(self.source_path.absolute()),
            "experiments": [
                "perfect_progressions",
                "optimal_temporal",
                "amplified_comorbidity", 
                "minimal_complexity"
            ],
            "goals": {
                "current_baseline_aps": "~0.40",
                "target_aps": "0.85-0.90",
                "original_paper_aps": "0.462-0.525"
            },
            "usage": [
                "1. cd BEHRT_run_reverse_eng",
                "2. python run_reverse_engineering_experiments.py",
                "3. python train_experimental_models.py", 
                "4. python analyze_experimental_results.py"
            ]
        }
        
        with open(self.target_path / "setup_info.json", 'w') as f:
            json.dump(setup_info, f, indent=2)
        
        print(f"ℹ️  Created setup_info.json")

def main():
    """Run the setup"""
    
    # Determine source directory
    if Path("BEHRT_run").exists():
        source = "BEHRT_run"
    else:
        print("❌ BEHRT_run directory not found!")
        print("Please run this script from the directory containing BEHRT_run/")
        return
    
    # Setup environment
    setup = ReverseEngineeringSetup(source_dir=source)
    setup.setup_environment()

if __name__ == "__main__":
    main()
