#!/usr/bin/env python3
"""
BEHRT Analysis & Cleaning Runner
===============================

Main script to run various BEHRT analysis and cleaning tools.

Usage:
    python run_analysis.py --help
    python run_analysis.py quick-analysis
    python run_analysis.py filtered-analysis  
    python run_analysis.py data-quality
    python run_analysis.py clean-phase1
    python run_analysis.py clean-phase2
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a script and handle errors"""
    print(f"🚀 Running {description}...")
    print(f"📍 Script: {script_path}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              check=True, cwd=Path(__file__).parent)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_path}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="BEHRT Analysis & Cleaning Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  quick-analysis    : Quick data overview and basic statistics
  filtered-analysis : Comprehensive analysis with XXX000 filtering
  data-quality      : Advanced data quality investigation
  clean-phase1      : Phase 1 cleaning (XXX000 removal)
  clean-phase2      : Phase 2 advanced cleaning (rare diseases, duplicates)
  
Examples:
  python run_analysis.py quick-analysis
  python run_analysis.py clean-phase1
  python run_analysis.py filtered-analysis
        """
    )
    
    parser.add_argument('command', 
                       choices=['quick-analysis', 'filtered-analysis', 'data-quality', 
                               'clean-phase1', 'clean-phase2'],
                       help='Analysis or cleaning command to run')
    
    args = parser.parse_args()
    
    # Map commands to scripts
    command_map = {
        'quick-analysis': ('analysis/analyze_data.py', 'Quick Data Analysis'),
        'filtered-analysis': ('analysis/analyze_filtered.py', 'Filtered Analysis (No XXX000)'),
        'data-quality': ('analysis/investigate_data_quality.py', 'Data Quality Investigation'),
        'clean-phase1': ('cleaning/clean_data.py', 'Phase 1 Data Cleaning'), 
        'clean-phase2': ('cleaning/clean_data_phase2.py', 'Phase 2 Advanced Cleaning')
    }
    
    script_path, description = command_map[args.command]
    
    print("🔬 BEHRT Analysis & Cleaning Runner")
    print("=" * 60)
    print(f"Command: {args.command}")
    print(f"Description: {description}")
    print()
    
    success = run_script(script_path, description)
    
    if success:
        # Provide next step suggestions
        print("\n💡 WHAT'S NEXT?")
        print("=" * 60)
        
        if args.command == 'quick-analysis':
            print("✓ Run: python run_analysis.py data-quality")
            print("  - Investigate additional data quality issues")
            
        elif args.command == 'data-quality':
            print("✓ Run: python run_analysis.py clean-phase1") 
            print("  - Clean XXX000 codes and basic issues")
            
        elif args.command == 'clean-phase1':
            print("✓ Run: python run_analysis.py filtered-analysis")
            print("  - Analyze performance on cleaned data")
            print("✓ Or: python run_analysis.py clean-phase2")
            print("  - Further cleaning (rare diseases, duplicates)")
            
        elif args.command == 'clean-phase2':
            print("✓ Use ultra-clean datasets for full model training")
            print("  - Files: *_ultraclean.parquet + vocab_ccsr_ultraclean.pkl")
            
        elif args.command == 'filtered-analysis':
            print("✓ Ready for full model training!")
            print("  - Use cleaned datasets for optimal results")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())