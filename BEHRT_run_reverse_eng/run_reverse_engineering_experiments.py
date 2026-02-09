"""
BEHRT Reverse Engineering Experiments
=====================================

Main experimental framework for systematic data manipulation to achieve 85-90% APS performance.

This is designed to run in the isolated BEHRT_run_reverse_eng environment without 
modifying the working BEHRT_run implementation.

Target: Understand what data characteristics enable near-perfect BEHRT performance
Current baseline: ~0.40 APS
Target goal: 0.85-0.90 APS  
Original paper: 0.462-0.525 APS
"""

import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging

class BEHRTReverseEngineering:
    """Main class for BEHRT reverse engineering experiments"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.data_path = self.base_path / "data"
        self.experiments_path = self.base_path / "experiments"
        self.results_path = self.base_path / "results"
        
        # Ensure directories exist
        self.experiments_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_path / 'experiments.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load baseline metrics
        self.baseline_metrics = self._load_baseline_metrics()
        
        self.logger.info("BEHRT Reverse Engineering initialized")
        self.logger.info(f"Baseline APS: {self.baseline_metrics.get('aps', 'Unknown')}")
        self.logger.info(f"Target APS: 0.85-0.90")
    
    def _load_baseline_metrics(self) -> Dict:
        """Load baseline performance metrics"""
        baseline_file = self.results_path / "baseline_metrics.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        return {'aps': 0.40, 'roc_auc': 0.90}  # Default estimates
    
    # === EXPERIMENT 1: PERFECT DISEASE PROGRESSIONS ===
    
    def create_perfect_progressions_experiment(self) -> str:
        """
        Create dataset with perfectly predictable disease progression patterns
        """
        experiment_name = "perfect_progressions"
        experiment_dir = self.experiments_path / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating {experiment_name} experiment")
        
        # Define deterministic disease progressions based on medical knowledge
        medical_progressions = {
            # Type 2 Diabetes progression (very predictable)
            'diabetes_progression': {
                'sequence': ['CCSR_END009', 'CCSR_END010', 'CCSR_CIR007', 'CCSR_EYE001'],
                'ages': [45, 47, 49, 52],
                'descriptions': ['Diabetes T2', 'Diabetes complications', 'Hypertension', 'Diabetic eye']
            },
            
            # Cardiovascular cascade (highly predictable)
            'cardiovascular_cascade': {
                'sequence': ['CCSR_CIR007', 'CCSR_CIR006', 'CCSR_CIR011', 'CCSR_CIR004'],
                'ages': [50, 55, 60, 62],
                'descriptions': ['Hypertension', 'CAD', 'MI', 'Heart failure']
            },
            
            # Mental health progression (predictable patterns)
            'mental_health_progression': {
                'sequence': ['CCSR_MBD006', 'CCSR_MBD005', 'CCSR_MBD011', 'CCSR_NVS012'],
                'ages': [25, 28, 32, 35],
                'descriptions': ['Anxiety', 'Depression', 'Substance abuse', 'Sleep disorders']
            },
            
            # Cancer progression (very predictable once started)
            'cancer_progression': {
                'sequence': ['CCSR_NEO019', 'CCSR_NEO038', 'CCSR_SYM010', 'CCSR_FAC007'],
                'ages': [60, 61, 62, 63],
                'descriptions': ['Primary cancer', 'Metastases', 'Cachexia', 'Palliative care']
            },
            
            # Degenerative progression (age-related, very predictable)
            'degenerative_progression': {
                'sequence': ['CCSR_MUS004', 'CCSR_NVS008', 'CCSR_NVS001', 'CCSR_GEN001'],
                'ages': [65, 70, 75, 80],
                'descriptions': ['Arthritis', 'Headaches', 'Dementia', 'Functional decline']
            }
        }
        
        # Create perfect dataset
        perfect_data = []
        patient_id_counter = 0
        
        for progression_name, progression_info in medical_progressions.items():
            sequence = progression_info['sequence']
            ages = progression_info['ages']
            
            # Create 150 perfect patients per progression (total: 750 patients)
            for patient_num in range(150):
                patient_id = f"perfect_{progression_name}_{patient_num:03d}"
                
                # Create perfect sequence
                for visit_idx, (disease, age) in enumerate(zip(sequence, ages)):
                    # Add small random variation to age (±1 year) to avoid perfect determinism
                    age_variation = age + np.random.randint(-1, 2)
                    
                    perfect_data.append({
                        'subject_id': patient_id,
                        'visit_concept_orders': visit_idx + 1,
                        'concept_ids': disease,
                        'ages': f'AGE[{age_variation}]',
                        'visit_dates': f'2020-{visit_idx+1:02d}-01'
                    })
        
        # Convert to DataFrame
        perfect_df = pd.DataFrame(perfect_data)
        
        # Save experimental dataset
        perfect_df.to_parquet(experiment_dir / "train_data_ccsr.parquet")
        
        # Save experiment configuration
        config = {
            'experiment_name': experiment_name,
            'description': 'Perfectly predictable disease progression sequences',
            'progressions': medical_progressions,
            'total_patients': len(perfect_df['subject_id'].unique()),
            'total_visits': len(perfect_df),
            'created': datetime.now().isoformat()
        }
        
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Created {len(perfect_df)} visits for {config['total_patients']} perfect patients")
        return str(experiment_dir)
    
    # === EXPERIMENT 2: OPTIMAL TEMPORAL SPACING ===
    
    def create_optimal_temporal_experiment(self) -> str:
        """
        Create dataset with optimal temporal spacing and age-disease correlations
        """
        experiment_name = "optimal_temporal"
        experiment_dir = self.experiments_path / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating {experiment_name} experiment")
        
        # Define optimal age-disease relationships (based on epidemiological knowledge)
        age_disease_optimal = {
            # Young adults (20-35): Mental health, reproductive health
            (20, 35): ['CCSR_MBD006', 'CCSR_MBD005', 'CCSR_PRG001', 'CCSR_INJ001'],
            
            # Middle age (35-50): Metabolic onset
            (35, 50): ['CCSR_CIR007', 'CCSR_END009', 'CCSR_DIG006', 'CCSR_MUS004'],
            
            # Older adults (50-65): Chronic disease complications
            (50, 65): ['CCSR_END010', 'CCSR_CIR006', 'CCSR_NEO019', 'CCSR_RES005'],
            
            # Elderly (65-80): Degenerative and complex conditions
            (65, 80): ['CCSR_NVS001', 'CCSR_CIR004', 'CCSR_GEN001', 'CCSR_SYM001'],
            
            # Very elderly (80+): End-stage conditions
            (80, 95): ['CCSR_FAC007', 'CCSR_CIR012', 'CCSR_NVS013', 'CCSR_GEN002']
        }
        
        optimal_data = []
        
        # Create optimal temporal sequences
        for patient_num in range(1000):  # 1000 patients with optimal patterns
            patient_id = f"optimal_temporal_{patient_num:04d}"
            
            # Random starting age between 20-30
            current_age = np.random.randint(20, 31)
            visit_num = 1
            
            # Create visits with perfect temporal progression
            while current_age < 90 and visit_num <= 12:  # Max 12 visits
                # Find appropriate diseases for current age
                suitable_diseases = []
                for (min_age, max_age), diseases in age_disease_optimal.items():
                    if min_age <= current_age < max_age:
                        suitable_diseases = diseases
                        break
                
                if suitable_diseases:
                    # Select disease(s) for this visit
                    n_diseases = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])  # Usually 1 disease
                    visit_diseases = np.random.choice(suitable_diseases, size=min(n_diseases, len(suitable_diseases)), replace=False)
                    
                    for disease in visit_diseases:
                        optimal_data.append({
                            'subject_id': patient_id,
                            'visit_concept_orders': visit_num,
                            'concept_ids': disease,
                            'ages': f'AGE[{current_age}]',
                            'visit_dates': f'2020-{visit_num:02d}-01'
                        })
                
                # Perfect temporal progression: advance age by optimal interval
                if current_age < 40:
                    age_increment = np.random.randint(2, 4)  # 2-3 years when young
                elif current_age < 65:
                    age_increment = np.random.randint(1, 3)  # 1-2 years in middle age
                else:
                    age_increment = 1  # 1 year when elderly
                
                current_age += age_increment
                visit_num += 1
        
        # Convert to DataFrame and save
        optimal_df = pd.DataFrame(optimal_data)
        optimal_df.to_parquet(experiment_dir / "train_data_ccsr.parquet")
        
        # Save configuration
        config = {
            'experiment_name': experiment_name,
            'description': 'Optimal temporal spacing and age-disease correlations',
            'age_disease_mapping': {f"{k[0]}-{k[1]}": v for k, v in age_disease_optimal.items()},
            'total_patients': len(optimal_df['subject_id'].unique()),
            'total_visits': len(optimal_df),
            'created': datetime.now().isoformat()
        }
        
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Created {len(optimal_df)} visits with optimal temporal patterns")
        return str(experiment_dir)
    
    # === EXPERIMENT 3: AMPLIFIED COMORBIDITY SIGNALS ===
    
    def create_amplified_comorbidity_experiment(self) -> str:
        """
        Create dataset with strongly amplified comorbidity patterns
        """
        experiment_name = "amplified_comorbidity"
        experiment_dir = self.experiments_path / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating {experiment_name} experiment")
        
        # Define strong comorbidity clusters with high co-occurrence rates
        comorbidity_clusters = {
            'metabolic_syndrome': {
                'core_diseases': ['CCSR_END009', 'CCSR_CIR007', 'CCSR_END001'],  # Diabetes, HTN, Obesity
                'associated': ['CCSR_CIR006', 'CCSR_END010', 'CCSR_CIR013'],    # CAD, Diabetes comp, CVD
                'co_occurrence_rate': 0.95
            },
            
            'heart_failure_complex': {
                'core_diseases': ['CCSR_CIR004', 'CCSR_CIR007', 'CCSR_CIR006'],  # HF, HTN, CAD
                'associated': ['CCSR_END009', 'CCSR_RES005', 'CCSR_GU001'],     # DM, COPD, Kidney
                'co_occurrence_rate': 0.90
            },
            
            'mental_health_complex': {
                'core_diseases': ['CCSR_MBD005', 'CCSR_MBD006', 'CCSR_MBD011'],  # Depression, anxiety, substance
                'associated': ['CCSR_NVS012', 'CCSR_INJ002', 'CCSR_FAC012'],    # Sleep, self-harm, social
                'co_occurrence_rate': 0.85
            },
            
            'cancer_cascade': {
                'core_diseases': ['CCSR_NEO019', 'CCSR_NEO038', 'CCSR_SYM010'],  # Cancer, mets, cachexia
                'associated': ['CCSR_BLD001', 'CCSR_MBD005', 'CCSR_FAC007'],    # Anemia, depression, palliative
                'co_occurrence_rate': 0.95
            }
        }
        
        amplified_data = []
        
        for cluster_name, cluster_info in comorbidity_clusters.items():
            core_diseases = cluster_info['core_diseases']
            associated_diseases = cluster_info['associated']
            co_occurrence_rate = cluster_info['co_occurrence_rate']
            
            # Create 200 patients per cluster
            for patient_num in range(200):
                patient_id = f"amplified_{cluster_name}_{patient_num:03d}"
                
                # Start with first core disease
                current_age = np.random.randint(45, 65)  # Middle-aged onset
                
                # Generate strongly correlated visits
                for visit_num in range(1, 6):  # 5 visits per patient
                    visit_diseases = []
                    
                    if visit_num == 1:
                        # First visit: introduce core disease
                        visit_diseases = [np.random.choice(core_diseases)]
                    
                    elif visit_num == 2:
                        # Second visit: high probability of adding another core disease
                        if np.random.random() < co_occurrence_rate:
                            remaining_core = [d for d in core_diseases if d not in visit_diseases]
                            if remaining_core:
                                visit_diseases.append(np.random.choice(remaining_core))
                    
                    else:
                        # Later visits: progressively add associated diseases
                        if np.random.random() < co_occurrence_rate:
                            # Continue core diseases
                            n_core = min(2, len(core_diseases))
                            visit_diseases.extend(np.random.choice(core_diseases, size=n_core, replace=False))
                            
                            # Add associated diseases with high probability
                            if np.random.random() < 0.8:
                                n_associated = np.random.randint(1, 3)
                                associated_sample = np.random.choice(associated_diseases, size=min(n_associated, len(associated_diseases)), replace=False)
                                visit_diseases.extend(associated_sample)
                    
                    # Remove duplicates and add to dataset
                    visit_diseases = list(set(visit_diseases))
                    for disease in visit_diseases:
                        amplified_data.append({
                            'subject_id': patient_id,
                            'visit_concept_orders': visit_num,
                            'concept_ids': disease,
                            'ages': f'AGE[{current_age}]',
                            'visit_dates': f'2020-{visit_num:02d}-01'
                        })
                    
                    current_age += 1  # Age by 1 year per visit
        
        # Convert to DataFrame and save
        amplified_df = pd.DataFrame(amplified_data)
        amplified_df.to_parquet(experiment_dir / "train_data_ccsr.parquet")
        
        # Save configuration
        config = {
            'experiment_name': experiment_name,
            'description': 'Strongly amplified comorbidity patterns',
            'comorbidity_clusters': comorbidity_clusters,
            'patients_per_cluster': 200,
            'total_patients': len(amplified_df['subject_id'].unique()),
            'total_visits': len(amplified_df),
            'created': datetime.now().isoformat()
        }
        
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Created {len(amplified_df)} visits with amplified comorbidity signals")
        return str(experiment_dir)
    
    # === EXPERIMENT 4: MINIMAL COMPLEXITY MAXIMUM PERFORMANCE ===
    
    def create_minimal_complexity_experiment(self) -> str:
        """
        Create minimal dataset focusing on only the most predictable patterns
        """
        experiment_name = "minimal_complexity"
        experiment_dir = self.experiments_path / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating {experiment_name} experiment")
        
        # Focus on just 5 most predictable disease relationships
        minimal_rules = {
            'rule_1_young_anxiety_depression': {
                'condition': 'age 25-35 AND anxiety',
                'prediction': 'depression in next visit',
                'sequence': ['CCSR_MBD006', 'CCSR_MBD005'],
                'ages': [28, 30],
                'probability': 0.98
            },
            
            'rule_2_diabetes_hypertension': {
                'condition': 'age 45-55 AND diabetes',
                'prediction': 'hypertension in next visit',
                'sequence': ['CCSR_END009', 'CCSR_CIR007'],
                'ages': [48, 50],
                'probability': 0.95
            },
            
            'rule_3_hypertension_cad': {
                'condition': 'age 55-65 AND hypertension',
                'prediction': 'coronary artery disease in next visit',
                'sequence': ['CCSR_CIR007', 'CCSR_CIR006'],
                'ages': [58, 61],
                'probability': 0.92
            },
            
            'rule_4_cad_mi': {
                'condition': 'age 60-70 AND CAD',
                'prediction': 'myocardial infarction in next visit', 
                'sequence': ['CCSR_CIR006', 'CCSR_CIR011'],
                'ages': [63, 65],
                'probability': 0.90
            },
            
            'rule_5_cancer_metastases': {
                'condition': 'age 60+ AND primary cancer',
                'prediction': 'metastases in next visit',
                'sequence': ['CCSR_NEO019', 'CCSR_NEO038'],
                'ages': [65, 66],
                'probability': 0.97
            }
        }
        
        minimal_data = []
        
        for rule_name, rule_info in minimal_rules.items():
            sequence = rule_info['sequence']
            ages = rule_info['ages']
            probability = rule_info['probability']
            
            # Create 300 patients per rule for strong signal
            for patient_num in range(300):
                patient_id = f"minimal_{rule_name}_{patient_num:03d}"
                
                # Create perfect 2-visit sequence
                for visit_idx, (disease, age) in enumerate(zip(sequence, ages)):
                    minimal_data.append({
                        'subject_id': patient_id,
                        'visit_concept_orders': visit_idx + 1,
                        'concept_ids': disease,
                        'ages': f'AGE[{age}]',
                        'visit_dates': f'2020-{visit_idx+1:02d}-01'
                    })
        
        # Convert to DataFrame and save
        minimal_df = pd.DataFrame(minimal_data)
        minimal_df.to_parquet(experiment_dir / "train_data_ccsr.parquet")
        
        # Save configuration
        config = {
            'experiment_name': experiment_name,
            'description': 'Minimal complexity with maximum predictability',
            'rules': minimal_rules,
            'patients_per_rule': 300,
            'total_patients': len(minimal_df['subject_id'].unique()),
            'total_visits': len(minimal_df),
            'created': datetime.now().isoformat()
        }
        
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Created {len(minimal_df)} visits with minimal complexity rules")
        return str(experiment_dir)
    
    # === MAIN EXPERIMENT RUNNER ===
    
    def run_all_experiments(self):
        """Run all reverse engineering experiments"""
        
        self.logger.info("="*80)
        self.logger.info("STARTING BEHRT REVERSE ENGINEERING EXPERIMENTS")
        self.logger.info("="*80)
        self.logger.info(f"Baseline APS: {self.baseline_metrics.get('aps', 'Unknown')}")
        self.logger.info(f"Target APS: 0.85-0.90")
        self.logger.info(f"Original BEHRT paper: 0.462-0.525 APS")
        self.logger.info("="*80)
        
        experiments_created = []
        
        try:
            # Run all experimental strategies
            exp1 = self.create_perfect_progressions_experiment()
            experiments_created.append(('perfect_progressions', exp1))
            
            exp2 = self.create_optimal_temporal_experiment()
            experiments_created.append(('optimal_temporal', exp2))
            
            exp3 = self.create_amplified_comorbidity_experiment()
            experiments_created.append(('amplified_comorbidity', exp3))
            
            exp4 = self.create_minimal_complexity_experiment()
            experiments_created.append(('minimal_complexity', exp4))
            
            # Create experiment summary
            summary = {
                'total_experiments': len(experiments_created),
                'baseline_aps': self.baseline_metrics.get('aps'),
                'target_aps': '0.85-0.90',
                'experiments': [
                    {
                        'name': name,
                        'path': path,
                        'created': datetime.now().isoformat()
                    }
                    for name, path in experiments_created
                ],
                'next_steps': [
                    'Train BEHRT model on each experimental dataset',
                    'Compare APS/ROC-AUC results against baseline',
                    'Identify which strategies achieve target performance',
                    'Analyze data characteristics that drive high performance'
                ]
            }
            
            with open(self.results_path / "experiment_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info("="*80)
            self.logger.info("✅ ALL EXPERIMENTS CREATED SUCCESSFULLY")
            self.logger.info("="*80)
            
            for name, path in experiments_created:
                self.logger.info(f"✓ {name}: {path}")
            
            self.logger.info("="*80)
            self.logger.info("NEXT STEPS:")
            self.logger.info("1. Train models: python train_experimental_models.py")
            self.logger.info("2. Analyze results: python analyze_experimental_results.py")
            self.logger.info("3. Find patterns that achieve 85-90% APS performance")
            self.logger.info("="*80)
            
            return experiments_created
            
        except Exception as e:
            self.logger.error(f"Experiment creation failed: {e}")
            raise
    
    def create_training_script(self):
        """Create script to train models on experimental data"""
        
        training_script = '''#!/usr/bin/env python3
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
    
    print("\\n" + "="*60)
    print("TRAINING SUMMARY:")
    for exp, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {exp}: {status}")
    print("="*60)

if __name__ == "__main__":
    main()
'''
        
        with open(self.base_path / "train_experimental_models.py", 'w') as f:
            f.write(training_script)
        
        self.logger.info("✓ Created training script: train_experimental_models.py")

def main():
    """Run the complete reverse engineering setup"""
    
    # Check if we're in the right environment
    if not Path("setup_info.json").exists():
        print("❌ Please run setup_reverse_engineering.py first!")
        return
    
    # Initialize and run experiments
    reverser = BEHRTReverseEngineering()
    experiments = reverser.run_all_experiments()
    reverser.create_training_script()
    
    print("\n🎯 EXPERIMENT CREATION COMPLETE!")
    print(f"📊 Created {len(experiments)} experimental datasets")
    print("📝 Next: Run python train_experimental_models.py")

if __name__ == "__main__":
    main()