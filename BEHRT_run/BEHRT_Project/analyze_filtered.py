#!/usr/bin/env python3
"""
Comprehensive BEHRT Analysis with XXX000 Filtering
==================================================

This script runs comprehensive analysis on the original model/data but 
filters out XXX000 codes from final metrics to show clean performance.
"""

import pandas as pd
import json
import numpy as np
from collections import Counter
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_and_filter_results():
    """Load comprehensive results and filter out XXX000"""
    print("🔬 BEHRT COMPREHENSIVE ANALYSIS (XXX000 FILTERED)")
    print("=" * 60)
    
    # Load original comprehensive results
    results_file = "utils/comprehensive_disease_analysis/comprehensive_disease_analysis_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("📊 ORIGINAL vs FILTERED METRICS COMPARISON")
    print("=" * 60)
    
    # Original metrics
    overall = results['overall_metrics']
    print("ORIGINAL METRICS (including XXX000):")
    print(f"  Weighted F1: {overall['weighted_f1']:.4f} ({overall['weighted_f1']*100:.2f}%)")
    print(f"  Macro F1:    {overall['macro_f1']:.4f} ({overall['macro_f1']*100:.2f}%)")
    print(f"  APS:         {overall['sample_wise_aps']:.4f} ({overall['sample_wise_aps']*100:.2f}%)")
    print(f"  ROC-AUC:     {overall['sample_wise_auc']:.4f} ({overall['sample_wise_auc']*100:.2f}%)")
    
    # Filter out XXX000 from disease results
    diseases_list = results['all_individual_diseases']
    filtered_diseases = [d for d in diseases_list if 'XXX000' not in d['disease_code']]
    
    # Create dictionary format for easier processing
    diseases_dict = {d['disease_code']: d for d in diseases_list}
    filtered_diseases_dict = {d['disease_code']: d for d in filtered_diseases}
    
    print(f"\nDISEASE COUNT:")
    print(f"  Total diseases: {len(diseases_list)}")
    print(f"  Real diseases (filtered): {len(filtered_diseases)}")
    print(f"  XXX000 removed: {len(diseases_list) - len(filtered_diseases)}")
    
    # Calculate filtered metrics
    filtered_f1_scores = [d['f1_score'] for d in filtered_diseases]
    filtered_precision_scores = [d['precision'] for d in filtered_diseases]
    filtered_recall_scores = [d['recall'] for d in filtered_diseases]
    filtered_support = [d['support'] for d in filtered_diseases]
    
    # Calculate weighted averages (excluding XXX000)
    total_support = sum(filtered_support)
    weighted_f1_filtered = sum(f1 * support for f1, support in zip(filtered_f1_scores, filtered_support)) / total_support
    weighted_precision_filtered = sum(p * support for p, support in zip(filtered_precision_scores, filtered_support)) / total_support
    weighted_recall_filtered = sum(r * support for r, support in zip(filtered_recall_scores, filtered_support)) / total_support
    
    # Macro averages (excluding XXX000)
    macro_f1_filtered = np.mean(filtered_f1_scores)
    macro_precision_filtered = np.mean(filtered_precision_scores)
    macro_recall_filtered = np.mean(filtered_recall_scores)
    
    print(f"\nFILTERED METRICS (excluding XXX000):")
    print(f"  Weighted F1: {weighted_f1_filtered:.4f} ({weighted_f1_filtered*100:.2f}%)")
    print(f"  Weighted Precision: {weighted_precision_filtered:.4f} ({weighted_precision_filtered*100:.2f}%)")
    print(f"  Weighted Recall: {weighted_recall_filtered:.4f} ({weighted_recall_filtered*100:.2f}%)")
    print(f"  Macro F1:    {macro_f1_filtered:.4f} ({macro_f1_filtered*100:.2f}%)")
    print(f"  Macro Precision: {macro_precision_filtered:.4f} ({macro_precision_filtered*100:.2f}%)")
    print(f"  Macro Recall: {macro_recall_filtered:.4f} ({macro_recall_filtered*100:.2f}%)")
    
    # Top performing real diseases
    print(f"\n🏆 TOP 10 REAL DISEASE PREDICTIONS (XXX000 excluded):")
    print("=" * 60)
    
    # Sort by F1 score
    top_diseases = sorted(
        filtered_diseases,
        key=lambda x: x['f1_score'],
        reverse=True
    )[:10]
    
    for i, disease in enumerate(top_diseases, 1):
        disease_name = disease['disease_code'].replace('CCSR_', '')
        print(f"{i:2}. {disease_name}:")
        print(f"     Precision: {disease['precision']:.3f} | Recall: {disease['recall']:.3f} | F1: {disease['f1_score']:.3f}")
        print(f"     Support: {disease['support']:,} patients")
        print()
    
    # Performance distribution analysis
    print(f"📈 PERFORMANCE DISTRIBUTION (Real diseases only):")
    print("=" * 60)
    
    f1_ranges = {
        'Excellent (F1 > 0.6)': sum(1 for f1 in filtered_f1_scores if f1 > 0.6),
        'Good (0.4 < F1 <= 0.6)': sum(1 for f1 in filtered_f1_scores if 0.4 < f1 <= 0.6),
        'Fair (0.2 < F1 <= 0.4)': sum(1 for f1 in filtered_f1_scores if 0.2 < f1 <= 0.4),
        'Poor (0.0 < F1 <= 0.2)': sum(1 for f1 in filtered_f1_scores if 0.0 < f1 <= 0.2),
        'No prediction (F1 = 0)': sum(1 for f1 in filtered_f1_scores if f1 == 0.0)
    }
    
    for range_name, count in f1_ranges.items():
        pct = (count / len(filtered_f1_scores)) * 100
        print(f"  {range_name}: {count} diseases ({pct:.1f}%)")
    
    # Category analysis (excluding XXX categories)
    categories = results.get('category_analysis', {})
    filtered_categories = {k: v for k, v in categories.items() if 'XXX' not in k and 'Unknown' not in k}
    
    if filtered_categories:
        print(f"\n🏥 TOP DISEASE CATEGORIES (Real categories only):")
        print("=" * 60)
        
        top_categories = sorted(
            filtered_categories.items(),
            key=lambda x: x[1].get('f1', 0),
            reverse=True
        )[:5]
        
        for category, metrics in top_categories:
            print(f"  {category}:")
            print(f"     F1: {metrics.get('f1', 0):.3f} | Classes: {metrics.get('num_classes', 'N/A')}")
            print(f"     Precision: {metrics.get('precision', 0):.3f} | Recall: {metrics.get('recall', 0):.3f}")
        
    # Key insights
    print(f"\n💡 KEY INSIGHTS (Clean Analysis):")
    print("=" * 60)
    
    excellent_diseases = [d['disease_code'] for d in filtered_diseases if d['f1_score'] > 0.6]
    good_diseases = [d['disease_code'] for d in filtered_diseases if 0.4 < d['f1_score'] <= 0.6]
    
    print(f"✓ {len(excellent_diseases)} diseases with excellent performance (F1 > 60%)")
    print(f"✓ {len(good_diseases)} diseases with good performance (F1 40-60%)")
    print(f"✓ Average F1 for real diseases: {macro_f1_filtered:.3f} ({macro_f1_filtered*100:.1f}%)")
    print(f"✓ Best performing disease: {top_diseases[0]['disease_code']} with F1={top_diseases[0]['f1_score']:.3f}")
    
    # Calculate improvement from removing XXX000
    xxx_info = next((d for d in diseases_list if 'XXX000' in d['disease_code']), None)
    if xxx_info:
        print(f"\n📊 IMPACT OF XXX000 REMOVAL:")
        print("=" * 60)
        print(f"  XXX000 F1: {xxx_info['f1_score']:.3f}")
        print(f"  XXX000 Support: {xxx_info['support']:,} cases")
        print(f"  XXX000 was skewing results due to high frequency")
        print(f"  Removing it reveals true clinical prediction quality")
    
    return {
        'filtered_diseases': filtered_diseases_dict,
        'filtered_metrics': {
            'weighted_f1': weighted_f1_filtered,
            'weighted_precision': weighted_precision_filtered, 
            'weighted_recall': weighted_recall_filtered,
            'macro_f1': macro_f1_filtered,
            'macro_precision': macro_precision_filtered,
            'macro_recall': macro_recall_filtered
        },
        'top_diseases': top_diseases,
        'performance_distribution': f1_ranges
    }

def main():
    """Main analysis function"""
    filtered_results = load_and_filter_results()
    
    print(f"\n✅ FILTERED ANALYSIS COMPLETE!")
    print("=" * 60)
    print("This analysis represents the TRUE clinical prediction capability")
    print("of the BEHRT model without XXX000 generic code contamination.")
    
    # For training the full model, recommend using cleaned data
    print(f"\n🚀 RECOMMENDATION FOR FULL MODEL:")
    print("=" * 60)
    print("✓ Use cleaned datasets (*_clean.parquet) for full model training")
    print("✓ Use clean vocabulary (vocab_ccsr_clean.pkl)")
    print("✓ This will provide cleaner, more interpretable results from the start")
    print("✓ Expected performance improvement due to focused learning")

if __name__ == "__main__":
    main()