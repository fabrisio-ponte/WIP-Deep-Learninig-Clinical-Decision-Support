# Comprehensive BEHRT Disease Analysis

## Overview

This tool provides comprehensive analysis of BEHRT model performance at both the overall and individual disease level, with a focus on clinical interpretability and structured reporting.

## Features

- **Overall Performance Metrics**: Weighted and macro-averaged precision, recall, F1, and accuracy
- **Disease Category Analysis**: Performance breakdown by medical specialties (Circulatory, Respiratory, etc.)
- **Individual Disease Performance**: Detailed metrics for each disease prediction
- **Clinical Report Format**: Structured output matching clinical evaluation standards
- **Detailed JSON Export**: Complete results saved for further analysis

## Usage

### Basic Usage

```bash
python comprehensive_disease_analysis.py
```

This will run with default configuration, looking for:
- Model at `../data/models/quick/bert_model.bin`
- Vocabulary at `../data/processed/vocab_ccsr.pkl`
- Test data at `../data/processed/test_data_ccsr.parquet`

### Custom Configuration

1. Copy the template configuration:
   ```bash
   cp analysis_config_template.json analysis_config.json
   ```

2. Edit `analysis_config.json` to point to your data paths:
   ```json
   {
     "model_path": "path/to/your/trained_model.bin",
     "vocab_path": "path/to/your/vocab.pkl",
     "test_data_path": "path/to/your/test_data.parquet",
     "batch_size": 64,
     "device": "cuda:0",
     "max_len_seq": 100,
     "max_age": 110
   }
   ```

3. Run analysis with your config:
   ```bash
   python comprehensive_disease_analysis.py --config analysis_config.json
   ```

## Expected Output

### Console Report
The analysis generates a structured report with:
- Overall weighted/macro performance metrics
- Top performing disease categories
- Best individual disease predictions
- Token-level accuracy statistics

### JSON Results
Detailed results are saved to `comprehensive_disease_analysis_results.json` containing:
- All calculated metrics
- Per-category performance breakdown
- Complete individual disease results
- Configuration used for reproducibility

## Requirements

- PyTorch
- scikit-learn
- pandas
- numpy
- BEHRT framework (already included in parent project)

## Understanding the Output

### Weighted vs Macro Metrics
- **Weighted**: Averages weighted by support (number of positive cases per disease)
- **Macro**: Simple average across all diseases (treats rare and common diseases equally)

### Disease Categories
Based on Clinical Classifications Software Refined (CCSR) categories:
- CIR: Circulatory system
- RES: Respiratory system  
- DIG: Digestive system
- MBD: Mental and behavioral disorders
- And 16 other major categories

### Top Performing Diseases
Individual diseases ranked by F1 score, showing:
- Precision (P): When model predicts positive, how often correct
- Recall (R): Of all actual positives, how many found
- F1: Harmonic mean of precision and recall

This analysis helps evaluate BEHRT's clinical utility and identify which medical conditions the model predicts most accurately.