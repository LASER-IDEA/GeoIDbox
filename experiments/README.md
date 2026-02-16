# Experiments Directory

This directory contains code and results for the IEEE TIM paper experiments.

## Structure

```
experiments/
├── README.md                           # This file
├── evaluation.py                       # Core evaluation functions
├── run_baseline_comparison.py          # Baseline comparison experiments
├── run_quick_improvements.py           # Quick improvement experiments
├── run_all_experiments.py              # Master experiment runner
├── reports/                            # Experiment reports
│   ├── MAML_RESULTS_SUMMARY.md
│   ├── PHYSICS_INFORMED_OPERATORS_REPORT.md
│   └── PINO_2D_RESULTS.md
├── ablation/                           # Ablation study results
├── baselines/                          # Baseline method results
├── deep_ensemble/                      # Deep ensemble results
├── maml_v2/                            # MAML training outputs
├── pino/                               # PINO training outputs
└── results/                            # Main experimental results
```

## Core Files

### evaluation.py
Main evaluation functions including:
- LOSO (Leave-One-Sensor-Out) evaluation
- Metrics computation (MAE, RMSE, etc.)
- Result aggregation and reporting

### run_baseline_comparison.py
Comparison experiments with baseline methods:
- Linear regression
- Random Forest
- Neural network baselines

### run_quick_improvements.py
Quick experiments for hyperparameter tuning and architecture variations.

### run_all_experiments.py
Master script to run all experiments in sequence.

## Results

The `results/` directory contains:
- `curriculum_history/` - Training history for curriculum learning
- `fold_*/` - Per-fold LOSO results
- `predictions.csv` - Model predictions
- `model_performance.json` - Performance metrics

## Usage

```bash
# Run baseline comparison
python experiments/run_baseline_comparison.py

# Run all experiments
python experiments/run_all_experiments.py

# Evaluate a specific model
python experiments/evaluation.py --model_path path/to/model.pt
```

## Paper-Related Code

The core paper experiments are in the repository root:
- `run_advanced_improvements.py` - Main training script (produces 3.79m MAE)
- `run_final_pipeline.py` - Complete data processing pipeline
- `run_curriculum_with_history.py` - Curriculum learning with history tracking
