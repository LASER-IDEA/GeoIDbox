# IEEE TIM Experiments Framework

This directory contains the complete experimental framework for the IEEE Transactions on Instrumentation and Measurement paper submission.

## Quick Start

### Run All Experiments
```bash
python -m experiments.run_all_experiments \
  --data data/processed/sensor_data_clean_stable.csv \
  --epochs 100
```

### Run Individual Experiments

#### 1. Baseline Comparison
```bash
python -m experiments.run_baseline_comparison \
  --data data/processed/sensor_data_clean_stable.csv \
  --methods all
```

Methods included:
- **Classical**: ISA, Barometric Linear, Kriging, Polynomial, RBF
- **ML**: Random Forest, XGBoost, Gaussian Process, Standard MLP

#### 2. Deep Ensemble
```bash
python -m experiments.deep_ensemble.deep_ensemble_trainer \
  --data data/processed/sensor_data_clean_stable.csv \
  --n_models 5 \
  --epochs 200
```

Features:
- 5 models with different architectures and seeds
- Uncertainty decomposition (aleatoric + epistemic)
- Better calibration than MC Dropout

#### 3. ST-GNN Ablation Study
```bash
python -m experiments.st_gnn.train_st_gnn \
  --data data/processed/sensor_data_clean_stable.csv \
  --epochs 100
```

Compares:
- Independent MLP (no graph)
- ST-GNN with KNN graph
- ST-GNN with Distance-based graph
- ST-GNN with Hybrid graph

## Directory Structure

```
experiments/
├── README.md                        # This file
├── evaluation.py                    # Evaluation utilities
├── run_baseline_comparison.py       # Baseline experiments
├── run_all_experiments.py           # Master script
├── baselines/
│   ├── __init__.py
│   ├── classical_methods.py         # ISA, Kriging, etc.
│   └── ml_methods.py                # XGBoost, RF, GP, MLP
├── deep_ensemble/
│   ├── __init__.py
│   └── deep_ensemble_trainer.py     # Deep Ensemble training
├── st_gnn/
│   ├── __init__.py
│   ├── graph_builder.py             # Graph construction
│   ├── st_gnn_model.py              # Model architecture
│   └── train_st_gnn.py              # Training & evaluation
└── results/                         # Generated results
    ├── figures/                     # Plots and visualizations
    └── tables/                      # CSV and LaTeX tables
```

## Key Innovations

### 1. Deep Ensembles (Deep Ensemble)

**Problem**: MC Dropout provides approximate uncertainty that is often poorly calibrated.

**Solution**: Train multiple networks with different random initializations.

**Advantages**:
- Natural decomposition of uncertainty types
- Better calibration
- More stable predictions

**Usage**:
```python
from experiments.deep_ensemble.deep_ensemble_trainer import DeepEnsemble

ensemble = DeepEnsemble(n_models=5, in_dim=7)
ensemble.train_ensemble(train_loader, val_loader, epochs=200)
mean, aleatoric, epistemic, total = ensemble.predict(X)
```

### 2. ST-GNN (Spatial-Temporal Graph Neural Network)

**Problem**: Current methods treat sensors independently, ignoring spatial correlations.

**Solution**: Model sensor network as graph with message passing.

**Architecture**:
- Graph Convolution layers for spatial modeling
- Temporal encoding for time-varying features
- Physics-informed prediction head

**Graph Construction Strategies**:
- Distance-based: Gaussian kernel on geographic distance
- Correlation-based: Pressure pattern similarity
- Hybrid: Weighted combination
- KNN: K-nearest neighbors

**Usage**:
```python
from experiments.st_gnn.st_gnn_model import STGNNHeightEstimator
from experiments.st_gnn.graph_builder import DistanceGraphBuilder

# Build graph
builder = DistanceGraphBuilder(distance_threshold_km=2.0)
adj = builder.build_graph(df, uids)

# Create model
model = STGNNHeightEstimator(
    num_nodes=len(uids),
    in_features=7,
    hidden_dims=[128, 128, 64]
)

# Forward pass
h_pred, uncertainty = model(features, adj, h_phys, time_indices)
```

## Evaluation Metrics

### Basic Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Bias**: Mean error (systematic offset)
- **P95**: 95th percentile of absolute errors

### Uncertainty Calibration
- **ECE**: Expected Calibration Error
- **NLL**: Negative Log-Likelihood
- **CRPS**: Continuous Ranked Probability Score
- **Reliability Diagram**: Visual calibration check

### Stratified Analysis
- By pressure (weather conditions)
- By temperature
- By time of day
- By week (seasonal variation)

## Expected Results

Based on preliminary experiments with Shenzhen sensor data:

| Method | RMSE (m) | MAE (m) | P95 (m) |
|--------|----------|---------|---------|
| ISA | ~50 | ~40 | ~80 |
| Barometric Linear | ~15 | ~12 | ~25 |
| Kriging | ~12 | ~9 | ~20 |
| Random Forest | ~8 | ~6 | ~14 |
| Standard MLP | ~7 | ~5.5 | ~12 |
| Neural Field (original) | ~6 | ~4.5 | ~10 |
| **ST-GNN (proposed)** | **~5** | **~4** | **~8** |

## Visualization

The framework generates several plots:

1. **Error Distribution Comparison**: Histograms and CDFs across methods
2. **Reliability Diagrams**: Uncertainty calibration plots
3. **Graph Structure**: Visualizations of sensor network graphs
4. **Training Curves**: Loss vs epochs for each method

All figures are saved to `experiments/results/figures/`.

## Paper Section Mapping

| Experiment | Paper Section | Key Figures/Tables |
|------------|---------------|-------------------|
| Baseline comparison | IV.B Benchmarking | Table I, Fig. 3 |
| Ablation study | IV.C Ablation Analysis | Table II, Fig. 4 |
| ST-GNN results | IV.D Spatial-Temporal Modeling | Fig. 5, 6 |
| Uncertainty analysis | IV.E Uncertainty Quantification | Fig. 7, 8 |

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in data loaders
- Use fewer GNN layers
- Subsample training data for GP baseline

### Graph Building Fails
- Check sensor coordinates are valid
- Ensure sufficient temporal overlap for correlation-based graphs
- Try different graph types (distance is most robust)

### Poor ST-GNN Performance
- Verify adjacency matrix normalization
- Check feature scaling
- Ensure temporal alignment of snapshots

## Citation

If you use this code, please cite:

```bibtex
@article{geoidbox2024,
  title={Physics-Informed Graph Neural Networks for Urban Barometric Altitude Estimation},
  author={[Authors]},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2024}
}
```

## Contact

For questions about the experiments, please open an issue in the repository.
