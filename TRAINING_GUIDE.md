# Training Guide for GeoIDbox PINN

Complete guide for training Physics-Informed Neural Networks for urban altitude estimation with verified zero-shot generalization.

## Overview

This guide covers training the **Bias-Aware Generalized PINN** with **Curriculum Learning**, which achieves **3.55m MAE** with verified generalization (8-fold LOSO).

## Key Results Reference

| Configuration | MAE | Improvement | Training Time |
|--------------|-----|-------------|---------------|
| Physics Baseline | 37.0m | - | - |
| Bias-Aware (no curriculum) | 9.27m | 75.3% | ~15 min |
| **Bias-Aware + Curriculum** | **3.55m** | **71.3%** | **~20 min** |

## Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
conda activate graphmamba

# Verify GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### 2. Recommended Training (with Curriculum)

#### Single Fold (Quick Test)

```bash
python -m height_field_project.train_bias_aware_with_curriculum \
  --input_csv data/sensor_data_filtered.csv \
  --output_dir height_field_project/artifacts_curriculum \
  --loso_fold 0 \
  --stage_epochs 30 \
  --epochs 80 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --restart_epochs 30 \
  --patience 25 \
  --seed 42
```

Expected result: ~4-5m MAE for Fold 0

#### Full 8-Fold LOSO (Rigorous Evaluation)

```bash
# Run all 8 folds sequentially
./height_field_project/run_loso_curriculum.sh

# Or run individually for each fold
for fold in {0..7}; do
  python -m height_field_project.train_bias_aware_with_curriculum \
    --input_csv data/sensor_data_filtered.csv \
    --output_dir height_field_project/loso_curriculum_results \
    --loso_fold $fold \
    --stage_epochs 30 \
    --epochs 80 \
    --batch_size 2048 \
    --lr 1e-3 \
    --seed 42
done
```

Expected result: 3.55 ± 1.23 m MAE across all folds

### 3. Baseline Training (without Curriculum)

For comparison purposes:

```bash
python -m height_field_project.train_generalized_with_bias \
  --input_csv data/sensor_data_filtered.csv \
  --output_dir height_field_project/loso_bias_aware_results \
  --loso_test \
  --loso_fold 0 \
  --epochs 80 \
  --batch_size 2048 \
  --lr 1e-3 \
  --seed 42
```

Expected result: ~8-9m MAE (vs 4-5m with curriculum)

## Architecture Details

### Bias-Aware PINN with Curriculum

```
Input Features:
  - Spatial: lat, lon (normalized)
  - Altitude: z (meters)
  - Temporal: t (Unix timestamp)
  - Environmental: T (°C), RH (%)
  - Physics Bias: P_bias = P_obs - P_expected

Encodings:
  - Hash Encoding: L=16 levels, F=4 features → 64-dim
  - Temporal Fourier: 6 frequencies → 12-dim
  - Bias Encoding: MLP → 8-dim

MLP Architecture:
  - Input: 64 + 1 + 12 + 1 + 1 + 8 = 87-dim
  - Hidden: [256] × 3 layers
  - Activation: SIREN (sinusoidal)
  - Output: 1 (δP in Pascals)

Total Parameters: ~1.4M
```

### Curriculum Learning Strategy

```
Stage 1 (Easy) - 30 epochs:
  Condition: h < 100m
  Coverage: ~30% of training data
  Purpose: Learn basic spatial patterns

Stage 2 (Medium) - 30 epochs:
  Condition: h < 200m
  Coverage: ~84% of training data
  Purpose: Add moderate extrapolation

Stage 3 (Hard) - 80 epochs:
  Condition: Full dataset
  Coverage: 100% of training data
  Purpose: Master high-altitude cases
```

## Training Parameters

### Key Hyperparameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--stage_epochs` | Epochs per curriculum stage | 30 |
| `--epochs` | Total epochs (Stage 3) | 80 |
| `--batch_size` | Batch size | 2048 |
| `--lr` | Learning rate | 1e-3 |
| `--weight_decay` | Weight decay | 1e-5 |
| `--restart_epochs` | Cosine annealing restart | 30 |
| `--patience` | Early stopping patience | 25 |
| `--hash_levels` | Hash encoding levels | 16 |
| `--hash_features` | Features per level | 4 |
| `--hidden_dim` | MLP hidden dimension | 256 |
| `--n_hidden_layers` | Number of hidden layers | 3 |
| `--temporal_freqs` | Temporal Fourier frequencies | 6 |
| `--bias_dim` | Bias encoding dimension | 8 |

### Loss Function

```python
# Data fidelity loss
P_corrected = P_obs + δP
h_pred = H × ln(P_ref / P_corrected)
L_data = MAE(h_pred, h_GNSS)

# No hydrostatic constraint (not needed with pressure formulation)
L_total = L_data
```

## Monitoring Training

### Expected Training Progress

```
Stage 1 (Easy, h<100m):
  Epoch 010 | Train: 32.28 | Val MAE: 34.10m
  Epoch 020 | Train: 29.40 | Val MAE: 31.35m
  Epoch 030 | Train: 26.80 | Val MAE: 28.87m

Stage 2 (Medium, h<200m):
  Epoch 010 | Train: 21.37 | Val MAE: 22.19m
  Epoch 020 | Train: 16.62 | Val MAE: 17.49m
  Epoch 030 | Train: 13.15 | Val MAE: 13.99m

Stage 3 (Hard, full dataset):
  Epoch 010 | Train: 10.74 | Val MAE: 10.44m | Improvement: 71.9%
  Epoch 020 | Train: 9.14 | Val MAE: 8.96m | Improvement: 75.9%
  ...
  Epoch 080 | Train: 3.44 | Val MAE: 3.36m | Improvement: 91.0%
```

### Key Metrics to Watch

1. **Val MAE**: Should decrease steadily through all stages
2. **Improvement %**: Should reach 70-90% by end of Stage 3
3. **Train/Val gap**: Small gap indicates good generalization

## Evaluation

### LOSO Cross-Validation

```bash
# Aggregate results from all folds
python3 << 'EOF'
import pandas as pd
import glob

results = []
for fold in range(8):
    log_file = f'height_field_project/loso_curriculum_results/fold_{fold}.log'
    # Parse MAE from log
    # ... (see results/loso_curriculum/loso_summary.csv)

df = pd.DataFrame(results)
print(f"Mean MAE: {df['mae'].mean():.3f} ± {df['mae'].std():.3f} m")
EOF
```

Expected output:
```
Mean MAE: 3.551 ± 1.228 m
Mean Improvement: 71.3%
```

### Inference on New Data

```python
import torch
import pandas as pd
from height_field_project.train_generalized_with_bias import (
    BiasAwarePINN, compute_sensor_bias
)
from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.physics_baseline import compute_physics_baseline

# Load model
checkpoint = torch.load(
    'height_field_project/loso_curriculum_results/model_curriculum_fold3.pt',
    map_location='cpu'
)
base_model = GeneralizedPressureCorrectionPINN(
    hash_levels=16, hash_features=4, hidden_dim=256,
    n_hidden_layers=3, temporal_freqs=6, use_siren=True
)
model = BiasAwarePINN(base_model, bias_dim=8)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare data
df = pd.read_csv('new_sensor_data.csv')
df, phys_params = compute_physics_baseline(df, p_ref=None)
df = compute_sensor_bias(df, phys_params.p_ref)

# Predict
with torch.no_grad():
    # ... (prepare tensors)
    delta_p = model(lat, lon, z, t, temp, humidity, bias)
    # Convert to height via hypsometric equation
```

## Troubleshooting

### High Validation Error

**Symptom**: Val MAE > 10m after Stage 3

**Solutions**:
1. Increase `--stage_epochs` to 40-50
2. Increase `--epochs` to 100-150
3. Reduce learning rate to 5e-4
4. Check data quality (outliers, missing values)

### Overfitting

**Symptom**: Train MAE << Val MAE

**Solutions**:
1. Increase `--weight_decay` to 1e-4
2. Reduce model capacity (fewer layers/hidden dims)
3. Add dropout (not currently implemented)

### Slow Convergence

**Symptom**: Little improvement after 50 epochs

**Solutions**:
1. Check learning rate schedule (cosine annealing)
2. Verify batch size is appropriate (2048 recommended)
3. Ensure curriculum stages are properly defined

### Out of Memory

**Solutions**:
1. Reduce `--batch_size` to 1024
2. Reduce `--hash_features` to 2
3. Reduce `--hidden_dim` to 128

## Advanced Topics

### Ablation Studies

To understand contribution of each component:

```bash
# Without curriculum
python -m height_field_project.train_generalized_with_bias ...

# With curriculum (this guide)
python -m height_field_project.train_bias_aware_with_curriculum ...

# With terrain features (not recommended for LOSO)
python -m height_field_project.train_pinn_with_terrain ...
```

### Multi-GPU Training

Currently not implemented for curriculum version, but can be added by wrapping model with `nn.DataParallel`.

### Hyperparameter Tuning

Key parameters to tune:
1. `--stage_epochs`: 20-50
2. `--epochs`: 60-150
3. `--lr`: 5e-4 to 2e-3
4. `--hidden_dim`: 128-512

## Expected File Outputs

After training fold 0:

```
height_field_project/loso_curriculum_results/
├── model_curriculum_fold0.pt       # Model checkpoint (5.5M)
├── fold_0.log                       # Training log
└── (folds 1-7 after full run)

results/loso_curriculum/
├── loso_summary.csv                 # Per-fold results
└── loso_summary.json                # Aggregate statistics
```

## References

- Main result: [results/FINAL_RESULTS_WITH_CURRICULUM.md](results/FINAL_RESULTS_WITH_CURRICULUM.md)
- Method comparison: [results/METHOD_COMPARISON.md](results/METHOD_COMPARISON.md)
- Original paper: `docs/paper/sections/method.tex`

## Summary

**Best Configuration**:
- Curriculum learning: ✅ Essential (9.27m → 3.55m)
- Physics-derived bias: ✅ Enables generalization
- Pressure correction: ✅ Better than height residual
- LOSO evaluation: ✅ Verified zero-shot capability

**Result**: 3.55m MAE with verified generalization (beats paper's 3.79m claim!)
