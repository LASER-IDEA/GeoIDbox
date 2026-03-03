# File Organization

## Directory Structure

```
GeoIDbox/
├── results/                          # FINAL RESULTS
│   ├── FINAL_RESULTS_WITH_CURRICULUM.md    # ⭐ Main result: 3.55m MAE
│   ├── FINAL_SUMMARY.md                    # Overall project summary
│   ├── METHOD_COMPARISON.md                # Paper vs. Implementation
│   ├── TERRAIN_VS_BIAS_COMPARISON.md       # Why terrain failed
│   ├── loso_summary.csv/json               # Copy of curriculum results
│   ├── loso_bias_aware/                    # Bias-aware results (no curriculum)
│   │   ├── RESULTS_SUMMARY.md
│   │   ├── loso_summary.csv/json
│   │   └── fold_*.log (8 files)
│   └── loso_curriculum/                    # Curriculum results (3.55m MAE)
│       ├── FINAL_RESULTS_WITH_CURRICULUM.md
│       ├── loso_summary.csv
│       └── loso_summary.json
│
├── height_field_project/             # SOURCE CODE
│   ├── train_bias_aware_with_curriculum.py  # ⭐ Final implementation
│   ├── train_generalized_with_bias.py       # Bias-aware implementation
│   ├── train_pinn_with_terrain.py           # Terrain-aware (paper method)
│   ├── neural_field_pinn_generalized.py     # Base architecture
│   ├── loso_curriculum_results/             # 8 model checkpoints (3.55m)
│   │   ├── model_curriculum_fold*.pt (8 files, 5.5M each)
│   │   ├── loso_summary.csv/json
│   │   └── fold_*.log (8 files)
│   ├── loso_bias_aware_results/             # 8 model checkpoints (9.27m)
│   │   └── model_bias_aware_fold*.pt (8 files)
│   └── (other implementation files)
│
├── test/                             # TEMPORARY FILES
│   ├── artifacts_archive/            # Old model checkpoints
│   ├── logs_archive/                 # Training logs
│   ├── scripts/                      # Shell scripts
│   ├── temp_outputs/                 # Intermediate outputs
│   └── usepx4/                       # PX4 test code
│
├── data/                             # DATA
│   ├── sensor_data_filtered.csv      # Main dataset (134,627 samples)
│   ├── dataprocess/                  # Data preprocessing
│   ├── era5_corrected/               # ERA5 weather data
│   ├── geoids/                       # Geoid models
│   └── rawdata/                      # Raw sensor data
│
└── docs/paper/                       # PAPER
    └── sections/
        ├── method.tex                # Original paper method
        ├── experiment.tex
        └── ...
```

## Key Files

### Results (Final)

| File | Description |
|------|-------------|
| `results/FINAL_RESULTS_WITH_CURRICULUM.md` | **Main result**: 3.55m MAE with curriculum |
| `results/loso_curriculum/loso_summary.csv` | Per-fold results (8 folds) |
| `results/loso_bias_aware/RESULTS_SUMMARY.md` | No-curriculum baseline (9.27m) |

### Models (Checkpoints)

| Location | MAE | Files |
|----------|-----|-------|
| `loso_curriculum_results/` | **3.55m** | 8 × 5.5MB checkpoints |
| `loso_bias_aware_results/` | 9.27m | 8 × 5.5MB checkpoints |

### Source Code

| File | Description |
|------|-------------|
| `train_bias_aware_with_curriculum.py` | **Final implementation** with curriculum |
| `train_generalized_with_bias.py` | Bias-aware without curriculum |
| `neural_field_pinn_generalized.py` | Base model architecture |

## Temporary Files (test/)

Moved to `test/` folder:
- Old training logs
- Intermediate model checkpoints
- Shell scripts
- Failed experiment outputs (terrain model)

## How to Use

### Load Best Model
```python
import torch
checkpoint = torch.load(
    'height_field_project/loso_curriculum_results/model_curriculum_fold3.pt',
    map_location='cpu'
)
# Fold 3 achieved 1.57m MAE (best fold)
```

### View Results
```bash
# Curriculum results (3.55m MAE)
cat results/loso_curriculum/loso_summary.csv

# Comparison with paper
cat results/FINAL_RESULTS_WITH_CURRICULUM.md
```

## Summary

- **Total model checkpoints**: 16 (8 curriculum + 8 bias-aware)
- **Total size**: ~176 MB
- **Final result**: 3.55m MAE (beats paper's 3.79m claim)
- **Evaluation**: 8-fold strict LOSO (verified generalization)
