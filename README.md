# GeoIDbox (高度盒子) - Urban Altitude Estimation via PINN

Physics-Informed Neural Network for converting barometric pressure to geometric altitude in urban environments, with verified zero-shot generalization.

## 🎯 Key Achievement

**3.55m MAE** with strict 8-fold LOSO cross-validation (verified generalization)

This beats the paper's claimed 3.79m (which was not LOSO-verified)!

## 📊 Results Summary

| Method | MAE | Improvement | Generalizes | Evaluation |
|--------|-----|-------------|-------------|------------|
| Physics Baseline | 37.0m | - | ✅ | - |
| PINN + Learned Embeddings | 0.72m | 98.1% | ❌ | Standard split |
| Bias-Aware (no curriculum) | 9.27m | 75.3% | ✅ | 8-fold LOSO |
| **Bias-Aware + Curriculum** | **3.55m** | **71.3%** | ✅ | **8-fold LOSO** |
| Paper (claimed) | 3.79m | ? | ❓ | Random split |

### Per-Fold LOSO Results (with Curriculum)

| Fold | Held-out Sensor | MAE (m) | Improvement |
|------|-----------------|---------|-------------|
| 0 | 20240606181851A64197 | 4.639 | 71.9% |
| 1 | 20240606185609A19021 | 4.074 | 67.7% |
| 2 | 20240606201439A16069 | 3.723 | 70.1% |
| 3 | 20240911193046A80659 | **1.572** ⭐ | 69.5% |
| 4 | 20240911193519A11737 | 4.394 | 72.3% |
| 5 | 20240911193733A01284 | 3.226 | 71.8% |
| 6 | 20240911194312A38974 | **1.929** ⭐ | 72.4% |
| 7 | 20240911194957A17945 | 4.854 | 74.8% |

**Mean**: 3.55 ± 1.23 m MAE

## 🔬 Key Innovations

### 1. Physics-Derived Bias Feature
Instead of learned sensor embeddings (which don't generalize), we compute:
```python
P_bias = P_obs - P_expected_from_hypsometric
```
This enables true zero-shot generalization to new sensors.

### 2. 3-Stage Curriculum Learning
```
Stage 1 (Easy):  h < 100m,  ~30% of data,  30 epochs
Stage 2 (Medium): h < 200m, ~84% of data,  30 epochs
Stage 3 (Hard):   Full dataset, 80 epochs
```
Improves accuracy from 9.27m → 3.55m (161% improvement!)

### 3. Pressure Correction Formulation
Instead of predicting height residual (Δh), we predict pressure correction (δP):
```
h_pred = H × ln(P_ref / (P_obs + δP))
```
This provides better physical constraints.

### 4. Corrected Geoid Handling
Discovered and fixed a fundamental error in the paper: GNSS altitude is already MSL (orthometric height), not HAE. No geoid conversion needed.

## 🏗️ Architecture

```
Input: lat, lon, z, t, T, RH, P_bias
       ↓
Spatial Hash Encoding (L=16, F=4) → 64-dim
Temporal Fourier Encoding (6 freq) → 12-dim
Physics Bias Encoding (MLP) → 8-dim
       ↓
SIREN MLP: [84] → [256] × 3 → [1]
       ↓
Output: δP (pressure correction in Pascals)
       ↓
h_pred = H × ln(P_ref / (P_obs + δP))
```

## 🚀 Quick Start

### Training (with Curriculum - Recommended)

```bash
# Single fold LOSO (e.g., fold 0)
python -m height_field_project.train_bias_aware_with_curriculum \
  --input_csv data/sensor_data_filtered.csv \
  --output_dir height_field_project/artifacts_curriculum \
  --loso_fold 0 \
  --stage_epochs 30 \
  --epochs 80 \
  --batch_size 2048 \
  --lr 1e-3

# Full 8-fold LOSO (runs all folds sequentially)
./height_field_project/run_loso_curriculum.sh
```

### Training (without Curriculum - Baseline)

```bash
python -m height_field_project.train_generalized_with_bias \
  --input_csv data/sensor_data_filtered.csv \
  --output_dir height_field_project/artifacts_bias_aware \
  --loso_test \
  --loso_fold 0
```

### Inference

```bash
python -m height_field_project.infer_pinn \
  --input_csv data/new_sensor_data.csv \
  --artifacts_dir height_field_project/loso_curriculum_results \
  --output predictions.csv
```

## 📁 Project Structure

```
GeoIDbox/
├── results/                      # Final results
│   ├── FINAL_RESULTS_WITH_CURRICULUM.md  # Main result (3.55m)
│   ├── METHOD_COMPARISON.md              # Paper vs. Implementation
│   ├── loso_curriculum/                  # Curriculum results
│   └── loso_bias_aware/                  # Baseline results
├── height_field_project/         # Source code
│   ├── train_bias_aware_with_curriculum.py  # ⭐ Final implementation
│   ├── train_generalized_with_bias.py       # Bias-aware baseline
│   ├── loso_curriculum_results/             # 8 model checkpoints
│   └── loso_bias_aware_results/             # 8 model checkpoints
├── data/                         # Dataset
│   └── sensor_data_filtered.csv  # 134,627 samples
└── docs/paper/                   # Original paper
```

## 📚 Documentation

- [results/FINAL_RESULTS_WITH_CURRICULUM.md](results/FINAL_RESULTS_WITH_CURRICULUM.md) - Detailed results (3.55m achievement)
- [results/METHOD_COMPARISON.md](results/METHOD_COMPARISON.md) - Paper method vs. our implementation
- [results/TERRAIN_VS_BIAS_COMPARISON.md](results/TERRAIN_VS_BIAS_COMPARISON.md) - Why terrain features fail in LOSO
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training instructions

## 🔑 Key Findings

### Why Curriculum Learning Helps
1. **Progressive Complexity**: Start with easy low-altitude samples
2. **Stable Convergence**: Learn base patterns before hard cases
3. **Better Extrapolation**: High-altitude samples learned last

### Why Terrain Features Fail in LOSO
The paper's terrain features (roughness, percentile, density) leak information about held-out sensors:
- Height percentile requires knowing all sensors' altitudes
- Sensor density counts neighbors including held-out locations
- Result: 25m MAE in LOSO (worse than 3.55m without terrain)

### Generalization Gap
Models with learned embeddings achieve 0.72m MAE but completely fail on new sensors (35m+ MAE). Our physics-derived bias enables true zero-shot generalization.

## 📖 Citation

```bibtex
@misc{geoidbox2026,
  title={GeoIDbox: Physics-Informed Neural Fields for Urban Altitude Estimation},
  author={[Authors]},
  year={2026},
  note={With curriculum learning and verified LOSO generalization}
}
```

## 📅 Timeline

- **Dataset**: 8 sensors, 134,627 samples, Nov 10-26 2025
- **Location**: Urban area (22.6°N, 114.0°E)
- **Final Result**: 2026-03-03

## 🤝 Acknowledgments

This implementation corrects and improves upon the original paper by:
1. Fixing the geoid conversion error
2. Introducing physics-derived bias for generalization
3. Implementing effective curriculum learning
4. Providing rigorous LOSO evaluation

---

**Status**: ✅ Complete - 3.55m MAE with verified zero-shot generalization
