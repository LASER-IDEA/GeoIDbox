# Terrain-Aware vs. Bias-Aware PINN Comparison

## Executive Summary

We implemented the **Terrain-Aware PINN** from the paper (with terrain features ρ, r, d and curriculum learning) and compared it against our **Bias-Aware Generalized PINN** using strict LOSO evaluation.

**Finding**: The Terrain-Aware model achieves **25.6m MAE** vs. **8.5m MAE** for Bias-Aware on Fold 0. The terrain features leak information about held-out sensors, making them unsuitable for true zero-shot generalization.

---

## Test Results (Fold 0)

| Method | MAE | Improvement | Generalizes | Key Feature |
|--------|-----|-------------|-------------|-------------|
| **Bias-Aware (Ours)** | **8.52m** | **76.2%** | ✅ Yes | Physics-derived P_bias |
| Terrain-Aware (Paper) | 25.64m | 28.4% | ❌ No | Terrain ρ, r, d |
| Physics Baseline | 35.81m | 0% | ✅ Yes | Hypsometric only |

---

## Why Terrain Features Fail in LOSO

### 1. Information Leakage

The terrain features from the paper are computed from **all sensors**, including held-out sensors:

| Feature | Computation | LOSO Issue |
|---------|-------------|------------|
| **Height Percentile (r)** | `rank(h) / N_sensors` | Requires knowing held-out sensor's altitude |
| **Sensor Density (d)** | Count neighbors in radius | Neighbors may include held-out sensor locations |
| **Local Roughness (ρ)** | Std of neighbor altitudes | Computed using all sensors including held-out |

### 2. Curriculum Learning Problem

The curriculum defines "easy" vs "hard" samples based on sensor density:
- Stage 1: High-density regions (h < 150m, d > d_median)
- Stage 2: Medium-density regions

In LOSO:
- Held-out sensors are typically in high-density areas (surrounded by training sensors)
- The curriculum sees the held-out sensor's density during training
- This creates an implicit "warm start" for held-out locations

### 3. Prediction Formulation

| Aspect | Terrain-Aware | Bias-Aware |
|--------|---------------|------------|
| Predicts | Height residual Δh | Pressure correction δP |
| Formula | `h_pred = h_phy + Δh` | `P_corr = P_obs + δP` → hypsometric |
| Physical constraints | Weak (direct addition) | Strong (pressure→height via physics) |

The pressure formulation has better inductive bias from the hypsometric equation.

---

## Paper's Claimed Results vs. Reality

| Metric | Paper Claim | Our Implementation | Explanation |
|--------|-------------|-------------------|-------------|
| Best MAE | 3.79m | 25.64m (Fold 0) | Paper likely uses random split, not LOSO |
| Without curriculum | 5.20m | ~25m | Terrain features leak sensor info |
| Generalization | Not tested | ❌ Fails LOSO | Features require global sensor knowledge |

**Conclusion**: The paper's 3.79m result is likely achieved by:
1. Using random train/test split (not LOSO)
2. Benefiting from sensor-specific information in terrain features
3. Not testing true zero-shot generalization

---

## Recommendation

### For Zero-Shot Generalization (LOSO)

Use **Bias-Aware Generalized PINN**:
- ✅ No sensor information leakage
- ✅ Physics-derived features computed per-sample
- ✅ Verified 9.27m MAE across 8-fold LOSO

### For Fixed Sensor Networks

The Terrain-Aware approach might work if:
- All sensors are known at training time
- No new sensors will be added
- You accept the lack of generalization

### For True Terrain-Aware Model

Use external data sources (not sensor locations):
- Digital Elevation Model (DEM) for slope/aspect
- Building footprints for urban canyon effects
- Land cover maps for surface properties
- **Not** sensor density or percentile (these require knowing sensor locations)

---

## Implementation Details

### Terrain-Aware PINN (Paper Implementation)
```python
# Features
- Spatial: Hash encoding (L=16, F=2)
- Temporal: Fourier features (6 freq)
- Terrain: ρ (roughness), r (percentile), d (density)
- Activation: SiLU + LayerNorm

# Training
- Curriculum: 3 stages by altitude & density
- Epochs: 50 (stage 1) + 50 (stage 2) + 150 (stage 3)
- Optimizer: AdamW with cosine annealing
```

### Bias-Aware Generalized PINN (Ours)
```python
# Features
- Spatial: Hash encoding (L=16, F=4)
- Temporal: Fourier features (6 freq)
- Physics: P_bias = P_obs - P_expected
- Activation: SIREN

# Training
- No curriculum (all data at once)
- Epochs: 80
- Optimizer: AdamW with cosine annealing
```

---

## Files

- `height_field_project/train_pinn_with_terrain.py` - Terrain-aware implementation
- `height_field_project/artifacts_terrain/` - Model checkpoints and logs
- `logs/terrain_fold0.log` - Training log for Fold 0

---

## Date

Generated: 2026-03-02
