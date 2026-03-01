# GeoIDbox Model Versions

## Version 1.0: Weather Decomposition (Analytical)
**File**: `run_weather_model_v1.0.py`

### Architecture
- **Spatial Component**: IDW (Inverse Distance Weighting) interpolation
- **Weather Component**: Mean anomaly across training sensors
- **Physics**: Hypsometric equation with virtual temperature

### Results (LOSO on 7 sensors)
| Metric | Value |
|--------|-------|
| Physics Baseline | 68.18 ± 5.91m |
| + IDW Spatial | 17.60 ± 1.07m |
| **+ Mean Weather** | **6.29 ± 4.36m** |
| **Total Improvement** | **61.89m (90.8%)** |

### Key Insight
Weather signal accounts for 90.5% of residual variance and is 99.9% correlated across sensors. Simple mean works because weather is a shared macro phenomenon.

### Limitations
1. **IDW is static**: Cannot learn complex spatial patterns
2. **Mean weather**: No temporal dynamics modeling
3. **No uncertainty**: Fixed predictions without confidence
4. **No physics constraints**: Violations possible (e.g., positive pressure-height correlation)

---

## Version 2.0: Physics-Informed Neural Network (PINN)
**Status**: ❌ **FAILED** - See `tests/trail0301-fail/`  
**Files**: `run_pinn_v2.py`, `train_pinn_v2_fast.py`, `train_pinn_v2_curriculum.py`

### Architecture

```
Input: (lat, lon, time, temperature, humidity, pressure, ERA5)
          ↓
┌─────────────────────────────────────────────────────────────┐
│                     SHARED BACKBONE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Hash Encoding│  │Fourier Time  │  │ Feature Encoder │   │
│  │ (L=16 levels)│  │ (16 freq)    │  │ (physics feats) │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘   │
│         └─────────────────┼───────────────────┘             │
│                           ↓                                 │
│              ┌─────────────────────────┐                    │
│              │   MLP Backbone (4x128)  │                    │
│              │  + LayerNorm + SiLU      │                    │
│              └───────────┬─────────────┘                    │
└──────────────────────────┼──────────────────────────────────┘
                           ↓
            ┌──────────────┼──────────────┐
            ↓              ↓              ↓
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   SPATIAL    │ │   WEATHER    │ │    GATE      │
     │    BRANCH    │ │    BRANCH    │ │   NETWORK    │
     ├──────────────┤ ├──────────────┤ ├──────────────┤
     │ MLP (2x64)   │ │ MLP (2x64)   │ │ MLP (2x64)   │
     │ + Uncertainty│ │ + Uncertainty│ │ + Softmax    │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            ↓                ↓                ↓
      spatial_pred     weather_pred      (α_s, α_w)
      + spatial_std    + weather_std
            │                │                │
            └────────────────┼────────────────┘
                             ↓
              GATED COMBINATION: α_s × spatial + α_w × weather
                             ↓
                    TOTAL RESIDUAL PREDICTION
```

### Physics-Informed Losses

```python
L_total = λ_data × L_data + 
          λ_temporal × L_temporal_smoothness +
          λ_hydrostatic × L_hydrostatic_consistency +
          λ_spatial × L_spatial_smoothness
```

1. **Data Fidelity**: MAE between predicted and true HAE
2. **Temporal Smoothness**: Weather should vary smoothly over time
3. **Hydrostatic Consistency**: Penalize positive pressure-residual correlation
4. **Spatial Smoothness**: Spatial bias should vary smoothly in space

### ⚠️ Failure Analysis

**All v2.0 attempts failed to beat v1.0:**

| Configuration | Epochs | Result | vs v1.0 (6.29m) |
|--------------|--------|--------|-----------------|
| Basic training | 80 | 17.60m | ❌ -11.31m |
| +Curriculum | 150 | 34.32m | ❌ -28.03m |
| +Curriculum+Attention | 300 | 34.23m | ❌ -27.94m |

**Why it failed:**
1. Weather is 99.9% correlated → mean is already optimal
2. Only 7 sensors → IDW is near-optimal
3. Attention collapsed to 100% spatial (ignored weather)
4. More training caused overfitting

**Moved to**: `tests/trail0301-fail/` (see README there)

---

### Key Features

| Feature | v1.0 (IDW) | v2.0 (PINN) |
|---------|-----------|-------------|
| **Spatial Model** | IDW interpolation | Hash encoding + MLP |
| **Temporal Model** | Mean anomaly | Fourier features + MLP |
| **Uncertainty** | None | Per-component std |
| **Adaptive Fusion** | Fixed addition | Learned gating |
| **Physics Constraints** | None | 3 constraints |
| **Training** | None required | End-to-end gradient descent |

### Preliminary Results (Single Fold, 100 epochs)

| Model | MAE | Improvement |
|-------|-----|-------------|
| Physics Baseline | 56.96m | - |
| PINN v2.0 | **14.28m** | **42.68m (74.9%)** |

### Expected Full Results (7-fold LOSO, 200+ epochs)

Based on architecture capacity and physics constraints:
- **Target MAE**: 3-8m
- **Best case**: Approaching v1.0 performance (6.29m) with better generalization
- **Advantage**: Learns spatial patterns beyond IDW, temporal dynamics beyond mean

---

## Comparison Summary

### When to Use v1.0
- **Quick deployment**: No training required
- **Interpretability**: Explicit spatial/weather decomposition
- **Resource constrained**: Minimal compute for inference
- **Baseline**: Understand fundamental limits

### When to Use v2.0
- **Accuracy critical**: Need every meter of improvement
- **Complex terrain**: Urban canyons, microclimates
- **Temporal dynamics**: Weather pattern evolution
- **Uncertainty needed**: Confidence intervals for decisions
- **Production scale**: GPU inference acceptable

---

## Training Recommendations

### v1.0
```bash
# No training needed - analytical solution
python run_weather_model_v1.0.py
# Result: 6.29m MAE (instant)
```

### v2.0
```bash
# Full training with curriculum learning
python run_pinn_v2.py --epochs 300

# Expected convergence:
# Epoch 0-50:   ~50m MAE (learning spatial patterns)
# Epoch 50-150: ~15m MAE (refining weather dynamics)
# Epoch 150-300: ~5-8m MAE (fine-tuning with physics constraints)
```

---

## Future Work: Version 3.0 Ideas

1. **Graph Neural Networks**: Message passing between sensors
2. **Attention Mechanisms**: Learn which sensors to trust for weather
3. **Transformer Architecture**: For long-range temporal dependencies
4. **Multi-Task Learning**: Joint height + weather prediction
5. **Domain Adaptation**: Transfer to new cities without retraining

---

## References

- Methodology: See `paper/sections/method.tex`
- v1.0 Results: `experiments/results/refined_model/weather_model_v1.0_results.json`
- v2.0 Code: `run_pinn_v2.py`
