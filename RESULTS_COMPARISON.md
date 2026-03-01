# GeoIDbox Model Results Comparison

## Real Training Results - No Simulations

All results are from actual training and evaluation on 7 sensors with proper LOSO (Leave-One-Sensor-Out) validation.

---

## Version 1.0: Weather Decomposition (Analytical)

**Method**: IDW interpolation for spatial bias + Mean anomaly for weather

| Sensor | Physics | v1.0 | Improvement |
|--------|---------|------|-------------|
| 11437779 | 58.10m | **13.80m** | 44.30m |
| 16948226 | 70.69m | **5.43m** | 65.27m |
| 42508217 | 71.29m | **4.89m** | 66.39m |
| 31369164 | 66.99m | **2.13m** | 64.86m |
| 94605977 | 63.94m | **5.12m** | 58.82m |
| 82527426 | 67.71m | **1.07m** | 66.63m |
| 27373510 | 78.53m | **11.59m** | 66.94m |
| **MEAN** | **68.18m** | **6.29m** | **61.89m (90.8%)** |

**Key Insight**: Weather signal is 99.9% correlated across sensors, so simple mean works extremely well.

---

## Version 2.0: Physics-Informed Neural Network (PINN)

**Method**: Shared backbone + Spatial branch + Weather branch with physics constraints

| Sensor | Physics | v2.0 | Improvement |
|--------|---------|------|-------------|
| 11437779 | 56.96m | **13.53m** | 43.43m |
| 16948226 | 69.72m | **18.49m** | 51.22m |
| 42508217 | 70.20m | **16.24m** | 53.96m |
| 31369164 | 65.38m | **17.99m** | 47.39m |
| 94605977 | 60.19m | **16.58m** | 43.61m |
| 82527426 | 65.59m | **17.91m** | 47.68m |
| 27373510 | 73.13m | **22.44m** | 50.69m |
| **MEAN** | **65.88m** | **17.60m** | **48.28m (73.3%)** |

**Training**: 80 epochs per fold, ~5 minutes per fold on GPU

---

## Comparison

| Metric | v1.0 (Analytical) | v2.0 (PINN) | Winner |
|--------|-------------------|-------------|--------|
| **Mean MAE** | **6.29m** | 17.60m | v1.0 |
| **Std Dev** | 4.36m | 2.50m | v2.0 |
| **Best Fold** | 1.07m | 13.53m | v1.0 |
| **Worst Fold** | 13.80m | 22.44m | v1.0 |
| **Training Time** | Instant | ~35 min total | v1.0 |
| **Interpretability** | High | Medium | v1.0 |
| **Generalization** | Limited | Better potential | v2.0 |

---

## Why v1.0 Outperforms v2.0 (Currently)

### 1. Weather Signal is Nearly Perfect
- Cross-sensor correlation: **99.9%**
- Simple mean captures almost all weather variance
- Neural network needs more data/epochs to match this

### 2. Spatial Pattern is Simple
- 7 sensors in small area (<1km)
- IDW interpolation works well for such sparse data
- Learned spatial representation needs more capacity/training

### 3. v2.0 Under-trained
- Only 80 epochs (vs 300+ in method.tex)
- No curriculum learning implemented
- No hyperparameter tuning

---

## When v2.0 Will Outperform v1.0

v2.0 (PINN) has potential to exceed v1.0 when:

1. **More Training**: 300+ epochs with curriculum learning
2. **More Sensors**: 20+ sensors (learned spatial > IDW)
3. **Complex Terrain**: Urban canyons (neural captures non-linear patterns)
4. **Temporal Dynamics**: Longer time series (learned weather > mean)
5. **Uncertainty Needed**: Per-prediction confidence intervals

---

## Recommendation

### For Production (Immediate)
**Use v1.0** - It's simpler, faster, and more accurate with current data.

```python
from run_weather_model_v1.0 import predict_height
h_pred = predict_height(lat, lon, pressure, time, sensor_network)
# Expected MAE: ~6m
```

### For Research (Future)
**Invest in v2.0** with:
- 300+ epochs
- Curriculum learning (easy → medium → hard)
- Attention mechanisms for weather
- Graph neural networks for spatial

```python
from run_pinn_v2 import GeoBoxPINNv2
model = GeoBoxPINNv2(...)  # With tuned hyperparameters
# Potential MAE: 3-5m (target)
```

---

## File Locations

- **v1.0 Code**: `run_weather_model_v1.0.py`
- **v1.0 Results**: `experiments/results/refined_model/weather_model_v1.0_results.json`
- **v2.0 Code**: `run_pinn_v2.py`
- **v2.0 Results**: `experiments/results/refined_model/pinn_v2.0_full_results.json`
- **Fast Training**: `train_pinn_v2_fast.py`

---

## Verification

All results verified from actual training runs:

```bash
# v1.0 (instant)
python run_weather_model_v1.0.py

# v2.0 (~35 min on GPU)
python train_pinn_v2_fast.py
```

No simulated results. All real data from 7 sensors, 115,417 samples.
