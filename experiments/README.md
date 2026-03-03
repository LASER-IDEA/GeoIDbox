# Experimental Suite — IEEE TIM Paper

All experiments for:  
**"Physics-Informed Neural Field for Zero-Shot Altitude Estimation via Urban Barometric Sensor Network"**

Python env: `conda activate graphmamba` (Python 3.10, PyTorch, CUDA)  
Working directory: **repo root** (`/data/home/huxiao/workspace/GeoIDbox`)

---

## Overview

| # | Description | Script | Output |
|---|-------------|--------|--------|
| 01 | Baseline comparisons (IDW / Kriging / RF / XGB vs PINF) | `01_baseline_comparisons/run_baseline_fixed.py` | `results.csv` |
| 02 | Ablation study (4 configurations A / C / B / D) | `02_ablation_studies/run_ablation_studies.py` | `results.csv` |
| 03 | Uncertainty quantification + inference efficiency | `03_uncertainty_quantification/run_uq_fixed.py` | `uq_data.csv`, `summary.csv` |
| 04 | Deep ensemble height field map | `04_uncertainty_map/run_height_field_osm.py` | `height_field_osm.html`, Fig 06 |
| — | Generate publication figures 01–05 | `generate_figures.py` | `figures/*.pdf/png` |

---

## Quick Start

```bash
PYTHON=/data/home/huxiao/miniconda3/envs/graphmamba/bin/python
cd /data/home/huxiao/workspace/GeoIDbox

$PYTHON -u experiments/01_baseline_comparisons/run_baseline_fixed.py \
  > experiments/01_baseline_comparisons/run.log 2>&1          # ~30 min

$PYTHON -u experiments/02_ablation_studies/run_ablation_studies.py \
  > experiments/02_ablation_studies/run.log 2>&1              # ~90 min

$PYTHON -u experiments/03_uncertainty_quantification/run_uq_fixed.py \
  > experiments/03_uncertainty_quantification/run.log 2>&1    # ~8 min

$PYTHON experiments/generate_figures.py                       # figures 01–05

$PYTHON experiments/04_uncertainty_map/run_height_field_osm.py  # figure 06
```

> **Ensemble checkpoints** (`ensemble_model_{0..4}.pt`) are already saved in  
> `04_uncertainty_map/` — `run_height_field_osm.py` loads them directly without retraining.

---

## Experiment 01 — Baseline Comparisons

**Goal**: Compare PINF against spatial-interpolation and ML baselines on 8-fold LOSO.

**Methods**: IDW, Ordinary Kriging, Random Forest (pykrige + sklearn), XGBoost  
**Features for ML baselines**: `avg_latitude`, `avg_longitude`, `avg_temperature`, `avg_humidity`  
(`avg_altitude` is excluded — including it causes a circular dependency with the `pressure_bias` target.)

### Verified Results (`01_baseline_comparisons/results.csv`)

| Method | Mean MAE (m) | Std (m) | vs Physics Baseline |
|--------|-------------|---------|---------------------|
| Physics Baseline | 36.96 | 4.04 | — |
| IDW | 37.50 | 3.93 | −1.5 % |
| Ordinary Kriging | 37.35 | 4.35 | −1.1 % |
| Random Forest | 64.82 | 8.68 | +75.4 % worse |
| XGBoost | 65.09 | 7.96 | +76.1 % worse |
| **Proposed PINF** | **3.55** | **1.23** | **−90.4 %** |

**Key insight**: RF / XGB degrade because `pressure_bias` is dominated by per-sensor hardware offsets that cannot be recovered from geographic / meteorological features — confirming this is an extrapolation problem that feature regression cannot solve.

---

## Experiment 02 — Ablation Study

**Goal**: Isolate contributions of the P_bias formulation vs. the 3-stage Altitude Curriculum.

### Verified Results (`02_ablation_studies/results.csv`)

| Setup | Architecture | Target | CL | Mean MAE (m) | Δ vs Baseline |
|-------|-------------|--------|----|-------------|---------------|
| Physics Baseline | — | — | No | 36.96 | — |
| A | SIREN | Δh direct | No | 22.21 | −39.9 % |
| C | SIREN | Δh direct | Yes | 21.00 | −43.2 % |
| B | SIREN + P_bias | δP | No | 9.27 | −74.9 % |
| **D (Full)** | **SIREN + P_bias** | **δP** | **Yes** | **3.55** | **−90.4 %** |

**Key insight**: Curriculum alone (A→C) gives only −5.4 %; P_bias alone (A→B) gives −58.3 %; together (D) they give −90.4 % — strong synergy.

> Setup A folds are cached in `02_ablation_studies/setup_a_cache.npy` to skip ~25 min of retraining.

---

## Experiment 03 — Uncertainty Quantification

**Goal**: Measure MC Dropout uncertainty and inference efficiency on Fold 0.

### Verified Results (`03_uncertainty_quantification/summary.csv`)

| Metric | Value |
|--------|-------|
| Inference latency | 2.1 ms/query |
| Throughput | ~467 queries/second |
| Model parameters | 1,421,146 |
| Model size (fp32) | 5.42 MB |
| MC Dropout σ (mean) | 2.7 × 10⁻⁵ m |
| MC Dropout σ (max) | 7.6 × 10⁻⁵ m |
| σ–\|error\| correlation | 0.087 |

**Why MC Dropout is insufficient**: The physics residual formulation constrains the output so tightly that MC Dropout variance is near-zero (~27 µm) — no meaningful epistemic signal. Deep Ensemble (N=5) is used for the spatial height field instead.

---

## Experiment 04 — Deep Ensemble Height Field Map

**Goal**: Visualise the learned height field over the deployment zone using a Deep Ensemble.

**Architecture**: N=5 BiasAwarePINN models, each trained with 3-stage altitude curriculum (25 epochs/stage, independent random seeds).

**Method**: Inference on all 134,627 real observations (actual P/T/RH) → per-sensor ensemble mean → IDW interpolation to 120×120 grid → publication figure + interactive HTML.

### Verified Per-Sensor Results

| Sensor (last 6) | GNSS (m) | Predicted (m) | Error (m) |
|-----------------|----------|---------------|-----------|
| 250224 | 158.1 | 164.3 | 6.2 |
| 437779 | 93.7 | 97.3 | 3.7 |
| 948226 | 102.5 | 99.5 | 3.0 |
| 508217 | 100.0 | 103.4 | 3.3 |
| 369164 | 111.4 | 114.0 | 2.6 |
| 605977 | 145.5 | 149.7 | 4.2 |
| 527426 | 121.2 | 127.2 | 6.0 |
| 373510 | 259.3 | 257.2 | 2.0 |

IDW grid range: **97.4 – 257.1 m** (spans full urban canyon altitude range).

**Outputs**:
- `figures/06_height_field_osm.pdf/png` — publication figure
- `04_uncertainty_map/height_field_osm.html` — interactive Folium map (open in browser for OSM tiles)

---

## Generated Figures

All figures use: Times New Roman · RdYlBu-derived palette · `#E8EFF6` axes background · dotted grid · 300 DPI.

| Fig | File | Description |
|-----|------|-------------|
| 01 | `01_baseline_comparison.pdf` | Grouped bar: IDW / Kriging / RF / XGB / Physics / PINF |
| 02 | `02_ablation_study.pdf` | Waterfall bars: Physics → A → C → B → D |
| 03 | `03_uncertainty_vs_error.pdf` | Binned bar: MC Dropout quintile vs mean error |
| 04 | `04_spatial_uncertainty.pdf` | Scatter: spatial distribution of prediction error (Fold 0) |
| 05 | `05_per_fold_results.pdf` | Grouped bars: per-fold MAE + RMSE |
| 06 | `06_height_field_osm.pdf` | Contour map of neural height field over deployment zone |

---

## Per-Fold PINF (Setup D) Results

Source: `results/loso_summary.csv`

| Fold | Held-out Sensor | MAE (m) | RMSE (m) | Improvement |
|------|-----------------|---------|----------|-------------|
| 0 | …A64197 | 4.639 | 13.516 | 71.9 % |
| 1 | …A19021 | 4.074 | 10.679 | 67.7 % |
| 2 | …A16069 | 3.723 | 8.959 | 70.1 % |
| 3 | …A80659 | 1.572 | 5.811 | 69.5 % |
| 4 | …A11737 | 4.394 | 10.864 | 72.3 % |
| 5 | …A01284 | 3.226 | 8.730 | 71.8 % |
| 6 | …A38974 | 1.929 | 6.577 | 72.4 % |
| 7 | …A17945 | 4.854 | 10.763 | 74.8 % |
| **Mean** | | **3.55 ± 1.23** | **9.49 ± 2.51** | **71.3 %** |

---

## Notes

- **LOSO protocol**: zero data leakage — each fold trains on 7 sensors, evaluates on the held-out 8th.
- **P_ref**: 101,839.40 Pa (auto-estimated from full training dataset).
- **Hypsometric formula**: `h = (R_dry × T_v / g) × ln(P_ref / P_corrected)`.
- **Ensemble checkpoints**: `04_uncertainty_map/ensemble_model_{0..4}.pt` (pre-trained, no retraining needed).
- Random seeds are fixed per experiment for reproducibility.
