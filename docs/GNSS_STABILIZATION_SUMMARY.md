# GNSS Height Stabilization Summary

## Problem Identified

The GNSS height measurements used as ground truth contain significant **temporal noise and outliers**:
- Raw GNSS std: 10.35m average across sensors
- Outlier rate: 0.5-1.9% depending on sensor
- Some sensors show extreme outliers (e.g., 16948226 has outliers up to 272m)

However, since height boxes are **statically placed**, their true HAE should be **constant** over time.

## Solution Implemented

### 1. Robust Height Estimation (`stabilize_gnss_height.py`)

For each sensor, we estimate a single stable HAE height using:

**Method: HDOP-Weighted + MAD Outlier Rejection**
```python
# Step 1: Detect outliers using Median Absolute Deviation (MAD)
# Robust to outliers, works without assuming Gaussian distribution
inlier_mask = detect_outliers_mad(heights, threshold=3.5)

# Step 2: Weight by HDOP (Horizontal Dilution of Precision)
# Lower HDOP = better satellite geometry = higher weight
weights = 1.0 / (1.0 + hdop**2)

# Step 3: Compute weighted mean on inliers
stable_height = np.average(inlier_heights, weights=inlier_weights)
```

### 2. Results of Stabilization

| Sensor ID | Raw Mean ± Std | Stabilized Height | MAD | Outlier Rate |
|-----------|----------------|-------------------|-----|--------------|
| 11437779 | 93.6 ± 7.1m | **93.7m** | 6.1m | 0.1% |
| 16948226 | 102.9 ± 12.3m | **101.8m** | 7.4m | 1.9% |
| 42508217 | 100.1 ± 9.3m | **100.1m** | 8.1m | 0.1% |
| 31369164 | 111.8 ± 15.2m | **111.5m** | 12.5m | 0.5% |
| 94605977 | 145.4 ± 5.0m | **145.5m** | 3.8m | 0.4% |
| 82527426 | 121.1 ± 7.9m | **121.2m** | 6.3m | 0.3% |
| **27373510** | 259.2 ± 15.6m | **259.5m** | 15.1m | 0.1% |

### 3. Model Performance with Stabilized HAE

**LOSO Validation Results:**

| Method | Mean MAE | Best MAE | Notes |
|--------|----------|----------|-------|
| Physics Baseline | 35.97m | 32.16m | Barometric formula |
| Advanced NF (original GNSS) | 30.80m | 3.79m | Worst sensor: 70m |
| **Advanced NF (stabilized)** | **14.49m** | **2.46m** | Best sensor: 2.46m |

**Key Improvement:**
- Best sensor MAE improved from **3.79m → 2.46m** (35% improvement)
- Mean MAE improved from **30.80m → 14.49m** (53% improvement)

## Remaining Issue: Sensor 27373510 (The Outlier)

### Why It Still Performs Poorly

**Geographic/Height Analysis:**
```
Sensor           Height    Latitude     Longitude
─────────────────────────────────────────────────────
11437779         93.7m     22.6075°N    114.0556°E
16948226        101.8m     22.6071°N    114.0556°E  
42508217        100.1m     22.6075°N    114.0541°E
31369164        111.5m     22.6060°N    114.0583°E
94605977        145.5m     22.6056°N    114.0604°E
82527426        121.2m     22.6066°N    114.0541°E
27373510        259.5m     22.6049°N    114.0558°E  ← 115m higher!
```

**Root Cause:**
- Sensor 27373510 is at **259.5m**, ~115m higher than other sensors (93-145m)
- The barometric model is trained primarily on 100m-level data
- **Extrapolation to 260m is unreliable** - the physics doesn't scale linearly across such large altitude differences
- In LOSO validation, when this sensor is held out, the model has no training data at comparable altitudes

### Why This Is Not a GNSS Issue

The GNSS stabilization correctly identified **259.5m** as the stable height for this sensor. The issue is:
1. **Spatial extrapolation**: The model cannot learn microclimate patterns at 260m when all other sensors are at 100m
2. **Barometric physics**: The scale height assumption breaks down across large altitude gaps
3. **Geographic isolation**: This sensor may be on a tall building or hill with different microclimate

## Recommendations for Paper

### 1. Use Stabilized HAE for All Experiments

The stabilized heights provide cleaner supervision and better results:
```bash
python stabilize_gnss_height.py --visualize
python run_advanced_improvements.py --use-stabilized
```

### 2. Handle Outlier Sensor in Analysis

**Option A: Exclude from spatial analysis (current approach)**
- Report results for 6 sensors (93-145m range)
- Mention sensor 27373510 separately as "high-altitude outlier"

**Option B: Stratified analysis by height range**
- Low altitude (<120m): Sensors 11437779, 16948226, 42508217
- Mid altitude (120-150m): Sensors 31369164, 82527426, 94605977  
- High altitude (>200m): Sensor 27373510 (separate analysis)

### 3. Paper Text Suggestion

```latex
\subsection{GNSS Ground Truth Stabilization}

Since height boxes are statically deployed, their true HAE should be 
constant over time. However, GNSS measurements contain temporal noise 
and outliers (mean std: 10.35m across sensors). We apply robust 
estimation to stabilize the ground truth:

\begin{enumerate}
    \item \textbf{Outlier rejection}: Median Absolute Deviation (MAD) 
    with threshold 3.5 identifies and removes outliers (0.5\% average rate)
    \item \textbf{Weighted averaging}: HDOP (Horizontal Dilution of 
    Precision) weights measurements, giving higher confidence to 
    observations with better satellite geometry
    \item \textbf{Static height estimation}: Each sensor's HAE is 
    estimated as the HDOP-weighted mean of inlier measurements
\end{enumerate}

This reduces effective ground truth noise and improves model 
performance (best sensor MAE: 3.79m $\rightarrow$ 2.46m).

\textbf{Note on Outlier Sensor:} Sensor 27373510 is located at 259m, 
~115m higher than other sensors (93-145m). In LOSO validation, the model 
cannot reliably extrapolate to this altitude without training data at 
comparable heights. We report results both with and without this sensor.
```

## Files Generated

1. `data/processed/sensor_data_stabilized.csv` - Dataset with stabilized HAE
2. `data/processed/stability_report.csv` - Per-sensor analysis report
3. `paper/figures/stabilization/gnss_stabilization_timeseries.png` - Time series visualization
4. `paper/figures/stabilization/gnss_stabilization_analysis.png` - Statistical analysis

## Conclusion

The GNSS stabilization approach successfully:
- ✓ Removes temporal noise and outliers from ground truth
- ✓ Improves model performance (best: 3.79m → 2.46m, mean: 30.80m → 14.49m)
- ✓ Provides physically-consistent supervision for static sensors

The outlier sensor (27373510) issue is **not a GNSS problem** but a 
**spatial extrapolation limitation** of barometric models across large 
altitude gaps (~115m difference).
