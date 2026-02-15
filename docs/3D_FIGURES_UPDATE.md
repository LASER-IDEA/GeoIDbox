# 3D Figures Update - Smooth RBF Interpolation

## Update Date: 2026-02-14

## Changes

### 3D Surface Smoothing
All 3D figures now use **RBF (Radial Basis Function) interpolation** with **Gaussian smoothing** to eliminate锯齿感:

- **Grid density**: Increased from 80×80 to 150×150
- **Interpolation**: RBF multiquadric for SRTM, RBF inverse for MAE
- **Smoothing**: Gaussian filter with sigma=1.5-2.0
- **Result**: Smooth, professional-looking surfaces

### Files Updated

| File | Before | After |
|------|--------|-------|
| fig7_3d_altitude_field.png | 1.5 MB (锯齿) | 2.3 MB (平滑) |
| fig8_3d_error_heatmap.png | 1.4 MB (锯齿) | 2.0 MB (平滑) |

### New File
- **fig9_osm_basemap.png**: 2D map with SRTM contours (1.2 MB)
  - Shows sensor locations with real SRTM elevation contours
  - Intended for OSM basemap overlay (network unavailable during generation)

## Technical Details

### RBF Interpolation
```python
from scipy.interpolate import Rbf

# For SRTM heights
rbf_srtm = Rbf(lons, lats, srtms, function='multiquadric', smooth=0.1)
ZI_srtm = rbf_srtm(XI, YI)

# Additional Gaussian smoothing
ZI_srtm_smooth = gaussian_filter(ZI_srtm, sigma=1.5)
```

### Data Sources
- **SRTM heights**: `data/processed/sensor_data_with_srtm.csv`
- **MAE values**: `experiments/results/advanced_improvements_results.json`
- **Coordinates**: Real GPS from 8 sensor locations

## Script
New script: `paper/generate_3d_figures_smooth.py`

Run:
```bash
source ~/miniconda3/bin/activate graphmamba
python paper/generate_3d_figures_smooth.py
```
