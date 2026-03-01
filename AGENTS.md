# GeoIDbox (高度盒子) - AI Agent Guide

## Project Overview

GeoIDbox (高度盒子) is a research project that builds a **Height Conversion Field** using dense urban sensor networks. It converts between barometric pressure, MSL (Mean Sea Level), and HAE (Height Above Ellipsoid) altitudes using a hybrid approach: physics-based baseline + neural residual field.

### Core Concept
- **Physics as Strong Prior**: Uses barometric formula fitting to establish baseline height estimates
- **Neural Residual Field**: A PyTorch MLP with Fourier features learns micro-meteorological corrections
- **Uncertainty Quantification**: MC Dropout provides prediction uncertainty estimates
- **ERA5 Integration**: Optional macro-scale weather context from ECMWF reanalysis data

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Deep Learning | PyTorch |
| Data Processing | NumPy, Pandas, Scikit-learn, xarray |
| Visualization | Matplotlib, Seaborn |
| Geospatial | pyproj, pykrige |
| Database | PostgreSQL (psycopg2, SQLAlchemy) |
| Weather Data | cdsapi (ERA5) |
| Experiment Tracking | Weights & Biases (optional) |
| Drone Logs | pyulog (PX4 ULog) |

## Project Structure

```
GeoIDbox/
├── height_field_project/       # Core neural field implementation
│   ├── config.py               # Training configuration dataclass
│   ├── neural_field.py         # FourierFeatures + ResidualNeuralField
│   ├── physics_baseline.py     # Barometric formula fitting
│   ├── train.py                # Training pipeline
│   ├── infer.py                # Inference with MC Dropout
│   ├── era5_utils.py           # ERA5 NetCDF enrichment
│   ├── era5_download.py        # ERA5 downloader CLI
│   ├── visualize_grid.py       # Grid visualization
│   └── artifacts/              # Model outputs (created at runtime)
│
├── data/                       # Data directory
│   ├── dataprocess/            # Data preprocessing scripts
│   ├── rawdata/                # Raw sensor CSV files
│   ├── geoids/                 # EGM2008 geoid model files
│   └── reports/                # Analysis outputs and figures
│
├── test/usepx4/                # PX4 drone log processing pipeline
│   ├── step1_parse_drone_log.py
│   ├── step2_download_era5.py
│   ├── step3_get_geoid.py
│   ├── step4_align_data.py
│   ├── step5_physics_baseline.py
│   └── step7_neural_residual_field.py
│
├── 1122demo/                   # Demo and visualization scripts
├── docs/                       # Documentation PDFs
└── neural_field_analysis.py    # Standalone analysis script
```

## Build and Run Commands

### Environment Setup
```bash
# Conda environment (recommended)
conda create -n py310 python=3.10
conda activate py310

# Core dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn xarray netCDF4

# Optional but recommended
pip install wandb cdsapi psycopg2-binary pyproj pykrige pyulog
```

### Training
```bash
# Basic training
python -m height_field_project.train \
  --input_csv sensor_data_clean_stable.csv \
  --epochs 300 \
  --pseudo_ratio 1.0 \
  --pseudo_weight 0.5

# With ERA5 enrichment
python -m height_field_project.train \
  --input_csv sensor_data_clean_stable.csv \
  --era5_nc era5_pl_2024-11-24.nc \
  --epochs 300

# With W&B logging
python -m height_field_project.train \
  --input_csv sensor_data_clean_stable.csv \
  --wandb_project GeoBox \
  --wandb_run_name experiment_1
```

### Inference
```bash
# CSV mode - predict for existing measurements
python -m height_field_project.infer \
  --input_csv sensor_data_clean_stable.csv \
  --samples 20 \
  --out_csv artifacts/predictions.csv

# Grid mode - generate height field slice
python -m height_field_project.infer \
  --grid_bbox 22.60 22.62 114.05 114.07 \
  --grid_res 80 \
  --grid_height 150 \
  --samples 30 \
  --out_csv artifacts/grid_slice.csv
```

### ERA5 Data Download
```bash
python -m height_field_project.era5_download \
  --date 2024-11-24 \
  --time 15:00 \
  --area "22.8,113.8,22.4,114.2" \
  --output era5_pl.nc
```

### Visualization
```bash
# Visualize grid inference results
python -m height_field_project.visualize_grid \
  --csv artifacts/grid_slice.csv \
  --out_dir artifacts/
```

## Data Format

### Input CSV Columns (Required)
```
avg_latitude      - Latitude (degrees)
avg_longitude     - Longitude (degrees)
avg_altitude      - GNSS altitude (meters)
avg_pressure      - Barometric pressure (Pa)
avg_temperature   - Temperature (Celsius)
avg_humidity      - Humidity (%)
week_seq          - Week sequence number (optional, defaults to 0)
processed_time    - Timestamp for ERA5 alignment (optional)
```

### ERA5 Enrichment (Optional)
When `--era5_nc` is provided, these additional features are extracted:
- `era5_tv1000_k`, `era5_tv900_k`: Virtual temperature at pressure levels
- `era5_lapse_1000_900`: Lapse rate between 1000-900 hPa
- `era5_z1000_m`, `era5_z900_m`: Geopotential heights

### Output Artifacts
```
artifacts/
├── model.pt          # Trained PyTorch model
├── scalers.pkl       # StandardScaler objects and feature columns
└── config.json       # Training configuration
```

### Inference Output Columns
```
h_phys_m           - Physics baseline height
residual_mean      - Predicted residual (neural correction)
residual_std       - Uncertainty (MC Dropout std)
h_pred_mean        - Final predicted height (h_phys + residual)
h_pred_std         - Final uncertainty
```

## Code Style Guidelines

### Language
- Comments and docstrings should be in **Chinese** (as per existing codebase convention)
- Variable names use English with Chinese comments explaining intent

### Import Style
```python
import argparse
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
```

### Function Documentation
```python
def fit_barometric_baseline(
    df: pd.DataFrame,
    pressure_col: str = "avg_pressure",
    altitude_col: str = "avg_altitude",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    拟合 ln(P) = a * h + b，得到尺度高度 Hs 和基准气压 P0。
    输出 h_phys_m，作为物理基线高度（近似 MSL）。
    """
```

### Configuration Pattern
Use dataclasses for configuration with `asdict` for serialization:
```python
@dataclass
class TrainingConfig:
    input_csv: str = "sensor_data_clean_stable.csv"
    epochs: int = 300
    lr: float = 1e-3
    # ... more fields
```

## Testing Strategy

### No Formal Unit Tests
This research codebase does not use pytest/unittest. Testing is done through:

1. **Demo Scripts**: Run `1122demo/demo.py` for end-to-end validation
2. **Step-by-step Pipeline**: Use `test/usepx4/step*.py` for incremental testing
3. **Analysis Scripts**: Run `neural_field_analysis.py` for data validation

### Manual Testing Workflow
```bash
# 1. Data preparation
cd data/dataprocess
python analyze.py  # Generates sensor_data_clean_stable.csv

# 2. Training test
python -m height_field_project.train --epochs 10  # Quick test run

# 3. Inference test
python -m height_field_project.infer --input_csv sensor_data_clean_stable.csv

# 4. Visualization check
python -m height_field_project.visualize_grid
```

## Development Conventions

### Pseudo-Points Strategy
Training uses "pseudo-points" (伪点) to regularize extrapolation:
- Randomly sampled in spatial extent of training data
- Target residual = 0 (fallback to physics baseline)
- Controlled by `--pseudo_ratio` (count multiplier) and `--pseudo_weight` (loss weight)

### Physics-Residual Loss
The training loss combines:
1. Huber loss on residual predictions
2. Physics consistency loss: `ln(p_obs) vs ln(p_hat)` from predicted height

### MC Dropout for Uncertainty
Always use `model.predict_mc(x, samples=20)` for inference, not `model.forward()`:
```python
mean, std = model.predict_mc(x_tensor, samples=20)
```

### ERA5 CDS API Setup
For ERA5 downloads, create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: <your-api-key>
```

## Security Considerations

1. **Database Credentials**: The file `1122demo/demo.py` contains hardcoded PostgreSQL credentials. These should be moved to environment variables:
   ```python
   # Instead of hardcoded:
   engine = create_engine("postgresql+psycopg2://dbadmin:IdeaRoot%402023@10.1.3.183:5432/silas-warehouse")
   
   # Use:
   import os
   DB_URL = os.environ.get("DATABASE_URL")
   ```

2. **API Keys**: ERA5 CDS API keys in `~/.cdsapirc` should have appropriate permissions

3. **Drone Logs**: ULog files may contain sensitive location data - handle appropriately

## Key Architectural Decisions

1. **Modular Design**: Each major function (train/infer/visualize) is a separate CLI module
2. **Artifact Persistence**: Models, scalers, and configs saved together for reproducibility
3. **Flexible Feature Engineering**: Optional ERA5 features are detected at runtime
4. **Physics-First Approach**: Neural network only learns residuals, not absolute heights
5. **Spatial Consistency**: Pseudo-points ensure smooth behavior outside observed regions

## Common Tasks

### Adding a New Feature
1. Add to `prepare_features()` in `train.py`
2. Update `build_pseudo_points()` to handle the new feature
3. Re-train and verify artifacts load correctly in `infer.py`

### Debugging Data Issues
Use `neural_field_analysis.py` as a template for exploring:
- Vertical precision tiering
- Environmental coupling
- Diurnal cycle patterns
- Physics residual distributions

### Extending to New Regions
1. Collect sensor data in target region
2. Download ERA5 for the same time/area
3. Run training pipeline
4. Use grid inference to generate field predictions
