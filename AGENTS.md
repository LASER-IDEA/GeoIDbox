# GeoIDbox (高度盒子) - AI Agent Documentation

## Project Overview

**GeoIDbox** (高度盒子, "Height Box") is a geospatial height measurement and modeling system that combines physical barometric models with neural residual fields to achieve accurate altitude estimation. The project processes sensor data from a distributed urban sensor network and drone flight logs to build a dense, queryable height conversion field between barometric pressure, MSL (Mean Sea Level), and HAE (Height Above Ellipsoid).

### Core Value Proposition
- Converts barometric pressure measurements to precise height estimates
- Uses physics-based models as strong priors with neural networks learning residual corrections
- Integrates ERA5 meteorological reanalysis data for macro-scale weather patterns
- Supports uncertainty quantification via MC Dropout

### Target Region
- Primary area: Shenzhen, China (深圳) - approximately 22.5°N, 114°E
- Sensor network covers ~1 km² urban area with 8-10 sensor nodes

---

## Technology Stack

### Programming Language
- **Python 3.7+** (developed primarily with Python 3.10)

### Core Dependencies

#### Deep Learning & ML
- `torch` (PyTorch) - Neural field implementation
- `torch.nn`, `torch.optim` - Model layers and optimizers
- `sklearn` (scikit-learn) - StandardScaler, train_test_split, LinearRegression

#### Data Processing
- `pandas` - DataFrame operations for sensor data
- `numpy` - Numerical computations
- `xarray` - ERA5 NetCDF data handling
- `scipy` - Statistical distributions, signal processing

#### Geospatial & Meteorological
- `cdsapi` - Copernicus Climate Data Store API for ERA5 downloads
- `geoid` / `pygeodesy` - EGM96/EGM2008 geoid undulation calculations
- `pyproj` - Coordinate projections
- `pykrige` - Kriging interpolation (in demo)

#### Visualization
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualization

#### Drone/Log Processing
- `pyulog` - PX4 ULog file parsing

#### Optional
- `wandb` - Experiment tracking (Weights & Biases)
- `psycopg2`, `sqlalchemy` - PostgreSQL database connectivity (legacy)

---

## Project Structure

```
GeoIDbox/
├── README.md                          # Main project documentation (Chinese)
├── AGENTS.md                          # AI Agent documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── height_field_project/              # ★ Core neural field module
│   ├── README.md                      # Module-specific documentation
│   ├── config.py                      # TrainingConfig dataclass + save/load
│   ├── physics_baseline.py            # Barometric height fitting (ln(P) linear regression)
│   ├── neural_field.py                # FourierFeatures + ResidualNeuralField (PyTorch)
│   ├── train.py                       # End-to-end training pipeline
│   ├── infer.py                       # Inference: CSV or grid mode with MC Dropout
│   ├── era5_utils.py                  # ERA5 NetCDF enrichment functions
│   ├── era5_download.py               # CDS API download script
│   ├── visualize_grid.py              # Grid visualization utilities
│   └── visualize_grid_osm.py          # OpenStreetMap overlay visualization
│
├── pipeline/                          # Data processing pipeline scripts
│   ├── step1_parse_drone_log.py       # ULog → CSV conversion
│   ├── step2_download_era5.py         # ERA5 data download (CDS API)
│   ├── step3_get_geoid.py             # EGM96 geoid undulation lookup
│   ├── step3b_downscale.py            # ERA5 downscaling utilities
│   ├── step4_align_data.py            # Data alignment/synchronization
│   ├── step5_physics_baseline.py      # Physical model baseline computation
│   └── step7_neural_residual_field.py # Legacy standalone neural field
│
├── analysis/                          # Analysis scripts
│   ├── analyze.py                     # Sensor network movement detection
│   ├── data_analyze.py                # Basic data analysis utilities
│   ├── deep_sensor_analysis.py        # Three-dimension analysis (precision tiers, environmental coupling, diurnal cycles)
│   ├── neural_field_analysis.py       # Neural field result analysis
│   └── height_box_analysis.py         # Height box specific analysis
│
├── aesm/                              # Altitude Error Statistical Modeling
│   ├── batch_data_analysis.py         # Batch process PX4 logs, generate statistics
│   ├── simpy_sim.py                   # Discrete event simulation
│   ├── bluesky_sim.py                 # BlueSky simulator integration
│   ├── complex_sim.py                 # Complex scenario simulation
│   └── sensor_uncertainty_stats.json  # Output: height error distributions
│
├── scripts/                           # Utility scripts
│   └── data_utils/                    # Data processing utilities
│       ├── analyze_weekly_sensor_data.py
│       ├── aggregate_weekly.py
│       ├── generate_weekly_sensor_figures.py
│       └── split_weekly_data.py
│
├── tests/                             # Test scripts
│   └── contour/                       # Contour analysis tests
│       ├── new_error_contour.py
│       └── analyze_contour_data.py
│
├── legacy_code/                       # Legacy demo code (GRU/LSTM, Kriging)
│   ├── demo.py
│   ├── visualize_demo.py
│   ├── new_error_contour_test.py
│   └── minimum_enclosing_box.py
│
├── data/                              # Data directory
│   ├── raw/                           # Raw sensor data (gitignored)
│   ├── processed/                     # Processed data (gitignored)
│   ├── external/                      # External data: ERA5, geoid (gitignored)
│   └── reports/                       # Generated analysis reports
│
├── models/                            # Trained model storage (gitignored)
├── artifacts/                         # Generated artifacts (gitignored)
├── geoids/                            # EGM2008 geoid model files
└── docs/                              # Documentation
    └── pdfs/                          # PDF documents
```

---

## Build and Execution Commands

### No Formal Build System
This project has no `setup.py`, `pyproject.toml`, or `requirements.txt`. Dependencies must be installed manually.

### Common Setup
```bash
# Install core dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn xarray scipy

# Install geospatial dependencies
pip install geoid pygeodesy pyproj cdsapi pyulog

# Optional: experiment tracking
pip install wandb
```

### Training the Neural Field

```bash
cd /path/to/GeoIDbox

# Basic training
python -m height_field_project.train \
  --input_csv data/processed/sensor_data_clean_stable.csv \
  --epochs 300 \
  --pseudo_ratio 1.0 \
  --pseudo_weight 0.5

# With ERA5 enrichment
python -m height_field_project.train \
  --input_csv data/processed/sensor_data_clean_stable.csv \
  --era5_nc data/external/era5_pl_2024-11-24.nc \
  --epochs 300
```

### Inference

```bash
# CSV mode - predict for existing measurements
python -m height_field_project.infer \
  --input_csv data/processed/sensor_data_clean_stable.csv \
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

### AESM Analysis (Drone Logs)

```bash
# Place parsed CSV logs in px4log/px4_logs_parsed/
python aesm/batch_data_analysis.py
# Outputs: height_uncertainty_model.png, sensor_uncertainty_stats.json
```

### Data Analysis Workflows

```bash
# Movement detection and cleaning
python analysis/analyze.py

# Deep sensor analysis (generates reports in data/reports/)
python analysis/deep_sensor_analysis.py

# Weekly sensor data analysis
python scripts/data_utils/analyze_weekly_sensor_data.py
```

---

## Code Style Guidelines

### Language Convention
- **Primary documentation language**: Chinese (Simplified)
- **Code comments**: Mix of Chinese and English
- **Variable names**: English with occasional Chinese comments
- **Docstrings**: Chinese with English translations for core functions

### Naming Conventions
- Functions: `snake_case` (e.g., `fit_barometric_baseline`, `prepare_features`)
- Classes: `PascalCase` (e.g., `ResidualNeuralField`, `TrainingConfig`)
- Constants: `UPPER_CASE` (e.g., `G0 = 9.80665`, `BIAS_WINDOW = 50`)
- Private functions: Leading underscore (e.g., `_interp_level`)

### Code Organization Patterns
1. **Step-by-step scripts**: Numbered workflow files (`step1_`, `step2_`, etc.)
2. **Module structure**: Core functionality in `height_field_project/` subpackage
3. **Config dataclass**: Centralized hyperparameters in `config.py`
4. **Artifacts directory**: Models, scalers, and outputs go to `artifacts/`

### File Header Template
```python
"""
Brief description (Chinese preferred)

Longer description if needed.
"""
```

---

## Key Data Schemas

### Sensor Data CSV (`sensor_data_clean_stable.csv`)
```
uid,processed_time,record_count,avg_temperature,avg_humidity,
avg_pressure,avg_altitude,avg_height,avg_vbat,
avg_satellites,avg_hdop,avg_latitude,avg_longitude,week_tag,week_seq
```

### Training Data CSV (`final_training_data.csv`)
```
timestamp,lat,lon,p_drone_pa,h_hae_true,p_ref_pa,h_ref_msl,
t_ref_k,q_ref,roughness,n_geoid
```

### Neural Field Output (`predictions.csv`)
```
avg_latitude,avg_longitude,h_phys_m,avg_temperature,avg_humidity,
avg_pressure,week_seq,residual_mean,residual_std,h_pred_mean,h_pred_std
```

---

## Testing Strategy

### No Formal Test Suite
The project does not use `pytest` or `unittest`. Testing is done via:

1. **Script-level validation**: Scripts print progress and sample outputs
2. **Visualization verification**: Generate plots to visually inspect results
3. **Data consistency checks**: Scripts validate required columns exist

### Manual Testing Approach
```bash
# Run individual step scripts with small data samples
python step1_parse_drone_log.py  # Check CSV output
python step3_get_geoid.py        # Verify geoid lookup
```

---

## Development Workflow

### Typical Development Flow
1. **Data Preparation**: Run `analysis/analyze.py` to clean sensor data
2. **ERA5 Download**: Run `pipeline/step2_download_era5.py` for weather data
3. **Training**: Execute `height_field_project/train.py`
4. **Inference**: Run `height_field_project/infer.py` for predictions
5. **Analysis**: Use `analysis/deep_sensor_analysis.py` for insights

### Artifact Management
- Models saved to: `height_field_project/artifacts/`
- Reports saved to: `data/reports/`
- Figures/plots: Generated in working directory or `data/`

---

## External Data Dependencies

### ERA5 Data (Copernicus CDS)
- Requires free CDS API account: https://cds.climate.copernicus.eu/
- API key must be configured in `~/.cdsapirc`
- Downloads pressure-level and single-level reanalysis data

### Geoid Models
- EGM96: Auto-downloaded via `geoid` package (~24MB)
- EGM2008: Manual download to `geoids/egm2008-5.pgm`

### PX4 Flight Logs
- Download from https://logs.px4.io/
- Requires PX4/flight_review tools for parsing

---

## Security Considerations

### Database Credentials (Legacy)
- **WARNING**: `legacy_code/demo.py` contains hardcoded PostgreSQL credentials
- Credentials are for internal network only but should be rotated
- Never commit production credentials to git

### API Keys
- CDS API key stored in `~/.cdsapirc` (standard location)
- No API keys should be committed to repository

---

## Notes for AI Agents

1. **Language**: Comments and documentation are primarily in Chinese. Use Chinese for any new documentation.

2. **Data Assumptions**:
   - Shenzhen region (22-23°N, 113-121°E)
   - UTC timestamps
   - Pressure in Pascals (not hPa)
   - Height in meters

3. **Physics Model**:
   - Uses barometric formula: `ln(P) = a * h + b`
   - Scale height `Hs = -1/a`, reference pressure `P0 = exp(b)`
   - Residual = observed_altitude - physics_predicted_altitude

4. **Neural Field Architecture**:
   - Fourier feature positional encoding (L=6 frequencies)
   - 5-layer MLP with SiLU activation
   - MC Dropout for uncertainty (train mode during inference)

5. **Common Issues**:
   - ERA5 data may have coverage gaps → check NaN handling
   - ULog parsing requires specific PX4 message types
   - Geoid files are large and not in git

6. **Extending the Code**:
   - Add new features to `prepare_features()` in `train.py`
   - Modify `ResidualNeuralField` architecture in `neural_field.py`
   - Update `TrainingConfig` dataclass for new hyperparameters
