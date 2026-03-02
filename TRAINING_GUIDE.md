# PINN Training Guide for GeoIDbox

## Corrected Methodology

The implementation has been corrected to properly handle the physics:

1. **Barometer measures pressure → converts to MSL** (orthometric height) via hypsometric equation
2. **GNSS altitude is already MSL** (orthometric height), NOT HAE
3. **No geoid conversion needed** - both barometer and GNSS output MSL
4. **PINN learns pressure correction field** δP(x, y, z, t) to account for:
   - Sensor calibration biases
   - Local microclimate variations
   - Temperature/humidity effects
   - Non-standard atmospheric conditions

## Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
source /data/home/huxiao/miniconda3/bin/activate graphmamba

# Verify GPUs
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### 2. Download ERA5 Data (Optional but Recommended)

```bash
# Download ERA5 for the correct date range (Nov 10-26, 2025)
python3 -m height_field_project.era5_download_corrected \
  --output_dir data/era5_corrected \
  --start_date 2025-11-10 \
  --end_date 2025-11-26 \
  --area_n 22.8 \
  --area_w 113.8 \
  --area_s 22.4 \
  --area_e 114.2
```

### 3. Training Commands

#### Option A: Standard Training (Single/Multi-GPU)

```bash
source /data/home/huxiao/miniconda3/bin/activate graphmamba

cd /data/home/huxiao/workspace/GeoIDbox

python3 -m height_field_project.train_pinn_multigpu \
  --input_csv data/sensor_data_clean_stable.csv \
  --artifacts_dir height_field_project/artifacts_pinn \
  --era5_nc data/era5_corrected/era5_surface_2025-11.nc \
  --epochs 500 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --restart_epochs 50 \
  --patience 60 \
  --lambda_hydro 0.01 \
  --sensor_embedding_dim 8 \
  --hash_levels 16 \
  --hash_features 2 \
  --hidden_dim 256 \
  --n_hidden_layers 4 \
  --temporal_freqs 4 \
  --use_siren \
  --use_multi_gpu \
  --use_amp \
  --wandb_project GeoIDbox-PINN \
  --wandb_run_name "pinn_v1_full"
```

#### Option B: LOSO Cross-Validation (Rigorous Evaluation)

```bash
source /data/home/huxiao/miniconda3/bin/activate graphmamba

cd /data/home/huxiao/workspace/GeoIDbox

python3 -m height_field_project.loso_validation \
  --input_csv data/sensor_data_clean_stable.csv \
  --output_dir height_field_project/loso_results \
  --epochs 300 \
  --batch_size 1024 \
  --lr 1e-3 \
  --lambda_hydro 0.01 \
  --hash_levels 16 \
  --hidden_dim 256 \
  --n_hidden_layers 4 \
  --use_siren
```

### 4. Inference

```bash
source /data/home/huxiao/miniconda3/bin/activate graphmamba

cd /data/home/huxiao/workspace/GeoIDbox

python3 -m height_field_project.infer_pinn \
  --input_csv data/sensor_data_clean_stable.csv \
  --artifacts_dir height_field_project/artifacts_pinn \
  --output artifacts/pinn_predictions.csv \
  --era5_nc data/era5_corrected/era5_surface_2025-11.nc \
  --mc_samples 30
```

## Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--lambda_hydro` | Hydrostatic constraint weight | 0.01-0.1 |
| `--hash_levels` | Number of hash encoding levels | 16 |
| `--hash_features` | Features per hash level | 2 |
| `--hidden_dim` | MLP hidden dimension | 256 |
| `--n_hidden_layers` | Number of MLP layers | 4 |
| `--sensor_embedding_dim` | Per-sensor embedding size | 8 |
| `--use_siren` | Use SIREN activation | True |
| `--use_amp` | Mixed precision training | True |

## Architecture Summary

```
Input: [lat, lon, z, t, T, RH, sensor_id]
  ↓
Hash Encoding (L=16, F=2) → 32-dim spatial features
Fourier Encoding (4 freq) → 8-dim temporal features
Sensor Embedding → 8-dim sensor-specific features
  ↓
Concatenate: 32 + 1 (z) + 8 + 1 (T) + 1 (RH) + 8 = 51-dim
  ↓
SIREN MLP: [51] → [256] × 4 → [1]
  ↓
Output: δP (pressure correction in Pa)
  ↓
P_corrected = P_obs + δP
  ↓
Hypsometric equation → H_MSL
  ↓
Compare to GNSS (MSL)
```

## Loss Function

```
L_total = L_data + λ_hydro * L_hydrostatic

L_data = MAE(H_pred, H_GNSS)
L_hydrostatic = ||dP/dz + ρg||²
```

## Expected Results

- **Physics Baseline**: ~30-40m MAE
- **PINN (basic)**: ~10-15m MAE
- **PINN (with ERA5 + hydrostatic)**: ~3-5m MAE
- **Target**: < 3m MAE for meter-level accuracy

## Monitoring

With wandb enabled, track:
- `train_loss`: Total training loss
- `val_mae`: Validation MAE
- `val_improvement`: % improvement over physics baseline
- `test_mae`: Final test MAE
- `test_improvement`: Final improvement %

## Troubleshooting

1. **High physics baseline error (>50m)**:
   - Check `--p_ref_method auto` is enabled
   - Verify temperature/humidity data quality

2. **No improvement from PINN**:
   - Increase model capacity (`--hidden_dim 256`, `--n_hidden_layers 4`)
   - Train longer (`--epochs 500`)
   - Adjust learning rate (`--lr 5e-4`)

3. **Out of memory**:
   - Reduce batch size (`--batch_size 1024`)
   - Reduce hash levels (`--hash_levels 12`)
   - Use gradient accumulation (`--grad_accum_steps 2`)

4. **Slow training**:
   - Enable AMP (`--use_amp`)
   - Increase batch size (up to GPU memory limit)
   - Use DataParallel (automatic with `--use_multi_gpu`)
