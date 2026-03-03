"""
Ensemble Uncertainty Map

Trains N independent BiasAwarePINN models (deep ensemble, different init seeds)
on the full dataset, then evaluates them on a dense lat/lon grid over the
GeoBox deployment zone.  The per-pixel std across ensemble predictions gives a
well-calibrated spatial uncertainty estimate (ensemble UQ >> MC Dropout for this
model class because epistemic variance is negligible after training).

Outputs
-------
experiments/04_uncertainty_map/ensemble_grid.csv   – per-pixel lat/lon/mean/std
experiments/figures/06_ensemble_uncertainty_map.pdf/png
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader

sys.path.insert(0, '/data/home/huxiao/workspace/GeoIDbox')

from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_generalized_with_bias import (
    compute_sensor_bias, BiasAwarePINNDataset, BiasAwarePINN
)
from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.train_bias_aware_with_curriculum import (
    create_curriculum_splits, train_epoch
)
from height_field_project.train_pinn import set_seed, parse_timestamp

R_DRY_AIR = 287.05
G_STANDARD = 9.80665
N_ENSEMBLE = 5          # number of independently-seeded models
STAGE_EPOCHS = 25       # epochs per curriculum stage
GRID_RES = 60           # grid resolution (pixels per axis)
EVAL_HEIGHT_M = 100.0   # fixed altitude slice for the grid (m)
OUT_DIR = 'experiments/04_uncertainty_map'
FIG_DIR = 'experiments/figures'


# ── colour palette (matches generate_figures.py) ──────────────────────────
COLORS = {
    'background': '#F6F7F8',
    'grid':       '#E5E7EB',
    'text':       '#1F2937',
}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size']   = 12


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_bias_aware_dataset(sub_df, phys_params):
    """Build a BiasAwarePINNDataset from a sub-DataFrame."""
    return BiasAwarePINNDataset(
        lat=sub_df['avg_latitude'].values,
        lon=sub_df['avg_longitude'].values,
        z=sub_df['avg_altitude'].values,
        t=sub_df['timestamp'].values,
        temperature=sub_df['avg_temperature'].values,
        humidity=sub_df['avg_humidity'].values,
        pressure_bias=sub_df['pressure_bias'].values,
        sensor_id=np.zeros(len(sub_df), dtype=np.int64),
        p_obs=sub_df['avg_pressure'].values,
        h_gnss=sub_df['avg_altitude'].values,
        h_phys=sub_df['h_phys_hae'].values,
    )


def build_model():
    """Instantiate a fresh BiasAwarePINN."""
    base = GeneralizedPressureCorrectionPINN(
        hash_levels=16, hash_features=4, hidden_dim=256,
        n_hidden_layers=3, temporal_freqs=6, use_siren=True
    )
    return BiasAwarePINN(base, bias_dim=8)


def train_one_model(df, phys_params, device, seed=0):
    """Train a single ensemble member on the full dataset with curriculum."""
    set_seed(seed)
    model = build_model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    altitudes = df['avg_altitude'].values
    mask_1, mask_2, mask_3 = create_curriculum_splits(df)

    for stage_idx, mask in enumerate([mask_1, mask_2, mask_3], start=1):
        stage_df = df[mask].copy()
        if len(stage_df) == 0:
            stage_df = df.copy()
        ds = make_bias_aware_dataset(stage_df, phys_params)
        loader = DataLoader(ds, batch_size=2048, shuffle=True)
        print(f"  Seed {seed} Stage {stage_idx}: {len(stage_df)} samples")
        for epoch in range(STAGE_EPOCHS):
            train_epoch(model, loader, optimizer, scheduler, phys_params, device)
        # scheduler is stepped inside train_epoch each epoch; no extra step needed

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Grid inference
# ══════════════════════════════════════════════════════════════════════════════

def predict_grid(models, grid_df, phys_params, device):
    """
    Run all ensemble models on the grid DataFrame.
    Returns arrays (n_models, n_points) of predicted heights in metres.
    """
    all_preds = []

    for model in models:
        model.eval()
        ds = make_bias_aware_dataset(grid_df, phys_params)
        loader = DataLoader(ds, batch_size=4096, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                delta_p = model(
                    batch['lat'].to(device),
                    batch['lon'].to(device),
                    batch['z'].to(device),
                    batch['t'].to(device),
                    batch['temperature'].to(device),
                    batch['humidity'].to(device),
                    batch['pressure_bias'].to(device),
                )
                p_obs = batch['p_obs'].to(device)
                p_corrected = p_obs + delta_p
                t_c = batch['temperature'].to(device)
                e_sat = 610.94 * torch.exp(17.625 * t_c / (t_c + 243.04))
                e = (batch['humidity'].to(device) / 100.0) * e_sat
                r = 0.62198 * e / (p_corrected - e)
                t_v = (t_c + 273.15) * (1.0 + 0.608 * r)
                H = R_DRY_AIR * t_v / G_STANDARD
                h_pred = H * torch.log(
                    torch.tensor(phys_params.p_ref, device=device) / p_corrected
                )
                preds.append(h_pred.cpu().numpy())

        all_preds.append(np.concatenate(preds))

    return np.stack(all_preds)  # (n_models, n_points)


def build_grid_dataframe(df, phys_params, grid_res=GRID_RES,
                          eval_height=EVAL_HEIGHT_M):
    """
    Build a dense grid over the deployment bounding box.
    Uses mean temperature/humidity and mean pressure_bias from training data
    as representative values for each (lat, lon) cell.
    """
    lat_min, lat_max = df['avg_latitude'].min(), df['avg_latitude'].max()
    lon_min, lon_max = df['avg_longitude'].min(), df['avg_longitude'].max()

    # Slight padding
    pad_lat = (lat_max - lat_min) * 0.05
    pad_lon = (lon_max - lon_min) * 0.05
    lats = np.linspace(lat_min - pad_lat, lat_max + pad_lat, grid_res)
    lons = np.linspace(lon_min - pad_lon, lon_max + pad_lon, grid_res)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    n_pts = lat_grid.size

    # Representative atmospheric values (spatial median from training data)
    med_temp = float(df['avg_temperature'].median())
    med_hum  = float(df['avg_humidity'].median())
    med_bias = float(df['pressure_bias'].median())
    med_p    = float(df['avg_pressure'].median())
    med_t    = float(df['timestamp'].median())

    grid_df = pd.DataFrame({
        'avg_latitude':   lat_grid.ravel(),
        'avg_longitude':  lon_grid.ravel(),
        'avg_altitude':   np.full(n_pts, eval_height),
        'avg_temperature': np.full(n_pts, med_temp),
        'avg_humidity':   np.full(n_pts, med_hum),
        'pressure_bias':  np.full(n_pts, med_bias),
        'avg_pressure':   np.full(n_pts, med_p),
        'h_phys_hae':     np.full(n_pts, eval_height),
        'timestamp':      np.full(n_pts, med_t),
    })

    return grid_df, lats, lons


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_ensemble_map(lats, lons, mean_grid, std_grid, sensor_df):
    """Two-panel figure: predicted height field (left) + uncertainty (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor('white')

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    sensor_lats = sensor_df.groupby('uid')['avg_latitude'].mean().values
    sensor_lons = sensor_df.groupby('uid')['avg_longitude'].mean().values

    # ── left panel: predicted height ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(COLORS['background'])
    cf = ax.contourf(lon_mesh, lat_mesh, mean_grid, levels=30, cmap='RdYlBu_r')
    cs = ax.contour(lon_mesh, lat_mesh, mean_grid, levels=10,
                    colors='k', linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f m')
    cbar = plt.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label('Predicted Height (m)', fontsize=12)
    ax.scatter(sensor_lons, sensor_lats, c='black', s=60, zorder=5,
               marker='^', label='Sensor nodes')
    ax.set_xlabel('Longitude (°)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=13, fontweight='bold')
    ax.set_title(f'Predicted Height Field\n(z = {EVAL_HEIGHT_M:.0f} m slice, Ensemble Mean)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])

    # ── right panel: ensemble uncertainty ────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(COLORS['background'])
    cf2 = ax.contourf(lon_mesh, lat_mesh, std_grid, levels=25, cmap='YlOrRd')
    cbar2 = plt.colorbar(cf2, ax=ax, shrink=0.85)
    cbar2.set_label('Uncertainty σ (m)', fontsize=12)
    ax.scatter(sensor_lons, sensor_lats, c='black', s=60, zorder=5,
               marker='^', label='Sensor nodes')
    ax.set_xlabel('Longitude (°)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=13, fontweight='bold')
    ax.set_title(f'Prediction Uncertainty Map\n(Deep Ensemble, N={N_ENSEMBLE} models)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])

    # Format tick labels for lat/lon
    for ax in axes:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f°'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f°'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout(pad=2.0)

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f'06_ensemble_uncertainty_map.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("EXPERIMENT 4: Deep Ensemble Spatial Uncertainty Map")
    print(f"  Ensemble size : {N_ENSEMBLE}")
    print(f"  Grid resolution: {GRID_RES}×{GRID_RES}")
    print(f"  Altitude slice : {EVAL_HEIGHT_M} m")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load & preprocess ────────────────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_csv('data/sensor_data_filtered.csv')
    df, phys_params = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    print(f"  {len(df)} samples, {df['uid'].nunique()} sensors")

    # ── Train ensemble on full dataset ───────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    models = []
    for i in range(N_ENSEMBLE):
        print(f"\n[{i+1}/{N_ENSEMBLE}] Training ensemble member (seed={i*7})...")
        m = train_one_model(df, phys_params, device, seed=i * 7)
        # Optionally cache model
        ckpt_path = os.path.join(OUT_DIR, f'ensemble_model_{i}.pt')
        torch.save(m.state_dict(), ckpt_path)
        models.append(m)
        print(f"  -> Saved: {ckpt_path}")

    # ── Build grid ───────────────────────────────────────────────────────
    print(f"\nBuilding {GRID_RES}×{GRID_RES} grid over deployment zone...")
    grid_df, lats, lons = build_grid_dataframe(df, phys_params)
    print(f"  Bbox: lat [{lats[0]:.5f}, {lats[-1]:.5f}], "
          f"lon [{lons[0]:.5f}, {lons[-1]:.5f}]")

    # ── Predict ──────────────────────────────────────────────────────────
    print("\nRunning ensemble grid inference...")
    preds = predict_grid(models, grid_df, phys_params, device)  # (N, H*W)

    mean_flat = preds.mean(axis=0)
    std_flat  = preds.std(axis=0)

    print(f"  Height mean range : {mean_flat.min():.1f} – {mean_flat.max():.1f} m")
    print(f"  Uncertainty σ mean: {std_flat.mean():.3f} m")
    print(f"  Uncertainty σ max : {std_flat.max():.3f} m")

    # ── Save CSV ─────────────────────────────────────────────────────────
    grid_df['h_mean'] = mean_flat
    grid_df['h_std']  = std_flat
    csv_path = os.path.join(OUT_DIR, 'ensemble_grid.csv')
    grid_df[['avg_latitude', 'avg_longitude', 'h_mean', 'h_std']].to_csv(
        csv_path, index=False
    )
    print(f"\nGrid data saved to: {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    mean_grid = mean_flat.reshape(len(lats), len(lons))
    std_grid  = std_flat.reshape(len(lats), len(lons))

    print("\nGenerating Figure 6: Ensemble Uncertainty Map...")
    plot_ensemble_map(lats, lons, mean_grid, std_grid, df)

    print("\nDone.")


if __name__ == '__main__':
    main()
