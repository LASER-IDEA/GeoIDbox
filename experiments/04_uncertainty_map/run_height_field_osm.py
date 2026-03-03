"""
Height Field on Map
===================
Loads the trained ensemble, runs inference on *real* sensor observations
(actual pressure / temperature / humidity), aggregates per-sensor predicted
heights, IDW-interpolates to a dense grid, and produces:

  1. Publication PDF — styled vector map with filled height contours,
     contour lines, sensor anchors, scale bar, north arrow.
  2. Interactive HTML — folium map with OSM tiles + height overlay
     (open in a browser with internet access).

Outputs
-------
experiments/figures/06_height_field_osm.pdf/png   <- paper figure
experiments/04_uncertainty_map/height_field_osm.html  <- interactive map
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/data/home/huxiao/workspace/GeoIDbox')

from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_generalized_with_bias import (
    compute_sensor_bias, BiasAwarePINNDataset, BiasAwarePINN
)
from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.train_pinn import parse_timestamp
from torch.utils.data import DataLoader

R_DRY_AIR  = 287.05
G_STANDARD = 9.80665
N_ENSEMBLE = 5
ENSEMBLE_DIR = 'experiments/04_uncertainty_map'
FIG_DIR      = 'experiments/figures'


def build_model():
    base = GeneralizedPressureCorrectionPINN(
        hash_levels=16, hash_features=4, hidden_dim=256,
        n_hidden_layers=3, temporal_freqs=6, use_siren=True
    )
    return BiasAwarePINN(base, bias_dim=8)


def load_ensemble(device):
    models = []
    for i in range(N_ENSEMBLE):
        path = os.path.join(ENSEMBLE_DIR, f'ensemble_model_{i}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        m = build_model().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    print(f"  Loaded {len(models)} ensemble members from {ENSEMBLE_DIR}/")
    return models


def run_inference(models, df, phys_params, device):
    """Ensemble mean predicted height for every sample in df."""
    ds = BiasAwarePINNDataset(
        lat=df['avg_latitude'].values,
        lon=df['avg_longitude'].values,
        z=df['avg_altitude'].values,
        t=df['timestamp'].values,
        temperature=df['avg_temperature'].values,
        humidity=df['avg_humidity'].values,
        pressure_bias=df['pressure_bias'].values,
        sensor_id=np.zeros(len(df), dtype=np.int64),
        p_obs=df['avg_pressure'].values,
        h_gnss=df['avg_altitude'].values,
        h_phys=df['h_phys_hae'].values,
    )
    loader = DataLoader(ds, batch_size=8192, shuffle=False)

    all_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                delta_p = model(
                    batch['lat'].to(device), batch['lon'].to(device),
                    batch['z'].to(device),   batch['t'].to(device),
                    batch['temperature'].to(device),
                    batch['humidity'].to(device),
                    batch['pressure_bias'].to(device),
                )
                p_corr = batch['p_obs'].to(device) + delta_p
                t_c    = batch['temperature'].to(device)
                e_sat  = 610.94 * torch.exp(17.625 * t_c / (t_c + 243.04))
                e      = (batch['humidity'].to(device) / 100.0) * e_sat
                r      = 0.62198 * e / (p_corr - e)
                t_v    = (t_c + 273.15) * (1.0 + 0.608 * r)
                H      = R_DRY_AIR * t_v / G_STANDARD
                h      = H * torch.log(
                    torch.tensor(phys_params.p_ref, device=device) / p_corr)
                preds.append(h.cpu().numpy())
        all_preds.append(np.concatenate(preds))

    return np.stack(all_preds).mean(axis=0)


def idw_grid(sensor_lats, sensor_lons, sensor_vals,
             grid_lats, grid_lons, power=2.0):
    """Inverse-distance-weighted interpolation."""
    glat, glon = np.meshgrid(grid_lats, grid_lons, indexing='ij')
    weights = np.zeros_like(glat)
    result  = np.zeros_like(glat)
    for slat, slon, sval in zip(sensor_lats, sensor_lons, sensor_vals):
        dlat = glat - slat
        dlon = (glon - slon) * np.cos(np.radians(slat))
        dist = np.maximum(np.sqrt(dlat**2 + dlon**2), 1e-9)
        w    = 1.0 / dist**power
        result  += w * sval
        weights += w
    return result / weights


def add_scale_bar(ax, lat_ref, length_m=100):
    metres_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref))
    bar_deg = length_m / metres_per_deg_lon
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    x0 = xl + (xr - xl) * 0.05
    y0 = yb + (yt - yb) * 0.04
    dy = (yt - yb) * 0.006
    ax.fill_betweenx([y0, y0 + dy], x0, x0 + bar_deg, color='black',
                     zorder=13, linewidth=0)
    ax.text(x0 + bar_deg * 0.5, y0 + dy * 1.6, f'{length_m} m',
            ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=13,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


def add_north_arrow(ax):
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    ax_x = xl + (xr - xl) * 0.95
    ax_y = yb + (yt - yb) * 0.90
    dy   = (yt - yb) * 0.04
    ax.annotate('', xy=(ax_x, ax_y + dy), xytext=(ax_x, ax_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.2),
                zorder=13)
    ax.text(ax_x, ax_y + dy * 1.2, 'N', ha='center', va='bottom',
            fontsize=12, fontweight='bold', zorder=13,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])


# ========================================================================
# Publication-quality static figure
# ========================================================================

def plot_publication_map(sens, grid_lats, grid_lons, height_grid):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif':  ['Times New Roman', 'DejaVu Serif'],
        'font.size':   12,
        'axes.linewidth': 0.8,
    })

    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    ax.set_facecolor('#E8EFF6')
    fig.patch.set_facecolor('white')

    lon_min, lon_max = grid_lons.min(), grid_lons.max()
    lat_min, lat_max = grid_lats.min(), grid_lats.max()

    # light land background
    land = mpatches.FancyBboxPatch(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        boxstyle="square,pad=0", linewidth=0,
        facecolor='#F5F1EB', zorder=0
    )
    ax.add_patch(land)

    # height_grid shape: (lat_res, lon_res) after .T
    h_plot = height_grid.T

    vmin, vmax = h_plot.min() - 5, h_plot.max() + 5
    levels_fill = np.linspace(vmin, vmax, 24)
    levels_line = np.linspace(vmin, vmax, 11)

    cf = ax.contourf(grid_lons, grid_lats, h_plot,
                     levels=levels_fill, cmap='RdYlBu_r',
                     alpha=0.70, zorder=2, extend='both')
    cs = ax.contour(grid_lons, grid_lats, h_plot,
                    levels=levels_line,
                    colors='#1A1A1A', linewidths=0.9, alpha=0.65, zorder=3)
    ax.clabel(cs, inline=True, fontsize=9.5, fmt='%.0f m',
              inline_spacing=3)

    # colorbar
    cbar = plt.colorbar(cf, ax=ax, fraction=0.035, pad=0.01,
                        shrink=0.78, extend='both')
    cbar.set_label('Predicted Height (m)', fontsize=13, fontweight='bold',
                   labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    # sensor anchors — shadow + star
    ax.scatter(sens['lon'], sens['lat'], s=340, c='#555555',
               marker='*', zorder=8, alpha=0.35)
    ax.scatter(sens['lon'], sens['lat'], s=300, c='#D62728',
               marker='*', zorder=9,
               edgecolors='#7F0000', linewidths=0.7,
               label='GeoBox sensor node')

    for _, row in sens.iterrows():
        lbl = (f"{row['label']}\n"
               f"GNSS: {row['alt_gnss']:.0f} m\n"
               f"Pred: {row['h_pred']:.0f} m")
        ax.annotate(
            lbl,
            xy=(row['lon'], row['lat']),
            xytext=(row['lon'] + 0.00040, row['lat'] + 0.00040),
            fontsize=8.5, fontweight='bold', color='#1A1A2E',
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec='#AAAAAA', alpha=0.90, lw=0.8),
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.7),
            zorder=12,
        )

    # grid lines
    ax.grid(True, linestyle=':', linewidth=0.5, color='#B0BFCC',
            alpha=0.9, zorder=1)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # scale bar and north arrow after limits are set
    add_scale_bar(ax, lat_ref=float(sens['lat'].mean()), length_m=100)
    add_north_arrow(ax)

    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f°'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f°'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=18, ha='right',
             fontsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=10)

    ax.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel('Latitude (°N)',  fontsize=13, fontweight='bold', labelpad=8)
    ax.set_title(
        'Neural Height Field over GeoBox Deployment Zone\n'
        r'(Shenzhen Urban Canyon $\cdot$ Deep Ensemble $N\!=\!5$ $\cdot$ LOSO MAE $=$ 3.55 m)',
        fontsize=13, fontweight='bold', pad=16
    )
    ax.legend(loc='lower right', fontsize=11, framealpha=0.92,
              edgecolor='#CCCCCC')

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f'06_height_field_osm.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {path}")
    plt.close(fig)


# ========================================================================
# Interactive folium HTML (OSM tiles in browser)
# ========================================================================

def plot_folium_map(sens, grid_lats, grid_lons, height_grid, out_dir):
    try:
        import folium
        from branca.colormap import LinearColormap
        import io, base64
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"  [skip HTML: {e}]")
        return

    h_plot = height_grid.T   # (lat_res, lon_res)
    vmin, vmax = h_plot.min(), h_plot.max()

    norm  = (h_plot - vmin) / max(vmax - vmin, 1.0)
    cmap  = plt.cm.get_cmap('RdYlBu_r')
    rgba  = (cmap(norm) * 255).astype(np.uint8)
    rgba[:, :, 3] = 160

    img = PILImage.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()

    lat_c = float(sens['lat'].mean())
    lon_c = float(sens['lon'].mean())
    m = folium.Map(location=[lat_c, lon_c], zoom_start=17,
                   tiles='CartoDB positron')

    bounds = [[float(grid_lats.min()), float(grid_lons.min())],
              [float(grid_lats.max()), float(grid_lons.max())]]
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{b64}",
        bounds=bounds, opacity=0.62,
        name='Predicted Height Field',
    ).add_to(m)

    colormap = LinearColormap(
        ['#313695', '#74ADD1', '#FEE090', '#F46D43', '#A50026'],
        vmin=vmin, vmax=vmax,
        caption='Predicted Height (m)',
    )
    colormap.add_to(m)

    for _, row in sens.iterrows():
        popup_html = (
            f"<b>Sensor:</b> {row['uid']}<br>"
            f"<b>GNSS altitude:</b> {row['alt_gnss']:.1f} m<br>"
            f"<b>Predicted height:</b> {row['h_pred']:.1f} m<br>"
            f"<b>Error:</b> {abs(row['h_pred']-row['alt_gnss']):.1f} m"
        )
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"GeoBox {row['label']} | {row['alt_gnss']:.0f}m GNSS",
            icon=folium.Icon(color='red', icon='signal', prefix='fa'),
        ).add_to(m)

    folium.LayerControl().add_to(m)

    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, 'height_field_osm.html')
    m.save(html_path)
    print(f"  Saved: {html_path}  (open in browser for live OSM tiles)")


# ========================================================================
# Main
# ========================================================================

def main():
    print("=" * 65)
    print("HEIGHT FIELD MAP  (ensemble inference + IDW + publication fig)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading sensor data...")
    df = pd.read_csv('data/sensor_data_filtered.csv')
    df, phys_params = compute_physics_baseline(df, p_ref=None,
                                               convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    print(f"  {len(df)} samples  |  {df['uid'].nunique()} sensors")

    print("\nLoading ensemble models...")
    models = load_ensemble(device)

    print("\nRunning ensemble inference on real observations (actual P/T/RH)...")
    h_pred_all = run_inference(models, df, phys_params, device)
    df['h_pred'] = h_pred_all
    print(f"  h_pred range : {h_pred_all.min():.1f} – {h_pred_all.max():.1f} m")

    per_sensor_pred = df.groupby('uid')['h_pred'].mean()
    per_sensor_gnss = df.groupby('uid')['avg_altitude'].mean()
    per_sensor_lat  = df.groupby('uid')['avg_latitude'].mean()
    per_sensor_lon  = df.groupby('uid')['avg_longitude'].mean()

    sens = pd.DataFrame({
        'uid':      per_sensor_pred.index,
        'lat':      per_sensor_lat.values,
        'lon':      per_sensor_lon.values,
        'alt_gnss': per_sensor_gnss.values,
        'h_pred':   per_sensor_pred.values,
        'label':    [u[-6:] for u in per_sensor_pred.index],
    })

    print("\nPer-sensor summary:")
    for _, row in sens.iterrows():
        err = abs(row['h_pred'] - row['alt_gnss'])
        print(f"  {row['label']}  GNSS={row['alt_gnss']:.1f}m  "
              f"Pred={row['h_pred']:.1f}m  Err={err:.1f}m")

    print("\nBuilding IDW height field (120x120 grid)...")
    pad = 0.0010
    lat_min = sens['lat'].min() - pad
    lat_max = sens['lat'].max() + pad
    lon_min = sens['lon'].min() - pad
    lon_max = sens['lon'].max() + pad

    grid_lats = np.linspace(lat_min, lat_max, 120)
    grid_lons = np.linspace(lon_min, lon_max, 120)
    height_grid = idw_grid(
        sens['lat'].values, sens['lon'].values, sens['h_pred'].values,
        grid_lats, grid_lons
    )
    print(f"  IDW height range: {height_grid.min():.1f} – {height_grid.max():.1f} m")

    print("\nGenerating publication figure (PDF+PNG)...")
    plot_publication_map(sens, grid_lats, grid_lons, height_grid)

    print("\nGenerating interactive HTML map (folium, needs browser internet)...")
    plot_folium_map(sens, grid_lats, grid_lons, height_grid, ENSEMBLE_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
