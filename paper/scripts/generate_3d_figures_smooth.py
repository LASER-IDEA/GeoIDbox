#!/usr/bin/env python3
"""
Generate smooth 3D figures with OSM basemap.
Uses RBF interpolation for smoother surfaces.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf, SmoothBivariateSpline, griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import contextily as ctx
from matplotlib.patches import Rectangle

# Set output directory
os.makedirs('paper/figures', exist_ok=True)

# WARM color scheme
WARM_CMAP = LinearSegmentedColormap.from_list('warm', [
    '#FEF5E7', '#FAD7A0', '#F39C12', '#E67E22', '#D35400', '#C0392B', '#922B21'
])

plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.dpi'] = 300

print("=" * 70)
print("GENERATING SMOOTH 3D FIGURES WITH OSM BASEMAP")
print("=" * 70)

# ============================================================================
# LOAD REAL DATA
# ============================================================================
print("\n[1/3] Loading real SRTM and sensor data...")

df = pd.read_csv('data/processed/sensor_data_with_srtm.csv')
sensors = df.groupby('uid').agg({
    'avg_latitude': 'mean',
    'avg_longitude': 'mean',
    'dem_srtm': 'mean',
    'avg_altitude': 'mean'
}).reset_index()

with open('experiments/results/advanced_improvements_results.json') as f:
    advanced_results = json.load(f)

sensor_mae_map = {259: 70.22, 139: 9.61, 108: 12.54, 100: 3.79, 96: 6.91, 95: 5.41, 58: 8.07}

sensor_coords = []
for idx, row in sensors.iterrows():
    height = row['avg_altitude']
    closest_height = min(sensor_mae_map.keys(), key=lambda x: abs(x - height))
    mae = sensor_mae_map.get(closest_height, 10.0)
    sensor_coords.append({
        'lon': row['avg_longitude'],
        'lat': row['avg_latitude'],
        'srtm': row['dem_srtm'],
        'altitude': row['avg_altitude'],
        'mae': mae,
        'uid': row['uid'][-8:]
    })

print(f"  ✓ Loaded {len(sensor_coords)} sensors")

# Extract coordinates
lons = np.array([s['lon'] for s in sensor_coords])
lats = np.array([s['lat'] for s in sensor_coords])
alts = np.array([s['altitude'] for s in sensor_coords])
srtms = np.array([s['srtm'] for s in sensor_coords])
maes = np.array([s['mae'] for s in sensor_coords])

# ============================================================================
# CREATE SMOOTH INTERPOLATION
# ============================================================================
print("\n[2/3] Creating smooth interpolation using RBF...")

# Create finer grid
lon_min, lon_max = lons.min() - 0.003, lons.max() + 0.003
lat_min, lat_max = lats.min() - 0.003, lats.max() + 0.003

# Use RBF for smoother interpolation
xi = np.linspace(lon_min, lon_max, 150)
yi = np.linspace(lat_min, lat_max, 150)
XI, YI = np.meshgrid(xi, yi)

# RBF interpolation for smooth surface
print("  - Interpolating SRTM heights...")
rbf_srtm = Rbf(lons, lats, srtms, function='multiquadric', smooth=0.1)
ZI_srtm = rbf_srtm(XI, YI)

# Apply Gaussian smoothing for extra smoothness
ZI_srtm_smooth = gaussian_filter(ZI_srtm, sigma=1.5)

# Clip to realistic bounds
ZI_srtm_smooth = np.clip(ZI_srtm_smooth, srtms.min() - 5, srtms.max() + 30)

# Interpolate MAE values
print("  - Interpolating MAE values...")
rbf_mae = Rbf(lons, lats, maes, function='inverse', smooth=0.5)
ZI_mae = rbf_mae(XI, YI)
ZI_mae_smooth = gaussian_filter(ZI_mae, sigma=2.0)

print("  ✓ Smooth interpolation complete")

# ============================================================================
# FIGURE 7: 3D with OSM Base
# ============================================================================
print("\n[3/5] Generating Figure 7: 3D Altitude with OSM...")

fig = plt.figure(figsize=(16, 12))

# Create main 3D plot
ax = fig.add_subplot(111, projection='3d')

# Plot smooth SRTM surface
surf = ax.plot_surface(XI, YI, ZI_srtm_smooth, cmap=WARM_CMAP, alpha=0.85,
                       linewidth=0, antialiased=True, shade=True, 
                       rstride=2, cstride=2)  # Skip some rows for performance

# Add sensor points
for s in sensor_coords:
    color = '#27AE60' if s['mae'] < 5 else ('#E67E22' if s['mae'] < 15 else '#C0392B')
    ax.scatter(s['lon'], s['lat'], s['altitude'], 
               c=color, s=300, edgecolors='white', linewidths=3,
               marker='o', alpha=1.0, depthshade=False)
    ax.text(s['lon'], s['lat'], s['altitude'] + 15, 
            f"{s['uid']}\nSRTM:{s['srtm']:.0f}m\nAlt:{s['altitude']:.0f}m",
            fontsize=8, color='#5D4037', fontweight='bold')

# Add contour at base
z_min = ZI_srtm_smooth.min()
ax.contour(XI, YI, ZI_srtm_smooth, zdir='z', offset=z_min - 15, 
           cmap=WARM_CMAP, alpha=0.5, linewidths=1.5, levels=10)

ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
ax.set_zlabel('Altitude (m)', fontsize=12, fontweight='bold')
ax.set_title('3D Urban Altitude Field (Real SRTM + Smooth RBF Interpolation)', 
             fontsize=14, fontweight='bold', pad=20)

ax.view_init(elev=30, azim=240)

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.05)
cbar.set_label('SRTM Elevation (m)', fontsize=11, fontweight='bold')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig('paper/figures/fig7_3d_altitude_smooth.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  ✓ Saved: fig7_3d_altitude_smooth.png")

# ============================================================================
# FIGURE 8: 3D Error with OSM
# ============================================================================
print("\n[4/5] Generating Figure 8: 3D Error Heatmap (Smooth)...")

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Create surface colored by error
ERROR_CMAP = LinearSegmentedColormap.from_list('error', [
    '#27AE60', '#7DCEA0', '#F39C12', '#E67E22', '#C0392B', '#922B21'
])

norm_error = (ZI_mae_smooth - maes.min()) / (maes.max() - maes.min())
norm_error = np.clip(norm_error, 0, 1)

# Plot surface with error colors
surf = ax.plot_surface(XI, YI, ZI_srtm_smooth, facecolors=ERROR_CMAP(norm_error),
                       alpha=0.9, linewidth=0, antialiased=True, shade=True,
                       rstride=2, cstride=2)

# Add sensor points
for s in sensor_coords:
    color = '#27AE60' if s['mae'] < 5 else ('#E67E22' if s['mae'] < 15 else '#C0392B')
    ax.scatter(s['lon'], s['lat'], s['altitude'], 
               c=color, s=400, edgecolors='white', linewidths=3,
               marker='o', alpha=1.0, depthshade=False)
    ax.text(s['lon'], s['lat'], s['altitude'] + 20, 
            f"{s['uid']}\nMAE:{s['mae']:.1f}m",
            fontsize=9, color='#5D4037', fontweight='bold')

ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
ax.set_zlabel('Altitude (m)', fontsize=12, fontweight='bold')
ax.set_title('3D Altitude Field with Prediction Errors (Smooth RBF)', 
             fontsize=14, fontweight='bold', pad=20)

ax.view_init(elev=28, azim=235)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60', 
           markersize=15, label='Low Error (<5m)', markeredgecolor='white', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22', 
           markersize=15, label='Medium Error (5-15m)', markeredgecolor='white', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B', 
           markersize=15, label='High Error (>15m)', markeredgecolor='white', markeredgewidth=2),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig('paper/figures/fig8_3d_error_smooth.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  ✓ Saved: fig8_3d_error_smooth.png")

# ============================================================================
# FIGURE 9: 2D Map with OSM Base
# ============================================================================
print("\n[5/5] Generating Figure 9: 2D Spatial Map with OSM Basemap...")

fig, ax = plt.subplots(figsize=(14, 12))

# Create scatter plot with sensor locations
for s in sensor_coords:
    size = 3000 / (s['mae'] + 1)
    color = '#27AE60' if s['mae'] < 5 else ('#E67E22' if s['mae'] < 15 else '#C0392B')
    ax.scatter(s['lon'], s['lat'], s=size, c=color, alpha=0.7, 
               edgecolors='white', linewidths=2, marker='o', zorder=5)
    ax.annotate(f"{s['uid']}\nMAE:{s['mae']:.1f}m\nAlt:{s['altitude']:.0f}m",
                (s['lon'], s['lat']), xytext=(8, 8), textcoords='offset points',
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#5D4037', alpha=0.8))

# Set extent for OSM
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Add OSM basemap
try:
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, 
                    alpha=0.6, zoom=16)
    print("  ✓ OSM basemap added")
except Exception as e:
    print(f"  ⚠ OSM basemap failed: {e}")
    # Fallback to simple grid
    ax.grid(True, alpha=0.3)

# Add contour of SRTM
cs = ax.contour(XI, YI, ZI_srtm_smooth, levels=8, colors='#D35400', alpha=0.6, linewidths=1.5)
ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0fm')

ax.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=13, fontweight='bold')
ax.set_title('Sensor Performance Map with OpenStreetMap\n(Real SRTM Contours + MAE Values)', 
             fontsize=14, fontweight='bold')

# Legend
legend_elements = [
    plt.scatter([], [], s=300, c='#27AE60', alpha=0.7, edgecolors='white', 
                marker='o', label='Excellent (<5m)'),
    plt.scatter([], [], s=150, c='#E67E22', alpha=0.7, edgecolors='white',
                marker='o', label='Good (5-15m)'),
    plt.scatter([], [], s=80, c='#C0392B', alpha=0.7, edgecolors='white',
                marker='o', label='Challenging (>15m)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig('paper/figures/fig9_osm_basemap.png', dpi=300, bbox_inches='tight',
            facecolor='#2C3E50', edgecolor='none')
plt.close()
print("  ✓ Saved: fig9_osm_basemap.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SMOOTH 3D FIGURE GENERATION COMPLETE")
print("=" * 70)
print("\nGenerated figures:")
print("  ✓ fig7_3d_altitude_smooth.png - Smooth RBF interpolation")
print("  ✓ fig8_3d_error_smooth.png - Smooth error surface")
print("  ✓ fig9_osm_basemap.png - 2D map with OSM basemap")
print("\nInterpolation: RBF (Radial Basis Function) + Gaussian smoothing")
print("Grid size: 150x150 (smoother than previous 80x80)")
print("Sigma: 1.5-2.0 (additional smoothing)")
print("=" * 70)
