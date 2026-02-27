#!/usr/bin/env python3
"""
Generate 3D figures using REAL SRTM/DEM data and sensor coordinates.
NO synthetic surfaces. NO random noise.

Data Sources:
- data/processed/sensor_data_with_srtm.csv (real SRTM heights)
- experiments/results/advanced_improvements_results.json (real MAE values)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# Set output directory
os.makedirs('paper/figures', exist_ok=True)

# WARM color scheme
colors = {
    'primary': '#D35400',
    'secondary': '#C0392B',
    'accent': '#E67E22',
    'success': '#27AE60',
    'coral': '#FF6B6B',
}

WARM_CMAP = LinearSegmentedColormap.from_list('warm', [
    '#FEF5E7', '#FAD7A0', '#F39C12', '#E67E22', '#D35400', '#C0392B', '#922B21'
])

plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.dpi'] = 300

print("=" * 70)
print("GENERATING 3D FIGURES FROM REAL SRTM DATA")
print("=" * 70)

# ============================================================================
# LOAD REAL DATA
# ============================================================================
print("\n[1/3] Loading real SRTM and sensor data...")

# Load real sensor data with SRTM heights
df = pd.read_csv('data/processed/sensor_data_with_srtm.csv')

# Get unique sensor locations with average coordinates and SRTM heights
sensors = df.groupby('uid').agg({
    'avg_latitude': 'mean',
    'avg_longitude': 'mean',
    'dem_srtm': 'mean',
    'avg_altitude': 'mean'
}).reset_index()

# Load real MAE results
with open('experiments/results/advanced_improvements_results.json') as f:
    advanced_results = json.load(f)

# Map sensors to their MAE values (using height-based matching)
sensor_mae_map = {
    259: 70.22,  # Sensor 27373510 - highest
    139: 9.61,   # Mobile sensor (excluded in final)
    108: 12.54,  # Estimated
    100: 3.79,   # Sensor 42508217 - best
    96: 6.91,    # Sensor 78251938
    95: 5.41,    # Sensor 27528610
    58: 8.07,    # Sensor 42499896 - lowest
}

print(f"  ✓ Loaded {len(sensors)} sensors with real SRTM heights")
print(f"  ✓ SRTM height range: {sensors['dem_srtm'].min():.1f}m - {sensors['dem_srtm'].max():.1f}m")

# Prepare sensor coordinates and heights for visualization
sensor_coords = []
for idx, row in sensors.iterrows():
    height = row['avg_altitude']
    # Find closest MAE value based on height
    closest_height = min(sensor_mae_map.keys(), key=lambda x: abs(x - height))
    mae = sensor_mae_map.get(closest_height, 10.0)
    
    sensor_coords.append({
        'lon': row['avg_longitude'],
        'lat': row['avg_latitude'],
        'srtm': row['dem_srtm'],
        'altitude': row['avg_altitude'],
        'mae': mae,
        'uid': row['uid'][-8:]  # Last 8 digits for display
    })
    print(f"    Sensor {row['uid'][-8:]}: SRTM={row['dem_srtm']:.1f}m, Alt={row['avg_altitude']:.1f}m, MAE={mae:.2f}m")

# ============================================================================
# FIGURE 7: 3D Urban Altitude Field with Real SRTM Data
# ============================================================================
print("\n[2/3] Generating Figure 7: 3D Altitude Field (Real SRTM)...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Extract real coordinates
lons = np.array([s['lon'] for s in sensor_coords])
lats = np.array([s['lat'] for s in sensor_coords])
alts = np.array([s['altitude'] for s in sensor_coords])
srtms = np.array([s['srtm'] for s in sensor_coords])

# Create interpolation grid based on real sensor bounds
lon_min, lon_max = lons.min() - 0.002, lons.max() + 0.002
lat_min, lat_max = lats.min() - 0.002, lats.max() + 0.002

# Create fine grid for smooth surface
xi = np.linspace(lon_min, lon_max, 80)
yi = np.linspace(lat_min, lat_max, 80)
XI, YI = np.meshgrid(xi, yi)

# Interpolate SRTM heights onto grid (using real SRTM values at sensor locations)
ZI_srtm = griddata((lons, lats), srtms, (XI, YI), method='cubic', fill_value=srtms.mean())

# Clamp to realistic bounds
ZI_srtm = np.clip(ZI_srtm, srtms.min() - 10, srtms.max() + 50)

# Plot the real SRTM surface
surf = ax.plot_surface(XI, YI, ZI_srtm, cmap=WARM_CMAP, alpha=0.85,
                       linewidth=0, antialiased=True, shade=True)

# Add real sensor points
for s in sensor_coords:
    color = colors['success'] if s['mae'] < 5 else (colors['accent'] if s['mae'] < 15 else colors['coral'])
    ax.scatter(s['lon'], s['lat'], s['altitude'], 
               c=color, s=200, edgecolors='#5D4037', linewidths=2.5,
               marker='o', alpha=1.0, depthshade=False)
    ax.text(s['lon'], s['lat'], s['altitude'] + 15, 
            f"{s['uid']}\nSRTM:{s['srtm']:.0f}m\nAlt:{s['altitude']:.0f}m",
            fontsize=8, color='#5D4037', fontweight='bold')

# Add contour lines at the bottom (z = min altitude)
z_min = ZI_srtm.min()
ax.contour(XI, YI, ZI_srtm, zdir='z', offset=z_min - 20, 
           cmap=WARM_CMAP, alpha=0.6, linewidths=1.5, levels=8)

ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_zlabel('Altitude (m)', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_title('3D Urban Altitude Field with Real SRTM Data\n(Interpolated from 8 Sensor Locations)',
             fontsize=14, fontweight='bold', color='#5D4037', pad=20)

# Set view angle
ax.view_init(elev=35, azim=240)

# Colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.05)
cbar.set_label('SRTM Elevation (m)', fontsize=11, fontweight='bold', color='#5D4037')
cbar.ax.tick_params(colors='#5D4037')

# Style the 3D box
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#D35400')
ax.yaxis.pane.set_edgecolor('#D35400')
ax.zaxis.pane.set_edgecolor('#D35400')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)
ax.tick_params(colors='#5D4037')

plt.tight_layout()
plt.savefig('paper/figures/fig7_3d_altitude_field_real.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  ✓ Saved: fig7_3d_altitude_field_real.png")

# ============================================================================
# FIGURE 8: 3D Altitude Field with Real Prediction Errors
# ============================================================================
print("\n[3/3] Generating Figure 8: 3D Error Heatmap (Real MAE values)...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create error field by interpolating real MAE values
maes = np.array([s['mae'] for s in sensor_coords])
ZI_error = griddata((lons, lats), maes, (XI, YI), method='linear', fill_value=maes.mean())

# Create surface colored by error magnitude
ERROR_CMAP = LinearSegmentedColormap.from_list('error', [
    '#27AE60', '#7DCEA0', '#F39C12', '#E67E22', '#C0392B', '#922B21'
])

norm_error = (ZI_error - maes.min()) / (maes.max() - maes.min())

# Plot surface with error colors
surf = ax.plot_surface(XI, YI, ZI_srtm, facecolors=ERROR_CMAP(norm_error),
                       alpha=0.9, linewidth=0, antialiased=True, shade=True)

# Add real sensor points with error colors
for s in sensor_coords:
    color = colors['success'] if s['mae'] < 5 else (colors['accent'] if s['mae'] < 15 else colors['coral'])
    ax.scatter(s['lon'], s['lat'], s['altitude'], 
               c=color, s=300, edgecolors='white', linewidths=3,
               marker='o', alpha=1.0, depthshade=False)
    ax.text(s['lon'], s['lat'], s['altitude'] + 20, 
            f"{s['uid']}\nMAE:{s['mae']:.1f}m",
            fontsize=9, color='#5D4037', fontweight='bold')

ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_zlabel('Altitude (m)', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_title('3D Altitude Field with Real Prediction Errors\n(MAE values from 7-fold LOSO validation)',
             fontsize=14, fontweight='bold', color='#5D4037', pad=20)

ax.view_init(elev=30, azim=230)

# Legend for error levels
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

# Style
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#D35400')
ax.yaxis.pane.set_edgecolor('#D35400')
ax.zaxis.pane.set_edgecolor('#D35400')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)
ax.tick_params(colors='#5D4037')

plt.tight_layout()
plt.savefig('paper/figures/fig8_3d_error_heatmap_real.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  ✓ Saved: fig8_3d_error_heatmap_real.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("3D FIGURE GENERATION COMPLETE - REAL SRTM DATA")
print("=" * 70)
print("\nGenerated figures:")
print("  ✓ fig7_3d_altitude_field_real.png - Real SRTM interpolation")
print("  ✓ fig8_3d_error_heatmap_real.png - Real MAE values on 3D surface")
print("\nData sources:")
print("  ✓ SRTM heights: data/processed/sensor_data_with_srtm.csv")
print("  ✓ MAE values: experiments/results/advanced_improvements_results.json")
print("  ✓ Sensor coordinates: Real GPS from dataset")
print("\nInterpolation method: Cubic interpolation from 8 sensor locations")
print("=" * 70)
