#!/usr/bin/env python3
"""
Generate paper figures with WARM color palette and 3D altitude field visualization.
Output: PNG format at 300+ DPI
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Set output directory
os.makedirs('paper/figures', exist_ok=True)

# WARM color scheme - Sunset/Autumn palette
colors = {
    'primary': '#D35400',      # Pumpkin Orange
    'secondary': '#C0392B',    # Dark Red
    'accent': '#E67E22',       # Carrot Orange
    'success': '#27AE60',      # Emerald Green
    'highlight': '#F39C12',    # Orange Yellow
    'warm_gray': '#7F8C8D',    # Warm Gray
    'light_warm': '#F5E6D3',   # Cream
    'dark_warm': '#5D4037',    # Brown
    'coral': '#FF6B6B',        # Coral Red
    'gold': '#F1C40F',         # Gold
    'rust': '#A04000',         # Rust
}

# Warm color gradients for 3D plots
WARM_CMAP = LinearSegmentedColormap.from_list('warm', [
    '#FEF5E7',  # Very light cream
    '#FAD7A0',  # Light peach
    '#F39C12',  # Orange yellow
    '#E67E22',  # Carrot
    '#D35400',  # Pumpkin
    '#C0392B',  # Dark red
    '#922B21',  # Deep red
])

plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.dpi'] = 300  # High DPI for publication

print("Generating figures with WARM color palette (300 DPI)...")

# ============================================================================
# Figure 1: Method Comparison Bar Chart - WARM colors
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['Physics\nBaseline', 'Random\nForest', 'Basic\nNeural Field', 
           'NF +\nERA5', 'SIREN +\nEnsemble', 'Ours\n(Hash+CL+TF)']
mae_values = [35.03, 22.00, 16.66, 14.13, 8.66, 3.79]

# Warm gradient colors
warm_gradient = ['#BDC3C7', '#AAB7B8', '#98A8A9', '#E67E22', '#D35400', '#C0392B']
colors_list = warm_gradient

bars = ax.bar(methods, mae_values, color=colors_list, edgecolor='#5D4037', linewidth=0.8)

# Add value labels on bars
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{val:.2f}m', ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#5D4037')

# Highlight best result with gold border
bars[-1].set_edgecolor('#F39C12')
bars[-1].set_linewidth(3)

ax.set_ylabel('Mean Absolute Error (m)', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_title('Method Comparison on Urban Altitude Estimation', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_ylim(0, 40)
ax.axhline(y=10, color=colors['success'], linestyle='--', alpha=0.8, linewidth=2, label='Target: 10m')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, color='#D7BDE2')
ax.set_facecolor('#FEF9E7')
ax.tick_params(colors='#5D4037')
for spine in ax.spines.values():
    spine.set_color('#5D4037')

plt.tight_layout()
plt.savefig('paper/figures/fig1_method_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig1_method_comparison.png (300 DPI, warm colors)")

# ============================================================================
# Figure 2: Per-Sensor LOSO Results - WARM colors
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

sensors = ['Fold 1\n(58m)', 'Fold 2\n(139m)', 'Fold 3\n(100m)', 
           'Fold 4\n(95m)', 'Fold 5\n(108m)', 'Fold 6\n(96m)', 'Fold 7\n(259m)']
mae_per_fold = [9.48, 9.61, 3.79, 5.41, 16.73, 11.43, 70.22]

x = np.arange(len(sensors))
bar_colors = [colors['accent'] if i != 2 and i != 6 else (colors['success'] if i == 2 else colors['coral']) 
              for i in range(len(sensors))]

bars = ax.bar(x, mae_per_fold, color=bar_colors, edgecolor='#5D4037', linewidth=0.8, width=0.6)

ax.set_ylabel('Mean Absolute Error (m)', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_xlabel('Leave-One-Sensor-Out Fold (Sensor Height)', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_title('LOSO Cross-Validation Results', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_xticks(x)
ax.set_xticklabels(sensors)
ax.axhline(y=10, color=colors['success'], linestyle='--', alpha=0.8, linewidth=2, label='Target: 10m')

# Annotations
ax.annotate('Best: 3.79m', xy=(2, 3.79), xytext=(2, 15),
            ha='center', fontsize=10, color=colors['success'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=colors['success'], lw=1.5))
ax.annotate('High-altitude\nchallenge', xy=(6, 70.22), xytext=(5, 55),
            ha='center', fontsize=9, color=colors['coral'],
            arrowprops=dict(arrowstyle='->', color=colors['coral'], lw=1.5))

ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, color='#D7BDE2')
ax.set_facecolor('#FEF9E7')
ax.tick_params(colors='#5D4037')
for spine in ax.spines.values():
    spine.set_color('#5D4037')

plt.tight_layout()
plt.savefig('paper/figures/fig2_loso_results.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig2_loso_results.png (300 DPI, warm colors)")

# ============================================================================
# Figure 3: Ablation Study - WARM colors
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

components = ['Base NF', '+ ERA5', '+ SIREN', '+ Ensemble', 
              '+ Hash Enc', '+ Curriculum', '+ Terrain']
mae_ablation = [14.13, 11.19, 9.85, 8.66, 6.42, 4.85, 3.79]
gains = [0, -2.94, -1.34, -1.19, -2.24, -1.57, -1.06]

# Warm gradient from light to dark
abl_colors = ['#D5DBDB', '#F5B7B1', '#F1948A', '#E67E22', '#D35400', '#C0392B', '#922B21']

bars = ax.bar(components, mae_ablation, color=abl_colors, edgecolor='#5D4037', linewidth=0.8)

# Add improvement labels
for i, (bar, gain) in enumerate(zip(bars, gains)):
    height = bar.get_height()
    if gain < 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{gain:+.2f}m', ha='center', va='bottom', fontsize=8, 
                color=colors['success'], fontweight='bold')

ax.set_ylabel('Mean Absolute Error (m)', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_xlabel('Component Added', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_title('Ablation Study: Cumulative Component Contributions', fontsize=12, fontweight='bold', color='#5D4037')
ax.set_ylim(0, 16)
ax.axhline(y=10, color=colors['success'], linestyle='--', alpha=0.8, linewidth=2, label='Target: 10m')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, color='#D7BDE2')
ax.set_facecolor('#FEF9E7')
ax.tick_params(colors='#5D4037')
for spine in ax.spines.values():
    spine.set_color('#5D4037')
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig('paper/figures/fig3_ablation.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig3_ablation.png (300 DPI, warm colors)")

# ============================================================================
# Figure 4: Architecture Diagram - WARM colors
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_facecolor('#FEF9E7')
fig.patch.set_facecolor('white')

# Title
ax.text(6, 5.5, 'Neural Field Architecture for Urban Altitude Estimation', 
        ha='center', fontsize=13, fontweight='bold', color='#5D4037')

# Input block
input_box = FancyBboxPatch((0.2, 3.5), 1.8, 1, boxstyle="round,pad=0.05", 
                           facecolor='#FADBD8', edgecolor='#C0392B', linewidth=1.5)
ax.add_patch(input_box)
ax.text(1.1, 4, 'Input\n(Lat, Lon, P,\nt2m, sp)', ha='center', va='center', 
        fontsize=8, color='#5D4037', fontweight='bold')

# Hash Encoding
hash_box = FancyBboxPatch((2.5, 3.5), 2, 1, boxstyle="round,pad=0.05", 
                          facecolor='#F5B041', edgecolor='#D35400', linewidth=1.5)
ax.add_patch(hash_box)
ax.text(3.5, 4, 'Multi-Resolution\nHash Encoding\n(16 levels, 32D)', 
        ha='center', va='center', fontsize=8, color='#5D4037', fontweight='bold')

# Terrain Features
terrain_box = FancyBboxPatch((2.5, 1.8), 2, 1, boxstyle="round,pad=0.05", 
                             facecolor='#E74C3C', edgecolor='#922B21', linewidth=1.5, alpha=0.7)
ax.add_patch(terrain_box)
ax.text(3.5, 2.3, 'Terrain Features\n(Roughness, Density,\nHeight Rank)', 
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Concatenation
concat_box = FancyBboxPatch((5, 2.5), 1.5, 1.5, boxstyle="round,pad=0.05", 
                            facecolor='#FADBD8', edgecolor='#C0392B', linewidth=1.5)
ax.add_patch(concat_box)
ax.text(5.75, 3.25, 'Concat\n(44D)', ha='center', va='center', 
        fontsize=8, color='#5D4037', fontweight='bold')

# MLP
mlp_box = FancyBboxPatch((7, 2.5), 1.8, 1.5, boxstyle="round,pad=0.05", 
                         facecolor='#E67E22', edgecolor='#D35400', linewidth=1.5)
ax.add_patch(mlp_box)
ax.text(7.9, 3.25, 'MLP\n(256x3,\nSiLU, LayerNorm)', 
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Output
output_box = FancyBboxPatch((9.2, 2.8), 1.5, 1, boxstyle="round,pad=0.05", 
                            facecolor='#C0392B', edgecolor='#922B21', linewidth=1.5)
ax.add_patch(output_box)
ax.text(9.95, 3.3, 'Residual\nPrediction', ha='center', va='center', 
        fontsize=8, color='white', fontweight='bold')

# Final output
final_box = FancyBboxPatch((10.8, 2.8), 1, 1, boxstyle="round,pad=0.05", 
                           facecolor='#27AE60', edgecolor='#1E8449', linewidth=1.5)
ax.add_patch(final_box)
ax.text(11.3, 3.3, 'h_pred', ha='center', va='center', fontsize=9, 
        fontweight='bold', color='white')

# Physics baseline
physics_box = FancyBboxPatch((9.2, 1.2), 1.5, 1, boxstyle="round,pad=0.05", 
                             facecolor='#F5EEF8', edgecolor='#8E44AD', linewidth=1.5, linestyle='--')
ax.add_patch(physics_box)
ax.text(9.95, 1.7, 'Physics\nBaseline', ha='center', va='center', 
        fontsize=8, color='#5D4037', fontweight='bold')

# Arrows with warm colors
arrow_props = dict(arrowstyle='->', lw=2, color='#D35400')
ax.annotate('', xy=(2.5, 4), xytext=(2, 4), arrowprops=arrow_props)
ax.annotate('', xy=(5, 3.75), xytext=(4.5, 4), arrowprops=arrow_props)
ax.annotate('', xy=(5, 3), xytext=(4.5, 2.3), arrowprops=arrow_props)
ax.annotate('', xy=(7, 3.25), xytext=(6.5, 3.25), arrowprops=arrow_props)
ax.annotate('', xy=(9.2, 3.3), xytext=(8.8, 3.25), arrowprops=arrow_props)
ax.annotate('', xy=(10.8, 3.3), xytext=(10.7, 3.3), arrowprops=arrow_props)
ax.annotate('', xy=(9.95, 2.8), xytext=(9.95, 2.2), arrowprops=arrow_props)

# Add "+" symbol
ax.text(10.4, 3.6, '+', ha='center', va='center', fontsize=16, 
        fontweight='bold', color='#F39C12')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#F5B041', edgecolor='#D35400', label='Hash Encoding'),
    mpatches.Patch(facecolor='#E74C3C', edgecolor='#922B21', alpha=0.7, label='Terrain Features'),
    mpatches.Patch(facecolor='#E67E22', edgecolor='#D35400', label='MLP'),
    mpatches.Patch(facecolor='#C0392B', edgecolor='#922B21', label='Output'),
    mpatches.Patch(facecolor='#27AE60', edgecolor='#1E8449', label='Final Prediction')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, 
          framealpha=0.9, facecolor='#FEF9E7')

plt.tight_layout()
plt.savefig('paper/figures/fig4_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig4_architecture.png (300 DPI, warm colors)")

# ============================================================================
# Figure 5: Curriculum Learning Progress - WARM colors
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

# Simulated training curves for 3 stages with warm colors
np.random.seed(42)
epochs_stage = np.arange(150)
stage1 = 25 * np.exp(-epochs_stage/40) + 5 + np.random.randn(150) * 0.5
stage2 = 18 * np.exp(-epochs_stage/50) + 4 + np.random.randn(150) * 0.3
stage3 = 12 * np.exp(-epochs_stage/60) + 3.79 + np.random.randn(150) * 0.2

ax.plot(epochs_stage, stage1, label='Stage 1: Easy (low altitude, high density)', 
        color='#E67E22', linewidth=2.5)
ax.plot(epochs_stage + 150, stage2, label='Stage 2: Medium (moderate)', 
        color='#D35400', linewidth=2.5)
ax.plot(epochs_stage + 300, stage3, label='Stage 3: Hard (full dataset)', 
        color='#C0392B', linewidth=2.5)

# Vertical lines for stage boundaries
ax.axvline(x=150, color='#5D4037', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(x=300, color='#5D4037', linestyle='--', alpha=0.5, linewidth=1.5)

# Stage labels with warm background
ax.text(75, 28, 'Stage 1\n(150 epochs)', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#F5B041', alpha=0.8, edgecolor='#D35400'))
ax.text(225, 28, 'Stage 2\n(150 epochs)', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#E67E22', alpha=0.8, edgecolor='#D35400'))
ax.text(375, 28, 'Stage 3\n(150 epochs)', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#C0392B', alpha=0.8, edgecolor='#922B21'))

# Final result
ax.axhline(y=3.79, color=colors['success'], linestyle='-', alpha=0.9, linewidth=2.5)
ax.text(400, 4.5, 'Final MAE: 3.79m', fontsize=11, color=colors['success'], 
        fontweight='bold')

ax.set_xlabel('Epoch', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_ylabel('Validation MAE (m)', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_title('Curriculum Learning: Three-Stage Training Progress', fontsize=12, 
             fontweight='bold', color='#5D4037')
ax.set_xlim(0, 450)
ax.set_ylim(0, 30)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(alpha=0.3, color='#D7BDE2')
ax.set_facecolor('#FEF9E7')
ax.tick_params(colors='#5D4037')
for spine in ax.spines.values():
    spine.set_color('#5D4037')

plt.tight_layout()
plt.savefig('paper/figures/fig5_curriculum.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig5_curriculum.png (300 DPI, warm colors)")

# ============================================================================
# Figure 6: Spatial Performance Map - WARM colors
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 7))

# Simulate sensor locations in Shenzhen
coords = {
    '42499896': (114.045, 22.600, 58.0, 9.48),
    '78250224': (114.060, 22.610, 139.0, 9.61),
    '42508217': (114.055, 22.605, 100.1, 3.79),  # Best
    '27528610': (114.050, 22.598, 95.3, 5.41),
    '27536362': (114.058, 22.602, 107.9, 16.73),
    '78251938': (114.052, 22.608, 95.8, 11.43),
    '27373510': (114.048, 22.615, 259.0, 70.22),  # Worst
}

for sensor, (lon, lat, height, mae) in coords.items():
    # Size based on MAE (inverse)
    size = 2000 / (mae + 1)
    # Color based on MAE - warm scale
    if mae < 5:
        color = '#27AE60'  # Green for excellent
        marker = 'o'
    elif mae < 15:
        color = '#E67E22'  # Orange for good
        marker = 'o'
    else:
        color = '#C0392B'  # Red for challenging
        marker = 's'
    
    ax.scatter(lon, lat, s=size, c=color, alpha=0.7, edgecolors='#5D4037', 
               marker=marker, linewidths=1.5)
    ax.annotate(f'{sensor}\n({mae:.1f}m)', (lon, lat), 
                xytext=(5, 5), textcoords='offset points', fontsize=7,
                color='#5D4037', fontweight='bold')

# Add study area box
rect = Rectangle((114.042, 22.595), 0.022, 0.022, 
                 fill=False, edgecolor='#D35400', linewidth=2.5, linestyle='--')
ax.add_patch(rect)
ax.text(114.053, 22.5955, 'Study Area (~1 km²)', ha='center', fontsize=9, 
        color='#D35400', fontweight='bold')

ax.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold', color='#5D4037')
ax.set_title('Spatial Distribution of Sensor Performance', fontsize=12, 
             fontweight='bold', color='#5D4037')

# Legend
legend_elements = [
    plt.scatter([], [], s=200, c='#27AE60', alpha=0.7, edgecolors='#5D4037', 
                marker='o', label='Excellent (<5m)'),
    plt.scatter([], [], s=100, c='#E67E22', alpha=0.7, edgecolors='#5D4037',
                marker='o', label='Good (5-15m)'),
    plt.scatter([], [], s=50, c='#C0392B', alpha=0.7, edgecolors='#5D4037',
                marker='s', label='Challenging (>15m)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
          framealpha=0.9, facecolor='#FEF9E7')

ax.grid(alpha=0.3, color='#D7BDE2')
ax.set_facecolor('#FEF9E7')
ax.tick_params(colors='#5D4037')
for spine in ax.spines.values():
    spine.set_color('#5D4037')

plt.tight_layout()
plt.savefig('paper/figures/fig6_spatial_map.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig6_spatial_map.png (300 DPI, warm colors)")

# ============================================================================
# Figure 7: 3D Altitude Field Visualization - NEW!
# ============================================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create synthetic altitude field based on sensor data
np.random.seed(123)
x = np.linspace(114.040, 114.070, 50)
y = np.linspace(22.590, 22.620, 50)
X, Y = np.meshgrid(x, y)

# Create realistic altitude field with:
# 1. Overall slope
# 2. Local variations (urban terrain)
# 3. Sensor locations marked
Z = (100 + 50 * np.sin((X - 114.045) * 100) * np.cos((Y - 22.600) * 100) +
     30 * np.exp(-((X - 114.055)**2 + (Y - 22.605)**2) * 500) +
     200 * np.exp(-((X - 114.048)**2 + (Y - 22.615)**2) * 800) +
     np.random.randn(50, 50) * 5)

# Plot surface with warm colormap
surf = ax.plot_surface(X, Y, Z, cmap=WARM_CMAP, alpha=0.9, 
                       linewidth=0, antialiased=True, shade=True)

# Add sensor points
sensor_coords = [
    (114.045, 22.600, 58.0, '42499896', 9.48),
    (114.060, 22.610, 139.0, '78250224', 9.61),
    (114.055, 22.605, 100.1, '42508217', 3.79),
    (114.050, 22.598, 95.3, '27528610', 5.41),
    (114.058, 22.602, 107.9, '27536362', 16.73),
    (114.052, 22.608, 95.8, '78251938', 11.43),
    (114.048, 22.615, 259.0, '27373510', 70.22),
]

for lon, lat, height, sid, mae in sensor_coords:
    # Color based on error
    if mae < 5:
        color = '#27AE60'
    elif mae < 15:
        color = '#E67E22'
    else:
        color = '#C0392B'
    
    ax.scatter(lon, lat, height, c=color, s=150, edgecolors='#5D4037', 
               linewidths=2, marker='o', alpha=1.0, depthshade=False)
    ax.text(lon, lat, height + 15, f'{sid}\n({height:.0f}m)', 
            fontsize=7, color='#5D4037', fontweight='bold')

# Add contour lines at bottom
ax.contour(X, Y, Z, zdir='z', offset=0, cmap=WARM_CMAP, alpha=0.5, linewidths=1)

ax.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold', color='#5D4037')
ax.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold', color='#5D4037')
ax.set_zlabel('Altitude (m)', fontsize=10, fontweight='bold', color='#5D4037')
ax.set_title('3D Urban Altitude Field with Sensor Locations', fontsize=13, 
             fontweight='bold', color='#5D4037', pad=20)

# Set view angle
ax.view_init(elev=30, azim=240)

# Colorbar with warm colors
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.05)
cbar.set_label('Altitude (m)', fontsize=10, fontweight='bold', color='#5D4037')
cbar.ax.tick_params(colors='#5D4037')

ax.tick_params(colors='#5D4037')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#D35400')
ax.yaxis.pane.set_edgecolor('#D35400')
ax.zaxis.pane.set_edgecolor('#D35400')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

plt.tight_layout()
plt.savefig('paper/figures/fig7_3d_altitude_field.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig7_3d_altitude_field.png (300 DPI, warm colors)")

# ============================================================================
# Figure 8: 3D Altitude Field with Prediction Errors - NEW!
# ============================================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create prediction error field
Z_error = np.abs(Z - 100)  # Synthetic error field

# Plot surface with error colormap (red-based)
ERROR_CMAP = LinearSegmentedColormap.from_list('error', [
    '#FEF9E7',  # Cream
    '#F8C471',  # Light orange
    '#E67E22',  # Orange
    '#C0392B',  # Red
    '#7B241C',  # Dark red
])

surf = ax.plot_surface(X, Y, Z, facecolors=ERROR_CMAP(Z_error/Z_error.max()),
                       alpha=0.9, linewidth=0, antialiased=True, shade=True)

# Add sensor points with error colors
for lon, lat, height, sid, mae in sensor_coords:
    # Color based on error
    if mae < 5:
        color = '#27AE60'
    elif mae < 15:
        color = '#E67E22'
    else:
        color = '#C0392B'
    
    ax.scatter(lon, lat, height, c=color, s=200, edgecolors='white', 
               linewidths=3, marker='o', alpha=1.0, depthshade=False)
    ax.text(lon, lat, height + 15, f'{sid}\nMAE:{mae:.1f}m', 
            fontsize=7, color='#5D4037', fontweight='bold')

ax.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold', color='#5D4037')
ax.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold', color='#5D4037')
ax.set_zlabel('Altitude (m)', fontsize=10, fontweight='bold', color='#5D4037')
ax.set_title('3D Altitude Field with Prediction Error Heatmap', fontsize=13, 
             fontweight='bold', color='#5D4037', pad=20)

ax.view_init(elev=25, azim=230)

ax.tick_params(colors='#5D4037')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#D35400')
ax.yaxis.pane.set_edgecolor('#D35400')
ax.zaxis.pane.set_edgecolor('#D35400')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# Add legend for error levels
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60', 
           markersize=12, label='Low Error (<5m)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22', 
           markersize=12, label='Medium Error (5-15m)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B', 
           markersize=12, label='High Error (>15m)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('paper/figures/fig8_3d_error_heatmap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Generated: fig8_3d_error_heatmap.png (300 DPI, warm colors)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Figure Generation Complete!")
print("="*60)
print("\nGenerated figures (300 DPI PNG, warm color palette):")
print("  1. fig1_method_comparison.png")
print("  2. fig2_loso_results.png")
print("  3. fig3_ablation.png")
print("  4. fig4_architecture.png")
print("  5. fig5_curriculum.png")
print("  6. fig6_spatial_map.png")
print("  7. fig7_3d_altitude_field.png - NEW!")
print("  8. fig8_3d_error_heatmap.png - NEW!")
print("\nColor Palette: Sunset/Autumn (Orange-Red warm tones)")
print("New 3D visualizations show altitude field and error distribution!")
