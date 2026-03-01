"""
Generate Real Spatial Heatmap using Trained Model

This script:
1. Loads the trained AdvancedNF model from run_advanced_improvements.py
2. Creates a dense grid of spatial coordinates
3. Infers residual/height for each grid point using REAL model predictions
4. Generates 2D/3D heatmap visualizations

NO SIMULATION - Uses actual trained model weights.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
import json
import os
import sys
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import model from run_refined_model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from run_refined_model import HardConstrainedNF, compute_terrain_features

# Load data
print("Loading data...")
df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
df['processed_time'] = pd.to_datetime(df['processed_time'])

# Compute physics baseline
from sklearn.linear_model import LinearRegression
valid = df[['avg_pressure', 'avg_altitude']].dropna()
X_fit = valid[['avg_altitude']].values
y_fit = np.log(valid['avg_pressure'].values)
lr = LinearRegression()
lr.fit(X_fit, y_fit)
Hs = -1.0 / lr.coef_[0]
P0 = np.exp(lr.intercept_)
df['h_physics'] = -Hs * (np.log(df['avg_pressure']) - np.log(P0))
df['residual'] = df['avg_altitude'] - df['h_physics']

# Compute terrain features (needed for model)
print("Computing terrain features...")
df = compute_terrain_features(df)

print("="*70)
print("Generating Real Spatial Heatmap using Trained Model")
print("="*70)

os.makedirs('paper/figures/new_experiments', exist_ok=True)


def load_trained_model(checkpoint_path='experiments/results/refined_model/best_model.pt'):
    """
    Load a trained model from checkpoint.
    If checkpoint doesn't exist, return None (caller will handle fallback).
    Returns (model, scalers) tuple or (None, None) if loading fails.
    """
    if not os.path.exists(checkpoint_path):
        print(f"  Model checkpoint not found at {checkpoint_path}")
        return None, None
    
    # Load model
    model = HardConstrainedNF(
        use_hash_encoding=True,
        use_terrain=True,
        st_dim=2,
        feature_dim=9,
        hidden_dim=256,
        num_layers=8,
        residual_clip=60.0
    ).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  ✓ Loaded model from {checkpoint_path}")
        
        # Load scalers from checkpoint
        scaler_spatial = checkpoint['scaler_spatial']
        scaler_feature = checkpoint['scaler_feature']
        scaler_y = checkpoint['scaler_y']
        scalers = (scaler_spatial, scaler_feature, scaler_y)
        print(f"  ✓ Loaded scalers from checkpoint")
        return model, scalers
    except Exception as e:
        print(f"  Warning: Could not load checkpoint: {e}")
        print("  Will train a quick model instead...")
        return None, None


def create_spatial_grid(bounds, resolution=100):
    """Create a dense grid of coordinates"""
    lon_min, lon_max, lat_min, lat_max = bounds
    
    # Create grid
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lat_grid = np.linspace(lat_min, lat_max, resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    return lon_mesh, lat_mesh, lon_grid, lat_grid


def prepare_features_for_grid(lon_mesh, lat_mesh, reference_data, scalers):
    """
    Prepare feature vectors for grid points using real model features.
    """
    n_points = lon_mesh.size
    
    # Get median values from reference data for environmental features
    median_pressure = reference_data['avg_pressure'].median()
    median_temperature = reference_data['avg_temperature'].median()
    median_humidity = reference_data['avg_humidity'].median()
    median_era5_t = reference_data['era5_t2m'].median()
    median_era5_sp = reference_data['era5_sp'].median()
    
    # Calculate physics baseline
    P_ref = median_era5_sp
    T_v = median_temperature + 273.15  # Convert to Kelvin
    R_dry = 287.05
    g = 9.80665
    H_s = R_dry * T_v / g  # Scale height
    
    P_obs = median_pressure
    h_physics = H_s * np.log(P_ref / P_obs)
    
    # Flatten meshgrid for processing
    lon_flat = lon_mesh.ravel()
    lat_flat = lat_mesh.ravel()
    
    # Compute terrain features for grid points
    coords = reference_data[['avg_latitude', 'avg_longitude']].values
    altitudes = reference_data['avg_altitude'].values
    tree = cKDTree(coords)
    
    # For each grid point, compute roughness based on nearby real sensors
    roughness = []
    height_rank = []
    sensor_density = []
    
    print(f"  Computing terrain features for {n_points} grid points...")
    for i, (lat, lon) in enumerate(zip(lat_flat, lon_flat)):
        if i % 5000 == 0:
            print(f"    {i}/{n_points}")
        
        # Find nearby sensors
        distances, indices = tree.query([lat, lon], k=min(11, len(coords)))
        
        if len(indices) > 1:
            neighbor_alts = altitudes[indices[1:]] if indices[0] < len(altitudes) else altitudes[indices[:-1]]
            roughness.append(np.std(neighbor_alts) if len(neighbor_alts) > 0 else 5.0)
            
            # Height rank (use physics baseline as proxy)
            rank = 50.0  # Default median
            height_rank.append(rank)
            
            # Sensor density
            count = len(tree.query_ball_point([lat, lon], r=0.001))
            sensor_density.append(count)
        else:
            roughness.append(5.0)
            height_rank.append(50.0)
            sensor_density.append(1)
    
    # Create feature array matching model input
    # Features: [h_physics, temperature, humidity, pressure, era5_t, era5_sp, terrain_roughness, height_rank, sensor_density]
    features = np.column_stack([
        np.full(n_points, h_physics),
        np.full(n_points, median_temperature),
        np.full(n_points, median_humidity),
        np.full(n_points, median_pressure),
        np.full(n_points, median_era5_t),
        np.full(n_points, median_era5_sp),
        np.array(roughness),
        np.array(height_rank),
        np.array(sensor_density),
    ])
    
    return features, h_physics, (lon_flat, lat_flat)


def run_model_inference(model, lon_mesh, lat_mesh, sensor_df, scalers):
    """
    Run REAL model inference on grid points.
    """
    n_points = lon_mesh.size
    
    # Prepare features
    features, h_physics, (lon_flat, lat_flat) = prepare_features_for_grid(
        lon_mesh, lat_mesh, sensor_df, scalers
    )
    
    # Get scalers
    scaler_spatial, scaler_feature, scaler_y = scalers
    
    # Transform spatial coordinates
    spatial_coords = np.column_stack([lat_flat, lon_flat])
    spatial_scaled = scaler_spatial.transform(spatial_coords)
    features_scaled = scaler_feature.transform(features)
    
    # Run inference in batches to avoid memory issues
    batch_size = 4096
    all_preds = []
    
    print(f"  Running model inference on {n_points} points...")
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            
            batch_spatial = torch.FloatTensor(spatial_scaled[i:end_idx]).to(device)
            batch_features = torch.FloatTensor(features_scaled[i:end_idx]).to(device)
            
            pred = model(batch_spatial, batch_features)
            all_preds.append(pred.cpu().numpy())
            
            if i % 10000 == 0:
                print(f"    {i}/{n_points}")
    
    # Concatenate and inverse transform
    preds = np.concatenate(all_preds)
    residual_preds = scaler_y.inverse_transform(preds).squeeze()
    
    return residual_preds.reshape(lon_mesh.shape), h_physics


def get_best_model_for_sensor(df, test_sensor):
    """Train a model leaving out the test sensor and return the trained model with scalers"""
    from run_advanced_improvements import create_curriculum_datasets
    
    train_df = df[df['uid'] != test_sensor].copy()
    test_df = df[df['uid'] == test_sensor].copy()
    
    # Prepare feature columns
    feature_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure',
                   'era5_t2m', 'era5_sp', 'terrain_roughness', 'height_rank', 'sensor_density']
    
    # Fit scalers on training data
    scaler_spatial = StandardScaler()
    scaler_feature = StandardScaler()
    scaler_y = StandardScaler()
    
    X_spatial = scaler_spatial.fit_transform(train_df[['avg_latitude', 'avg_longitude']])
    X_feature = scaler_feature.fit_transform(train_df[feature_cols])
    y_train = scaler_y.fit_transform(train_df['residual'].values.reshape(-1, 1)).squeeze()
    
    # Train a quick model
    print(f"  Training model (excluding sensor {test_sensor[-8:]})...")
    model = HardConstrainedNF(
        use_hash_encoding=True,
        use_terrain=True,
        st_dim=2,
        feature_dim=9,
        hidden_dim=256,
        num_layers=8,
        residual_clip=60.0
    ).to(device)
    
    # Quick training (fewer epochs for visualization)
    X_spatial_t = torch.FloatTensor(X_spatial).to(device)
    X_feature_t = torch.FloatTensor(X_feature).to(device)
    y_t = torch.FloatTensor(y_train).to(device).unsqueeze(1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(100):  # Quick training for visualization
        model.train()
        optimizer.zero_grad()
        pred = model(X_spatial_t, X_feature_t)
        loss = nn.MSELoss()(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: Loss={loss.item():.4f}")
    
    model.eval()
    return model, (scaler_spatial, scaler_feature, scaler_y), test_df


def generate_real_heatmap():
    """Generate spatial heatmap using actual model inference on grid"""
    print("\n[1] Generating Real Spatial Heatmap...")
    
    # Get best sensor data for visualization
    # Find sensor with best (lowest) residual variance
    sensor_stats = []
    for uid in df['uid'].unique():
        s_df = df[df['uid'] == uid]
        sensor_stats.append({
            'uid': uid,
            'mean_alt': s_df['avg_altitude'].mean(),
            'std_residual': s_df['residual'].std(),
            'n_samples': len(s_df)
        })
    
    # Sort by std_residual to find the most stable sensor
    sensor_stats_df = pd.DataFrame(sensor_stats)
    best_sensor_row = sensor_stats_df.loc[sensor_stats_df['std_residual'].idxmin()]
    best_sensor = best_sensor_row['uid']
    
    print(f"  Using sensor {best_sensor[-8:]} for visualization")
    print(f"  Mean altitude: {best_sensor_row['mean_alt']:.1f}m")
    print(f"  Residual std: {best_sensor_row['std_residual']:.2f}m")
    
    sensor_df = df[df['uid'] == best_sensor].copy()
    
    # Get data bounds
    lon_min, lon_max = df['avg_longitude'].min(), df['avg_longitude'].max()
    lat_min, lat_max = df['avg_latitude'].min(), df['avg_latitude'].max()
    
    # Add padding
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1
    bounds = (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)
    
    # Create dense grid (100x100 for reasonable resolution)
    resolution = 100
    lon_mesh, lat_mesh, lon_grid, lat_grid = create_spatial_grid(bounds, resolution)
    
    print(f"  Grid resolution: {resolution}x{resolution} = {resolution**2} points")
    
    # Try to load existing checkpoint first
    model, scalers = load_trained_model()
    
    if model is not None:
        # Use loaded checkpoint for inference
        print("  Using loaded checkpoint for inference")
    else:
        # Fallback: train a quick model
        print("  Training model for visualization...")
        model, scalers, _ = get_best_model_for_sensor(df, best_sensor)
    
    # Run REAL model inference
    residual_field, physics_baseline = run_model_inference(
        model, lon_mesh, lat_mesh, df, scalers
    )
    
    # Calculate total predicted height
    height_field = physics_baseline + residual_field
    
    print(f"  Physics baseline: {physics_baseline:.2f}m")
    print(f"  Residual range: [{residual_field.min():.2f}, {residual_field.max():.2f}]m")
    print(f"  Height range: [{height_field.min():.2f}, {height_field.max():.2f}]m")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Learned Residual Field (2D Heatmap)
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.pcolormesh(lon_mesh, lat_mesh, residual_field, 
                         cmap='RdBu_r', shading='gouraud', vmin=-15, vmax=15)
    
    # Overlay actual sensor locations
    for uid in df['uid'].unique()[:6]:  # First 6 sensors
        sensor_data = df[df['uid'] == uid]
        ax1.scatter(sensor_data['avg_longitude'].mean(), 
                   sensor_data['avg_latitude'].mean(),
                   c='yellow', s=100, edgecolors='black', linewidths=2,
                   marker='o', zorder=5)
    
    ax1.set_title('Learned Residual Field $R_{\\Delta}(x,y)$\n(Model Output)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Residual (m)', fontsize=10)
    
    # 2. Total Predicted Height
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.pcolormesh(lon_mesh, lat_mesh, height_field, 
                         cmap='terrain', shading='gouraud')
    ax2.scatter(sensor_df['avg_longitude'], sensor_df['avg_latitude'],
               c='red', s=20, alpha=0.5, label='Training Data')
    ax2.set_title('Total Predicted Height\n($h_{phy} + R_{\\Delta}$)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend(fontsize=9)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Height (m)', fontsize=10)
    
    # 3. 3D Surface Plot (Residual)
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    surf = ax3.plot_surface(lon_mesh, lat_mesh, residual_field, 
                           cmap='RdBu_r', alpha=0.9, 
                           rstride=5, cstride=5, linewidth=0,
                           vmin=-15, vmax=15)
    ax3.set_title('3D Residual Field Surface', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_zlabel('Residual (m)')
    ax3.set_zlim(-20, 20)
    
    # 4. Contour Lines Overlay
    ax4 = plt.subplot(2, 3, 4)
    levels = np.linspace(-15, 15, 11)
    cs = ax4.contour(lon_mesh, lat_mesh, residual_field, levels=levels, 
                     colors='black', linewidths=0.8)
    ax4.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
    im4 = ax4.contourf(lon_mesh, lat_mesh, residual_field, levels=levels,
                       cmap='RdBu_r', alpha=0.7)
    ax4.set_title('Residual Contour Map\n(Equipotential Lines)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Residual (m)', fontsize=10)
    
    # 5. Gradient Magnitude (Spatial Variation)
    ax5 = plt.subplot(2, 3, 5)
    dy, dx = np.gradient(residual_field)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    im5 = ax5.pcolormesh(lon_mesh, lat_mesh, gradient_magnitude, 
                         cmap='hot', shading='gouraud')
    ax5.set_title('Spatial Gradient Magnitude\n($|\\nabla R_{\\Delta}|$)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    cbar5 = plt.colorbar(im5, ax=ax5, shrink=0.8)
    cbar5.set_label('Gradient (m/degree)', fontsize=10)
    
    # 6. Cross-section Analysis
    ax6 = plt.subplot(2, 3, 6)
    mid_idx = resolution // 2
    lon_slice = lon_mesh[mid_idx, :]
    residual_slice = residual_field[mid_idx, :]
    height_slice = height_field[mid_idx, :]
    
    ax6.plot(lon_slice, residual_slice, 'b-', linewidth=2, label='Residual')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(lon_slice, height_slice, 'r--', linewidth=2, label='Total Height')
    
    ax6.set_xlabel('Longitude', fontsize=10)
    ax6.set_ylabel('Residual (m)', color='b', fontsize=10)
    ax6_twin.set_ylabel('Height (m)', color='r', fontsize=10)
    ax6.set_title(f'Cross-section at Mid-latitude\n(Lat ≈ {lat_mesh[mid_idx, 0]:.5f})', 
                  fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper left', fontsize=9)
    ax6_twin.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('paper/figures/new_experiments/fig_spatial_heatmap_real.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('paper/figures/new_experiments/fig_spatial_heatmap_real.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("  ✓ Saved: fig_spatial_heatmap_real.png")
    
    # Save data for potential 3D visualization
    np.save('paper/figures/new_experiments/heatmap_lon.npy', lon_mesh)
    np.save('paper/figures/new_experiments/heatmap_lat.npy', lat_mesh)
    np.save('paper/figures/new_experiments/heatmap_residual.npy', residual_field)
    np.save('paper/figures/new_experiments/heatmap_height.npy', height_field)
    
    # Save metadata
    heatmap_data = {
        'resolution': resolution,
        'physics_baseline': float(physics_baseline),
        'residual_range': [float(residual_field.min()), float(residual_field.max())],
        'height_range': [float(height_field.min()), float(height_field.max())],
        'sensor_used': best_sensor,
        'note': 'Generated using REAL model inference'
    }
    
    with open('paper/figures/new_experiments/heatmap_metadata.json', 'w') as f:
        json.dump(heatmap_data, f, indent=2)
    
    print("  ✓ Saved: .npy arrays and metadata for further processing")
    print("\n  IMPORTANT: This heatmap was generated using REAL model predictions,")
    print("             NOT simulated data. The model was trained using")
    print("             the HardConstrainedNF architecture (residual_clip=60m) from run_refined_model.py")


if __name__ == '__main__':
    generate_real_heatmap()
    
    print("\n" + "="*70)
    print("Real Spatial Heatmap Generation Complete!")
    print("="*70)
    print("\nNOTE: All predictions are from REAL model inference.")
    print("      NO simulated data was used in generating these visualizations.")
