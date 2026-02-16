"""
2D Physics-Informed Neural Operator (PINO-2D)

Maps sensor data to 2D spatial grid and applies Fourier Neural Operator.
Key innovations:
1. Spatial grid mapping with Gaussian interpolation
2. 2D Spectral convolutions for spatial pattern learning
3. Physics-informed loss functions
4. Multi-scale feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fix random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_seed(42)


# ==============================================================================
# 2D Spectral Convolution
# ==============================================================================

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Does FFT, linear transform, and IFFT.
    """
    def __init__(self, in_channels, out_channels, modes1=12, modes2=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # First dim modes
        self.modes2 = modes2  # Second dim modes
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Learnable weights for low-frequency Fourier modes
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
    
    def compl_mul2d(self, input, weights):
        """Complex multiplication: (batch, in_ch, x, y) * (in_ch, out_ch, x, y)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # FFT
        x_ft = fft.rfft2(x, dim=(-2, -1))
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)
        
        # Lower modes (near center)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights1)
        )
        
        # Higher modes (far from center)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2],
            torch.view_as_complex(self.weights2)
        )
        
        # IFFT
        x = fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        return x


# ==============================================================================
# 2D FNO Model
# ==============================================================================

class FNO2D(nn.Module):
    """
    2D Fourier Neural Operator for spatial altitude estimation.
    
    Architecture:
    Input: [batch, channels, height, width] - spatial grid of features
      ↓
    Lifting: Linear projection to high-dimensional space
      ↓
    2D FNO Layers: Multiple spectral convolutions
      ↓
    Projection: Linear to output (altitude residual)
    """
    def __init__(self, in_channels=8, out_channels=1, width=64, 
                 modes1=12, modes2=12, n_layers=4):
        super().__init__()
        
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_layers = n_layers
        
        # Lifting layer
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO layers
        self.fno_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()  # Skip connections
        
        for _ in range(n_layers):
            self.fno_layers.append(
                SpectralConv2d(width, width, modes1, modes2)
            )
            self.w_layers.append(
                nn.Conv2d(width, width, 1)
            )
        
        # Projection layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width] - input features on grid
        Returns:
            [batch, 1, height, width] - predicted residual on grid
        """
        # Lifting: [batch, channels, h, w] -> [batch, h, w, width]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [batch, width, h, w]
        
        # FNO layers
        for i, (fno, w) in enumerate(zip(self.fno_layers, self.w_layers)):
            x1 = fno(x)
            x2 = w(x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        # Projection: [batch, width, h, w] -> [batch, 1, h, w]
        x = x.permute(0, 2, 3, 1)  # [batch, h, w, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # [batch, 1, h, w]
        
        return x


# ==============================================================================
# Spatial Grid Mapping
# ==============================================================================

def create_spatial_grid(df, grid_size=32, method='linear'):
    """
    Map irregular sensor data to regular 2D grid.
    
    Args:
        df: DataFrame with 'avg_latitude', 'avg_longitude', and features
        grid_size: Resolution of output grid
        method: Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
        grid_data: [channels, height, width] - gridded features
        grid_mask: [height, width] - mask of valid grid points
        grid_coords: (grid_lon, grid_lat) - coordinate meshes
    """
    # Get bounds
    lon_min, lon_max = df['avg_longitude'].min(), df['avg_longitude'].max()
    lat_min, lat_max = df['avg_latitude'].min(), df['avg_latitude'].max()
    
    # Add padding
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1
    lon_min -= lon_pad
    lon_max += lon_pad
    lat_min -= lat_pad
    lat_max += lat_pad
    
    # Create grid
    grid_lon = np.linspace(lon_min, lon_max, grid_size)
    grid_lat = np.linspace(lat_min, lat_max, grid_size)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Points for interpolation
    points = df[['avg_longitude', 'avg_latitude']].values
    
    # Interpolate each feature
    feature_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 
                   'avg_pressure', 'era5_t2m', 'era5_sp', 'height_rank']
    
    grid_features = []
    
    for col in feature_cols:
        values = df[col].values
        grid = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), 
                       method=method, fill_value=np.nan)
        
        # Fill NaN with nearest neighbor
        if np.isnan(grid).any():
            mask = ~np.isnan(grid)
            if mask.any():
                grid_flat = grid.flatten()
                mask_flat = mask.flatten()
                grid_flat[~mask_flat] = np.interp(
                    np.where(~mask_flat)[0],
                    np.where(mask_flat)[0],
                    grid_flat[mask_flat]
                )
                grid = grid_flat.reshape(grid.shape)
        
        grid_features.append(grid)
    
    # Stack features: [channels, height, width]
    grid_data = np.stack(grid_features, axis=0)
    
    # Create mask (valid sensor coverage area)
    # Use distance to nearest sensor
    grid_points = np.column_stack([grid_lon_mesh.ravel(), grid_lat_mesh.ravel()])
    distances = cdist(grid_points, points).min(axis=1)
    max_sensor_dist = 0.005  # degrees (~500m)
    mask = (distances < max_sensor_dist).reshape(grid_size, grid_size)
    
    return (torch.FloatTensor(grid_data), 
            torch.FloatTensor(mask),
            (grid_lon_mesh, grid_lat_mesh))


def sensor_to_grid(df, grid_size=32):
    """
    Create grid representation of sensor data.
    
    Returns grid with sensor values at sensor locations, zeros elsewhere.
    """
    # Get bounds
    lon_min, lon_max = df['avg_longitude'].min(), df['avg_longitude'].max()
    lat_min, lat_max = df['avg_latitude'].min(), df['avg_latitude'].max()
    
    # Padding
    lon_pad = (lon_max - lon_min) * 0.05
    lat_pad = (lat_max - lat_min) * 0.05
    
    # Create grid
    grid_lon = np.linspace(lon_min - lon_pad, lon_max + lon_pad, grid_size)
    grid_lat = np.linspace(lat_min - lat_pad, lat_max + lat_pad, grid_size)
    
    # Assign sensors to grid cells
    lon_bins = np.digitize(df['avg_longitude'].values, grid_lon) - 1
    lat_bins = np.digitize(df['avg_latitude'].values, grid_lat) - 1
    
    # Clip to valid range
    lon_bins = np.clip(lon_bins, 0, grid_size - 1)
    lat_bins = np.clip(lat_bins, 0, grid_size - 1)
    
    # Create feature grids
    feature_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 
                   'avg_pressure', 'era5_t2m', 'era5_sp', 'height_rank']
    
    grid = np.zeros((len(feature_cols) + 1, grid_size, grid_size))  # +1 for residual target
    count = np.zeros((grid_size, grid_size))
    
    # Fill grid with sensor values
    for i, (_, row) in enumerate(df.iterrows()):
        li, lj = lat_bins[i], lon_bins[i]
        
        # Input features
        for fi, col in enumerate(feature_cols):
            grid[fi, li, lj] += row[col]
        
        # Target residual
        grid[-1, li, lj] += row['residual']
        count[li, lj] += 1
    
    # Average where multiple sensors in same cell
    mask = count > 0
    for c in range(grid.shape[0]):
        grid[c][mask] /= count[mask]
    
    return torch.FloatTensor(grid), torch.FloatTensor(mask), (grid_lon, grid_lat)


# ==============================================================================
# Physics-Informed Loss
# ==============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss: Data fidelity + Physics constraints
    """
    def __init__(self, lambda_data=1.0, lambda_smooth=0.1, lambda_bound=0.01):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_smooth = lambda_smooth
        self.lambda_bound = lambda_bound
        self.mse = nn.MSELoss()
    
    def spatial_gradient(self, x):
        """Compute spatial gradients using finite differences"""
        # x: [batch, 1, h, w]
        grad_h = x[:, :, 1:, :] - x[:, :, :-1, :]  # Vertical gradient
        grad_w = x[:, :, :, 1:] - x[:, :, :, :-1]  # Horizontal gradient
        return grad_h, grad_w
    
    def forward(self, pred, target, mask):
        """
        Compute combined loss
        
        Args:
            pred: [batch, 1, h, w] - predicted residual
            target: [batch, 1, h, w] - true residual
            mask: [batch, 1, h, w] - valid region mask
        """
        # Data fidelity loss (only where we have data)
        diff = (pred - target) * mask
        loss_data = (diff ** 2).sum() / (mask.sum() + 1e-8)
        
        # Spatial smoothness loss (encourage gradual changes)
        grad_h, grad_w = self.spatial_gradient(pred)
        loss_smooth = (grad_h ** 2).mean() + (grad_w ** 2).mean()
        
        # Boundary smoothness (edges should be smooth)
        pred_padded = F.pad(pred, (1, 1, 1, 1), mode='replicate')
        loss_bound = ((pred - pred_padded[:, :, 1:-1, 1:-1]) ** 2).mean()
        
        # Combined
        loss = (self.lambda_data * loss_data + 
                self.lambda_smooth * loss_smooth + 
                self.lambda_bound * loss_bound)
        
        return loss, {
            'data': loss_data.item(),
            'smooth': loss_smooth.item(),
            'bound': loss_bound.item()
        }


# ==============================================================================
# Training
# ==============================================================================

def prepare_data_grid(df, grid_size=32):
    """Prepare grid data for all sensors"""
    sensors = df['uid'].unique()
    
    all_grids = []
    all_masks = []
    all_residuals = []
    sensor_ids = []
    
    for sensor in sensors:
        sensor_df = df[df['uid'] == sensor].copy()
        
        # Compute height rank
        sensor_df['height_rank'] = sensor_df['avg_altitude'].rank(pct=True) * 100
        
        # Create grid
        grid, mask, coords = sensor_to_grid(sensor_df, grid_size)
        
        # Split: first channels are input, last is target
        input_grid = grid[:-1]  # [channels-1, h, w]
        target_grid = grid[-1:]  # [1, h, w]
        
        all_grids.append(input_grid)
        all_masks.append(mask)
        all_residuals.append(target_grid)
        sensor_ids.append(sensor)
    
    return {
        'inputs': torch.stack(all_grids),  # [n_sensors, channels, h, w]
        'masks': torch.stack(all_masks),   # [n_sensors, h, w]
        'targets': torch.stack(all_residuals),  # [n_sensors, 1, h, w]
        'sensors': sensor_ids
    }


def train_pino_2d(data, n_epochs=200, lr=1e-3, save_dir='experiments/pino2d'):
    """Train 2D PINO"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    X = data['inputs'].to(device)
    masks = data['masks'].unsqueeze(1).to(device)  # [n, 1, h, w]
    y = data['targets'].to(device)
    
    n_sensors, n_channels, h, w = X.shape
    print(f"Data: {n_sensors} sensors, {n_channels} channels, grid {h}x{w}")
    
    # Create model
    model = FNO2D(
        in_channels=n_channels,
        out_channels=1,
        width=64,
        modes1=8,
        modes2=8,
        n_layers=4
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = PhysicsInformedLoss(lambda_data=1.0, lambda_smooth=0.1)
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(n_epochs):
        model.train()
        
        # Forward pass
        pred = model(X)
        loss, loss_dict = criterion(pred, y, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Log
        if epoch % 20 == 0:
            # Compute MAE on valid pixels only
            with torch.no_grad():
                diff = torch.abs(pred - y) * masks
                mae = diff.sum() / masks.sum()
            
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'mae': mae.item(),
                **loss_dict
            })
            
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, MAE={mae.item():.3f}m, "
                  f"Data={loss_dict['data']:.4f}, Smooth={loss_dict['smooth']:.4f}")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'loss': loss.item()
                }, f'{save_dir}/pino2d_best.pt')
    
    return model, history


def evaluate_loso_2d(df, grid_size=32, n_epochs=200):
    """Leave-one-sensor-out evaluation"""
    
    sensors = df['uid'].unique()
    results = []
    
    print(f"\n{'='*60}")
    print(f"2D PINO LOSO Evaluation ({len(sensors)} sensors)")
    print(f"{'='*60}\n")
    
    for fold_idx, test_sensor in enumerate(sensors):
        print(f"Fold {fold_idx+1}/{len(sensors)}: {test_sensor[-8:]}")
        
        # Split data
        train_df = df[df['uid'] != test_sensor].copy()
        test_df = df[df['uid'] == test_sensor].copy()
        
        # Add height rank
        train_df['height_rank'] = train_df['avg_altitude'].rank(pct=True) * 100
        test_df['height_rank'] = test_df['avg_altitude'].rank(pct=True) * 100
        
        # Prepare grids
        train_data = prepare_data_grid(train_df, grid_size)
        
        # Train
        save_dir = f'experiments/pino2d/fold_{fold_idx}'
        model, history = train_pino_2d(train_data, n_epochs=n_epochs, save_dir=save_dir)
        
        # Evaluate on test sensor
        model.eval()
        with torch.no_grad():
            test_grid, test_mask, _ = sensor_to_grid(test_df, grid_size)
            test_input = test_grid[:-1].unsqueeze(0).to(device)
            test_target = test_grid[-1:].unsqueeze(0).to(device)
            test_mask = test_mask.unsqueeze(0).unsqueeze(0).to(device)
            
            pred = model(test_input)
            
            # Compute MAE
            diff = torch.abs(pred - test_target) * test_mask
            mae = diff.sum() / test_mask.sum()
            
            # Also compute per-sample MAE
            valid_pixels = test_mask.squeeze().bool()
            pred_vals = pred.squeeze()[valid_pixels].cpu().numpy()
            true_vals = test_target.squeeze()[valid_pixels].cpu().numpy()
            
        print(f"  Grid MAE: {mae.item():.3f}m")
        
        # Calculate total height error
        h_physics_test = test_df['h_physics'].values
        y_test_alt = test_df['avg_altitude'].values
        residual_test = test_df['residual'].values
        
        # Average predicted residual for this sensor
        pred_residual_mean = pred_vals.mean()
        
        # Total height prediction
        h_pred = h_physics_test + pred_residual_mean
        total_mae = np.abs(h_pred - y_test_alt).mean()
        
        print(f"  Total Height MAE: {total_mae:.3f}m")
        
        results.append({
            'sensor': test_sensor[-8:],
            'grid_mae': mae.item(),
            'total_mae': total_mae,
            'n_samples': len(test_df)
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_maes = [r['total_mae'] for r in results]
    for r in results:
        print(f"{r['sensor']}: {r['total_mae']:.3f}m ({r['n_samples']} samples)")
    
    print(f"\nMean Total MAE: {np.mean(total_maes):.3f}m ± {np.std(total_maes):.3f}m")
    print(f"Best: {np.min(total_maes):.3f}m, Worst: {np.max(total_maes):.3f}m")
    print(f"Baseline: 3.79m")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D PINO')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'loso', 'test'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--grid_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--modes', type=int, default=8)
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
    
    if args.mode == 'train':
        # Single sensor test
        sensor = df['uid'].unique()[0]
        sensor_df = df[df['uid'] == sensor].copy()
        sensor_df['height_rank'] = sensor_df['avg_altitude'].rank(pct=True) * 100
        
        data = prepare_data_grid(sensor_df, args.grid_size)
        model, history = train_pino_2d(data, n_epochs=args.epochs)
    
    elif args.mode == 'loso':
        results = evaluate_loso_2d(df, grid_size=args.grid_size, n_epochs=args.epochs)
        
        # Save results
        with open('experiments/pino2d/loso_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    elif args.mode == 'test':
        # Quick test on one sensor
        sensor = df['uid'].unique()[0]
        print(f"Testing on sensor: {sensor[-8:]}")
        
        sensor_df = df[df['uid'] == sensor].copy()
        sensor_df['height_rank'] = sensor_df['avg_altitude'].rank(pct=True) * 100
        
        # Split train/test by time
        split = int(len(sensor_df) * 0.8)
        train_df = sensor_df.iloc[:split]
        test_df = sensor_df.iloc[split:]
        
        train_data = prepare_data_grid(train_df, args.grid_size)
        
        model, _ = train_pino_2d(train_data, n_epochs=100, 
                                 save_dir='experiments/pino2d_test')
        
        # Evaluate
        model.eval()
        test_grid, test_mask, _ = sensor_to_grid(test_df, args.grid_size)
        test_input = test_grid[:-1].unsqueeze(0).to(device)
        test_target = test_grid[-1:].unsqueeze(0).to(device)
        test_mask_t = test_mask.unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(test_input)
            diff = torch.abs(pred - test_target) * test_mask_t
            mae = diff.sum() / test_mask_t.sum()
        
        print(f"\nTest MAE: {mae.item():.3f}m")
