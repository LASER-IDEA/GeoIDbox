"""
Generalized PINN with physics-derived sensor bias as input feature.

Instead of learned embeddings, we compute:
  bias = P_observed - P_expected_from_physics

This is a PHYSICAL feature that generalizes to new sensors.
"""
import argparse
import os
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.physics_baseline import compute_physics_baseline, compute_virtual_temperature
from height_field_project.train_pinn import (
    PINNDataset, set_seed, parse_timestamp, 
    compute_prediction, train_epoch, evaluate
)

R_DRY_AIR = 287.05
G_STANDARD = 9.80665


def compute_sensor_bias(df: pd.DataFrame, p_ref: float) -> pd.DataFrame:
    """
    Compute physics-derived sensor bias.
    
    bias = P_obs - P_expected
    where P_expected = P_ref * exp(-h/H)
    
    This captures barometer calibration offset.
    """
    # Compute expected pressure from altitude
    t_celsius = df['avg_temperature'].values
    h = df['avg_altitude'].values
    p_obs = df['avg_pressure'].values
    
    # Virtual temperature
    e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
    rh = df['avg_humidity'].values / 100.0
    e = rh * e_sat
    r = 0.62198 * e / (p_obs - e)
    t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
    
    # Scale height
    H = R_DRY_AIR * t_v / G_STANDARD
    
    # Expected pressure from hypsometric equation
    p_expected = p_ref * np.exp(-h / H)
    
    # Bias is the residual
    df['pressure_bias'] = p_obs - p_expected
    
    return df


class BiasAwarePINNDataset(Dataset):
    """Dataset with pressure bias as additional feature."""
    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        z: np.ndarray,
        t: np.ndarray,
        temperature: np.ndarray,
        humidity: np.ndarray,
        pressure_bias: np.ndarray,  # NEW: physics-derived bias
        sensor_id: np.ndarray,
        p_obs: np.ndarray,
        h_gnss: np.ndarray,
        h_phys: np.ndarray,
        weights: Optional[np.ndarray] = None
    ):
        self.lat = torch.tensor(lat, dtype=torch.float32)
        self.lon = torch.tensor(lon, dtype=torch.float32)
        self.z = torch.tensor(z, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.temperature = torch.tensor(temperature, dtype=torch.float32)
        self.humidity = torch.tensor(humidity, dtype=torch.float32)
        self.pressure_bias = torch.tensor(pressure_bias, dtype=torch.float32)  # NEW
        self.sensor_id = torch.tensor(sensor_id, dtype=torch.long)
        self.p_obs = torch.tensor(p_obs, dtype=torch.float32)
        self.h_gnss = torch.tensor(h_gnss, dtype=torch.float32)
        self.h_phys = torch.tensor(h_phys, dtype=torch.float32)
        self.weights = torch.ones(len(lat)) if weights is None else torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self):
        return len(self.lat)
    
    def __getitem__(self, idx):
        return {
            'lat': self.lat[idx],
            'lon': self.lon[idx],
            'z': self.z[idx],
            't': self.t[idx],
            'temperature': self.temperature[idx],
            'humidity': self.humidity[idx],
            'pressure_bias': self.pressure_bias[idx],  # NEW
            'sensor_id': self.sensor_id[idx],
            'p_obs': self.p_obs[idx],
            'h_gnss': self.h_gnss[idx],
            'h_phys': self.h_phys[idx],
            'weight': self.weights[idx]
        }


class BiasAwarePINN(nn.Module):
    """PINN with pressure bias as input feature."""
    def __init__(
        self,
        base_model: GeneralizedPressureCorrectionPINN,
        bias_dim: int = 8  # Embedding dimension for bias
    ):
        super().__init__()
        self.base_model = base_model
        self.bias_dim = bias_dim
        
        # Bias encoding (simple MLP)
        self.bias_encoder = nn.Sequential(
            nn.Linear(1, bias_dim),
            nn.SiLU(),
            nn.Linear(bias_dim, bias_dim)
        )
        
        # Compute input dimensions
        # Original: hash(64) + z(1) + temporal(12) + T(1) + RH(1) = 79
        # New: 79 + bias_dim
        original_in_dim = 64 + 1 + 12 + 1 + 1  # 79
        new_in_dim = original_in_dim + bias_dim
        hidden_dim = 256
        
        # Build new MLP with correct input size
        from height_field_project.neural_field_pinn_generalized import SirenLayer
        
        layers = []
        # First layer takes new input size
        layers.append(SirenLayer(new_in_dim, hidden_dim, w0=1.0, is_first=True))
        # Hidden layers
        for _ in range(2):  # 2 hidden layers to match original
            layers.append(SirenLayer(hidden_dim, hidden_dim, w0=1.0))
        # Output
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize output to near zero
        with torch.no_grad():
            self.mlp[-1].weight.fill_(0.0)
            self.mlp[-1].bias.fill_(0.0)
    
    def forward(self, lat, lon, z, t, temperature, humidity, pressure_bias, sensor_id=None):
        # Encode bias
        bias_feat = self.bias_encoder(pressure_bias.unsqueeze(-1))  # [B, bias_dim]
        
        # Get original features from base model encoders
        lat_norm = (lat + 90.0) / 180.0
        lon_norm = lon % 360.0 / 360.0
        coords = torch.stack([lat_norm, lon_norm], dim=-1)
        
        h_spatial = self.base_model.hash_encoding(coords)  # [B, 64]
        h_temporal = self.base_model.temporal_encoding(t / 3600.0)  # [B, 12]
        
        features = torch.cat([
            h_spatial,  # 64
            z.unsqueeze(-1),  # 1
            h_temporal,  # 12
            temperature.unsqueeze(-1),  # 1
            humidity.unsqueeze(-1),  # 1
            bias_feat  # bias_dim
        ], dim=-1)
        
        return self.mlp(features).squeeze(-1)


def train_bias_aware_pinn(args: argparse.Namespace):
    """Train PINN with physics-derived bias feature."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print(f"Loading data: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Total samples: {len(df)}, Sensors: {df['uid'].nunique()}")
    
    # Compute physics baseline
    df, phys_params = compute_physics_baseline(
        df, p_ref=args.p_ref, t_ref_method="mean", convert_to_hae=False
    )
    print(f"P_ref: {phys_params.p_ref:.2f} Pa")
    
    # Compute sensor bias
    df = compute_sensor_bias(df, phys_params.p_ref)
    bias_stats = df.groupby('uid')['pressure_bias'].agg(['mean', 'std'])
    print(f"\nSensor bias statistics (Pa):")
    print(bias_stats.round(2))
    print(f"Overall bias std: {df['pressure_bias'].std():.2f} Pa")
    
    # Parse timestamps
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # LOSO split for testing
    if args.loso_test:
        sensors = df['uid'].unique()
        held_out_sensor = sensors[args.loso_fold % len(sensors)]
        
        test_df = df[df['uid'] == held_out_sensor].copy()
        train_df = df[df['uid'] != held_out_sensor].copy()
        
        # Train/val split
        n_val = int(len(train_df) * args.val_ratio)
        indices = np.random.permutation(len(train_df))
        val_df = train_df.iloc[indices[:n_val]]
        train_df = train_df.iloc[indices[n_val:]]
        
        print(f"\nLOSO Fold {args.loso_fold}: Held-out = {held_out_sensor[:25]}")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    else:
        # Standard random split
        n_test = int(len(df) * 0.15)
        n_val = int(len(df) * 0.15)
        indices = np.random.permutation(len(df))
        
        test_df = df.iloc[indices[:n_test]]
        val_df = df.iloc[indices[n_test:n_test+n_val]]
        train_df = df.iloc[n_test+n_val:]
        
        print(f"\nRandom Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create datasets
    def create_dataset(data_df):
        return BiasAwarePINNDataset(
            lat=data_df['avg_latitude'].values,
            lon=data_df['avg_longitude'].values,
            z=data_df['avg_altitude'].values,
            t=data_df['timestamp'].values,
            temperature=data_df['avg_temperature'].values,
            humidity=data_df['avg_humidity'].values,
            pressure_bias=data_df['pressure_bias'].values,
            sensor_id=np.zeros(len(data_df), dtype=np.int64),
            p_obs=data_df['avg_pressure'].values,
            h_gnss=data_df['avg_altitude'].values,
            h_phys=data_df['h_phys_hae'].values
        )
    
    train_ds = create_dataset(train_df)
    val_ds = create_dataset(val_df)
    test_ds = create_dataset(test_df)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Create model with bias feature
    base_model = GeneralizedPressureCorrectionPINN(
        hash_levels=args.hash_levels,
        hash_features=args.hash_features,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        temporal_freqs=args.temporal_freqs,
        dropout=args.dropout,
        use_siren=args.use_siren
    )
    
    # Wrap with bias-aware layer
    model = BiasAwarePINN(base_model, bias_dim=args.bias_dim).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.restart_epochs, T_mult=2
    )
    
    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0
    
    print(f"\nTraining for up to {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_data = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward with bias
            delta_p = model(
                batch['lat'].to(device),
                batch['lon'].to(device),
                batch['z'].to(device),
                batch['t'].to(device),
                batch['temperature'].to(device),
                batch['humidity'].to(device),
                batch['pressure_bias'].to(device)
            )
            
            # Compute predicted height
            p_corrected = batch['p_obs'].to(device) + delta_p
            t_celsius = batch['temperature'].to(device)
            e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (batch['humidity'].to(device) / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            H = 287.05 * t_v / 9.80665
            h_pred = H * torch.log(phys_params.p_ref / p_corrected)
            
            # Loss
            loss = torch.mean(torch.abs(h_pred - batch['h_gnss'].to(device)))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch['lat'])
            total_data += len(batch['lat'])
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_gnss = []
        val_phys = []
        
        with torch.no_grad():
            for batch in val_loader:
                delta_p = model(
                    batch['lat'].to(device),
                    batch['lon'].to(device),
                    batch['z'].to(device),
                    batch['t'].to(device),
                    batch['temperature'].to(device),
                    batch['humidity'].to(device),
                    batch['pressure_bias'].to(device)
                )
                
                p_corrected = batch['p_obs'].to(device) + delta_p
                t_celsius = batch['temperature'].to(device)
                e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
                e = (batch['humidity'].to(device) / 100.0) * e_sat
                r = 0.62198 * e / (p_corrected - e)
                t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
                H = 287.05 * t_v / 9.80665
                h_pred = H * torch.log(phys_params.p_ref / p_corrected)
                
                val_preds.append(h_pred.cpu())
                val_gnss.append(batch['h_gnss'])
                val_phys.append(batch['h_phys'])
        
        val_preds = torch.cat(val_preds)
        val_gnss = torch.cat(val_gnss)
        val_phys = torch.cat(val_phys)
        
        val_mae = torch.mean(torch.abs(val_preds - val_gnss)).item()
        val_baseline_mae = torch.mean(torch.abs(val_phys - val_gnss)).item()
        val_improvement = (1 - val_mae / val_baseline_mae) * 100
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:04d} | Train: {total_loss/total_data:.4f} | "
                  f"Val MAE: {val_mae:.3f}m | Improvement: {val_improvement:.1f}%")
        
        # Early stopping
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best and test
    model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_gnss = []
    test_phys = []
    
    with torch.no_grad():
        for batch in test_loader:
            delta_p = model(
                batch['lat'].to(device),
                batch['lon'].to(device),
                batch['z'].to(device),
                batch['t'].to(device),
                batch['temperature'].to(device),
                batch['humidity'].to(device),
                batch['pressure_bias'].to(device)
            )
            
            p_corrected = batch['p_obs'].to(device) + delta_p
            t_celsius = batch['temperature'].to(device)
            e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (batch['humidity'].to(device) / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            H = 287.05 * t_v / 9.80665
            h_pred = H * torch.log(phys_params.p_ref / p_corrected)
            
            test_preds.append(h_pred.cpu())
            test_gnss.append(batch['h_gnss'])
            test_phys.append(batch['h_phys'])
    
    test_preds = torch.cat(test_preds)
    test_gnss = torch.cat(test_gnss)
    test_phys = torch.cat(test_phys)
    
    test_mae = torch.mean(torch.abs(test_preds - test_gnss)).item()
    test_rmse = torch.sqrt(torch.mean((test_preds - test_gnss)**2)).item()
    test_baseline_mae = torch.mean(torch.abs(test_phys - test_gnss)).item()
    test_improvement = (1 - test_mae / test_baseline_mae) * 100
    
    print(f"\n{'='*60}")
    print("BIAS-AWARE GENERALIZED PINN RESULTS")
    print(f"{'='*60}")
    print(f"Test Baseline MAE: {test_baseline_mae:.3f} m")
    print(f"Test PINN MAE:     {test_mae:.3f} m")
    print(f"Test PINN RMSE:    {test_rmse:.3f} m")
    print(f"Improvement:       {test_improvement:.1f}%")
    print(f"{'='*60}")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'phys_params': phys_params,
        'args': vars(args),
    }
    torch.save(checkpoint, os.path.join(args.output_dir, f"model_bias_aware_fold{args.loso_fold}.pt"))
    
    return {
        'mae': test_mae,
        'rmse': test_rmse,
        'mae_baseline': test_baseline_mae,
        'improvement': test_improvement
    }


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, default="data/sensor_data_filtered.csv")
    p.add_argument("--output_dir", type=str, default="height_field_project/artifacts_bias_aware")
    
    p.add_argument("--loso_test", action="store_true")
    p.add_argument("--loso_fold", type=int, default=0)
    
    p.add_argument("--p_ref", type=float, default=None)
    p.add_argument("--bias_dim", type=int, default=8)
    
    p.add_argument("--hash_levels", type=int, default=16)
    p.add_argument("--hash_features", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_hidden_layers", type=int, default=3)
    p.add_argument("--temporal_freqs", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_siren", action="store_true", default=True)
    
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--restart_epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train_bias_aware_pinn(args)
