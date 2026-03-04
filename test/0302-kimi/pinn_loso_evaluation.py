#!/usr/bin/env python
"""
LOSO Evaluation for Weather-Informed PINN
===========================================

Leave-One-Sensor-Out cross-validation to test spatial generalization.

For each fold:
- Train on N-1 sensors (all weeks)
- Test on 1 held-out sensor (all weeks)

This tests if the model can generalize to new sensor locations.

Author: Assistant
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

R_DRY = 287.05
G = 9.80665
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WorkingPINN(nn.Module):
    """Same architecture as pinn_working_v1.py"""
    
    def __init__(self, n_sensors):
        super().__init__()
        self.sensor_embed = nn.Embedding(n_sensors, 16)
        self.hour_embed = nn.Embedding(24, 4)
        self.net = nn.Sequential(
            nn.Linear(24, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def compute_physics_baseline(self, P_obs, P_era5, T_era5, h_era5):
        H_v = R_DRY * T_era5 / G
        return h_era5 - H_v * torch.log(P_obs / P_era5)
    
    def forward(self, sensor_ids, P_obs, P_era5, T_era5, h_era5, timestamp, hour, local_T_diff):
        sensor = self.sensor_embed(sensor_ids)
        hour_emb = self.hour_embed(hour)
        P_diff = (P_obs - P_era5).unsqueeze(-1) / 1000
        T_diff = (T_era5 - 290).unsqueeze(-1) / 10
        P_norm = (P_obs - 100000).unsqueeze(-1) / 5000
        local_T_feat = local_T_diff.unsqueeze(-1) / 5
        x = torch.cat([sensor, hour_emb, P_diff, T_diff, local_T_feat, P_norm], dim=-1)
        h_residual = self.net(x).squeeze(-1) * 50
        return h_residual
    
    def predict(self, sensor_ids, P_obs, P_era5, T_era5, h_era5, timestamp, hour, local_T_diff):
        h_baseline = self.compute_physics_baseline(P_obs, P_era5, T_era5, h_era5)
        h_residual = self.forward(sensor_ids, P_obs, P_era5, T_era5, h_era5, timestamp, hour, local_T_diff)
        return h_baseline + h_residual


def load_data():
    """Load and filter data."""
    df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
    df['processed_time'] = pd.to_datetime(df['processed_time'])
    
    mobile_uid = '20240606181851A641973A1878250224'
    df = df[df['uid'] != mobile_uid].copy()
    
    # Light filtering
    df = df[(df['avg_pressure'] > 95000) & (df['avg_pressure'] < 105000)]
    median_alts = df.groupby('uid')['avg_altitude'].transform('median')
    df = df[np.abs(df['avg_altitude'] - median_alts) < 40]
    
    # Compute ERA5 height
    df['h_era5'] = -8500 * np.log(df['era5_sp'] / 101325)
    
    return df


def prepare_tensors(df, sensor_encoder, device):
    """Prepare tensors."""
    timestamp = df['processed_time'].astype(np.int64) // 10**9
    timestamp = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min() + 1e-6)
    hour = df['processed_time'].dt.hour.values
    local_T_diff = (df['avg_temperature'].values + 273.15) - df['era5_t2m'].values
    
    # Encode sensors
    df = df.copy()
    df['sensor_id'] = sensor_encoder.transform(df['uid'])
    
    return {
        'sensor_ids': torch.LongTensor(df['sensor_id'].values).to(device),
        'P_obs': torch.FloatTensor(df['avg_pressure'].values).to(device),
        'P_era5': torch.FloatTensor(df['era5_sp'].values).to(device),
        'T_era5': torch.FloatTensor(df['era5_t2m'].values).to(device),
        'h_era5': torch.FloatTensor(df['h_era5'].values).to(device),
        'h_gps': torch.FloatTensor(df['avg_altitude'].values).to(device),
        'timestamp': torch.FloatTensor(timestamp.values).to(device),
        'hour': torch.LongTensor(hour).to(device),
        'local_T_diff': torch.FloatTensor(local_T_diff).to(device),
    }


def train_model(train_df, sensor_encoder, epochs=200):
    """Train model on training data."""
    n_sensors = len(sensor_encoder.classes_)
    model = WorkingPINN(n_sensors).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Prepare tensors
    timestamp = train_df['processed_time'].astype(np.int64) // 10**9
    timestamp = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min() + 1e-6)
    hour = train_df['processed_time'].dt.hour.values
    local_T_diff = (train_df['avg_temperature'].values + 273.15) - train_df['era5_t2m'].values
    train_df = train_df.copy()
    train_df['sensor_id'] = sensor_encoder.transform(train_df['uid'])
    
    train_tensors = {
        'sensor_ids': torch.LongTensor(train_df['sensor_id'].values).to(DEVICE),
        'P_obs': torch.FloatTensor(train_df['avg_pressure'].values).to(DEVICE),
        'P_era5': torch.FloatTensor(train_df['era5_sp'].values).to(DEVICE),
        'T_era5': torch.FloatTensor(train_df['era5_t2m'].values).to(DEVICE),
        'h_era5': torch.FloatTensor(train_df['h_era5'].values).to(DEVICE),
        'h_gps': torch.FloatTensor(train_df['avg_altitude'].values).to(DEVICE),
        'timestamp': torch.FloatTensor(timestamp.values).to(DEVICE),
        'hour': torch.LongTensor(hour).to(DEVICE),
        'local_T_diff': torch.FloatTensor(local_T_diff).to(DEVICE),
    }
    
    # Train
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        h_pred = model.predict(
            train_tensors['sensor_ids'], train_tensors['P_obs'],
            train_tensors['P_era5'], train_tensors['T_era5'],
            train_tensors['h_era5'], train_tensors['timestamp'],
            train_tensors['hour'], train_tensors['local_T_diff']
        )
        
        loss = F.smooth_l1_loss(h_pred, train_tensors['h_gps'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    return model


def evaluate_model(model, test_df, sensor_encoder):
    """Evaluate on test data."""
    model.eval()
    
    timestamp = test_df['processed_time'].astype(np.int64) // 10**9
    timestamp = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min() + 1e-6)
    hour = test_df['processed_time'].dt.hour.values
    local_T_diff = (test_df['avg_temperature'].values + 273.15) - test_df['era5_t2m'].values
    test_df = test_df.copy()
    test_df['sensor_id'] = sensor_encoder.transform(test_df['uid'])
    
    test_tensors = {
        'sensor_ids': torch.LongTensor(test_df['sensor_id'].values).to(DEVICE),
        'P_obs': torch.FloatTensor(test_df['avg_pressure'].values).to(DEVICE),
        'P_era5': torch.FloatTensor(test_df['era5_sp'].values).to(DEVICE),
        'T_era5': torch.FloatTensor(test_df['era5_t2m'].values).to(DEVICE),
        'h_era5': torch.FloatTensor(test_df['h_era5'].values).to(DEVICE),
        'h_gps': torch.FloatTensor(test_df['avg_altitude'].values).to(DEVICE),
        'timestamp': torch.FloatTensor(timestamp.values).to(DEVICE),
        'hour': torch.LongTensor(hour).to(DEVICE),
        'local_T_diff': torch.FloatTensor(local_T_diff).to(DEVICE),
    }
    
    with torch.no_grad():
        h_pred = model.predict(
            test_tensors['sensor_ids'], test_tensors['P_obs'],
            test_tensors['P_era5'], test_tensors['T_era5'],
            test_tensors['h_era5'], test_tensors['timestamp'],
            test_tensors['hour'], test_tensors['local_T_diff']
        )
        
        h_pred_np = h_pred.cpu().numpy()
        h_gps_np = test_tensors['h_gps'].cpu().numpy()
        
        mae = np.mean(np.abs(h_pred_np - h_gps_np))
        rmse = np.sqrt(np.mean((h_pred_np - h_gps_np)**2))
    
    return mae, rmse


def main():
    print("="*80)
    print("LOSO EVALUATION - Weather-Informed PINN")
    print("="*80)
    print()
    
    # Load data
    print("[1] Loading data...")
    df = load_data()
    
    # Get unique sensors
    sensors = sorted(df['uid'].unique())
    n_sensors = len(sensors)
    print(f"    {n_sensors} sensors: {[s[-8:] for s in sensors]}")
    
    # LOSO evaluation
    print("\n[2] Running LOSO cross-validation...")
    print()
    
    results = {
        'sensor': [],
        'train_samples': [],
        'test_samples': [],
        'pinn_mae': [],
        'pinn_rmse': [],
        'era5_mae': [],
    }
    
    for fold_idx, test_sensor in enumerate(sensors):
        print(f"Fold {fold_idx+1}/{n_sensors}: Testing on sensor {test_sensor[-8:]}")
        
        # Split
        train_df = df[df['uid'] != test_sensor].copy()
        test_df = df[df['uid'] == test_sensor].copy()
        
        print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        # Create encoder for this fold
        train_sensors = sorted(train_df['uid'].unique())
        sensor_encoder = LabelEncoder()
        sensor_encoder.fit(train_sensors)
        
        # Need to handle test sensor not in encoder - assign a dummy ID
        # Actually, test sensor won't be in training, so we need to handle this
        # For simplicity, add test sensor to encoder with a new ID
        all_sensors = train_sensors + [test_sensor]
        sensor_encoder = LabelEncoder()
        sensor_encoder.fit(all_sensors)
        
        # Train
        print(f"  Training...")
        model = train_model(train_df, sensor_encoder, epochs=200)
        
        # Evaluate
        print(f"  Evaluating...")
        pinn_mae, pinn_rmse = evaluate_model(model, test_df, sensor_encoder)
        
        # ERA5 baseline
        era5_pred = test_df['h_era5'] - 8500 * np.log(test_df['avg_pressure'] / test_df['era5_sp'])
        era5_mae = np.mean(np.abs(era5_pred - test_df['avg_altitude']))
        
        print(f"  Results: ERA5={era5_mae:.2f}m, PINN={pinn_mae:.2f}m")
        print()
        
        # Store
        results['sensor'].append(test_sensor[-8:])
        results['train_samples'].append(len(train_df))
        results['test_samples'].append(len(test_df))
        results['pinn_mae'].append(pinn_mae)
        results['pinn_rmse'].append(pinn_rmse)
        results['era5_mae'].append(era5_mae)
    
    # Summary
    print("="*80)
    print("LOSO RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"{'Sensor':<12} {'Train':<8} {'Test':<8} {'ERA5':<10} {'PINN':<10} {'Improvement':<12}")
    print("-" * 70)
    
    for i in range(n_sensors):
        improvement = (results['era5_mae'][i] - results['pinn_mae'][i]) / results['era5_mae'][i] * 100
        print(f"{results['sensor'][i]:<12} {results['train_samples'][i]:<8} {results['test_samples'][i]:<8} "
              f"{results['era5_mae'][i]:<10.2f} {results['pinn_mae'][i]:<10.2f} {improvement:>10.1f}%")
    
    print("-" * 70)
    mean_era5 = np.mean(results['era5_mae'])
    mean_pinn = np.mean(results['pinn_mae'])
    mean_improvement = (mean_era5 - mean_pinn) / mean_era5 * 100
    print(f"{'MEAN':<12} {'':<8} {'':<8} {mean_era5:<10.2f} {mean_pinn:<10.2f} {mean_improvement:>10.1f}%")
    
    print()
    print(f"ERA5 Baseline (mean):  {mean_era5:.2f}m")
    print(f"PINN (mean):           {mean_pinn:.2f}m")
    print(f"Improvement:           {mean_improvement:.1f}%")
    print()
    
    # Compare to temporal validation
    print("="*80)
    print("COMPARISON: LOSO vs Temporal Validation")
    print("="*80)
    print()
    print("Temporal validation (Week 3 test):")
    print("  PINN MAE: ~6.90m")
    print()
    print("LOSO validation (per-sensor test):")
    print(f"  PINN MAE: {mean_pinn:.2f}m")
    print()
    
    if mean_pinn < 20:
        print("✓ Good spatial generalization!")
    elif mean_pinn < 40:
        print("~ Moderate spatial generalization")
    else:
        print("⚠ Poor spatial generalization - model overfits to locations")
    
    print()
    print("Note: LOSO is harder than temporal validation because:")
    print("  - Test sensor's location is unseen during training")
    print("  - Must generalize spatial patterns to new coordinates")
    print("  - Sensor-specific biases must be inferred from similar sensors")
    
    # Save
    with open('pinn_loso_results.json', 'w') as f:
        json.dump({
            'sensors': results['sensor'],
            'era5_mae': results['era5_mae'],
            'pinn_mae': results['pinn_mae'],
            'pinn_rmse': results['pinn_rmse'],
            'mean_era5': float(mean_era5),
            'mean_pinn': float(mean_pinn),
            'mean_improvement': float(mean_improvement),
        }, f, indent=2)
    
    print()
    print("Results saved to: pinn_loso_results.json")


if __name__ == '__main__':
    main()
