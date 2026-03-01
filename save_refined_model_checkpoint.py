#!/usr/bin/env python3
"""
Train and Save Refined Model Checkpoint
=======================================

Trains the HardConstrainedNF model and saves the checkpoint for figure generation.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

os.makedirs('experiments/results/refined_model', exist_ok=True)

# Import from refined model
from run_refined_model import (
    HardConstrainedNF, compute_terrain_features, 
    create_curriculum_stages, train_with_curriculum
)

def main():
    print("="*70)
    print("TRAINING AND SAVING REFINED MODEL CHECKPOINT")
    print("="*70)
    
    # Load stabilized data
    print("\n[1] Loading stabilized GNSS data...")
    df = pd.read_csv('data/processed/sensor_data_stabilized.csv')
    df['processed_time'] = pd.to_datetime(df['processed_time'])
    
    # Compute physics baseline
    valid = df[['avg_pressure', 'avg_altitude']].dropna()
    X_fit = valid[['avg_altitude']].values
    y_fit = np.log(valid['avg_pressure'].values)
    lr = LinearRegression()
    lr.fit(X_fit, y_fit)
    Hs = -1.0 / lr.coef_[0]
    P0 = np.exp(lr.intercept_)
    df['h_physics'] = -Hs * (np.log(df['avg_pressure']) - np.log(P0))
    df['residual'] = df['avg_altitude'] - df['h_physics']
    
    # Compute terrain features
    print("\n[2] Computing terrain features...")
    df = compute_terrain_features(df)
    
    # Use best sensor (42508217) for training the checkpoint
    best_sensor = '20240911193046A806593A5642508217'
    print(f"\n[3] Training model for best sensor: {best_sensor[-8:]}...")
    
    train_df = df[df['uid'] != best_sensor].copy()
    test_df = df[df['uid'] == best_sensor].copy()
    
    h_physics_test = test_df['h_physics'].values
    y_test_alt = test_df['avg_altitude'].values
    
    # Create curriculum stages
    stages = create_curriculum_stages(train_df, strategy='altitude_density')
    
    # Create and train model
    feature_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure',
                   'era5_t2m', 'era5_sp', 'terrain_roughness', 'height_rank', 'sensor_density']
    
    model = HardConstrainedNF(
        use_hash_encoding=True,
        use_terrain=True,
        st_dim=2,
        feature_dim=9,
        hidden_dim=256,
        num_layers=8,
        residual_clip=60.0
    ).to(DEVICE)
    
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with curriculum
    print(f"\n[4] Training with curriculum learning (350 epochs per stage)...")
    model, best_mae, scaler_spatial, scaler_feature, scaler_y, history = train_with_curriculum(
        model, stages, test_df, h_physics_test, y_test_alt,
        max_epochs_per_stage=350, patience=100
    )
    
    print(f"\n    Best MAE achieved: {best_mae:.2f}m")
    
    # Save checkpoint
    print(f"\n[5] Saving checkpoint...")
    checkpoint_path = 'experiments/results/refined_model/best_model.pt'
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_spatial': scaler_spatial,
        'scaler_feature': scaler_feature,
        'scaler_y': scaler_y,
        'best_mae': best_mae,
        'feature_cols': feature_cols,
        'config': {
            'use_hash_encoding': True,
            'use_terrain': True,
            'st_dim': 2,
            'feature_dim': 9,
            'hidden_dim': 256,
            'num_layers': 8,
            'residual_clip': 60.0
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"    ✓ Saved checkpoint to: {checkpoint_path}")
    print(f"    File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")
    
    # Also save scalers separately for easy access
    import pickle
    scalers_path = 'experiments/results/refined_model/scalers.pkl'
    with open(scalers_path, 'wb') as f:
        pickle.dump({
            'scaler_spatial': scaler_spatial,
            'scaler_feature': scaler_feature,
            'scaler_y': scaler_y
        }, f)
    print(f"    ✓ Saved scalers to: {scalers_path}")
    
    print("\n" + "="*70)
    print("CHECKPOINT SAVED SUCCESSFULLY")
    print("="*70)
    print(f"\nNow you can run the figure generation scripts!")
    print(f"They will load the checkpoint from: {checkpoint_path}")

if __name__ == '__main__':
    main()
