"""
Training script for Generalized PINN (sensor-agnostic).

Trains a model WITHOUT sensor embeddings for true spatial generalization.
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


def create_datasets_no_sensor(
    df: pd.DataFrame,
    val_ratio: float = 0.15
) -> Tuple[PINNDataset, PINNDataset]:
    """
    Create train/val datasets without sensor stratification.
    Pure random split for spatial generalization testing.
    """
    # Parse timestamps
    df = df.copy()
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # Random split (NOT stratified by sensor)
    n_val = int(len(df) * val_ratio)
    indices = np.random.permutation(len(df))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # Dummy sensor_idx = 0 for all (not used by model)
    train_ds = PINNDataset(
        lat=train_df['avg_latitude'].values,
        lon=train_df['avg_longitude'].values,
        z=train_df['avg_altitude'].values,
        t=train_df['timestamp'].values,
        temperature=train_df['avg_temperature'].values,
        humidity=train_df['avg_humidity'].values,
        sensor_id=np.zeros(len(train_df), dtype=np.int64),  # Dummy
        p_obs=train_df['avg_pressure'].values,
        h_gnss=train_df['avg_altitude'].values,
        h_phys=train_df['h_phys_hae'].values
    )
    
    val_ds = PINNDataset(
        lat=val_df['avg_latitude'].values,
        lon=val_df['avg_longitude'].values,
        z=val_df['avg_altitude'].values,
        t=val_df['timestamp'].values,
        temperature=val_df['avg_temperature'].values,
        humidity=val_df['avg_humidity'].values,
        sensor_id=np.zeros(len(val_df), dtype=np.int64),  # Dummy
        p_obs=val_df['avg_pressure'].values,
        h_gnss=val_df['avg_altitude'].values,
        h_phys=val_df['h_phys_hae'].values
    )
    
    return train_ds, val_ds


def train_generalized_pinn(args: argparse.Namespace):
    """Train sensor-agnostic PINN."""
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
    
    # Create train/val datasets (random split, NOT stratified)
    train_ds, val_ds = create_datasets_no_sensor(df, val_ratio=args.val_ratio)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Create test set (hold out specific sensors)
    if args.loso_test:
        # Strict LOSO: hold out one sensor completely
        held_out_sensor = df['uid'].unique()[0]
        test_df = df[df['uid'] == held_out_sensor].copy()
        train_df = df[df['uid'] != held_out_sensor].copy()
        
        # Recompute train/val from remaining sensors
        train_val_df = train_df
        train_val_df['timestamp'] = train_val_df['processed_time'].apply(parse_timestamp)
        test_df['timestamp'] = test_df['processed_time'].apply(parse_timestamp)
        
        n_val = int(len(train_val_df) * args.val_ratio)
        indices = np.random.permutation(len(train_val_df))
        val_df = train_val_df.iloc[indices[:n_val]]
        train_df = train_val_df.iloc[indices[n_val:]]
        
        train_ds = PINNDataset(
            lat=train_df['avg_latitude'].values, lon=train_df['avg_longitude'].values,
            z=train_df['avg_altitude'].values, t=train_df['timestamp'].values,
            temperature=train_df['avg_temperature'].values, humidity=train_df['avg_humidity'].values,
            sensor_id=np.zeros(len(train_df), dtype=np.int64),
            p_obs=train_df['avg_pressure'].values, h_gnss=train_df['avg_altitude'].values,
            h_phys=train_df['h_phys_hae'].values
        )
        val_ds = PINNDataset(
            lat=val_df['avg_latitude'].values, lon=val_df['avg_longitude'].values,
            z=val_df['avg_altitude'].values, t=val_df['timestamp'].values,
            temperature=val_df['avg_temperature'].values, humidity=val_df['avg_humidity'].values,
            sensor_id=np.zeros(len(val_df), dtype=np.int64),
            p_obs=val_df['avg_pressure'].values, h_gnss=val_df['avg_altitude'].values,
            h_phys=val_df['h_phys_hae'].values
        )
        test_ds = PINNDataset(
            lat=test_df['avg_latitude'].values, lon=test_df['avg_longitude'].values,
            z=test_df['avg_altitude'].values, t=test_df['timestamp'].values,
            temperature=test_df['avg_temperature'].values, humidity=test_df['avg_humidity'].values,
            sensor_id=np.zeros(len(test_df), dtype=np.int64),
            p_obs=test_df['avg_pressure'].values, h_gnss=test_df['avg_altitude'].values,
            h_phys=test_df['h_phys_hae'].values
        )
        
        print(f"LOSO Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} (held-out: {held_out_sensor[:20]})")
    else:
        test_ds = val_ds  # Use same for simplicity
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Create model (NO sensor embedding!)
    model = GeneralizedPressureCorrectionPINN(
        hash_levels=args.hash_levels,
        hash_features=args.hash_features,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        temporal_freqs=args.temporal_freqs,
        dropout=args.dropout,
        use_siren=args.use_siren
    ).to(device)
    
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
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, scheduler,
            phys_params.p_ref, device, args.lambda_hydro
        )
        
        val_metrics = evaluate(model, val_loader, phys_params.p_ref, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:04d} | Train: {train_loss:.4f} | "
                  f"Val MAE: {val_metrics['mae']:.3f}m | "
                  f"Improvement: {val_metrics['improvement']:.1f}%")
        
        # Early stopping
        if val_metrics['mae'] < best_val_mae - 1e-4:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best and evaluate on test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, phys_params.p_ref, device)
    
    print(f"\n{'='*60}")
    print("GENERALIZED PINN RESULTS (NO SENSOR EMBEDDINGS)")
    print(f"{'='*60}")
    print(f"Test Baseline MAE: {test_metrics['mae_baseline']:.3f} m")
    print(f"Test PINN MAE:     {test_metrics['mae']:.3f} m")
    print(f"Test PINN RMSE:    {test_metrics['rmse']:.3f} m")
    print(f"Improvement:       {test_metrics['improvement']:.1f}%")
    print(f"{'='*60}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'phys_params': phys_params,
        'args': vars(args),
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "model_generalized.pt"))
    print(f"\nModel saved to: {args.output_dir}/model_generalized.pt")
    
    return test_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train sensor-agnostic PINN")
    
    p.add_argument("--input_csv", type=str, default="data/sensor_data_filtered.csv")
    p.add_argument("--output_dir", type=str, default="height_field_project/artifacts_generalized")
    
    # Evaluation mode
    p.add_argument("--loso_test", action="store_true", help="Use strict LOSO (hold out 1 sensor)")
    
    # Physics
    p.add_argument("--p_ref", type=float, default=None)
    p.add_argument("--lambda_hydro", type=float, default=0.001)
    
    # Model (sensor-agnostic - no sensor_embedding_dim!)
    p.add_argument("--hash_levels", type=int, default=16)
    p.add_argument("--hash_features", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_hidden_layers", type=int, default=3)
    p.add_argument("--temporal_freqs", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_siren", action="store_true", default=True)
    
    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--restart_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train_generalized_pinn(args)
