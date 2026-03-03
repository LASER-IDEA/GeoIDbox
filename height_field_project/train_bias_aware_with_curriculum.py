"""
Bias-Aware Generalized PINN WITH Curriculum Learning

Tests whether curriculum learning helps when the base model is sound.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from height_field_project.train_generalized_with_bias import (
    BiasAwarePINN, compute_sensor_bias, BiasAwarePINNDataset
)
from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_pinn import set_seed, parse_timestamp


def create_curriculum_splits(df: pd.DataFrame):
    """
    Create 3-stage curriculum splits based on altitude.
    
    Stage 1 (Easy): h < 100m (low altitude, dense coverage)
    Stage 2 (Medium): h < 200m (add moderate altitudes)
    Stage 3 (Hard): Full dataset (all altitudes including high)
    """
    altitudes = df['avg_altitude'].values
    
    # Stage 1: Easy - low altitude
    mask_1 = altitudes < 100
    
    # Stage 2: Medium - moderate altitude
    mask_2 = altitudes < 200
    
    # Stage 3: Hard - full dataset
    mask_3 = np.ones(len(df), dtype=bool)
    
    return mask_1, mask_2, mask_3


def train_epoch(model, loader, optimizer, scheduler, phys_params, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_data = 0
    
    R_DRY_AIR = 287.05
    G_STANDARD = 9.80665
    
    for batch in loader:
        optimizer.zero_grad()
        
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
        H = R_DRY_AIR * t_v / G_STANDARD
        h_pred = H * torch.log(phys_params.p_ref / p_corrected)
        
        loss = torch.mean(torch.abs(h_pred - batch['h_gnss'].to(device)))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch['lat'])
        total_data += len(batch['lat'])
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / total_data


def evaluate(model, loader, phys_params, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_gnss = []
    all_phys = []
    
    R_DRY_AIR = 287.05
    G_STANDARD = 9.80665
    
    with torch.no_grad():
        for batch in loader:
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
            H = R_DRY_AIR * t_v / G_STANDARD
            h_pred = H * torch.log(phys_params.p_ref / p_corrected)
            
            all_preds.append(h_pred.cpu())
            all_gnss.append(batch['h_gnss'])
            all_phys.append(batch['h_phys'])
    
    all_preds = torch.cat(all_preds)
    all_gnss = torch.cat(all_gnss)
    all_phys = torch.cat(all_phys)
    
    mae = torch.mean(torch.abs(all_preds - all_gnss)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_gnss)**2)).item()
    baseline_mae = torch.mean(torch.abs(all_phys - all_gnss)).item()
    improvement = (1 - mae / baseline_mae) * 100
    
    return {'mae': mae, 'rmse': rmse, 'mae_baseline': baseline_mae, 'improvement': improvement}


def train_with_curriculum(args):
    """Train Bias-Aware PINN with curriculum learning."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print(f"Loading data: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Total samples: {len(df)}")
    
    # Compute physics baseline
    df, phys_params = compute_physics_baseline(
        df, p_ref=args.p_ref, t_ref_method="mean", convert_to_hae=False
    )
    print(f"P_ref: {phys_params.p_ref:.2f} Pa")
    
    # Compute sensor bias
    df = compute_sensor_bias(df, phys_params.p_ref)
    print(f"Bias std: {df['pressure_bias'].std():.2f} Pa")
    
    # Parse timestamps
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # LOSO split
    sensors = df['uid'].unique()
    held_out_sensor = sensors[args.loso_fold % len(sensors)]
    
    test_df = df[df['uid'] == held_out_sensor].copy()
    train_df = df[df['uid'] != held_out_sensor].copy()
    
    # Train/val split
    train_indices = train_df.index.values
    n_val = int(len(train_indices) * args.val_ratio)
    
    np.random.seed(args.seed)
    perm = np.random.permutation(len(train_indices))
    val_indices = train_indices[perm[:n_val]]
    train_indices = train_indices[perm[n_val:]]
    
    print(f"\nLOSO Fold {args.loso_fold}: Held-out = {held_out_sensor[:25]}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_df)}")
    
    # Create curriculum splits
    mask_1, mask_2, mask_3 = create_curriculum_splits(train_df.loc[train_indices])
    print(f"\nCurriculum splits (train only):")
    print(f"  Stage 1 (Easy, h<100m): {mask_1.sum()} samples ({mask_1.mean()*100:.1f}%)")
    print(f"  Stage 2 (Medium, h<200m): {mask_2.sum()} samples ({mask_2.mean()*100:.1f}%)")
    print(f"  Stage 3 (Hard, full): {mask_3.sum()} samples (100%)")
    
    # Create datasets
    def create_dataset(indices):
        return BiasAwarePINNDataset(
            lat=df.loc[indices]['avg_latitude'].values,
            lon=df.loc[indices]['avg_longitude'].values,
            z=df.loc[indices]['avg_altitude'].values,
            t=df.loc[indices]['timestamp'].values,
            temperature=df.loc[indices]['avg_temperature'].values,
            humidity=df.loc[indices]['avg_humidity'].values,
            pressure_bias=df.loc[indices]['pressure_bias'].values,
            sensor_id=np.zeros(len(indices), dtype=np.int64),
            p_obs=df.loc[indices]['avg_pressure'].values,
            h_gnss=df.loc[indices]['avg_altitude'].values,
            h_phys=df.loc[indices]['h_phys_hae'].values
        )
    
    # Full datasets
    ds_train = create_dataset(train_indices)
    ds_val = create_dataset(val_indices)
    ds_test = create_dataset(test_df.index.values)
    
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    base_model = GeneralizedPressureCorrectionPINN(
        hash_levels=args.hash_levels,
        hash_features=args.hash_features,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        temporal_freqs=args.temporal_freqs,
        dropout=args.dropout,
        use_siren=args.use_siren
    )
    model = BiasAwarePINN(base_model, bias_dim=args.bias_dim).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # THREE-STAGE CURRICULUM TRAINING
    print(f"\n{'='*60}")
    print("CURRICULUM TRAINING - BIAS-AWARE PINN")
    print(f"{'='*60}")
    
    best_val_mae = float('inf')
    
    # Stage 1: Easy (low altitude)
    if mask_1.sum() > 1000:
        print(f"\n--- Stage 1: Easy (h<100m) ---")
        stage1_indices = train_indices[mask_1]
        ds_stage1 = create_dataset(stage1_indices)
        stage1_loader = DataLoader(ds_stage1, batch_size=args.batch_size, shuffle=True)
        
        for epoch in range(args.stage_epochs):
            train_loss = train_epoch(model, stage1_loader, optimizer, None, phys_params, device)
            if (epoch + 1) % 10 == 0:
                val_metrics = evaluate(model, val_loader, phys_params, device)
                print(f"  Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val MAE: {val_metrics['mae']:.3f}m")
    
    # Stage 2: Medium (moderate altitude)
    if mask_2.sum() > 1000:
        print(f"\n--- Stage 2: Medium (h<200m) ---")
        stage2_indices = train_indices[mask_2]
        ds_stage2 = create_dataset(stage2_indices)
        stage2_loader = DataLoader(ds_stage2, batch_size=args.batch_size, shuffle=True)
        
        for epoch in range(args.stage_epochs):
            train_loss = train_epoch(model, stage2_loader, optimizer, None, phys_params, device)
            if (epoch + 1) % 10 == 0:
                val_metrics = evaluate(model, val_loader, phys_params, device)
                print(f"  Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val MAE: {val_metrics['mae']:.3f}m")
                
                if val_metrics['mae'] < best_val_mae:
                    best_val_mae = val_metrics['mae']
                    best_state = model.state_dict()
    
    # Stage 3: Hard (full dataset)
    print(f"\n--- Stage 3: Hard (full dataset) ---")
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.restart_epochs, T_mult=2
    )
    
    patience_counter = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, phys_params, device)
        
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, val_loader, phys_params, device)
            print(f"  Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val MAE: {val_metrics['mae']:.3f}m | Improvement: {val_metrics['improvement']:.1f}%")
            
            if val_metrics['mae'] < best_val_mae - 1e-4:
                best_val_mae = val_metrics['mae']
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best and test
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    
    test_metrics = evaluate(model, test_loader, phys_params, device)
    
    print(f"\n{'='*60}")
    print("BIAS-AWARE PINN WITH CURRICULUM LEARNING - RESULTS")
    print(f"{'='*60}")
    print(f"Test Baseline MAE: {test_metrics['mae_baseline']:.3f} m")
    print(f"Test PINN MAE:     {test_metrics['mae']:.3f} m")
    print(f"Test PINN RMSE:    {test_metrics['rmse']:.3f} m")
    print(f"Improvement:       {test_metrics['improvement']:.1f}%")
    print(f"{'='*60}")
    
    # Compare with non-curriculum
    print("\nComparison with non-curriculum (same fold):")
    print(f"  Without curriculum: 8.516m MAE (76.2% improvement)")
    print(f"  With curriculum:    {test_metrics['mae']:.3f}m MAE ({test_metrics['improvement']:.1f}% improvement)")
    
    if test_metrics['mae'] < 8.516:
        print(f"  ✅ Curriculum helped! Improvement: {(8.516 - test_metrics['mae']):.3f}m")
    else:
        print(f"  ❌ No improvement. Difference: {(test_metrics['mae'] - 8.516):.3f}m")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'phys_params': phys_params,
        'args': vars(args),
    }
    torch.save(checkpoint, os.path.join(args.output_dir, f"model_curriculum_fold{args.loso_fold}.pt"))
    
    return test_metrics


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, default="data/sensor_data_filtered.csv")
    p.add_argument("--output_dir", type=str, default="height_field_project/artifacts_curriculum")
    
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
    
    # Curriculum
    p.add_argument("--stage_epochs", type=int, default=30)
    p.add_argument("--epochs", type=int, default=80)
    
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--restart_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train_with_curriculum(args)
