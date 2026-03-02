"""
Leave-One-Sensor-Out (LOSO) cross-validation for PINN.

This is the rigorous metrological protocol where:
- In each fold, one sensor is held out completely
- Model is trained on remaining K-1 sensors
- Evaluation is on the held-out sensor's location

This tests true spatial generalization capability.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple
import json

from height_field_project.train_pinn import (
    set_seed,
    create_sensor_mapping,
    parse_timestamp,
    PINNDataset,
    train_epoch,
    evaluate,
    compute_prediction
)
from height_field_project.neural_field_pinn import PressureCorrectionPINN
from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.geoid_utils import lookup_geoid


R_DRY_AIR = 287.05
G_STANDARD = 9.80665
P_ISA_SL = 101325.0


def loso_split(df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
    """
    Generate LOSO splits.
    
    Returns:
        List of (train_df, test_df, held_out_sensor_uid) tuples
    """
    sensors = df['uid'].unique()
    splits = []
    
    for held_out in sensors:
        test_df = df[df['uid'] == held_out].copy()
        train_df = df[df['uid'] != held_out].copy()
        splits.append((train_df, test_df, held_out))
    
    return splits


def train_single_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_idx: int,
    held_out_sensor: str,
    args: argparse.Namespace,
    device: torch.device
) -> Dict:
    """Train and evaluate a single LOSO fold."""
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}: Held-out sensor = {held_out_sensor}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Compute physics baseline (GNSS altitude is already MSL)
    train_df, phys_params = compute_physics_baseline(
        train_df, p_ref=args.p_ref, t_ref_method="mean", convert_to_hae=False
    )
    
    # Apply same parameters to val/test
    val_df, _ = compute_physics_baseline(val_df, p_ref=phys_params.p_ref, t_ref_method="mean", convert_to_hae=False)
    test_df, _ = compute_physics_baseline(test_df, p_ref=phys_params.p_ref, t_ref_method="mean", convert_to_hae=False)
    
    # Create sensor mapping (only from training sensors)
    sensor_mapping = {uid: idx for idx, uid in enumerate(train_df['uid'].unique())}
    
    # Add held-out sensor to mapping (for inference, though it won't be trained)
    if held_out_sensor not in sensor_mapping:
        sensor_mapping[held_out_sensor] = len(sensor_mapping)
    
    train_df['sensor_idx'] = train_df['uid'].map(sensor_mapping)
    val_df['sensor_idx'] = val_df['uid'].map(sensor_mapping)
    test_df['sensor_idx'] = test_df['uid'].map(sensor_mapping)
    
    # Parse timestamps
    train_df['timestamp'] = train_df['processed_time'].apply(parse_timestamp)
    val_df['timestamp'] = val_df['processed_time'].apply(parse_timestamp)
    test_df['timestamp'] = test_df['processed_time'].apply(parse_timestamp)
    
    # Create datasets
    train_ds = PINNDataset(
        lat=train_df['avg_latitude'].values,
        lon=train_df['avg_longitude'].values,
        z=train_df['avg_altitude'].values,
        t=train_df['timestamp'].values,
        temperature=train_df['avg_temperature'].values,
        humidity=train_df['avg_humidity'].values,
        sensor_id=train_df['sensor_idx'].values,
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
        sensor_id=val_df['sensor_idx'].values,
        p_obs=val_df['avg_pressure'].values,
        h_gnss=val_df['avg_altitude'].values,
        h_phys=val_df['h_phys_hae'].values
    )
    
    test_ds = PINNDataset(
        lat=test_df['avg_latitude'].values,
        lon=test_df['avg_longitude'].values,
        z=test_df['avg_altitude'].values,
        t=test_df['timestamp'].values,
        temperature=test_df['avg_temperature'].values,
        humidity=test_df['avg_humidity'].values,
        sensor_id=test_df['sensor_idx'].values,
        p_obs=test_df['avg_pressure'].values,
        h_gnss=test_df['avg_altitude'].values,
        h_phys=test_df['h_phys_hae'].values
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    model = PressureCorrectionPINN(
        n_sensors=len(sensor_mapping),
        embedding_dim=args.sensor_embedding_dim,
        hash_levels=args.hash_levels,
        hash_features=args.hash_features,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        temporal_freqs=args.temporal_freqs,
        dropout=args.dropout,
        use_siren=args.use_siren
    ).to(device)
    
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
    
    for epoch in range(args.epochs):
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, None,
            phys_params.p_ref, device, args.lambda_hydro
        )
        
        val_metrics = evaluate(model, val_loader, phys_params.p_ref, device)
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:04d} | Val MAE: {val_metrics['mae']:.3f}m | "
                  f"Improvement: {val_metrics['improvement']:.1f}%")
        
        if val_metrics['mae'] < best_val_mae - 1e-4:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            # Save best model for this fold
            best_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, phys_params.p_ref, device)
    
    print(f"\n  Fold {fold_idx + 1} Results:")
    print(f"    Baseline MAE: {test_metrics['mae_baseline']:.3f} m")
    print(f"    PINN MAE: {test_metrics['mae']:.3f} m")
    print(f"    PINN RMSE: {test_metrics['rmse']:.3f} m")
    print(f"    Improvement: {test_metrics['improvement']:.1f}%")
    
    return {
        'fold': fold_idx,
        'held_out_sensor': held_out_sensor,
        'test_mae': test_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'baseline_mae': test_metrics['mae_baseline'],
        'improvement': test_metrics['improvement'],
        'n_train': len(train_df),
        'n_test': len(test_df)
    }


def main(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print(f"Loading data: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Total samples: {len(df)}, Sensors: {df['uid'].nunique()}")
    
    # Generate LOSO splits
    splits = loso_split(df)
    print(f"\nRunning LOSO cross-validation with {len(splits)} folds")
    
    # Run each fold
    results = []
    for fold_idx, (train_df, test_df, held_out) in enumerate(splits):
        # Further split train into train/val
        n_val = int(len(train_df) * 0.15)
        val_df = train_df.tail(n_val).copy()
        train_df = train_df.head(len(train_df) - n_val).copy()
        
        fold_result = train_single_fold(
            train_df, val_df, test_df,
            fold_idx, held_out, args, device
        )
        results.append(fold_result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("LOSO Cross-Validation Summary")
    print(f"{'='*60}")
    
    mae_values = [r['test_mae'] for r in results]
    rmse_values = [r['test_rmse'] for r in results]
    improvement_values = [r['improvement'] for r in results]
    
    print(f"Mean MAE: {np.mean(mae_values):.3f} ± {np.std(mae_values):.3f} m")
    print(f"Mean RMSE: {np.mean(rmse_values):.3f} ± {np.std(rmse_values):.3f} m")
    print(f"Mean Improvement: {np.mean(improvement_values):.1f}%")
    
    # Per-sensor breakdown
    print(f"\nPer-Sensor Results:")
    for r in results:
        print(f"  {r['held_out_sensor'][:20]:20s} | "
              f"MAE: {r['test_mae']:.3f}m | "
              f"Improvement: {r['improvement']:.1f}%")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.output_dir, "loso_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save summary
    summary = {
        'mean_mae': float(np.mean(mae_values)),
        'std_mae': float(np.std(mae_values)),
        'mean_rmse': float(np.mean(rmse_values)),
        'std_rmse': float(np.std(rmse_values)),
        'mean_improvement': float(np.mean(improvement_values)),
        'n_folds': len(results)
    }
    summary_path = os.path.join(args.output_dir, "loso_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LOSO cross-validation for PINN")
    
    p.add_argument("--input_csv", type=str, default="data/sensor_data_clean_stable.csv")
    p.add_argument("--output_dir", type=str, default="height_field_project/loso_results")
    
    # Physics
    p.add_argument("--p_ref", type=float, default=P_ISA_SL)
    p.add_argument("--lambda_hydro", type=float, default=0.01)
    
    # Model
    p.add_argument("--sensor_embedding_dim", type=int, default=8)
    p.add_argument("--hash_levels", type=int, default=16)
    p.add_argument("--hash_features", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_hidden_layers", type=int, default=3)
    p.add_argument("--temporal_freqs", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_siren", action="store_true", default=True)
    
    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--restart_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
