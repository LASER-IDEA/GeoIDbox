"""
Training pipeline for Physics-Informed Neural Field (PINN).

Corrected methodology:
1. Barometer → MSL via hypsometric equation
2. MSL → HAE via geoid undulation
3. PINN learns pressure correction field δP(x, y, z, t)
4. Multi-component loss: data fidelity + hydrostatic constraint
"""
import argparse
import os
import pickle
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

try:
    import wandb
except ImportError:
    wandb = None

from height_field_project.config import TrainingConfig, save_config
from height_field_project.physics_baseline import (
    compute_physics_baseline,
    pressure_to_msl_hypsometric,
    msl_to_hae,
    compute_virtual_temperature
)
from height_field_project.neural_field_pinn import (
    PressureCorrectionPINN,
    PhysicsInformedLoss
)
from height_field_project.geoid_utils import lookup_geoid
from height_field_project.era5_utils import enrich_with_era5


# Physical constants
R_DRY_AIR = 287.05
G_STANDARD = 9.80665
P_ISA_SL = 101325.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PINNDataset(Dataset):
    """
    Dataset for PINN training.
    
    Each sample contains:
    - lat, lon: Spatial coordinates
    - z: Physical altitude (for derivative computation)
    - t: Timestamp
    - temperature, humidity: Environmental measurements
    - sensor_id: Device identifier
    - p_obs: Observed pressure
    - h_gnss: GNSS ground truth (HAE)
    """
    
    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        z: np.ndarray,
        t: np.ndarray,
        temperature: np.ndarray,
        humidity: np.ndarray,
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
        self.sensor_id = torch.tensor(sensor_id, dtype=torch.long)
        self.p_obs = torch.tensor(p_obs, dtype=torch.float32)
        self.h_gnss = torch.tensor(h_gnss, dtype=torch.float32)
        self.h_phys = torch.tensor(h_phys, dtype=torch.float32)
        
        if weights is None:
            self.weights = torch.ones_like(self.h_gnss)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.h_gnss)
    
    def __getitem__(self, idx: int):
        return {
            'lat': self.lat[idx],
            'lon': self.lon[idx],
            'z': self.z[idx],
            't': self.t[idx],
            'temperature': self.temperature[idx],
            'humidity': self.humidity[idx],
            'sensor_id': self.sensor_id[idx],
            'p_obs': self.p_obs[idx],
            'h_gnss': self.h_gnss[idx],
            'h_phys': self.h_phys[idx],
            'weight': self.weights[idx]
        }


def create_sensor_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """Create mapping from sensor UID to integer index."""
    unique_uids = df['uid'].unique()
    return {uid: idx for idx, uid in enumerate(unique_uids)}


def parse_timestamp(ts_str: str) -> float:
    """Parse timestamp string to Unix timestamp."""
    dt = datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
    return dt.timestamp()


def compute_prediction(
    model: PressureCorrectionPINN,
    batch: dict,
    p_ref: float,
    device: torch.device,
    use_correction: bool = True
) -> torch.Tensor:
    """
    Compute HAE prediction from model.
    
    Args:
        model: PINN model
        batch: Batch data dictionary
        p_ref: Reference pressure (Pa)
        device: torch device
        use_correction: Whether to apply pressure correction
    
    Returns:
        Predicted HAE height
    """
    # Move batch to device
    lat = batch['lat'].to(device)
    lon = batch['lon'].to(device)
    z = batch['z'].to(device)
    t = batch['t'].to(device)
    temperature = batch['temperature'].to(device)
    humidity = batch['humidity'].to(device)
    sensor_id = batch['sensor_id'].to(device)
    p_obs = batch['p_obs'].to(device)
    
    if use_correction:
        # Get pressure correction from PINN
        delta_p = model(lat, lon, z, t, temperature, humidity, sensor_id)
        p_corrected = p_obs + delta_p
    else:
        p_corrected = p_obs
    
    # Compute virtual temperature
    t_celsius = temperature
    e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
    e = (humidity / 100.0) * e_sat
    r = 0.62198 * e / (p_corrected - e)
    t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
    
    # Hypsometric equation: MSL from pressure
    scale_height = R_DRY_AIR * t_v / G_STANDARD
    h_msl = scale_height * torch.log(p_ref / p_corrected)
    
    # Output is MSL (same as GNSS altitude which is orthometric height)
    h_hae = h_msl
    
    return h_hae


def train_epoch(
    model: PressureCorrectionPINN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedLoss,
    p_ref: float,
    device: torch.device,
    lambda_hydro: float
) -> Tuple[float, dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_data = 0.0
    loss_components = {}
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Compute prediction
        h_pred = compute_prediction(model, batch, p_ref, device, use_correction=True)
        h_gnss = batch['h_gnss'].to(device)
        weights = batch['weight'].to(device)
        
        # Data fidelity loss
        loss_data = torch.mean(weights * torch.abs(h_pred - h_gnss))
        loss = loss_data
        
        # Hydrostatic constraint (if enabled)
        if lambda_hydro > 0:
            # Enable gradient for z (must be on correct device first)
            z_with_grad = batch['z'].to(device).clone().requires_grad_(True)
            
            # Recompute with gradient-enabled z
            delta_p = model(
                batch['lat'].to(device),
                batch['lon'].to(device),
                z_with_grad,
                batch['t'].to(device),
                batch['temperature'].to(device),
                batch['humidity'].to(device),
                batch['sensor_id'].to(device)
            )
            
            p_obs = batch['p_obs'].to(device)
            p_corrected = p_obs + delta_p
            
            # dP/dz
            dp_dz = torch.autograd.grad(
                outputs=p_corrected,
                inputs=z_with_grad,
                grad_outputs=torch.ones_like(p_corrected),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Virtual temperature
            temperature = batch['temperature'].to(device)
            humidity = batch['humidity'].to(device)
            t_celsius = temperature
            e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (humidity / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            
            # Hydrostatic residual: dP/dz + P*g/(R*T)
            hydro_residual = dp_dz + (p_corrected * G_STANDARD) / (R_DRY_AIR * t_v)
            loss_hydro = torch.mean(hydro_residual ** 2)
            
            loss = loss + lambda_hydro * loss_hydro
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        batch_size = len(h_gnss)
        total_loss += loss.item() * batch_size
        total_data += batch_size
        
        # Track components
        if 'data' not in loss_components:
            loss_components['data'] = 0.0
        loss_components['data'] += loss_data.item() * batch_size
        
        if lambda_hydro > 0 and 'hydrostatic' not in loss_components:
            loss_components['hydrostatic'] = 0.0
        if lambda_hydro > 0:
            loss_components['hydrostatic'] += loss_hydro.item() * batch_size
    
    avg_loss = total_loss / total_data
    for key in loss_components:
        loss_components[key] /= total_data
    
    return avg_loss, loss_components


def evaluate(
    model: PressureCorrectionPINN,
    loader: DataLoader,
    p_ref: float,
    device: torch.device
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    all_preds = []
    all_gts = []
    all_phys = []
    
    with torch.no_grad():
        for batch in loader:
            h_pred = compute_prediction(model, batch, p_ref, device, use_correction=True)
            h_gnss = batch['h_gnss']
            h_phys = batch['h_phys']
            
            all_preds.append(h_pred.cpu().numpy())
            all_gts.append(h_gnss.numpy())
            all_phys.append(h_phys.numpy())
    
    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gts)
    phys = np.concatenate(all_phys)
    
    # Metrics
    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))
    
    # Baseline metrics (no correction)
    mae_baseline = np.mean(np.abs(phys - gts))
    rmse_baseline = np.sqrt(np.mean((phys - gts) ** 2))
    
    improvement = (mae_baseline - mae) / mae_baseline * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mae_baseline': mae_baseline,
        'rmse_baseline': rmse_baseline,
        'improvement': improvement
    }


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Loading data from: {args.input_csv}")
    
    # Initialize wandb
    run = None
    if args.wandb_project and wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples from {df['uid'].nunique()} sensors")
    
    # Optional: ERA5 enrichment
    if args.era5_nc and os.path.exists(args.era5_nc):
        print(f"Enriching with ERA5: {args.era5_nc}")
        df = enrich_with_era5(df, args.era5_nc)
    
    # Compute physics baseline (GNSS altitude is already MSL/orthometric height)
    print("Computing physics baseline...")
    df, phys_params = compute_physics_baseline(
        df,
        p_ref=args.p_ref,
        h_ref=0.0,
        t_ref_method="mean",
        convert_to_hae=False
    )
    print(f"Reference pressure: {phys_params.p_ref:.2f} Pa")
    print(f"Reference temperature: {phys_params.t_ref:.2f} K")
    print(f"Scale height: {phys_params.scale_height:.2f} m")
    
    # Compute initial baseline error
    baseline_error = np.mean(np.abs(df['h_phys_hae'] - df['avg_altitude']))
    print(f"Initial physics baseline MAE: {baseline_error:.2f} m")
    
    # Create sensor ID mapping
    sensor_mapping = create_sensor_mapping(df)
    df['sensor_idx'] = df['uid'].map(sensor_mapping)
    print(f"Sensors: {list(sensor_mapping.keys())}")
    
    # Parse timestamps
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # Prepare dataset
    dataset = PINNDataset(
        lat=df['avg_latitude'].values,
        lon=df['avg_longitude'].values,
        z=df['avg_altitude'].values,  # Use GNSS altitude as z for derivative
        t=df['timestamp'].values,
        temperature=df['avg_temperature'].values,
        humidity=df['avg_humidity'].values,
        sensor_id=df['sensor_idx'].values,
        p_obs=df['avg_pressure'].values,
        h_gnss=df['avg_altitude'].values,  # Ground truth
        h_phys=df['h_phys_hae'].values  # Physics baseline
    )
    
    # Train/val/test split
    n_total = len(dataset)
    n_test = int(n_total * args.test_ratio)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val - n_test
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
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
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.restart_epochs,
        T_mult=2
    )
    
    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer,
            None, phys_params.p_ref, device, args.lambda_hydro
        )
        
        # Validation
        val_metrics = evaluate(model, val_loader, phys_params.p_ref, device)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:04d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val MAE: {val_metrics['mae']:.3f}m | "
                  f"Improvement: {val_metrics['improvement']:.1f}%")
        
        if run:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_improvement': val_metrics['improvement'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Early stopping
        if val_metrics['mae'] < best_val_mae - 1e-4:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'sensor_mapping': sensor_mapping,
                'phys_params': phys_params,
                'args': vars(args)
            }, os.path.join(args.artifacts_dir, "model.pt"))
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(args.artifacts_dir, "model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, phys_params.p_ref, device)
    print(f"\nTest Results:")
    print(f"  Physics Baseline MAE: {test_metrics['mae_baseline']:.3f} m")
    print(f"  PINN MAE: {test_metrics['mae']:.3f} m")
    print(f"  PINN RMSE: {test_metrics['rmse']:.3f} m")
    print(f"  Improvement: {test_metrics['improvement']:.1f}%")
    
    if run:
        wandb.log({
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_improvement': test_metrics['improvement']
        })
        wandb.finish()
    
    print(f"\nArtifacts saved to: {args.artifacts_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PINN for barometric altitude correction")
    
    # Data
    p.add_argument("--input_csv", type=str, default="data/sensor_data_clean_stable.csv")
    p.add_argument("--artifacts_dir", type=str, default="height_field_project/artifacts_pinn")
    p.add_argument("--era5_nc", type=str, default=None)
    
    # Physics
    p.add_argument("--p_ref", type=float, default=P_ISA_SL, help="Reference pressure (Pa)")
    p.add_argument("--lambda_hydro", type=float, default=0.01, help="Hydrostatic constraint weight")
    
    # Model architecture
    p.add_argument("--sensor_embedding_dim", type=int, default=8)
    p.add_argument("--hash_levels", type=int, default=16)
    p.add_argument("--hash_features", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_hidden_layers", type=int, default=3)
    p.add_argument("--temporal_freqs", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_siren", action="store_true", default=True)
    
    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--restart_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    
    # Logging
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
