"""
PINN with Terrain Features and Curriculum Learning (Paper Implementation)

This implements the full paper method including:
1. Terrain features: local roughness, height percentile, sensor density
2. Three-stage curriculum learning by altitude and density
3. Height residual prediction (as in paper)
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
from sklearn.neighbors import NearestNeighbors

try:
    import wandb
except ImportError:
    wandb = None

from height_field_project.neural_field_pinn_generalized import (
    GeneralizedPressureCorrectionPINN, SirenLayer
)
from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_pinn import set_seed, parse_timestamp


class TerrainFeatureExtractor:
    """Extract terrain features as described in the paper."""
    
    def __init__(self, df: pd.DataFrame, k_neighbors: int = 5, radius: float = 0.001):
        """
        Args:
            df: DataFrame with lat, lon, altitude
            k_neighbors: K for local roughness calculation
            radius: Radius (in degrees) for sensor density calculation
        """
        self.df = df.copy()
        self.k = k_neighbors
        self.radius = radius
        
        # Build KD-tree for spatial queries
        coords = df[['avg_latitude', 'avg_longitude']].values
        self.kdtree = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        self.kdtree.fit(coords)
        
        # Pre-compute features
        self._compute_all_features()
    
    def _compute_all_features(self):
        """Pre-compute terrain features for all samples."""
        coords = self.df[['avg_latitude', 'avg_longitude']].values
        altitudes = self.df['avg_altitude'].values
        
        # 1. Local Roughness (ρ)
        distances, indices = self.kdtree.kneighbors(coords)
        roughness = []
        for i, neighbors in enumerate(indices):
            neighbor_alts = altitudes[neighbors]
            rho = np.sqrt(np.mean((neighbor_alts - neighbor_alts.mean())**2))
            roughness.append(rho)
        self.df['roughness'] = roughness
        
        # 2. Height Percentile Rank (r)
        sorted_indices = np.argsort(np.argsort(altitudes))
        percentile_ranks = (sorted_indices / len(altitudes)) * 100
        self.df['height_percentile'] = percentile_ranks
        
        # 3. Sensor Density (d) - count neighbors within radius
        densities = []
        for coord in coords:
            count = self.kdtree.radius_neighbors([coord], radius=self.radius, return_distance=False)[0]
            densities.append(len(count))
        self.df['sensor_density'] = densities
    
    def get_features(self, indices: np.ndarray) -> np.ndarray:
        """Get terrain features for given indices."""
        return self.df.iloc[indices][['roughness', 'height_percentile', 'sensor_density']].values


class TerrainAwareDataset(Dataset):
    """Dataset with terrain features."""
    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        z: np.ndarray,
        t: np.ndarray,
        temperature: np.ndarray,
        humidity: np.ndarray,
        terrain_features: np.ndarray,  # [N, 3] roughness, percentile, density
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
        self.terrain = torch.tensor(terrain_features, dtype=torch.float32)
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
            'terrain': self.terrain[idx],
            'p_obs': self.p_obs[idx],
            'h_gnss': self.h_gnss[idx],
            'h_phys': self.h_phys[idx],
            'weight': self.weights[idx]
        }


class TerrainAwarePINN(nn.Module):
    """
    PINN with terrain features and height residual prediction (as in paper).
    """
    def __init__(
        self,
        hash_levels: int = 16,
        hash_features: int = 2,  # Paper uses F=2
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        temporal_freqs: int = 6,
        use_siren: bool = False,  # Paper uses SiLU
        terrain_dim: int = 3
    ):
        super().__init__()
        
        from height_field_project.neural_field_pinn_generalized import (
            MultiResolutionHashEncoding, FourierTemporalEncoding
        )
        
        # Spatial encoding
        self.hash_encoding = MultiResolutionHashEncoding(
            n_levels=hash_levels,
            n_features=hash_features,
            min_res=16,
            max_res=512  # Paper uses N_max=512
        )
        
        # Temporal encoding
        self.temporal_encoding = FourierTemporalEncoding(
            n_frequencies=temporal_freqs,
            max_period_hours=168.0
        )
        
        # Input dimension
        in_dim = (
            self.hash_encoding.out_dim +  # Spatial
            1 +  # Z
            self.temporal_encoding.out_dim +  # Temporal
            1 + 1 +  # T, RH
            terrain_dim  # Terrain features
        )
        
        # MLP with SiLU + LayerNorm (as in paper)
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize output to near zero
        with torch.no_grad():
            self.mlp[-1].weight.fill_(0.0)
            self.mlp[-1].bias.fill_(0.0)
    
    def forward(self, lat, lon, z, t, temperature, humidity, terrain):
        # Normalize coordinates
        lat_norm = (lat + 90.0) / 180.0
        lon_norm = lon % 360.0 / 360.0
        coords = torch.stack([lat_norm, lon_norm], dim=-1)
        
        # Encode features
        h_spatial = self.hash_encoding(coords)
        h_temporal = self.temporal_encoding(t / 3600.0)
        
        # Concatenate all features
        features = torch.cat([
            h_spatial,
            z.unsqueeze(-1),
            h_temporal,
            temperature.unsqueeze(-1),
            humidity.unsqueeze(-1),
            terrain
        ], dim=-1)
        
        # Predict height residual (as in paper)
        delta_h = self.mlp(features).squeeze(-1)
        
        return delta_h


def create_curriculum_splits(df: pd.DataFrame):
    """
    Create 3-stage curriculum splits as described in paper.
    
    Stage 1 (Easy): h < 150m, d > d_median
    Stage 2 (Medium): h < 200m OR d > d_p25
    Stage 3 (Hard): Full dataset
    """
    altitudes = df['avg_altitude'].values
    densities = df['sensor_density'].values
    
    d_median = np.median(densities)
    d_p25 = np.percentile(densities, 25)
    
    # Stage 1: Easy - low altitude, high density
    mask_1 = (altitudes < 150) & (densities > d_median)
    
    # Stage 2: Medium - moderate altitude or moderate density
    mask_2 = (altitudes < 200) | (densities > d_p25)
    
    # Stage 3: Hard - full dataset
    mask_3 = np.ones(len(df), dtype=bool)
    
    return mask_1, mask_2, mask_3


def train_epoch(model, loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_data = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        delta_h = model(
            batch['lat'].to(device),
            batch['lon'].to(device),
            batch['z'].to(device),
            batch['t'].to(device),
            batch['temperature'].to(device),
            batch['humidity'].to(device),
            batch['terrain'].to(device)
        )
        
        # Predicted height = physical baseline + residual
        h_pred = batch['h_phys'].to(device) + delta_h
        
        # Loss
        loss = torch.mean(torch.abs(h_pred - batch['h_gnss'].to(device)))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch['lat'])
        total_data += len(batch['lat'])
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / total_data


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_gnss = []
    all_phys = []
    
    with torch.no_grad():
        for batch in loader:
            delta_h = model(
                batch['lat'].to(device),
                batch['lon'].to(device),
                batch['z'].to(device),
                batch['t'].to(device),
                batch['temperature'].to(device),
                batch['humidity'].to(device),
                batch['terrain'].to(device)
            )
            
            h_pred = batch['h_phys'].to(device) + delta_h
            
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
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mae_baseline': baseline_mae,
        'improvement': improvement
    }


def train_terrain_pinn(args: argparse.Namespace):
    """Train PINN with terrain features and curriculum learning."""
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
    
    # Extract terrain features
    print("\nExtracting terrain features...")
    terrain_extractor = TerrainFeatureExtractor(df, k_neighbors=args.k_neighbors, radius=args.density_radius)
    
    # Copy features back to main df
    df['roughness'] = terrain_extractor.df['roughness'].values
    df['height_percentile'] = terrain_extractor.df['height_percentile'].values
    df['sensor_density'] = terrain_extractor.df['sensor_density'].values
    
    print(f"Roughness: mean={df['roughness'].mean():.2f}, std={df['roughness'].std():.2f}")
    print(f"Height percentile: mean={df['height_percentile'].mean():.2f}")
    print(f"Sensor density: mean={df['sensor_density'].mean():.2f}, std={df['sensor_density'].std():.2f}")
    
    # Create curriculum splits
    mask_1, mask_2, mask_3 = create_curriculum_splits(df)
    print(f"\nCurriculum splits:")
    print(f"  Stage 1 (Easy): {mask_1.sum()} samples ({mask_1.mean()*100:.1f}%)")
    print(f"  Stage 2 (Medium): {mask_2.sum()} samples ({mask_2.mean()*100:.1f}%)")
    print(f"  Stage 3 (Hard): {mask_3.sum()} samples (100%)")
    
    # Parse timestamps
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # Create datasets for each stage
    def create_dataset(mask):
        indices = np.where(mask)[0]
        terrain_feat = df.iloc[indices][['roughness', 'height_percentile', 'sensor_density']].values
        
        return TerrainAwareDataset(
            lat=df.iloc[indices]['avg_latitude'].values,
            lon=df.iloc[indices]['avg_longitude'].values,
            z=df.iloc[indices]['avg_altitude'].values,
            t=df.iloc[indices]['timestamp'].values,
            temperature=df.iloc[indices]['avg_temperature'].values,
            humidity=df.iloc[indices]['avg_humidity'].values,
            terrain_features=terrain_feat,
            p_obs=df.iloc[indices]['avg_pressure'].values,
            h_gnss=df.iloc[indices]['avg_altitude'].values,
            h_phys=df.iloc[indices]['h_phys_hae'].values
        )
    
    ds_stage1 = create_dataset(mask_1)
    ds_stage2 = create_dataset(mask_2)
    ds_full = create_dataset(mask_3)
    
    print(f"\nDataset sizes: Stage1={len(ds_stage1)}, Stage2={len(ds_stage2)}, Full={len(ds_full)}")
    
    # LOSO split for final evaluation
    if args.loso_test:
        sensors = df['uid'].unique()
        held_out_sensor = sensors[args.loso_fold % len(sensors)]
        
        test_mask = df['uid'] == held_out_sensor
        train_mask = ~test_mask
        
        # Train/val split
        train_indices = np.where(train_mask)[0]
        n_val = int(len(train_indices) * args.val_ratio)
        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]
        
        print(f"\nLOSO Fold {args.loso_fold}: Held-out = {held_out_sensor[:25]}")
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {test_mask.sum()}")
        
        # Create datasets
        def create_dataset_from_indices(indices):
            terrain_feat = df.iloc[indices][['roughness', 'height_percentile', 'sensor_density']].values
            return TerrainAwareDataset(
                lat=df.iloc[indices]['avg_latitude'].values,
                lon=df.iloc[indices]['avg_longitude'].values,
                z=df.iloc[indices]['avg_altitude'].values,
                t=df.iloc[indices]['timestamp'].values,
                temperature=df.iloc[indices]['avg_temperature'].values,
                humidity=df.iloc[indices]['avg_humidity'].values,
                terrain_features=terrain_feat,
                p_obs=df.iloc[indices]['avg_pressure'].values,
                h_gnss=df.iloc[indices]['avg_altitude'].values,
                h_phys=df.iloc[indices]['h_phys_hae'].values
            )
        
        ds_train = create_dataset_from_indices(train_indices)
        ds_val = create_dataset_from_indices(val_indices)
        ds_test = create_dataset_from_indices(np.where(test_mask)[0])
    else:
        # Standard split
        n_test = int(len(ds_full) * 0.15)
        n_val = int(len(ds_full) * 0.15)
        
        indices = np.random.permutation(len(ds_full))
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test+n_val]
        train_indices = indices[n_test+n_val:]
        
        # Create subsets
        def subset_dataset(ds, indices):
            terrain_feat = ds.terrain.numpy()[indices]
            return TerrainAwareDataset(
                lat=ds.lat.numpy()[indices],
                lon=ds.lon.numpy()[indices],
                z=ds.z.numpy()[indices],
                t=ds.t.numpy()[indices],
                temperature=ds.temperature.numpy()[indices],
                humidity=ds.humidity.numpy()[indices],
                terrain_features=terrain_feat,
                p_obs=ds.p_obs.numpy()[indices],
                h_gnss=ds.h_gnss.numpy()[indices],
                h_phys=ds.h_phys.numpy()[indices]
            )
        
        ds_train = subset_dataset(ds_full, train_indices)
        ds_val = subset_dataset(ds_full, val_indices)
        ds_test = subset_dataset(ds_full, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = TerrainAwarePINN(
        hash_levels=args.hash_levels,
        hash_features=args.hash_features,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        temporal_freqs=args.temporal_freqs,
        use_siren=args.use_siren,
        terrain_dim=3
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Three-stage curriculum training
    print(f"\n{'='*60}")
    print("CURRICULUM TRAINING")
    print(f"{'='*60}")
    
    best_val_mae = float('inf')
    
    # Stage 1: Easy samples
    if not args.loso_test and len(ds_stage1) > 1000:
        print(f"\n--- Stage 1: Easy (h<150m, d>d_median) ---")
        stage1_loader = DataLoader(ds_stage1, batch_size=args.batch_size, shuffle=True)
        
        for epoch in range(args.stage_epochs):
            train_loss = train_epoch(model, stage1_loader, optimizer, None, device)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:03d} | Train: {train_loss:.4f}")
    
    # Stage 2: Medium samples
    if not args.loso_test and len(ds_stage2) > 1000:
        print(f"\n--- Stage 2: Medium (h<200m OR d>d_p25) ---")
        stage2_loader = DataLoader(ds_stage2, batch_size=args.batch_size, shuffle=True)
        
        for epoch in range(args.stage_epochs):
            train_loss = train_epoch(model, stage2_loader, optimizer, None, device)
            if (epoch + 1) % 20 == 0:
                val_metrics = evaluate(model, val_loader, device)
                print(f"  Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val MAE: {val_metrics['mae']:.3f}m")
    
    # Stage 3: Full dataset (or LOSO training)
    print(f"\n--- Stage 3: Full Dataset ---")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.restart_epochs, T_mult=2
    )
    
    patience_counter = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, val_loader, device)
            print(f"  Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val MAE: {val_metrics['mae']:.3f}m | Improvement: {val_metrics['improvement']:.1f}%")
            
            # Early stopping
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
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print("TERRAIN-AWARE PINN WITH CURRICULUM LEARNING RESULTS")
    print(f"{'='*60}")
    print(f"Test Baseline MAE: {test_metrics['mae_baseline']:.3f} m")
    print(f"Test PINN MAE:     {test_metrics['mae']:.3f} m")
    print(f"Test PINN RMSE:    {test_metrics['rmse']:.3f} m")
    print(f"Improvement:       {test_metrics['improvement']:.1f}%")
    print(f"{'='*60}")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'phys_params': phys_params,
        'args': vars(args),
    }
    torch.save(checkpoint, os.path.join(args.output_dir, f"model_terrain_fold{args.loso_fold}.pt"))
    
    return test_metrics


def build_parser():
    p = argparse.ArgumentParser(description="Train PINN with terrain features and curriculum learning")
    
    p.add_argument("--input_csv", type=str, default="data/sensor_data_filtered.csv")
    p.add_argument("--output_dir", type=str, default="height_field_project/artifacts_terrain")
    
    # LOSO
    p.add_argument("--loso_test", action="store_true")
    p.add_argument("--loso_fold", type=int, default=0)
    
    # Physics
    p.add_argument("--p_ref", type=float, default=None)
    
    # Model (paper settings)
    p.add_argument("--hash_levels", type=int, default=16)
    p.add_argument("--hash_features", type=int, default=2)  # Paper uses F=2
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_hidden_layers", type=int, default=3)
    p.add_argument("--temporal_freqs", type=int, default=6)
    p.add_argument("--use_siren", action="store_true", default=False)  # Paper uses SiLU
    
    # Terrain features
    p.add_argument("--k_neighbors", type=int, default=5)
    p.add_argument("--density_radius", type=float, default=0.001)
    
    # Curriculum learning
    p.add_argument("--stage_epochs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=150)
    
    # Training
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--restart_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train_terrain_pinn(args)
