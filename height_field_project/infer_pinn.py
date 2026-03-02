"""
Inference script for PINN model.

Loads trained model and performs prediction with uncertainty quantification.
"""
import argparse
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datetime import datetime

from height_field_project.neural_field_pinn import PressureCorrectionPINN
from height_field_project.physics_baseline import (
    compute_physics_baseline,
    compute_virtual_temperature
)
from height_field_project.geoid_utils import lookup_geoid
from height_field_project.era5_utils import enrich_with_era5


R_DRY_AIR = 287.05
G_STANDARD = 9.80665


def parse_timestamp(ts_str: str) -> float:
    """Parse timestamp to Unix timestamp."""
    dt = datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
    return dt.timestamp()


def load_model(artifacts_dir: str, device: torch.device):
    """Load trained PINN model."""
    checkpoint_path = os.path.join(artifacts_dir, "model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    args = checkpoint['args']
    sensor_mapping = checkpoint['sensor_mapping']
    phys_params = checkpoint['phys_params']
    
    # Create model
    model = PressureCorrectionPINN(
        n_sensors=len(sensor_mapping),
        embedding_dim=args.get('sensor_embedding_dim', 8),
        hash_levels=args.get('hash_levels', 16),
        hash_features=args.get('hash_features', 2),
        hidden_dim=args.get('hidden_dim', 128),
        n_hidden_layers=args.get('n_hidden_layers', 3),
        temporal_freqs=args.get('temporal_freqs', 4),
        dropout=args.get('dropout', 0.0),
        use_siren=args.get('use_siren', True)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, sensor_mapping, phys_params


def predict_with_correction(
    model: PressureCorrectionPINN,
    df: pd.DataFrame,
    sensor_mapping: dict,
    phys_params,
    device: torch.device,
    mc_samples: int = 30
) -> pd.DataFrame:
    """
    Predict HAE with pressure correction and uncertainty.
    
    Args:
        model: Trained PINN model
        df: DataFrame with sensor data
        sensor_mapping: UID to index mapping
        phys_params: Physics baseline parameters
        device: torch device
        mc_samples: Number of MC dropout samples for uncertainty
    
    Returns:
        DataFrame with predictions
    """
    df = df.copy()
    
    # Map sensor IDs
    df['sensor_idx'] = df['uid'].map(sensor_mapping)
    
    # Parse timestamps
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # Prepare tensors
    lat = torch.tensor(df['avg_latitude'].values, dtype=torch.float32).to(device)
    lon = torch.tensor(df['avg_longitude'].values, dtype=torch.float32).to(device)
    z = torch.tensor(df['avg_altitude'].values, dtype=torch.float32).to(device)
    t = torch.tensor(df['timestamp'].values, dtype=torch.float32).to(device)
    temperature = torch.tensor(df['avg_temperature'].values, dtype=torch.float32).to(device)
    humidity = torch.tensor(df['avg_humidity'].values, dtype=torch.float32).to(device)
    sensor_id = torch.tensor(df['sensor_idx'].values, dtype=torch.long).to(device)
    p_obs = torch.tensor(df['avg_pressure'].values, dtype=torch.float32).to(device)
    
    # Get pressure correction with uncertainty
    with torch.no_grad():
        delta_p_mean, delta_p_std = model.predict_mc(
            lat, lon, z, t, temperature, humidity, sensor_id,
            samples=mc_samples
        )
    
    delta_p_mean = delta_p_mean.cpu().numpy()
    delta_p_std = delta_p_std.cpu().numpy()
    
    # Apply correction
    p_corrected = df['avg_pressure'].values + delta_p_mean
    
    # Compute virtual temperature
    t_celsius = df['avg_temperature'].values
    e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
    e = (df['avg_humidity'].values / 100.0) * e_sat
    r = 0.62198 * e / (p_corrected - e)
    t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
    
    # Hypsometric equation: MSL from pressure
    p_ref = phys_params.get('p_ref', 101325.0) if isinstance(phys_params, dict) else phys_params.p_ref
    scale_height = R_DRY_AIR * t_v / G_STANDARD
    h_msl = scale_height * np.log(p_ref / p_corrected)
    
    # Output is already MSL (GNSS altitude is MSL)
    h_pred = h_msl
    
    # Propagate uncertainty (approximate)
    # dh = -H * dP/P
    dh_dP = -scale_height / p_corrected
    h_pred_std = np.abs(dh_dP) * delta_p_std
    
    # Add results to DataFrame
    df['delta_p_mean'] = delta_p_mean
    df['delta_p_std'] = delta_p_std
    df['p_corrected'] = p_corrected
    df['h_msl_pred'] = h_msl
    df['h_pred_hae'] = h_pred
    df['h_pred_std'] = h_pred_std
    
    # Compute error if ground truth available
    if 'avg_altitude' in df.columns:
        df['error'] = df['h_pred_hae'] - df['avg_altitude']
        df['abs_error'] = np.abs(df['error'])
    
    return df


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.artifacts_dir}")
    model, sensor_mapping, phys_params = load_model(args.artifacts_dir, device)
    print(f"Loaded model with {len(sensor_mapping)} sensors")
    
    # Load input data
    print(f"Loading data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Optional ERA5 enrichment
    if args.era5_nc and os.path.exists(args.era5_nc):
        print(f"Enriching with ERA5: {args.era5_nc}")
        df = enrich_with_era5(df, args.era5_nc)
    
    # Compute physics baseline for comparison (GNSS altitude is already MSL)
    p_ref = phys_params.get('p_ref', 101325.0) if isinstance(phys_params, dict) else phys_params.p_ref
    df, _ = compute_physics_baseline(df, p_ref=p_ref, convert_to_hae=False)
    
    # Filter to known sensors
    known_sensors = set(sensor_mapping.keys())
    df_known = df[df['uid'].isin(known_sensors)].copy()
    
    if len(df_known) == 0:
        print(f"Warning: No known sensors found in input data")
        print(f"Available sensors: {list(known_sensors)}")
        return
    
    print(f"Predicting for {len(df_known)} samples from {df_known['uid'].nunique()} sensors")
    
    # Run inference
    df_pred = predict_with_correction(
        model, df_known, sensor_mapping, phys_params,
        device, mc_samples=args.mc_samples
    )
    
    # Compute metrics
    if 'avg_altitude' in df_pred.columns:
        mae = df_pred['abs_error'].mean()
        rmse = np.sqrt((df_pred['error'] ** 2).mean())
        
        # Baseline error
        baseline_error = np.abs(df_pred['h_phys_hae'] - df_pred['avg_altitude'])
        mae_baseline = baseline_error.mean()
        
        improvement = (mae_baseline - mae) / mae_baseline * 100
        
        print(f"\nResults:")
        print(f"  Physics Baseline MAE: {mae_baseline:.3f} m")
        print(f"  PINN MAE: {mae:.3f} m")
        print(f"  PINN RMSE: {rmse:.3f} m")
        print(f"  Improvement: {improvement:.1f}%")
    
    # Save predictions
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df_pred.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PINN inference for altitude correction")
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--artifacts_dir", type=str, default="height_field_project/artifacts_pinn")
    p.add_argument("--output", type=str, default="artifacts/pinn_predictions.csv")
    p.add_argument("--era5_nc", type=str, default=None)
    p.add_argument("--mc_samples", type=int, default=30, help="MC dropout samples for uncertainty")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
