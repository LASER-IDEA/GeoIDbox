"""
Uncertainty Quantification Experiment

1. Confidence Intervals vs. Residuals scatter plot
2. Spatial Uncertainty Map
3. Inference Latency measurement
4. Memory Footprint
"""
import numpy as np
import pandas as pd
import torch
import time
import sys
import os
sys.path.insert(0, '/data/home/huxiao/workspace/GeoIDbox')

from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_generalized_with_bias import compute_sensor_bias, BiasAwarePINN
from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.train_pinn import parse_timestamp


R_DRY_AIR = 287.05
G_STANDARD = 9.80665


def measure_uncertainty_vs_error(model, test_loader, phys_params, device, mc_samples=30):
    """
    Measure uncertainty vs actual error for MC Dropout.
    Returns uncertainty and absolute error for each sample.
    """
    print("Running MC Dropout uncertainty estimation...")
    model.train()  # Enable dropout
    
    all_uncertainties = []
    all_abs_errors = []
    all_coords = []
    
    with torch.no_grad():
        for batch in test_loader:
            lat = batch['lat'].to(device)
            lon = batch['lon'].to(device)
            z = batch['z'].to(device)
            t = batch['t'].to(device)
            temp = batch['temperature'].to(device)
            hum = batch['humidity'].to(device)
            bias = batch['pressure_bias'].to(device)
            p_obs = batch['p_obs'].to(device)
            h_gnss = batch['h_gnss'].to(device)
            
            # Multiple forward passes with dropout
            predictions = []
            for _ in range(mc_samples):
                delta_p = model(lat, lon, z, t, temp, hum, bias)
                
                # Convert to height
                p_corrected = p_obs + delta_p
                t_celsius = temp
                e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
                e = (hum / 100.0) * e_sat
                r = 0.62198 * e / (p_corrected - e)
                t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
                H = R_DRY_AIR * t_v / G_STANDARD
                h_pred = H * torch.log(phys_params.p_ref / p_corrected)
                
                predictions.append(h_pred.cpu().numpy())
            
            predictions = np.array(predictions)  # [mc_samples, batch_size]
            
            # Uncertainty = std across MC samples
            uncertainty = np.std(predictions, axis=0)
            
            # Mean prediction
            mean_pred = np.mean(predictions, axis=0)
            
            # Absolute error
            abs_error = np.abs(mean_pred - h_gnss.cpu().numpy())
            
            all_uncertainties.extend(uncertainty)
            all_abs_errors.extend(abs_error)
            all_coords.extend(torch.stack([lat, lon], dim=1).cpu().numpy())
    
    return np.array(all_uncertainties), np.array(all_abs_errors), np.array(all_coords)


def measure_inference_latency(model, device, n_runs=1000):
    """Measure inference latency in milliseconds."""
    print("\nMeasuring inference latency...")
    model.eval()
    
    # Create dummy input
    batch_size = 1
    lat = torch.randn(batch_size).to(device)
    lon = torch.randn(batch_size).to(device)
    z = torch.randn(batch_size).to(device)
    t = torch.randn(batch_size).to(device)
    temp = torch.randn(batch_size).to(device)
    hum = torch.randn(batch_size).to(device)
    bias = torch.randn(batch_size).to(device)
    
    # Warmup
    for _ in range(100):
        _ = model(lat, lon, z, t, temp, hum, bias)
    
    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(n_runs):
        _ = model(lat, lon, z, t, temp, hum, bias)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    end = time.time()
    
    avg_latency_ms = (end - start) / n_runs * 1000
    
    print(f"  Average latency: {avg_latency_ms:.3f} ms per query")
    print(f"  Throughput: {1000/avg_latency_ms:.1f} queries/second")
    
    return avg_latency_ms


def measure_memory_footprint(model):
    """Measure model memory footprint."""
    print("\nMeasuring memory footprint...")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # Estimate size (4 bytes per float32 parameter)
    size_mb = n_params * 4 / (1024 * 1024)
    
    print(f"  Total parameters: {n_params:,}")
    print(f"  Model size (float32): {size_mb:.2f} MB")
    
    return n_params, size_mb


def run_uncertainty_quantification():
    """Run all UQ experiments."""
    print("="*70)
    print("EXPERIMENT 3: Uncertainty Quantification")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/sensor_data_filtered.csv')
    df, phys_params = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # Load trained model (Fold 0)
    print("\nLoading trained model...")
    checkpoint_path = 'height_field_project/loso_curriculum_results/model_curriculum_fold0.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found: {checkpoint_path}")
        print("Please run curriculum training first or use a different checkpoint.")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    base_model = GeneralizedPressureCorrectionPINN(
        hash_levels=16, hash_features=4, hidden_dim=256,
        n_hidden_layers=3, temporal_freqs=6, use_siren=True
    )
    model = BiasAwarePINN(base_model, bias_dim=8).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Get test data (Fold 0)
    sensors = df['uid'].unique()
    held_out_sensor = sensors[0]
    test_df = df[df['uid'] == held_out_sensor].copy()
    
    print(f"\nTest set: {len(test_df)} samples from {held_out_sensor[:20]}")
    
    # Create dataset
    from height_field_project.train_generalized_with_bias import BiasAwarePINNDataset
    test_ds = BiasAwarePINNDataset(
        lat=test_df['avg_latitude'].values,
        lon=test_df['avg_longitude'].values,
        z=test_df['avg_altitude'].values,
        t=test_df['timestamp'].values,
        temperature=test_df['avg_temperature'].values,
        humidity=test_df['avg_humidity'].values,
        pressure_bias=test_df['pressure_bias'].values,
        sensor_id=np.zeros(len(test_df), dtype=np.int64),
        p_obs=test_df['avg_pressure'].values,
        h_gnss=test_df['avg_altitude'].values,
        h_phys=test_df['h_phys_hae'].values
    )
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # 1. Measure Uncertainty vs Error
    print("\n" + "="*70)
    print("1. Uncertainty vs. Actual Error Analysis")
    print("="*70)
    uncertainties, abs_errors, coords = measure_uncertainty_vs_error(
        model, test_loader, phys_params, device, mc_samples=30
    )
    
    # Save data for plotting
    uq_data = pd.DataFrame({
        'uncertainty': uncertainties,
        'abs_error': abs_errors,
        'latitude': coords[:, 0],
        'longitude': coords[:, 1]
    })
    uq_data.to_csv('experiments/03_uncertainty_quantification/uq_data.csv', index=False)
    
    # Compute correlation
    correlation = np.corrcoef(uncertainties, abs_errors)[0, 1]
    print(f"\n  Correlation (uncertainty vs error): {correlation:.3f}")
    
    # Bin analysis
    n_bins = 5
    bins = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    print(f"\n  Uncertainty Bins Analysis:")
    for i in range(n_bins):
        mask = (uncertainties >= bins[i]) & (uncertainties <= bins[i+1])
        if mask.sum() > 0:
            mean_unc = uncertainties[mask].mean()
            mean_err = abs_errors[mask].mean()
            print(f"    Bin {i+1}: Uncertainty={mean_unc:.3f}m, Error={mean_err:.3f}m, Count={mask.sum()}")
    
    # 2. Inference Latency
    print("\n" + "="*70)
    print("2. Inference Latency")
    print("="*70)
    latency_ms = measure_inference_latency(model, device, n_runs=1000)
    
    # 3. Memory Footprint
    print("\n" + "="*70)
    print("3. Memory Footprint")
    print("="*70)
    n_params, size_mb = measure_memory_footprint(model)
    
    # Save summary
    summary = {
        'uncertainty_error_correlation': correlation,
        'mean_uncertainty': uncertainties.mean(),
        'mean_abs_error': abs_errors.mean(),
        'inference_latency_ms': latency_ms,
        'throughput_qps': 1000 / latency_ms,
        'n_parameters': n_params,
        'model_size_mb': size_mb
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('experiments/03_uncertainty_quantification/summary.csv', index=False)
    
    print("\n" + "="*70)
    print("UNCERTAINTY QUANTIFICATION SUMMARY")
    print("="*70)
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nResults saved to:")
    print("  - experiments/03_uncertainty_quantification/uq_data.csv")
    print("  - experiments/03_uncertainty_quantification/summary.csv")
    
    return uq_data, summary


if __name__ == "__main__":
    uq_data, summary = run_uncertainty_quantification()
