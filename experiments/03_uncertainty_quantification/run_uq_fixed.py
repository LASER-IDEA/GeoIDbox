"""
Uncertainty Quantification Experiment (Fixed Version)

Analyzes MC Dropout uncertainty and model efficiency.
"""
import numpy as np
import pandas as pd
import torch
import time
import sys
import os
sys.path.insert(0, '/data/home/huxiao/workspace/GeoIDbox')

from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_generalized_with_bias import compute_sensor_bias, BiasAwarePINNDataset
from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.train_generalized_with_bias import BiasAwarePINN
from height_field_project.train_pinn import parse_timestamp
from torch.utils.data import DataLoader

R_DRY_AIR = 287.05
G_STANDARD = 9.80665


def compute_prediction_variance(model, dataloader, device, n_samples=30):
    """
    Compute prediction variance using MC Dropout.
    Returns both mean prediction and variance across samples.
    """
    model.train()  # Keep dropout active
    
    all_predictions = []
    
    # Collect predictions from multiple forward passes
    for _ in range(n_samples):
        batch_preds = []
        with torch.no_grad():
            for batch in dataloader:
                delta_p = model(
                    batch['lat'].to(device), batch['lon'].to(device),
                    batch['z'].to(device), batch['t'].to(device),
                    batch['temperature'].to(device), batch['humidity'].to(device),
                    batch['pressure_bias'].to(device)
                )
                
                # Compute height
                p_corrected = batch['p_obs'].to(device) + delta_p
                t_celsius = batch['temperature'].to(device)
                e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
                e = (batch['humidity'].to(device) / 100.0) * e_sat
                r = 0.62198 * e / (p_corrected - e)
                t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
                H = R_DRY_AIR * t_v / G_STANDARD
                h_pred = H * torch.log(101839.40 / p_corrected)
                
                batch_preds.append(h_pred.cpu())
        
        all_predictions.append(torch.cat(batch_preds).numpy())
    
    # Stack predictions: shape (n_samples, n_data)
    predictions = np.stack(all_predictions)
    
    # Compute statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    variance_pred = np.var(predictions, axis=0)
    
    return mean_pred, std_pred, variance_pred


def run_uncertainty_quantification():
    """Run uncertainty quantification experiments."""
    print("="*70)
    print("EXPERIMENT 3: Uncertainty Quantification (Fixed)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/sensor_data_filtered.csv')
    df, phys_params = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)
    
    # Load model
    print("\nLoading trained model...")
    # Find available model
    model_paths = [
        'test/temp_outputs/artifacts_curriculum/model_curriculum_fold0.pt',
        'height_field_project/loso_bias_aware_results/model_bias_aware_fold0.pt',
        'height_field_project/artifacts/model.pt',
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("ERROR: Could not find any trained model!")
        return
    
    print(f"Using model: {model_path}")
    
    if not os.path.exists(model_path):
        print("ERROR: Could not find trained model!")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    base_model = GeneralizedPressureCorrectionPINN(
        hash_levels=16, hash_features=4, hidden_dim=256,
        n_hidden_layers=3, temporal_freqs=6, use_siren=True
    )
    model = BiasAwarePINN(base_model, bias_dim=8).to(device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
    
    # Use fold 0 test data
    sensors = df['uid'].unique()
    held_out_sensor = sensors[0]
    test_df = df[df['uid'] == held_out_sensor].copy()
    
    print(f"\nTest set: {len(test_df)} samples from {held_out_sensor[:20]}")
    
    # Create dataset
    ds_test = BiasAwarePINNDataset(
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
    
    test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)
    
    # 1. Uncertainty vs Error Analysis
    print("\n" + "="*70)
    print("1. Uncertainty vs. Actual Error Analysis")
    print("="*70)
    print("Running MC Dropout uncertainty estimation...")
    
    mean_pred, std_pred, variance_pred = compute_prediction_variance(
        model, test_loader, device, n_samples=30
    )
    
    # Ground truth
    h_gnss = test_df['avg_altitude'].values
    abs_error = np.abs(mean_pred - h_gnss)
    
    # Uncertainty metrics
    correlation = np.corrcoef(std_pred, abs_error)[0, 1]
    
    print(f"\n  Correlation (σ vs |error|): {correlation:.4f}")
    print(f"  Mean predicted σ: {np.mean(std_pred):.6f} m")
    print(f"  Std predicted σ: {np.std(std_pred):.6f} m")
    print(f"  Max predicted σ: {np.max(std_pred):.6f} m")
    print(f"  Mean absolute error: {np.mean(abs_error):.3f} m")
    
    # Bin analysis
    print("\n  Uncertainty Bins Analysis:")
    n_bins = 5
    bins = np.percentile(std_pred, np.linspace(0, 100, n_bins + 1))
    
    for i in range(n_bins):
        mask = (std_pred >= bins[i]) & (std_pred <= bins[i+1])
        if mask.sum() > 0:
            bin_unc = std_pred[mask].mean()
            bin_err = abs_error[mask].mean()
            bin_count = mask.sum()
            print(f"    Bin {i+1}: σ={bin_unc:.6f}m, |error|={bin_err:.3f}m, n={bin_count}")
    
    # 2. Inference Latency
    print("\n" + "="*70)
    print("2. Inference Latency")
    print("="*70)
    
    print("\nMeasuring inference latency...")
    model.eval()
    
    # Warmup
    sample_input = torch.randn(1, 7).to(device)  # dummy input
    for _ in range(10):
        with torch.no_grad():
            _ = model.base_model.hash_encoding(sample_input[:, :2])
    
    # Measure latency
    latencies = []
    test_tensor = torch.utils.data.Subset(ds_test, range(min(1000, len(ds_test))))
    
    for i in range(min(100, len(test_tensor))):
        sample = test_tensor[i]
        lat_start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(
                sample['lat'].unsqueeze(0).to(device),
                sample['lon'].unsqueeze(0).to(device),
                sample['z'].unsqueeze(0).to(device),
                sample['t'].unsqueeze(0).to(device),
                sample['temperature'].unsqueeze(0).to(device),
                sample['humidity'].unsqueeze(0).to(device),
                sample['pressure_bias'].unsqueeze(0).to(device)
            )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        lat_end = time.perf_counter()
        latencies.append((lat_end - lat_start) * 1000)  # Convert to ms
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = 1000.0 / avg_latency  # queries per second
    
    print(f"  Average latency: {avg_latency:.3f} ± {std_latency:.3f} ms per query")
    print(f"  Throughput: {throughput:.1f} queries/second")
    
    # 3. Memory Footprint
    print("\n" + "="*70)
    print("3. Memory Footprint")
    print("="*70)
    
    print("\nMeasuring memory footprint...")
    n_params = sum(p.numel() for p in model.parameters())
    model_size_mb = n_params * 4 / (1024 * 1024)  # float32
    
    print(f"  Total parameters: {n_params:,}")
    print(f"  Model size (float32): {model_size_mb:.2f} MB")
    
    # Save results
    print("\n" + "="*70)
    print("UNCERTAINTY QUANTIFICATION SUMMARY")
    print("="*70)
    
    summary = {
        'uncertainty_error_correlation': correlation,
        'mean_uncertainty': np.mean(std_pred),
        'std_uncertainty': np.std(std_pred),
        'max_uncertainty': np.max(std_pred),
        'mean_abs_error': np.mean(abs_error),
        'inference_latency_ms': avg_latency,
        'latency_std_ms': std_latency,
        'throughput_qps': throughput,
        'n_parameters': n_params,
        'model_size_mb': model_size_mb
    }
    
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save detailed data
    uq_data = pd.DataFrame({
        'latitude': test_df['avg_latitude'].values,
        'longitude': test_df['avg_longitude'].values,
        'altitude': test_df['avg_altitude'].values,
        'h_pred': mean_pred,
        'abs_error': abs_error,
        'uncertainty': std_pred,
        'variance': variance_pred
    })
    uq_data.to_csv('experiments/03_uncertainty_quantification/uq_data.csv', index=False)
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('experiments/03_uncertainty_quantification/summary.csv', index=False)
    
    print("\nResults saved to:")
    print("  - experiments/03_uncertainty_quantification/uq_data.csv")
    print("  - experiments/03_uncertainty_quantification/summary.csv")
    
    return summary


if __name__ == "__main__":
    results = run_uncertainty_quantification()
