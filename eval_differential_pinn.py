
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.interpolate
import os
import matplotlib.pyplot as plt

# --- 1. Model Definitions (Copied to avoid import issues) ---

# Constants
g = 9.80665
R = 287.05
L_b = -0.0065  # Temperature lapse rate (K/m)
T0 = 288.15    # Standard temperature at sea level (K)
P0 = 101325.0  # Standard pressure at sea level (Pa)

def standard_atmosphere(h):
    # Valid for troposphere
    T = T0 + L_b * h
    # Avoid division by zero or invalid power if base is negative (unlikely here)
    base = 1 + L_b/T0 * h
    # Clamp base to avoid issues at extreme heights if model queries there
    # base = torch.clamp(base, min=1e-6)
    P = P0 * (base) ** (-g / (R * L_b))
    return P, T

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_freqs, sigma=1.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.sigma = sigma
        # Initialize frequencies - fixed, not trainable
        self.B = nn.Parameter(torch.randn(input_dim, num_freqs) * sigma, requires_grad=False)

    def forward(self, x):
        # x is (batch, input_dim)
        # projected is (batch, num_freqs)
        projected = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

class WeatherField(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, num_freqs=10, sigma=1.0):
        super().__init__()

        self.input_dim = 4 # x, y, z, t
        self.fourier = FourierFeatures(self.input_dim, num_freqs, sigma)

        feature_dim = 2 * num_freqs

        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, 2)) # Outputs: P_residual, T_residual
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.fourier(x))

class DataNormalizer:
    def __init__(self):
        self.mins = None
        self.maxs = None
        self.p_stats = None
        self.t_stats = None

    def fit(self, df):
        # Handle cases where column names might differ if fitting on diverse data
        # We assume standard names or renamed before passing here
        self.mins = {
            'lat': df['lat'].min(), 'lon': df['lon'].min(),
            'alt': df['alt'].min(), 'timestamp': df['timestamp'].min()
        }
        self.maxs = {
            'lat': df['lat'].max(), 'lon': df['lon'].max(),
            'alt': df['alt'].max(), 'timestamp': df['timestamp'].max()
        }

    def normalize_coords(self, lat, lon, h, t):
        eps = 1e-6
        # lat, lon -> [-1, 1]
        n_lat = 2 * (lat - self.mins['lat']) / (self.maxs['lat'] - self.mins['lat'] + eps) - 1
        n_lon = 2 * (lon - self.mins['lon']) / (self.maxs['lon'] - self.mins['lon'] + eps) - 1

        # h -> [0, 1]
        n_h = (h - self.mins['alt']) / (self.maxs['alt'] - self.mins['alt'] + eps)

        # t -> [0, 1]
        n_t = (t - self.mins['timestamp']) / (self.maxs['timestamp'] - self.mins['timestamp'] + eps)

        return torch.stack([n_lat, n_lon, n_h, n_t], dim=-1)

    def unnormalize_coords_h(self, n_h):
        eps = 1e-6
        return n_h * (self.maxs['alt'] - self.mins['alt'] + eps) + self.mins['alt']

    def scale_outputs(self, p_res_raw, t_res_raw):
        # Hardcoded scaling based on training script
        p_res = p_res_raw * 5000.0
        t_res = t_res_raw * 20.0
        return p_res, t_res

def solve_height(model, normalizer, lat, lon, time_ts, p_measured, device):
    """
    Inverse problem: Find h such that P_model(lat, lon, h, t) = P_measured
    Uses binary search on altitude.
    """
    model.eval()

    # Define search range
    h_min = normalizer.mins['alt'] - 500
    h_max = normalizer.maxs['alt'] + 500

    low = h_min
    high = h_max

    # Pre-create tensors for fixed inputs
    t_lat = torch.tensor([lat], dtype=torch.float32, device=device)
    t_lon = torch.tensor([lon], dtype=torch.float32, device=device)
    t_time = torch.tensor([time_ts], dtype=torch.float32, device=device)

    # Normalize fixed inputs
    eps = 1e-6
    n_lat = 2 * (t_lat - normalizer.mins['lat']) / (normalizer.maxs['lat'] - normalizer.mins['lat'] + eps) - 1
    n_lon = 2 * (t_lon - normalizer.mins['lon']) / (normalizer.maxs['lon'] - normalizer.mins['lon'] + eps) - 1
    n_t = (t_time - normalizer.mins['timestamp']) / (normalizer.maxs['timestamp'] - normalizer.mins['timestamp'] + eps)

    with torch.no_grad():
        for _ in range(20):
            mid = (low + high) / 2

            # Normalize height
            n_h = (mid - normalizer.mins['alt']) / (normalizer.maxs['alt'] - normalizer.mins['alt'] + eps)
            t_h = torch.tensor([n_h], dtype=torch.float32, device=device)

            inp = torch.stack([n_lat, n_lon, t_h, n_t], dim=-1)

            # Predict
            preds = model(inp)
            p_res_raw = preds[:, 0]

            p_res = p_res_raw * 5000.0

            # Physical P
            # Note: standard_atmosphere expects numpy or scaler if input is scalar
            # But we are using it inside torch context usually?
            # Implemented standard_atmosphere works with tensors or scalars
            p_base, _ = standard_atmosphere(mid)
            p_total = p_base + p_res.item()

            if p_total > p_measured:
                low = mid
            else:
                high = mid

    return (low + high) / 2

# --- 2. Main Evaluation Script ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # A. Init and Fit Normalizer (Crucial for correct inference)
    # We must fit the normalizer on the TRAINING data distribution if possible
    # to maintain the coordinate mapping learned by the model.
    print("Fitting Normalizer on Training Data...")
    train_data_path = 'GeoIDbox/data/processed/sensor_data_with_real_era5.csv'
    if not os.path.exists(train_data_path):
        train_data_path = 'data/processed/sensor_data_with_real_era5.csv'

    normalizer = DataNormalizer()

    if os.path.exists(train_data_path):
        print(f"Loading training data from {train_data_path} for normalization context.")
        train_df = pd.read_csv(train_data_path)

        # Determine strict timestamp range
        if 'processed_time' in train_df.columns:
            train_df['dt'] = pd.to_datetime(train_df['processed_time'])
            t_min = train_df['dt'].min()
            train_df['timestamp'] = (train_df['dt'] - t_min).dt.total_seconds()

        # Prepare fit dataframe with correct columns
        # Note: Depending on the file, columns might be different.
        # 'sensor_data_with_real_era5.csv' has 'avg_latitude' etc.
        fit_df = pd.DataFrame({
            'lat': train_df['avg_latitude'],
            'lon': train_df['avg_longitude'],
            'alt': train_df['avg_altitude'],
            'timestamp': train_df['timestamp']
        })
        normalizer.fit(fit_df)
    else:
        print("Warning: Training data not found. Fitting on inference data distributions (Risk of mismatch).")
        t_min = None

    # B. Load Evaluation Data
    print("Loading Evaluation Data...")
    eval_data_path = 'GeoIDbox/data/processed/sensor_data_filtered_outliers.csv'
    if not os.path.exists(eval_data_path):
        eval_data_path = 'data/processed/sensor_data_filtered_outliers.csv'

    if not os.path.exists(eval_data_path):
        print(f"Error: {eval_data_path} not found.")
        return

    df = pd.read_csv(eval_data_path)

    # Time processing for Eval set
    if 'processed_time' in df.columns:
        df['dt'] = pd.to_datetime(df['processed_time'])

        # If t_min was not set by training data, set it now
        if t_min is None:
            t_min = df['dt'].min()
            df['timestamp'] = (df['dt'] - t_min).dt.total_seconds()

            # If we fit on this data, we need to fit normalizer now
            fit_df = pd.DataFrame({
                'lat': df['avg_latitude'],
                'lon': df['avg_longitude'],
                'alt': df['avg_altitude'],
                'timestamp': df['timestamp']
            })
            normalizer.fit(fit_df)
        else:
            # Use t_min from training data to keep time definition consistent
            # Critical: Use the same t_min as training!
            df['timestamp'] = (df['dt'] - t_min).dt.total_seconds()

    print("Normalizer ready.")

    # Init Model
    model = WeatherField(num_freqs=10).to(device)
    model_path = 'pinn_model.pth' # Assuming in current root or GeoIDbox?
    if not os.path.exists(model_path):
        model_path = 'GeoIDbox/pinn_model.pth'

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Error: pinn_model.pth not found.")
        return

    model.eval()

    # B. Identify Reference Station
    ref_uid = '20240911193733A012843A9994605977'
    print(f"Extracting Reference Station: {ref_uid}")

    # Check for UID column name
    uid_col = None
    if 'uid' in df.columns:
        uid_col = 'uid'
    elif ' uid' in df.columns:
        uid_col = ' uid'
    else:
        print(f"Error: UID column not found. Available columns: {df.columns.tolist()}")
        return

    # Filter Reference Data
    ref_df = df[df[uid_col] == ref_uid].sort_values('timestamp')

    if len(ref_df) == 0:
        print("Error: Reference UID not found in data.")
        return

    # Create Lookup Interpolation for Reference Pressure P_ref(t)
    # We interpolate Pressure over Time
    ref_times = ref_df['timestamp'].values
    ref_pressures = ref_df['avg_pressure'].values

    # Create interpolator
    # kind='linear' is safe. 'cubic' might be better but risky if gaps.
    p_ref_interp = scipy.interpolate.interp1d(ref_times, ref_pressures, kind='linear', fill_value="extrapolate")

    # Reference Coordinates (Assume static)
    ref_lat = ref_df['avg_latitude'].mean()
    ref_lon = ref_df['avg_longitude'].mean()
    ref_alt_true = ref_df['avg_altitude'].mean() # H_ref_true

    print(f"Reference Station: {len(ref_df)} samples. Pos: ({ref_lat:.4f}, {ref_lon:.4f}, {ref_alt_true:.2f})")

    # C. Evaluation Loop
    rovers = df[uid_col].unique()
    rovers = [r for r in rovers if r != ref_uid]

    results = []

    print(f"Evaluating on {len(rovers)} rovers...")

    SAMPLES_PER_ROVER = 50

    for rover_id in rovers:
        # Get rover data
        rover_data = df[df[uid_col] == rover_id]

        if len(rover_data) < 10:
            continue

        # Sampling
        test_samples = rover_data.sample(min(len(rover_data), SAMPLES_PER_ROVER), random_state=42)

        station_errors_raw = []
        station_errors_corr = []

        for idx, row in test_samples.iterrows():
            t_curr = row['timestamp']

            # 1. Predicted Rover Height (Raw)
            # H_raw = solve(model, P_rover)
            try:
                h_raw = solve_height(model, normalizer, row['avg_latitude'], row['avg_longitude'], t_curr, row['avg_pressure'], device)
            except Exception as e:
                print(f"Error solving raw: {e}")
                continue

            # 2. Reference Correction
            # Get P_ref at t_curr
            try:
                p_ref_meas = p_ref_interp(t_curr)
            except:
                continue

            # Predict Reference Height using its P_ref_meas
            # H_ref_pred = solve(model, P_ref_meas) at Ref Location
            h_ref_pred = solve_height(model, normalizer, ref_lat, ref_lon, t_curr, float(p_ref_meas), device)

            # Correction Delta
            # If model says Ref is at 105m, but it is actually at 100m -> Error is +5m.
            # We should subtract 5m from Rover.
            # Delta = H_ref_pred - H_ref_true
            delta = h_ref_pred - ref_alt_true

            # 3. Corrected Height
            h_final = h_raw - delta

            # Truth
            h_true = row['avg_altitude']

            station_errors_raw.append(h_raw - h_true)
            station_errors_corr.append(h_final - h_true)

        if station_errors_raw:
            mae_raw = np.mean(np.abs(station_errors_raw))
            mae_corr = np.mean(np.abs(station_errors_corr))
            rmse_raw = np.sqrt(np.mean(np.array(station_errors_raw)**2))
            rmse_corr = np.sqrt(np.mean(np.array(station_errors_corr)**2))

            results.append({
                'rover_id': rover_id,
                'mae_raw': mae_raw,
                'mae_corr': mae_corr,
                'rmse_raw': rmse_raw,
                'rmse_corr': rmse_corr,
                'improvement_mae': mae_raw - mae_corr,
                'improvement_percent': (mae_raw - mae_corr) / mae_raw * 100
            })
            print(f"Rover {rover_id[-6:]}: MAE {mae_raw:.2f} -> {mae_corr:.2f} ({mae_raw - mae_corr:+.2f}m)")

    # D. Summary
    if results:
        res_df = pd.DataFrame(results)
        print("\n--- Summary Results ---")
        print(f"Mean MAE Raw:      {res_df['mae_raw'].mean():.2f} m")
        print(f"Mean MAE Corrected:{res_df['mae_corr'].mean():.2f} m")
        print(f"Mean RMSE Raw:     {res_df['rmse_raw'].mean():.2f} m")
        print(f"Mean RMSE Corrected:{res_df['rmse_corr'].mean():.2f} m")
        print(f"Mean Improvement:  {res_df['improvement_mae'].mean():.2f} m")

        # Save results
        res_df.to_csv('differential_pinn_results.csv', index=False)
        print("Results saved to differential_pinn_results.csv")
    else:
        print("No results computed.")

if __name__ == "__main__":
    main()
