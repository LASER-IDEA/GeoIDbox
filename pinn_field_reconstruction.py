
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.interpolate
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Constants
g = 9.80665
R = 287.05
L_b = -0.0065  # Temperature lapse rate (K/m)
T0 = 288.15    # Standard temperature at sea level (K)
P0 = 101325.0  # Standard pressure at sea level (Pa)

# Fourier Feature Mapping
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

# Coordinate-based Neural Field Model (SIREN-like / Fourier)
class WeatherField(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, num_freqs=10, sigma=1.0):
        super().__init__()

        self.input_dim = 4 # x, y, z, t
        self.fourier = FourierFeatures(self.input_dim, num_freqs, sigma)

        feature_dim = 2 * num_freqs

        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.SiLU()) # SiLU is Swish, smooth and works well for physics

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, 2)) # Outputs: P_residual, T_residual
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Input x: (batch, 4) -> (lat_norm, lon_norm, z_norm, t_norm)
        # Returns: (P_res_norm, T_res_norm)
        return self.net(self.fourier(x))

class DataNormalizer:
    def __init__(self):
        self.mins = None
        self.maxs = None
        self.p_stats = None
        self.t_stats = None

    def fit(self, df):
        # Coordinates (x, y, z, t) -> [-1, 1] for x,y; [0, 1] for z, t usually
        self.mins = {
            'lat': df['lat'].min(), 'lon': df['lon'].min(),
            'alt': df['alt'].min(), 'timestamp': df['timestamp'].min()
        }
        self.maxs = {
            'lat': df['lat'].max(), 'lon': df['lon'].max(),
            'alt': df['alt'].max(), 'timestamp': df['timestamp'].max()
        }

        # Pressure / Temperature Residual Scaling
        self.p_stats = {'mean': df['pressure'].mean(), 'std': df['pressure'].std()}
        self.t_stats = {'mean': df['temperature'].mean(), 'std': df['temperature'].std()}

    def normalize_coords(self, lat, lon, h, t):
        # lat, lon -> [-1, 1]
        n_lat = 2 * (lat - self.mins['lat']) / (self.maxs['lat'] - self.mins['lat'] + 1e-6) - 1
        n_lon = 2 * (lon - self.mins['lon']) / (self.maxs['lon'] - self.mins['lon'] + 1e-6) - 1

        # h -> [0, 1]
        n_h = (h - self.mins['alt']) / (self.maxs['alt'] - self.mins['alt'] + 1e-6)

        # t -> [0, 1]
        n_t = (t - self.mins['timestamp']) / (self.maxs['timestamp'] - self.mins['timestamp'] + 1e-6)

        return torch.stack([n_lat, n_lon, n_h, n_t], dim=-1)

    def unnormalize_coords_h(self, n_h):
        return n_h * (self.maxs['alt'] - self.mins['alt'] + 1e-6) + self.mins['alt']

    # We predict residuals. We need scaling for P and T to keep gradients happy.
    # We will predict outputs in roughly [-1, 1] range.
    # Real P ~ 100,000 Pa. Residual might be 100-1000 Pa. Scale factor 1000.
    # Real T ~ 288 K. Residual might be 1-5 K. Scale factor 10.

    def scale_outputs(self, p_res_raw, t_res_raw):
        # Raw model output -> Physical units
        # Increased scaling factor to cover larger residuals (e.g. >10hPa)
        # 5000.0 means a raw output of 1.0 = 5000 Pa (50 hPa)
        p_res = p_res_raw * 5000.0
        t_res = t_res_raw * 20.0   # Increased T scaling slightly too
        return p_res, t_res

class CombinedDataset(Dataset):
    def __init__(self, sensor_df, era5_df, normalizer):
        self.normalizer = normalizer

        # Process Sensor Data
        self.s_coords = normalizer.normalize_coords(
            torch.tensor(sensor_df['lat'].values, dtype=torch.float32),
            torch.tensor(sensor_df['lon'].values, dtype=torch.float32),
            torch.tensor(sensor_df['alt'].values, dtype=torch.float32),
            torch.tensor(sensor_df['timestamp'].values, dtype=torch.float32)
        )
        self.s_p = torch.tensor(sensor_df['pressure'].values, dtype=torch.float32)
        self.s_t = torch.tensor(sensor_df['temperature'].values, dtype=torch.float32)

        # Process ERA5 Data (Anchors)
        # Assuming era5_df has 'static_height' calculated
        self.e_coords = normalizer.normalize_coords(
            torch.tensor(era5_df['lat'].values, dtype=torch.float32),
            torch.tensor(era5_df['lon'].values, dtype=torch.float32),
            torch.tensor(era5_df['static_height'].values, dtype=torch.float32),
            torch.tensor(era5_df['timestamp'].values, dtype=torch.float32)
        )
        self.e_p = torch.tensor(era5_df['sp'].values, dtype=torch.float32)
        self.e_t = torch.tensor(era5_df['t2m'].values, dtype=torch.float32)

    def __len__(self):
        return max(len(self.s_coords), len(self.e_coords))

    def __getitem__(self, idx):
        # Sampling strategy: Return one sensor point and one ERA5 point per batch item
        s_idx = idx % len(self.s_coords)
        e_idx = idx % len(self.e_coords)

        return {
            's_x': self.s_coords[s_idx], 's_p': self.s_p[s_idx], 's_t': self.s_t[s_idx],
            'e_x': self.e_coords[e_idx], 'e_p': self.e_p[e_idx], 'e_t': self.e_t[e_idx]
        }

def standard_atmosphere(h):
    # Valid for troposphere
    T = T0 + L_b * h
    P = P0 * (1 + L_b/T0 * h) ** (-g / (R * L_b))
    return P, T

def physics_loss(model, x_colloc, normalizer):
    """
    Compute physics residual loss (hydrostatic balance)
    x_colloc: (N, 4) autograd-enabled coordinates
    """
    x_colloc.requires_grad_(True)

    # Predict residuals
    preds = model(x_colloc)
    p_res_raw, t_res_raw = preds[:, 0], preds[:, 1]

    # Scale to physical units
    p_res, t_res = normalizer.scale_outputs(p_res_raw, t_res_raw)

    # Recover physical height for standard atmosphere baseline
    h_phys = normalizer.unnormalize_coords_h(x_colloc[:, 2])

    # Base state (Standard Atmosphere or ERA5 mean could be used)
    # Using Standard Atmosphere as base for simplicity here
    p_base, t_base = standard_atmosphere(h_phys)

    # Total fields
    P_total = p_base + p_res
    T_total = t_base + t_res

    # 1. Compute dP/dz
    # We need gradient w.r.t physical z.
    # z_norm = (z - min) / (max - min) => z = z_norm * range + min
    # d/dz = d/dz_norm * dz_norm/dz = d/dz_norm * (1/range)

    h_range = normalizer.maxs['alt'] - normalizer.mins['alt'] + 1e-6

    # Gradients of outputs w.r.t inputs
    # We want d(P_total)/dz. Since P_base is analytical, we can either:
    # A) Differentiate P_total numerically via autograd (includes P_base grad)
    # B) Differentiate P_res and add analytical d(P_base)/dz

    # Let's use Method A (pure autograd) for generality
    grad_P = torch.autograd.grad(
        outputs=P_total,
        inputs=x_colloc,
        grad_outputs=torch.ones_like(P_total),
        create_graph=True
    )[0]

    # grad_P is [dP/dlat_n, dP/dlon_n, dP/dz_n, dP/dt_n]
    dP_dzn = grad_P[:, 2]
    dP_dz = dP_dzn / h_range

    # 2. Compute Density (Ideal Gas Law)
    # rho = P / (R * T)
    rho = P_total / (R * T_total)

    # 3. Hydrostatic Balance Residual
    # dP/dz = -rho * g
    # Residual = dP/dz + rho * g
    pde_res = dP_dz + rho * g

    return torch.mean(pde_res ** 2)

def train(model, dataset, normalizer, device, epochs=1000, batch_size=256):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    losses = []

    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # 1. Sensor Loss (MSE)
            s_x = batch['s_x'].to(device)
            s_p_true = batch['s_p'].to(device)
            s_t_true = batch['s_t'].to(device)

            s_preds = model(s_x)
            s_p_res_raw, s_t_res_raw = s_preds[:, 0], s_preds[:, 1]
            s_p_res, s_t_res = normalizer.scale_outputs(s_p_res_raw, s_t_res_raw)

            # Base
            s_h_phys = normalizer.unnormalize_coords_h(s_x[:, 2])
            s_p_base, s_t_base = standard_atmosphere(s_h_phys)

            loss_obs_p = torch.mean(((s_p_base + s_p_res) - s_p_true)**2)
            loss_obs_t = torch.mean(((s_t_base + s_t_res) - s_t_true)**2)

            # 2. ERA5 Anchor Loss (MSE - slightly lower weight typically)
            e_x = batch['e_x'].to(device)
            e_p_true = batch['e_p'].to(device)
            e_t_true = batch['e_t'].to(device)

            e_preds = model(e_x)
            e_p_res_raw, e_t_res_raw = e_preds[:, 0], e_preds[:, 1]
            e_p_res, e_t_res = normalizer.scale_outputs(e_p_res_raw, e_t_res_raw)

            e_h_phys = normalizer.unnormalize_coords_h(e_x[:, 2])
            e_p_base, e_t_base = standard_atmosphere(e_h_phys)

            loss_era5_p = torch.mean(((e_p_base + e_p_res) - e_p_true)**2)
            loss_era5_t = torch.mean(((e_t_base + e_t_res) - e_t_true)**2)

            # 3. Physics Loss (Hydrostatic)
            # Generate random collocation points in the domain
            # (Or mix sensor/ERA5 points)
            # Let's perturb sensor points slightly to create volume
            noise = (torch.rand_like(s_x) - 0.5) * 0.1 # Small perturbation
            x_colloc = (s_x + noise).clamp(-1, 1)
            # Ensure z and t are in [0, 1]
            x_colloc[:, 2:] = x_colloc[:, 2:].clamp(0, 1)

            loss_pde = physics_loss(model, x_colloc, normalizer)

            # Total Loss
            # P_residual ~ 5000 Pa -> Loss ~ 25e6. with 1e-4 -> 2500.
            # RMSE 500Pa -> MSE 250,000 -> * 1e-4 = 25.
            # PDE Loss (dP/dz + rho*g)^2 ~ 1-10.
            # To balance, we need obs_loss * weight ~ pde_loss * weight
            # 25 * w_obs ~ 5 * w_pde
            # If w_pde = 1.0, w_obs should be 0.2.
            # But we are using 1e-4. 250,000 * 1e-4 = 25. 25 vs 1.
            # The observation term dominates.

            # Let's keep 1e-4 but increase PDE weight, or normalize.
            # Let's try to balance them better.
            # If we want equal contribution at ~200Pa error (MSE 40000).
            # 40000 * 1e-4 = 4. PDE ~ 4.
            # So 1e-4 and 1.0 is actually reasonable IF error is small.
            # But initially error is large.

            # Revert to original weights but boost PDE slightly to enforce physics
            # and rely on the new scaling to help convergence.

            # Balance at ~500Pa error (MSE 2.5e5):
            # Obs * 1e-5 = 2.5
            # PDE * 1.0  = 1~5

            total_loss = (loss_obs_p + loss_era5_p) * 1e-5 + \
                         (loss_obs_t + loss_era5_t) * 1e-1 + \
                         loss_pde * 1.0

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.5f} | OBS_P={loss_obs_p.item():.1f} | PDE={loss_pde.item():.3f}")

    print(f"Training finished in {time.time() - start_time:.1f}s")
    return model, losses

def solve_height(model, normalizer, lat, lon, time_ts, p_measured, device):
    """
    Inverse problem: Find h such that P_model(lat, lon, h, t) = P_measured
    Uses binary search on altitude.
    """
    model.eval()

    # Define search range (in meters, based on dataset min/max)
    # Expand slightly to ensure we brackets the solution
    h_min = normalizer.mins['alt'] - 500
    h_max = normalizer.maxs['alt'] + 500

    # Binary search
    low = h_min
    high = h_max

    # Create input tensors for lat, lon, t (fixed)
    # Ensure they are on the correct device
    t_lat = torch.tensor([lat], dtype=torch.float32, device=device)
    t_lon = torch.tensor([lon], dtype=torch.float32, device=device)
    t_time = torch.tensor([time_ts], dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(20): # Precision: ~10000m / 2^20 ~ 0.01m
            mid = (low + high) / 2

            # Normalize inputs manually
            n_lat = 2 * (t_lat - normalizer.mins['lat']) / (normalizer.maxs['lat'] - normalizer.mins['lat'] + 1e-6) - 1
            n_lon = 2 * (t_lon - normalizer.mins['lon']) / (normalizer.maxs['lon'] - normalizer.mins['lon'] + 1e-6) - 1
            n_h = (mid - normalizer.mins['alt']) / (normalizer.maxs['alt'] - normalizer.mins['alt'] + 1e-6)
            n_t = (t_time - normalizer.mins['timestamp']) / (normalizer.maxs['timestamp'] - normalizer.mins['timestamp'] + 1e-6)

            # Form batch of 1
            t_h = torch.tensor([n_h], dtype=torch.float32, device=device)
            # n_lat, n_lon, n_t are tensors, mid is float.
            # Wait, I used 'mid' in n_h calculation above, so 'n_h' is float.
            # Need to make n_h a tensor.

            inp = torch.stack([n_lat, n_lon, t_h, n_t], dim=-1)

            # Predict
            preds = model(inp)
            p_res_raw = preds[:, 0]

            p_res = p_res_raw * 5000.0 # Match new scaling

            # Physical P
            p_base, _ = standard_atmosphere(mid)
            p_total = p_base + p_res.item()

            # Logic: P(h) is monotonically decreasing.
            # If P_total(mid) > P_measured, we are too low (pressure is too high).
            # We need to increase height.
            if p_total > p_measured:
                low = mid
            else:
                high = mid

    return (low + high) / 2

def barometric_formula_height(P, P0=101325.0, T0=288.15):
    """Estimates height from pressure using standard atmosphere"""
    return (T0 / L_b) * ((P / P0) ** (-R * L_b / g) - 1)

def load_data():
    dataset_path = 'data/processed/sensor_data_filtered_outliers.csv'
    if not os.path.exists(dataset_path):
        # Fallback for running from parent directory
        dataset_path = 'GeoIDbox/data/processed/sensor_data_filtered_outliers.csv'
        if not os.path.exists(dataset_path):
             print(f"Error: dataset not found at {dataset_path} or data/processed/...")
             return None, None

    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Parse timestamps
    # The clean CSV has 'processed_time' as strings?
    if 'processed_time' in df.columns:
        df['dt'] = pd.to_datetime(df['processed_time'])
        # Convert to float timestamp (seconds from start)
        t_min = df['dt'].min()
        df['timestamp'] = (df['dt'] - t_min).dt.total_seconds()

        # Save t_min for reference if needed
        print(f"Time range: {df['dt'].min()} to {df['dt'].max()}")
        print(f"Total duration: {df['timestamp'].max() / 3600:.1f} hours")
    else:
        # Fallback if processed_time missing, use row index or other
        print("Warning: 'processed_time' column missing, strictly using index is risky.")
        df['timestamp'] = df.index * 60 # Assume 1 min?

    # --- 1. Prepare Sensor Data ---
    # Filter valid data
    # avg_altitude is HAE

    cols = ['avg_latitude', 'avg_longitude', 'avg_altitude', 'avg_pressure', 'avg_temperature', 'timestamp']
    if 'uid' in df.columns:
        cols.append('uid')

    sensor_df = df[cols].copy()

    # Rename columns carefully
    rename_dict = {
        'avg_latitude': 'lat', 'avg_longitude': 'lon', 'avg_altitude': 'alt',
        'avg_pressure': 'pressure', 'avg_temperature': 'temperature', 'timestamp': 'timestamp',
        'uid': 'uid'
    }
    sensor_df = sensor_df.rename(columns=rename_dict)

    # Drop NaNs
    sensor_df = sensor_df.dropna()
    print(f"Sensor samples: {len(sensor_df)}")

    # --- 2. Prepare ERA5 Data (Robust Estimation) ---
    # We want to use ERA5 surface pressure as 'anchors' at ground level.
    # But we don't have ERA5 geopotential z. We have 'era5_sp' (Surface Pressure).
    # We estimate a STATIC height for the ERA5 surface reference.

    # Extract ERA5 columns from the merged dataset
    # Note: These values are likely interpolated or NN at the sensor location.
    # To stabilize, we treat them as valid "ground truth" reference points for the field
    # at some estimated height z_era5.

    era5_cols = ['avg_latitude', 'avg_longitude', 'era5_sp', 'era5_t2m', 'timestamp']
    era5_df = df[era5_cols].copy()
    era5_df.columns = ['lat', 'lon', 'sp', 't2m', 'timestamp']
    era5_df = era5_df.dropna()

    # Strategy:
    # 1. Round coordinates to creating "Grid bins" (e.g. 0.01 deg approx 1km)
    #    This groups nearby points to stabilize the static height estimation.
    #    (The real ERA5 grid is 0.25deg, but our data is along a track)

    # Rounding to 3 decimals (~100m) or 2 decimals (~1km)?
    # Let's use 3 decimals to be safe but group effectively.
    era5_df['lat_grid'] = era5_df['lat'].round(3)
    era5_df['lon_grid'] = era5_df['lon'].round(3)

    # 2. Calculate Mean Surface Pressure per Grid Point
    #    P_mean(lat, lon) = mean(era5_sp) over all time
    #    Reset index to ensure column names are flat
    mean_sp = era5_df.groupby(['lat_grid', 'lon_grid'])['sp'].mean().reset_index()
    mean_sp.rename(columns={'sp': 'mean_sp'}, inplace=True)

    # 3. Estimate Static Grid Height using Barometric Formula on Mean Pressure
    #    z_grid = Barometric(P_mean)
    #    Ref: P = P0 * ... => z = ...
    #    Using global standard atmosphere P0=1013.25hPa is a rough approx but consistent.
    #    Goal: Fix a z value so the 'ground' doesn't dance up and down.
    mean_sp['static_height'] = barometric_formula_height(mean_sp['mean_sp'])

    # 4. Merge back to ERA5 df
    era5_df = pd.merge(era5_df, mean_sp, on=['lat_grid', 'lon_grid'], how='left')

    print(f"ERA5 samples: {len(era5_df)}")
    print("Static height stats:", era5_df['static_height'].describe())

    return sensor_df, era5_df

def evaluate_differential(model, normalizer, sensor_df, device):
    """
    Evaluate the model using differential altimetry concept.
    1. Identify Reference Station (most stable/most data).
    2. Use Reference Station to correct bias for other stations (Rovers).
    """
    model.eval()

    if 'uid' not in sensor_df.columns:
        print("Warning: 'uid' column not found in sensor_df. Skipping differential evaluation.")
        return

    # 1. Identify Reference Station (Highest sample count)
    uid_counts = sensor_df['uid'].value_counts()
    if len(uid_counts) < 2:
        print("Not enough unique UIDs for differential evaluation.")
        return

    ref_uid = uid_counts.idxmax()
    print(f"\n--- Differential Evaluation ---")
    print(f"Reference Station UID: {ref_uid} (Count: {uid_counts[ref_uid]})")

    # Filter Reference Data
    ref_df = sensor_df[sensor_df['uid'] == ref_uid].sort_values('timestamp')

    # Create Interpolator for Reference Pressure P_ref(t)
    # We need to look up Reference Pressure at any time t
    # Group by timestamp if duplicates exist
    # Fix: Select only numeric columns before mean()
    numeric_cols = ['pressure', 'lat', 'lon', 'alt', 'temperature']
    ref_df_clean = ref_df.groupby('timestamp')[numeric_cols].mean().reset_index()

    if len(ref_df_clean) < 10:
        print("Reference station has too few samples.")
        return

    # Create interpolator
    try:
        p_ref_interp = scipy.interpolate.interp1d(
            ref_df_clean['timestamp'].values,
            ref_df_clean['pressure'].values,
            kind='linear',
            fill_value="extrapolate"
        )
    except Exception as e:
        print(f"Error creating interpolator: {e}")
        return

    # Reference coordinates (Fixed)
    ref_lat = ref_df_clean['lat'].mean()
    ref_lon = ref_df_clean['lon'].mean()
    ref_alt_true = ref_df_clean['alt'].mean()

    print(f"Reference Location: ({ref_lat:.4f}, {ref_lon:.4f}, {ref_alt_true:.2f} m)")

    # 2. Evaluate on Rovers
    rovers = [u for u in uid_counts.index if u != ref_uid]
    results = []

    print(f"Evaluating on {len(rovers)} rovers...")

    for rover_id in rovers:
        rover_data = sensor_df[sensor_df['uid'] == rover_id]
        if len(rover_data) < 10:
            continue

        # Sample for speed if too large
        test_samples = rover_data.sample(min(len(rover_data), 50), random_state=42)

        errors_raw = []
        errors_diff = []

        for idx, row in test_samples.iterrows():
            t_curr = row['timestamp']

            # A. Raw Prediction (Single Station)
            try:
                h_raw = solve_height(model, normalizer, row['lat'], row['lon'], t_curr, row['pressure'], device)
            except:
                continue

            # B. Differential Correction
            try:
                # 1. Get Reference Pressure at this time
                p_ref_curr = float(p_ref_interp(t_curr))

                # 2. Predict Reference Height using Model + Reference Pressure
                # H_ref_pred = Model(lat_ref, lon_ref, P_ref_curr, t)
                h_ref_pred = solve_height(model, normalizer, ref_lat, ref_lon, t_curr, p_ref_curr, device)

                # 3. Calculate Model Bias at Reference
                # Bias = Predicted - True
                bias = h_ref_pred - ref_alt_true

                # 4. Correct Rover Height
                h_diff = h_raw - bias

                # Truth
                h_true = row['alt']

                errors_raw.append(h_raw - h_true)
                errors_diff.append(h_diff - h_true)

            except Exception as e:
                # print(f"Diff eval error: {e}")
                pass

        if errors_raw:
            mae_raw = np.mean(np.abs(errors_raw))
            mae_diff = np.mean(np.abs(errors_diff))
            rmse_raw = np.sqrt(np.mean(np.array(errors_raw)**2))
            rmse_diff = np.sqrt(np.mean(np.array(errors_diff)**2))

            results.append({
                'uid': rover_id,
                'mae_raw': mae_raw,
                'mae_diff': mae_diff,
                'improvement': mae_raw - mae_diff
            })
            print(f"Rover {rover_id[-6:]}: MAE Raw={mae_raw:.2f}m -> Diff={mae_diff:.2f}m (Imp: {mae_raw - mae_diff:.2f}m)")

    if results:
        res_df = pd.DataFrame(results)
        print("\n--- Summary ---")
        print(f"Mean MAE Raw: {res_df['mae_raw'].mean():.2f}")
        print(f"Mean MAE Diff: {res_df['mae_diff'].mean():.2f}")
        print(f"Mean Improvement: {res_df['improvement'].mean():.2f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sensor_df, era5_df = load_data()

    if sensor_df is None:
        return

    # Normalizer
    # For normalization, we should consider the range of BOTH datasets
    # Create a temp combined DF for fitting normalizer

    # Align column names for concatenation
    s_temp = sensor_df[['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']]
    e_temp = era5_df[['lat', 'lon', 'static_height', 'timestamp', 'sp', 't2m']]
    e_temp.columns = ['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']

    combined_temp = pd.concat([s_temp, e_temp], ignore_index=True)

    normalizer = DataNormalizer()
    normalizer.fit(combined_temp)

    # Dataset
    dataset = CombinedDataset(sensor_df, era5_df, normalizer)

    # Model
    model = WeatherField(num_freqs=10).to(device)

    # Train
    model, losses = train(model, dataset, normalizer, device, epochs=500)

    # Save
    torch.save(model.state_dict(), 'pinn_model.pth')

    # Plot Loss
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    print("Saved loss curve to loss_curve.png")

    model.eval()
    print("\n--- Evaluation (Height Solver) ---")
    test_samples = sensor_df.sample(100)
    errors = []

    for idx, row in test_samples.iterrows():
        try:
            h_pred = solve_height(model, normalizer, row['lat'], row['lon'], row['timestamp'], row['pressure'], device)
            diff = h_pred - row['alt']
            errors.append(diff)
        except Exception as e:
            print(f"Error solving for sample {idx}: {e}")

    errors = np.array(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    print(f"Test MAE: {mae:.2f} m")
    print(f"Test RMSE: {rmse:.2f} m")

    # Differential Evaluation
    evaluate_differential(model, normalizer, sensor_df, device)

if __name__ == "__main__":
    main()
