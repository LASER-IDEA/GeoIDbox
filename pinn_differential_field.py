
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

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

    def unnormalize_coords_t(self, n_t):
        return n_t * (self.maxs['timestamp'] - self.mins['timestamp'] + 1e-6) + self.mins['timestamp']

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
    def __init__(self, sensor_df, era5_df, normalizer, p_ref_interp, t_ref_interp):
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

        # Lookup reference values for sensor data
        s_timestamps = sensor_df['timestamp'].values
        self.s_p_ref = torch.tensor(p_ref_interp(s_timestamps), dtype=torch.float32)
        self.s_t_ref = torch.tensor(t_ref_interp(s_timestamps), dtype=torch.float32)

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

        # Lookup reference values for ERA5 data
        e_timestamps = era5_df['timestamp'].values
        self.e_p_ref = torch.tensor(p_ref_interp(e_timestamps), dtype=torch.float32)
        self.e_t_ref = torch.tensor(t_ref_interp(e_timestamps), dtype=torch.float32)

    def __len__(self):
        return max(len(self.s_coords), len(self.e_coords))

    def __getitem__(self, idx):
        # Sampling strategy: Return one sensor point and one ERA5 point per batch item
        s_idx = idx % len(self.s_coords)
        e_idx = idx % len(self.e_coords)

        return {
            's_x': self.s_coords[s_idx], 's_p': self.s_p[s_idx], 's_t': self.s_t[s_idx],
            's_p_ref': self.s_p_ref[s_idx], 's_t_ref': self.s_t_ref[s_idx],
            'e_x': self.e_coords[e_idx], 'e_p': self.e_p[e_idx], 'e_t': self.e_t[e_idx],
            'e_p_ref': self.e_p_ref[e_idx], 'e_t_ref': self.e_t_ref[e_idx]
        }

def differential_baseline(h, p_ref, t_ref, h_ref=145.4):
    """
    Differential physical base model relative to a reference station.

    Args:
        h: Target height (m) [Tensor]
        p_ref: Pressure at reference station (Pa) [Tensor]
        t_ref: Temperature at reference station (K) [Tensor]
        h_ref: Height of reference station (m) [float, default=145.4]

    Returns:
        P_base, T_base
    """
    # Standard lapse rate
    L_b = -0.0065

    # 1. Estimate average temperature between h_ref and h
    # Simple linear interpolation of temperature
    # T(h) = T_ref + L_b * (h - h_ref)
    # T_avg = (T_ref + T(h)) / 2 = T_ref + L_b/2 * (h - h_ref)
    T_avg = t_ref + (L_b * (h - h_ref) / 2.0)

    # 2. Hypsometric equation / Barometric formula with T_avg
    # P = P_ref * exp( -g * (h - h_ref) / (R * T_avg) )
    exponent = -g * (h - h_ref) / (R * T_avg)
    P_base = p_ref * torch.exp(exponent)

    # 3. Base Temperature
    T_base = t_ref + L_b * (h - h_ref)

    return P_base, T_base

def physics_loss(model, x_colloc, normalizer, p_ref_interp, t_ref_interp):
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

    # Recover physical coordinates
    h_phys = normalizer.unnormalize_coords_h(x_colloc[:, 2])
    t_phys = normalizer.unnormalize_coords_t(x_colloc[:, 3])

    # Get reference values for these timestamps
    # Need to detach tensor, convert to numpy for interpolation, then back to tensor
    # IMPORTANT: t_phys is part of the graph? NO, unnormalize_coords_t is linear, so gradients flow.
    # But for p_ref lookup, we treat p_ref as exogenous/constant for that time.
    # The gradient of Loss w.r.t time does involve d(P_base)/dt?
    # physics_loss computes dP/dz.
    # P_total = P_base(h, t) + P_res(h, t).
    # dP/dz = dP_base/dm * ... + dP_res/dz.
    # P_base depends on p_ref(t). That doesn't depend on z (vertical z).
    # However, differential_baseline(h, p_ref, t_ref).
    # P_base = p_ref * exp(-g*(h-h_ref)/(R*T_avg)).
    # T_avg depends on h and t_ref.
    # So P_base depends on h. So we need to compute gradients through P_base w.r.t h.
    # This means p_ref_val and t_ref_val should differ from h in the graph?
    # p_ref_val depends on t. t is an input coordinate.
    # If we are computing dP/dt, we need p_ref_val to be differentiable w.r.t t.
    # But scipy.interp1d is NOT differentiable by PyTorch.
    # But wait, physics_loss ONLY computes dP/dz.
    # "dP/dz = dP_dzn / h_range".
    # Only gradients w.r.t Z are needed.
    # P_base depends on h explicitly.
    # Does P_base depend on h via p_ref_val? No, p_ref_val depends on t.
    # So treating p_ref_val as constant w.r.t z is CORRECT.
    # So detach().cpu().numpy() is safe for computing dP/dz.

    t_phys_np = t_phys.detach().cpu().numpy()
    p_ref_val = torch.tensor(p_ref_interp(t_phys_np), dtype=torch.float32, device=x_colloc.device)
    t_ref_val = torch.tensor(t_ref_interp(t_phys_np), dtype=torch.float32, device=x_colloc.device)

    # Differential Baseline
    p_base, t_base = differential_baseline(h_phys, p_ref_val, t_ref_val)

    # Total fields
    P_total = p_base + p_res
    T_total = t_base + t_res

    # 1. Compute dP/dz
    h_range = normalizer.maxs['alt'] - normalizer.mins['alt'] + 1e-6

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

def train(model, dataset, normalizer, p_ref_interp, t_ref_interp, device, epochs=1000, batch_size=256):
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

            # 1. Sensor Loss
            s_x = batch['s_x'].to(device)
            s_p_true = batch['s_p'].to(device)
            s_t_true = batch['s_t'].to(device)
            s_p_ref = batch['s_p_ref'].to(device)
            s_t_ref = batch['s_t_ref'].to(device)

            s_preds = model(s_x)
            s_p_res_raw, s_t_res_raw = s_preds[:, 0], s_preds[:, 1]
            s_p_res, s_t_res = normalizer.scale_outputs(s_p_res_raw, s_t_res_raw)

            # Differential Base
            s_h_phys = normalizer.unnormalize_coords_h(s_x[:, 2])
            s_p_base, s_t_base = differential_baseline(s_h_phys, s_p_ref, s_t_ref)

            loss_obs_p = torch.mean(((s_p_base + s_p_res) - s_p_true)**2)
            loss_obs_t = torch.mean(((s_t_base + s_t_res) - s_t_true)**2)

            # 2. ERA5 Anchor Loss
            e_x = batch['e_x'].to(device)
            e_p_true = batch['e_p'].to(device)
            e_t_true = batch['e_t'].to(device)
            e_p_ref = batch['e_p_ref'].to(device)
            e_t_ref = batch['e_t_ref'].to(device)

            e_preds = model(e_x)
            e_p_res_raw, e_t_res_raw = e_preds[:, 0], e_preds[:, 1]
            e_p_res, e_t_res = normalizer.scale_outputs(e_p_res_raw, e_t_res_raw)

            e_h_phys = normalizer.unnormalize_coords_h(e_x[:, 2])
            e_p_base, e_t_base = differential_baseline(e_h_phys, e_p_ref, e_t_ref)

            loss_era5_p = torch.mean(((e_p_base + e_p_res) - e_p_true)**2)
            loss_era5_t = torch.mean(((e_t_base + e_t_res) - e_t_true)**2)

            # 3. Physics Loss
            noise = (torch.rand_like(s_x) - 0.5) * 0.1
            x_colloc = (s_x + noise).clamp(-1, 1)
            x_colloc[:, 2:] = x_colloc[:, 2:].clamp(0, 1)

            loss_pde = physics_loss(model, x_colloc, normalizer, p_ref_interp, t_ref_interp)

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

def solve_height(model, normalizer, lat, lon, time_ts, p_measured, p_ref_interp, t_ref_interp, device):
    """
    Inverse problem: Find h such that P_model(lat, lon, h, t) = P_measured
    Uses binary search on altitude.
    """
    model.eval()

    h_min = normalizer.mins['alt'] - 500
    h_max = normalizer.maxs['alt'] + 500

    low = h_min
    high = h_max

    t_lat = torch.tensor([lat], dtype=torch.float32, device=device)
    t_lon = torch.tensor([lon], dtype=torch.float32, device=device)
    t_time = torch.tensor([time_ts], dtype=torch.float32, device=device)

    # Get reference values for this time
    # Ensure inputs are floats
    ts_val = float(time_ts)
    p_ref_val_float = float(p_ref_interp(ts_val))
    t_ref_val_float = float(t_ref_interp(ts_val))

    p_ref_val = torch.tensor([p_ref_val_float], dtype=torch.float32, device=device)
    t_ref_val = torch.tensor([t_ref_val_float], dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(20):
            mid = (low + high) / 2

            n_lat = 2 * (t_lat - normalizer.mins['lat']) / (normalizer.maxs['lat'] - normalizer.mins['lat'] + 1e-6) - 1
            n_lon = 2 * (t_lon - normalizer.mins['lon']) / (normalizer.maxs['lon'] - normalizer.mins['lon'] + 1e-6) - 1
            n_h = (mid - normalizer.mins['alt']) / (normalizer.maxs['alt'] - normalizer.mins['alt'] + 1e-6)
            n_t = (t_time - normalizer.mins['timestamp']) / (normalizer.maxs['timestamp'] - normalizer.mins['timestamp'] + 1e-6)

            t_h = torch.tensor([n_h], dtype=torch.float32, device=device)
            inp = torch.stack([n_lat, n_lon, t_h, n_t], dim=-1)

            preds = model(inp)
            p_res_raw = preds[:, 0]
            p_res = p_res_raw * 5000.0

            # Differential Base
            t_h_phys = torch.tensor([mid], dtype=torch.float32, device=device)
            p_base, _ = differential_baseline(t_h_phys, p_ref_val, t_ref_val)

            p_total = p_base + p_res

            if p_total.item() > p_measured:
                low = mid
            else:
                high = mid

    return (low + high) / 2

def barometric_formula_height(P, P0=101325.0, T0=288.15):
    """Estimates height from pressure using standard atmosphere"""
    return (T0 / L_b) * ((P / P0) ** (-R * L_b / g) - 1)

def load_data():
    dataset_path = 'data/processed/sensor_data_with_real_era5.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'GeoIDbox/data/processed/sensor_data_with_real_era5.csv'
        if not os.path.exists(dataset_path):
             print(f"Error: dataset not found at {dataset_path} or data/processed/...")
             return None, None

    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    if 'processed_time' in df.columns:
        df['dt'] = pd.to_datetime(df['processed_time'])
        t_min = df['dt'].min()
        df['timestamp'] = (df['dt'] - t_min).dt.total_seconds()
        print(f"Time range: {df['dt'].min()} to {df['dt'].max()}")
    else:
        print("Warning: 'processed_time' column missing.")
        df['timestamp'] = df.index * 60

    return df

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_data()

    if df is None:
        return

    # --- 1. Extract Reference Station Data ---
    REF_UID = "20240911193733A012843A9994605977"
    ref_df = df[df['uid'] == REF_UID].copy()
    ref_df = ref_df.sort_values('timestamp')

    if len(ref_df) == 0:
        print(f"Error: Reference station {REF_UID} not found in data.")
        return

    print(f"Reference station samples: {len(ref_df)}")

    # Create Interpolation Functions
    # Use fill_value="extrapolate" to handle edge cases safely
    # timestamps are floats. pressure/temperature are floats.
    p_ref_interp = interp1d(ref_df['timestamp'].values, ref_df['avg_pressure'].values, kind='linear', fill_value="extrapolate")
    t_ref_interp = interp1d(ref_df['timestamp'].values, ref_df['avg_temperature'].values, kind='linear', fill_value="extrapolate")

    # --- 2. Prepare Sensor Data ---
    sensor_df = df[['avg_latitude', 'avg_longitude', 'avg_altitude', 'avg_pressure', 'avg_temperature', 'timestamp']].copy()
    sensor_df.columns = ['lat', 'lon', 'alt', 'pressure', 'temperature', 'timestamp']
    sensor_df = sensor_df.dropna()
    print(f"Sensor samples: {len(sensor_df)}")

    # --- 3. Prepare ERA5 Data ---
    era5_cols = ['avg_latitude', 'avg_longitude', 'era5_sp', 'era5_t2m', 'timestamp']
    era5_df = df[era5_cols].copy()
    era5_df.columns = ['lat', 'lon', 'sp', 't2m', 'timestamp']
    era5_df = era5_df.dropna()

    era5_df['lat_grid'] = era5_df['lat'].round(3)
    era5_df['lon_grid'] = era5_df['lon'].round(3)

    mean_sp = era5_df.groupby(['lat_grid', 'lon_grid'])['sp'].mean().reset_index()
    mean_sp.rename(columns={'sp': 'mean_sp'}, inplace=True)
    mean_sp['static_height'] = barometric_formula_height(mean_sp['mean_sp'])

    era5_df = pd.merge(era5_df, mean_sp, on=['lat_grid', 'lon_grid'], how='left')
    print(f"ERA5 samples: {len(era5_df)}")

    # Normalizer
    s_temp = sensor_df[['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']]
    e_temp = era5_df[['lat', 'lon', 'static_height', 'timestamp', 'sp', 't2m']]
    e_temp.columns = ['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']
    combined_temp = pd.concat([s_temp, e_temp], ignore_index=True)

    normalizer = DataNormalizer()
    normalizer.fit(combined_temp)

    # Dataset
    dataset = CombinedDataset(sensor_df, era5_df, normalizer, p_ref_interp, t_ref_interp)

    # Model
    model = WeatherField(num_freqs=10).to(device)

    # Train
    model, losses = train(model, dataset, normalizer, p_ref_interp, t_ref_interp, device, epochs=500)

    # Save
    torch.save(model.state_dict(), 'pinn_differential_model.pth')

    # Plot Loss
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss (Differential)')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve_diff.png')

    model.eval()
    print("\n--- Evaluation (Height Solver) ---")
    test_samples = sensor_df.sample(100)
    errors = []

    for idx, row in test_samples.iterrows():
        try:
            h_pred = solve_height(
                model, normalizer,
                row['lat'], row['lon'], row['timestamp'],
                row['pressure'],
                p_ref_interp, t_ref_interp,
                device
            )
            diff = h_pred - row['alt']
            errors.append(diff)
        except Exception as e:
            print(f"Error solving for sample {idx}: {e}")

    errors = np.array(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    print(f"Test MAE: {mae:.2f} m")
    print(f"Test RMSE: {rmse:.2f} m")

if __name__ == "__main__":
    main()
