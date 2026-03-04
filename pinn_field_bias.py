
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# Coordinate-based Neural Field Model with Sensor Bias
class WeatherFieldWithBias(nn.Module):
    def __init__(self, num_sensors, hidden_dim=256, num_layers=4, num_freqs=10, sigma=1.0):
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

        layers.append(nn.Linear(hidden_dim, 2)) # Outputs: P_residual_norm, T_residual_norm
        self.net = nn.Sequential(*layers)

        # Initialize final layer bias to 0.0 so we start with near-zero residuals
        self.net[-1].bias.data.fill_(0.0)
        self.net[-1].weight.data *= 0.1 # Small weights for initial zero output

        # Learnable sensor biases
        # Storing bias in Pascals directly
        self.sensor_biases = nn.Embedding(num_sensors, 1)

        # Initialize sensor biases to 0.0
        self.sensor_biases.weight.data.fill_(0.0)

    def forward(self, x, sensor_id):
        # Input x: (batch, 4) -> (lat_norm, lon_norm, z_norm, t_norm)
        # sensor_id: (batch,)

        field_out = self.net(self.fourier(x)) # (batch, 2)

        p_res_norm = field_out[:, 0]
        t_res_norm = field_out[:, 1]

        # Scaling logic: p_pred = (p_res_norm * 5000.0) + bias
        # We output the COMPONENTS, not the sum, to allow flexibility in loss
        # But for convenience, let's return the scaled physical values

        p_res_field_pa = p_res_norm * 5000.0
        t_res_field = t_res_norm * 20.0 # Scaling T roughly

        # Get bias for this batch
        # bias shape: (batch, 1) -> (batch,)
        # These are in Pa
        p_bias_pa = self.sensor_biases(sensor_id).squeeze(-1)

        # Total Residual = Field Residual + Sensor Bias
        p_res_total_pa = p_res_field_pa + p_bias_pa

        # We do not apply bias to Temperature for now
        t_res_total = t_res_field

        return torch.stack([p_res_total_pa, t_res_total], dim=-1)

    def predict_base(self, x):
        """Predict without adding sensor bias (for physics loss or reference queries)"""
        field_out = self.net(self.fourier(x))
        p_res_norm = field_out[:, 0]
        t_res_norm = field_out[:, 1]

        p_res_field_pa = p_res_norm * 5000.0
        t_res_field = t_res_norm * 20.0

        return torch.stack([p_res_field_pa, t_res_field], dim=-1)

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
        self.p_stats = {'mean': df['pressure'].mean(), 'std': df['pressure'].std()}
        self.t_stats = {'mean': df['temperature'].mean(), 'std': df['temperature'].std()}

    def normalize_coords(self, lat, lon, h, t):
        n_lat = 2 * (lat - self.mins['lat']) / (self.maxs['lat'] - self.mins['lat'] + 1e-6) - 1
        n_lon = 2 * (lon - self.mins['lon']) / (self.maxs['lon'] - self.mins['lon'] + 1e-6) - 1
        n_h = (h - self.mins['alt']) / (self.maxs['alt'] - self.mins['alt'] + 1e-6)
        n_t = (t - self.mins['timestamp']) / (self.maxs['timestamp'] - self.mins['timestamp'] + 1e-6)
        return torch.stack([n_lat, n_lon, n_h, n_t], dim=-1)

    def unnormalize_coords_h(self, n_h):
        return n_h * (self.maxs['alt'] - self.mins['alt'] + 1e-6) + self.mins['alt']

class CombinedDataset(Dataset):
    def __init__(self, sensor_df, era5_df, normalizer, era5_id):
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
        self.s_ids = torch.tensor(sensor_df['sensor_id_enc'].values, dtype=torch.long)

        # Process ERA5 Data (Anchors)
        self.e_coords = normalizer.normalize_coords(
            torch.tensor(era5_df['lat'].values, dtype=torch.float32),
            torch.tensor(era5_df['lon'].values, dtype=torch.float32),
            torch.tensor(era5_df['static_height'].values, dtype=torch.float32),
            torch.tensor(era5_df['timestamp'].values, dtype=torch.float32)
        )
        self.e_p = torch.tensor(era5_df['sp'].values, dtype=torch.float32)
        self.e_t = torch.tensor(era5_df['t2m'].values, dtype=torch.float32)
        # Assign ERA5 ID to all ERA5 samples
        self.e_ids = torch.full((len(era5_df),), era5_id, dtype=torch.long)

    def __len__(self):
        return max(len(self.s_coords), len(self.e_coords))

    def __getitem__(self, idx):
        # Sampling strategy: Return one sensor point and one ERA5 point per batch item
        s_idx = idx % len(self.s_coords)
        e_idx = idx % len(self.e_coords)

        return {
            's_x': self.s_coords[s_idx], 's_p': self.s_p[s_idx], 's_t': self.s_t[s_idx], 's_id': self.s_ids[s_idx],
            'e_x': self.e_coords[e_idx], 'e_p': self.e_p[e_idx], 'e_t': self.e_t[e_idx], 'e_id': self.e_ids[e_idx]
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
    Using BASE field prediction (no bias chould affect physics consistency of the field itself)
    """
    x_colloc.requires_grad_(True)

    # Predict residuals (BASE ONLY, no specific sensor bias)
    preds = model.predict_base(x_colloc)
    p_res_pa, t_res_pa = preds[:, 0], preds[:, 1]

    # Recover physical height for standard atmosphere baseline
    h_phys = normalizer.unnormalize_coords_h(x_colloc[:, 2])

    # Base state
    p_base, t_base = standard_atmosphere(h_phys)

    # Total fields
    P_total = p_base + p_res_pa
    T_total = t_base + t_res_pa

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
    rho = P_total / (R * T_total)

    # 3. Hydrostatic Balance Residual
    pde_res = dP_dz + rho * g

    return torch.mean(pde_res ** 2)

def train(model, dataset, normalizer, device, epochs=1000, batch_size=256):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    losses = []
    lambda_bias = 1e-6  # Weak Regularization strength for biases

    print("Starting training with learned bias...")
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
            s_ids = batch['s_id'].to(device)

            # Forward pass now returns PA scaled residuals + bias
            s_preds = model(s_x, s_ids)
            s_p_res_total, s_t_res_total = s_preds[:, 0], s_preds[:, 1]

            # Base
            s_h_phys = normalizer.unnormalize_coords_h(s_x[:, 2])
            s_p_base, s_t_base = standard_atmosphere(s_h_phys)

            # Loss in Pascals (approx 10^0 to 10^2)
            loss_obs_p = torch.mean(((s_p_base + s_p_res_total) - s_p_true)**2)
            loss_obs_t = torch.mean(((s_t_base + s_t_res_total) - s_t_true)**2)

            # 2. ERA5 Anchor Loss (MSE)
            e_x = batch['e_x'].to(device)
            e_p_true = batch['e_p'].to(device)
            e_t_true = batch['e_t'].to(device)
            e_ids = batch['e_id'].to(device)

            e_preds = model(e_x, e_ids)
            e_p_res_total, e_t_res_total = e_preds[:, 0], e_preds[:, 1]

            e_h_phys = normalizer.unnormalize_coords_h(e_x[:, 2])
            e_p_base, e_t_base = standard_atmosphere(e_h_phys)

            loss_era5_p = torch.mean(((e_p_base + e_p_res_total) - e_p_true)**2)
            loss_era5_t = torch.mean(((e_t_base + e_t_res_total) - e_t_true)**2)

            # 3. Physics Loss (Hydrostatic)
            noise = (torch.rand_like(s_x) - 0.5) * 0.1
            x_colloc = (s_x + noise).clamp(-1, 1)
            x_colloc[:, 2:] = x_colloc[:, 2:].clamp(0, 1)

            loss_pde = physics_loss(model, x_colloc, normalizer)

            # 4. Bias Regularization (L2 on bias weights)
            # bias weights are in Pa. We want to penalize large biases.
            # If bias is 100 Pa, bias^2 is 10000.
            loss_bias_reg = lambda_bias * torch.mean(model.sensor_biases.weight**2)

            # Weights adjustment
            # Sensors and ERA5 are both trusted sources, but ERA5 is larger scale.
            # Measurements are more precise.

            total_loss = (loss_obs_p + loss_era5_p) * 1e-5 + \
                         (loss_obs_t + loss_era5_t) * 1e-1 + \
                         loss_pde * 1.0 + \
                         loss_bias_reg

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.5f} | OBS_P={loss_obs_p.item():.1f} | PDE={loss_pde.item():.3f} | BiasReg={loss_bias_reg.item():.3f}")

    print(f"Training finished in {time.time() - start_time:.1f}s")
    return model, losses

def solve_height_with_bias(model, normalizer, lat, lon, time_ts, p_measured, sensor_id_tensor, device):
    """
    Inverse problem with specific sensor bias support
    """
    model.eval()

    h_min = normalizer.mins['alt'] - 500
    h_max = normalizer.maxs['alt'] + 500

    low = h_min
    high = h_max

    t_lat = torch.tensor([lat], dtype=torch.float32, device=device)
    t_lon = torch.tensor([lon], dtype=torch.float32, device=device)
    t_time = torch.tensor([time_ts], dtype=torch.float32, device=device)

    # Ensure sensor_id_tensor is correct shape/device
    if not isinstance(sensor_id_tensor, torch.Tensor):
        sensor_id_tensor = torch.tensor([sensor_id_tensor], dtype=torch.long, device=device)
    else:
        sensor_id_tensor = sensor_id_tensor.to(device)

    # Ensure it's a 1D tensor (Scalar or size 1)
    if sensor_id_tensor.dim() == 0:
         sensor_id_tensor = sensor_id_tensor.unsqueeze(0)

    # Pre-fetch bias for this sensor (constant for the solving process)
    with torch.no_grad():
        # model.sensor_biases returns (1, 1), squeeze to scalar
        bias_pa = model.sensor_biases(sensor_id_tensor).squeeze().item()

    with torch.no_grad():
        for _ in range(20):
            mid = (low + high) / 2

            n_lat = 2 * (t_lat - normalizer.mins['lat']) / (normalizer.maxs['lat'] - normalizer.mins['lat'] + 1e-6) - 1
            n_lon = 2 * (t_lon - normalizer.mins['lon']) / (normalizer.maxs['lon'] - normalizer.mins['lon'] + 1e-6) - 1
            n_h = (mid - normalizer.mins['alt']) / (normalizer.maxs['alt'] - normalizer.mins['alt'] + 1e-6)
            n_t = (t_time - normalizer.mins['timestamp']) / (normalizer.maxs['timestamp'] - normalizer.mins['timestamp'] + 1e-6)

            t_h = torch.tensor([n_h], dtype=torch.float32, device=device)

            inp = torch.stack([n_lat, n_lon, t_h, n_t], dim=-1)

            # Predict BASE FIELD (without bias)
            preds = model.predict_base(inp)
            p_res_field_pa = preds[0, 0].item()

            # Calculate total predicted pressure: Base + FieldRes + Bias
            p_base_chem, _ = standard_atmosphere(mid)
            p_total = p_base_chem + p_res_field_pa + bias_pa

            if p_total > p_measured:
                # Pressure decreases with height
                # If predicted > measured, we are at a lower altitude than the target
                # We need to go higher to reduce pressure
                low = mid
            else:
                high = mid

    return (low + high) / 2

def barometric_formula_height(P, P0=101325.0, T0=288.15):
    return (T0 / L_b) * ((P / P0) ** (-R * L_b / g) - 1)

def load_data():
    # UPDATED PATH as requested
    dataset_path = 'GeoIDbox/data/processed/sensor_data_filtered_outliers.csv'

    if not os.path.exists(dataset_path):
         # Fallback to local
         dataset_path = 'data/processed/sensor_data_filtered_outliers.csv'
         if not os.path.exists(dataset_path):
             print(f"Error: dataset not found at {dataset_path}")
             return None, None

    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    if 'processed_time' in df.columns:
        df['dt'] = pd.to_datetime(df['processed_time'])
        t_min = df['dt'].min()
        df['timestamp'] = (df['dt'] - t_min).dt.total_seconds()
    elif 'timestamp' in df.columns:
        # If timestamp already exists
        pass
    else:
        df['timestamp'] = df.index * 60

    if 'uid' not in df.columns:
        if 'sensor_id' in df.columns:
            df['uid'] = df['sensor_id']
        else:
             print("Warning: No UID found, using random assignment.")
             df['uid'] = np.random.randint(0, 5, len(df))

    sensor_df = df[['uid', 'avg_latitude', 'avg_longitude', 'avg_altitude', 'avg_pressure', 'avg_temperature', 'timestamp']].copy()
    sensor_df.columns = ['uid', 'lat', 'lon', 'alt', 'pressure', 'temperature', 'timestamp']
    sensor_df = sensor_df.dropna()
    print(f"Sensor samples: {len(sensor_df)}")

    # Check for ERA5 columns, if not present we need a fallback or they might be in the file
    # The processed file usually has them merged.
    if 'era5_sp' in df.columns:
        era5_cols = ['avg_latitude', 'avg_longitude', 'era5_sp', 'era5_t2m', 'timestamp']
        era5_df = df[era5_cols].copy()
        era5_df.columns = ['lat', 'lon', 'sp', 't2m', 'timestamp']
        era5_df = era5_df.dropna()
    else:
        print("Warning: No ERA5 columns found in CSV. Using dummy ERA5 data derived from sensors.")
        era5_df = sensor_df.copy()
        # Fake ERA5 from standard atm + simple noise
        era5_df['sp'] = era5_df['pressure']
        era5_df['t2m'] = era5_df['temperature']

    # Simple grid aggregation for ERA5 visualization/anchor points
    era5_df['lat_grid'] = era5_df['lat'].round(3)
    era5_df['lon_grid'] = era5_df['lon'].round(3)

    mean_sp = era5_df.groupby(['lat_grid', 'lon_grid'])['sp'].mean().reset_index()
    if 'static_height' not in era5_df.columns:
        mean_sp.rename(columns={'sp': 'mean_sp'}, inplace=True)
        mean_sp['static_height'] = barometric_formula_height(mean_sp['mean_sp'])
        era5_df = pd.merge(era5_df, mean_sp, on=['lat_grid', 'lon_grid'], how='left')

    print(f"ERA5 samples: {len(era5_df)}")

    return sensor_df, era5_df

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sensor_df, era5_df = load_data()

    if sensor_df is None:
        return

    # --- Encode Sensor IDs ---
    le = LabelEncoder()
    sensor_df['sensor_id_enc'] = le.fit_transform(sensor_df['uid'])

    num_sensors = len(le.classes_)
    print(f"Found {num_sensors} unique sensors: {le.classes_}")

    era5_id = num_sensors
    print(f"ERA5 assigned ID: {era5_id}")

    total_ids = num_sensors + 1

    # --- Hold-out Strategy ---
    held_out_sensor_id = None
    if num_sensors > 1:
        # Hold out the last one for demonstration
        held_out_sensor_id = num_sensors - 1
        held_out_mask = sensor_df['sensor_id_enc'] == held_out_sensor_id

        train_sensor_df = sensor_df[~held_out_mask].copy()
        test_sensor_df = sensor_df[held_out_mask].copy()

        print(f"Holding out sensor {le.classes_[held_out_sensor_id]} ({len(test_sensor_df)} samples) for calibration test.")
    else:
        train_sensor_df = sensor_df.copy()
        test_sensor_df = sensor_df.sample(0)
        print("Not enough sensors to hold out one.")

    s_temp = train_sensor_df[['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']]
    # Rename columns for e_temp to match s_temp for normalization fitting
    e_temp = era5_df[['lat', 'lon', 'static_height', 'timestamp', 'sp', 't2m']].copy()
    e_temp.columns = ['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']

    combined_temp = pd.concat([s_temp, e_temp], ignore_index=True)

    normalizer = DataNormalizer()
    normalizer.fit(combined_temp)

    dataset = CombinedDataset(train_sensor_df, era5_df, normalizer, era5_id)

    model = WeatherFieldWithBias(num_sensors=total_ids, num_freqs=10).to(device)

    model, losses = train(model, dataset, normalizer, device, epochs=500)

    torch.save(model.state_dict(), 'pinn_bias_model.pth')

    print("\n--- Learned Sensor Biases ---")
    with torch.no_grad():
        biases_pa = model.sensor_biases.weight.data.cpu().numpy().squeeze()
        # biases_pa is already in Pascals

        for i, uid in enumerate(le.classes_):
             if held_out_sensor_id is None or i != held_out_sensor_id:
                print(f"Sensor {uid} (ID {i}): Bias = {biases_pa[i]:.2f} Pa")

        if era5_id < len(biases_pa):
            print(f"ERA5 (ID {era5_id}): Bias = {biases_pa[era5_id]:.2f} Pa")

    print("\n--- Evaluation: Known Sensors (Training Set) ---")
    model.eval()
    if len(train_sensor_df) > 0:
        known_samples = train_sensor_df.sample(min(200, len(train_sensor_df)))
        errors = []

        for idx, row in known_samples.iterrows():
            sid = torch.tensor([row['sensor_id_enc']], dtype=torch.long, device=device)
            h_pred = solve_height_with_bias(model, normalizer, row['lat'], row['lon'], row['timestamp'], row['pressure'], sid, device)
            # h_pred is tensor or float? solve_height returns (low+high)/2 (float/tensor)
            # Convert to CPU numpy for diff
            if isinstance(h_pred, torch.Tensor):
                h_pred = h_pred.item()
            errors.append(h_pred - row['alt'])

        errors = np.array(errors)
        print(f"Known Sensors MAE: {np.mean(np.abs(errors)):.2f} m")
        print(f"Known Sensors RMSE: {np.sqrt(np.mean(errors**2)):.2f} m")

    # --- Evaluation 2: New Sensor Calibration ---
    if len(test_sensor_df) > 10:
        print("\n--- Evaluation: New Sensor Calibration ---")
        test_sensor_df = test_sensor_df.sort_values('timestamp')

        calib_df = test_sensor_df.iloc[:5]
        eval_df = test_sensor_df.iloc[5:]

        print(f"Calibrating using {len(calib_df)} samples...")

        # Calculate Quick Calibration Bias
        biases_needed = []

        with torch.no_grad():
             for idx, row in calib_df.iterrows():
                # Prepare input
                n_lat = 2 * (row['lat'] - normalizer.mins['lat']) / (normalizer.maxs['lat'] - normalizer.mins['lat'] + 1e-6) - 1
                n_lon = 2 * (row['lon'] - normalizer.mins['lon']) / (normalizer.maxs['lon'] - normalizer.mins['lon'] + 1e-6) - 1
                n_h = (row['alt'] - normalizer.mins['alt']) / (normalizer.maxs['alt'] - normalizer.mins['alt'] + 1e-6)
                n_t = (row['timestamp'] - normalizer.mins['timestamp']) / (normalizer.maxs['timestamp'] - normalizer.mins['timestamp'] + 1e-6)

                inp = torch.tensor([[n_lat, n_lon, n_h, n_t]], dtype=torch.float32, device=device)

                # Predict BASE FIELD (no bias)
                preds = model.predict_base(inp)
                p_res_field_pa = preds[0, 0].item()

                p_base_chem, _ = standard_atmosphere(row['alt'])

                # P_meas = P_base_chem + P_res_field + Bias
                # Bias = P_meas - P_base_chem - P_res_field

                bias_sample = row['pressure'] - p_base_chem - p_res_field_pa
                biases_needed.append(bias_sample)

        new_bias_pa = np.mean(biases_needed)
        print(f"Calculated Quick Calibration Bias: {new_bias_pa:.2f} Pa")

        # Update model for this ID
        # Since embedding stores Pa directly, we just store it
        with torch.no_grad():
             # We can't update weights in place easily if tracking gradients, but here we are in no_grad
             model.sensor_biases.weight[held_out_sensor_id] = float(new_bias_pa)

        errors_calib = []
        # Sample eval set to save time if large
        eval_subset = eval_df.sample(min(100, len(eval_df)))
        for idx, row in eval_subset.iterrows():
            sid = torch.tensor([held_out_sensor_id], dtype=torch.long, device=device)
            h_pred = solve_height_with_bias(model, normalizer, row['lat'], row['lon'], row['timestamp'], row['pressure'], sid, device)

            if isinstance(h_pred, torch.Tensor):
                h_pred = h_pred.item()
            errors_calib.append(h_pred - row['alt'])

        errors_calib = np.array(errors_calib)
        print(f"New Sensor (Calibrated) MAE: {np.mean(np.abs(errors_calib)):.2f} m")
        print(f"New Sensor (Calibrated) RMSE: {np.sqrt(np.mean(errors_calib**2)):.2f} m")

if __name__ == "__main__":
    main()
