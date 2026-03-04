
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt

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
            'h_hae': df['h_hae'].min(), 'timestamp': df['timestamp'].min()
        }
        self.maxs = {
            'lat': df['lat'].max(), 'lon': df['lon'].max(), 
            'h_hae': df['h_hae'].max(), 'timestamp': df['timestamp'].max()
        }
        
        # Pressure / Temperature Residual Scaling
        # We model residuals. Let's compute approx stats of residuals 
        # based on standard atmosphere as a rough prior
        # (Optimally we'd use ERA5 here but keeping it simple for standalone)
        self.p_stats = {'mean': df['pressure'].mean(), 'std': df['pressure'].std()}
        self.t_stats = {'mean': df['temperature'].mean(), 'std': df['temperature'].std()}
        
    def normalize_coords(self, lat, lon, h, t):
        # lat, lon -> [-1, 1]
        n_lat = 2 * (lat - self.mins['lat']) / (self.maxs['lat'] - self.mins['lat'] + 1e-6) - 1
        n_lon = 2 * (lon - self.mins['lon']) / (self.maxs['lon'] - self.mins['lon'] + 1e-6) - 1
        
        # h -> [0, 1]
        n_h = (h - self.mins['h_hae']) / (self.maxs['h_hae'] - self.mins['h_hae'] + 1e-6)
        
        # t -> [0, 1]
        n_t = (t - self.mins['timestamp']) / (self.maxs['timestamp'] - self.mins['timestamp'] + 1e-6)
        
        return torch.stack([n_lat, n_lon, n_h, n_t], dim=-1)

    def unnormalize_coords_h(self, n_h):
        return n_h * (self.maxs['h_hae'] - self.mins['h_hae'] + 1e-6) + self.mins['h_hae']

    # We predict residuals. We need scaling for P and T to keep gradients happy.
    # We will predict outputs in roughly [-1, 1] range.
    # Real P ~ 100,000 Pa. Residual might be 100-1000 Pa. Scale factor 1000.
    # Real T ~ 288 K. Residual might be 1-5 K. Scale factor 10.
    
    def scale_outputs(self, p_res_raw, t_res_raw):
        # Raw model output -> Physical units
        p_res = p_res_raw * 1000.0 
        t_res = t_res_raw * 10.0
        return p_res, t_res

class CombinedDataset(Dataset):
    def __init__(self, sensor_df, era5_df, normalizer):
        self.normalizer = normalizer
        
        # Process Sensor Data
        self.s_coords = normalizer.normalize_coords(
            torch.tensor(sensor_df['lat'].values, dtype=torch.float32),
            torch.tensor(sensor_df['lon'].values, dtype=torch.float32),
            torch.tensor(sensor_df['h_hae'].values, dtype=torch.float32),
            torch.tensor(sensor_df['timestamp'].values, dtype=torch.float32)
        )
        self.s_p = torch.tensor(sensor_df['pressure'].values, dtype=torch.float32)
        self.s_t = torch.tensor(sensor_df['temperature'].values, dtype=torch.float32)
        
        # Process ERA5 Data (Anchors)
        self.e_coords = normalizer.normalize_coords(
            torch.tensor(era5_df['lat'].values, dtype=torch.float32),
            torch.tensor(era5_df['lon'].values, dtype=torch.float32),
            torch.tensor(era5_df['h_hae'].values, dtype=torch.float32),
            torch.tensor(era5_df['timestamp'].values, dtype=torch.float32)
        )
        self.e_p = torch.tensor(era5_df['pressure'].values, dtype=torch.float32)
        self.e_t = torch.tensor(era5_df['temperature'].values, dtype=torch.float32)
        
    def __len__(self):
        return max(len(self.s_coords), len(self.e_coords))
    
    def __getitem__(self, idx):
        # Sampling strategy: Return one sensor point and one ERA5 point per batch item
        s_idx = idx % len(self.s_coords)
        e_idx = idx % len(self.e_coords) # Random sampling might be better but this is deterministic
        
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
    
    h_range = normalizer.maxs['h_hae'] - normalizer.mins['h_hae'] + 1e-6
    
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
            # Weights need tuning based on magnitude
            # Obs/ERA5 are in Pa^2 (10^4~10^6). PDE is (Pa/m)^2 ~ 1-100.
            # Normalizing loss values or coefficients helps.
            
            total_loss = (loss_obs_p + loss_era5_p) * 1e-4 + \
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
    Using Newton-Raphson.
    """
    model.eval()
    
    # 1. Initial Guess using Barometric Formula
    # h = T0/Lb * ((P/P0)^(-R*Lb/g) - 1)
    h_guess = (T0 / L_b) * ((p_measured / P0) ** (-R * L_b / g) - 1)
    h_current = torch.tensor([h_guess], dtype=torch.float32, device=device, requires_grad=True)
    
    # Static inputs
    lat_ts = torch.tensor([lat], dtype=torch.float32, device=device)
    lon_ts = torch.tensor([lon], dtype=torch.float32, device=device)
    t_ts = torch.tensor([time_ts], dtype=torch.float32, device=device)
    target_p = torch.tensor([p_measured], dtype=torch.float32, device=device)
    
    optimizer_inv = optim.LBFGS([h_current], lr=1, max_iter=20) # LBFGS is good for inversion
    
    # Prepare normalized coords function
    def get_input_tensor(h_val):
        h_norm = (h_val - normalizer.mins['h_hae']) / (normalizer.maxs['h_hae'] - normalizer.mins['h_hae'] + 1e-6)
        
        l_norm = 2 * (lat_ts - normalizer.mins['lat']) / (normalizer.maxs['lat'] - normalizer.mins['lat'] + 1e-6) - 1
        n_norm = 2 * (lon_ts - normalizer.mins['lon']) / (normalizer.maxs['lon'] - normalizer.mins['lon'] + 1e-6) - 1
        t_norm = (t_ts - normalizer.mins['timestamp']) / (normalizer.maxs['timestamp'] - normalizer.mins['timestamp'] + 1e-6)
        
        return torch.stack([l_norm, n_norm, h_norm, t_norm], dim=-1)

    # Newton Step manually or via optimizer
    # Let's use simple Newton: z_{n+1} = z_n - f(z_n) / f'(z_n)
    # where f(z) = P_pred(z) - P_meas
    
    h_range = normalizer.maxs['h_hae'] - normalizer.mins['h_hae'] + 1e-6
    
    for i in range(15):
        # Create input
        x_in = get_input_tensor(h_current)
        
        # Ensure we track gradients for h_current (via the input tensor)
        # Note: x_in is created from h_current, so graph is connected
        
        # Forward pass for P
        preds = model(x_in)
        p_res_raw = preds[:, 0]
        p_res, _ = normalizer.scale_outputs(p_res_raw, torch.zeros_like(p_res_raw))
        p_base, _ = standard_atmosphere(h_current)
        p_pred = p_base + p_res
        
        diff = p_pred - target_p
        
        if torch.abs(diff) < 0.1: # Converged to 0.1 Pa
            break
            
        # Compute gradient dP/dh
        grad_p = torch.autograd.grad(p_pred, h_current, create_graph=False)[0]
        
        # Newton update
        with torch.no_grad():
            h_current = h_current - diff / (grad_p + 1e-8)
            
        h_current.requires_grad_(True)
        
    return h_current.item()

def generate_synthetic_data():
    # Helper to generate dummy data if files are missing or for testing structure
    print("Generating synthetic data for testing...")
    
    # Lat/Lon center (Shenzhen approx)
    lat0, lon0 = 22.54, 114.05
    
    # 1. Sensor Data (Low altitude, sparse)
    n_sensor = 200
    s_lat = lat0 + np.random.randn(n_sensor) * 0.01
    s_lon = lon0 + np.random.randn(n_sensor) * 0.01
    s_h = np.random.uniform(0, 100, n_sensor) # Low altitude
    s_t = np.sort(np.random.uniform(0, 10000, n_sensor))
    
    s_p_base, s_t_base = standard_atmosphere(s_h)
    # Add some noise/bias
    s_pressure = s_p_base + np.random.randn(n_sensor) * 10
    s_temperature = s_t_base + np.random.randn(n_sensor) * 1.0
    
    sensor_df = pd.DataFrame({
        'lat': s_lat, 'lon': s_lon, 'h_hae': s_h, 'timestamp': s_t,
        'pressure': s_pressure, 'temperature': s_temperature
    })
    
    # 2. ERA5 Data (Full column, grid)
    n_era = 500
    e_lat = lat0 + np.random.randn(n_era) * 0.05 # Wider area
    e_lon = lon0 + np.random.randn(n_era) * 0.05
    e_h = np.random.uniform(0, 5000, n_era) # Up to 5km
    e_t = np.random.uniform(0, 10000, n_era)
    
    e_p_base, e_t_base = standard_atmosphere(e_h)
    e_pressure = e_p_base + np.random.randn(n_era) * 20
    e_temperature = e_t_base + np.random.randn(n_era) * 2.0
    
    era5_df = pd.DataFrame({
        'lat': e_lat, 'lon': e_lon, 'h_hae': e_h, 'timestamp': e_t,
        'pressure': e_pressure, 'temperature': e_temperature
    })
    
    return sensor_df, era5_df

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    try:
        # User should replace this with actual loading logic
        # Loading logic similar to pinn_working_v1.py
        # For now, using synthetic data to ensure runnability
        if os.path.exists("pinn_data.csv"): # Placeholder for real file
            df = pd.read_csv("pinn_data.csv")
            # Splitting logic would go here
            sensor_df = df.iloc[:100]
            era5_df = df.iloc[100:]
        else:
            sensor_df, era5_df = generate_synthetic_data()
            
    except Exception as e:
        print(f"Error loading data: {e}")
        sensor_df, era5_df = generate_synthetic_data()

    # Normalization
    normalizer = DataNormalizer()
    normalizer.fit(pd.concat([sensor_df, era5_df]))
    
    # Dataset & Model
    dataset = CombinedDataset(sensor_df, era5_df, normalizer)
    model = WeatherField(num_freqs=10).to(device)
    
    # Train
    model, history = train(model, dataset, normalizer, device, epochs=500)
    
    # Evaluation (Inversion Test)
    print("\n--- Evaluation (Height Solver) ---")
    test_idx = 0
    sample = sensor_df.iloc[test_idx]
    
    real_h = sample['h_hae']
    p_meas = sample['pressure']
    
    pred_h = solve_height(model, normalizer, sample['lat'], sample['lon'], sample['timestamp'], p_meas, device)
    
    print(f"True Height: {real_h:.2f} m")
    print(f"Pred Height: {pred_h:.2f} m")
    print(f"Error: {abs(real_h - pred_h):.2f} m")
    
    # Plot Loss
    plt.figure()
    plt.plot(history)
    plt.yscale('log')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('pinn_field_loss.png')
    print("Loss plot saved to pinn_field_loss.png")

if __name__ == "__main__":
    main()
