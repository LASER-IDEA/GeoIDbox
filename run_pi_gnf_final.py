import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin, asin, sqrt
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_isa_height(p_pa, p0=101325.0):
    p = np.clip(p_pa, 1000.0, None)
    return (288.15 / 0.0065) * (1.0 - (p / p0) ** (0.0065 * 287.05 / 9.80665))

class GatedFusionPINN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 2) 
        )
        
        self.micro_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        # ZERO INITIALIZATION TRICK for proper PINN anchoring
        self.gate_net[-1].weight.data.fill_(0.0)
        self.gate_net[-1].bias.data.fill_(0.0) # sigmoid(0)=0.5 * 2.0 = 1.0!
        
        self.micro_net[-1].weight.data.fill_(0.0)
        self.micro_net[-1].bias.data.fill_(0.0) # exactly 0.0 additive bias
        
    def forward(self, x_features, spatial_prior, weather_prior):
        gates = torch.sigmoid(self.gate_net(x_features)) * 2.0 
        g_s, g_w = gates[:, 0], gates[:, 1]
        
        micro_res = self.micro_net(x_features).squeeze(-1)
        # Bound micro_res heavily so it doesn't randomly diverge to 20m off
        micro_res = torch.tanh(micro_res / 10.0) * 10.0 
        
        total_residual_pred = (g_s * spatial_prior) + (g_w * weather_prior) + micro_res
        return total_residual_pred, micro_res

print("Loading Data...")
df = pd.read_csv('data/processed/sensor_data_stabilized.csv')
df['processed_time'] = pd.to_datetime(df['processed_time'])
df['time_key'] = df['processed_time'].dt.round('10min')

df['n_geoid'] = 20.0
df['h_true_msl'] = df['avg_altitude'] - df['n_geoid']
df['h_isa'] = compute_isa_height(df['avg_pressure'].values)
df['residual'] = df['h_true_msl'] - df['h_isa']

hours = df['processed_time'].dt.hour + df['processed_time'].dt.minute / 60.0
df['time_sin'] = np.sin(2 * np.pi * hours / 24.0)
df['time_cos'] = np.cos(2 * np.pi * hours / 24.0)

physics_features = ['avg_pressure', 'avg_temperature', 'avg_humidity', 'time_sin', 'time_cos']
uids = sorted(df['uid'].unique())
lat_lon = df.groupby('uid').agg({'avg_latitude':'first', 'avg_longitude':'first'})
results = []

for fold, test_sensor in enumerate(uids):
    train_df = df[df['uid'] != test_sensor].copy()
    test_df = df[df['uid'] == test_sensor].copy()
    
    spatial_bias_map = train_df.groupby('uid')['residual'].mean()
    train_sensors = train_df.groupby('uid').agg({'avg_latitude':'first', 'avg_longitude':'first'})
    train_sensors['spatial_bias'] = spatial_bias_map
    
    def get_idw_bias(lat, lon):
        dists = np.sqrt((train_sensors['avg_latitude']-lat)**2 + (train_sensors['avg_longitude']-lon)**2)
        weights = 1.0 / np.maximum(dists, 1e-6)
        weights = weights / weights.sum()
        return np.average(train_sensors['spatial_bias'], weights=weights)
    
    train_df['spatial_prior'] = train_df['uid'].map(spatial_bias_map)
    test_lat, test_lon = lat_lon.loc[test_sensor]
    test_df['spatial_prior'] = get_idw_bias(test_lat, test_lon)
    
    train_df['weather_residual'] = train_df['residual'] - train_df['spatial_prior']
    weather_map = train_df.groupby('time_key')['weather_residual'].mean()
    
    train_df['weather_prior'] = train_df['time_key'].map(weather_map).fillna(0)
    test_df['weather_prior'] = test_df['time_key'].map(weather_map).fillna(0)
    
    scaler = StandardScaler()
    X_train_phys = scaler.fit_transform(train_df[physics_features].values)
    X_test_phys = scaler.transform(test_df[physics_features].values)
    
    X_tr = torch.FloatTensor(X_train_phys).to(device)
    S_tr = torch.FloatTensor(train_df['spatial_prior'].values).to(device)
    W_tr = torch.FloatTensor(train_df['weather_prior'].values).to(device)
    Y_tr = torch.FloatTensor(train_df['residual'].values).to(device)
    
    X_te = torch.FloatTensor(X_test_phys).to(device)
    S_te = torch.FloatTensor(test_df['spatial_prior'].values).to(device)
    W_te = torch.FloatTensor(test_df['weather_prior'].values).to(device)
    Y_te = torch.FloatTensor(test_df['residual'].values).to(device)
    
    model = GatedFusionPINN(feature_dim=len(physics_features)).to(device)
    # L2 decay strictly applies to non-zeroed weights!
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    
    batch_size = 4096
    num_samples = len(X_tr)
    
    for epoch in range(15):
        model.train()
        idx = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            b_idx = idx[i:i+batch_size]
            pred, micro = model(X_tr[b_idx], S_tr[b_idx], W_tr[b_idx])
            # Strict regularization on gate divergence
            gate_penalty = torch.mean((pred - (S_tr[b_idx] + W_tr[b_idx]))**2)
            loss = nn.L1Loss()(pred, Y_tr[b_idx]) + 0.1 * gate_penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_pred, _ = model(X_te, S_te, W_te)
        v1_pred = S_te + W_te
        nn_err = torch.abs(test_pred - Y_te).mean().item()
        v1_err = torch.abs(v1_pred - Y_te).mean().item()
        
    print(f"Fold {fold+1:d} ({test_sensor[-8:]}) | V1 Base: {v1_err:5.2f}m -> PINN: {nn_err:5.2f}m")
    results.append({'v1': v1_err, 'nn': nn_err})

print(f"\n=============================================")
print(f"Classical V1.0 Baseline Total: {np.mean([r['v1'] for r in results]):.2f}m")
print(f"Gated PINN Fusion Total MAE:   {np.mean([r['nn'] for r in results]):.2f}m")
print(f"Improvement:                   {np.mean([r['v1'] for r in results]) - np.mean([r['nn'] for r in results]):.2f}m")
print(f"=============================================")
