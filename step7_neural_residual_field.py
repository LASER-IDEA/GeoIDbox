import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 定义模型组件：位置编码 & MLP
# ==========================================

class PositionalEncoding(nn.Module):
    """
    位置编码层。将低维坐标映射到高维空间，帮助网络学习高频细节。
    Inputs: [x, y, z, t]
    """
    def __init__(self, input_dim, L=10):
        super().__init__()
        self.L = L # 编码频率的数量
        self.input_dim = input_dim
        # 创建频率序列 [2^0, 2^1, ..., 2^(L-1)]
        self.freq_bands = torch.pow(2, torch.linspace(0, L-1, L)).to(device)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        encoded = [x]
        for freq in self.freq_bands:
            # 对每个输入维度应用 sin 和 cos 变换
            for i in range(self.input_dim):
                encoded.append(torch.sin(x[:, i:i+1] * freq * np.pi))
                encoded.append(torch.cos(x[:, i:i+1] * freq * np.pi))
        # 连接所有编码特征
        return torch.cat(encoded, dim=-1)

class ResidualNeuralField(nn.Module):
    """
    高级神经场模型
    Inputs:
        - Spatiotemporal: Lat, Lon, Alt, Time (Encoded)
        - Physics: P_phy, T_ref (Raw/Scaled)
        - Environment: Roughness (Raw/Scaled)
    """
    def __init__(self, st_dim=4, phys_dim=2, env_dim=1, hidden_dim=256, num_layers=6):
        super().__init__()

        # 1. 位置编码层 for Spatiotemporal coordinates
        self.pe = PositionalEncoding(input_dim=st_dim, L=6)
        st_encoded_dim = st_dim * (2 * 6 + 1)

        # Total input dimension = Encoded(ST) + Phys + Env
        total_input_dim = st_encoded_dim + phys_dim + env_dim

        layers = []
        # 2. 输入层
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.Tanh()) # Tanh or GELU/Swish often better for INRs

        # 3. 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # 4. 输出层 (输出 1 个标量：残差值)
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x_st, x_phys, x_env):
        # x_st: [batch, 4] (lat, lon, alt, time)
        # x_phys: [batch, 2] (p_phy, t_ref)
        # x_env: [batch, 1] (roughness)

        st_encoded = self.pe(x_st)

        # Concatenate all features
        x_combined = torch.cat([st_encoded, x_phys, x_env], dim=-1)

        return self.net(x_combined)

# ==========================================
# 2. Loss Function (Physics Guided)
# ==========================================

class PhysicsGuidedLoss(nn.Module):
    def __init__(self, lambda_smooth=0.01, lambda_phys=0.001):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_smooth = lambda_smooth
        self.lambda_phys = lambda_phys

    def forward(self, pred_residual, true_residual, inputs_st, model_output):
        # 1. Data Term
        data_loss = self.mse(pred_residual, true_residual)

        # 2. Physics/Constraint Term (Optional)
        # Example: Penalty for excessive residual magnitude (e.g., > 500m is unlikely)
        # In standardized space, let's say > 3 sigma
        phys_loss = torch.mean(torch.relu(torch.abs(pred_residual) - 3.0)**2)

        # 3. Smoothness Term (Gradient penalty w.r.t input space)
        # This is expensive to compute for every batch, usually done via autograd
        # For this POC, we skip explicit gradient computation or use a simplified weight regularization
        smooth_loss = 0.0

        total_loss = data_loss + self.lambda_phys * phys_loss + self.lambda_smooth * smooth_loss
        return total_loss, data_loss

# ==========================================
# 3. 数据准备 & 交叉验证
# ==========================================

class DroneDataset(Dataset):
    def __init__(self, df, scaler_st, scaler_phys, scaler_env, scaler_target):
        # Features
        st_cols = ['lat', 'lon', 'h_msl_pred_phy', 'timestamp_norm'] # Use physics predicted height as 'z' input
        phys_cols = ['h_msl_pred_phy', 't_ref_k'] # Features fed to network
        env_cols = ['roughness']

        self.X_st = torch.tensor(scaler_st.transform(df[st_cols].values), dtype=torch.float32).to(device)
        self.X_phys = torch.tensor(scaler_phys.transform(df[phys_cols].values), dtype=torch.float32).to(device)
        self.X_env = torch.tensor(scaler_env.transform(df[env_cols].values), dtype=torch.float32).to(device)

        # Target
        self.y = torch.tensor(scaler_target.transform(df['residual_hae'].values.reshape(-1, 1)), dtype=torch.float32).reshape(-1, 1).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_st[idx], self.X_phys[idx], self.X_env[idx], self.y[idx]

def run_training_pipeline():
    print("加载数据...")
    df = pd.read_csv('data_with_residual.csv')

    # Preprocessing Time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_norm'] = (df['timestamp'].astype(np.int64) // 10**9).astype(float)
    df['timestamp_norm'] = (df['timestamp_norm'] - df['timestamp_norm'].min()) / (df['timestamp_norm'].max() - df['timestamp_norm'].min() + 1e-5)

    # Scalers
    scaler_st = StandardScaler()
    scaler_phys = StandardScaler()
    scaler_env = StandardScaler()
    scaler_target = StandardScaler()

    st_cols = ['lat', 'lon', 'h_msl_pred_phy', 'timestamp_norm']
    phys_cols = ['h_msl_pred_phy', 't_ref_k']
    env_cols = ['roughness']

    scaler_st.fit(df[st_cols])
    scaler_phys.fit(df[phys_cols])
    scaler_env.fit(df[env_cols])
    scaler_target.fit(df['residual_hae'].values.reshape(-1, 1))

    # --- Spatial Cross-Validation ---
    # We use Grid Search CV concept: split domain into grids.
    # For simplicity here, we use K-Fold on the sorted spatial data or random K-Fold if dense enough.
    # The prompt asked for Spatial Grid Search CV.
    # Let's split by Lat/Lon grid.

    n_splits = 5
    # Create spatial bins
    df['lat_bin'] = pd.cut(df['lat'], bins=n_splits, labels=False)
    df['lon_bin'] = pd.cut(df['lon'], bins=n_splits, labels=False)
    df['grid_id'] = df['lat_bin'] * n_splits + df['lon_bin']

    unique_grids = df['grid_id'].unique()
    kf = KFold(n_splits=min(len(unique_grids), 5), shuffle=True, random_state=42)

    results = []

    for fold, (train_grid_idx, test_grid_idx) in enumerate(kf.split(unique_grids)):
        print(f"\n--- Fold {fold+1} ---")
        train_grids = unique_grids[train_grid_idx]
        test_grids = unique_grids[test_grid_idx]

        train_df = df[df['grid_id'].isin(train_grids)]
        test_df = df[df['grid_id'].isin(test_grids)]

        if len(train_df) == 0 or len(test_df) == 0:
            print("Fold empty, skipping.")
            continue

        train_dataset = DroneDataset(train_df, scaler_st, scaler_phys, scaler_env, scaler_target)
        test_dataset = DroneDataset(test_df, scaler_st, scaler_phys, scaler_env, scaler_target)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Model
        model = ResidualNeuralField(st_dim=4, phys_dim=2, env_dim=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsGuidedLoss()

        # Train
        for epoch in range(50): # 50 epochs for demo
            model.train()
            train_loss_sum = 0
            for x_st, x_phys, x_env, y in train_loader:
                optimizer.zero_grad()
                pred = model(x_st, x_phys, x_env)
                loss, data_loss = loss_fn(pred, y, x_st, model)
                loss.backward()
                optimizer.step()
                train_loss_sum += data_loss.item()

            if (epoch+1) % 25 == 0:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss_sum/len(train_loader):.4f}")

        # Evaluate
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x_st, x_phys, x_env, y in test_loader:
                pred = model(x_st, x_phys, x_env)
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        # Inverse transform
        preds_real = scaler_target.inverse_transform(preds)
        trues_real = scaler_target.inverse_transform(trues)

        rmse = np.sqrt(np.mean((preds_real - trues_real)**2))
        print(f"Fold {fold+1} RMSE: {rmse:.4f} m")
        results.append(rmse)

        # Visualization for the first fold
        if fold == 0:
            visualize_field(model, df, scaler_st, scaler_phys, scaler_env, scaler_target)

    print("\nMean CV RMSE:", np.mean(results))

def visualize_field(model, df, s_st, s_phys, s_env, s_target):
    print("生成残差场可视化...")
    # Generate Grid
    res = 50
    lats = np.linspace(df['lat'].min(), df['lat'].max(), res)
    lons = np.linspace(df['lon'].min(), df['lon'].max(), res)
    lat_mesh, lon_mesh = np.meshgrid(lats, lons)

    # Fixed other vars
    avg_h = df['h_msl_pred_phy'].mean()
    avg_t = df['t_ref_k'].mean()
    avg_rough = df['roughness'].mean()
    avg_time = 0.5 # Normalized time

    # Flatten
    n_points = res * res
    flat_lats = lat_mesh.flatten()
    flat_lons = lon_mesh.flatten()

    # Prepare Inputs
    # ST: lat, lon, h, t
    st_input = np.stack([flat_lats, flat_lons, np.full(n_points, avg_h), np.full(n_points, avg_time)], axis=1)
    phys_input = np.stack([np.full(n_points, avg_h), np.full(n_points, avg_t)], axis=1)
    env_input = np.stack([np.full(n_points, avg_rough)], axis=1)

    # Scale
    st_tensor = torch.tensor(s_st.transform(st_input), dtype=torch.float32).to(device)
    phys_tensor = torch.tensor(s_phys.transform(phys_input), dtype=torch.float32).to(device)
    env_tensor = torch.tensor(s_env.transform(env_input), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(st_tensor, phys_tensor, env_tensor).cpu().numpy()

    pred_real = s_target.inverse_transform(pred_scaled).reshape(res, res)

    plt.figure(figsize=(10, 8))
    plt.contourf(lon_mesh, lat_mesh, pred_real, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Predicted Residual (m)')
    plt.scatter(df['lon'], df['lat'], c=df['residual_hae'], cmap='RdBu_r', edgecolors='k', s=20)
    plt.title('Reconstructed Residual Field (Neural Field)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('neural_field_vis.png')
    print("可视化已保存到 neural_field_vis.png")

if __name__ == "__main__":
    run_training_pipeline()
