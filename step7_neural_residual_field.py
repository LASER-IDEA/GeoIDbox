import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 定义模型组件：位置编码 & MLP
# ==========================================

class PositionalEncoding(nn.Module):
    """
    位置编码层。将低维坐标映射到高维空间，帮助网络学习高频细节。
    类似于 NeRF 中的做法。
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
        # 输出维度 = input_dim * (2 * L + 1)
        return torch.cat(encoded, dim=-1)

class ResidualNeuralField(nn.Module):
    """
    核心神经场模型：一个简单的 MLP。
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=6):
        super().__init__()

        # 1. 位置编码层 (L=8 是一个经验值，可调)
        self.pe = PositionalEncoding(input_dim=input_dim, L=8)
        # 计算编码后的特征维度
        encoded_dim = input_dim * (2 * 8 + 1)

        layers = []
        # 2. 输入层
        layers.append(nn.Linear(encoded_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # 3. 隐藏层 (使用残差连接的思想可以加深网络，这里先用简单的MLP)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # 4. 输出层 (输出 1 个标量：残差值)
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 先进行位置编码，再通过 MLP
        x_encoded = self.pe(x)
        return self.net(x_encoded)

# ==========================================
# 2. 数据准备
# ==========================================

class DroneResidualDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 加载数据
print("加载数据...")
df = pd.read_csv('data_with_residual.csv')

# 选择特征 (Inputs)
# 关键：我们使用物理基准高度(h_hae_pred_phy)作为输入，
# 因为残差与当前的绝对高度高度相关。
feature_cols = ['lat', 'lon', 'h_hae_pred_phy']
# 如果数据跨度大，一定要加入时间特征 (需标准化到0-1之间)
# feature_cols.append('timestamp_normalized')

X_raw = df[feature_cols].values
y_raw = df['residual_hae'].values

# **至关重要**：数据标准化 (Standardization)
# 神经网络对输入范围非常敏感。经纬度、高度必须缩放到相似的范围。
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)
# 目标值也建议标准化，训练更稳定，预测时再反变换回来
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

train_dataset = DroneResidualDataset(X_train, y_train)
test_dataset = DroneResidualDataset(X_test, y_test)
# 批次大小可以根据显存调整
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ==========================================
# 3. 训练循环
# ==========================================

# 初始化模型
input_dim = X_scaled.shape[1] # 特征数量
model = ResidualNeuralField(input_dim=input_dim).to(device)
criterion = nn.MSELoss() # 均方误差损失
# 使用 Adam 优化器，学习率是一个关键超参数
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 1000 # 训练轮数
train_losses = []
test_losses = []

print(f"开始训练神经场 (Epochs: {num_epochs})...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # 验证循环
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)
    epoch_test_loss = running_test_loss / len(test_dataset)
    test_losses.append(epoch_test_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}")

print("训练完成。")

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss (MSE)')
plt.plot(test_losses, label='Test Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (Standardized scale)')
plt.legend()
plt.title('Neural Field Training Progress')
plt.show()

# ==========================================
# 4. 评估与场可视化 (关键!)
# ==========================================

model.eval()
with torch.no_grad():
    # --- 定量评估 (RMSE) ---
    # 获取测试集的预测结果 (标准化后的)
    y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    # 反标准化回真实单位（米）
    y_pred_residual = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true_residual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # 计算最终 RMSE
    rmse = np.sqrt(np.mean((y_true_residual - y_pred_residual)**2))
    print(f"\n神经场测试集 RMSE: {rmse:.3f} 米")

    # --- 定性评估：可视化重建的残差场 ---
    # 我们创建一个密集的网格来“查询”我们的神经场，看看它学到了什么结构
    print("正在生成残差场热力图...")

    # 1. 定义网格范围 (根据数据范围)
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    # 固定一个高度进行切片可视化，例如平均飞行高度
    avg_h_phy = df['h_hae_pred_phy'].mean()

    # 2. 生成密集网格点 (例如 100x100 的分辨率)
    resolution = 100
    lat_grid = np.linspace(lat_min, lat_max, resolution)
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # 3. 构建查询输入
    query_points = np.stack([
        lat_mesh.flatten(),
        lon_mesh.flatten(),
        np.full(resolution*resolution, avg_h_phy) # 所有点高度相同
    ], axis=1)

    # 4. 标准化查询输入 (非常重要! 必须用训练时的Scaler)
    query_points_scaled = scaler_X.transform(query_points)
    query_tensor = torch.tensor(query_points_scaled, dtype=torch.float32).to(device)

    # 5. 查询神经场
    predicted_residual_scaled = model(query_tensor).cpu().numpy()

    # 6. 反标准化结果
    predicted_residual_field = scaler_y.inverse_transform(predicted_residual_scaled)
    residual_grid = predicted_residual_field.reshape(resolution, resolution)

    # 7. 绘图
    plt.figure(figsize=(12, 10))
    # 绘制残差场热力图
    contour = plt.contourf(lon_mesh, lat_mesh, residual_grid, levels=50, cmap='RdBu_r', alpha=0.8)
    cbar = plt.colorbar(contour)
    cbar.set_label('Predicted Residual (m) - Red is positive, Blue is negative')

    # 叠加真实的无人机轨迹点，颜色表示真实残差
    scatter = plt.scatter(df['lon'], df['lat'], c=df['residual_hae'], cmap='RdBu_r',
                          edgecolor='k', s=20, vmin=residual_grid.min(), vmax=residual_grid.max())

    plt.title(f'Reconstructed Residual Field at H={avg_h_phy:.0f}m (Neural Field View)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

print("可视化完成。热力图展示了神经场学习到的空间结构。")
print("散点是真实的训练数据。如果神经场工作正常，热力图的颜色应该与散点颜色大致吻合，并且在散点之间平滑过渡。")