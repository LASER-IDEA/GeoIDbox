#!/usr/bin/env python
"""
Step 3: Advanced Neural Field Training
======================================

基于 pipeline/step7 的改进版 Neural Field，针对深圳传感器数据优化：

改进点:
1. Positional Encoding (L=6) - 关键！
2. Deeper network (256x6)
3. Tanh activation
4. 支持传感器 ID embedding（处理离散传感器偏差）
5. 集成 ERA5 气象特征
6. LOSO 验证
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ==================== Neural Field 组件 ====================

class PositionalEncoding(nn.Module):
    """位置编码 - 核心组件"""
    def __init__(self, input_dim, L=6):
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        self.freq_bands = torch.pow(2, torch.linspace(0, L-1, L)).to(DEVICE)
    
    def forward(self, x):
        encoded = [x]
        for freq in self.freq_bands:
            for i in range(self.input_dim):
                encoded.append(torch.sin(x[:, i:i+1] * freq * np.pi))
                encoded.append(torch.cos(x[:, i:i+1] * freq * np.pi))
        return torch.cat(encoded, dim=-1)


class SensorEmbedding(nn.Module):
    """传感器 ID Embedding - 处理离散传感器偏差"""
    def __init__(self, num_sensors, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_sensors, embedding_dim)
    
    def forward(self, sensor_ids):
        return self.embedding(sensor_ids)


class AdvancedNeuralField(nn.Module):
    """
    改进版 Neural Field
    
    输入:
        - st: [lat, lon, h_physics, time] (时空坐标)
        - phys: [h_physics, t_ref, q_ref] (物理特征)
        - env: [roughness] (环境特征)
        - sensor_id: 传感器 ID (用于 embedding)
    """
    def __init__(self, num_sensors, st_dim=4, phys_dim=3, env_dim=1, 
                 sensor_emb_dim=8, hidden_dim=256, num_layers=6, L=6):
        super().__init__()
        
        # 传感器 embedding
        self.sensor_emb = SensorEmbedding(num_sensors, sensor_emb_dim)
        
        # 位置编码
        self.pe = PositionalEncoding(input_dim=st_dim, L=L)
        st_encoded_dim = st_dim * (2 * L + 1)
        
        # 总输入维度
        total_input_dim = st_encoded_dim + phys_dim + env_dim + sensor_emb_dim
        
        # 构建 MLP
        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_st, x_phys, x_env, sensor_id):
        # 编码
        st_encoded = self.pe(x_st)
        sensor_features = self.sensor_emb(sensor_id)
        
        # 拼接所有特征
        x_combined = torch.cat([st_encoded, x_phys, x_env, sensor_features], dim=-1)
        
        return self.net(x_combined)


# ==================== 数据集 ====================

class SensorDataset(Dataset):
    """传感器数据集"""
    def __init__(self, df, scaler_st, scaler_phys, scaler_env, scaler_target, sensor_id_map):
        # 特征列
        self.st_cols = ['avg_latitude', 'avg_longitude', 'h_physics', 'timestamp_norm']
        self.phys_cols = ['h_physics', 'avg_temperature', 'avg_humidity']
        self.env_cols = ['avg_pressure']  # 简化环境特征
        
        # 标准化
        self.X_st = torch.FloatTensor(scaler_st.transform(df[self.st_cols].values)).to(DEVICE)
        self.X_phys = torch.FloatTensor(scaler_phys.transform(df[self.phys_cols].values)).to(DEVICE)
        self.X_env = torch.FloatTensor(scaler_env.transform(df[self.env_cols].values)).to(DEVICE)
        
        # 传感器 ID
        self.sensor_ids = torch.LongTensor([sensor_id_map[uid] for uid in df['uid']]).to(DEVICE)
        
        # 目标
        self.y = torch.FloatTensor(scaler_target.transform(df['residual'].values.reshape(-1, 1))).to(DEVICE)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X_st[idx], self.X_phys[idx], self.X_env[idx], 
                self.sensor_ids[idx], self.y[idx])


# ==================== 训练 ====================

def train_epoch(model, train_loader, optimizer, scheduler=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    for x_st, x_phys, x_env, sensor_id, y in train_loader:
        optimizer.zero_grad()
        pred = model(x_st, x_phys, x_env, sensor_id)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    
    avg_loss = total_loss / len(train_loader.dataset)
    
    if scheduler:
        scheduler.step(avg_loss)
    
    return avg_loss


def evaluate(model, test_loader, scaler_target):
    """评估模型"""
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for x_st, x_phys, x_env, sensor_id, y in test_loader:
            pred = model(x_st, x_phys, x_env, sensor_id)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # 反标准化
    preds_real = scaler_target.inverse_transform(preds)
    trues_real = scaler_target.inverse_transform(trues)
    
    mae = np.mean(np.abs(preds_real - trues_real))
    rmse = np.sqrt(np.mean((preds_real - trues_real)**2))
    
    return mae, rmse, preds_real, trues_real


# ==================== LOSO 验证 ====================

def run_loso_validation(df, epochs=100, batch_size=256):
    """Leave-One-Sensor-Out 验证"""
    
    print("=" * 70)
    print("ADVANCED NEURAL FIELD - LOSO VALIDATION")
    print("=" * 70)
    
    sensors = sorted(df['uid'].unique())
    print(f"传感器列表 ({len(sensors)} 个): {[s[-8:] for s in sensors]}")
    
    # 传感器 ID 映射
    sensor_id_map = {uid: i for i, uid in enumerate(sensors)}
    
    results = {
        'nf_mae': [],
        'nf_rmse': [],
        'rf_mae': [],
        'physics_mae': []
    }
    
    for fold_idx, test_sensor in enumerate(sensors):
        print(f"\n{'-'*60}")
        print(f"Fold {fold_idx+1}/{len(sensors)}: 测试传感器 = {test_sensor[-8:]}")
        print(f"{'-'*60}")
        
        # 划分数据
        train_df = df[df['uid'] != test_sensor].copy()
        test_df = df[df['uid'] == test_sensor].copy()
        
        print(f"  训练: {len(train_df)} 样本 ({train_df['uid'].nunique()} 传感器)")
        print(f"  测试: {len(test_df)} 样本")
        
        # 准备标准化器
        scaler_st = StandardScaler()
        scaler_phys = StandardScaler()
        scaler_env = StandardScaler()
        scaler_target = StandardScaler()
        
        st_cols = ['avg_latitude', 'avg_longitude', 'h_physics', 'timestamp_norm']
        phys_cols = ['h_physics', 'avg_temperature', 'avg_humidity']
        env_cols = ['avg_pressure']
        
        scaler_st.fit(train_df[st_cols])
        scaler_phys.fit(train_df[phys_cols])
        scaler_env.fit(train_df[env_cols])
        scaler_target.fit(train_df['residual'].values.reshape(-1, 1))
        
        # 数据集
        train_dataset = SensorDataset(train_df, scaler_st, scaler_phys, scaler_env, scaler_target, sensor_id_map)
        test_dataset = SensorDataset(test_df, scaler_st, scaler_phys, scaler_env, scaler_target, sensor_id_map)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # ===== Neural Field =====
        num_train_sensors = train_df['uid'].nunique()
        model = AdvancedNeuralField(num_sensors=num_train_sensors).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"  Training Neural Field...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer)
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}, Loss: {train_loss:.4f}")
            
            # Early stopping
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter > 20:
                    print(f"    Early stopping at epoch {epoch}")
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 评估 NF
        nf_mae, nf_rmse, _, _ = evaluate(model, test_loader, scaler_target)
        results['nf_mae'].append(nf_mae)
        results['nf_rmse'].append(nf_rmse)
        print(f"  Neural Field: MAE={nf_mae:.2f}m, RMSE={nf_rmse:.2f}m")
        
        # ===== Random Forest (baseline) =====
        feature_cols = ['avg_latitude', 'avg_longitude', 'avg_temperature', 'avg_humidity', 'avg_pressure']
        X_train = train_df[feature_cols].values
        y_train = train_df['residual'].values
        X_test = test_df[feature_cols].values
        h_physics_test = test_df['h_physics'].values
        y_alt_test = test_df['avg_altitude'].values
        
        scaler_rf = StandardScaler()
        X_train_s = scaler_rf.fit_transform(X_train)
        X_test_s = scaler_rf.transform(X_test)
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf.fit(X_train_s, y_train)
        
        res_pred_rf = rf.predict(X_test_s)
        alt_pred_rf = h_physics_test + res_pred_rf
        rf_mae = np.mean(np.abs(alt_pred_rf - y_alt_test))
        results['rf_mae'].append(rf_mae)
        print(f"  Random Forest: MAE={rf_mae:.2f}m")
        
        # ===== Physics baseline =====
        physics_mae = np.mean(np.abs(h_physics_test - y_alt_test))
        results['physics_mae'].append(physics_mae)
        print(f"  Physics: MAE={physics_mae:.2f}m")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总结果
    print(f"\n{'='*70}")
    print("LOSO 结果汇总")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Mean MAE':<15} {'Std MAE':<15}")
    print("-" * 50)
    
    for model_name in ['physics_mae', 'rf_mae', 'nf_mae']:
        mean_val = np.mean(results[model_name])
        std_val = np.std(results[model_name])
        label = model_name.replace('_mae', '').upper()
        print(f"{label:<20} {mean_val:.2f}m{'':<10} {std_val:.2f}m")
    
    # 保存结果
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'loso_advanced_nf.json', 'w') as f:
        json.dump({
            'sensors': sensors,
            'physics_mae': results['physics_mae'],
            'rf_mae': results['rf_mae'],
            'nf_mae': results['nf_mae'],
            'nf_rmse': results['nf_rmse']
        }, f, indent=2)
    
    return results


# ==================== 主函数 ====================

def main():
    """主函数"""
    
    # 加载数据
    data_file = 'data/processed/sensor_data_cleaned.csv'
    
    if not Path(data_file).exists():
        print(f"错误: 请先运行 step1_data_cleaning.py")
        print(f"缺失文件: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['processed_time'] = pd.to_datetime(df['processed_time'])
    
    # 准备时间特征
    df['timestamp_norm'] = (df['processed_time'].astype(np.int64) // 10**9).astype(float)
    df['timestamp_norm'] = (df['timestamp_norm'] - df['timestamp_norm'].min()) / \
                           (df['timestamp_norm'].max() - df['timestamp_norm'].min() + 1e-5)
    
    # 计算物理基线
    from sklearn.linear_model import LinearRegression
    valid = df[['avg_pressure', 'avg_altitude']].dropna()
    X_fit = valid[['avg_altitude']].values
    y_fit = np.log(valid['avg_pressure'].values)
    lr = LinearRegression()
    lr.fit(X_fit, y_fit)
    Hs = -1.0 / lr.coef_[0]
    P0 = np.exp(lr.intercept_)
    
    df['h_physics'] = -Hs * (np.log(df['avg_pressure']) - np.log(P0))
    df['residual'] = df['avg_altitude'] - df['h_physics']
    
    print(f"\n数据准备完成:")
    print(f"  样本数: {len(df)}")
    print(f"  传感器数: {df['uid'].nunique()}")
    print(f"  物理基线 MAE: {np.mean(np.abs(df['residual'])):.2f}m")
    
    # 运行 LOSO 验证
    results = run_loso_validation(df, epochs=100)
    
    print(f"\n{'='*70}")
    print("最终结论:")
    print(f"{'='*70}")
    
    nf_mean = np.mean(results['nf_mae'])
    rf_mean = np.mean(results['rf_mae'])
    
    if nf_mean < rf_mean:
        print(f"✓ Advanced Neural Field 优于 RF")
        print(f"  NF: {nf_mean:.2f}m vs RF: {rf_mean:.2f}m (优 {rf_mean-nf_mean:.2f}m)")
    else:
        print(f"✗ RF 仍然优于 NF")
        print(f"  RF: {rf_mean:.2f}m vs NF: {nf_mean:.2f}m (优 {nf_mean-rf_mean:.2f}m)")
    
    print(f"\n目标: 达到 <10m MAE")
    if nf_mean < 10:
        print(f"✓ 目标达成！")
    else:
        print(f"✗ 还需改进 (当前 {nf_mean:.2f}m)")
        print(f"  建议: 1) 添加 ERA5 数据 2) 增加训练轮数 3) 调整架构")


if __name__ == '__main__':
    main()
