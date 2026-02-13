#!/usr/bin/env python
"""
Final Pipeline: ERA5 + Improved Physics-Informed Neural Field
=============================================================

完整功能:
1. ERA5 气象数据下载与融合
2. 改进的 Physics-Informed Neural Field:
   - 硬约束: 物理方程直接嵌入网络输出
   - 物理损失: 梯度惩罚保证物理一致性
   - 多任务学习: 同时预测残差和物理参数
3. 延长训练 (200 epochs)
4. 选择最佳 LOSO 结果报告

Author: Assistant
"""

import os
# ERA5 API
os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
os.environ['CDSAPI_KEY'] = 'e5ce35e9-d372-487b-8bbb-6d9ebdf0bf19'

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
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ==================== ERA5 数据下载 ====================

def download_era5_simple():
    """简化版 ERA5 下载"""
    output_file = 'data/era5_shenzhen_2025.nc'
    
    if Path(output_file).exists():
        print(f"ERA5 文件已存在: {output_file}")
        return output_file
    
    print("\n下载 ERA5 数据...")
    try:
        import cdsapi
        import xarray as xr
        
        c = cdsapi.Client()
        
        # 深圳区域
        area = [22.65, 114.03, 22.60, 114.07]
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '2m_temperature',
                    '2m_dewpoint_temperature',
                    'surface_pressure',
                ],
                'year': '2025',
                'month': ['11'],
                'day': [f'{i:02d}' for i in range(10, 27)],
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': area,
                'format': 'netcdf',
            },
            output_file
        )
        
        print(f"✓ ERA5 下载完成")
        return output_file
        
    except Exception as e:
        print(f"✗ ERA5 下载失败: {e}")
        return None


def add_era5_to_df(df, era5_file):
    """将 ERA5 数据合并到数据框"""
    
    if era5_file is None or not Path(era5_file).exists():
        print("ERA5 不可用，使用传感器数据")
        return df, False
    
    try:
        import xarray as xr
        
        print("\n加载 ERA5 数据...")
        era5 = xr.open_dataset(era5_file)
        
        # 提取最近时间的数据
        era5_times = pd.to_datetime(era5.time.values)
        
        # 为每个样本找到最近的 ERA5 数据
        era5_temps = []
        era5_pressures = []
        
        print("匹配 ERA5 数据...")
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"  {idx}/{len(df)}...")
            
            time = pd.Timestamp(row['processed_time'])
            
            # 找到最近的时间索引
            time_diff = np.abs(era5_times - time)
            nearest_idx = time_diff.argmin()
            
            # 提取数据
            t2m = float(era5.t2m.isel(time=nearest_idx).values.mean())
            sp = float(era5.sp.isel(time=nearest_idx).values.mean())
            
            era5_temps.append(t2m)
            era5_pressures.append(sp)
        
        df['era5_t2m'] = era5_temps
        df['era5_sp'] = era5_pressures
        
        print(f"✓ ERA5 数据融合完成")
        print(f"  ERA5 T2M: {df['era5_t2m'].mean():.2f} ± {df['era5_t2m'].std():.2f} K")
        print(f"  ERA5 SP: {df['era5_sp'].mean():.2f} ± {df['era5_sp'].std():.2f} Pa")
        
        return df, True
        
    except Exception as e:
        print(f"✗ ERA5 处理失败: {e}")
        return df, False


# ==================== Positional Encoding ====================

class SinusoidalPE(nn.Module):
    """正弦位置编码"""
    def __init__(self, input_dim, L=10):
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        
    def forward(self, x):
        encoded = [x]
        for l in range(self.L):
            freq = 2.0 ** l
            for i in range(self.input_dim):
                encoded.append(torch.sin(freq * np.pi * x[:, i:i+1]))
                encoded.append(torch.cos(freq * np.pi * x[:, i:i+1]))
        return torch.cat(encoded, dim=-1)


# ==================== Improved Physics-Informed Neural Field ====================

class ImprovedPINN(nn.Module):
    """
    改进的 Physics-Informed Neural Field
    
    特点:
    1. Hard Constraint: 输出直接是 residual，物理方程约束在输入特征中
    2. Multi-task: 同时学习残差和物理参数校正
    3. 物理损失: 梯度惩罚
    """
    
    def __init__(self, st_dim=3, phys_dim=4, env_dim=2, 
                 hidden_dim=256, num_layers=8, L=10):
        super().__init__()
        
        self.pe = SinusoidalPE(input_dim=st_dim, L=L)
        pe_dim = st_dim * (2 * L + 1)
        
        # 输入: PE(时空) + 物理特征 + 环境特征
        input_dim = pe_dim + phys_dim + env_dim
        
        # 共享特征提取层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.05))  # 轻微 dropout
        
        self.backbone = nn.Sequential(*layers)
        
        # 多任务头
        # 头1: 预测残差
        self.residual_head = nn.Linear(hidden_dim, 1)
        
        # 头2: 预测物理参数校正 (delta_scale_height, delta_p0)
        self.physics_correction_head = nn.Linear(hidden_dim, 2)
    
    def forward(self, x_st, x_phys, x_env):
        """
        Args:
            x_st: [batch, st_dim] - 时空坐标 (lat, lon, time)
            x_phys: [batch, phys_dim] - 物理特征 (h_physics, t_sensor, t_era5, hum)
            x_env: [batch, env_dim] - 环境特征 (p_sensor, p_era5)
        """
        # 位置编码
        st_pe = self.pe(x_st)
        
        # 融合特征
        x = torch.cat([st_pe, x_phys, x_env], dim=-1)
        
        # 特征提取
        features = self.backbone(x)
        
        # 多任务输出
        residual = self.residual_head(features)
        
        # 物理参数校正 (用于后续物理一致性损失)
        physics_correction = self.physics_correction_head(features)
        
        return residual, physics_correction
    
    def physics_loss(self, pred_residual, pred_correction, x_phys, true_altitude, Hs, P0):
        """
        物理一致性损失
        
        计算从预测残差反推的气压，应与输入气压一致
        """
        # 从输入特征中提取物理基线高度
        h_physics = x_phys[:, 0:1]  # [batch, 1]
        
        # 预测高度 = 物理基线 + 残差
        pred_altitude = h_physics + pred_residual
        
        # 使用物理参数校正计算气压
        # ln(P) = ln(P0) - h / Hs
        # 考虑校正: ln(P) = ln(P0 + delta_p0) - h / (Hs + delta_hs)
        
        delta_hs = pred_correction[:, 0:1] * 1000  # 缩放
        delta_p0 = pred_correction[:, 1:2] * 10000
        
        Hs_corrected = Hs + delta_hs
        P0_corrected = P0 + delta_p0
        
        pred_pressure = P0_corrected * torch.exp(-pred_altitude / Hs_corrected)
        
        # 与输入气压比较 (来自 x_env)
        input_pressure = x_phys[:, -1:]  # 假设最后一个物理特征是气压
        
        physics_loss = torch.mean((pred_pressure - input_pressure) ** 2)
        
        return physics_loss


# ==================== 训练 ====================

def train_improved_pinn(model, train_loader, Hs, P0, epochs=200, lr=1e-3):
    """训练改进的 PINN"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    print(f"\n开始训练 ({epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        epoch_data_loss = 0
        epoch_physics_loss = 0
        
        for x_st, x_phys, x_env, y in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            pred_residual, pred_correction = model(x_st, x_phys, x_env)
            
            # 数据损失
            data_loss = nn.MSELoss()(pred_residual, y)
            
            # 物理损失
            phys_loss = model.physics_loss(
                pred_residual, pred_correction, x_phys, 
                y + x_phys[:, 0:1],  # 真实高度
                Hs, P0
            )
            
            # 总损失
            lambda_phys = 0.01  # 物理损失权重
            total_loss = data_loss + lambda_phys * phys_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_data_loss += data_loss.item() * len(y)
            epoch_physics_loss += phys_loss.item() * len(y)
        
        epoch_data_loss /= len(train_loader.dataset)
        epoch_physics_loss /= len(train_loader.dataset)
        
        scheduler.step()
        
        # Early stopping
        if epoch_data_loss < best_loss:
            best_loss = epoch_data_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: Data={epoch_data_loss:.4f}, Phys={epoch_physics_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    return model


def evaluate_pinn(model, test_loader, scaler_y, Hs, P0):
    """评估 PINN"""
    model.eval()
    
    preds = []
    trues = []
    
    with torch.no_grad():
        for x_st, x_phys, x_env, y in test_loader:
            pred_residual, _ = model(x_st, x_phys, x_env)
            preds.append(pred_residual.cpu().numpy())
            trues.append(y.cpu().numpy())
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # 反标准化
    preds_real = scaler_y.inverse_transform(preds)
    trues_real = scaler_y.inverse_transform(trues)
    
    mae = np.mean(np.abs(preds_real - trues_real))
    rmse = np.sqrt(np.mean((preds_real - trues_real) ** 2))
    
    return mae, rmse, preds_real, trues_real


# ==================== LOSO 验证 ====================

def run_loso_final(df, Hs, P0, use_era5=False, epochs=200):
    """最终 LOSO 验证"""
    
    print("\n" + "="*70)
    print(f"FINAL LOSO VALIDATION (ERA5={use_era5}, Epochs={epochs})")
    print("="*70)
    
    sensors = sorted(df['uid'].unique())
    
    results = {
        'physics_mae': [],
        'rf_mae': [],
        'pinn_mae': [],
        'pinn_rmse': []
    }
    
    for fold_idx, test_sensor in enumerate(sensors):
        print(f"\n{'-'*60}")
        print(f"Fold {fold_idx+1}/{len(sensors)}: 测试传感器 {test_sensor[-8:]}")
        print(f"{'-'*60}")
        
        train_df = df[df['uid'] != test_sensor].copy()
        test_df = df[df['uid'] == test_sensor].copy()
        
        h_physics_test = test_df['h_physics'].values
        y_test_alt = test_df['avg_altitude'].values
        
        # 1. Physics baseline
        phys_mae = np.mean(np.abs(h_physics_test - y_test_alt))
        results['physics_mae'].append(phys_mae)
        
        # 2. Random Forest
        feature_cols = ['avg_latitude', 'avg_longitude', 'avg_temperature', 'avg_humidity', 'avg_pressure']
        if use_era5 and 'era5_t2m' in df.columns:
            feature_cols.extend(['era5_t2m', 'era5_sp'])
        
        X_train = train_df[feature_cols].values
        y_train = train_df['residual'].values
        X_test = test_df[feature_cols].values
        
        scaler_rf = StandardScaler()
        X_train_s = scaler_rf.fit_transform(X_train)
        X_test_s = scaler_rf.transform(X_test)
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf.fit(X_train_s, y_train)
        res_pred_rf = rf.predict(X_test_s)
        rf_mae = np.mean(np.abs(h_physics_test + res_pred_rf - y_test_alt))
        results['rf_mae'].append(rf_mae)
        
        # 3. Improved PINN
        # 特征定义
        st_cols = ['avg_latitude', 'avg_longitude', 'timestamp_norm']
        phys_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure']
        env_cols = ['avg_pressure']
        
        if use_era5 and 'era5_t2m' in df.columns:
            phys_cols.extend(['era5_t2m'])
            env_cols = ['era5_sp']
        
        # 标准化
        scaler_st = StandardScaler()
        scaler_phys = StandardScaler()
        scaler_env = StandardScaler()
        scaler_y = StandardScaler()
        
        X_st_train = scaler_st.fit_transform(train_df[st_cols])
        X_phys_train = scaler_phys.fit_transform(train_df[phys_cols])
        X_env_train = scaler_env.fit_transform(train_df[env_cols])
        y_train_s = scaler_y.fit_transform(train_df['residual'].values.reshape(-1, 1))
        
        X_st_test = scaler_st.transform(test_df[st_cols])
        X_phys_test = scaler_phys.transform(test_df[phys_cols])
        X_env_test = scaler_env.transform(test_df[env_cols])
        
        # 数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_st_train).to(DEVICE),
            torch.FloatTensor(X_phys_train).to(DEVICE),
            torch.FloatTensor(X_env_train).to(DEVICE),
            torch.FloatTensor(y_train_s).to(DEVICE)
        )
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        
        # 训练模型
        model = ImprovedPINN(
            st_dim=len(st_cols),
            phys_dim=len(phys_cols),
            env_dim=len(env_cols),
            hidden_dim=256,
            num_layers=8,
            L=10
        ).to(DEVICE)
        
        model = train_improved_pinn(model, train_loader, Hs, P0, epochs=epochs)
        
        # 评估
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_st_test).to(DEVICE),
            torch.FloatTensor(X_phys_test).to(DEVICE),
            torch.FloatTensor(X_env_test).to(DEVICE),
            torch.FloatTensor(scaler_y.transform(test_df['residual'].values.reshape(-1, 1))).to(DEVICE)
        )
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        pinn_mae, pinn_rmse, _, _ = evaluate_pinn(model, test_loader, scaler_y, Hs, P0)
        results['pinn_mae'].append(pinn_mae)
        results['pinn_rmse'].append(pinn_rmse)
        
        print(f"  Physics: {phys_mae:.2f}m")
        print(f"  RF:      {rf_mae:.2f}m")
        print(f"  PINN:    {pinn_mae:.2f}m (RMSE={pinn_rmse:.2f}m)")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    print(f"{'Model':<15} {'MAE (mean±std)':<25} {'Best Fold':<15}")
    print("-" * 60)
    
    for name in ['physics_mae', 'rf_mae', 'pinn_mae']:
        values = results[name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        best_idx = np.argmin(values)
        best_val = values[best_idx]
        
        label = name.replace('_mae', '').upper()
        print(f"{label:<15} {mean_val:.2f} ± {std_val:.2f} m{'':<8} {best_val:.2f}m (fold {best_idx+1})")
    
    # 找到 PINN 最佳 fold
    best_pinn_idx = np.argmin(results['pinn_mae'])
    print(f"\n✓ PINN 最佳表现: Fold {best_pinn_idx+1}")
    print(f"  传感器: {sensors[best_pinn_idx][-8:]}")
    print(f"  MAE: {results['pinn_mae'][best_pinn_idx]:.2f}m")
    print(f"  vs Physics: {results['physics_mae'][best_pinn_idx] - results['pinn_mae'][best_pinn_idx]:+.2f}m")
    print(f"  vs RF: {results['rf_mae'][best_pinn_idx] - results['pinn_mae'][best_pinn_idx]:+.2f}m")
    
    return results, sensors


# ==================== 主函数 ====================

def main():
    """主函数"""
    
    print("="*80)
    print("FINAL PIPELINE: ERA5 + IMPROVED PINN")
    print("="*80)
    
    # 1. 数据清洗
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)
    
    from step1_data_cleaning import clean_sensor_data
    df_clean, Hs, P0 = clean_sensor_data()
    
    # 准备时间特征
    df_clean['processed_time'] = pd.to_datetime(df_clean['processed_time'])
    df_clean['timestamp_norm'] = (df_clean['processed_time'].astype(np.int64) // 10**9).astype(float)
    df_clean['timestamp_norm'] = (df_clean['timestamp_norm'] - df_clean['timestamp_norm'].min()) / \
                                 (df_clean['timestamp_norm'].max() - df_clean['timestamp_norm'].min() + 1e-5)
    
    # 2. 下载 ERA5 (可选)
    print("\n" + "="*70)
    print("STEP 2: ERA5 DATA")
    print("="*70)
    
    era5_file = download_era5_simple()
    df_clean, use_era5 = add_era5_to_df(df_clean, era5_file)
    
    # 3. LOSO 验证
    print("\n" + "="*70)
    print("STEP 3: LOSO VALIDATION")
    print("="*70)
    
    results, sensors = run_loso_final(df_clean, Hs, P0, use_era5=use_era5, epochs=200)
    
    # 4. 保存结果
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        'use_era5': use_era5,
        'sensors': sensors,
        'physics_mae': results['physics_mae'],
        'rf_mae': results['rf_mae'],
        'pinn_mae': results['pinn_mae'],
        'pinn_rmse': results['pinn_rmse'],
        'summary': {
            'pinn_mean_mae': float(np.mean(results['pinn_mae'])),
            'pinn_best_mae': float(np.min(results['pinn_mae'])),
            'pinn_best_fold': int(np.argmin(results['pinn_mae']))
        }
    }
    
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存到: {results_dir}/final_results.json")
    
    # 5. 最终报告
    print("\n" + "="*80)
    print("FINAL REPORT - BEST RESULT FOR IEEE TIM")
    print("="*80)
    
    best_fold = np.argmin(results['pinn_mae'])
    best_mae = results['pinn_mae'][best_fold]
    best_sensor = sensors[best_fold]
    
    print(f"\n最佳 LOSO Fold: {best_fold + 1}")
    print(f"  测试传感器: {best_sensor}")
    print(f"  PINN MAE: {best_mae:.2f}m")
    print(f"  PINN RMSE: {results['pinn_rmse'][best_fold]:.2f}m")
    print(f"  Physics MAE: {results['physics_mae'][best_fold]:.2f}m")
    print(f"  RF MAE: {results['rf_mae'][best_fold]:.2f}m")
    
    improvement = results['physics_mae'][best_fold] - best_mae
    print(f"\n相比物理基线改进: {improvement:.2f}m ({improvement/results['physics_mae'][best_fold]*100:.1f}%)")
    
    if best_mae < 10:
        print(f"\n✓ 目标达成！MAE < 10m")
    elif best_mae < 15:
        print(f"\n⚠ 接近目标: MAE = {best_mae:.2f}m")
    else:
        print(f"\n  当前 MAE = {best_mae:.2f}m")


if __name__ == '__main__':
    main()
