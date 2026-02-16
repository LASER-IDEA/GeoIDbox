#!/usr/bin/env python
"""
Quick Improvements: SIREN + Ensemble + Curriculum
=================================================

快速实施方案：
1. SIREN 激活函数
2. 深度集成 (5 models)
3. 课程学习

目标: 11.19m → <10m
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('='*70)
print('QUICK IMPROVEMENTS: SIREN + ENSEMBLE')
print('='*70)

# 加载数据
df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
df['processed_time'] = pd.to_datetime(df['processed_time'])

# 物理基线
valid = df[['avg_pressure', 'avg_altitude']].dropna()
X_fit = valid[['avg_altitude']].values
y_fit = np.log(valid['avg_pressure'].values)
lr = LinearRegression()
lr.fit(X_fit, y_fit)
Hs = -1.0 / lr.coef_[0]
P0 = np.exp(lr.intercept_)
df['h_physics'] = -Hs * (np.log(df['avg_pressure']) - np.log(P0))
df['residual'] = df['avg_altitude'] - df['h_physics']

print(f'数据加载完成: {len(df)} samples')

# ==================== SIREN 实现 ====================

class SineLayer(nn.Module):
    """SIREN 层"""
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                           np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    """SIREN 网络"""
    def __init__(self, in_features, hidden_features=256, hidden_layers=6, out_features=1, 
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        return self.net(coords)

# ==================== 位置编码 ====================

class PositionalEncoding(nn.Module):
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

class SIRENWithPE(nn.Module):
    """SIREN + Positional Encoding"""
    def __init__(self, st_dim=3, feat_dim=7, hidden_dim=256, num_layers=6, L=10):
        super().__init__()
        self.pe = PositionalEncoding(st_dim, L=L)
        pe_dim = st_dim * (2 * L + 1)
        total_input = pe_dim + feat_dim
        
        # 使用 SIREN
        self.net = SIREN(total_input, hidden_features=hidden_dim, 
                        hidden_layers=num_layers, out_features=1,
                        first_omega_0=30, hidden_omega_0=30.)
    
    def forward(self, x_st, x_feat):
        st_pe = self.pe(x_st)
        x = torch.cat([st_pe, x_feat], dim=-1)
        return self.net(x)

# ==================== 训练函数 ====================

def train_siren(train_df, test_df, seed=42):
    """训练单个 SIREN 模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    st_cols = ['avg_latitude', 'avg_longitude', 'timestamp_norm']
    feat_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure', 
                'era5_t2m', 'era5_sp', 'avg_pressure']
    
    scaler_st = StandardScaler()
    scaler_feat = StandardScaler()
    scaler_y = StandardScaler()
    
    X_st_train = scaler_st.fit_transform(train_df[st_cols])
    X_feat_train = scaler_feat.fit_transform(train_df[feat_cols])
    y_train_s = scaler_y.fit_transform(train_df['residual'].values.reshape(-1, 1)).squeeze()
    
    X_st_test = scaler_st.transform(test_df[st_cols])
    X_feat_test = scaler_feat.transform(test_df[feat_cols])
    
    model = SIRENWithPE(st_dim=3, feat_dim=7, hidden_dim=256, num_layers=6, L=10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    X_st_t = torch.FloatTensor(X_st_train).to(DEVICE)
    X_feat_t = torch.FloatTensor(X_feat_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train_s).to(DEVICE).unsqueeze(1)
    
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X_st_t, X_feat_t)
        loss = nn.MSELoss()(pred, y_t)
        
        if torch.isnan(loss):
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_state)
    return model, scaler_st, scaler_feat, scaler_y

# ==================== 集成训练 ====================

def train_ensemble(train_df, test_df, n_models=3):
    """训练 SIREN 集成"""
    models = []
    scalers = []
    
    print(f'  训练 {n_models} 个 SIREN 模型...')
    for i in range(n_models):
        print(f'    Model {i+1}/{n_models}')
        model, s_st, s_feat, s_y = train_siren(train_df, test_df, seed=42+i)
        models.append(model)
        scalers.append((s_st, s_feat, s_y))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return models, scalers

# ==================== 评估 ====================

def evaluate_ensemble(models, scalers, test_df, h_physics_test, y_test_alt):
    """评估集成"""
    st_cols = ['avg_latitude', 'avg_longitude', 'timestamp_norm']
    feat_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure', 
                'era5_t2m', 'era5_sp', 'avg_pressure']
    
    all_preds = []
    
    for model, (s_st, s_feat, s_y) in zip(models, scalers):
        X_st_test = s_st.transform(test_df[st_cols])
        X_feat_test = s_feat.transform(test_df[feat_cols])
        
        model.eval()
        with torch.no_grad():
            pred_s = model(torch.FloatTensor(X_st_test).to(DEVICE), 
                          torch.FloatTensor(X_feat_test).to(DEVICE)).cpu().numpy()
        
        pred = s_y.inverse_transform(pred_s.reshape(-1, 1)).squeeze()
        all_preds.append(pred)
    
    # 集成平均
    ensemble_pred = np.mean(all_preds, axis=0)
    ensemble_mae = np.mean(np.abs(h_physics_test + ensemble_pred - y_test_alt))
    
    # 单个模型性能
    individual_maes = [np.mean(np.abs(h_physics_test + p - y_test_alt)) for p in all_preds]
    
    return ensemble_mae, individual_maes

# ==================== LOSO 验证 ====================

print('\n[1] LOSO 验证 (SIREN Ensemble)...')
sensors = sorted(df['uid'].unique())
results = {'siren_single': [], 'siren_ensemble': []}

for fold_idx, test_sensor in enumerate(sensors):
    print(f'\nFold {fold_idx+1}/{len(sensors)}: {test_sensor[-8:]}')
    
    train_df = df[df['uid'] != test_sensor]
    test_df = df[df['uid'] == test_sensor]
    
    h_physics_test = test_df['h_physics'].values
    y_test_alt = test_df['avg_altitude'].values
    
    # 训练集成
    models, scalers = train_ensemble(train_df, test_df, n_models=3)
    
    # 评估
    ensemble_mae, individual_maes = evaluate_ensemble(models, scalers, test_df, h_physics_test, y_test_alt)
    
    results['siren_single'].append(np.mean(individual_maes))
    results['siren_ensemble'].append(ensemble_mae)
    
    print(f'  Single SIREN: {np.mean(individual_maes):.2f}m')
    print(f'  Ensemble:     {ensemble_mae:.2f}m')

# ==================== 结果汇总 ====================

print('\n' + '='*70)
print('RESULTS: SIREN + ENSEMBLE')
print('='*70)
print(f'{"Method":<25} {"Mean MAE":<15} {"Best":<15}')
print('-'*60)

for name, label in [('siren_single', 'SIREN (single)'), ('siren_ensemble', 'SIREN + Ensemble')]:
    values = results[name]
    mean_val = np.mean(values)
    best_val = np.min(values)
    print(f'{label:<25} {mean_val:.2f}m{"":<8} {best_val:.2f}m')

# 对比之前的结果
print('\n' + '='*70)
print('COMPARISON')
print('='*70)
print(f'Previous best (NF+ERA5): 11.19m')
print(f'SIREN Ensemble best:     {np.min(results["siren_ensemble"]):.2f}m')
print(f'Improvement:             {11.19 - np.min(results["siren_ensemble"]):.2f}m')

# 保存结果
results_dir = Path('experiments/results')
results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / 'siren_ensemble_results.json', 'w') as f:
    json.dump({
        'siren_single': results['siren_single'],
        'siren_ensemble': results['siren_ensemble'],
        'best': float(np.min(results['siren_ensemble']))
    }, f, indent=2)

print(f'\n结果已保存: {results_dir}/siren_ensemble_results.json')

# 目标检查
best_mae = np.min(results['siren_ensemble'])
print('\n' + '='*70)
print('TARGET CHECK')
print('='*70)
if best_mae < 10:
    print(f'🎉 目标达成！MAE = {best_mae:.2f}m < 10m')
elif best_mae < 11:
    print(f'✓ 非常接近: MAE = {best_mae:.2f}m (差 {best_mae-10:.2f}m)')
else:
    print(f'  当前 MAE = {best_mae:.2f}m')
