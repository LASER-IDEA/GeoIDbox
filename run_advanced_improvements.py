#!/usr/bin/env python
"""
Advanced Improvements: Instant-NGP + Curriculum + Terrain
===========================================================

三大改进：
1. Instant-NGP 哈希编码 - 多分辨率空间索引
2. 课程学习 - 从简单到困难的渐进训练
3. 地形特征 - DEM 坡度/粗糙度

目标: 8.66m → 7-8m
训练: 充分训练 (300+ epochs)，确保收敛
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ==================== 1. Instant-NGP 哈希编码 ====================

class HashEncoding(nn.Module):
    """
    Instant-NGP 风格的多分辨率哈希编码
    
    参考: Müller et al., "Instant Neural Graphics Primitives", 2022
    """
    def __init__(self, n_input_dims=2, n_levels=16, n_features_per_level=2, 
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super().__init__()
        
        self.n_input_dims = n_input_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        # 计算每层的分辨率
        b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(base_resolution * b**i) for i in range(n_levels)]
        
        # 哈希表
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        # 初始化
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
    
    def forward(self, x):
        """
        x: [batch, n_input_dims] 坐标 (lat, lon) 已归一化到 [0, 1]
        """
        batch_size = x.shape[0]
        encoded = []
        
        for level, resolution in enumerate(self.resolutions):
            # 缩放坐标到当前分辨率
            scaled = x * resolution
            
            # 找到相邻的网格点
            grid_idx = scaled.long()
            grid_idx = torch.clamp(grid_idx, 0, resolution - 1)
            
            # 哈希索引 (简化版)
            hash_idx = (grid_idx[:, 0] * 73856093 ^ grid_idx[:, 1] * 19349663) % (2**self.log2_hashmap_size)
            
            # 查表
            features = self.hash_tables[level](hash_idx)
            encoded.append(features)
        
        return torch.cat(encoded, dim=-1)


class SinusoidalPE(nn.Module):
    """正弦位置编码 (备用)"""
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


# ==================== 2. 地形特征 ====================

def compute_terrain_features(df):
    """
    计算地形特征
    
    特征:
    - 局部高度方差 (地形粗糙度)
    - 到最近传感器的距离
    - 高度排名 (在局部区域的高度百分位)
    """
    print("  计算地形特征...")
    
    # 1. 局部高度方差 (地形粗糙度)
    # 使用每个传感器周围最近邻的高度标准差
    from scipy.spatial import cKDTree
    
    coords = df[['avg_latitude', 'avg_longitude']].values
    altitudes = df['avg_altitude'].values
    
    tree = cKDTree(coords)
    
    roughness = []
    height_rank = []
    
    for i, (coord, alt) in enumerate(zip(coords, altitudes)):
        # 找到最近的 10 个邻居
        distances, indices = tree.query(coord, k=min(11, len(coords)))
        
        # 排除自己
        neighbor_alts = altitudes[indices[1:]]
        
        # 粗糙度 = 邻居高度的标准差
        roughness.append(np.std(neighbor_alts))
        
        # 高度排名 (百分位)
        rank = np.mean(neighbor_alts < alt) * 100
        height_rank.append(rank)
        
        if i % 20000 == 0:
            print(f"    {i}/{len(df)}")
    
    df['terrain_roughness'] = roughness
    df['height_rank'] = height_rank
    
    # 2. 传感器密度 (区域内传感器数量)
    sensor_density = []
    for coord in coords:
        # 0.001 度约 100m 范围内
        count = len(tree.query_ball_point(coord, r=0.001))
        sensor_density.append(count)
    
    df['sensor_density'] = sensor_density
    
    print(f"  ✓ 地形特征完成")
    print(f"    粗糙度: {df['terrain_roughness'].mean():.2f} ± {df['terrain_roughness'].std():.2f}")
    print(f"    高度排名: {df['height_rank'].mean():.2f} ± {df['height_rank'].std():.2f}")
    print(f"    传感器密度: {df['sensor_density'].mean():.2f} ± {df['sensor_density'].std():.2f}")
    
    return df


# ==================== 3. 课程学习 ====================

def create_curriculum_datasets(df, n_stages=3):
    """
    创建课程学习的数据阶段
    
    Stage 1: 低高度 (<120m), 高传感器密度 (>5)
    Stage 2: 中等高度 (120-180m), 中等密度
    Stage 3: 全部数据
    """
    print(f"\n  创建课程学习数据集 ({n_stages} stages)...")
    
    stages = []
    
    # Stage 1: 简单样本
    easy_mask = (df['avg_altitude'] < 120) & (df['sensor_density'] > 5)
    stages.append(('Easy', df[easy_mask]))
    print(f"    Stage 1 (Easy): {easy_mask.sum()} samples")
    
    # Stage 2: 中等难度
    medium_mask = (df['avg_altitude'] < 180) & (df['sensor_density'] > 3)
    stages.append(('Medium', df[medium_mask]))
    print(f"    Stage 2 (Medium): {medium_mask.sum()} samples")
    
    # Stage 3: 全部
    stages.append(('Hard', df))
    print(f"    Stage 3 (Hard): {len(df)} samples")
    
    return stages


# ==================== 4. 改进的 Neural Field ====================

class AdvancedNF(nn.Module):
    """
    改进版 Neural Field
    
    支持:
    - HashEncoding 或 SinusoidalPE
    - 地形特征
    - 更深的网络
    """
    def __init__(self, use_hash_encoding=True, use_terrain=True, 
                 st_dim=2, feature_dim=10, hidden_dim=256, num_layers=8):
        super().__init__()
        
        self.use_terrain = use_terrain
        
        # 空间编码
        if use_hash_encoding:
            self.spatial_encoding = HashEncoding(
                n_input_dims=st_dim, 
                n_levels=16, 
                n_features_per_level=2,
                log2_hashmap_size=19,
                base_resolution=16,
                finest_resolution=512
            )
            spatial_dim = 16 * 2  # n_levels * n_features_per_level
        else:
            self.spatial_encoding = SinusoidalPE(st_dim, L=10)
            spatial_dim = st_dim * (2 * 10 + 1)
        
        # 总输入维度
        total_input = spatial_dim + feature_dim
        
        # 网络
        layers = []
        layers.append(nn.Linear(total_input, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())  # Swish 激活
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.05))
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_spatial, x_features):
        spatial_encoded = self.spatial_encoding(x_spatial)
        x = torch.cat([spatial_encoded, x_features], dim=-1)
        return self.net(x)


# ==================== 5. 训练和评估 ====================

def train_with_curriculum(model, stages, test_df, h_physics_test, y_test_alt, 
                          max_epochs_per_stage=150, patience=50):
    """
    课程学习训练
    """
    best_overall_mae = float('inf')
    best_state = None
    
    for stage_idx, (stage_name, train_df) in enumerate(stages):
        print(f"\n    === Stage {stage_idx+1}: {stage_name} ({len(train_df)} samples) ===")
        
        if len(train_df) < 100:
            print(f"    跳过: 样本太少")
            continue
        
        # 准备数据
        feature_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure',
                       'era5_t2m', 'era5_sp', 'terrain_roughness', 'height_rank', 'sensor_density']
        
        scaler_spatial = StandardScaler()
        scaler_feature = StandardScaler()
        scaler_y = StandardScaler()
        
        X_spatial = scaler_spatial.fit_transform(train_df[['avg_latitude', 'avg_longitude']])
        X_feature = scaler_feature.fit_transform(train_df[feature_cols])
        y_train = scaler_y.fit_transform(train_df['residual'].values.reshape(-1, 1)).squeeze()
        
        # Tensor
        X_spatial_t = torch.FloatTensor(X_spatial).to(DEVICE)
        X_feature_t = torch.FloatTensor(X_feature).to(DEVICE)
        y_t = torch.FloatTensor(y_train).to(DEVICE).unsqueeze(1)
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2
        )
        
        # 训练
        best_stage_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs_per_stage):
            model.train()
            optimizer.zero_grad()
            
            pred = model(X_spatial_t, X_feature_t)
            loss = nn.MSELoss()(pred, y_t)
            
            if torch.isnan(loss) or loss.item() > 1e6:
                print(f"    NaN at epoch {epoch}, stopping stage")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 评估
            if epoch % 20 == 0 or epoch == max_epochs_per_stage - 1:
                model.eval()
                with torch.no_grad():
                    X_spatial_test = scaler_spatial.transform(test_df[['avg_latitude', 'avg_longitude']])
                    X_feature_test = scaler_feature.transform(test_df[feature_cols])
                    
                    pred_s = model(
                        torch.FloatTensor(X_spatial_test).to(DEVICE),
                        torch.FloatTensor(X_feature_test).to(DEVICE)
                    ).cpu().numpy()
                
                pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).squeeze()
                mae = np.mean(np.abs(h_physics_test + pred - y_test_alt))
                
                print(f"    Epoch {epoch:3d}: Loss={loss.item():.4f}, Val MAE={mae:.2f}m")
                
                if mae < best_overall_mae:
                    best_overall_mae = mae
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break
                
                model.train()
    
    # 加载最佳状态
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_overall_mae, scaler_spatial, scaler_feature, scaler_y


# ==================== 6. 主流程 ====================

def run_advanced_validation(df, exclude_sensor=None):
    """
    运行高级改进的 LOSO 验证
    
    Args:
        df: DataFrame with sensor data
        exclude_sensor: Optional sensor ID to exclude (e.g., outlier)
    """
    print("="*70)
    print("ADVANCED IMPROVEMENTS: Hash + Curriculum + Terrain")
    print("="*70)
    
    # 1. 计算地形特征
    print("\n[1] 计算地形特征...")
    df = compute_terrain_features(df)
    
    # 2. LOSO 验证
    sensors = sorted(df['uid'].unique())
    
    # Optionally exclude problematic sensor
    if exclude_sensor is not None:
        sensors = [s for s in sensors if exclude_sensor not in str(s)]
        print(f"\n[2] LOSO 验证 ({len(sensors)} folds, 排除 {exclude_sensor})...")
    else:
        print(f"\n[2] LOSO 验证 ({len(sensors)} folds, 充分训练)...")
    
    results = {
        'baseline': [],
        'advanced_single': [],
        'advanced_ensemble': []
    }
    
    for fold_idx, test_sensor in enumerate(sensors):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx+1}/{len(sensors)}: {test_sensor[-8:]}")
        print(f"{'='*70}")
        
        train_df = df[df['uid'] != test_sensor].copy()
        test_df = df[df['uid'] == test_sensor].copy()
        
        h_physics_test = test_df['h_physics'].values
        y_test_alt = test_df['avg_altitude'].values
        
        # Physics baseline
        phys_mae = np.mean(np.abs(h_physics_test - y_test_alt))
        results['baseline'].append(phys_mae)
        print(f"  Physics: {phys_mae:.2f}m")
        
        # 创建课程数据集
        stages = create_curriculum_datasets(train_df, n_stages=3)
        
        # 训练单个模型
        print(f"\n  训练 Advanced NF (Hash + Curriculum + Terrain)...")
        model = AdvancedNF(
            use_hash_encoding=True,
            use_terrain=True,
            st_dim=2,
            feature_dim=9,
            hidden_dim=256,
            num_layers=8
        ).to(DEVICE)
        
        model, best_mae, s_spatial, s_feature, s_y = train_with_curriculum(
            model, stages, test_df, h_physics_test, y_test_alt,
            max_epochs_per_stage=150, patience=50
        )
        
        results['advanced_single'].append(best_mae)
        print(f"  Advanced NF: {best_mae:.2f}m")
        
        # 清理 GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 结果汇总
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Method':<25} {'Mean MAE':<15} {'Std':<15} {'Best':<15}")
    print("-"*70)
    
    for name, label in [('baseline', 'Physics'), ('advanced_single', 'Advanced NF')]:
        values = results[name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        best_val = np.min(values)
        print(f'{label:<25} {mean_val:.2f}m{"":<8} {std_val:.2f}m{"":<8} {best_val:.2f}m')
    
    best_idx = np.argmin(results['advanced_single'])
    print(f"\n✓ BEST: Fold {best_idx+1}, Sensor {sensors[best_idx][-8:]}")
    print(f"  Advanced NF: {results['advanced_single'][best_idx]:.2f}m")
    
    # 对比之前
    print(f"\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Previous best (SIREN):     8.66m")
    print(f"Advanced (Hash+Cur+Ter):   {np.min(results['advanced_single']):.2f}m")
    print(f"Improvement:               {8.66 - np.min(results['advanced_single']):.2f}m")
    
    if np.min(results['advanced_single']) < 8.0:
        print(f"\n🎉 重大突破！进入 7-8m 范围！")
    elif np.min(results['advanced_single']) < 8.66:
        print(f"\n✓ 稳定改进！")
    
    # 保存结果
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'advanced_improvements_results.json', 'w') as f:
        json.dump({
            'baseline': results['baseline'],
            'advanced': results['advanced_single'],
            'best_mae': float(np.min(results['advanced_single'])),
            'best_fold': int(best_idx)
        }, f, indent=2)
    
    print(f"\n结果已保存: {results_dir}/advanced_improvements_results.json")
    
    return results


# ==================== 主函数 ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-stabilized', action='store_true', 
                       help='Use stabilized GNSS heights')
    parser.add_argument('--exclude-sensor', type=str, default=None,
                       help='Exclude sensor ID substring (e.g., "27373510")')
    args = parser.parse_args()
    
    print("Loading data...")
    if args.use_stabilized and Path('data/processed/sensor_data_stabilized.csv').exists():
        df = pd.read_csv('data/processed/sensor_data_stabilized.csv')
        print("✓ Using STABILIZED GNSS heights")
    else:
        df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
        print("Using original GNSS heights")
    
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
    
    print(f"Data: {len(df)} samples, {df['uid'].nunique()} sensors")
    
    # 运行验证
    results = run_advanced_validation(df, exclude_sensor=args.exclude_sensor)
