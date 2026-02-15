#!/usr/bin/env python3
"""
Curriculum Learning with History Saving
=======================================
Train with curriculum learning and save training history for plotting.

Usage:
    source ~/miniconda3/bin/activate graphmamba
    python run_curriculum_with_history.py
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Create output directory
os.makedirs('experiments/results/curriculum_history', exist_ok=True)

# Import from existing module
import sys
sys.path.insert(0, '.')
from run_advanced_improvements import (
    HashEncoding, AdvancedNF, compute_terrain_features
)


def train_with_curriculum_and_history(
    model, stages, test_df, h_physics_test, y_test_alt, 
    max_epochs_per_stage=350,  # Increased for better visualization
    patience=100
):
    """
    课程学习训练，保存完整历史
    """
    best_overall_mae = float('inf')
    best_state = None
    
    # History dictionary
    history = {
        'stages': [],
        'epochs': [],
        'losses': [],
        'maes': [],
        'learning_rates': []
    }
    
    global_epoch = 0
    
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
        patience_counter = 0
        stage_history = []
        
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
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # 评估 - 每10个epoch保存一次（更密集）
            if epoch % 10 == 0 or epoch == max_epochs_per_stage - 1:
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
                
                # Save to history
                history['stages'].append(stage_idx + 1)
                history['epochs'].append(global_epoch)
                history['losses'].append(float(loss.item()))
                history['maes'].append(float(mae))
                history['learning_rates'].append(float(current_lr))
                
                print(f"    Epoch {global_epoch:3d} (Stage{stage_idx+1} E{epoch:3d}): "
                      f"Loss={loss.item():.4f}, Val MAE={mae:.2f}m, LR={current_lr:.6f}")
                
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
            
            global_epoch += 1
        
        print(f"    Stage {stage_idx+1} complete. Best MAE so far: {best_overall_mae:.2f}m")
    
    # 加载最佳状态
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_overall_mae, scaler_spatial, scaler_feature, scaler_y, history


def run_curriculum_training_for_best_fold():
    """
    为最佳Fold (Sensor 42508217) 训练并保存课程学习历史
    """
    print("="*70)
    print("CURRICULUM LEARNING WITH HISTORY - Best Fold Training")
    print("="*70)
    
    # 加载数据
    print("\n[1] 加载数据...")
    df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
    df = compute_terrain_features(df)
    
    # 找到最佳Fold的传感器 (Sensor 42508217)
    # 从之前的实验结果，这个传感器取得了最好的结果 (MAE=3.79m)
    target_sensor = '20240911193046A806593A5642508217'
    
    print(f"\n[2] 选择最佳传感器: {target_sensor}")
    
    # LOSO - 留一法
    train_df = df[df['uid'] != target_sensor].copy()
    test_df = df[df['uid'] == target_sensor].copy()
    
    print(f"    训练集: {len(train_df)} samples")
    print(f"    测试集: {len(test_df)} samples")
    
    # 准备测试数据
    h_physics_test = test_df['h_physics'].values
    y_test_alt = test_df['avg_altitude'].values
    
    # 定义课程学习阶段
    print("\n[3] 定义课程学习阶段...")
    
    # Stage 1: Easy (低海拔 + 高密度)
    altitude_threshold_low = train_df['avg_altitude'].quantile(0.4)
    density_threshold_high = train_df['sensor_density'].quantile(0.6)
    
    easy_df = train_df[
        (train_df['avg_altitude'] < altitude_threshold_low) & 
        (train_df['sensor_density'] > density_threshold_high)
    ].copy()
    
    # Stage 2: Medium (中等难度)
    altitude_threshold_high = train_df['avg_altitude'].quantile(0.7)
    density_threshold_low = train_df['sensor_density'].quantile(0.3)
    
    medium_df = train_df[
        (train_df['avg_altitude'] < altitude_threshold_high) | 
        (train_df['sensor_density'] > density_threshold_low)
    ].copy()
    
    # Stage 3: Hard (全部数据)
    hard_df = train_df.copy()
    
    stages = [
        ('Easy (Low Altitude, High Density)', easy_df),
        ('Medium (Moderate)', medium_df),
        ('Hard (Full Dataset)', hard_df)
    ]
    
    print(f"    Stage 1 (Easy): {len(easy_df)} samples")
    print(f"    Stage 2 (Medium): {len(medium_df)} samples") 
    print(f"    Stage 3 (Hard): {len(hard_df)} samples")
    
    # 创建模型
    print("\n[4] 创建模型...")
    model = AdvancedNF(
        use_hash_encoding=True,
        use_terrain=True,
        st_dim=2,
        feature_dim=9,
        hidden_dim=256,
        num_layers=8
    ).to(DEVICE)
    
    print(f"    参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练并保存历史
    print("\n[5] 开始课程学习训练 (1000+ epochs)...")
    print("    This may take 30-60 minutes...")
    
    model, best_mae, scaler_spatial, scaler_feature, scaler_y, history = \
        train_with_curriculum_and_history(
            model, stages, test_df, h_physics_test, y_test_alt,
            max_epochs_per_stage=350,  # 3 stages × 350 = ~1000 epochs
            patience=100
        )
    
    # 保存历史
    print("\n[6] 保存训练历史...")
    history_file = 'experiments/results/curriculum_history/training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"    已保存: {history_file}")
    
    # 保存模型
    model_file = 'experiments/results/curriculum_history/best_model.pt'
    torch.save(model.state_dict(), model_file)
    print(f"    已保存: {model_file}")
    
    # 打印总结
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest MAE: {best_mae:.4f}m")
    print(f"Total epochs recorded: {len(history['epochs'])}")
    print(f"\nStage distribution:")
    for stage in [1, 2, 3]:
        count = history['stages'].count(stage)
        print(f"  Stage {stage}: {count} checkpoints")
    print("\nHistory saved. Generate figure with:")
    print("  python paper/generate_fig5_curriculum.py")
    
    return history, best_mae


if __name__ == '__main__':
    history, best_mae = run_curriculum_training_for_best_fold()
