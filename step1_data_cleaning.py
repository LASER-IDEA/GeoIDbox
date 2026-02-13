#!/usr/bin/env python
"""
Step 1: Data Cleaning
=====================

清洗深圳传感器数据：
1. 识别并移除/修复被移动的传感器
2. 检测并处理异常值
3. 生成高质量训练数据

问题传感器: 20240606181851A641973A1878250224 (累积移动 122km)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def clean_sensor_data():
    """清洗传感器数据"""
    
    print("=" * 70)
    print("STEP 1: DATA CLEANING")
    print("=" * 70)
    
    # 加载原始数据
    df = pd.read_csv('data/processed/sensor_data_clean_stable.csv')
    df['processed_time'] = pd.to_datetime(df['processed_time'])
    
    original_count = len(df)
    original_sensors = df['uid'].nunique()
    
    print(f"\n原始数据:")
    print(f"  总样本: {original_count}")
    print(f"  传感器数: {original_sensors}")
    
    # ==================== 清洗策略 1: 移除移动传感器 ====================
    
    MOBILE_SENSOR = '20240606181851A641973A1878250224'
    
    print(f"\n[清洗 1] 移除移动传感器: {MOBILE_SENSOR[-8:]}")
    df_clean = df[df['uid'] != MOBILE_SENSOR].copy()
    removed_mobile = original_count - len(df_clean)
    print(f"  移除样本: {removed_mobile} ({removed_mobile/original_count*100:.1f}%)")
    
    # ==================== 清洗策略 2: 移除异常气压值 ====================
    
    print(f"\n[清洗 2] 移除异常气压值")
    pressure_mask = (df_clean['avg_pressure'] > 95000) & (df_clean['avg_pressure'] < 105000)
    removed_pressure = (~pressure_mask).sum()
    df_clean = df_clean[pressure_mask].copy()
    print(f"  移除样本: {removed_pressure}")
    
    # ==================== 清洗策略 3: 移除异常高度值 ====================
    
    print(f"\n[清洗 3] 移除异常高度值 (负值或 >500m)")
    altitude_mask = (df_clean['avg_altitude'] > 0) & (df_clean['avg_altitude'] < 500)
    removed_altitude = (~altitude_mask).sum()
    df_clean = df_clean[altitude_mask].copy()
    print(f"  移除样本: {removed_altitude}")
    
    # ==================== 清洗策略 4: 基于物理一致性检测 ====================
    
    print(f"\n[清洗 4] 基于物理一致性检测")
    
    # 计算物理基线
    from sklearn.linear_model import LinearRegression
    valid = df_clean[['avg_pressure', 'avg_altitude']].dropna()
    X_fit = valid[['avg_altitude']].values
    y_fit = np.log(valid['avg_pressure'].values)
    lr = LinearRegression()
    lr.fit(X_fit, y_fit)
    
    Hs = -1.0 / lr.coef_[0]
    P0 = np.exp(lr.intercept_)
    
    df_clean['h_physics'] = -Hs * (np.log(df_clean['avg_pressure']) - np.log(P0))
    df_clean['residual'] = df_clean['avg_altitude'] - df_clean['h_physics']
    
    # 标记异常残差 (>3 sigma)
    residual_mean = df_clean['residual'].mean()
    residual_std = df_clean['residual'].std()
    residual_threshold = 3 * residual_std
    
    normal_mask = np.abs(df_clean['residual']) < residual_threshold
    removed_outliers = (~normal_mask).sum()
    df_clean = df_clean[normal_mask].copy()
    
    print(f"  残差阈值 (3σ): ±{residual_threshold:.1f}m")
    print(f"  移除异常样本: {removed_outliers}")
    
    # ==================== 清洗后统计 ====================
    
    print(f"\n{'='*70}")
    print("清洗后数据:")
    print(f"{'='*70}")
    print(f"  剩余样本: {len(df_clean)} ({len(df_clean)/original_count*100:.1f}%)")
    print(f"  剩余传感器: {df_clean['uid'].nunique()}")
    print(f"  传感器列表: {[uid[-8:] for uid in sorted(df_clean['uid'].unique())]}")
    
    # 各传感器统计
    print(f"\n各传感器统计:")
    for uid in sorted(df_clean['uid'].unique()):
        s_df = df_clean[df_clean['uid'] == uid]
        print(f"  {uid[-8:]}: {len(s_df)} samples, "
              f"alt={s_df['avg_altitude'].mean():.1f}±{s_df['avg_altitude'].std():.1f}m, "
              f"residual={s_df['residual'].mean():.1f}±{s_df['residual'].std():.1f}m")
    
    # 保存清洗后数据
    output_path = 'data/processed/sensor_data_cleaned.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\n清洗后数据已保存: {output_path}")
    
    return df_clean, Hs, P0


if __name__ == '__main__':
    df_clean, Hs, P0 = clean_sensor_data()
