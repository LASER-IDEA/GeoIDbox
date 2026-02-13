#!/usr/bin/env python
"""
Step 2: Download ERA5 Data
==========================

下载 ERA5 再分析数据作为气象特征：
- 温度 (t): 2米温度 (K)
- 湿度 (q): 2米比湿 (kg/kg)
- 地表气压 (sp): 地表气压 (Pa)

需要 CDS API Key，请设置环境变量：
export CDSAPI_URL="https://cds.climate.copernicus.eu/api/v2"
export CDSAPI_KEY="你的API-Key"

或在 ~/.cdsapirc 中配置
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import cdsapi
import xarray as xr
import warnings
warnings.filterwarnings('ignore')


def get_era5_for_location(lat, lon, time, era5_data):
    """
    从 ERA5 数据中提取特定位置和时间的数据
    
    Args:
        lat: 纬度
        lon: 经度  
        time: 时间
        era5_data: xarray Dataset
    
    Returns:
        dict with t, q, sp
    """
    # 找到最近的时间点
    era5_times = pd.to_datetime(era5_data.time.values)
    nearest_time_idx = np.argmin(np.abs(era5_times - pd.Timestamp(time)))
    nearest_time = era5_times[nearest_time_idx]
    
    # 空间插值 (最近邻)
    data_at_time = era5_data.isel(time=nearest_time_idx)
    
    # 找到最近的经纬度索引
    lat_idx = np.argmin(np.abs(data_at_time.latitude.values - lat))
    lon_idx = np.argmin(np.abs(data_at_time.longitude.values - lon))
    
    # 提取数据
    result = {
        't_ref_k': float(data_at_time.t.values[lat_idx, lon_idx]),
        'q_ref': float(data_at_time.q.values[lat_idx, lon_idx]),
        'sp_ref': float(data_at_time.sp.values[lat_idx, lon_idx]),
        'era5_time': nearest_time
    }
    
    return result


def download_era5_for_period(start_date, end_date, area, output_file):
    """
    下载指定时间段和区域的 ERA5 数据
    
    Args:
        start_date: 开始日期 '2025-11-10'
        end_date: 结束日期 '2025-11-26'
        area: [N, W, S, E] 例如 [22.65, 114.03, 22.60, 114.07]
        output_file: 输出文件名
    """
    
    print(f"下载 ERA5 数据:")
    print(f"  时间: {start_date} 至 {end_date}")
    print(f"  区域: {area}")
    
    try:
        c = cdsapi.Client()
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '2m_temperature',
                    '2m_dewpoint_temperature',  # 用于计算湿度
                    'surface_pressure',
                ],
                'year': start_date[:4],
                'month': start_date[5:7],
                'day': [f"{d:02d}" for d in range(int(start_date[8:10]), int(end_date[8:10]) + 1)],
                'time': [f"{h:02d}:00" for h in range(24)],
                'area': area,  # North, West, South, East
                'format': 'netcdf',
            },
            output_file
        )
        
        print(f"  下载完成: {output_file}")
        return True
        
    except Exception as e:
        print(f"  下载失败: {e}")
        print(f"  请检查 CDS API Key 是否正确配置")
        return False


def add_era5_features(df, era5_file='data/era5_shenzhen.nc'):
    """
    为数据框添加 ERA5 特征
    
    Args:
        df: 输入数据框
        era5_file: ERA5 NetCDF 文件路径
    
    Returns:
        添加了 ERA5 特征的 df
    """
    
    print(f"\n添加 ERA5 特征...")
    
    # 检查 ERA5 文件是否存在
    if not Path(era5_file).exists():
        print(f"  警告: ERA5 文件不存在: {era5_file}")
        print(f"  请先运行 download_era5_for_period() 下载数据")
        return df
    
    # 加载 ERA5 数据
    era5_data = xr.open_dataset(era5_file)
    
    # 提取每个样本的 ERA5 数据
    era5_features = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  处理 {idx}/{len(df)}...")
        
        try:
            features = get_era5_for_location(
                row['avg_latitude'], 
                row['avg_longitude'],
                row['processed_time'],
                era5_data
            )
            era5_features.append(features)
        except Exception as e:
            print(f"  错误 at idx {idx}: {e}")
            era5_features.append({'t_ref_k': np.nan, 'q_ref': np.nan, 'sp_ref': np.nan})
    
    # 添加到数据框
    era5_df = pd.DataFrame(era5_features)
    df = pd.concat([df.reset_index(drop=True), era5_df.reset_index(drop=True)], axis=1)
    
    print(f"  ERA5 特征添加完成")
    print(f"  有 ERA5 数据的样本: {df['t_ref_k'].notna().sum()}/{len(df)}")
    
    return df


def main():
    """主函数"""
    
    print("=" * 70)
    print("STEP 2: DOWNLOAD ERA5 DATA")
    print("=" * 70)
    
    # 加载清洗后的数据
    df = pd.read_csv('data/processed/sensor_data_cleaned.csv')
    df['processed_time'] = pd.to_datetime(df['processed_time'])
    
    # 确定下载范围
    start_date = df['processed_time'].min().strftime('%Y-%m-%d')
    end_date = df['processed_time'].max().strftime('%Y-%m-%d')
    
    # 区域 (稍微扩大以包含所有传感器)
    lat_min, lat_max = df['avg_latitude'].min(), df['avg_latitude'].max()
    lon_min, lon_max = df['avg_longitude'].min(), df['avg_longitude'].max()
    
    # 添加缓冲
    buffer = 0.01  # 约 1km
    area = [
        lat_max + buffer,  # North
        lon_min - buffer,  # West
        lat_min - buffer,  # South
        lon_max + buffer   # East
    ]
    
    era5_file = 'data/era5_shenzhen.nc'
    
    # 下载 ERA5 数据
    if not Path(era5_file).exists():
        print(f"\nERA5 文件不存在，开始下载...")
        success = download_era5_for_period(start_date, end_date, area, era5_file)
        if not success:
            print("\n下载失败。请手动下载或检查 API Key。")
            print("替代方案: 使用传感器自身的温湿度数据作为特征")
            return df
    else:
        print(f"\n使用已存在的 ERA5 文件: {era5_file}")
    
    # 添加 ERA5 特征
    df = add_era5_features(df, era5_file)
    
    # 保存
    output_file = 'data/processed/sensor_data_with_era5.csv'
    df.to_csv(output_file, index=False)
    print(f"\n数据已保存: {output_file}")
    
    return df


if __name__ == '__main__':
    main()
