import cdsapi
import os
import xarray as xr
import numpy as np
import pandas as pd

# --- 配置区域 ---
# 假设无人机在深圳飞行
TARGET_DATE = '2024-11-24'
TARGET_TIME = '15:00' # UTC时间，取整点
# 深圳周边范围
# The drone data is at: 22.908927, 120.3140614
# The previous AREA was: [23.0, 113.8, 22.4, 114.2] (Near Shenzhen, but lon is 114)
# The drone seems to be in Taiwan (lon 120) or somewhere else.
# Correcting AREA to cover the drone location.
# Drone: 22.91, 120.31
AREA = [23.2, 120.0, 22.6, 120.6] # North, West, South, East

def download_era5_data():
    """
    下载指定时间和区域的ERA5气压层数据和地面数据。
    """
    c = cdsapi.Client()

    # 1. 下载气压层数据 (Pressure Levels)
    # 用于构建垂直剖面
    output_pl = f'data/era5_pl_{TARGET_DATE}.nc'
    if not os.path.exists(output_pl):
        print(f"开始下载 ERA5 气压层数据到 {output_pl} ...")
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'geopotential', 'specific_humidity', 'temperature',
                    'u_component_of_wind', 'v_component_of_wind',
                    'vertical_velocity'
                ],
                'pressure_level': [
                    '50', '100', '150', '200', '250',
                    '300', '400', '500', '600', '700',
                    '800', '850', '900', '925', '950',
                    '975', '1000',
                ],
                'year': TARGET_DATE.split('-')[0],
                'month': TARGET_DATE.split('-')[1],
                'day': TARGET_DATE.split('-')[2],
                'time': TARGET_TIME,
                'area': AREA,
            },
            output_pl)
    else:
        print(f"{output_pl} 已存在，跳过下载。")

    # 2. 下载地面数据 (Single Levels)
    # 用于获取高精度的地表信息 (Surface Pressure, 2m T, etc)
    output_sl = f'data/era5_sl_{TARGET_DATE}.nc'
    if not os.path.exists(output_sl):
        print(f"开始下载 ERA5 地面数据到 {output_sl} ...")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_temperature', '2m_dewpoint_temperature', 'surface_pressure',
                    'mean_sea_level_pressure', 'geopotential', # 地表位势 -> 地形高度
                    '10m_u_component_of_wind', '10m_v_component_of_wind'
                ],
                'year': TARGET_DATE.split('-')[0],
                'month': TARGET_DATE.split('-')[1],
                'day': TARGET_DATE.split('-')[2],
                'time': TARGET_TIME,
                'area': AREA,
            },
            output_sl)
    else:
        print(f"{output_sl} 已存在，跳过下载。")

    print("ERA5 数据准备完成。")

def generate_mock_era5_data():
    """
    生成假的 ERA5 数据用于测试流程 (当无法连接 CDS API 时使用)。
    """
    print("生成模拟 ERA5 数据...")

    # 经纬度网格
    lats = np.linspace(AREA[2], AREA[0], 10)
    lons = np.linspace(AREA[1], AREA[3], 10)
    levels = np.array([50, 100, 200, 500, 850, 925, 1000])
    times = pd.to_datetime([f"{TARGET_DATE} {TARGET_TIME}"])

    # 1. Mock Pressure Levels
    n_times = len(times)
    n_levels = len(levels)
    n_lats = len(lats)
    n_lons = len(lons)

    temp_data = 300 - 0.0065 * np.tile(levels[None,:,None,None], (1,1,10,10)) * 10 + np.random.randn(n_times, n_levels, n_lats, n_lons)
    sh_data = 0.01 * np.exp(-levels[None,:,None,None]/1000) + np.random.rand(n_times, n_levels, n_lats, n_lons)*0.001
    geo_data = np.tile(levels[None,:,None,None], (1,1,10,10)) * 100 * 9.8

    data_pl = {
        'temperature': (['time', 'level', 'latitude', 'longitude'], temp_data),
        'specific_humidity': (['time', 'level', 'latitude', 'longitude'], sh_data),
        'geopotential': (['time', 'level', 'latitude', 'longitude'], geo_data),
    }
    coords_pl = {
        'time': times,
        'level': levels,
        'latitude': lats,
        'longitude': lons
    }
    ds_pl = xr.Dataset(data_pl, coords=coords_pl)
    ds_pl.to_netcdf(f'data/era5_pl_{TARGET_DATE}.nc')

    # 2. Mock Surface Levels
    t2m_data = 290 + np.random.randn(n_times, n_lats, n_lons)
    sp_data = 101325 + np.random.randn(n_times, n_lats, n_lons)*100
    z_data = np.abs(np.random.randn(n_times, n_lats, n_lons))*100 * 9.8

    data_sl = {
        't2m': (['time', 'latitude', 'longitude'], t2m_data),
        'sp': (['time', 'latitude', 'longitude'], sp_data),
        'z': (['time', 'latitude', 'longitude'], z_data),
    }
    ds_sl = xr.Dataset(data_sl, coords={'time': times, 'latitude': lats, 'longitude': lons})
    ds_sl.to_netcdf(f'data/era5_sl_{TARGET_DATE}.nc')

    print("模拟数据生成完成。")

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')

    try:
        download_era5_data()
    except Exception as e:
        print(f"下载失败 (可能是没有 API Key): {e}")
        print("切换到模拟数据模式...")
        generate_mock_era5_data()
