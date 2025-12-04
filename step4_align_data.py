import pandas as pd
import xarray as xr
import numpy as np
# from geographiclib.geoid import Geoid
from step3_get_geoid import get_geoid_undulation

# 1. 加载数据
print("加载无人机数据和气象数据...")
# 假设这是您之前步骤生成的CSV
df_drone = pd.read_csv('./data/drone_data_synced.csv')
# 假设这是您下载的NetCDF文件
ds_era5 = xr.open_dataset('./data/era5_pl_2024-11-24.nc')
# geoid = Geoid('egm96-5')
geoid = get_geoid_undulation

# 确保时间格式正确以便比较
df_drone['timestamp_dt'] = pd.to_datetime(df_drone['timestamp_us'], unit='us', origin='unix')
# 注意：这里假设日志的时间戳已经是UTC。如果不是，需要调整。
# ERA5的时间通常是UTC。

print("开始数据对齐（这可能需要一点时间）...")

aligned_data = []

# 为了加速，我们不逐行循环，而是采用一种简化的策略用于PoC：
# 策略：对于无人机的每个点，找到ERA5中【最近的时间点】和【最近的网格点】
# 进阶做法是做4D插值，但现在先跑通流程。

# 选取一个ERA5的参考气压层作为基准，比如最接近地面的层，假设是1000hPa或975hPa
# 您需要查看ds_era5['level']来确定有哪些层。这里假设用 1000 hPa 层。todo: how to get the closest level to the ground?
ref_level = 1000
print(ds_era5['pressure_level'])
ds_ref = ds_era5.sel(pressure_level=ref_level, method='nearest')

for index, row in df_drone.iterrows():
    drone_time = row['timestamp_dt']
    drone_lat = row['lat']
    drone_lon = row['lon']

    # 1. 在ERA5中找到最近的时间和空间点
    # 使用 .sel(method='nearest') 进行最近邻查找
    era5_point = ds_ref.sel(valid_time=drone_time, latitude=drone_lat, longitude=drone_lon, method='nearest')

    # 2. 提取气象参数
    # 注意单位转换：ERA5温度是开尔文(K)，位势是 m^2/s^2
    t_ref_k = float(era5_point['t']) # 参考层温度 (Kelvin)
    q_ref = float(era5_point['q'])   # 参考层比湿 (kg/kg)
    z_ref_geo = float(era5_point['z']) # 参考层位势 (Geopotential)

    # 将位势转换为大致的MSL高度 (除以重力加速度 g)
    h_ref_msl = z_ref_geo / 9.80665
    p_ref_pa = ref_level * 100.0 # 将hPa转换为Pa

    # 3. 计算该点的大地水准面差距 N
    n_value = get_geoid_undulation(drone_lat, drone_lon)

    # 4. 收集所有数据
    aligned_data.append({
        'timestamp': drone_time,
        'lat': drone_lat,
        'lon': drone_lon,
        'p_drone_pa': row['pressure_pa'],      # 输入 X1
        'h_hae_true': row['hae_alt_m'],        # 目标真值 Y
        # --- 气象背景参数 ---
        'p_ref_era5_pa': p_ref_pa,             # 输入 X2 (基准气压)
        'h_ref_era5_msl': h_ref_msl,           # 输入 X3 (基准高度)
        't_ref_era5_k': t_ref_k,               # 输入 X4 (基准温度)
        'q_ref_era5': q_ref,                   # 输入 X5 (基准湿度)
        # --- 地理参数 ---
        'n_geoid': n_value                     # 用于MSL/HAE转换
    })

# 创建最终的数据集
df_final = pd.DataFrame(aligned_data)
print("数据对齐完成！预览：")
print(df_final.head())

df_final.to_csv('final_training_data.csv', index=False)