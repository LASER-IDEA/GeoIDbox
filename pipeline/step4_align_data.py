import pandas as pd
import xarray as xr
import numpy as np
from step3_get_geoid import get_geoid_undulation
from step3b_downscale import Downscaler

# 1. 加载数据
print("加载无人机数据和气象数据...")
# 假设这是您之前步骤生成的CSV
df_drone = pd.read_csv('drone_data_synced.csv')
# df_drone = pd.read_csv('test.csv') # Use this if testing with small set

# 初始化 Downscaler
# 确保 step2 已经生成了这些文件
ds = Downscaler('data/era5_pl_2024-11-24.nc', 'data/era5_sl_2024-11-24.nc')

# 确保时间格式正确以便比较
df_drone['timestamp_dt'] = pd.to_datetime(df_drone['timestamp_us'], unit='us', origin='unix')

print("开始数据对齐与降尺度 (这可能需要一点时间)...")

aligned_data = []

# 为了提高效率，可以预先加载 Geoid
# 也可以在这里做 batch 处理，但为了清晰展示逻辑，我们逐点处理
count = 0
total = len(df_drone)

for index, row in df_drone.iterrows():
    drone_time = row['timestamp_dt']
    drone_lat = row['lat']
    drone_lon = row['lon']

    # --- 1. Downscaling ---
    # 获取该位置的高精度气象背景场 (基于 DSM 高度)
    # 注意：这里我们获取的是 DSM 高度处的气象数据，作为物理模型的 "Reference"
    # 物理模型将从这个 Reference 高度/气压 推算到 无人机测量的气压处的高度
    meteo_bg = ds.downscale_point(drone_lat, drone_lon)

    # --- 2. Geoid Undulation ---
    # 获取 EGM08/96 的 N 值
    n_value = get_geoid_undulation(drone_lat, drone_lon)

    # --- 3. 收集数据 ---
    aligned_data.append({
        'timestamp': drone_time,
        'lat': drone_lat,
        'lon': drone_lon,
        'p_drone_pa': row['pressure_pa'],      # 传感器实测气压 (Pa)
        'h_hae_true': row['hae_alt_m'],        # RTK 实测高度 (HAE)

        # --- Downscaled ERA5 Reference Data ---
        # 这些是物理模型的输入 (Reference values)
        # 这里的 Reference Point 是地面 (DSM surface)
        'p_ref_pa': meteo_bg['p_downscaled_pa'],
        'h_ref_msl': meteo_bg['h_dsm'],        # DSM height is approx MSL (orthometric)
        't_ref_k': meteo_bg['t_downscaled_k'],
        'q_ref': meteo_bg['q_downscaled'],

        # --- Environmental Features for Neural Field ---
        'roughness': meteo_bg['roughness'],

        # --- Geoid ---
        'n_geoid': n_value
    })

    count += 1
    if count % 100 == 0:
        print(f"Processed {count}/{total}")

# 创建最终的数据集
df_final = pd.DataFrame(aligned_data)
print("数据对齐完成！预览：")
print(df_final.head())

df_final.to_csv('final_training_data.csv', index=False)
print("已保存到 final_training_data.csv")
