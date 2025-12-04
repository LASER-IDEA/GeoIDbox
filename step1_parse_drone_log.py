import pandas as pd
import numpy as np
from pyulog import ULog
from datetime import datetime, timezone

def parse_ulog(log_path):
    """
    解析ULog文件，提取气压和高精度HAE高度，并对齐时间戳。
    """
    print(f"正在解析日志: {log_path}...")
    ulog = ULog(log_path)

    # print uuid
    print(ulog['vehicle_uuid'])

    # 1. 提取气压数据 (来自 vehicle_air_data 主题)
    # baro_alt_meter 是基于标准大气的粗略高度，baro_pressure_pa 是原始气压
    try:
        air_data = ulog.get_dataset('vehicle_air_data').data
        df_baro = pd.DataFrame({
            'timestamp_us': air_data['timestamp'],
            'pressure_pa': air_data['baro_pressure_pa'],
            # 'baro_temp_c': air_data['baro_temp_celcius'] # 如果有需要可以提取温度
        })
    except Exception as e:
        print(f"错误: 无法在日志中找到气压数据 (vehicle_air_data). {e}")
        return None

    # 2. 提取高精度 GPS 数据 (来自 vehicle_gps_position 主题)
    # alt 是 MSL 高度, alt_ellipsoid 是 HAE 高度 (我们需要这个!)
    try:
        gps_data = ulog.get_dataset('vehicle_gps_position').data
        # 检查是否为 RTK 数据 (fix_type: 3=2D fix, 4=3D fix, 5=RTK Float, 6=RTK Fixed)
        fix_types = np.unique(gps_data['fix_type'])
        print(f"GPS Fix Types found: {fix_types} (期望包含 5 或 6)")

        df_gps = pd.DataFrame({
            'timestamp_us': gps_data['timestamp'],
            'lat': gps_data['lat'] / 1e7,  # 转换为度
            'lon': gps_data['lon'] / 1e7,  # 转换为度
            'hae_alt_m': gps_data['alt_ellipsoid'] / 1000.0, # 转换为米
            'fix_type': gps_data['fix_type']
        })
        # 仅保留良好的3D Fix或RTK数据
        df_gps = df_gps[df_gps['fix_type'] >= 4].copy()

    except Exception as e:
        print(f"错误: 无法在日志中找到GPS数据 (vehicle_gps_position). {e}")
        return None

    # 3. 数据对齐与合并
    # 因为气压计和GPS的采样率不同，我们需要按时间戳对齐。
    # 这里以气压计的时间戳为基准，插值 GPS 数据。
    df_baro = df_baro.set_index('timestamp_us').sort_index()
    df_gps = df_gps.set_index('timestamp_us').sort_index()

    # 使用最近邻插值或线性插值将GPS数据对齐到气压数据的时间点
    # 注意：实际应用中可能需要更复杂的同步处理
    df_merged = pd.merge_asof(df_baro, df_gps, on='timestamp_us', direction='nearest', tolerance=100000) # 容差100ms

    # 删除没有对齐的数据
    df_merged.dropna(inplace=True)

    # 将微秒时间戳转换为 UTC 时间 (假设日志基准时间正确)
    # 注意：PX4日志的时间戳通常是系统启动时间，获取绝对UTC时间比较麻烦。
    # 这里为了简化，我们假设我们知道飞行的确切日期，或者日志中包含了 UTC 时间基准。
    # **关键：在实际操作中，您需要知道这个日志是哪一天飞的，以便下载对应的气象数据。**
    # 这里我们先只保留相对时间用于分析。

    print(f"解析完成! 共提取 {len(df_merged)} 条对齐数据。")
    print("数据预览:")
    print(df_merged[['pressure_pa', 'hae_alt_m', 'lat', 'lon']].head())

    # 保存为 CSV 供后续使用
    output_file = 'drone_data_synced.csv'
    df_merged.to_csv(output_file, index=False)
    print(f"已保存对齐后的数据到: {output_file}")
    return df_merged

# --- 执行 ---
# 请将您下载的日志文件名替换如下：
log_file_name = './data/ae8d1e38-ec70-4b4b-bc45-650a48a8effb.ulg'
parse_ulog(log_file_name) # 取消注释并运行
# print("请先下载ULog文件，修改脚本中的文件名后再运行。")