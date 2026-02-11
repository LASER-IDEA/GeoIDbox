import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import radians, cos, sin, asin, sqrt

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def haversine(lon1, lat1, lon2, lat2):
    """
    计算两点间的大圆距离（单位：米）
    """
    if pd.isna(lon1) or pd.isna(lon2):
        return 0

    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 * 1000 # 地球平均半径，单位为米
    return c * r

def load_and_preprocess(file_paths):
    print("正在加载数据...")
    df_list = []
    for file in file_paths:
        try:
            df_temp = pd.read_csv(file)
            # 简单的周数标记，实际应用中可以更精确
            if '2025-11-10' in file:
                df_temp['week_tag'] = 'Week 1 (Nov 10)'
                df_temp['week_seq'] = 1
            elif '2025-11-17' in file:
                df_temp['week_tag'] = 'Week 2 (Nov 17)'
                df_temp['week_seq'] = 2
            elif '2025-11-24' in file:
                df_temp['week_tag'] = 'Week 3 (Nov 24)'
                df_temp['week_seq'] = 3
            else:
                df_temp['week_tag'] = 'Unknown Week'
                df_temp['week_seq'] = 0

            df_list.append(df_temp)
        except Exception as e:
            print(f"无法读取文件 {file}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    df['processed_time'] = pd.to_datetime(df['processed_time'])

    print(f"数据加载完成。总记录数: {len(df)}")
    print(f"包含 UUID 数量: {df['uid'].nunique()}")
    return df

def detect_and_visualize_movement(df, threshold_meters=50):
    """
    检测移动设备，并可视化其轨迹
    返回:
    1. moved_uuids: 发生移动的设备ID列表
    2. clean_df: 剔除移动设备后的纯净数据
    """
    print(f"\n--- 开始移动检测 (阈值: {threshold_meters}米) ---")

    # 1. 计算每周的质心位置
    weekly_pos = df.groupby(['uid', 'week_seq', 'week_tag'])[['avg_latitude', 'avg_longitude']].mean().reset_index()

    moved_devices_info = []
    moved_uuids = set()

    # 2. 遍历每个设备，检查周与周之间的距离
    for uid in weekly_pos['uid'].unique():
        uid_data = weekly_pos[weekly_pos['uid'] == uid].sort_values('week_seq')

        if len(uid_data) < 2:
            continue

        weeks = uid_data['week_seq'].values
        tags = uid_data['week_tag'].values
        lats = uid_data['avg_latitude'].values
        lons = uid_data['avg_longitude'].values

        is_moved = False
        max_dist = 0

        for i in range(len(weeks) - 1):
            dist = haversine(lons[i], lats[i], lons[i+1], lats[i+1])
            if dist > threshold_meters:
                is_moved = True
                max_dist = max(max_dist, dist)
                moved_devices_info.append({
                    'uid': uid,
                    'move_from': tags[i],
                    'move_to': tags[i+1],
                    'distance': dist,
                    'lat_from': lats[i],
                    'lon_from': lons[i]
                })

        if is_moved:
            moved_uuids.add(uid)

    # 3. 输出检测结果
    print(f"检测结果: 共有 {len(moved_uuids)} 个设备发生了 >{threshold_meters}m 的位移。")
    if len(moved_uuids) > 0:
        print("移动设备列表及最大单次位移:")
        # 简单打印前10个，避免刷屏
        for info in moved_devices_info[:10]:
             print(f"  - {info['uid']}: {info['move_from']} -> {info['move_to']} 移动了 {info['distance']:.1f}米")
        if len(moved_devices_info) > 10:
            print(f"  ... 以及其他 {len(moved_devices_info)-10} 次移动记录")

    # 4. 可视化轨迹
    plt.figure(figsize=(12, 10))

    # 绘制所有设备的点（灰色背景）
    plt.scatter(weekly_pos['avg_longitude'], weekly_pos['avg_latitude'],
                c='lightgray', alpha=0.5, s=20, label='Static Position')

    # 高亮绘制移动设备的轨迹
    if moved_uuids:
        moved_data = weekly_pos[weekly_pos['uid'].isin(moved_uuids)].sort_values(['uid', 'week_seq'])

        # 使用不同的颜色标记不同的移动设备
        for uid in moved_uuids:
            subset = moved_data[moved_data['uid'] == uid]
            plt.plot(subset['avg_longitude'], subset['avg_latitude'],
                     marker='o', linestyle='-', linewidth=2, markersize=8, alpha=0.8, label=f'Moved: {uid[-4:]}')

            # 标注起点和终点
            plt.text(subset.iloc[0]['avg_longitude'], subset.iloc[0]['avg_latitude'], 'Start', fontsize=8)
            plt.text(subset.iloc[-1]['avg_longitude'], subset.iloc[-1]['avg_latitude'], 'End', fontsize=8)

    plt.title(f'设备位置分布及异常移动轨迹检测 (阈值={threshold_meters}m)')
    plt.xlabel('经度 (Longitude)')
    plt.ylabel('纬度 (Latitude)')
    # 如果图例太多，只显示前几个
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > 10:
        plt.legend(handles[:10], labels[:10], title="部分设备示例")
    else:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. 返回清洗后的数据
    clean_df = df[~df['uid'].isin(moved_uuids)].copy()
    print(f"\n数据清洗完成: 原始记录 {len(df)} -> 清洗后记录 {len(clean_df)}")
    print(f"保留了 {clean_df['uid'].nunique()} 个位置稳定的设备用于深度分析。")

    return list(moved_uuids), clean_df

# --- 主执行逻辑 ---
# 替换为您的实际文件路径
file_list = [
    'sensor_data_week_2025-11-10_1min_agg.csv',
    'sensor_data_week_2025-11-17_1min_agg.csv',
    'sensor_data_week_2025-11-24_1min_agg.csv'
]

# 1. 加载
df = load_and_preprocess(file_list)

# 2. 检测移动并清洗
moved_uuids, clean_df = detect_and_visualize_movement(df, threshold_meters=50)

# 3. (可选) 保存清洗后的数据供后续使用
clean_df.to_csv('sensor_data_clean_stable.csv', index=False)