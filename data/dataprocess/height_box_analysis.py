import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_clean_data(file_path):
    print(f"正在加载清洗后的数据: {file_path} ...")
    try:
        df = pd.read_csv(file_path)
        df['processed_time'] = pd.to_datetime(df['processed_time'])
        print(f"加载完成。包含 {df['uid'].nunique()} 个设备，共 {len(df)} 条记录。")
        return df
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def analyze_vertical_precision(df):
    """
    1. 垂直精度分级分析
    """
    print("\n--- 1. 垂直精度分级 (Vertical Precision Tiering) ---")

    # 计算每个设备的 MSL 高度标准差 (衡量稳定性)
    # 同时计算平均卫星数，看是否卫星数少导致了不稳定
    device_stats = df.groupby('uid').agg({
        'avg_altitude': ['std', 'mean'],
        'avg_satellites': 'mean',
        'avg_hdop': 'mean'
    })
    device_stats.columns = ['alt_std', 'alt_mean', 'sat_mean', 'hdop_mean']
    device_stats = device_stats.sort_values('alt_std')

    # 分级定义
    tier1 = device_stats[device_stats['alt_std'] <= 1.5]
    tier2 = device_stats[(device_stats['alt_std'] > 1.5) & (device_stats['alt_std'] <= 5.0)]
    tier3 = device_stats[device_stats['alt_std'] > 5.0]

    print(f"Tier 1 (高精级 STD<=1.5m): {len(tier1)} 个设备")
    print(f"Tier 2 (标准级 STD<=5.0m): {len(tier2)} 个设备")
    print(f"Tier 3 (预警级 STD> 5.0m): {len(tier3)} 个设备")

    if len(tier3) > 0:
        print("\n[警告] 极不稳定的设备 (Top 3):")
        print(tier3.tail(3)[['alt_std', 'sat_mean', 'hdop_mean']])

    # 可视化：精度分布与卫星数的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=device_stats, x='sat_mean', y='alt_std', hue='alt_std', palette='viridis', size='hdop_mean', sizes=(20, 200))
    plt.axhline(y=1.5, color='g', linestyle='--', label='Tier 1 Threshold (1.5m)')
    plt.axhline(y=5.0, color='r', linestyle='--', label='Tier 2 Threshold (5.0m)')
    plt.title('垂直精度(STD) vs 平均卫星数')
    plt.xlabel('平均卫星数')
    plt.ylabel('高度标准差 (米) - 越低越好')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return device_stats

def analyze_env_coupling(df):
    """
    2. 环境耦合分析 (气压/温度 vs 高度)
    """
    print("\n--- 2. 环境耦合分析 (Environmental Coupling) ---")

    # 计算相关性矩阵
    cols = ['avg_altitude', 'avg_pressure', 'avg_temperature', 'avg_humidity', 'avg_satellites']
    corr = df[cols].corr()

    # 热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('高度与环境因子相关性热力图')
    plt.tight_layout()
    plt.show()

    # 重点分析：气压 vs 高度 (物理一致性检查)
    # 选取一个数据量最足的稳定设备进行展示，避免不同位置设备的干扰
    sample_uid = df['uid'].value_counts().idxmax()
    sample_data = df[df['uid'] == sample_uid]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('时间')
    ax1.set_ylabel('GNSS 高度 (m)', color=color)
    ax1.plot(sample_data['processed_time'], sample_data['avg_altitude'], color=color, alpha=0.6, label='GNSS Altitude')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 共享x轴
    color = 'tab:red'
    ax2.set_ylabel('气压 (Pa)', color=color)
    # 气压通常与高度成反比，为了视觉一致性，可以反转坐标轴或者只要看趋势
    ax2.plot(sample_data['processed_time'], sample_data['avg_pressure'], color=color, alpha=0.6, linestyle='--', label='Pressure')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'单设备时序分析 (UID: {sample_uid[-4:]}): 验证气压与GNSS高度的物理一致性')
    plt.tight_layout()
    plt.show()

    print(f"对于设备 {sample_uid}，高度与气压的相关系数为: {sample_data['avg_altitude'].corr(sample_data['avg_pressure']):.4f}")
    print("解读: 如果相关系数接近 -1 (负相关)，说明GNSS高度非常敏锐地捕捉到了真实高度变化或气压高度变化规律。")

def analyze_diurnal_cycle(df, device_stats):
    """
    3. 日变化周期分析 (检测多路径效应)
    """
    print("\n--- 3. 日变化周期分析 (Multipath Detection) ---")

    # 选取最稳定(Tier 1)和最不稳定(Tier 3)的代表设备
    stable_uid = device_stats.index[0] # STD 最小
    unstable_uid = device_stats.index[-1] # STD 最大

    targets = [stable_uid, unstable_uid]
    labels = ['最稳定 (Best)', '最不稳定 (Worst)']

    plt.figure(figsize=(12, 6))

    for uid, label in zip(targets, labels):
        subset = df[df['uid'] == uid].copy()
        # 提取小时 (0-23)
        subset['hour'] = subset['processed_time'].dt.hour

        # 计算每个小时的高度相对于该设备总平均值的偏差
        mean_alt = subset['avg_altitude'].mean()
        hourly_dev = subset.groupby('hour')['avg_altitude'].mean() - mean_alt

        plt.plot(hourly_dev.index, hourly_dev.values, marker='o', label=f'{label}: {uid[-4:]}')

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title('GNSS 高度误差的日变化周期 (24小时叠加)')
    plt.xlabel('小时 (Hour of Day)')
    plt.ylabel('高度偏差 (Dev from Mean, m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("解读: 如果曲线呈现明显的正弦波状（例如中午高、半夜低），这通常是多路径效应或温度对天线相位中心影响的特征。")

# --- 主执行逻辑 ---
# 此时我们直接加载清洗后的文件
clean_file = 'sensor_data_clean_stable.csv'

if os.path.exists(clean_file):
    df_clean = load_clean_data(clean_file)

    if df_clean is not None and not df_clean.empty:
        # 1. 精度分级
        device_stats = analyze_vertical_precision(df_clean)

        # 2. 环境分析
        analyze_env_coupling(df_clean)

        # 3. 周期性分析
        analyze_diurnal_cycle(df_clean, device_stats)
else:
    print(f"找不到文件 {clean_file}，请确保上一步已成功运行并保存了文件。")