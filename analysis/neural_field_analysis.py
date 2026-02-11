import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

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

    device_stats = df.groupby('uid').agg({
        'avg_altitude': ['std', 'mean'],
        'avg_satellites': 'mean',
        'avg_hdop': 'mean'
    })
    device_stats.columns = ['alt_std', 'alt_mean', 'sat_mean', 'hdop_mean']
    device_stats = device_stats.sort_values('alt_std')

    tier1 = device_stats[device_stats['alt_std'] <= 1.5]
    print(f"Tier 1 (高精级 STD<=1.5m): {len(tier1)} 个设备")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=device_stats, x='sat_mean', y='alt_std', hue='alt_std', palette='viridis', size='hdop_mean', sizes=(20, 200))
    plt.axhline(y=1.5, color='g', linestyle='--', label='Tier 1 Threshold (1.5m)')
    plt.title('垂直精度(STD) vs 平均卫星数')
    plt.tight_layout()
    plt.show()

    return device_stats

def analyze_env_coupling(df):
    """
    2. 环境耦合分析
    """
    print("\n--- 2. 环境耦合分析 (Environmental Coupling) ---")
    cols = ['avg_altitude', 'avg_pressure', 'avg_temperature', 'avg_humidity']
    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('高度与环境因子相关性')
    plt.tight_layout()
    plt.show()

def analyze_diurnal_cycle(df, device_stats):
    """
    3. 日变化周期分析
    """
    print("\n--- 3. 日变化周期分析 (Multipath Detection) ---")
    stable_uid = device_stats.index[0]
    unstable_uid = device_stats.index[-1]

    targets = [stable_uid, unstable_uid]

    plt.figure(figsize=(12, 6))
    for uid in targets:
        subset = df[df['uid'] == uid].copy()
        subset['hour'] = subset['processed_time'].dt.hour
        mean_alt = subset['avg_altitude'].mean()
        hourly_dev = subset.groupby('hour')['avg_altitude'].mean() - mean_alt
        plt.plot(hourly_dev.index, hourly_dev.values, marker='o', label=f'Device: {uid[-4:]}')

    plt.title('GNSS 高度误差的日变化周期 (24小时叠加)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Deviation (m)')
    plt.legend()
    plt.show()

def analyze_physics_residuals(df):
    """
    4. 物理残差分析 (Target for Neural Field)
    - 拟合 Barometric Formula (物理骨架)
    - 提取 Residuals (需要神经网络学习的微气象修正量)
    """
    print("\n--- 4. 物理残差分析 (Physics Residual Mining) ---")

    # 过滤掉极度异常的 GNSS 数据，保证拟合的物理公式参数准确
    # 假设 1000m < P < 110000 (正常气压范围)
    valid_df = df[(df['avg_pressure'] > 90000) & (df['avg_altitude'] > -100) & (df['avg_altitude'] < 5000)].copy()

    # A. 拟合物理基准: ln(P) = -h/Hs + ln(P0)
    # y = ln(P), x = h
    # slope = -1/Hs, intercept = ln(P0)

    X = valid_df['avg_altitude'].values.reshape(-1, 1)
    y = np.log(valid_df['avg_pressure'].values)

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    Hs_fit = -1 / slope
    P0_fit = np.exp(intercept)

    print(f"拟合得到的物理参数: 标高 Hs = {Hs_fit:.2f} m, 基准气压 P0 = {P0_fit:.2f} Pa")
    print(f"物理公式拟合 R^2: {model.score(X, y):.4f}")

    # B. 计算残差 (Residual)
    # h_phys = -Hs * (ln(P) - ln(P0))
    valid_df['h_phys'] = -Hs_fit * (np.log(valid_df['avg_pressure']) - np.log(P0_fit))
    valid_df['residual'] = valid_df['avg_altitude'] - valid_df['h_phys']

    print(f"残差统计: Mean={valid_df['residual'].mean():.2f}m, Std={valid_df['residual'].std():.2f}m")

    # C. 可视化残差与微气象的关系 (这就是 Neural Field 要学的东西)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 残差 vs 温度
    sns.scatterplot(data=valid_df, x='avg_temperature', y='residual', ax=axes[0], alpha=0.3, s=10)
    # 拟合一条线看趋势
    z = np.polyfit(valid_df['avg_temperature'], valid_df['residual'], 1)
    p = np.poly1d(z)
    axes[0].plot(valid_df['avg_temperature'], p(valid_df['avg_temperature']), "r--", linewidth=2)
    axes[0].set_title(f'Residual vs Temperature (Corr: {valid_df["residual"].corr(valid_df["avg_temperature"]):.2f})')
    axes[0].set_ylabel('Height Residual (GNSS - PhysModel) [m]')

    # 2. 残差 vs 湿度
    sns.scatterplot(data=valid_df, x='avg_humidity', y='residual', ax=axes[1], alpha=0.3, s=10, color='green')
    z_h = np.polyfit(valid_df['avg_humidity'], valid_df['residual'], 1)
    p_h = np.poly1d(z_h)
    axes[1].plot(valid_df['avg_humidity'], p_h(valid_df['avg_humidity']), "r--", linewidth=2)
    axes[1].set_title(f'Residual vs Humidity (Corr: {valid_df["residual"].corr(valid_df["avg_humidity"]):.2f})')

    plt.tight_layout()
    plt.show()

    print("\n深度结论:")
    if abs(valid_df["residual"].corr(valid_df["avg_temperature"])) > 0.3:
        print(">> 发现显著的【温度漂移】效应。Neural Field 将能有效利用温度数据修正气压高度。")
    else:
        print(">> 温度对残差影响较小，可能是因为数据已做过温度补偿，或垂直温差不显著。")

    if abs(valid_df["residual"].corr(valid_df["avg_humidity"])) > 0.3:
        print(">> 发现显著的【湿度/水汽】效应。这证明了微气象数据对高度修正的必要性。")

# --- 主执行逻辑 ---
clean_file = 'sensor_data_clean_stable.csv'

if os.path.exists(clean_file):
    df_clean = load_clean_data(clean_file)
    if df_clean is not None and not df_clean.empty:
        device_stats = analyze_vertical_precision(df_clean)
        analyze_env_coupling(df_clean)
        analyze_diurnal_cycle(df_clean, device_stats)
        # 这一步替换了之前的 Feasibility Check，直接做残差挖掘
        analyze_physics_residuals(df_clean)
else:
    print(f"找不到文件 {clean_file}")