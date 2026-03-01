import pandas as pd
import numpy as np

# 常量定义
R_dry = 287.05  # 干空气气体常数
g0 = 9.80665    # 标准重力加速度

df = pd.read_csv('final_training_data.csv')

print("计算物理基准高度...")

# 1. 计算虚拟温度 (Virtual Temperature)
# 湿度会使空气变轻，相当于温度升高。这一步对低空非常重要。
# 公式: Tv = T * (1 + 0.61 * q)
df['t_virtual_k'] = df['t_ref_era5_k'] * (1 + 0.61 * df['q_ref_era5'])

# 2. 应用压高公式 (Hypsometric Equation)
# 我们使用 ERA5 的参考层作为起点，推算无人机所在气压层的高度。
# 使用对数形式，假设在参考层和无人机之间温度是线性的（或使用平均虚温）
# 简化版公式: h_diff = (R * Tv_avg / g) * ln(P1 / P2)

# 这里使用参考层的虚温近似平均虚温（对于低空短距离推算尚可）
# 计算无人机相对于ERA5参考层的高度差 (MSL高度差)
df['delta_h_msl'] = (R_dry * df['t_virtual_k'] / g0) * \
                    np.log(df['p_ref_era5_pa'] / df['p_drone_pa'])

# 3. 得到物理模型预测的 MSL 高度
df['h_msl_pred_phy'] = df['h_ref_era5_msl'] + df['delta_h_msl']

# 4. 关键一步：转换为 HAE 高度
# HAE = MSL + N
df['h_hae_pred_phy'] = df['h_msl_pred_phy'] + df['n_geoid']

# 5. 计算物理模型的误差 (Residual)
# 残差 = RTK真值 - 物理预测值
df['residual_hae'] = df['h_hae_true'] - df['h_hae_pred_phy']

print("物理基准计算完成。误差统计(米):")
print(df['residual_hae'].describe())

# 保存带有残差结果的数据，这是下一步AI训练的目标
df.to_csv('data_with_residual.csv', index=False)

# 可视化检查 (强烈建议)
# 如果您有matplotlib，画出残差看看。它应该不是纯白噪声，而是有一定趋势。
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(df['residual_hae'], label='Physics Residual (m)')
# plt.legend()
# plt.title("Step 2 Residual: What Physics Missed")
# plt.show()