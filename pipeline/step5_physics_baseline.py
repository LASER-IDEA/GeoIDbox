import pandas as pd
import numpy as np

# 常量定义
R_dry = 287.05  # 干空气气体常数
g0 = 9.80665    # 标准重力加速度
# L = -0.0065   # Standard Lapse Rate (K/m), not used in the user's specific formula, but implicit in alpha if needed.
# User Formula: Z_p = (T_ref / alpha) * [1 - (p / p_ref)^(alpha * R / g0)] + Z_ref
# This looks like the standard barometric formula where alpha is the lapse rate.
# Standard lapse rate alpha is usually denoted as L = 0.0065 K/m (positive value in formula usually)
ALPHA = 0.0065 # K/m

df = pd.read_csv('final_training_data.csv')

print("计算物理基准高度...")

# 1. 准备变量
# T_ref, P_ref, Z_ref (MSL) are from the downscaled ERA5 at the DSM surface.
# P_drone is the measured pressure.
t_ref = df['t_ref_k']
p_ref = df['p_ref_pa']
z_ref_msl = df['h_ref_msl']
p_drone = df['p_drone_pa']

# 2. 应用用户提供的压高公式 (Hypsometric Formula)
# Formula: Z = (T_ref / alpha) * [1 - (p / p_ref)^(alpha * R / g0)] + Z_ref
# Note: This assumes linear temperature lapse rate.

# Exponent factor
exponent = (ALPHA * R_dry) / g0

# Calculate Height
# We need to handle the case where alpha is 0 (isothermal), but here we assume standard atmosphere behavior locally.
term1 = t_ref / ALPHA
term2 = 1 - (p_drone / p_ref) ** exponent
h_msl_pred_phy = term1 * term2 + z_ref_msl

df['h_msl_pred_phy'] = h_msl_pred_phy

# 3. 转换为 HAE 高度
# HAE = MSL + N
df['h_hae_pred_phy'] = df['h_msl_pred_phy'] + df['n_geoid']

# 4. 计算物理模型的误差 (Residual)
# 残差 = RTK真值 - 物理预测值
df['residual_hae'] = df['h_hae_true'] - df['h_hae_pred_phy']

print("物理基准计算完成。误差统计(米):")
print(df['residual_hae'].describe())

# 保存带有残差结果的数据，这是下一步AI训练的目标
df.to_csv('data_with_residual.csv', index=False)
print("已保存 data_with_residual.csv")
