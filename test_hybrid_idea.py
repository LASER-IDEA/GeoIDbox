import pandas as pd
import numpy as np

def compute_isa_height_msl(pressure_pa, era5_sp=None):
    P0_ISA = 101325.0
    T0_ISA = 288.15
    ALPHA_ISA = 0.0065
    p0 = era5_sp if era5_sp is not None else P0_ISA
    p = np.clip(pressure_pa, 1000.0, None)
    return (T0_ISA / ALPHA_ISA) * (1.0 - (p / p0) ** (ALPHA_ISA * 287.05 / 9.80665))

df = pd.read_csv('data/processed/sensor_data_stabilized.csv')
df['time_key'] = pd.to_datetime(df['processed_time']).dt.round('10min')
df['n_geoid'] = 20.0
df['h_isa'] = compute_isa_height_msl(df['avg_pressure'].values)
df['h_true_msl'] = df['avg_altitude'] - df['n_geoid']
df['residual'] = df['h_true_msl'] - df['h_isa']

# Let's just do Fold 1
test_sensor = df['uid'].unique()[0]
train_df = df[df['uid'] != test_sensor].copy()
test_df = df[df['uid'] == test_sensor].copy()

spatial_bias_map = train_df.groupby('uid')['residual'].mean().to_dict()
train_df['spatial_bias'] = train_df['uid'].map(spatial_bias_map)
train_df['temporal_anomaly'] = train_df['residual'] - train_df['spatial_bias']
weather_signal_map = train_df.groupby('time_key')['temporal_anomaly'].mean().to_dict()
train_df['weather_signal'] = train_df['time_key'].map(weather_signal_map)

# Micro residual
train_df['micro_residual'] = train_df['temporal_anomaly'] - train_df['weather_signal']

print(f"Micro residual var: {train_df['micro_residual'].var():.2f}")
print(f"Micro residual MAE: {train_df['micro_residual'].abs().mean():.2f}")
