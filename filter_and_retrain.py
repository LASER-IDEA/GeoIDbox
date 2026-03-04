
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Ensure we can import from GeoIDbox
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import necessary classes and functions from pinn_field_reconstruction
from pinn_field_reconstruction import (
    WeatherField,
    DataNormalizer,
    FourierFeatures,
    CombinedDataset,
    load_data,
    train,
    standard_atmosphere,
    barometric_formula_height,
    g, R, T0, L_b, P0  # Constants
)

def fast_height_solve_approximation(model, normalizer, df, device):
    """
    Vectorized approximation of height error using linearization around reported altitude.
    h_pred = h_true + (P_measured - P_model(h_true)) / (dP/dh)
    dP/dh ≈ -rho * g ≈ -P / (R * T) * g
    """
    model.eval()

    # Prepare batch input
    lat = torch.tensor(df['lat'].values, dtype=torch.float32, device=device)
    lon = torch.tensor(df['lon'].values, dtype=torch.float32, device=device)
    alt = torch.tensor(df['alt'].values, dtype=torch.float32, device=device) # use observed alt as initial guess
    timestamp = torch.tensor(df['timestamp'].values, dtype=torch.float32, device=device)

    # Normalize coordinates
    n_coords = normalizer.normalize_coords(lat, lon, alt, timestamp).to(device)

    with torch.no_grad():
        preds = model(n_coords)
        p_res_raw = preds[:, 0]
        # p_res_raw, t_res_raw = preds[:, 0], preds[:, 1]

        # Scale outputs (using logic from DataNormalizer.scale_outputs in pinn_field_reconstruction)
        # Note: We need to access the scaler values or method directly if possible.
        # Since we imported DataNormalizer, we can use the instance logic if we have the instance.
        # But here we are passing 'normalizer' object.
        p_res, _ = normalizer.scale_outputs(p_res_raw, preds[:, 1])

        # Calculate Base P at h_true
        # standard_atmosphere can handle tensors?
        # Let's check implementation of standard_atmosphere.
        # It uses simple math operations, so it should work with tensors on device.
        p_base, t_base = standard_atmosphere(alt)

        # Predicted Pressure at h_true
        p_model_at_h_true = p_base + p_res

        # Measured Pressure
        p_measured = torch.tensor(df['pressure'].values, dtype=torch.float32, device=device)

        # DEBUG: Print first 5 comparisons
        if len(df) > 0:
             print("\nDEBUG fast_height_solve_approximation:")
             print(f"Alt (True): {alt[:5].cpu().numpy()}")
             print(f"P (Meas): {p_measured[:5].cpu().numpy()}")
             print(f"P (Base @ Alt): {p_base[:5].cpu().numpy()}")
             print(f"P (Res Model): {p_res[:5].cpu().numpy()}")
             print(f"P (Model Total): {p_model_at_h_true[:5].cpu().numpy()}")
             print(f"Diff P: {(p_measured - p_model_at_h_true)[:5].cpu().numpy()}")

        # Calculate dP/dh approx
        # dP/dz = -rho * g = -P / (R * T) * g
        # Use model T or base T? Base T is stable.
        rho = p_model_at_h_true / (R * t_base)
        dp_dh = -rho * g

        # Delta P = P_measured - P_model(h_true)
        # Delta P ≈ dP/dh * (h_pred - h_true)
        # h_pred - h_true ≈ Delta P / dP/dh

        diff_p = p_measured - p_model_at_h_true

        # Avoid division by zero
        dp_dh = torch.where(torch.abs(dp_dh) < 1e-6, torch.tensor(-1.0, device=device), dp_dh)

        delta_h = diff_p / dp_dh

        # predicted_h = alt + delta_h
        # Error = |predicted_h - alt| = |delta_h|

        return delta_h.abs().cpu().numpy()

def main():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    sensor_df, era5_df = load_data()
    if sensor_df is None:
        print("Failed to load data.")
        return

    # Initialize Normalizer (re-fit on current data to be consistent with training logic)
    # Note: If the saved model depends on a specific normalization state that was not saved,
    # this might be slightly off if data changed. Assuming data is same.

    # Create temp combined DF for fitting normalizer (same as in pinn_field_reconstruction.py)
    s_temp = sensor_df[['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']]
    e_temp = era5_df[['lat', 'lon', 'static_height', 'timestamp', 'sp', 't2m']]
    e_temp.columns = ['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']
    combined_temp = pd.concat([s_temp, e_temp], ignore_index=True)

    normalizer = DataNormalizer()
    normalizer.fit(combined_temp)

    # Load Model
    model_path = 'pinn_model.pth'
    # Try looking in GeoIDbox if not in current dir
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, 'pinn_model.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train model first or adjust path.")
        return

    print(f"Loading model from {model_path}...")
    model = WeatherField(num_freqs=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Analyze: Calculate MAE per UID
    print("\n--- Analyzing Sensors ---")

    # Add error column to dataframe
    # Using fast approximation
    mae_errors = fast_height_solve_approximation(model, normalizer, sensor_df, device)
    sensor_df['mae_error'] = mae_errors

    # Group by UID using string formatting for clarity if needed
    if 'uid' not in sensor_df.columns:
        print("Error: 'uid' column missing in sensor data.")
        return

    uid_stats = sensor_df.groupby('uid')['mae_error'].mean().reset_index()
    uid_stats.columns = ['uid', 'mae']
    uid_stats = uid_stats.sort_values('mae', ascending=False)

    print("\nSensor MAE Statistics:")
    print(uid_stats)

    # 3. Filter
    # Check if we have any good sensors
    min_mae = uid_stats['mae'].min()
    print(f"Minimum MAE observed: {min_mae:.2f} m")

    if min_mae > 15.0:
        print("WARNING: All sensors have MAE > 15m. Filtering by threshold will result in empty dataset.")
        print("Switching strategy: Keep top 50% of sensors based on MAE.")
        threshold = uid_stats['mae'].median()
        good_sensors = uid_stats[uid_stats['mae'] <= threshold]
        print(f"New relative threshold: {threshold:.2f} m")
    else:
        MAE_THRESHOLD = 15.0 # meters
        good_sensors = uid_stats[uid_stats['mae'] <= MAE_THRESHOLD]
        print(f"Filtering Criteria: MAE <= {MAE_THRESHOLD}m")

    bad_sensors = uid_stats[~uid_stats['uid'].isin(good_sensors['uid'])]

    print(f"Bad Sensors (Count: {len(bad_sensors)}):")
    if not bad_sensors.empty:
        print(bad_sensors)
    else:
        print("None found.")

    print(f"Good Sensors (Count: {len(good_sensors)})")

    good_uids = set(good_sensors['uid'].values)

    # Filter original dataframe using 'uid'
    # Note: sensor_df was a copy with dropped NaNs etc. load_data reads from disk.
    # We should reload raw or apply mask to loaded df. 'sensor_df' is derived from raw.
    # But for creating 'sensor_data_clean_elite.csv', we probably want to filter the source file?
    # Or just save the DataFrame we have?
    # The 'load_data' function reads 'sensor_data_filtered_outliers.csv'.
    # Whatever pre-processing load_data did (renaming columns), we want to keep that or revert?
    # Usually better to load source, filter UIDs, save.

    source_path = 'data/processed/sensor_data_filtered_outliers.csv'
    if not os.path.exists(source_path):
        source_path = 'GeoIDbox/data/processed/sensor_data_filtered_outliers.csv'

    df_source = pd.read_csv(source_path)

    # Filter
    # Check if uid is in good_uids
    # Ensure uid column exists
    if 'uid' not in df_source.columns:
        print("Using 'avg_latitude' etc implies processed file might have different schema than raw?")
        # load_data mapped 'uid' -> 'uid'. So it should be there.
        pass

    df_elite = df_source[df_source['uid'].isin(good_uids)].copy()

    output_path = 'data/processed/sensor_data_clean_elite.csv'
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed', exist_ok=True)
        # If running from root, check path
        if os.path.exists('GeoIDbox/data/processed'):
             output_path = 'GeoIDbox/data/processed/sensor_data_clean_elite.csv'

    print(f"Saving {len(df_elite)} records to {output_path}...")
    df_elite.to_csv(output_path, index=False)

    # 4. Retrain
    print("\n--- Retraining with Elite Data ---")

    # Create new dataset with elite data
    # We can reuse DataNormalizer and CombinedDataset classes
    # But we need to reload 'sensor_df' from the new elite file (or just use the filtered dataframe in memory)

    # Reloading to ensure clean slate and correct column mappings from 'load_data' equivalent logic
    # Actually, we can just pass the filtered dataframe to a modified load_data or construct manually.
    # But 'load_data' inside pinn_field_reconstruction is hardcoded to a path.
    # We can monkeypatch it or just construct the dataset manually here.

    # Let's construct manually to avoid changing source code or complex patching

    # Prepare Sensor Data (Elite)
    # Using df_elite, we need to apply same rename/processing as load_data
    cols = ['avg_latitude', 'avg_longitude', 'avg_altitude', 'avg_pressure', 'avg_temperature', 'timestamp']
    if 'uid' in df_elite.columns:
        cols.append('uid')

    # Recalculate timestamp if needed (load_data does this: df['dt'] - min).
    # Important: Normalizer needs to be consistent.
    # If we filter data, the min/max time/lat/lon might shrink.
    # Should we re-fit normalizer on the subset? Ideally yes, for best training dynamics.

    if 'processed_time' in df_elite.columns:
        df_elite['dt'] = pd.to_datetime(df_elite['processed_time'])
        t_min = df_elite['dt'].min() # This might be different start time!
        df_elite['timestamp'] = (df_elite['dt'] - t_min).dt.total_seconds()
    else:
        # Fallback
        df_elite['timestamp'] = df_elite.index * 60

    sensor_df_elite = df_elite[cols].copy()
    rename_dict = {
        'avg_latitude': 'lat', 'avg_longitude': 'lon', 'avg_altitude': 'alt',
        'avg_pressure': 'pressure', 'avg_temperature': 'temperature', 'timestamp': 'timestamp',
        'uid': 'uid'
    }
    sensor_df_elite = sensor_df_elite.rename(columns=rename_dict).dropna()

    # Prepare ERA5 Data (Same as before)
    # We can reuse the already loaded 'era5_df', but we should probably re-align timestamps if t_min changed?
    # Wait, t_min changed means timestamp 0 is different.
    # ERA5 timestamp must align with sensor timestamp.
    # The ERA5 timestamps in 'era5_df' from load_data were computed using the ORIGINAL t_min.
    # If we change t_min, we shift the time axis.
    # For simplicity, let's Stick to the ORIGINAL t_min and normalizer if possible,
    # OR re-process ERA5 with the new t_min.
    # Re-processing ERA5 is safer.

    # Re-extract ERA5 from df_elite (assuming filtered file still covers the range?
    # Wait, df_elite is a subset of rows. ERA5 columns in the CSV are likely same for all rows at same time?
    # NO, the CSV merges ERA5 data onto sensor rows. So if we drop sensor rows, we drop ERA5 points too.
    # That's fine, we treat them as training points.

    era5_cols = ['avg_latitude', 'avg_longitude', 'era5_sp', 'era5_t2m', 'timestamp']
    era5_df_elite = df_elite[era5_cols].copy()
    era5_df_elite.columns = ['lat', 'lon', 'sp', 't2m', 'timestamp']
    era5_df_elite = era5_df_elite.dropna()

    # ERA5 Static Height Logic (Group by grid and average)
    era5_df_elite['lat_grid'] = era5_df_elite['lat'].round(3)
    era5_df_elite['lon_grid'] = era5_df_elite['lon'].round(3)
    mean_sp = era5_df_elite.groupby(['lat_grid', 'lon_grid'])['sp'].mean().reset_index()
    mean_sp.rename(columns={'sp': 'mean_sp'}, inplace=True)
    mean_sp['static_height'] = barometric_formula_height(mean_sp['mean_sp'])
    era5_df_elite = pd.merge(era5_df_elite, mean_sp, on=['lat_grid', 'lon_grid'], how='left')

    # Create temp combined DF for fitting normalizer
    # Note: ERA5 dataframe column names might be different than expected by normalizer or combined dataset
    # pinn_field_reconstruction logic:
    #   if 'static_height' in era5_df: use it as alt
    #   if 'sp' in era5_df: use it as pressure
    #   if 't2m' in era5_df: use it as temperature

    # We must match the column names used in CombinedDataset.__init__
    # Reading pinn_field_reconstruction.py CombinedDataset:
    #   self.e_coords = normalizer.normalize_coords(..., era5_df['static_height'], ...)
    #   self.e_p = ... era5_df['sp']
    #   self.e_t = ... era5_df['t2m']

    # So era5_df passed to CombinedDataset MUST have 'static_height', 'sp', 't2m'

    # But for normalizer.fit(combined_temp_elite), we need unified names 'alt', 'pressure', 'temperature'

    # Rename for concatenation
    s_part = sensor_df_elite[['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']]
    e_part = era5_df_elite[['lat', 'lon', 'static_height', 'timestamp', 'sp', 't2m']].rename(
        columns={'static_height': 'alt', 'sp': 'pressure', 't2m': 'temperature'}
    )

    combined_temp_elite = pd.concat([s_part, e_part], ignore_index=True)

    normalizer_elite = DataNormalizer()
    normalizer_elite.fit(combined_temp_elite)

    # Dataset
    # Re-initialize dataset with correct dataframes
    dataset_elite = CombinedDataset(sensor_df_elite, era5_df_elite, normalizer_elite)

    # New Model
    print("Initializing new model...")
    model_elite = WeatherField(num_freqs=10).to(device)

    # Train
    print("Starting retraining...")
    model_elite, losses = train(model_elite, dataset_elite, normalizer_elite, device, epochs=500)

    # Save
    new_model_path = 'pinn_model_elite.pth'
    if 'GeoIDbox' in current_dir:
         new_model_path = os.path.join(current_dir, 'pinn_model_elite.pth')

    torch.save(model_elite.state_dict(), new_model_path)
    print(f"Saved refined model to {new_model_path}")

    # Evaluate again
    print("\n--- Final Evaluation (Elite Model) ---")
    model_elite.eval()

    mae_errors_elite = fast_height_solve_approximation(model_elite, normalizer_elite, sensor_df_elite, device)

    print(f"Original Mean MAE: {np.mean(mae_errors):.2f} m")
    print(f"Refined Mean MAE: {np.mean(mae_errors_elite):.2f} m (on elite subset)")

    # Optional: Check on bad sensors?
    # Usually we don't care about bad sensors anymore if we decided they are bad.

if __name__ == "__main__":
    main()
