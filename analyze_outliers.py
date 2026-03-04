
import pandas as pd
import numpy as np

# Constants for ISA Model
P0 = 101325.0  # Sea level standard pressure (Pa)
T0 = 288.15    # Sea level standard temperature (K)
L = 0.0065     # Temperature lapse rate (K/m)
R = 287.05     # Specific gas constant for dry air (J/(kg·K))
g = 9.80665    # Gravity (m/s^2)

def calculate_h_isa(p_obs):
    """
    Calculate ISA altitude from observed pressure.
    Formula: h = (T0 / L) * (1 - (P / P0)^(R * L / g))
    """
    exponent = (R * L) / g
    scaling_factor = T0 / L
    return scaling_factor * (1 - (p_obs / P0) ** exponent)

def main():
    # 1. Load Data
    input_csv = 'GeoIDbox/data/processed/sensor_data_with_real_era5.csv'
    output_csv = 'GeoIDbox/data/processed/sensor_data_filtered_outliers.csv'

    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        # Fallback for running from root
        input_csv = 'data/processed/sensor_data_with_real_era5.csv' # Try different path if needed, but absolute path is safer
        # Let's just use the path relative to workspace root as I am in the root
        df = pd.read_csv('GeoIDbox/data/processed/sensor_data_with_real_era5.csv')

    print(f"Initial rows: {len(df)}")

    # 2. Calculate ISA Error
    # Assuming 'avg_pressure' is in Pascals and 'avg_altitude' is GNSS altitude in meters
    if 'avg_pressure' not in df.columns or 'avg_altitude' not in df.columns:
        print("Error: Missing 'avg_pressure' or 'avg_altitude' columns.")
        return

    # Calculate H_isa
    df['h_isa'] = calculate_h_isa(df['avg_pressure'])

    # Calculate error: H_isa - H_gps
    # H_gps is avg_altitude
    df['isa_error'] = df['h_isa'] - df['avg_altitude']
    df['abs_isa_error'] = df['isa_error'].abs()

    # 3. Analyze Distribution
    print("\n--- Outlier Analysis ---")
    percentiles = [50, 90, 95, 99]
    percentile_values = np.percentile(df['abs_isa_error'].dropna(), percentiles)

    for p, val in zip(percentiles, percentile_values):
        print(f"{p}th percentile of |error|: {val:.2f} m")

    max_error = df['abs_isa_error'].dropna().max()
    print(f"Max error: {max_error:.2f} m")

    # Suggested threshold
    threshold = 200.0
    print(f"\nUsing threshold: {threshold} m")

    # 4. Filter
    df_filtered = df[df['abs_isa_error'] <= threshold].copy()

    removed_count = len(df) - len(df_filtered)
    removed_percent = (removed_count / len(df)) * 100

    print(f"\nOriginal count: {len(df)}")
    print(f"Filtered count: {len(df_filtered)}")
    print(f"Removed: {removed_count} samples ({removed_percent:.2f}%)")

    # Save filtered data
    print(f"\nSaving filtered data to {output_csv}...")
    # Drop the helper columns before saving to keep it clean, or keep them?
    # User just said "Create a filtered dataframe". Usually good to keep original structure.
    # But usually good to drop temporary calculation columns. I will drop them to maintain schema.
    df_filtered_clean = df_filtered.drop(columns=['h_isa', 'isa_error', 'abs_isa_error'])
    df_filtered_clean.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
