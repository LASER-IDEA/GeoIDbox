import pandas as pd
import numpy as np

# Path to the dataset
file_path = "GeoIDbox/data/processed/sensor_data_with_real_era5.csv"

# Load the CSV
print(f"Loading data from {file_path}...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}. Trying absolute path...")
    file_path = "/data/home/huxiao/workspace/geobox/GeoIDbox/data/processed/sensor_data_with_real_era5.csv"
    df = pd.read_csv(file_path)

# Convert processed_time to datetime
df['processed_time'] = pd.to_datetime(df['processed_time'])

print(f"Total rows: {len(df)}")
print(f"Calculating statistics for each UID...")

# Group by UID
grouped = df.groupby('uid')

results = []

for uid, group in grouped:
    # 1. Stationary Check (Standard Deviation)
    lat_std = group['avg_latitude'].std()
    lon_std = group['avg_longitude'].std()
    alt_std = group['avg_altitude'].std()

    # Combined spatial stability metric (simple sum of stds, heavily weighting lat/lon)
    # Lat/Lon degrees are roughly 111km, so small changes are big distances.
    # But usually we look at small scale movements.
    # Let's just store the values.

    # 2. Continuous / Data Volume
    count = len(group)
    min_time = group['processed_time'].min()
    max_time = group['processed_time'].max()
    time_range_days = (max_time - min_time).total_seconds() / (24 * 3600)

    # 3. Altitude (Mean)
    mean_alt = group['avg_altitude'].mean()

    results.append({
        'uid': uid,
        'count': count,
        'lat_std': lat_std,
        'lon_std': lon_std,
        'alt_std': alt_std,
        'time_range_days': time_range_days,
        'mean_alt': mean_alt,
        'min_time': min_time,
        'max_time': max_time
    })

results_df = pd.DataFrame(results)

# Filter for "good" candidates
# Criteria 1: Stability. Let's look for very low lat/lon std.
# A stationary sensor should have lat/lon std due to GPS noise, but it should be small.
# 1e-5 degrees is approx 1 meter.

# Let's clean up NaNs if any (e.g. single sample groups have NaN std)
results_df = results_df.dropna()

# We want high count, high time_range, low lat_std, low lon_std.
# Let's sort by stability first, then look at count.

# Define a stability score. Lower is better.
# We normalize the stds to give them equal weight? Or just look at lat/lon.
# Let's just sort by lat_std + lon_std (spatial stability).
results_df['spatial_instability'] = results_df['lat_std'] + results_df['lon_std']

# Filter out sensors with very little data (e.g. less than 1 day or less than 10 samples)
min_samples = 50
min_days = 1.0

candidates = results_df[
    (results_df['count'] >= min_samples) &
    (results_df['time_range_days'] >= min_days)
].copy()

# Sort by stability (most stable first)
candidates = candidates.sort_values(by='spatial_instability', ascending=True)

print("\n--- Top Candidates (Most Stable) ---")
print(candidates[['uid', 'count', 'time_range_days', 'lat_std', 'lon_std', 'alt_std', 'mean_alt']].head(10))

# Also, we might want the ones with the MOST data that are "stable enough".
# Let's define "stable enough" as lat_std < 1e-4 and lon_std < 1e-4 (approx 10m noise).
stable_enough = candidates[
    (candidates['lat_std'] < 1e-4) &
    (candidates['lon_std'] < 1e-4)
].copy()

if not stable_enough.empty:
    print("\n--- Longest Duration Stable Candidates ---")
    longest_stable = stable_enough.sort_values(by='time_range_days', ascending=False)
    print(longest_stable[['uid', 'count', 'time_range_days', 'lat_std', 'lon_std', 'alt_std', 'mean_alt']].head(5))

    print("\n--- Best Reference Station Recommendation ---")
    # Picking the top from longest stable as the best ref station usually
    best = longest_stable.iloc[0]
    print(f"UID: {best['uid']}")
    print(f"  Duration: {best['time_range_days']:.2f} days")
    print(f"  Samples: {best['count']}")
    print(f"  Avg Alt: {best['mean_alt']:.2f} m")
    print(f"  Lat Std: {best['lat_std']:.2e}")
    print(f"  Lon Std: {best['lon_std']:.2e}")
else:
    print("\nNo candidates met the 'stable enough' criteria.")
