import polars as pl
import os
import time

def aggregate_weekly_files(input_dir, output_dir):
    print(f"Starting aggregation of files in {input_dir}...")
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])

    if not files:
        print("No CSV files found in input directory.")
        return

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".csv", "_1min_agg.csv"))

        print(f"Processing {filename}...")

        # Scan the raw weekly file
        q = (
            pl.scan_csv(file_path)
            .with_columns(
                pl.col("processed_time").str.to_datetime(strict=False) # Auto-infer ISO format
            )
            .filter(
                pl.col("processed_time").is_not_null() &
                (pl.col("latitude") != 0) &
                (pl.col("longitude") != 0) &
                pl.col("latitude").is_not_null() &
                pl.col("longitude").is_not_null()
            )
            .sort("processed_time")
            .group_by_dynamic("processed_time", every="1m", group_by="uid")
            .agg([
                pl.col("uid").count().alias("record_count"),
                pl.col("temperature").mean().alias("avg_temperature"),
                pl.col("humidity").mean().alias("avg_humidity"),
                pl.col("pressure").mean().alias("avg_pressure"),
                pl.col("altitude").mean().alias("avg_altitude"),
                pl.col("height").mean().alias("avg_height"),
                pl.col("vbat").mean().alias("avg_vbat"),
                pl.col("satellites").mean().alias("avg_satellites"),
                pl.col("hdop").mean().alias("avg_hdop"),
                pl.col("latitude").mean().alias("avg_latitude"),
                pl.col("longitude").mean().alias("avg_longitude")
            ])
            .sort(["uid", "processed_time"])
        )

        # Execute and save
        df_agg = q.collect()
        df_agg.write_csv(output_path)
        print(f"  - Saved {os.path.basename(output_path)} ({df_agg.height} rows)")

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "weekly_sensor_data")
    output_dir = os.path.join(script_dir, "weekly_sensor_data_1min")

    if os.path.exists(input_dir):
        aggregate_weekly_files(input_dir, output_dir)
    else:
        print(f"Error: Input directory {input_dir} not found.")
