import polars as pl
import os
import time
from shapely import wkb

def parse_wkb_hex(hex_str):
    """Parses a WKB hex string and returns (lon, lat). Returns (None, None) on error."""
    try:
        if not hex_str:
            return None
        binary = bytes.fromhex(hex_str)
        point = wkb.loads(binary)
        return (point.x, point.y)
    except Exception:
        return None

def split_by_week(input_path, output_dir):
    print(f"Starting weekly split of {input_path}...")
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # We need to extract Lat/Lon for raw data (or aggregated).
    # User said "split the raw data per week", but also "remember to conver lat and long".
    # Raw data is 3.7GB. Splitting it into weeks without aggregation is fine,
    # but converting Lat/Lon for EVERY row (20M rows) in Python is SLOW.
    # However, Polars can handle large writes.

    # Strategy:
    # 1. Scan CSV.
    # 2. Add 'week_start' column.
    # 3. Iterate over weeks (filter -> collect -> parse lat/lon -> save).
    # This avoids loading 3.7GB into RAM + Python objects.

    # Note: Polars `dt.week()` gives week number. `dt.truncate("1w")` gives week start date.

    q = (
        pl.scan_csv(input_path)
        .filter(pl.col("is_delete") == 0)
        .with_columns(
            pl.col("processed_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        )
        .filter(pl.col("processed_time").is_not_null())
        .with_columns(
            # Truncate to week start (Monday)
            pl.col("processed_time").dt.truncate("1w").dt.date().alias("week_start")
        )
    )

    # Get unique weeks first to iterate
    # We can do a quick aggregation to find unique weeks
    print("Scanning for unique weeks...")
    weeks_df = q.select("week_start").unique().collect().sort("week_start")
    unique_weeks = weeks_df["week_start"].to_list()

    print(f"Found {len(unique_weeks)} weeks.")

    for week_start in unique_weeks:
        week_str = str(week_start)
        print(f"Processing week: {week_str}...")

        # Filter for this week
        # We select relevant columns to keep file size manageable, or keep all?
        # User said "raw data", so likely wants most columns.
        # We MUST convert location.

        df_week = (
            q.filter(pl.col("week_start") == week_start)
             .collect()
        )

        # Parse Lat/Lon for this chunk
        # Converting WKB hex to lat/lon in Python
        # This part is the bottleneck.
        # If a week has 2M rows, this loop runs 2M times.
        # Optimization: If 'location' repeats, cache it?
        # Sensors are static? Probably.

        print(f"  - Parsing locations for {df_week.height} rows...")

        # To speed up, let's see if we can use map_elements with return_dtype or similar,
        # but a simple list comp is often fastest for mixed types in Python.

        hex_data = df_week["location"].to_list()

        # Check distinct locations to see if caching helps
        # (Optional optimization, maybe premature)

        coords = [parse_wkb_hex(x) for x in hex_data]

        lons = [c[0] if c else None for c in coords]
        lats = [c[1] if c else None for c in coords]

        df_week = df_week.with_columns([
            pl.Series("longitude", lons),
            pl.Series("latitude", lats)
        ])

        # Drop WKB location and metadata we don't need?
        # User asked for "split raw data", implying keeping original content but adding lat/lon.
        # I'll drop the WKB 'location' to save space as 'lat/lon' replaces it.
        df_week = df_week.drop(["location", "week_start"])

        filename = f"sensor_data_week_{week_str}.csv"
        file_path = os.path.join(output_dir, filename)

        df_week.write_csv(file_path)
        print(f"  - Saved {filename}")

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "rt_alg_env_sensor_processed.csv")
    output_dir = os.path.join(script_dir, "weekly_sensor_data")

    if os.path.exists(input_file):
        split_by_week(input_file, output_dir)
    else:
        print(f"Error: File {input_file} not found.")
