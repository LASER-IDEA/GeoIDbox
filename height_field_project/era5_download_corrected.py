"""
ERA5 data download script for correct date range.

Data collection period: 2025-11-10 to 2025-11-26
Downloads surface and pressure-level data for Shenzhen region.
"""
import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional


def download_era5_surface(
    output_path: str,
    start_date: str = "2025-11-10",
    end_date: str = "2025-11-26",
    area: List[float] = [22.8, 113.8, 22.4, 114.2],
    times: Optional[List[str]] = None
):
    """
    Download ERA5 surface data for the sensor collection period.
    
    Args:
        output_path: Output NetCDF file path
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        area: Bounding box [N, W, S, E]
        times: List of times to download (default: every 6 hours)
    """
    try:
        import cdsapi
    except ImportError:
        print("Error: cdsapi not installed. Install with: pip install cdsapi")
        print("Also configure ~/.cdsapirc with your CDS API credentials")
        return False
    
    if times is None:
        times = ["00:00", "06:00", "12:00", "18:00"]
    
    print(f"Downloading ERA5 surface data...")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Area: {area}")
    print(f"  Times: {times}")
    
    c = cdsapi.Client()
    
    # Use parameter IDs instead of names to avoid ambiguity
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "surface_pressure",           # 134
            "2m_temperature",             # 167
            "2m_dewpoint_temperature",    # 168
            "geopotential",               # 129 (surface geopotential)
        ],
        "year": ["2025"],
        "month": ["11"],
        "day": [f"{d:02d}" for d in range(10, 27)],
        "time": times,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            request,
            output_path
        )
        print(f"Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_era5_pressure_levels(
    output_path: str,
    start_date: str = "2025-11-10",
    end_date: str = "2025-11-26",
    area: List[float] = [22.8, 113.8, 22.4, 114.2],
    pressure_levels: Optional[List[str]] = None,
    times: Optional[List[str]] = None
):
    """
    Download ERA5 pressure-level data for lapse rate calculations.
    
    Args:
        output_path: Output NetCDF file path
        start_date: Start date
        end_date: End date
        area: Bounding box [N, W, S, E]
        pressure_levels: List of pressure levels in hPa
        times: List of times to download
    """
    try:
        import cdsapi
    except ImportError:
        print("Error: cdsapi not installed")
        return False
    
    if pressure_levels is None:
        pressure_levels = ["1000", "950", "900", "850"]
    
    if times is None:
        times = ["00:00", "06:00", "12:00", "18:00"]
    
    print(f"Downloading ERA5 pressure-level data...")
    print(f"  Levels: {pressure_levels} hPa")
    
    c = cdsapi.Client()
    
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "temperature",
            "relative_humidity",
        ],
        "pressure_level": pressure_levels,
        "year": ["2025"],
        "month": ["11"],
        "day": [f"{d:02d}" for d in range(10, 27)],
        "time": times,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    try:
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            request,
            output_path
        )
        print(f"Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 data for correct dates")
    parser.add_argument("--output_dir", type=str, default="data/era5_corrected")
    parser.add_argument("--start_date", type=str, default="2025-11-10")
    parser.add_argument("--end_date", type=str, default="2025-11-26")
    parser.add_argument("--area_n", type=float, default=22.8)
    parser.add_argument("--area_w", type=float, default=113.8)
    parser.add_argument("--area_s", type=float, default=22.4)
    parser.add_argument("--area_e", type=float, default=114.2)
    parser.add_argument("--download_surface", action="store_true", default=True)
    parser.add_argument("--download_pressure", action="store_true", default=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    area = [args.area_n, args.area_w, args.area_s, args.area_e]
    
    success = True
    
    if args.download_surface:
        surface_path = os.path.join(args.output_dir, "era5_surface_2025-11.nc")
        if not os.path.exists(surface_path):
            success = download_era5_surface(surface_path, args.start_date, args.end_date, area)
        else:
            print(f"Surface data already exists: {surface_path}")
    
    if args.download_pressure:
        pressure_path = os.path.join(args.output_dir, "era5_pressure_2025-11.nc")
        if not os.path.exists(pressure_path):
            success = download_era5_pressure_levels(pressure_path, args.start_date, args.end_date, area)
        else:
            print(f"Pressure-level data already exists: {pressure_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
