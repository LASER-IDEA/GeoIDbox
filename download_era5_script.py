#!/usr/bin/env python3
"""
Download ERA5 data for the correct dates (Nov 10-26, 2025).
"""
import sys
sys.path.insert(0, '/data/home/huxiao/workspace/GeoIDbox')

from height_field_project.era5_download_corrected import download_era5_surface, download_era5_pressure_levels

# Download surface data
print("="*60)
print("Downloading ERA5 Surface Data")
print("="*60)
result1 = download_era5_surface(
    output_path="/data/home/huxiao/workspace/GeoIDbox/data/era5_corrected/era5_surface_2025-11.nc",
    start_date="2025-11-10",
    end_date="2025-11-26",
    area=[22.8, 113.8, 22.4, 114.2],
    times=["00:00", "06:00", "12:00", "18:00"]
)

# Download pressure level data  
print("\n" + "="*60)
print("Downloading ERA5 Pressure Level Data")
print("="*60)
result2 = download_era5_pressure_levels(
    output_path="/data/home/huxiao/workspace/GeoIDbox/data/era5_corrected/era5_pressure_2025-11.nc",
    start_date="2025-11-10",
    end_date="2025-11-26",
    area=[22.8, 113.8, 22.4, 114.2],
    pressure_levels=["1000", "950", "900", "850"],
    times=["00:00", "06:00", "12:00", "18:00"]
)

print("\n" + "="*60)
print(f"Surface download: {'SUCCESS' if result1 else 'FAILED'}")
print(f"Pressure download: {'SUCCESS' if result2 else 'FAILED'}")
print("="*60)
