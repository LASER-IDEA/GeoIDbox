"""
ERA5 pressure-level downloader (NetCDF) for macro weather context.
Requires CDS API key configured in ~/.cdsapirc
"""
import argparse
import cdsapi


def parse_area(area_str: str):
    """
    Parse area as N,W,S,E (comma or space separated).
    """
    parts = area_str.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("area must have four numbers: N W S E")
    return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]


def download_era5_pressure_levels(date: str, time_utc: str, area, output: str):
    """
    Download ERA5 pressure levels (geopotential, specific humidity, temperature) to NetCDF.
    """
    client = cdsapi.Client()
    req = {
        "product_type": "reanalysis",
        "data_format": "netcdf",
        "variable": ["geopotential", "specific_humidity", "temperature"],
        "pressure_level": [
            "50",
            "100",
            "150",
            "200",
            "250",
            "300",
            "350",
            "400",
            "450",
            "500",
            "550",
            "600",
            "650",
            "700",
            "750",
            "800",
            "850",
            "900",
            "950",
            "1000",
        ],
        "year": date.split("-")[0],
        "month": date.split("-")[1],
        "day": date.split("-")[2],
        "time": time_utc,
        "area": area,  # N,W,S,E
    }
    print(f"Requesting ERA5 for {date} {time_utc} UTC, area {area} -> {output}")
    client.retrieve("reanalysis-era5-pressure-levels", req, output)
    print("Download completed:", output)


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 pressure levels (NetCDF)")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--time", required=True, help="HH:MM or HH (UTC)")
    parser.add_argument(
        "--area",
        required=True,
        help="N,W,S,E (e.g., '22.8,113.8,22.4,114.2' for Shenzhen region)",
    )
    parser.add_argument("--output", default="era5_pl.nc", help="output NetCDF path")
    args = parser.parse_args()

    area = parse_area(args.area)
    download_era5_pressure_levels(args.date, args.time, area, args.output)


if __name__ == "__main__":
    main()
