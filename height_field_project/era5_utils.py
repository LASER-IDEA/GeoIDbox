import pandas as pd
import numpy as np
import xarray as xr
from datetime import timezone

G0 = 9.80665


def _interp_level(ds, var, level, lat, lon, time):
    return (
        ds[var]
        .sel(time=time, method="nearest")
        .sel(level=level, method="nearest")
        .interp(latitude=lat, longitude=lon)
        .item()
    )


def enrich_with_era5(df: pd.DataFrame, nc_path: str) -> pd.DataFrame:
    """
    Enrich input dataframe with ERA5 pressure-level features:
    - temperature and specific humidity at 1000/900 hPa
    - virtual temperature at those levels
    - lapse rate between 1000 and 900 hPa
    - geopotential heights at 1000/900 hPa (meters)
    """
    if "processed_time" not in df.columns:
        raise ValueError("processed_time column required to align ERA5")

    ds = xr.open_dataset(nc_path)
    df = df.copy()
    df["processed_time"] = pd.to_datetime(df["processed_time"], utc=True)

    feats = {
        "era5_t1000_k": [],
        "era5_q1000": [],
        "era5_tv1000_k": [],
        "era5_t900_k": [],
        "era5_q900": [],
        "era5_tv900_k": [],
        "era5_z1000_m": [],
        "era5_z900_m": [],
        "era5_lapse_1000_900": [],
    }

    for _, row in df.iterrows():
        lat = float(row["avg_latitude"])
        lon = float(row["avg_longitude"])
        t = row["processed_time"]
        # ERA5 time is hourly; use nearest
        time_sel = np.datetime64(t.to_pydatetime().replace(tzinfo=timezone.utc))

        try:
            T1000 = _interp_level(ds, "temperature", 1000, lat, lon, time_sel)
            q1000 = _interp_level(ds, "specific_humidity", 1000, lat, lon, time_sel)
            z1000 = _interp_level(ds, "geopotential", 1000, lat, lon, time_sel) / G0
            T900 = _interp_level(ds, "temperature", 900, lat, lon, time_sel)
            q900 = _interp_level(ds, "specific_humidity", 900, lat, lon, time_sel)
            z900 = _interp_level(ds, "geopotential", 900, lat, lon, time_sel) / G0
        except Exception:
            # If interpolation fails, fill NaN
            T1000 = q1000 = z1000 = T900 = q900 = z900 = np.nan

        tv1000 = T1000 * (1 + 0.61 * q1000) if np.isfinite(T1000) else np.nan
        tv900 = T900 * (1 + 0.61 * q900) if np.isfinite(T900) else np.nan
        lapse = np.nan
        if np.isfinite(T1000) and np.isfinite(T900) and np.isfinite(z1000) and np.isfinite(z900) and (z900 - z1000) != 0:
            lapse = (T900 - T1000) / (z900 - z1000)

        feats["era5_t1000_k"].append(T1000)
        feats["era5_q1000"].append(q1000)
        feats["era5_tv1000_k"].append(tv1000)
        feats["era5_t900_k"].append(T900)
        feats["era5_q900"].append(q900)
        feats["era5_tv900_k"].append(tv900)
        feats["era5_z1000_m"].append(z1000)
        feats["era5_z900_m"].append(z900)
        feats["era5_lapse_1000_900"].append(lapse)

    for k, v in feats.items():
        df[k] = v

    return df
