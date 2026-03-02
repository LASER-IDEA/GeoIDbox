import pandas as pd
import numpy as np
import xarray as xr
from datetime import timezone
import warnings

G0 = 9.80665
R_DRY = 287.05


def _rh_to_specific_humidity(rh_percent, t_k, p_pa):
    """Convert relative humidity to specific humidity."""
    e_sat = 610.94 * np.exp(17.625 * (t_k - 273.15) / (t_k - 30.11))
    e = (rh_percent / 100.0) * e_sat
    q = 0.62198 * e / (p_pa - e)
    return q


def enrich_with_era5(df: pd.DataFrame, nc_path: str) -> pd.DataFrame:
    """
    Enrich input dataframe with ERA5 pressure-level features.
    
    Optimized version using xarray's advanced interpolation.
    """
    if "processed_time" not in df.columns:
        raise ValueError("processed_time column required to align ERA5")

    print(f"Loading ERA5 data from {nc_path}...")
    ds = xr.open_dataset(nc_path)
    
    df = df.copy()
    df["processed_time"] = pd.to_datetime(df["processed_time"], utc=True)
    
    # Convert sensor times to numpy datetime64
    times = df["processed_time"].apply(lambda t: np.datetime64(t.to_pydatetime().replace(tzinfo=timezone.utc))).values
    lats = df["avg_latitude"].values
    lons = df["avg_longitude"].values
    
    print(f"Enriching {len(df)} samples with ERA5 features...")
    
    # Determine coordinate names
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    level_dim = 'pressure_level' if 'pressure_level' in ds.dims else 'level'
    
    # Create output arrays
    n_samples = len(df)
    features = {}
    
    # Process each pressure level
    levels = [1000, 950, 900]
    for level in levels:
        print(f"  Processing {level} hPa level...")
        
        # Get data at this level
        ds_level = ds.sel({level_dim: float(level)}, method='nearest')
        
        # Interpolate for all samples at once using xarray's interp
        # Create a DataArray with sample index as a new dimension
        interp_dict = {
            time_dim: xr.DataArray(times, dims='sample'),
            'latitude': xr.DataArray(lats, dims='sample'),
            'longitude': xr.DataArray(lons, dims='sample'),
        }
        
        # Interpolate temperature
        t_interp = ds_level['t'].interp(interp_dict).values
        features[f'era5_t{level}_k'] = t_interp
        
        # Interpolate relative humidity
        rh_interp = ds_level['r'].interp(interp_dict).values
        features[f'era5_rh{level}'] = rh_interp
        
        # Interpolate geopotential and convert to height
        z_interp = ds_level['z'].interp(interp_dict).values / G0
        features[f'era5_z{level}_m'] = z_interp
        
        # Convert RH to specific humidity
        p_pa = float(level) * 100.0
        q = _rh_to_specific_humidity(rh_interp, t_interp, p_pa)
        features[f'era5_q{level}'] = q
        
        # Compute virtual temperature
        tv = t_interp * (1 + 0.61 * q)
        features[f'era5_tv{level}_k'] = tv
    
    # Compute lapse rates
    print("  Computing lapse rates...")
    t1000 = features['era5_t1000_k']
    t950 = features['era5_t950_k']
    t900 = features['era5_t900_k']
    z1000 = features['era5_z1000_m']
    z950 = features['era5_z950_m']
    z900 = features['era5_z900_m']
    
    # Lapse rate between 1000-950 hPa
    dz_1000_950 = z950 - z1000
    features['era5_lapse_1000_950'] = np.where(
        np.abs(dz_1000_950) > 0.1,
        (t950 - t1000) / dz_1000_950,
        np.nan
    )
    
    # Lapse rate between 950-900 hPa
    dz_950_900 = z900 - z950
    features['era5_lapse_950_900'] = np.where(
        np.abs(dz_950_900) > 0.1,
        (t900 - t950) / dz_950_900,
        np.nan
    )
    
    # Add features to dataframe
    for k, v in features.items():
        df[k] = v
    
    ds.close()
    print(f"✅ ERA5 enrichment complete! Added {len(features)} features.")
    
    return df
