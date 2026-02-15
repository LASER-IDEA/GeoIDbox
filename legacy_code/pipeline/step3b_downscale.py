import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import os

class Downscaler:
    """
    Downscaling module for NWP data using High-Resolution DSM/DTM.
    Handles Horizontal and Vertical interpolation.
    """
    def __init__(self, era5_pl_path, era5_sl_path, dsm_path=None):
        """
        Initialize the downscaler.
        :param era5_pl_path: Path to ERA5 Pressure Level NetCDF.
        :param era5_sl_path: Path to ERA5 Surface Level NetCDF.
        :param dsm_path: Path to DSM/DTM file (optional, can use synthetic).
        """
        self.ds_pl = xr.open_dataset(era5_pl_path)
        self.ds_sl = xr.open_dataset(era5_sl_path)

        # Pre-compute interpolators for speed
        self._prepare_interpolators()

        self.dsm = None
        if dsm_path and os.path.exists(dsm_path):
            # TODO: Load real DSM using rasterio
            pass
        else:
            print("Warning: No DSM file provided, using synthetic DSM generator.")

    def _prepare_interpolators(self):
        """
        Create 3D/2D interpolators for ERA5 variables.
        """
        # ERA5 coordinates (assuming regular grid)
        # Note: ERA5 lat is usually descending (90 -> -90), lon is ascending.
        # We need sorted arrays for RegularGridInterpolator

        self.lats = self.ds_pl['latitude'].values
        self.lons = self.ds_pl['longitude'].values
        self.levels = self.ds_pl['level'].values # Pressure levels in hPa
        self.times = self.ds_pl['time'].values

        # Check sorting
        if self.lats[0] > self.lats[-1]:
            self.lats = self.lats[::-1]
            flip_lat = True
        else:
            flip_lat = False

        # We construct interpolators for the first time step for simplicity in this POC
        # In production, you'd handle time interpolation too.
        t_idx = 0

        def get_data_3d(var_name):
            data = self.ds_pl[var_name].isel(time=t_idx).values
            if flip_lat:
                data = data[:, ::-1, :]
            return data

        # 3D Interpolators (Level, Lat, Lon) -> Value
        # Variables: temperature, specific_humidity, geopotential
        self.interp_t = RegularGridInterpolator((self.levels, self.lats, self.lons), get_data_3d('temperature'))
        self.interp_q = RegularGridInterpolator((self.levels, self.lats, self.lons), get_data_3d('specific_humidity'))
        self.interp_z = RegularGridInterpolator((self.levels, self.lats, self.lons), get_data_3d('geopotential'))

        # Surface Interpolators (Lat, Lon) -> Value
        def get_data_2d(var_name):
            data = self.ds_sl[var_name].isel(time=t_idx).values
            if flip_lat:
                data = data[::-1, :]
            return data

        self.interp_sp = RegularGridInterpolator((self.lats, self.lons), get_data_2d('sp')) # Surface Pressure
        self.interp_t2m = RegularGridInterpolator((self.lats, self.lons), get_data_2d('t2m')) # 2m Temp

    def get_dsm_height(self, lat, lon):
        """
        Get high-resolution terrain height from DSM.
        If no DSM is loaded, generate synthetic city terrain.
        """
        if self.dsm:
            # TODO: Implement rasterio lookup
            return 0
        else:
            # Synthetic City Terrain:
            # Base terrain + random buildings
            # Use deterministic hash of coords for consistency
            np.random.seed(int((lat+lon)*10000) % 100000)
            base_h = 20.0 # Base ground height
            is_building = np.random.rand() > 0.7
            building_h = np.random.uniform(10, 100) if is_building else 0
            return base_h + building_h

    def get_local_roughness(self, lat, lon, radius_m=100):
        """
        Calculate local roughness (std dev of height) around the point.
        """
        # Synthetic roughness
        np.random.seed(int((lat+lon)*10000) % 100000)
        return np.random.uniform(1, 15)

    def downscale_point(self, lat, lon, time=None):
        """
        Perform Downscaling for a specific location.
        Returns: High-res Meteo Variables (P, T, Q, Z) at the surface/drone location.
        """
        # 1. Get High-Res Terrain Height (DSM)
        h_dsm = self.get_dsm_height(lat, lon)

        # 2. Get Coarse Meteo Profiles from ERA5 (Vertical Column)
        # We extract the column at the specific (lat, lon)
        # We can query the 3D interpolator at (all_levels, lat, lon)
        pts = np.array([(lev, lat, lon) for lev in self.levels])

        # T_profile: Temperature at each pressure level
        t_profile = self.interp_t(pts)
        q_profile = self.interp_q(pts)
        z_profile = self.interp_z(pts) # Geopotential (m^2/s^2)
        h_profile = z_profile / 9.80665 # Geopotential Height (m)

        # 3. Vertical Interpolation to DSM Height
        # We need to find P at h_dsm.
        # We use the Hypsometric equation / Log-P interpolation.

        # Find the two levels sandwiching h_dsm
        # Since pressure decreases with height, h increases as level index increases (if levels are 1000->50)
        # Wait, ERA5 levels usually go 50, ..., 1000.
        # So low index = High Altitude. High index = Low Altitude.
        # We want to find i such that h_profile[i] > h_dsm > h_profile[i+1] (or vice versa)

        # Let's sort profile by height (ascending) for np.interp
        sort_idx = np.argsort(h_profile)
        h_sorted = h_profile[sort_idx]
        p_sorted = self.levels[sort_idx] # hPa
        t_sorted = t_profile[sort_idx]
        q_sorted = q_profile[sort_idx]

        # Interpolate Pressure at h_dsm
        # Log-linear interpolation for Pressure: ln(P) is linear with h
        # ln(P) = m * h + c
        # P = exp(interp(h, h_levels, ln(P_levels)))

        p_dsm_hpa = np.exp(np.interp(h_dsm, h_sorted, np.log(p_sorted)))
        t_dsm = np.interp(h_dsm, h_sorted, t_sorted)
        q_dsm = np.interp(h_dsm, h_sorted, q_sorted)

        # Get Surface Pressure from ERA5 (2D) for reference
        p_sfc_era5 = self.interp_sp((lat, lon)) # Pa

        return {
            'lat': lat, 'lon': lon,
            'h_dsm': h_dsm,
            'p_downscaled_pa': p_dsm_hpa * 100.0,
            't_downscaled_k': t_dsm,
            'q_downscaled': q_dsm,
            'roughness': self.get_local_roughness(lat, lon)
        }

if __name__ == "__main__":
    # Test the module
    print("Testing Downscaler...")
    # Generate dummy data first if needed
    if not os.path.exists('data/era5_pl_2024-11-24.nc'):
        from step2_download_era5 import generate_mock_era5_data
        generate_mock_era5_data()

    ds = Downscaler('data/era5_pl_2024-11-24.nc', 'data/era5_sl_2024-11-24.nc')

    # Test point (Shenzhen)
    res = ds.downscale_point(22.5431, 114.0579)
    print("Downscaled Result:", res)
