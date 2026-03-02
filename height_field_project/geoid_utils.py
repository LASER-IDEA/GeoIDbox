"""
Geoid undulation utilities for MSL to HAE conversion.
Uses EGM2008 geoid model.
"""
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from typing import Union


class GeoidInterpolator:
    """
    Interpolator for EGM2008 geoid undulation values.
    
    The geoid undulation N(φ, λ) represents the separation between:
    - MSL (Mean Sea Level, orthometric height)
    - WGS-84 ellipsoid (HAE - Height Above Ellipsoid)
    
    Relationship: HAE = MSL + N(φ, λ)
    """
    
    def __init__(self, geoid_path: str = None):
        """
        Initialize geoid interpolator.
        
        Args:
            geoid_path: Path to EGM2008 .pgm file. If None, searches in data/geoids/
        """
        if geoid_path is None:
            # Search relative to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            geoid_path = os.path.join(project_root, "data", "geoids", "egm2008-5.pgm")
        
        if not os.path.exists(geoid_path):
            raise FileNotFoundError(f"Geoid file not found: {geoid_path}")
        
        self.geoid_path = geoid_path
        self._load_geoid()
    
    def _load_geoid(self):
        """Load EGM2008 geoid data from PGM file."""
        # Read PGM header
        with open(self.geoid_path, 'rb') as f:
            # Check magic number
            magic = f.readline().strip()
            if magic != b'P5':
                raise ValueError(f"Unsupported PGM format: {magic}")
            
            # Skip comments
            while True:
                line = f.readline()
                if not line.startswith(b'#'):
                    break
            
            # Parse dimensions
            dims = line.strip().split()
            width = int(dims[0])
            height = int(dims[1])
            
            # Parse max value
            maxval = int(f.readline().strip())
            
            # Read binary data
            data = np.fromfile(f, dtype=np.uint16 if maxval > 255 else np.uint8)
            data = data.reshape((height, width))
            
            # Convert to meters (EGM2008 data is in meters * scale factor)
            # Standard EGM2008 PGM uses 0.01m resolution for 16-bit data
            if maxval > 255:
                self.geoid_grid = data.astype(np.float32) * 0.01
            else:
                self.geoid_grid = data.astype(np.float32)
        
        # EGM2008-5 grid parameters
        # Grid spans: longitude 0 to 360, latitude 90 to -90
        self.nlat, self.nlon = self.geoid_grid.shape
        
        # Create coordinate grids
        # Latitude: 90 to -90 (descending)
        self.lats = np.linspace(90, -90, self.nlat)
        # Longitude: 0 to 360 (ascending)
        self.lons = np.linspace(0, 360, self.nlon)
        
        # Create interpolator
        # Note: RegularGridInterpolator expects ascending coordinates
        self.interpolator = RegularGridInterpolator(
            (self.lats, self.lons),
            self.geoid_grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    def lookup(self, lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Look up geoid undulation N(φ, λ) at given coordinates.
        
        Args:
            lat: Latitude in degrees (-90 to 90)
            lon: Longitude in degrees (-180 to 180 or 0 to 360)
        
        Returns:
            Geoid undulation in meters. Positive means geoid is above ellipsoid.
        """
        # Convert to numpy arrays
        lat = np.atleast_1d(lat).astype(np.float64)
        lon = np.atleast_1d(lon).astype(np.float64)
        
        # Normalize longitude to 0-360 range
        lon = lon % 360
        
        # Create query points
        points = np.stack([lat, lon], axis=-1)
        
        # Interpolate
        result = self.interpolator(points)
        
        # Return scalar if single point
        if result.size == 1:
            return float(result[0])
        return result
    
    def msl_to_hae(self, msl: Union[float, np.ndarray], lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert orthometric height (MSL) to HAE.
        
        Args:
            msl: Height above mean sea level (orthometric height) in meters
            lat: Latitude in degrees
            lon: Longitude in degrees
        
        Returns:
            Height above WGS-84 ellipsoid (HAE) in meters
        """
        n = self.lookup(lat, lon)
        return msl + n
    
    def hae_to_msl(self, hae: Union[float, np.ndarray], lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert HAE to orthometric height (MSL).
        
        Args:
            hae: Height above WGS-84 ellipsoid in meters
            lat: Latitude in degrees
            lon: Longitude in degrees
        
        Returns:
            Height above mean sea level (orthometric height) in meters
        """
        n = self.lookup(lat, lon)
        return hae - n


# Global singleton instance
_geoid_interp = None


def get_geoid_interpolator() -> GeoidInterpolator:
    """Get or create global geoid interpolator instance."""
    global _geoid_interp
    if _geoid_interp is None:
        _geoid_interp = GeoidInterpolator()
    return _geoid_interp


def lookup_geoid(lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convenience function to lookup geoid undulation."""
    return get_geoid_interpolator().lookup(lat, lon)
