"""
Corrected physics baseline for barometric altitude conversion.

Key physics relationships:
1. Barometer measures pressure P at sensor location
2. Pressure relates to orthometric height (MSL) via barometric formula
3. MSL converts to HAE via geoid undulation: HAE = MSL + N(φ, λ)

This module implements:
- ISA standard atmosphere model
- Hypsometric equation with virtual temperature correction
- Proper MSL to HAE conversion using EGM2008 geoid
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from height_field_project.geoid_utils import lookup_geoid


# Physical constants
R_DRY_AIR = 287.05  # J/(kg·K), specific gas constant for dry air
G_STANDARD = 9.80665  # m/s², standard gravity
T_ISA_SL = 288.15  # K, ISA sea level temperature
P_ISA_SL = 101325.0  # Pa, ISA sea level pressure
LAPSE_ISA = 0.0065  # K/m, ISA temperature lapse rate


@dataclass
class BarometricParams:
    """Container for barometric baseline parameters."""
    p_ref: float  # Reference pressure (Pa)
    h_ref: float  # Reference height (m, MSL)
    t_ref: float  # Reference temperature (K)
    method: str  # Method used for fitting
    
    # Derived parameters
    scale_height: float = None  # H = R*T/g
    
    def __post_init__(self):
        if self.scale_height is None:
            self.scale_height = R_DRY_AIR * self.t_ref / G_STANDARD


def compute_virtual_temperature(t_celsius: float, rh_percent: float, p_pa: float) -> float:
    """
    Compute virtual temperature accounting for humidity.
    
    Virtual temperature is the temperature that dry air would need to have
    the same density as moist air at the same pressure.
    
    Args:
        t_celsius: Temperature in Celsius
        rh_percent: Relative humidity in percent (0-100)
        p_pa: Pressure in Pascals
    
    Returns:
        Virtual temperature in Kelvin
    """
    t_kelvin = t_celsius + 273.15
    
    # Saturation vapor pressure (Magnus formula)
    # e_sat in Pa
    e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
    
    # Actual vapor pressure
    e = (rh_percent / 100.0) * e_sat
    
    # Mixing ratio
    r = 0.62198 * e / (p_pa - e)
    
    # Virtual temperature
    t_v = t_kelvin * (1 + 0.608 * r)
    
    return t_v


def pressure_to_msl_hypsometric(
    p_obs: float,
    p_ref: float,
    t_v: float,
    h_ref: float = 0.0
) -> float:
    """
    Convert pressure to orthometric height (MSL) using hypsometric equation.
    
    Hypsometric equation:
        H = H_ref + (R_dry * T_v / g) * ln(P_ref / P_obs)
    
    Args:
        p_obs: Observed pressure at sensor (Pa)
        p_ref: Reference pressure (Pa)
        t_v: Virtual temperature (K)
        h_ref: Reference height (m, MSL)
    
    Returns:
        Orthometric height (MSL) in meters
    """
    scale_height = R_DRY_AIR * t_v / G_STANDARD
    h_msl = h_ref + scale_height * np.log(p_ref / p_obs)
    return h_msl


def msl_to_hae(h_msl: float, lat: float, lon: float) -> float:
    """
    Convert MSL (orthometric height) to HAE.
    
    Args:
        h_msl: Height above mean sea level (m)
        lat: Latitude (degrees)
        lon: Longitude (degrees)
    
    Returns:
        Height above WGS-84 ellipsoid (m)
    """
    n = lookup_geoid(lat, lon)
    return h_msl + n


def hae_to_msl(hae: float, lat: float, lon: float) -> float:
    """
    Convert HAE to MSL (orthometric height).
    
    Args:
        hae: Height above WGS-84 ellipsoid (m)
        lat: Latitude (degrees)
        lon: Longitude (degrees)
    
    Returns:
        Height above mean sea level (m)
    """
    n = lookup_geoid(lat, lon)
    return hae - n


def estimate_reference_pressure(
    df: pd.DataFrame,
    method: str = "lowest_altitude"
) -> float:
    """
    Estimate reference pressure (sea level equivalent) from sensor data.
    
    Methods:
    - 'lowest_altitude': Use the lowest altitude sensor as reference
    - 'regression': Fit ln(P) vs H and extrapolate to H=0
    
    Args:
        df: DataFrame with pressure and altitude data
        method: Estimation method
    
    Returns:
        Estimated reference pressure at sea level (Pa)
    """
    if method == "lowest_altitude":
        # Find the lowest altitude sensor
        sensor_stats = df.groupby('uid').agg({
            'avg_altitude': 'mean',
            'avg_pressure': 'mean',
            'avg_temperature': 'mean',
            'avg_humidity': 'mean'
        }).reset_index()
        
        lowest_sensor = sensor_stats.loc[sensor_stats['avg_altitude'].idxmin()]
        p_low = lowest_sensor['avg_pressure']
        h_low = lowest_sensor['avg_altitude']
        t_low = lowest_sensor['avg_temperature']
        rh_low = lowest_sensor['avg_humidity']
        
        # Compute virtual temperature
        t_v = compute_virtual_temperature(t_low, rh_low, p_low)
        
        # Extrapolate to sea level
        scale_height = R_DRY_AIR * t_v / G_STANDARD
        p_ref = p_low * np.exp(h_low / scale_height)
        
        return p_ref
    
    elif method == "regression":
        from sklearn.linear_model import LinearRegression
        
        # Get valid data
        valid = df[['avg_pressure', 'avg_altitude']].dropna()
        mask = (valid['avg_pressure'] > 90000) & (valid['avg_pressure'] < 110000)
        valid = valid[mask]
        
        if len(valid) < 10:
            raise ValueError("Insufficient data for regression")
        
        # Fit ln(P) = a*H + b
        X = valid[['avg_altitude']].values.reshape(-1, 1)
        y = np.log(valid['avg_pressure'].values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Extrapolate to H=0
        intercept = model.intercept_
        p_ref = np.exp(intercept)
        
        return p_ref
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_physics_baseline(
    df: pd.DataFrame,
    p_ref: Optional[float] = None,
    h_ref: float = 0.0,
    t_ref_method: str = "mean",
    p_ref_method: str = "auto",
    convert_to_hae: bool = False
) -> Tuple[pd.DataFrame, BarometricParams]:
    """
    Compute physics baseline height (HAE) from barometric measurements.
    
    This is the CORRECTED implementation that:
    1. Uses hypsometric equation to get MSL from pressure
    2. Applies geoid undulation to convert MSL → HAE
    
    Args:
        df: DataFrame with columns: avg_pressure, avg_temperature, avg_humidity, avg_latitude, avg_longitude
        p_ref: Reference pressure (Pa). If None, uses ISA sea level pressure
        h_ref: Reference height (m, MSL). Default 0 (sea level)
        t_ref_method: Method for reference temperature ('mean', 'isa', or float value)
    
    Returns:
        Tuple of (DataFrame with added columns, BarometricParams)
    """
    df = df.copy()
    
    # Determine reference pressure
    if p_ref is None:
        if p_ref_method == "auto":
            # Auto-estimate from data
            p_ref = estimate_reference_pressure(df, method="lowest_altitude")
            print(f"Auto-estimated P_ref: {p_ref:.2f} Pa")
        elif p_ref_method == "isa":
            p_ref = P_ISA_SL
        else:
            p_ref = estimate_reference_pressure(df, method=p_ref_method)
    
    # Determine reference temperature
    if t_ref_method == "mean":
        t_ref = df["avg_temperature"].mean() + 273.15
    elif t_ref_method == "isa":
        t_ref = T_ISA_SL
    else:
        t_ref = float(t_ref_method)
    
    # Compute virtual temperature for each observation
    df["t_virtual"] = df.apply(
        lambda row: compute_virtual_temperature(
            row["avg_temperature"],
            row["avg_humidity"],
            row["avg_pressure"]
        ),
        axis=1
    )
    
    # Compute MSL from pressure using hypsometric equation
    df["h_msl"] = df.apply(
        lambda row: pressure_to_msl_hypsometric(
            row["avg_pressure"],
            p_ref,
            row["t_virtual"],
            h_ref
        ),
        axis=1
    )
    
    # Optionally convert MSL to HAE using geoid undulation
    if convert_to_hae:
        df["h_phys_hae"] = df.apply(
            lambda row: msl_to_hae(
                row["h_msl"],
                row["avg_latitude"],
                row["avg_longitude"]
            ),
            axis=1
        )
    else:
        # Assume GNSS altitude is already MSL (orthometric height)
        df["h_phys_hae"] = df["h_msl"]
    
    # Create parameter container
    params = BarometricParams(
        p_ref=p_ref,
        h_ref=h_ref,
        t_ref=t_ref,
        method=f"hypsometric_{t_ref_method}"
    )
    
    return df, params


def compute_pressure_correction_residual(
    df: pd.DataFrame,
    p_correction: np.ndarray
) -> pd.DataFrame:
    """
    Apply pressure correction and recompute height.
    
    This is used by the PINN to apply the learned pressure correction field.
    
    Args:
        df: DataFrame with physics baseline columns
        p_correction: Pressure correction values (Pa) to add to observed pressure
    
    Returns:
        DataFrame with corrected height
    """
    df = df.copy()
    
    # Apply pressure correction
    df["p_corrected"] = df["avg_pressure"] + p_correction
    
    # Recompute MSL with corrected pressure
    df["h_msl_corrected"] = df.apply(
        lambda row: pressure_to_msl_hypsometric(
            row["p_corrected"],
            row.get("p_ref", P_ISA_SL),
            row["t_virtual"],
            row.get("h_ref", 0.0)
        ),
        axis=1
    )
    
    # Convert to HAE
    df["h_pred_hae"] = df.apply(
        lambda row: msl_to_hae(
            row["h_msl_corrected"],
            row["avg_latitude"],
            row["avg_longitude"]
        ),
        axis=1
    )
    
    return df


def fit_barometric_baseline_legacy(
    df: pd.DataFrame,
    pressure_col: str = "avg_pressure",
    altitude_col: str = "avg_altitude",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    LEGACY: Original barometric baseline fitting (ln(P) = a*h + b).
    
    Kept for backward compatibility. New code should use compute_physics_baseline().
    
    Note: This method assumes pressure directly relates to HAE, which is incorrect.
    It should only be used for comparison with old results.
    """
    from sklearn.linear_model import LinearRegression
    
    valid = df[[pressure_col, altitude_col]].dropna()
    mask = (valid[pressure_col] > 90000) & (valid[pressure_col] < 110000)
    valid = valid[mask]
    if len(valid) < 10:
        raise ValueError("有效压力样本过少，无法拟合物理基线")

    X = valid[[altitude_col]].values.reshape(-1, 1)
    y = np.log(valid[pressure_col].values)
    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    Hs = -1.0 / slope
    P0 = np.exp(intercept)

    df = df.copy()
    df["h_phys_m"] = -Hs * (np.log(df[pressure_col]) - np.log(P0))

    params = {"Hs_m": float(Hs), "P0_Pa": float(P0), "slope": float(slope), "intercept": float(intercept)}
    return df, params
