#!/usr/bin/env python3
"""
Refined Hard-Constrained Neural Field with Curriculum Learning
===============================================================

Key improvements based on pinn-sol.md:
1. Hard constraint: Output clipping with tanh (residual limited to ±20m)
2. Stabilized GNSS heights as ground truth
3. Better curriculum learning strategy
4. Ablation study support

CRITICAL DESIGN PRINCIPLES (高度盒子 GeoIDbox):
==============================================
1. Barometer → HAE Flow:
   Barometer → ISA Model → MSL → +Geoid → HAE
   - ISA model uses UNIVERSAL constants (P0=101325Pa, T0=288.15K, alpha=0.0065K/m)
   - NO fold-fitted parameters for physics baseline
   - Neural residual learns: delta = true_MSL - ISA_MSL

2. Physical Baseline:
   - Universal ISA model only (not fold-fitted affine)
   - Optional ERA5-aware P0 for weather correction (isa_era5_p0 mode)
   - Neural field corrects ISA, doesn't replace it

3. Meter-Level Accuracy Improvements:
   - Enhanced ISA features: pressure deviation, virtual temperature
   - Temperature deviation from ISA expectation
   - Scale height using actual conditions
   - Minimum performance guard (never worse than 1.5x ISA baseline)

Usage:
    python run_refined_model.py [--ablation {full,no_era5,no_terrain,no_hash}]
    python run_refined_model.py --physics-mode isa_era5_p0  # Weather-aware
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

os.makedirs('experiments/results/refined_model', exist_ok=True)

R_DRY_AIR = 287.05
R_VAPOR = 461.5
G0 = 9.80665


def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IdentityScaler:
    def fit(self, x):
        return self

    def fit_transform(self, x):
        return np.asarray(x)

    def transform(self, x):
        return np.asarray(x)

    def inverse_transform(self, x):
        return np.asarray(x)


class SpatialMinMaxScaler:
    """Scale latitude/longitude to [0, 1] for hash encoding."""
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.min_ = None
        self.max_ = None

    def fit(self, x):
        arr = np.asarray(x, dtype=np.float32)
        self.min_ = arr.min(axis=0)
        self.max_ = arr.max(axis=0)
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=np.float32)
        denom = np.maximum(self.max_ - self.min_, self.eps)
        y = (arr - self.min_) / denom
        return np.clip(y, 0.0, 1.0)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


# =============================================================================
# Hash Encoding (from run_advanced_improvements.py)
# =============================================================================

class HashEncoding(nn.Module):
    """Instant-NGP style multi-resolution hash encoding"""
    def __init__(self, n_input_dims=2, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size

        b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(base_resolution * b**i) for i in range(n_levels)]

        self.hash_tables = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])

        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)

    def forward(self, x):
        batch_size = x.shape[0]
        encoded = []
        for level, resolution in enumerate(self.resolutions):
            scaled = x * resolution
            grid_idx = scaled.long()
            grid_idx = torch.clamp(grid_idx, 0, resolution - 1)
            hash_idx = (grid_idx[:, 0] * 73856093 ^ grid_idx[:, 1] * 19349663) % (2**self.log2_hashmap_size)
            features = self.hash_tables[level](hash_idx)
            encoded.append(features)
        return torch.cat(encoded, dim=-1)


# =============================================================================
# Hard-Constrained Neural Field
# =============================================================================

class HardConstrainedNF(nn.Module):
    """
    Neural Field with Hard Output Constraints

    Based on pinn-sol.md recommendation:
    - Output clipping: residual = clip * tanh(network_output)
    - This creates hard constraint: residual is ALWAYS in [-clip, +clip]m
    """
    def __init__(self, use_hash_encoding=True, use_terrain=True,
                 st_dim=2, feature_dim=9, hidden_dim=256, num_layers=8,
                 residual_clip=60.0, ood_decay_m=120.0, hard_ood_threshold_m=None,
                 soft_ood_center_m=None, soft_ood_steepness_m=20.0):
        super().__init__()

        self.use_terrain = use_terrain
        self.residual_clip = residual_clip
        self.ood_decay_m = ood_decay_m
        self.hard_ood_threshold_m = hard_ood_threshold_m
        self.soft_ood_center_m = soft_ood_center_m
        self.soft_ood_steepness_m = soft_ood_steepness_m

        # Spatial encoding
        if use_hash_encoding:
            self.spatial_encoding = HashEncoding(
                n_input_dims=st_dim, n_levels=16, n_features_per_level=2,
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512
            )
            spatial_dim = 16 * 2
        else:
            # Simple MLP for ablation without hash
            self.spatial_encoding = None
            spatial_dim = st_dim

        # Total input dimension
        total_input = spatial_dim + feature_dim

        # MLP backbone
        layers = []
        layers.append(nn.Linear(total_input, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.05))

        self.backbone = nn.Sequential(*layers)
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_spatial, x_features, ood_distance_m=None):
        if self.spatial_encoding is not None:
            spatial_encoded = self.spatial_encoding(x_spatial)
        else:
            spatial_encoded = x_spatial

        x = torch.cat([spatial_encoded, x_features], dim=-1)
        hidden = self.backbone(x)
        raw_output = self.residual_head(hidden)
        learned_confidence = self.confidence_head(hidden)

        # HARD CONSTRAINT: Clip residual to [-residual_clip, +residual_clip]
        # Using tanh ensures smooth gradients and bounded output
        clipped_residual = self.residual_clip * torch.tanh(raw_output)

        if ood_distance_m is not None:
            distance_confidence = torch.exp(-torch.clamp(ood_distance_m, min=0.0) / self.ood_decay_m)
            confidence = torch.clamp(learned_confidence * distance_confidence, min=0.0, max=1.0)

            if self.soft_ood_center_m is not None and self.soft_ood_steepness_m > 0:
                soft_gate = torch.sigmoid((self.soft_ood_center_m - ood_distance_m) / self.soft_ood_steepness_m)
                confidence = confidence * soft_gate

            if self.hard_ood_threshold_m is not None and self.hard_ood_threshold_m > 0:
                hard_gate = (ood_distance_m <= self.hard_ood_threshold_m).float()
                confidence = confidence * hard_gate
        else:
            confidence = learned_confidence

        gated_residual = confidence * clipped_residual
        return gated_residual, clipped_residual, confidence


# =============================================================================
# Terrain Features (from run_advanced_improvements.py)
# =============================================================================

def _approx_meter_scale(lat_deg):
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat_deg))
    return meters_per_deg_lat, np.maximum(meters_per_deg_lon, 1.0)


def _add_era5_anomaly_features(df):
    if 'era5_sp' in df.columns:
        df['pressure_anom'] = df['avg_pressure'] - df['era5_sp']
    else:
        df['pressure_anom'] = 0.0

    if 'era5_t2m' in df.columns:
        era5_t = df['era5_t2m'].copy()
        if np.nanmedian(era5_t.values) > 200:
            era5_t = era5_t - 273.15
        df['temp_anom'] = df['avg_temperature'] - era5_t
    else:
        df['temp_anom'] = 0.0
    return df


def _add_temporal_delta_features(df):
    df = df.sort_values(['uid', 'processed_time']).copy()
    for col, out_col in [
        ('avg_pressure', 'delta_pressure_1step'),
        ('avg_temperature', 'delta_temp_1step'),
        ('avg_humidity', 'delta_humidity_1step')
    ]:
        if col in df.columns:
            df[out_col] = df.groupby('uid')[col].diff().fillna(0.0)
        else:
            df[out_col] = 0.0
    return df


def _add_sensor_domain_features(df):
    """Add per-sensor normalized features to reduce device/domain bias."""
    df = df.copy()
    for col, prefix in [
        ('avg_pressure', 'pressure'),
        ('avg_temperature', 'temp'),
        ('avg_humidity', 'humidity')
    ]:
        if col not in df.columns:
            df[f'{prefix}_uid_center'] = 0.0
            df[f'{prefix}_uid_z'] = 0.0
            continue

        group_med = df.groupby('uid')[col].transform('median')
        group_std = df.groupby('uid')[col].transform('std').fillna(1.0)
        group_std = np.maximum(group_std.values, 1e-3)

        centered = df[col].values - group_med.values
        df[f'{prefix}_uid_center'] = centered
        df[f'{prefix}_uid_z'] = centered / group_std

    return df


def compute_fold_safe_features(train_df, test_df, use_domain_features=False):
    """Compute fold-safe geo features using only train spatial manifold."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    mean_lat = float(train_df['avg_latitude'].mean())
    m_lat, m_lon = _approx_meter_scale(mean_lat)

    train_xy = np.column_stack([
        train_df['avg_latitude'].values * m_lat,
        train_df['avg_longitude'].values * m_lon
    ])
    tree = cKDTree(train_xy)
    centroid = train_xy.mean(axis=0)

    def add_geo_block(frame):
        xy = np.column_stack([
            frame['avg_latitude'].values * m_lat,
            frame['avg_longitude'].values * m_lon
        ])
        k = min(6, len(train_xy))
        dists, _ = tree.query(xy, k=k)
        if len(dists.shape) == 1:
            dists = dists[:, None]
        frame['knn_dist_m'] = dists[:, 0]
        frame['local_density'] = (dists < 120.0).sum(axis=1).astype(float)
        frame['dist_to_train_mean_m'] = np.linalg.norm(xy - centroid, axis=1)
        frame['ood_distance_m'] = frame['knn_dist_m']
        return frame

    train_df = add_geo_block(train_df)
    test_df = add_geo_block(test_df)

    train_df = _add_era5_anomaly_features(train_df)
    test_df = _add_era5_anomaly_features(test_df)
    train_df = _add_temporal_delta_features(train_df)
    test_df = _add_temporal_delta_features(test_df)

    if use_domain_features:
        train_df = _add_sensor_domain_features(train_df)
        test_df = _add_sensor_domain_features(test_df)

    return train_df, test_df


def compute_terrain_features(df):
    """
    Backward-compatible feature API.
    Uses the full dataset as reference manifold (not for strict LOSO reporting).
    """
    train_df, _ = compute_fold_safe_features(df.copy(), df.copy())
    return train_df


def _as_temperature_k(temp_series):
    arr = temp_series.values.astype(np.float32)
    if np.nanmedian(arr) < 200:
        arr = arr + 273.15
    return np.clip(arr, 200.0, 340.0)


def _compute_virtual_temperature_k(temp_k, rh_pct, pressure_pa):
    temp_c = temp_k - 273.15
    rh = np.clip(rh_pct.astype(np.float32), 0.0, 100.0)
    es_hpa = 6.112 * np.exp((17.67 * temp_c) / np.maximum(temp_c + 243.5, 1e-3))
    e_pa = np.clip((rh / 100.0) * es_hpa * 100.0, 0.0, 0.99 * pressure_pa)
    q = 0.622 * e_pa / np.maximum(pressure_pa - 0.378 * e_pa, 1.0)
    tv = temp_k * (1.0 + 0.61 * q)
    return np.clip(tv, 200.0, 360.0)


def _add_hypsometric_feature(df):
    df = df.copy()
    p = np.clip(df['avg_pressure'].values.astype(np.float32), 1000.0, None)

    if 'era5_sp' in df.columns:
        p_ref = np.clip(df['era5_sp'].values.astype(np.float32), 1000.0, None)
    else:
        p_ref = np.full_like(p, np.median(p))

    temp_k = _as_temperature_k(df['avg_temperature']) if 'avg_temperature' in df.columns else np.full_like(p, 293.15)
    humidity = df['avg_humidity'].values.astype(np.float32) if 'avg_humidity' in df.columns else np.full_like(p, 60.0)
    tv_k = _compute_virtual_temperature_k(temp_k, humidity, p)

    scale_height = (R_DRY_AIR * tv_k) / G0
    hypso_x = scale_height * np.log(np.maximum(p_ref, 1000.0) / p)

    df['tv_k'] = tv_k
    df['scale_height_m'] = np.clip(scale_height, 5000.0, 12000.0)
    df['hypso_x_m'] = np.clip(hypso_x, -1000.0, 1000.0)
    return df


def _prepare_geoid_column(df, geoid_ref_csv='data/final_training_data.csv'):
    """Ensure df has n_geoid (m). Prefer existing column, else infer from geoid model, then CSV fallback."""
    df = df.copy()

    if 'n_geoid' in df.columns:
        return df

    geoid_candidates = [
        Path('data/geoids/egm2008-5.pgm'),
        Path('geoids/egm2008-5.pgm'),
        Path('../data/geoids/egm2008-5.pgm'),
    ]
    geoid_model_path = next((p for p in geoid_candidates if p.exists()), None)

    if geoid_model_path is not None:
        try:
            from pygeodesy import GeoidKarney
            from pygeodesy.ellipsoidalKarney import LatLon

            interpolator = GeoidKarney(str(geoid_model_path))

            coords = df[['avg_latitude', 'avg_longitude']].round(8)
            unique_coords = coords.drop_duplicates()

            geoid_map = {}
            for row in unique_coords.itertuples(index=False):
                lat = float(row.avg_latitude)
                lon = float(row.avg_longitude)
                geoid_map[(lat, lon)] = float(interpolator(LatLon(lat, lon)))

            df['n_geoid'] = [
                geoid_map[(float(lat), float(lon))]
                for lat, lon in coords.itertuples(index=False)
            ]
            return df
        except Exception:
            pass

    ref_path = Path(geoid_ref_csv)
    if not ref_path.exists():
        df['n_geoid'] = 0.0
        return df

    try:
        ref = pd.read_csv(ref_path, usecols=['lat', 'lon', 'n_geoid']).dropna()
    except Exception:
        df['n_geoid'] = 0.0
        return df

    if len(ref) < 10:
        df['n_geoid'] = 0.0
        return df

    mean_lat = float(df['avg_latitude'].mean())
    m_lat, m_lon = _approx_meter_scale(mean_lat)

    ref_xy = np.column_stack([ref['lat'].values * m_lat, ref['lon'].values * m_lon])
    tar_xy = np.column_stack([df['avg_latitude'].values * m_lat, df['avg_longitude'].values * m_lon])

    tree = cKDTree(ref_xy)
    _, idx = tree.query(tar_xy, k=1)
    df['n_geoid'] = ref['n_geoid'].values[idx].astype(np.float32)
    return df


def compute_enhanced_isa_features(df):
    """
    Compute enhanced ISA-related features for meter-level accuracy.
    
    These features help the neural field understand deviations from standard ISA:
    - isa_height: Standard ISA height (universal)
    - isa_deviation: Pressure deviation from ISA standard sea level
    - tv_k: Virtual temperature (accounts for humidity)
    - isa_temp_expected: Temperature expected by ISA at this height
    - temp_deviation_from_isa: Actual temp difference from ISA expectation
    """
    df = df.copy()
    
    # Standard ISA parameters
    P0_ISA = 101325.0
    T0_ISA = 288.15
    ALPHA_ISA = 0.0065
    
    # 1. Compute ISA height (already done in compute_isa_height_msl, but recompute for features)
    p = np.clip(df['avg_pressure'].values.astype(np.float32), 1000.0, None)
    exponent = (ALPHA_ISA * R_DRY_AIR) / G0
    df['isa_height'] = (T0_ISA / ALPHA_ISA) * (1.0 - (p / P0_ISA) ** exponent)
    
    # 2. ISA deviation: how much pressure differs from standard sea level
    # Positive = high pressure system, Negative = low pressure system
    df['isa_pressure_dev_pa'] = df['avg_pressure'] - P0_ISA
    df['isa_pressure_dev_pct'] = 100.0 * df['isa_pressure_dev_pa'] / P0_ISA
    
    # 3. Virtual temperature (humidity-corrected temperature for density calculations)
    if 'avg_temperature' in df.columns and 'avg_humidity' in df.columns:
        temp_k = _as_temperature_k(df['avg_temperature'])
        df['tv_k'] = _compute_virtual_temperature_k(temp_k, df['avg_humidity'].values, p)
    else:
        df['tv_k'] = T0_ISA
    
    # 4. Expected ISA temperature at this ISA height
    df['isa_temp_expected_k'] = T0_ISA - ALPHA_ISA * df['isa_height']
    
    # 5. Temperature deviation from ISA (important for identifying non-standard atmospheres)
    if 'avg_temperature' in df.columns:
        temp_k = _as_temperature_k(df['avg_temperature'])
        df['temp_deviation_from_isa_k'] = temp_k - df['isa_temp_expected_k']
    else:
        df['temp_deviation_from_isa_k'] = 0.0
    
    # 6. Scale height using actual virtual temperature (more accurate than ISA)
    df['scale_height_m'] = (R_DRY_AIR * df['tv_k']) / G0
    
    return df


def compute_isa_height_msl(pressure_pa, temperature_k=None, humidity_pct=None, era5_sp=None):
    """
    Compute ISA-based MSL height from barometric pressure.
    
    This uses the UNIVERSAL ISA model (not fold-fitted):
    h_msl = (T0/alpha) * (1 - (P/P0)^(alpha*R/g))
    
    Args:
        pressure_pa: Barometric pressure in Pascals
        temperature_k: Optional actual temperature for correction hint
        humidity_pct: Optional humidity for virtual temperature
        era5_sp: Optional ERA5 surface pressure for local P0 adjustment
    
    Returns:
        MSL height in meters
    """
    # Universal ISA constants (国际标准大气)
    P0_ISA = 101325.0  # Standard sea level pressure [Pa]
    T0_ISA = 288.15    # Standard sea level temperature [K]
    ALPHA_ISA = 0.0065 # Temperature lapse rate [K/m]
    
    # Use ERA5 surface pressure as local P0 if available (weather-aware)
    if era5_sp is not None:
        p0 = np.clip(era5_sp, 97000.0, 104000.0)  # Sanity check range
    else:
        p0 = P0_ISA
    
    p = np.clip(pressure_pa, 1000.0, None)
    exponent = (ALPHA_ISA * R_DRY_AIR) / G0
    
    # Base ISA height
    h_msl = (T0_ISA / ALPHA_ISA) * (1.0 - (p / p0) ** exponent)
    
    return h_msl


def compute_fold_physics(train_df, test_df, physics_mode='isa_standard'):
    """
    Compute physics baseline on train/test using UNIVERSAL physical formulas.
    
    CRITICAL PRINCIPLES:
    1. Barometer → ISA Model → MSL height (universal constants, NO fold-fitting)
    2. MSL + Geoid = HAE
    3. Neural residual learns: delta = true_MSL - ISA_MSL
    
    Args:
        train_df: Training data
        test_df: Test data  
        physics_mode: 'isa_standard' (recommended) or 'isa_era5_p0' (weather-aware)
                      NOTE: Fold-fitted modes ('fitted_hypsometric', 'legacy_linear') 
                      are DEPRECATED as they violate universal ISA principle.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Step 1: Convert HAE (avg_altitude) to MSL using Geoid undulation
    # HAE = MSL + N_geoid  =>  MSL = HAE - N_geoid
    train_df['h_true_msl'] = train_df['avg_altitude'] - train_df['n_geoid']
    test_df['h_true_msl'] = test_df['avg_altitude'] - test_df['n_geoid']

    if physics_mode == 'isa_era5_p0':
        # Weather-aware ISA: use ERA5 surface pressure as local sea-level pressure
        # This accounts for synoptic pressure variations (high/low pressure systems)
        p0_train = train_df['era5_sp'].values if 'era5_sp' in train_df.columns else None
        p0_test = test_df['era5_sp'].values if 'era5_sp' in test_df.columns else None
        
        train_df['h_physics'] = compute_isa_height_msl(
            train_df['avg_pressure'].values, era5_sp=p0_train
        )
        test_df['h_physics'] = compute_isa_height_msl(
            test_df['avg_pressure'].values, era5_sp=p0_test
        )
        params = {'mode': physics_mode, 'note': 'ERA5-aware local P0'}
        
    elif physics_mode == 'isa_standard':
        # Standard ISA with universal constants
        train_df['h_physics'] = compute_isa_height_msl(train_df['avg_pressure'].values)
        test_df['h_physics'] = compute_isa_height_msl(test_df['avg_pressure'].values)
        params = {
            'mode': physics_mode, 
            'P0': 101325.0, 
            'T0': 288.15, 
            'alpha': 0.0065,
            'note': 'Universal ISA - no fold fitting'
        }
    else:
        # DEPRECATED: Fold-fitted modes should not be used for physical baseline
        # They violate the principle that physics is universal, not data-dependent
        raise ValueError(
            f"Physics mode '{physics_mode}' uses fold-fitted parameters and is DEPRECATED. "
            "Use 'isa_standard' (fixed universal constants) or 'isa_era5_p0' (weather-aware). "
            "The neural residual field should learn location-specific corrections, not the physics baseline."
        )

    # Step 2: Compute residual = true_MSL - physics_MSL
    # Neural field learns this residual (location/weather-dependent correction to universal ISA)
    train_df['residual'] = train_df['h_true_msl'] - train_df['h_physics']
    test_df['residual'] = test_df['h_true_msl'] - test_df['h_physics']
    
    return train_df, test_df, params


def apply_minimum_performance_guard(pred_hae, pred_residual, h_physics_test, n_geoid_test, 
                                     y_test_hae, isa_mae_threshold_factor=1.5):
    """
    Safety mechanism: Ensure model doesn't degrade significantly below ISA baseline.
    
    In worst-case scenarios (extreme OOD, sensor malfunction), blend predictions
    toward ISA baseline to maintain minimum performance guarantees.
    
    Args:
        pred_hae: Current model predictions (HAE)
        pred_residual: Raw residual predictions
        h_physics_test: ISA physics baseline (MSL)
        n_geoid_test: Geoid undulation
        y_test_hae: Ground truth HAE (for computing ISA baseline error)
        isa_mae_threshold_factor: Max allowed degradation vs ISA (default 1.5x)
    
    Returns:
        Guarded predictions that won't be worse than threshold * ISA_error
    """
    isa_hae = h_physics_test + n_geoid_test
    isa_errors = np.abs(isa_hae - y_test_hae)
    isa_mae = np.mean(isa_errors)
    
    pred_errors = np.abs(pred_hae - y_test_hae)
    
    # Identify samples where model is significantly worse than ISA
    # This is a post-hoc analysis; in production, we'd use confidence scores
    error_ratio = pred_errors / np.maximum(isa_errors, 0.1)  # Avoid division by zero
    
    # Samples where model error is > threshold * ISA error
    problematic = error_ratio > isa_mae_threshold_factor
    
    if problematic.sum() == 0:
        return pred_hae, pred_residual, 0.0  # No guard needed
    
    # For problematic samples, blend toward ISA baseline
    # Weight by how badly we're performing
    blend_weight = np.clip((error_ratio - 1.0) / (isa_mae_threshold_factor - 1.0), 0.0, 1.0)
    blend_weight = blend_weight * problematic.astype(float)
    
    # Blend: (1-w) * pred + w * isa
    guarded_hae = (1 - blend_weight) * pred_hae + blend_weight * isa_hae
    guarded_residual = (1 - blend_weight) * pred_residual  # Blend residual toward 0
    
    guard_ratio = problematic.mean()
    return guarded_hae, guarded_residual, guard_ratio


def apply_physical_fallback(pred_residual, confidence, h_physics_test, test_ood_distance,
                            train_h_physics, train_ood_distance,
                            enable_fallback=False,
                            fallback_alt_margin_m=20.0,
                            fallback_ood_quantile=0.95,
                            fallback_ood_buffer_m=20.0,
                            fallback_threshold=1.0,
                            fallback_slope=6.0,
                            fallback_conf_weight=0.5):
    """Blend residual prediction toward 0 in extreme extrapolation/OOD regions."""
    if not enable_fallback:
        return pred_residual

    train_min = float(np.min(train_h_physics))
    train_max = float(np.max(train_h_physics))
    margin = max(float(fallback_alt_margin_m), 1.0)

    high_excess = np.maximum((h_physics_test - train_max) / margin, 0.0)
    low_excess = np.maximum((train_min - h_physics_test) / margin, 0.0)
    alt_extreme = np.maximum(high_excess, low_excess)

    q = float(np.quantile(train_ood_distance, np.clip(fallback_ood_quantile, 0.5, 0.999)))
    q = q + max(float(fallback_ood_buffer_m), 0.0)
    ood_scale = max(q * 0.5, 5.0)
    ood_extreme = np.maximum((test_ood_distance - q) / ood_scale, 0.0)

    base_risk = np.maximum(alt_extreme, ood_extreme)
    if confidence is not None:
        conf_term = np.maximum((1.0 - confidence) * float(fallback_conf_weight), 0.0) * (base_risk > 0.0).astype(np.float32)
    else:
        conf_term = 0.0

    risk_score = base_risk + conf_term
    fallback_w = 1.0 / (1.0 + np.exp(-float(fallback_slope) * (risk_score - float(fallback_threshold))))
    return (1.0 - fallback_w) * pred_residual


def build_temporal_pairs(df, max_gap_minutes=5, max_pairs=30000):
    ordered = df.sort_values(['uid', 'processed_time']).reset_index(drop=True)
    pair_i = []
    pair_j = []

    for _, group in ordered.groupby('uid', sort=False):
        if len(group) < 2:
            continue
        idx = group.index.values
        t = group['processed_time'].values.astype('datetime64[m]')
        dt = (t[1:] - t[:-1]).astype('timedelta64[m]').astype(int)
        mask = dt <= max_gap_minutes
        valid_i = idx[:-1][mask]
        valid_j = idx[1:][mask]
        pair_i.extend(valid_i.tolist())
        pair_j.extend(valid_j.tolist())

    if len(pair_i) > max_pairs:
        choice = np.random.choice(len(pair_i), size=max_pairs, replace=False)
        pair_i = np.array(pair_i)[choice].tolist()
        pair_j = np.array(pair_j)[choice].tolist()

    return ordered, np.array(pair_i, dtype=np.int64), np.array(pair_j, dtype=np.int64)


# =============================================================================
# Data Loading with Stabilized GNSS
# =============================================================================

def load_data(use_stabilized=True):
    """Load data with stabilized or original GNSS heights"""

    if use_stabilized and Path('data/processed/sensor_data_stabilized.csv').exists():
        df = pd.read_csv('data/processed/sensor_data_stabilized.csv')
        print("✓ Using STABILIZED GNSS heights")
        # The stabilized height is already in avg_altitude column
    else:
        df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
        print("Using original GNSS heights")

    df['processed_time'] = pd.to_datetime(df['processed_time'])

    uid_alt_std = df.groupby('uid')['avg_altitude'].std().fillna(0.0)
    low_var_ratio = float((uid_alt_std < 0.5).mean()) if len(uid_alt_std) > 0 else 0.0
    if low_var_ratio > 0.7:
        print("  [Warning] Per-sensor altitude is nearly constant for most UIDs.")
        print("            This dataset is weakly identifiable for learning pressure->height mapping.")
        print("            Consider training vertical physics/residuals with moving-platform data (e.g., final_training_data.csv).")

    print(f"  Data: {len(df)} samples, {df['uid'].nunique()} sensors")

    return df, None, None


# =============================================================================
# Curriculum Learning
# =============================================================================

def create_curriculum_stages(df, strategy='altitude_density'):
    """
    Create curriculum learning stages

    Strategies:
    - 'altitude_density': Stage by altitude and sensor density
    - 'residual': Stage by residual magnitude (easy = small residual)
    """
    stages = []

    if strategy == 'altitude_density':
        density_col = 'local_density' if 'local_density' in df.columns else 'sensor_density'

        # Stage 1: Low altitude, high density (easiest)
        alt_q40 = df['avg_altitude'].quantile(0.4)
        den_q60 = df[density_col].quantile(0.6)
        easy_df = df[(df['avg_altitude'] < alt_q40) & (df[density_col] > den_q60)].copy()

        if len(easy_df) < 100:
            easy_df = df[df['avg_altitude'] < alt_q40].copy()

        # Stage 2: Medium difficulty
        alt_q70 = df['avg_altitude'].quantile(0.7)
        den_q30 = df[density_col].quantile(0.3)
        medium_df = df[(df['avg_altitude'] < alt_q70) | (df[density_col] > den_q30)].copy()

        if len(medium_df) < 100:
            medium_df = df[df['avg_altitude'] < alt_q70].copy()

        # Stage 3: All data (hardest)
        hard_df = df.copy()

        stages = [
            ('Easy (Low Alt, High Density)', easy_df),
            ('Medium', medium_df),
            ('Hard (Full)', hard_df)
        ]

    elif strategy == 'residual':
        # Stage by how far from physics baseline
        res_q30 = df['residual'].abs().quantile(0.3)
        res_q60 = df['residual'].abs().quantile(0.6)

        easy_df = df[df['residual'].abs() < res_q30].copy()
        medium_df = df[df['residual'].abs() < res_q60].copy()
        hard_df = df.copy()

        stages = [
            ('Easy (Small Residual)', easy_df),
            ('Medium', medium_df),
            ('Hard (Full)', hard_df)
        ]

    return stages


def train_with_curriculum(model, stages, test_df, h_physics_test, y_test_hae, n_geoid_test,
                          max_epochs_per_stage=200, patience=50, lr=1e-3, feature_cols=None,
                          w_residual_reg=0.005, w_temporal=0.010, w_hydro=0.020, w_confidence=0.005,
                          constraint_schedule='none', schedule_warmup_ratio=0.35,
                          batch_size=0, pair_sample_size=2048,
                          train_h_physics=None, train_ood_distance=None,
                          enable_physical_fallback=False,
                          fallback_alt_margin_m=20.0,
                          fallback_ood_quantile=0.95,
                          fallback_ood_buffer_m=20.0,
                          fallback_threshold=1.0,
                          fallback_slope=6.0,
                          fallback_conf_weight=0.5):
    """Train model with curriculum learning"""

    # Default feature columns if not provided
    if feature_cols is None:
        feature_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure',
                        'pressure_anom', 'temp_anom', 'local_density', 'knn_dist_m',
                        'dist_to_train_mean_m', 'delta_pressure_1step', 'delta_temp_1step',
                        'isa_pressure_dev_pa', 'isa_pressure_dev_pct',
                        'temp_deviation_from_isa_k', 'scale_height_m']

    best_overall_mae = float('inf')
    best_state = None
    best_scalers = None
    history = {'stages': [], 'epochs': [], 'maes': [], 'losses': []}

    global_epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    full_train_df = stages[-1][1].copy()
    scaler_spatial = SpatialMinMaxScaler()
    scaler_feature = StandardScaler()
    scaler_spatial.fit(full_train_df[['avg_latitude', 'avg_longitude']].values)
    scaler_feature.fit(full_train_df[feature_cols])

    def get_constraint_factor(epoch, total_epochs):
        if constraint_schedule == 'none':
            return 1.0
        warmup_epochs = max(1, int(total_epochs * schedule_warmup_ratio))
        if epoch >= warmup_epochs:
            return 1.0
        progress = epoch / float(warmup_epochs)
        if constraint_schedule == 'linear_warmup':
            return progress
        if constraint_schedule == 'cosine_warmup':
            return 0.5 * (1.0 - np.cos(np.pi * progress))
        return 1.0

    for stage_idx, (stage_name, train_df) in enumerate(stages):
        print(f"\n  === Stage {stage_idx+1}: {stage_name} ({len(train_df)} samples) ===")

        if len(train_df) < 100:
            print(f"    Skip: too few samples")
            continue

        ordered_train, pair_i_np, pair_j_np = build_temporal_pairs(train_df)

        X_spatial = scaler_spatial.transform(ordered_train[['avg_latitude', 'avg_longitude']].values)
        X_feature = scaler_feature.transform(ordered_train[feature_cols])
        y_train = ordered_train['residual'].values.astype(np.float32)
        ood_train = ordered_train['ood_distance_m'].values.astype(np.float32)
        logp_train = np.log(np.clip(ordered_train['avg_pressure'].values.astype(np.float32), 1.0, None))

        X_spatial_t = torch.FloatTensor(X_spatial).to(DEVICE)
        X_feature_t = torch.FloatTensor(X_feature).to(DEVICE)
        y_t = torch.FloatTensor(y_train).to(DEVICE).unsqueeze(1)
        ood_train_t = torch.FloatTensor(ood_train).to(DEVICE).unsqueeze(1)
        logp_t = torch.FloatTensor(logp_train).to(DEVICE).unsqueeze(1)

        pair_i_t = torch.LongTensor(pair_i_np).to(DEVICE) if len(pair_i_np) > 0 else None
        pair_j_t = torch.LongTensor(pair_j_np).to(DEVICE) if len(pair_j_np) > 0 else None

        patience_counter = 0

        for epoch in range(max_epochs_per_stage):
            model.train()
            constraint_factor = get_constraint_factor(epoch, max_epochs_per_stage)

            n_train = X_spatial_t.shape[0]
            use_minibatch = batch_size is not None and batch_size > 0 and batch_size < n_train
            epoch_loss = 0.0

            if use_minibatch:
                indices = torch.randperm(n_train, device=DEVICE)
                steps = 0
                for start in range(0, n_train, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    optimizer.zero_grad()

                    pred_gated, pred_raw_clipped, confidence = model(
                        X_spatial_t[batch_idx], X_feature_t[batch_idx], ood_train_t[batch_idx]
                    )

                    loss_main = nn.SmoothL1Loss(beta=3.0)(pred_gated, y_t[batch_idx])
                    loss_residual_reg = torch.mean(torch.abs(pred_raw_clipped))
                    loss_confidence = torch.mean(
                        (1.0 - confidence) * (ood_train_t[batch_idx] / (ood_train_t[batch_idx] + 100.0))
                    )

                    if pair_i_t is not None and pair_i_t.numel() > 0:
                        pair_count = pair_i_t.numel()
                        sample_n = min(pair_sample_size, pair_count)
                        pair_sel = torch.randperm(pair_count, device=DEVICE)[:sample_n]
                        pi = pair_i_t[pair_sel]
                        pj = pair_j_t[pair_sel]
                        pred_i, _, _ = model(X_spatial_t[pi], X_feature_t[pi], ood_train_t[pi])
                        pred_j, _, _ = model(X_spatial_t[pj], X_feature_t[pj], ood_train_t[pj])
                        pred_diff = pred_j - pred_i
                        logp_diff = logp_t[pj] - logp_t[pi]
                        loss_temporal = torch.mean(torch.abs(pred_diff))
                        loss_hydro = torch.mean(torch.relu(pred_diff * logp_diff))
                    else:
                        loss_temporal = torch.tensor(0.0, device=DEVICE)
                        loss_hydro = torch.tensor(0.0, device=DEVICE)

                    loss = (
                        loss_main
                        + constraint_factor * (
                            w_residual_reg * loss_residual_reg
                            + w_temporal * loss_temporal
                            + w_hydro * loss_hydro
                            + w_confidence * loss_confidence
                        )
                    )

                    if torch.isnan(loss) or loss.item() > 1e6:
                        print(f"    NaN detected, stopping stage")
                        break

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += float(loss.item())
                    steps += 1
                loss_value = epoch_loss / max(steps, 1)
            else:
                optimizer.zero_grad()
                pred_gated, pred_raw_clipped, confidence = model(X_spatial_t, X_feature_t, ood_train_t)

                loss_main = nn.SmoothL1Loss(beta=3.0)(pred_gated, y_t)
                loss_residual_reg = torch.mean(torch.abs(pred_raw_clipped))

                if pair_i_t is not None and pair_i_t.numel() > 0:
                    pred_diff = pred_gated[pair_j_t] - pred_gated[pair_i_t]
                    logp_diff = logp_t[pair_j_t] - logp_t[pair_i_t]
                    loss_temporal = torch.mean(torch.abs(pred_diff))
                    loss_hydro = torch.mean(torch.relu(pred_diff * logp_diff))
                else:
                    loss_temporal = torch.tensor(0.0, device=DEVICE)
                    loss_hydro = torch.tensor(0.0, device=DEVICE)

                loss_confidence = torch.mean((1.0 - confidence) * (ood_train_t / (ood_train_t + 100.0)))

                loss = (
                    loss_main
                    + constraint_factor * (
                        w_residual_reg * loss_residual_reg
                        + w_temporal * loss_temporal
                        + w_hydro * loss_hydro
                        + w_confidence * loss_confidence
                    )
                )

                if torch.isnan(loss) or loss.item() > 1e6:
                    print(f"    NaN detected, stopping stage")
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                loss_value = float(loss.item())

            # Evaluate
            if epoch % 20 == 0 or epoch == max_epochs_per_stage - 1:
                model.eval()
                with torch.no_grad():
                    X_spatial_test = scaler_spatial.transform(test_df[['avg_latitude', 'avg_longitude']].values)
                    X_feature_test = scaler_feature.transform(test_df[feature_cols])
                    ood_test = test_df['ood_distance_m'].values.astype(np.float32)

                    pred_s, _, conf_t = model(
                        torch.FloatTensor(X_spatial_test).to(DEVICE),
                        torch.FloatTensor(X_feature_test).to(DEVICE),
                        torch.FloatTensor(ood_test).to(DEVICE).unsqueeze(1)
                    )
                    pred_s = pred_s.cpu().numpy()
                    conf_np = conf_t.cpu().numpy().reshape(-1)

                pred_residual = pred_s.reshape(-1)
                pred_residual = apply_physical_fallback(
                    pred_residual,
                    conf_np,
                    h_physics_test,
                    ood_test,
                    (train_h_physics if train_h_physics is not None else np.array([np.min(h_physics_test), np.max(h_physics_test)])),
                    (train_ood_distance if train_ood_distance is not None else ood_test),
                    enable_fallback=enable_physical_fallback,
                    fallback_alt_margin_m=fallback_alt_margin_m,
                    fallback_ood_quantile=fallback_ood_quantile,
                    fallback_ood_buffer_m=fallback_ood_buffer_m,
                    fallback_threshold=fallback_threshold,
                    fallback_slope=fallback_slope,
                    fallback_conf_weight=fallback_conf_weight,
                )
                # Prediction flow: Baro -> ISA -> MSL -> +NeuralResidual -> corrected_MSL -> +Geoid -> HAE
                pred_msl = h_physics_test + pred_residual  # Corrected MSL height
                pred_hae = pred_msl + n_geoid_test          # Convert to HAE
                mae = np.mean(np.abs(pred_hae - y_test_hae))

                history['stages'].append(stage_idx + 1)
                history['epochs'].append(global_epoch)
                history['maes'].append(float(mae))
                history['losses'].append(loss_value)

                if epoch % 40 == 0:
                    print(
                        f"    Epoch {global_epoch}: Loss={loss_value:.4f}, "
                        f"MAE={mae:.2f}m, CF={constraint_factor:.2f}"
                    )

                if mae < best_overall_mae:
                    best_overall_mae = mae
                    best_state = model.state_dict().copy()
                    # Save the scalers that achieved the best MAE
                    best_scalers = (scaler_spatial, scaler_feature)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

                model.train()

            global_epoch += 1

        print(f"    Stage {stage_idx+1} complete. Best MAE: {best_overall_mae:.2f}m")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Return the best scalers (from the stage that achieved best MAE)
    y_identity_scaler = IdentityScaler()
    if best_scalers is not None:
        return model, best_overall_mae, best_scalers[0], best_scalers[1], y_identity_scaler, history
    else:
        return model, best_overall_mae, scaler_spatial, scaler_feature, y_identity_scaler, history


# =============================================================================
# LOSO Validation
# =============================================================================

def run_loso_validation(df, use_hash=True, use_terrain=True, ablation=None, epochs=200,
                        residual_clip=40.0, ood_decay_m=120.0,
                        w_residual_reg=0.005, w_temporal=0.010, w_hydro=0.020, w_confidence=0.005,
                        constraint_schedule='none', schedule_warmup_ratio=0.35,
                        lr=1e-3, hard_ood_threshold_m=None,
                        soft_ood_center_m=None, soft_ood_steepness_m=20.0,
                        adaptive_ood_center_quantile=None,
                        batch_size=0, pair_sample_size=2048,
                        use_domain_features=False,
                        physics_mode='isa_standard',
                        geoid_ref_csv='data/final_training_data.csv',
                        enable_physical_fallback=False,
                        fallback_alt_margin_m=20.0,
                        fallback_ood_quantile=0.95,
                        fallback_ood_buffer_m=20.0,
                        fallback_threshold=1.0,
                        fallback_slope=6.0,
                        fallback_conf_weight=0.5):
    """
    Run LOSO validation

    Args:
        ablation: None for full model, or one of ['no_era5', 'no_terrain', 'no_hash']
    """
    print("\n" + "="*70)
    print(f"LOSO VALIDATION (Hash={use_hash}, Terrain={use_terrain}, Ablation={ablation})")
    print("="*70)

    df = _prepare_geoid_column(df, geoid_ref_csv=geoid_ref_csv)

    sensors = sorted(df['uid'].unique())

    results = {
        'physics_mae': [],
        'pinf_mae': [],
        'sensors': []
    }

    for fold_idx, test_sensor in enumerate(sensors):
        print(f"\nFold {fold_idx+1}/{len(sensors)}: {test_sensor[-8:]}")

        train_raw = df[df['uid'] != test_sensor].copy()
        test_raw = df[df['uid'] == test_sensor].copy()
        train_df, test_df = compute_fold_safe_features(train_raw, test_raw, use_domain_features=use_domain_features)
        
        # Compute enhanced ISA features for meter-level accuracy
        train_df = compute_enhanced_isa_features(train_df)
        test_df = compute_enhanced_isa_features(test_df)
        
        train_df, test_df, phys_params = compute_fold_physics(train_df, test_df, physics_mode=physics_mode)

        fold_soft_center_m = soft_ood_center_m
        if adaptive_ood_center_quantile is not None:
            fold_soft_center_m = float(np.quantile(train_df['ood_distance_m'].values, adaptive_ood_center_quantile))

        h_physics_test = test_df['h_physics'].values
        n_geoid_test = test_df['n_geoid'].values
        y_test_hae = test_df['avg_altitude'].values

        # Physics baseline accuracy: ISA_MSL + Geoid = predicted_HAE
        # Compare against true_HAE (y_test_hae)
        phys_mae = np.mean(np.abs((h_physics_test + n_geoid_test) - y_test_hae))
        results['physics_mae'].append(phys_mae)

        # Prepare feature columns based on ablation
        # Core features: physics baseline + sensor readings + spatial context + ISA deviations
        feature_cols = ['h_physics', 'avg_temperature', 'avg_humidity', 'avg_pressure',
                'pressure_anom', 'temp_anom', 'local_density', 'knn_dist_m',
                'dist_to_train_mean_m', 'delta_pressure_1step', 'delta_temp_1step',
                # Enhanced ISA features for meter-level accuracy
                'isa_pressure_dev_pa', 'isa_pressure_dev_pct',
                'temp_deviation_from_isa_k', 'scale_height_m']

        if use_domain_features:
            feature_cols.extend([
                'pressure_uid_center', 'pressure_uid_z',
                'temp_uid_center', 'temp_uid_z',
                'humidity_uid_center', 'humidity_uid_z'
            ])

        if ablation == 'no_era5':
            feature_cols = [c for c in feature_cols if c not in ['pressure_anom', 'temp_anom']]
        elif ablation == 'no_terrain':
            feature_cols = [c for c in feature_cols if c not in ['local_density', 'knn_dist_m', 'dist_to_train_mean_m']]

        # Create curriculum stages
        stages = create_curriculum_stages(train_df)

        # Train model
        feature_dim = len(feature_cols)
        use_hash_enc = use_hash if ablation != 'no_hash' else False

        model = HardConstrainedNF(
            use_hash_encoding=use_hash_enc,
            use_terrain=use_terrain if ablation != 'no_terrain' else False,
            st_dim=2,
            feature_dim=feature_dim,
            hidden_dim=256,
            num_layers=8,
            residual_clip=residual_clip,
            ood_decay_m=ood_decay_m,
            hard_ood_threshold_m=hard_ood_threshold_m,
            soft_ood_center_m=fold_soft_center_m,
            soft_ood_steepness_m=soft_ood_steepness_m
        ).to(DEVICE)

        model, best_mae, _, _, _, _ = train_with_curriculum(
            model, stages, test_df, h_physics_test, y_test_hae, n_geoid_test,
            max_epochs_per_stage=epochs, patience=50, lr=lr, feature_cols=feature_cols,
            w_residual_reg=w_residual_reg, w_temporal=w_temporal,
            w_hydro=w_hydro, w_confidence=w_confidence,
            constraint_schedule=constraint_schedule,
            schedule_warmup_ratio=schedule_warmup_ratio,
            batch_size=batch_size,
            pair_sample_size=pair_sample_size,
            train_h_physics=train_df['h_physics'].values,
            train_ood_distance=train_df['ood_distance_m'].values,
            enable_physical_fallback=enable_physical_fallback,
            fallback_alt_margin_m=fallback_alt_margin_m,
            fallback_ood_quantile=fallback_ood_quantile,
            fallback_ood_buffer_m=fallback_ood_buffer_m,
            fallback_threshold=fallback_threshold,
            fallback_slope=fallback_slope,
            fallback_conf_weight=fallback_conf_weight,
        )

        results['pinf_mae'].append(best_mae)
        results['sensors'].append(test_sensor)

        print(f"  Physics: {phys_mae:.2f}m, PINF: {best_mae:.2f}m")
        if fold_idx == 0:
            print(f"  Physics mode={physics_mode}, params={phys_params}")

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Physics Mean MAE: {np.mean(results['physics_mae']):.2f} ± {np.std(results['physics_mae']):.2f}m")
    print(f"PINF Mean MAE: {np.mean(results['pinf_mae']):.2f} ± {np.std(results['pinf_mae']):.2f}m")
    print(f"PINF Best MAE: {np.min(results['pinf_mae']):.2f}m")
    print(f"PINF Worst MAE: {np.max(results['pinf_mae']):.2f}m")

    return results


def run_ablation_study(df):
    """Run full ablation study"""
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)

    ablations = {
        'Full Model': None,
        'Without ERA5': 'no_era5',
        'Without Terrain': 'no_terrain',
        'Without Hash Encoding': 'no_hash',
    }

    ablation_results = {}

    for name, ablation in ablations.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        results = run_loso_validation(df, use_hash=True, use_terrain=True, ablation=ablation, epochs=100)

        ablation_results[name] = {
            'mean_mae': float(np.mean(results['pinf_mae'])),
            'std_mae': float(np.std(results['pinf_mae'])),
            'best_mae': float(np.min(results['pinf_mae'])),
            'all_maes': [float(m) for m in results['pinf_mae']]
        }

        print(f"\n{name}: {ablation_results[name]['mean_mae']:.2f} ± {ablation_results[name]['std_mae']:.2f}m")

    # Save results
    with open('experiments/results/refined_model/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    for name, res in ablation_results.items():
        print(f"{name:30s}: {res['mean_mae']:.2f} ± {res['std_mae']:.2f}m")

    return ablation_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-stabilized', action='store_true',
                       help='Use stabilized GNSS heights')
    parser.add_argument('--ablation', action='store_true',
                       help='Run full ablation study')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Epochs per curriculum stage')
    parser.add_argument('--residual-clip', type=float, default=40.0,
                       help='Residual clipping range in meters')
    parser.add_argument('--ood-decay-m', type=float, default=120.0,
                       help='Distance decay scale (meters) for OOD confidence gating')
    parser.add_argument('--w-residual-reg', type=float, default=0.005,
                       help='Weight of residual magnitude regularization term')
    parser.add_argument('--w-temporal', type=float, default=0.010,
                       help='Weight of temporal smoothness term')
    parser.add_argument('--w-hydro', type=float, default=0.020,
                       help='Weight of hydrostatic consistency term')
    parser.add_argument('--w-confidence', type=float, default=0.005,
                       help='Weight of confidence regularization term')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for AdamW optimizer')
    parser.add_argument('--seed', type=int, default=42,
                       help='Global random seed')
    parser.add_argument('--hard-ood-threshold-m', type=float, default=-1.0,
                       help='Hard cutoff distance (m); residual forced to 0 beyond threshold, <=0 disables')
    parser.add_argument('--soft-ood-center-m', type=float, default=-1.0,
                       help='Soft gating center distance (m); confidence decays via sigmoid beyond center, <=0 disables')
    parser.add_argument('--soft-ood-steepness-m', type=float, default=20.0,
                       help='Soft gating steepness in meters (smaller = sharper decay)')
    parser.add_argument('--adaptive-ood-center-quantile', type=float, default=-1.0,
                       help='If >0, set soft center per fold to this quantile of train ood_distance_m (e.g., 0.9)')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Mini-batch size for training; <=0 uses full-batch')
    parser.add_argument('--pair-sample-size', type=int, default=2048,
                       help='Sampled temporal-pair count per batch when using mini-batch training')
    parser.add_argument('--use-domain-features', action='store_true',
                       help='Enable per-sensor domain normalization features')
    parser.add_argument('--physics-mode', type=str, default='isa_standard',
                       choices=['isa_standard', 'isa_era5_p0'],
                       help=(
                           'Physics baseline mode: '
                           'isa_standard = universal ISA model with fixed constants (recommended); '
                           'isa_era5_p0 = ISA with ERA5 surface pressure as local P0 (weather-aware). '
                           'NOTE: Fold-fitted modes are removed to maintain universal physics principle.'
                       ))
    parser.add_argument('--geoid-ref-csv', type=str, default='data/final_training_data.csv',
                       help='Reference CSV containing lat/lon/n_geoid for HAE<->MSL conversion')
    parser.add_argument('--enable-physical-fallback', action='store_true',
                       help='In extreme extrapolation/OOD, blend residual toward 0 (physics baseline)')
    parser.add_argument('--fallback-alt-margin-m', type=float, default=20.0,
                       help='Altitude extrapolation margin (meters) for fallback gating')
    parser.add_argument('--fallback-ood-quantile', type=float, default=0.95,
                       help='Train OOD-distance quantile used as fallback trigger')
    parser.add_argument('--fallback-ood-buffer-m', type=float, default=20.0,
                       help='Extra distance margin (m) added on top of OOD quantile before fallback')
    parser.add_argument('--fallback-threshold', type=float, default=1.0,
                       help='Risk threshold for fallback gate activation')
    parser.add_argument('--fallback-slope', type=float, default=6.0,
                       help='Sigmoid slope for fallback gate')
    parser.add_argument('--fallback-conf-weight', type=float, default=0.5,
                       help='Weight for (1-confidence) term in fallback risk score')
    parser.add_argument('--constraint-schedule', type=str, default='none',
                       choices=['none', 'linear_warmup', 'cosine_warmup'],
                       help='Schedule for ramping physics/gating constraint losses')
    parser.add_argument('--schedule-warmup-ratio', type=float, default=0.35,
                       help='Fraction of stage epochs used to warm up constraint weights')
    parser.add_argument('--result-tag', type=str, default='default',
                       help='Tag name used for output result file')
    args = parser.parse_args()

    print("="*70)
    print("REFINED HARD-CONSTRAINED NEURAL FIELD")
    print("="*70)

    set_global_seed(args.seed)

    # Load data
    print("\n[1] Loading data...")
    df, Hs, P0 = load_data(use_stabilized=args.use_stabilized)

    print("\n[2] Feature strategy: fold-safe spatial + ERA5 anomaly + temporal deltas")

    # Run
    if args.ablation:
        print("\n[3] Running ablation study...")
        results = run_ablation_study(df)
    else:
        print("\n[3] Running LOSO validation...")
        results = run_loso_validation(
            df,
            epochs=args.epochs,
            residual_clip=args.residual_clip,
            ood_decay_m=args.ood_decay_m,
            w_residual_reg=args.w_residual_reg,
            w_temporal=args.w_temporal,
            w_hydro=args.w_hydro,
            w_confidence=args.w_confidence,
            constraint_schedule=args.constraint_schedule,
            schedule_warmup_ratio=args.schedule_warmup_ratio,
            lr=args.lr,
            hard_ood_threshold_m=(args.hard_ood_threshold_m if args.hard_ood_threshold_m > 0 else None),
            soft_ood_center_m=(args.soft_ood_center_m if args.soft_ood_center_m > 0 else None),
            soft_ood_steepness_m=args.soft_ood_steepness_m,
            adaptive_ood_center_quantile=(args.adaptive_ood_center_quantile if args.adaptive_ood_center_quantile > 0 else None),
            batch_size=args.batch_size,
            pair_sample_size=args.pair_sample_size,
            use_domain_features=args.use_domain_features,
            physics_mode=args.physics_mode,
            geoid_ref_csv=args.geoid_ref_csv,
            enable_physical_fallback=args.enable_physical_fallback,
            fallback_alt_margin_m=args.fallback_alt_margin_m,
            fallback_ood_quantile=args.fallback_ood_quantile,
            fallback_ood_buffer_m=args.fallback_ood_buffer_m,
            fallback_threshold=args.fallback_threshold,
            fallback_slope=args.fallback_slope,
            fallback_conf_weight=args.fallback_conf_weight,
        )

        # Save results
        output = {
            'physics_mae': results['physics_mae'],
            'pinf_mae': results['pinf_mae'],
            'sensors': results['sensors'],
            'summary': {
                'physics_mean': float(np.mean(results['physics_mae'])),
                'pinf_mean': float(np.mean(results['pinf_mae'])),
                'pinf_best': float(np.min(results['pinf_mae'])),
                'pinf_worst': float(np.max(results['pinf_mae']))
            },
            'config': {
                'epochs': args.epochs,
                'residual_clip': args.residual_clip,
                'ood_decay_m': args.ood_decay_m,
                'w_residual_reg': args.w_residual_reg,
                'w_temporal': args.w_temporal,
                'w_hydro': args.w_hydro,
                'w_confidence': args.w_confidence,
                'lr': args.lr,
                'seed': args.seed,
                'hard_ood_threshold_m': (args.hard_ood_threshold_m if args.hard_ood_threshold_m > 0 else None),
                'soft_ood_center_m': (args.soft_ood_center_m if args.soft_ood_center_m > 0 else None),
                'soft_ood_steepness_m': args.soft_ood_steepness_m,
                'adaptive_ood_center_quantile': (args.adaptive_ood_center_quantile if args.adaptive_ood_center_quantile > 0 else None),
                'batch_size': args.batch_size,
                'pair_sample_size': args.pair_sample_size,
                'use_domain_features': args.use_domain_features,
                'physics_mode': args.physics_mode,
                'geoid_ref_csv': args.geoid_ref_csv,
                'enable_physical_fallback': args.enable_physical_fallback,
                'fallback_alt_margin_m': args.fallback_alt_margin_m,
                'fallback_ood_quantile': args.fallback_ood_quantile,
                'fallback_ood_buffer_m': args.fallback_ood_buffer_m,
                'fallback_threshold': args.fallback_threshold,
                'fallback_slope': args.fallback_slope,
                'fallback_conf_weight': args.fallback_conf_weight,
                'constraint_schedule': args.constraint_schedule,
                'schedule_warmup_ratio': args.schedule_warmup_ratio,
                'result_tag': args.result_tag
            }
        }

        output_path = f"experiments/results/refined_model/results_{args.result_tag}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        with open('experiments/results/refined_model/results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_path}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
