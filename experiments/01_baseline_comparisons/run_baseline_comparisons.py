"""
Baseline Comparisons Experiment

Compares PINN against:
1. IDW (Inverse Distance Weighting)
2. Ordinary Kriging
3. Random Forest
4. XGBoost
"""
import numpy as np
import pandas as pd
import torch
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging
import xgboost as xgb
import sys
import os
sys.path.insert(0, '/data/home/huxiao/workspace/GeoIDbox')

from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_generalized_with_bias import compute_sensor_bias


class IDWInterpolator:
    """Inverse Distance Weighting for pressure bias interpolation."""
    def __init__(self, power=2):
        self.power = power
        self.known_coords = None
        self.known_values = None
    
    def fit(self, coords, values):
        """Fit IDW with known coordinates and values."""
        self.known_coords = coords
        self.known_values = values
    
    def predict(self, coords):
        """Predict at new coordinates."""
        # Calculate distances
        distances = cdist(coords, self.known_coords)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Compute weights (inverse distance)
        weights = 1.0 / (distances ** self.power)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Weighted average
        predictions = np.sum(weights * self.known_values, axis=1)
        return predictions


def run_baseline_comparisons():
    """Run all baseline comparisons."""
    print("="*70)
    print("EXPERIMENT 1: Baseline Comparisons")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/sensor_data_filtered.csv')
    df, phys_params = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)
    
    # Prepare features
    features = ['avg_latitude', 'avg_longitude', 'avg_altitude', 'avg_temperature', 'avg_humidity']
    target = 'pressure_bias'
    
    results = []
    
    # LOSO evaluation
    sensors = df['uid'].unique()
    
    for fold, held_out_sensor in enumerate(sensors):
        print(f"\n{'='*70}")
        print(f"Fold {fold}: Held-out sensor {held_out_sensor[:20]}")
        print(f"{'='*70}")
        
        # Split data
        test_mask = df['uid'] == held_out_sensor
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
        
        # Prepare data
        X_train = train_df[features].values
        y_train = train_df[target].values
        X_test = test_df[features].values
        y_test = test_df[target].values
        
        # Also get coordinates for spatial methods
        coords_train = train_df[['avg_latitude', 'avg_longitude']].values
        coords_test = test_df[['avg_latitude', 'avg_longitude']].values
        
        fold_results = {'fold': fold, 'held_out': held_out_sensor[:20]}
        
        # 1. IDW (Inverse Distance Weighting)
        print("\n  Running IDW...")
        try:
            idw = IDWInterpolator(power=2)
            idw.fit(coords_train, y_train)
            y_pred_idw = idw.predict(coords_test)
            
            # Convert pressure bias back to height error for fair comparison
            mae_idw = np.mean(np.abs(y_pred_idw - y_test))
            rmse_idw = np.sqrt(np.mean((y_pred_idw - y_test)**2))
            
            fold_results['IDW_MAE'] = mae_idw
            fold_results['IDW_RMSE'] = rmse_idw
            print(f"    IDW - MAE: {mae_idw:.3f} Pa, RMSE: {rmse_idw:.3f} Pa")
        except Exception as e:
            print(f"    IDW failed: {e}")
            fold_results['IDW_MAE'] = np.nan
            fold_results['IDW_RMSE'] = np.nan
        
        # 2. Ordinary Kriging
        print("\n  Running Ordinary Kriging...")
        try:
            # Subsample for speed if too many points
            max_points = 2000
            if len(coords_train) > max_points:
                indices = np.random.choice(len(coords_train), max_points, replace=False)
                coords_train_sub = coords_train[indices]
                y_train_sub = y_train[indices]
            else:
                coords_train_sub = coords_train
                y_train_sub = y_train
            
            OK = OrdinaryKriging(
                coords_train_sub[:, 0], coords_train_sub[:, 1], y_train_sub,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False,
                nlags=10
            )
            
            y_pred_krig, ss = OK.execute('points', coords_test[:, 0], coords_test[:, 1])
            y_pred_krig = np.array(y_pred_krig).flatten()
            
            mae_krig = np.mean(np.abs(y_pred_krig - y_test))
            rmse_krig = np.sqrt(np.mean((y_pred_krig - y_test)**2))
            
            fold_results['Kriging_MAE'] = mae_krig
            fold_results['Kriging_RMSE'] = rmse_krig
            print(f"    Kriging - MAE: {mae_krig:.3f} Pa, RMSE: {rmse_krig:.3f} Pa")
        except Exception as e:
            print(f"    Kriging failed: {e}")
            fold_results['Kriging_MAE'] = np.nan
            fold_results['Kriging_RMSE'] = np.nan
        
        # 3. Random Forest
        print("\n  Running Random Forest...")
        try:
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            mae_rf = np.mean(np.abs(y_pred_rf - y_test))
            rmse_rf = np.sqrt(np.mean((y_pred_rf - y_test)**2))
            
            fold_results['RF_MAE'] = mae_rf
            fold_results['RF_RMSE'] = rmse_rf
            print(f"    Random Forest - MAE: {mae_rf:.3f} Pa, RMSE: {rmse_rf:.3f} Pa")
        except Exception as e:
            print(f"    Random Forest failed: {e}")
            fold_results['RF_MAE'] = np.nan
            fold_results['RF_RMSE'] = np.nan
        
        # 4. XGBoost
        print("\n  Running XGBoost...")
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            
            mae_xgb = np.mean(np.abs(y_pred_xgb - y_test))
            rmse_xgb = np.sqrt(np.mean((y_pred_xgb - y_test)**2))
            
            fold_results['XGB_MAE'] = mae_xgb
            fold_results['XGB_RMSE'] = rmse_xgb
            print(f"    XGBoost - MAE: {mae_xgb:.3f} Pa, RMSE: {rmse_xgb:.3f} Pa")
        except Exception as e:
            print(f"    XGBoost failed: {e}")
            fold_results['XGB_MAE'] = np.nan
            fold_results['XGB_RMSE'] = np.nan
        
        results.append(fold_results)
    
    # Aggregate results
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiments/01_baseline_comparisons/results.csv', index=False)
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON RESULTS (Mean ± Std)")
    print("="*70)
    
    methods = ['IDW', 'Kriging', 'RF', 'XGB']
    for method in methods:
        mae_col = f'{method}_MAE'
        if mae_col in results_df.columns:
            mae_mean = results_df[mae_col].mean()
            mae_std = results_df[mae_col].std()
            print(f"{method:15s} - MAE: {mae_mean:8.3f} ± {mae_std:6.3f} Pa")
    
    # Also show in meters (approximate conversion)
    print("\n" + "="*70)
    print("APPROXIMATE HEIGHT ERROR (assuming ~10 Pa/m sensitivity)")
    print("="*70)
    for method in methods:
        mae_col = f'{method}_MAE'
        if mae_col in results_df.columns:
            mae_mean_pa = results_df[mae_col].mean()
            mae_mean_m = mae_mean_pa / 10.0  # Rough conversion
            print(f"{method:15s} - MAE: ~{mae_mean_m:6.2f} m")
    
    print("\nResults saved to: experiments/01_baseline_comparisons/results.csv")
    return results_df


if __name__ == "__main__":
    results = run_baseline_comparisons()
