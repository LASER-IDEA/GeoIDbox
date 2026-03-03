"""
Baseline Comparisons Experiment (Fixed Version)

Compares PINN against:
1. IDW (Inverse Distance Weighting)
2. Ordinary Kriging
3. Random Forest
4. XGBoost

Uses full data for RF/XGB, subsampled for IDW/Kriging.
"""
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        self.known_coords = coords
        self.known_values = values

    def predict(self, coords):
        distances = cdist(coords, self.known_coords)
        distances = np.maximum(distances, 1e-10)
        weights = 1.0 / (distances ** self.power)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return np.sum(weights * self.known_values, axis=1)


def run_baseline_comparisons():
    """Run all baseline comparisons."""
    print("="*70)
    print("EXPERIMENT 1: Baseline Comparisons (Fixed Version)")
    print("="*70)

    print("\nLoading data...")
    df = pd.read_csv('data/sensor_data_filtered.csv')
    df, phys_params = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)

    features = ['avg_latitude', 'avg_longitude', 'avg_temperature', 'avg_humidity']
    # NOTE: avg_altitude is EXCLUDED intentionally.
    # pressure_bias = P_obs - P_barometric(avg_altitude), so including avg_altitude
    # creates a circular dependency between feature and target, and causes severe
    # distribution-shift failure (RF/XGB cannot extrapolate) under LOSO validation.
    # IDW/Kriging use only lat/lon; we use lat/lon + T + RH for a fair comparison.
    target = 'pressure_bias'

    results = []
    sensors = df['uid'].unique()

    for fold, held_out_sensor in enumerate(sensors):
        print(f"\n{'='*70}")
        print(f"Fold {fold}: Held-out sensor {held_out_sensor[:20]}")
        print(f"{'='*70}")

        test_mask = df['uid'] == held_out_sensor
        train_df_full = df[~test_mask].copy()
        test_df = df[test_mask].copy()

        # Full data for ML methods
        X_train_full = train_df_full[features].values
        y_train_full = train_df_full[target].values
        X_test = test_df[features].values
        y_test = test_df[target].values

        coords_train_full = train_df_full[['avg_latitude', 'avg_longitude']].values
        coords_test = test_df[['avg_latitude', 'avg_longitude']].values

        fold_results = {'fold': fold, 'held_out': held_out_sensor[:20]}

        # 1. IDW with subsampling
        print("\n  Running IDW (subsampled)...")
        try:
            t0 = time.time()
            # Subsample for IDW speed
            max_idw_points = 3000
            if len(coords_train_full) > max_idw_points:
                indices = np.random.choice(len(coords_train_full), max_idw_points, replace=False)
                coords_train_idw = coords_train_full[indices]
                y_train_idw = y_train_full[indices]
            else:
                coords_train_idw = coords_train_full
                y_train_idw = y_train_full

            idw = IDWInterpolator(power=2)
            idw.fit(coords_train_idw, y_train_idw)
            y_pred_idw = idw.predict(coords_test)

            # Convert pressure correction to height prediction in meters
            p_corrected = test_df['avg_pressure'].values + y_pred_idw

            t_celsius = test_df['avg_temperature'].values
            e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (test_df['avg_humidity'].values / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            H = 287.05 * t_v / 9.80665
            h_pred_idw = H * np.log(phys_params.p_ref / p_corrected)

            mae_idw = np.mean(np.abs(h_pred_idw - test_df['avg_altitude'].values))
            rmse_idw = np.sqrt(np.mean((h_pred_idw - test_df['avg_altitude'].values)**2))

            fold_results['IDW_MAE'] = mae_idw
            fold_results['IDW_RMSE'] = rmse_idw
            print(f"    IDW - MAE: {mae_idw:.3f} m, RMSE: {rmse_idw:.3f} m ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"    IDW failed: {e}")
            import traceback
            traceback.print_exc()
            fold_results['IDW_MAE'] = np.nan
            fold_results['IDW_RMSE'] = np.nan

        # 2. Ordinary Kriging with subsampling
        print("\n  Running Ordinary Kriging (subsampled)...")
        try:
            t0 = time.time()
            max_krig_points = 2000
            if len(coords_train_full) > max_krig_points:
                indices = np.random.choice(len(coords_train_full), max_krig_points, replace=False)
                coords_train_krig = coords_train_full[indices]
                y_train_krig = y_train_full[indices]
            else:
                coords_train_krig = coords_train_full
                y_train_krig = y_train_full

            OK = OrdinaryKriging(
                coords_train_krig[:, 0], coords_train_krig[:, 1], y_train_krig,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False,
                nlags=10
            )

            y_pred_krig, ss = OK.execute('points', coords_test[:, 0], coords_test[:, 1])
            y_pred_krig = np.array(y_pred_krig).flatten()

            # Convert pressure correction to height prediction in meters
            p_corrected = test_df['avg_pressure'].values + y_pred_krig

            t_celsius = test_df['avg_temperature'].values
            e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (test_df['avg_humidity'].values / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            H = 287.05 * t_v / 9.80665
            h_pred_krig = H * np.log(phys_params.p_ref / p_corrected)

            mae_krig = np.mean(np.abs(h_pred_krig - test_df['avg_altitude'].values))
            rmse_krig = np.sqrt(np.mean((h_pred_krig - test_df['avg_altitude'].values)**2))

            fold_results['Kriging_MAE'] = mae_krig
            fold_results['Kriging_RMSE'] = rmse_krig
            print(f"    Kriging - MAE: {mae_krig:.3f} m, RMSE: {rmse_krig:.3f} m ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"    Kriging failed: {e}")
            import traceback
            traceback.print_exc()
            fold_results['Kriging_MAE'] = np.nan
            fold_results['Kriging_RMSE'] = np.nan

        # 3. Random Forest (FULL DATA)
        print("\n  Running Random Forest (full data)...")
        try:
            t0 = time.time()
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_train_full, y_train_full)
            y_pred_rf = rf.predict(X_test)

            # Convert pressure correction to height prediction in meters
            p_corrected = test_df['avg_pressure'].values + y_pred_rf

            t_celsius = test_df['avg_temperature'].values
            e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (test_df['avg_humidity'].values / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            H = 287.05 * t_v / 9.80665
            h_pred_rf = H * np.log(phys_params.p_ref / p_corrected)

            mae_rf = np.mean(np.abs(h_pred_rf - test_df['avg_altitude'].values))
            rmse_rf = np.sqrt(np.mean((h_pred_rf - test_df['avg_altitude'].values)**2))

            fold_results['RF_MAE'] = mae_rf
            fold_results['RF_RMSE'] = rmse_rf
            print(f"    Random Forest - MAE: {mae_rf:.3f} m, RMSE: {rmse_rf:.3f} m ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"    Random Forest failed: {e}")
            import traceback
            traceback.print_exc()
            fold_results['RF_MAE'] = np.nan
            fold_results['RF_RMSE'] = np.nan

        # 4. XGBoost (FULL DATA)
        print("\n  Running XGBoost (full data)...")
        try:
            t0 = time.time()
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_full, y_train_full)
            y_pred_xgb = xgb_model.predict(X_test)

            # Convert pressure correction to height prediction in meters
            p_corrected = test_df['avg_pressure'].values + y_pred_xgb

            t_celsius = test_df['avg_temperature'].values
            e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (test_df['avg_humidity'].values / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            H = 287.05 * t_v / 9.80665
            h_pred_xgb = H * np.log(phys_params.p_ref / p_corrected)

            mae_xgb = np.mean(np.abs(h_pred_xgb - test_df['avg_altitude'].values))
            rmse_xgb = np.sqrt(np.mean((h_pred_xgb - test_df['avg_altitude'].values)**2))

            fold_results['XGB_MAE'] = mae_xgb
            fold_results['XGB_RMSE'] = rmse_xgb
            print(f"    XGBoost - MAE: {mae_xgb:.3f} m, RMSE: {rmse_xgb:.3f} m ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"    XGBoost failed: {e}")
            import traceback
            traceback.print_exc()
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
            mae_mean_m = results_df[mae_col].mean()
            mae_std_m = results_df[mae_col].std()
            print(f"{method:15s} - MAE: {mae_mean_m:6.3f} ± {mae_std_m:5.2f} m")

    print("\nResults saved to: experiments/01_baseline_comparisons/results.csv")
    return results_df


if __name__ == "__main__":
    results = run_baseline_comparisons()
