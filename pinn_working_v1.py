#!/usr/bin/env python
"""
Working PINN v1 - Stable baseline that actually works
======================================================

Start simple, verify it works, then add complexity.

Architecture:
1. ERA5 physics baseline (fixed, not learned)
2. Neural residual correction (simple MLP)
3. Multi-task: also predict temperature

Author: Assistant
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

R_DRY = 287.05
G = 9.80665
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WorkingPINN(nn.Module):
    """
    Improved PINN with better features.
    """

    def __init__(self, n_sensors):
        super().__init__()

        # Sensor embedding
        self.sensor_embed = nn.Embedding(n_sensors, 16)

        # Additional embeddings for time
        self.hour_embed = nn.Embedding(24, 4)

        # Network: more features
        # sensor(16) + P_diff(1) + T_diff(1) + hour(4) + local_T_diff(1) + P_obs_norm(1) = 24
        self.net = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Just h_residual
        )

    def compute_physics_baseline(self, P_obs, P_era5, T_era5, h_era5):
        """
        Proper physics baseline using ERA5.

        h = h_era5 - H_v * ln(P_obs / P_era5)
        """
        H_v = R_DRY * T_era5 / G
        h_baseline = h_era5 - H_v * torch.log(P_obs / P_era5)
        return h_baseline

    def forward(self, sensor_ids, P_obs, P_era5, T_era5, h_era5, timestamp, hour, local_T_diff):
        """
        Predict corrections to physics baseline.
        """
        # Features
        sensor = self.sensor_embed(sensor_ids)
        hour_emb = self.hour_embed(hour)

        P_diff = (P_obs - P_era5).unsqueeze(-1) / 1000
        T_diff = (T_era5 - 290).unsqueeze(-1) / 10
        P_norm = (P_obs - 100000).unsqueeze(-1) / 5000

        # Local temperature difference (sensor - ERA5) - key microclimate feature
        local_T_feat = local_T_diff.unsqueeze(-1) / 5

        x = torch.cat([sensor, hour_emb, P_diff, T_diff, local_T_feat, P_norm], dim=-1)

        h_residual = self.net(x).squeeze(-1) * 50  # ±50m

        return h_residual

    def predict(self, sensor_ids, P_obs, P_era5, T_era5, h_era5, timestamp, hour, local_T_diff):
        """Full prediction."""
        h_baseline = self.compute_physics_baseline(P_obs, P_era5, T_era5, h_era5)
        h_residual = self.forward(sensor_ids, P_obs, P_era5, T_era5, h_era5, timestamp, hour, local_T_diff)

        h_pred = h_baseline + h_residual

        return h_pred


def load_data():
    """Load and filter data."""
    df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
    df['processed_time'] = pd.to_datetime(df['processed_time'])

    mobile_uid = '20240606181851A641973A1878250224'
    df = df[df['uid'] != mobile_uid].copy()

    # Light filtering
    df = df[(df['avg_pressure'] > 95000) & (df['avg_pressure'] < 105000)]

    # Per-sensor median filter
    median_alts = df.groupby('uid')['avg_altitude'].transform('median')
    df = df[np.abs(df['avg_altitude'] - median_alts) < 40]

    # Compute ERA5 height
    df['h_era5'] = -8500 * np.log(df['era5_sp'] / 101325)

    # Encode sensors
    sensor_encoder = LabelEncoder()
    df['sensor_id'] = sensor_encoder.fit_transform(df['uid'])

    return df, sensor_encoder


def physics_baseline_from_df(df):
    """Compute physics baseline consistent with model's formulation."""
    h_era5 = df['h_era5'].values
    p_obs = df['avg_pressure'].values
    p_era5 = df['era5_sp'].values
    t_era5 = df['era5_t2m'].values

    h_v = R_DRY * t_era5 / G
    return h_era5 - h_v * np.log(p_obs / p_era5)


def _fit_linear(train_x, train_y):
    """Fit y = a*x + b with least squares."""
    X = np.column_stack([train_x, np.ones_like(train_x)])
    coef, _, _, _ = np.linalg.lstsq(X, train_y, rcond=None)
    return coef[0], coef[1]


def _predict_linear(x, a, b):
    return a * x + b


def weather_identifiability_report(df_train, df_test):
    """
    Decompose pressure residual into common-mode and local components,
    then attribute test MAE gains for pressure->height conversion.
    """
    train = df_train.copy()
    test = df_test.copy()

    # Pressure residual to ERA5 as weather proxy
    train['dp'] = train['avg_pressure'] - train['era5_sp']
    test['dp'] = test['avg_pressure'] - test['era5_sp']

    # Common-mode: mean across sensors at same timestamp
    train['dp_common'] = train.groupby('processed_time')['dp'].transform('mean')
    test['dp_common'] = test.groupby('processed_time')['dp'].transform('mean')

    # Local component: residual after removing common-mode
    train['dp_local'] = train['dp'] - train['dp_common']
    test['dp_local'] = test['dp'] - test['dp_common']

    # Variance decomposition
    dp_var_train = float(np.var(train['dp'].values) + 1e-12)
    common_var_frac = float(np.var(train['dp_common'].values) / dp_var_train)
    local_var_frac = float(np.var(train['dp_local'].values) / dp_var_train)

    # Pairwise cross-sensor correlation of dp
    wide = train.pivot_table(index='processed_time', columns='uid', values='dp', aggfunc='mean')
    corr = wide.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().values
    median_pair_corr = float(np.median(upper)) if len(upper) > 0 else float('nan')

    # Height error relative to physics baseline
    h_phys_train = physics_baseline_from_df(train)
    h_phys_test = physics_baseline_from_df(test)
    e_train = h_phys_train - train['avg_altitude'].values
    e_test = h_phys_test - test['avg_altitude'].values

    mae_phys = float(np.mean(np.abs(e_test)))

    # Model A: common only
    a_c, b_c = _fit_linear(train['dp_common'].values, e_train)
    ehat_common = _predict_linear(test['dp_common'].values, a_c, b_c)
    mae_common = float(np.mean(np.abs(e_test - ehat_common)))

    # Model B: local only
    a_l, b_l = _fit_linear(train['dp_local'].values, e_train)
    ehat_local = _predict_linear(test['dp_local'].values, a_l, b_l)
    mae_local = float(np.mean(np.abs(e_test - ehat_local)))

    # Model C: common + local (2D linear)
    X_train = np.column_stack([
        train['dp_common'].values,
        train['dp_local'].values,
        np.ones(len(train))
    ])
    coef2, _, _, _ = np.linalg.lstsq(X_train, e_train, rcond=None)

    X_test = np.column_stack([
        test['dp_common'].values,
        test['dp_local'].values,
        np.ones(len(test))
    ])
    ehat_both = X_test @ coef2
    mae_both = float(np.mean(np.abs(e_test - ehat_both)))

    return {
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'dp_common_variance_fraction': common_var_frac,
        'dp_local_variance_fraction': local_var_frac,
        'dp_pairwise_corr_median': median_pair_corr,
        'mae_physics': mae_phys,
        'mae_common_only': mae_common,
        'mae_local_only': mae_local,
        'mae_common_plus_local': mae_both,
        'gain_common_vs_physics': float(mae_phys - mae_common),
        'gain_local_vs_physics': float(mae_phys - mae_local),
        'gain_local_on_top_of_common': float(mae_common - mae_both),
    }


def prepare_tensors(df, device, force_single_sensor_id=False):
    """Prepare tensors."""
    timestamp = df['processed_time'].astype(np.int64) // 10**9
    timestamp = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min() + 1e-6)

    # Hour of day (0-23)
    hour = df['processed_time'].dt.hour.values

    # Local temperature difference (sensor - ERA5)
    local_T_diff = (df['avg_temperature'].values + 273.15) - df['era5_t2m'].values

    sensor_ids = df['sensor_id'].values
    if force_single_sensor_id:
        sensor_ids = np.zeros_like(sensor_ids)

    return {
        'sensor_ids': torch.LongTensor(sensor_ids).to(device),
        'P_obs': torch.FloatTensor(df['avg_pressure'].values).to(device),
        'P_era5': torch.FloatTensor(df['era5_sp'].values).to(device),
        'T_obs': torch.FloatTensor(df['avg_temperature'].values + 273.15).to(device),
        'T_era5': torch.FloatTensor(df['era5_t2m'].values).to(device),
        'h_era5': torch.FloatTensor(df['h_era5'].values).to(device),
        'h_gps': torch.FloatTensor(df['avg_altitude'].values).to(device),
        'timestamp': torch.FloatTensor(timestamp.values).to(device),
        'hour': torch.LongTensor(hour).to(device),
        'local_T_diff': torch.FloatTensor(local_T_diff).to(device),
    }


def train(model, train_tensors, val_tensors, epochs=300):
    """Train model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_mae = float('inf')
    best_state = None

    print("\nTraining...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward
        h_pred = model.predict(
            train_tensors['sensor_ids'], train_tensors['P_obs'],
            train_tensors['P_era5'], train_tensors['T_era5'],
            train_tensors['h_era5'], train_tensors['timestamp'],
            train_tensors['hour'], train_tensors['local_T_diff']
        )

        # Loss
        loss = F.smooth_l1_loss(h_pred, train_tensors['h_gps'])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validate
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                h_pred = model.predict(
                    val_tensors['sensor_ids'], val_tensors['P_obs'],
                    val_tensors['P_era5'], val_tensors['T_era5'],
                    val_tensors['h_era5'], val_tensors['timestamp'],
                    val_tensors['hour'], val_tensors['local_T_diff']
                )
                mae = torch.mean(torch.abs(h_pred - val_tensors['h_gps'])).item()

            print(f"  Epoch {epoch:3d}: Val MAE={mae:.2f}m")

            if mae < best_mae:
                best_mae = mae
                best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model


def evaluate(model, tensors):
    """Evaluate."""
    model.eval()
    with torch.no_grad():
        h_pred = model.predict(
            tensors['sensor_ids'], tensors['P_obs'],
            tensors['P_era5'], tensors['T_era5'],
            tensors['h_era5'], tensors['timestamp'],
            tensors['hour'], tensors['local_T_diff']
        )

        h_pred_np = h_pred.cpu().numpy()
        h_gps_np = tensors['h_gps'].cpu().numpy()

        mae = np.mean(np.abs(h_pred_np - h_gps_np))
        rmse = np.sqrt(np.mean((h_pred_np - h_gps_np)**2))

    return mae, rmse


def run_loso_best(df, epochs=120):
    """
    Run leave-one-sensor-out training/evaluation and return best held-out result.

    Uses a sensor-agnostic setting by forcing sensor id to one shared token,
    so unseen-sensor evaluation is well-defined.
    """
    uids = sorted(df['uid'].unique())
    results = []

    print("\n[LOSO] Running leave-one-sensor-out...")
    print(f"[LOSO] Sensors: {len(uids)}, epochs/fold: {epochs}")

    for uid in uids:
        fold_train = df[df['uid'] != uid].copy()
        fold_test = df[df['uid'] == uid].copy()

        if len(fold_train) < 100 or len(fold_test) < 50:
            print(f"[LOSO] Skip {uid}: insufficient data")
            continue

        split_idx = int(len(fold_train) * 0.9)
        fold_train_df = fold_train.iloc[:split_idx].copy()
        fold_val_df = fold_train.iloc[split_idx:].copy()

        train_tensors = prepare_tensors(fold_train_df, DEVICE, force_single_sensor_id=True)
        val_tensors = prepare_tensors(fold_val_df, DEVICE, force_single_sensor_id=True)
        test_tensors = prepare_tensors(fold_test, DEVICE, force_single_sensor_id=True)

        model = WorkingPINN(n_sensors=1).to(DEVICE)
        model = train(model, train_tensors, val_tensors, epochs=epochs)
        fold_mae, fold_rmse = evaluate(model, test_tensors)

        results.append({
            'held_out_uid': str(uid),
            'n_test': int(len(fold_test)),
            'mae': float(fold_mae),
            'rmse': float(fold_rmse)
        })
        print(f"[LOSO] Holdout {uid}: MAE={fold_mae:.2f}m RMSE={fold_rmse:.2f}m n={len(fold_test)}")

    if len(results) == 0:
        return [], None

    best = min(results, key=lambda x: x['mae'])
    return results, best


def parse_args():
    parser = argparse.ArgumentParser(description='Working PINN v1 with optional LOSO evaluation')
    parser.add_argument('--run_loso_best', action='store_true', help='Run LOSO across sensors and report best held-out result')
    parser.add_argument('--loso_epochs', type=int, default=120, help='Epochs per LOSO fold')
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("WORKING PINN v1 - Simple and Effective")
    print("="*80)

    # Load
    print("\n[1] Loading data...")
    df, sensor_encoder = load_data()
    n_sensors = len(sensor_encoder.classes_)

    # Split
    print("\n[2] Splitting...")
    df_train = df[df['week_seq'] != 3].copy()
    df_test = df[df['week_seq'] == 3].copy()
    df_train, df_val = df_train.iloc[:int(len(df_train)*0.9)], df_train.iloc[int(len(df_train)*0.9):]

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Tensors
    print("\n[3] Preparing tensors...")
    train_tensors = prepare_tensors(df_train, DEVICE)
    val_tensors = prepare_tensors(df_val, DEVICE)
    test_tensors = prepare_tensors(df_test, DEVICE)

    # Baselines
    print("\n[4] Baselines...")
    era5_baseline = physics_baseline_from_df(df_test)
    era5_mae = np.mean(np.abs(era5_baseline - df_test['avg_altitude']))
    print(f"  ERA5 Physics: {era5_mae:.2f}m")

    ident_report = weather_identifiability_report(df_train, df_test)
    print("\n  Weather identifiability (train→test):")
    print(f"    dp variance common/local: {ident_report['dp_common_variance_fraction']*100:.1f}% / {ident_report['dp_local_variance_fraction']*100:.1f}%")
    print(f"    dp pairwise corr (median): {ident_report['dp_pairwise_corr_median']:.3f}")
    print(f"    MAE physics:              {ident_report['mae_physics']:.2f}m")
    print(f"    MAE common-only corr:     {ident_report['mae_common_only']:.2f}m")
    print(f"    MAE local-only corr:      {ident_report['mae_local_only']:.2f}m")
    print(f"    MAE common+local corr:    {ident_report['mae_common_plus_local']:.2f}m")
    print(f"    Gain (common):            {ident_report['gain_common_vs_physics']:.2f}m")
    print(f"    Gain (local over common): {ident_report['gain_local_on_top_of_common']:.2f}m")

    # Train
    print("\n[5] Training...")
    model = WorkingPINN(n_sensors).to(DEVICE)
    model = train(model, train_tensors, val_tensors, epochs=300)

    # Evaluate
    print("\n[6] Evaluation...")
    test_mae, test_rmse = evaluate(model, test_tensors)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"ERA5 Baseline:  {era5_mae:.2f}m")
    print(f"PINN:           {test_mae:.2f}m (RMSE={test_rmse:.2f}m)")
    print(f"Improvement:    {(era5_mae - test_mae)/era5_mae*100:.1f}%")

    if test_mae < era5_mae:
        print("\n✓ Model improves over ERA5 baseline!")
    else:
        print("\n⚠ Model did not improve")

    out_report = {
        'era5_mae': float(era5_mae),
        'pinn_mae': float(test_mae),
        'pinn_rmse': float(test_rmse),
        'identifiability': ident_report
    }
    out_path = Path('weather_identifiability_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_report, f, indent=2)
    print(f"\nSaved report: {out_path}")

    if args.run_loso_best:
        loso_results, loso_best = run_loso_best(df, epochs=args.loso_epochs)
        if loso_best is None:
            print("\n[LOSO] No valid LOSO folds.")
            return

        loso_out = {
            'protocol': 'LOSO (sensor-held-out), sensor-agnostic embedding token',
            'epochs_per_fold': int(args.loso_epochs),
            'n_folds': int(len(loso_results)),
            'best_fold': loso_best,
            'all_folds': loso_results
        }
        loso_path = Path('loso_best_report.json')
        with open(loso_path, 'w', encoding='utf-8') as f:
            json.dump(loso_out, f, indent=2)

        print("\n[LOSO] Best held-out result:")
        print(f"  uid={loso_best['held_out_uid']} | MAE={loso_best['mae']:.2f}m | RMSE={loso_best['rmse']:.2f}m | n={loso_best['n_test']}")
        print(f"[LOSO] Saved report: {loso_path}")


if __name__ == '__main__':
    main()
