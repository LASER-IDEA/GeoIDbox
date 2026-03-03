"""
Ablation Studies Experiment

Tests 4 setups:
A: Base SIREN MLP only (No Curriculum, No P_bias)     → direct Δh, no physics guidance
B: Base SIREN + P_bias (No Curriculum)                → δP formulation, no curriculum
C: Base SIREN + Curriculum (No P_bias)                → direct Δh + curriculum, no δP
D: Full (SIREN + P_bias + Curriculum)                 → 3.55m MAE
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
from datetime import datetime
sys.path.insert(0, '/data/home/huxiao/workspace/GeoIDbox')

from height_field_project.physics_baseline import compute_physics_baseline, compute_virtual_temperature
from height_field_project.train_generalized_with_bias import compute_sensor_bias, BiasAwarePINNDataset
from height_field_project.neural_field_pinn_generalized import GeneralizedPressureCorrectionPINN
from height_field_project.train_pinn import set_seed, parse_timestamp
from torch.utils.data import DataLoader


R_DRY_AIR = 287.05
G_STANDARD = 9.80665


class DirectHeightPINN(nn.Module):
    """Base SIREN MLP that predicts direct height residual."""
    def __init__(self, hash_levels=16, hash_features=4, hidden_dim=256, temporal_freqs=6):
        super().__init__()
        from height_field_project.neural_field_pinn_generalized import (
            MultiResolutionHashEncoding, FourierTemporalEncoding, SirenLayer
        )

        self.hash_encoding = MultiResolutionHashEncoding(
            n_levels=hash_levels, n_features=hash_features
        )
        self.temporal_encoding = FourierTemporalEncoding(
            n_frequencies=temporal_freqs
        )

        # input: hash + z + temporal + T + RH
        in_dim = self.hash_encoding.out_dim + 1 + self.temporal_encoding.out_dim + 1 + 1

        layers = []
        layers.append(SirenLayer(in_dim, hidden_dim, w0=1.0, is_first=True))
        for _ in range(2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, w0=1.0))
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

        with torch.no_grad():
            self.mlp[-1].weight.fill_(0.0)
            self.mlp[-1].bias.fill_(0.0)

    def forward(self, lat, lon, z, t, temperature, humidity):
        lat_norm = (lat + 90.0) / 180.0
        lon_norm = lon % 360.0 / 360.0
        coords = torch.stack([lat_norm, lon_norm], dim=-1)

        h_spatial = self.hash_encoding(coords)
        h_temporal = self.temporal_encoding(t / 3600.0)

        features = torch.cat([
            h_spatial, z.unsqueeze(-1), h_temporal,
            temperature.unsqueeze(-1), humidity.unsqueeze(-1)
        ], dim=-1)

        return self.mlp(features).squeeze(-1)


def train_direct_height_pinn(model, train_loader, val_loader, test_loader,
                             device, epochs=80):
    """Train PINN that predicts direct height residual (Delta h)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            delta_h = model(
                batch['lat'].to(device), batch['lon'].to(device),
                batch['z'].to(device), batch['t'].to(device),
                batch['temperature'].to(device), batch['humidity'].to(device)
            )

            # Predict h_pred = h_phy + delta_h
            h_pred = batch['h_phys'].to(device) + delta_h

            loss = torch.mean(torch.abs(h_pred - batch['h_gnss'].to(device)))
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_preds = []
            val_gnss = []
            with torch.no_grad():
                for batch in val_loader:
                    delta_h = model(
                        batch['lat'].to(device), batch['lon'].to(device),
                        batch['z'].to(device), batch['t'].to(device),
                        batch['temperature'].to(device), batch['humidity'].to(device)
                    )
                    h_pred = batch['h_phys'].to(device) + delta_h

                    val_preds.append(h_pred.cpu())
                    val_gnss.append(batch['h_gnss'])

            val_preds = torch.cat(val_preds)
            val_gnss = torch.cat(val_gnss)
            val_mae = torch.mean(torch.abs(val_preds - val_gnss)).item()

            if val_mae < best_val_mae - 1e-4:
                best_val_mae = val_mae
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

    # Load best and test
    if 'best_state' in locals():
        model.load_state_dict(best_state)

    model.eval()
    test_preds = []
    test_gnss = []
    with torch.no_grad():
        for batch in test_loader:
            delta_h = model(
                batch['lat'].to(device), batch['lon'].to(device),
                batch['z'].to(device), batch['t'].to(device),
                batch['temperature'].to(device), batch['humidity'].to(device)
            )
            h_pred = batch['h_phys'].to(device) + delta_h

            test_preds.append(h_pred.cpu())
            test_gnss.append(batch['h_gnss'])

    test_preds = torch.cat(test_preds)
    test_gnss = torch.cat(test_gnss)

    test_mae = torch.mean(torch.abs(test_preds - test_gnss)).item()

    return test_mae


def create_curriculum_splits(altitudes: np.ndarray):
    """3-stage curriculum: Stage1 h<100m, Stage2 h<200m, Stage3 all."""
    mask_1 = altitudes < 100
    mask_2 = altitudes < 200
    mask_3 = np.ones(len(altitudes), dtype=bool)
    return mask_1, mask_2, mask_3


def train_direct_height_with_curriculum(model, train_df, test_df, phys_params,
                                         device, stage_epochs=30):
    """
    Setup C: DirectHeightPINN (no P_bias) trained with 3-stage altitude curriculum.
    """
    from height_field_project.train_pinn import PINNDataset

    def make_loader(sub_df, shuffle=True):
        ds = PINNDataset(
            lat=sub_df['avg_latitude'].values,
            lon=sub_df['avg_longitude'].values,
            z=sub_df['avg_altitude'].values,
            t=sub_df['timestamp'].values,
            temperature=sub_df['avg_temperature'].values,
            humidity=sub_df['avg_humidity'].values,
            sensor_id=np.zeros(len(sub_df), dtype=np.int64),
            p_obs=sub_df['avg_pressure'].values,
            h_gnss=sub_df['avg_altitude'].values,
            h_phys=sub_df['h_phys_hae'].values
        )
        return DataLoader(ds, batch_size=2048, shuffle=shuffle)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    altitudes = train_df['avg_altitude'].values
    mask_1, mask_2, mask_3 = create_curriculum_splits(altitudes)

    for stage_idx, mask in enumerate([mask_1, mask_2, mask_3], start=1):
        stage_df = train_df[mask].copy()
        if len(stage_df) == 0:
            stage_df = train_df.copy()
        stage_loader = make_loader(stage_df, shuffle=True)

        for _ in range(stage_epochs):
            model.train()
            for batch in stage_loader:
                optimizer.zero_grad()
                delta_h = model(
                    batch['lat'].to(device), batch['lon'].to(device),
                    batch['z'].to(device), batch['t'].to(device),
                    batch['temperature'].to(device), batch['humidity'].to(device)
                )
                h_pred = batch['h_phys'].to(device) + delta_h
                loss = torch.mean(torch.abs(h_pred - batch['h_gnss'].to(device)))
                loss.backward()
                optimizer.step()
            scheduler.step()

    # Evaluate on test set
    test_loader = make_loader(test_df, shuffle=False)
    model.eval()
    preds, gnss = [], []
    with torch.no_grad():
        for batch in test_loader:
            delta_h = model(
                batch['lat'].to(device), batch['lon'].to(device),
                batch['z'].to(device), batch['t'].to(device),
                batch['temperature'].to(device), batch['humidity'].to(device)
            )
            h_pred = batch['h_phys'].to(device) + delta_h
            preds.append(h_pred.cpu())
            gnss.append(batch['h_gnss'])

    preds = torch.cat(preds)
    gnss = torch.cat(gnss)
    return torch.mean(torch.abs(preds - gnss)).item()


def run_ablation_studies():

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/sensor_data_filtered.csv')
    df, phys_params = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)
    df['timestamp'] = df['processed_time'].apply(parse_timestamp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sensors = df['uid'].unique()

    def create_dataset(data_df):
        from height_field_project.train_pinn import PINNDataset
        return PINNDataset(
            lat=data_df['avg_latitude'].values,
            lon=data_df['avg_longitude'].values,
            z=data_df['avg_altitude'].values,
            t=data_df['timestamp'].values,
            temperature=data_df['avg_temperature'].values,
            humidity=data_df['avg_humidity'].values,
            sensor_id=np.zeros(len(data_df), dtype=np.int64),
            p_obs=data_df['avg_pressure'].values,
            h_gnss=data_df['avg_altitude'].values,
            h_phys=data_df['h_phys_hae'].values
        )

    direct_height_maes = []
    CACHE_A = 'experiments/02_ablation_studies/setup_a_cache.npy'

    if os.path.exists(CACHE_A):
        direct_height_maes = list(np.load(CACHE_A))
        print(f"\n[Cache] Loaded Setup A from {CACHE_A}: {[f'{v:.3f}' for v in direct_height_maes]}")
    else:
        print("\n" + "="*70)
        print("Setup A: Base Neural Network (Direct Height \u0394h, No P_bias, No Curriculum)")
        print("="*70)

        for fold, held_out_sensor in enumerate(sensors):
            print(f"\nEvaluating Fold {fold}: {held_out_sensor}")
            test_mask = df['uid'] == held_out_sensor
            train_df = df[~test_mask].copy()
            test_df = df[test_mask].copy()

            # Train/val split
            train_val = train_df
            n_val = int(len(train_val) * 0.15)
            val_df = train_val.tail(n_val)
            train_df = train_val.head(len(train_val) - n_val)

            try:
                set_seed(42 + fold)
                model_a = DirectHeightPINN().to(device)

                ds_train_a = create_dataset(train_df)
                ds_val_a = create_dataset(val_df)
                ds_test_a = create_dataset(test_df)

                train_loader_a = DataLoader(ds_train_a, batch_size=2048, shuffle=True)
                val_loader_a = DataLoader(ds_val_a, batch_size=2048, shuffle=False)
                test_loader_a = DataLoader(ds_test_a, batch_size=2048, shuffle=False)

                mae_a = train_direct_height_pinn(
                    model_a, train_loader_a, val_loader_a, test_loader_a,
                    device, epochs=80
                )

                print(f"Fold {fold} - Setup A (Direct Height) Test MAE: {mae_a:.3f} m")
                direct_height_maes.append(mae_a)
            except Exception as e:
                print(f"Setup A Fold {fold} failed: {e}")
                import traceback
                traceback.print_exc()
                direct_height_maes.append(np.nan)

        np.save(CACHE_A, np.array(direct_height_maes))
        print(f"\n[Cache] Saved Setup A results to {CACHE_A}")

    mean_mae_a = np.nanmean(direct_height_maes)
    print(f"\n-> Overall Setup A (Direct Height) MAE: {mean_mae_a:.3f} m")

    # ── Setup C: DirectHeightPINN + Curriculum (no P_bias) ────────────────
    print("\n" + "="*70)
    print("Setup C: Base SIREN + Curriculum (Direct Δh, No P_bias)")
    print("="*70)

    direct_curriculum_maes = []
    for fold, held_out_sensor in enumerate(sensors):
        print(f"\nEvaluating Fold {fold}: {held_out_sensor}")
        test_mask = df['uid'] == held_out_sensor
        train_df_fold = df[~test_mask].copy()
        test_df_fold = df[test_mask].copy()

        try:
            from height_field_project.train_pinn import set_seed as _set_seed
            _set_seed(42 + fold)
            model_c = DirectHeightPINN().to(device)
            mae_c = train_direct_height_with_curriculum(
                model_c, train_df_fold, test_df_fold, phys_params,
                device, stage_epochs=30
            )
            print(f"Fold {fold} - Setup C (Direct+Curriculum) Test MAE: {mae_c:.3f} m")
            direct_curriculum_maes.append(mae_c)
        except Exception as e:
            print(f"Setup C Fold {fold} failed: {e}")
            import traceback; traceback.print_exc()
            direct_curriculum_maes.append(np.nan)

    mean_mae_c = np.nanmean(direct_curriculum_maes)
    print(f"\n-> Overall Setup C (Direct+Curriculum) MAE: {mean_mae_c:.3f} m")

    results = []
    results.append({'setup': 'Pure Physics Baseline', 'mae': 36.96})
    results.append({'setup': 'Setup A: Base NN (Direct Height)', 'mae': mean_mae_a})
    results.append({'setup': 'Setup C: Direct Height + Curriculum (No P_bias)', 'mae': mean_mae_c})

    # Setup B & Setup D are read directly from their respective JSON files
    import json

    print("\n" + "="*70)
    print("Reading Results for Setup B and Setup D from previous LOSO runs")
    print("="*70)

    bias_aware_json = 'height_field_project/loso_bias_aware_results/loso_summary.json'
    if os.path.exists(bias_aware_json):
        with open(bias_aware_json, 'r') as f:
            data = json.load(f)
            mae_b = data.get('mean_mae', np.nan)
            results.append({'setup': 'Setup B: Bias-Aware Formulation (δP)', 'mae': mae_b})
            print(f"Setup B (Bias-Aware) loaded: {mae_b:.3f} m")
    else:
        print("WARNING: loso_bias_aware_results/loso_summary.json not found!")

    curriculum_json = 'height_field_project/loso_curriculum_results/loso_summary.json'
    if os.path.exists(curriculum_json):
        with open(curriculum_json, 'r') as f:
            data = json.load(f)
            mae_d = data.get('mean_mae', np.nan)
            results.append({'setup': 'Setup D: Bias-Aware + Curriculum Learning', 'mae': mae_d})
            print(f"Setup D (Curriculum) loaded: {mae_d:.3f} m")
    else:
        print("WARNING: loso_curriculum_results/loso_summary.json not found!")

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('experiments/02_ablation_studies', exist_ok=True)
    results_df.to_csv('experiments/02_ablation_studies/results.csv', index=False)

    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS (8-Fold LOSO Mean)")
    print("="*70)
    print(results_df.to_string(index=False))
    print("\nResults saved to: experiments/02_ablation_studies/results.csv")

    return results_df


if __name__ == "__main__":
    import numpy as np
    results = run_ablation_studies()
