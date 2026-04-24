"""
Deep Learning Baselines for R1.3 Rebuttal

Adds 4 DL baselines to the comparison table:
1. Standard MLP (ReLU) predicting Δh directly
2. SIREN MLP predicting Δh directly (from ablation Setup A cache)
3. Hash-SIREN predicting δP without P_bias feature
4. TabNet on tabular features

All evaluated under strict 8-fold LOSO.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_generalized_with_bias import compute_sensor_bias
from height_field_project.train_pinn import set_seed, parse_timestamp
from height_field_project.neural_field_pinn_generalized import (
    MultiResolutionHashEncoding,
    FourierTemporalEncoding,
    SirenLayer,
)

warnings.filterwarnings("ignore")

R_DRY_AIR = 287.05
G_STANDARD = 9.80665
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pressure_bias_to_height_m(p_bias_pred, p_obs, t_celsius, rh, p_ref):
    e_sat = 610.94 * np.exp(17.625 * t_celsius / (t_celsius + 243.04))
    e = (rh / 100.0) * e_sat
    r = 0.62198 * e / (p_obs + p_bias_pred - e)
    t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
    H = R_DRY_AIR * t_v / G_STANDARD
    h_pred = H * np.log(p_ref / (p_obs + p_bias_pred))
    return h_pred


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.fill_(0.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HashSirenNoBias(nn.Module):
    def __init__(
        self,
        hash_levels=16,
        hash_features=4,
        hidden_dim=256,
        n_hidden=3,
        temporal_freqs=6,
    ):
        super().__init__()
        self.hash_encoding = MultiResolutionHashEncoding(
            n_levels=hash_levels, n_features=hash_features
        )
        self.temporal_encoding = FourierTemporalEncoding(n_frequencies=temporal_freqs)
        in_dim = self.hash_encoding.out_dim + 1 + self.temporal_encoding.out_dim + 1 + 1
        layers = [SirenLayer(in_dim, hidden_dim, w0=1.0, is_first=True)]
        for _ in range(n_hidden - 1):
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
        features = torch.cat(
            [
                h_spatial,
                z.unsqueeze(-1),
                h_temporal,
                temperature.unsqueeze(-1),
                humidity.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)


class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def train_simple_mlp(
    train_loader,
    val_loader,
    test_loader,
    phys_params,
    p_obs_test,
    t_test,
    rh_test,
    h_gnss_test,
    device,
    epochs=80,
    lr=1e-3,
):
    model = SimpleMLP(in_dim=5, hidden_dim=128, n_hidden=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )
    best_val = float("inf")
    patience = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x.to(device))
            loss = nn.functional.l1_loss(pred, y.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    vp.append(model(x.to(device)).cpu())
                    vt.append(y)
            vp = torch.cat(vp)
            vt = torch.cat(vt)
            vm = torch.mean(torch.abs(vp - vt)).item()
            if vm < best_val - 1e-4:
                best_val = vm
                patience = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if patience >= 15:
                break

    if "best_state" in locals():
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).cpu().numpy())
    delta_h = np.concatenate(preds)

    h_pred = h_gnss_test + delta_h
    mae = np.mean(np.abs(h_pred - h_gnss_test))
    rmse = np.sqrt(np.mean((h_pred - h_gnss_test) ** 2))
    return mae, rmse


def train_hash_siren_no_bias(
    train_loader,
    val_loader,
    test_loader,
    phys_params,
    p_obs_test,
    t_test,
    rh_test,
    h_gnss_test,
    device,
    epochs=80,
    lr=1e-3,
):
    model = HashSirenNoBias().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )
    best_val = float("inf")
    patience = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            dp = model(
                batch["lat"].to(device),
                batch["lon"].to(device),
                batch["z"].to(device),
                batch["t"].to(device),
                batch["temperature"].to(device),
                batch["humidity"].to(device),
            )
            p_corr = batch["p_obs"].to(device) + dp
            tc = batch["temperature"].to(device)
            es = 610.94 * torch.exp(17.625 * tc / (tc + 243.04))
            ev = (batch["humidity"].to(device) / 100.0) * es
            rv = 0.62198 * ev / (p_corr - ev)
            tv = (tc + 273.15) * (1 + 0.608 * rv)
            H = R_DRY_AIR * tv / G_STANDARD
            h_pred = H * torch.log(phys_params.p_ref / p_corr)
            loss = torch.mean(torch.abs(h_pred - batch["h_gnss"].to(device)))
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            vp, vg = [], []
            with torch.no_grad():
                for batch in val_loader:
                    dp = model(
                        batch["lat"].to(device),
                        batch["lon"].to(device),
                        batch["z"].to(device),
                        batch["t"].to(device),
                        batch["temperature"].to(device),
                        batch["humidity"].to(device),
                    )
                    p_corr = batch["p_obs"].to(device) + dp
                    tc = batch["temperature"].to(device)
                    es = 610.94 * torch.exp(17.625 * tc / (tc + 243.04))
                    ev = (batch["humidity"].to(device) / 100.0) * es
                    rv = 0.62198 * ev / (p_corr - ev)
                    tv = (tc + 273.15) * (1 + 0.608 * rv)
                    H = R_DRY_AIR * tv / G_STANDARD
                    hp = H * torch.log(phys_params.p_ref / p_corr)
                    vp.append(hp.cpu())
                    vg.append(batch["h_gnss"])
            vp = torch.cat(vp)
            vg = torch.cat(vg)
            vm = torch.mean(torch.abs(vp - vg)).item()
            if vm < best_val - 1e-4:
                best_val = vm
                patience = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if patience >= 15:
                break

    if "best_state" in locals():
        model.load_state_dict(best_state)

    model.eval()
    all_dp = []
    with torch.no_grad():
        for batch in test_loader:
            dp = model(
                batch["lat"].to(device),
                batch["lon"].to(device),
                batch["z"].to(device),
                batch["t"].to(device),
                batch["temperature"].to(device),
                batch["humidity"].to(device),
            )
            all_dp.append(dp.cpu().numpy())
    dp_arr = np.concatenate(all_dp)

    h_pred = pressure_bias_to_height_m(
        dp_arr, p_obs_test, t_test, rh_test, phys_params.p_ref
    )
    mae = np.mean(np.abs(h_pred - h_gnss_test))
    rmse = np.sqrt(np.mean((h_pred - h_gnss_test) ** 2))
    return mae, rmse


def train_tabnet(
    X_train,
    y_train,
    X_test,
    p_obs_test,
    t_test,
    rh_test,
    h_gnss_test,
    phys_params,
    max_epochs=200,
):
    from pytorch_tabnet.tab_model import TabNetRegressor

    model = TabNetRegressor(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_params={"step_size": 30, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42,
        verbose=0,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )

    y_train_2d = y_train.reshape(-1, 1)

    model.fit(
        X_train,
        y_train_2d,
        max_epochs=max_epochs,
        patience=20,
        batch_size=4096,
        virtual_batch_size=512,
    )

    y_pred = model.predict(X_test).flatten()

    h_pred = pressure_bias_to_height_m(
        y_pred, p_obs_test, t_test, rh_test, phys_params.p_ref
    )
    mae = np.mean(np.abs(h_pred - h_gnss_test))
    rmse = np.sqrt(np.mean((h_pred - h_gnss_test) ** 2))
    return mae, rmse


def make_neural_dataset(sub_df, with_timestamp=True):
    from height_field_project.train_generalized_with_bias import BiasAwarePINNDataset

    ts = (
        sub_df["timestamp"].values
        if "timestamp" in sub_df.columns
        else np.zeros(len(sub_df))
    )
    return BiasAwarePINNDataset(
        lat=sub_df["avg_latitude"].values,
        lon=sub_df["avg_longitude"].values,
        z=sub_df["avg_altitude"].values,
        t=ts,
        temperature=sub_df["avg_temperature"].values,
        humidity=sub_df["avg_humidity"].values,
        pressure_bias=sub_df["pressure_bias"].values
        if "pressure_bias" in sub_df.columns
        else np.zeros(len(sub_df)),
        sensor_id=np.zeros(len(sub_df), dtype=np.int64),
        p_obs=sub_df["avg_pressure"].values,
        h_gnss=sub_df["avg_altitude"].values,
        h_phys=sub_df["h_phys_hae"].values
        if "h_phys_hae" in sub_df.columns
        else np.zeros(len(sub_df)),
    )


def normalize_features(train_arr, test_arr):
    mu = train_arr.mean(axis=0)
    sigma = train_arr.std(axis=0) + 1e-8
    return (train_arr - mu) / sigma, (test_arr - mu) / sigma


def run_dl_baselines():
    print("=" * 70)
    print("EXPERIMENT 5: Deep Learning Baselines (R1.3 Rebuttal)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv("data/sensor_data_filtered.csv")
    df, phys_params = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, phys_params.p_ref)
    df["timestamp"] = df["processed_time"].apply(parse_timestamp)

    sensors = df["uid"].unique()
    results = []

    CACHE_SIREN = "experiments/02_ablation_studies/setup_a_cache.npy"
    if os.path.exists(CACHE_SIREN):
        siren_maes = list(np.load(CACHE_SIREN))
        print(f"[Cache] Loaded SIREN Δh results: {[f'{v:.2f}' for v in siren_maes]}")
    else:
        siren_maes = [np.nan] * 8
        print("[Warning] SIREN Δh cache not found, will use NaN")

    for fold, held_out_sensor in enumerate(sensors):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold}: {held_out_sensor[:25]}")
        print(f"{'=' * 70}")

        test_mask = df["uid"] == held_out_sensor
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()

        n_val = int(len(train_df) * 0.15)
        val_df = train_df.iloc[-n_val:]
        train_df_trunc = train_df.iloc[:-n_val]

        fold_res = {"fold": fold, "held_out": held_out_sensor[:25]}

        # --- 1. Plain MLP (Δh) ---
        print("\n  [1/4] Plain MLP (ReLU, Δh)...")
        try:
            t0 = time.time()
            set_seed(42 + fold)
            feat_cols = [
                "avg_latitude",
                "avg_longitude",
                "avg_temperature",
                "avg_humidity",
                "avg_altitude",
            ]
            X_tr = train_df_trunc[feat_cols].values
            X_va = val_df[feat_cols].values
            X_te = test_df[feat_cols].values
            y_tr = (
                train_df_trunc["avg_altitude"].values
                - train_df_trunc["h_phys_hae"].values
            )
            y_va = val_df["avg_altitude"].values - val_df["h_phys_hae"].values

            X_tr_n, X_te_n = normalize_features(np.vstack([X_tr, X_va]), X_te)
            X_tr_n_full = X_tr_n[: len(X_tr)]
            X_va_n = X_tr_n[len(X_tr) :]

            ds_tr = TabularDataset(X_tr_n_full, y_tr)
            ds_va = TabularDataset(X_va_n, y_va)
            ds_te = TabularDataset(X_te_n, np.zeros(len(X_te_n)))

            dl_tr = DataLoader(ds_tr, batch_size=2048, shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=2048, shuffle=False)
            dl_te = DataLoader(ds_te, batch_size=2048, shuffle=False)

            mae, rmse = train_simple_mlp(
                dl_tr,
                dl_va,
                dl_te,
                phys_params,
                test_df["avg_pressure"].values,
                test_df["avg_temperature"].values,
                test_df["avg_humidity"].values,
                test_df["avg_altitude"].values,
                DEVICE,
                epochs=80,
            )
            fold_res["MLP_MAE"] = mae
            fold_res["MLP_RMSE"] = rmse
            print(
                f"    MLP MAE: {mae:.3f} m, RMSE: {rmse:.3f} m ({time.time() - t0:.1f}s)"
            )
        except Exception as e:
            print(f"    MLP failed: {e}")
            import traceback

            traceback.print_exc()
            fold_res["MLP_MAE"] = np.nan
            fold_res["MLP_RMSE"] = np.nan

        # --- 2. SIREN Δh (from cache) ---
        fold_res["SIREN_MAE"] = siren_maes[fold]
        fold_res["SIREN_RMSE"] = np.nan
        if not np.isnan(siren_maes[fold]):
            print(f"  [2/4] SIREN Δh (cached): MAE = {siren_maes[fold]:.3f} m")

        # --- 3. Hash-SIREN (δP, no P_bias) ---
        print("\n  [3/4] Hash-SIREN (δP, no P_bias)...")
        try:
            t0 = time.time()
            set_seed(42 + fold)
            ds_tr_n = make_neural_dataset(train_df_trunc)
            ds_va_n = make_neural_dataset(val_df)
            ds_te_n = make_neural_dataset(test_df)

            dl_tr_n = DataLoader(ds_tr_n, batch_size=2048, shuffle=True)
            dl_va_n = DataLoader(ds_va_n, batch_size=2048, shuffle=False)
            dl_te_n = DataLoader(ds_te_n, batch_size=2048, shuffle=False)

            mae, rmse = train_hash_siren_no_bias(
                dl_tr_n,
                dl_va_n,
                dl_te_n,
                phys_params,
                test_df["avg_pressure"].values,
                test_df["avg_temperature"].values,
                test_df["avg_humidity"].values,
                test_df["avg_altitude"].values,
                DEVICE,
                epochs=80,
            )
            fold_res["HashSIREN_MAE"] = mae
            fold_res["HashSIREN_RMSE"] = rmse
            print(
                f"    Hash-SIREN MAE: {mae:.3f} m, RMSE: {rmse:.3f} m ({time.time() - t0:.1f}s)"
            )
        except Exception as e:
            print(f"    Hash-SIREN failed: {e}")
            import traceback

            traceback.print_exc()
            fold_res["HashSIREN_MAE"] = np.nan
            fold_res["HashSIREN_RMSE"] = np.nan

        # --- 4. TabNet ---
        print("\n  [4/4] TabNet...")
        try:
            t0 = time.time()
            feat_cols_tb = [
                "avg_latitude",
                "avg_longitude",
                "avg_temperature",
                "avg_humidity",
            ]
            X_tr_tb = train_df[feat_cols_tb].values
            y_tr_tb = train_df["pressure_bias"].values
            X_te_tb = test_df[feat_cols_tb].values

            mae, rmse = train_tabnet(
                X_tr_tb,
                y_tr_tb,
                X_te_tb,
                test_df["avg_pressure"].values,
                test_df["avg_temperature"].values,
                test_df["avg_humidity"].values,
                test_df["avg_altitude"].values,
                phys_params,
                max_epochs=200,
            )
            fold_res["TabNet_MAE"] = mae
            fold_res["TabNet_RMSE"] = rmse
            print(
                f"    TabNet MAE: {mae:.3f} m, RMSE: {rmse:.3f} m ({time.time() - t0:.1f}s)"
            )
        except Exception as e:
            print(f"    TabNet failed: {e}")
            import traceback

            traceback.print_exc()
            fold_res["TabNet_MAE"] = np.nan
            fold_res["TabNet_RMSE"] = np.nan

        results.append(fold_res)

    results_df = pd.DataFrame(results)
    os.makedirs("experiments/05_dl_baselines", exist_ok=True)
    results_df.to_csv("experiments/05_dl_baselines/results.csv", index=False)

    print("\n" + "=" * 70)
    print("DEEP LEARNING BASELINE RESULTS (Mean ± Std, 8-fold LOSO)")
    print("=" * 70)

    for method in ["MLP", "SIREN", "HashSIREN", "TabNet"]:
        mae_col = f"{method}_MAE"
        if mae_col in results_df.columns:
            vals = results_df[mae_col].dropna()
            if len(vals) > 0:
                print(
                    f"  {method:15s} MAE: {vals.mean():7.3f} ± {vals.std():5.2f} m  (n={len(vals)})"
                )

    print(f"\n  {'Physics Baseline':15s} MAE:  36.96 ±  4.04 m")
    print(f"  {'PINF (Ours)':15s} MAE:   3.55 ±  1.23 m")
    print(f"\nResults saved to: experiments/05_dl_baselines/results.csv")
    return results_df


if __name__ == "__main__":
    run_dl_baselines()
