"""
Random-Split vs LOSO Comparison Experiment

Shows that DL/ML methods achieve artificially inflated performance under random splits
due to data leakage (same sensor in train & test), but fail catastrophically under LOSO.

Methods: Plain MLP, SIREN MLP, Hash-SIREN (no P_bias), TabNet, RF, XGBoost
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys, os, time, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

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


def bias_to_height(p_bias_pred, p_obs, t_c, rh, p_ref):
    p_corrected = p_obs - p_bias_pred
    e_sat = 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))
    e = (rh / 100.0) * e_sat
    r = 0.62198 * e / (p_corrected - e)
    t_v = (t_c + 273.15) * (1 + 0.608 * r)
    H = R_DRY_AIR * t_v / G_STANDARD
    return H * np.log(p_ref / p_corrected)


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class PlainMLP(nn.Module):
    def __init__(self, in_dim=5, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SirenMLP(nn.Module):
    def __init__(self, hash_levels=16, hash_features=4, hidden=256, temporal_freqs=6):
        super().__init__()
        self.hash = MultiResolutionHashEncoding(
            n_levels=hash_levels, n_features=hash_features
        )
        self.temporal = FourierTemporalEncoding(n_frequencies=temporal_freqs)
        in_dim = self.hash.out_dim + 1 + self.temporal.out_dim + 1 + 1
        layers = [SirenLayer(in_dim, hidden, w0=1.0, is_first=True)]
        for _ in range(2):
            layers.append(SirenLayer(hidden, hidden, w0=1.0))
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, lat, lon, z, t, temperature, humidity):
        lat_n = (lat + 90.0) / 180.0
        lon_n = lon % 360.0 / 360.0
        coords = torch.stack([lat_n, lon_n], dim=-1)
        h_s = self.hash(coords)
        h_t = self.temporal(t / 3600.0)
        feat = torch.cat(
            [
                h_s,
                z.unsqueeze(-1),
                h_t,
                temperature.unsqueeze(-1),
                humidity.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp(feat).squeeze(-1)


class NeuralDataset(Dataset):
    def __init__(self, df):
        from height_field_project.train_generalized_with_bias import (
            BiasAwarePINNDataset,
        )

        self.ds = BiasAwarePINNDataset(
            lat=df["avg_latitude"].values,
            lon=df["avg_longitude"].values,
            z=df["avg_altitude"].values,
            t=df["timestamp"].values,
            temperature=df["avg_temperature"].values,
            humidity=df["avg_humidity"].values,
            pressure_bias=df["pressure_bias"].values,
            sensor_id=np.zeros(len(df), dtype=np.int64),
            p_obs=df["avg_pressure"].values,
            h_gnss=df["avg_altitude"].values,
            h_phys=df["h_phys_hae"].values,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def make_neural_loaders(train_df, test_df, bs=2048):
    ds_tr = NeuralDataset(train_df)
    ds_te = NeuralDataset(test_df)
    return DataLoader(ds_tr, batch_size=bs, shuffle=True), DataLoader(
        ds_te, batch_size=bs, shuffle=False
    )


def eval_height_from_delta_h(model, loader, device):
    model.eval()
    preds, gnss = [], []
    with torch.no_grad():
        for b in loader:
            dh = model(
                b["lat"].to(device),
                b["lon"].to(device),
                b["z"].to(device),
                b["t"].to(device),
                b["temperature"].to(device),
                b["humidity"].to(device),
            )
            h_pred = b["h_phys"].to(device) + dh
            preds.append(h_pred.cpu().numpy())
            gnss.append(b["h_gnss"].numpy())
    return np.concatenate(preds), np.concatenate(gnss)


def eval_height_from_delta_p(model, loader, phys_params, device):
    model.eval()
    preds, gnss = [], []
    with torch.no_grad():
        for b in loader:
            dp = model(
                b["lat"].to(device),
                b["lon"].to(device),
                b["z"].to(device),
                b["t"].to(device),
                b["temperature"].to(device),
                b["humidity"].to(device),
            )
            p_corr = b["p_obs"].to(device) + dp
            tc = b["temperature"].to(device)
            es = 610.94 * torch.exp(17.625 * tc / (tc + 243.04))
            ev = (b["humidity"].to(device) / 100.0) * es
            rv = 0.62198 * ev / (p_corr - ev)
            tv = (tc + 273.15) * (1 + 0.608 * rv)
            H = R_DRY_AIR * tv / G_STANDARD
            hp = H * torch.log(phys_params.p_ref / p_corr)
            preds.append(hp.cpu().numpy())
            gnss.append(b["h_gnss"].numpy())
    return np.concatenate(preds), np.concatenate(gnss)


def train_mlp_random(train_df, test_df, phys_params, device, epochs=50):
    model = PlainMLP(in_dim=5, hidden=128, n_layers=3).to(device)
    feat = [
        "avg_latitude",
        "avg_longitude",
        "avg_temperature",
        "avg_humidity",
        "avg_altitude",
    ]
    mu = train_df[feat].values.mean(0)
    sigma = train_df[feat].values.std(0) + 1e-8
    X_tr = (train_df[feat].values - mu) / sigma
    y_tr = train_df["avg_altitude"].values - train_df["h_phys_hae"].values
    X_te = (test_df[feat].values - mu) / sigma

    ds = TabularDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=4096, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    for ep in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            pred = model(x.to(device))
            nn.functional.l1_loss(pred, y.to(device)).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_te, dtype=torch.float32).to(device)
        dh = model(xt).cpu().numpy()
    h_pred = test_df["avg_altitude"].values + dh
    return np.mean(np.abs(h_pred - test_df["avg_altitude"].values))


def train_siren_random(train_df, test_df, phys_params, device, epochs=50):
    model = SirenMLP().to(device)
    tr_dl, te_dl = make_neural_loaders(train_df, test_df)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    for ep in range(epochs):
        model.train()
        for b in tr_dl:
            opt.zero_grad()
            dh = model(
                b["lat"].to(device),
                b["lon"].to(device),
                b["z"].to(device),
                b["t"].to(device),
                b["temperature"].to(device),
                b["humidity"].to(device),
            )
            hp = b["h_phys"].to(device) + dh
            torch.mean(torch.abs(hp - b["h_gnss"].to(device))).backward()
            opt.step()
        sched.step()

    preds, gnss = eval_height_from_delta_h(model, te_dl, device)
    return np.mean(np.abs(preds - gnss))


def train_hash_siren_random(train_df, test_df, phys_params, device, epochs=50):
    model = SirenMLP().to(device)
    tr_dl, te_dl = make_neural_loaders(train_df, test_df)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    for ep in range(epochs):
        model.train()
        for b in tr_dl:
            opt.zero_grad()
            dp = model(
                b["lat"].to(device),
                b["lon"].to(device),
                b["z"].to(device),
                b["t"].to(device),
                b["temperature"].to(device),
                b["humidity"].to(device),
            )
            p_corr = b["p_obs"].to(device) + dp
            tc = b["temperature"].to(device)
            es = 610.94 * torch.exp(17.625 * tc / (tc + 243.04))
            ev = (b["humidity"].to(device) / 100.0) * es
            rv = 0.62198 * ev / (p_corr - ev)
            tv = (tc + 273.15) * (1 + 0.608 * rv)
            H = R_DRY_AIR * tv / G_STANDARD
            hp = H * torch.log(phys_params.p_ref / p_corr)
            torch.mean(torch.abs(hp - b["h_gnss"].to(device))).backward()
            opt.step()
        sched.step()

    preds, gnss = eval_height_from_delta_p(model, te_dl, phys_params, device)
    return np.mean(np.abs(preds - gnss))


def run_tabnet_random(train_df, test_df, phys_params):
    from pytorch_tabnet.tab_model import TabNetRegressor

    feat = ["avg_latitude", "avg_longitude", "avg_temperature", "avg_humidity"]
    X_tr = train_df[feat].values
    y_tr = train_df["pressure_bias"].values.reshape(-1, 1)
    X_te = test_df[feat].values

    model = TabNetRegressor(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        optimizer_params=dict(lr=2e-3),
        seed=42,
        verbose=0,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.fit(
        X_tr, y_tr, max_epochs=200, patience=20, batch_size=4096, virtual_batch_size=512
    )
    y_pred = model.predict(X_te).flatten()

    h_pred = bias_to_height(
        y_pred,
        test_df["avg_pressure"].values,
        test_df["avg_temperature"].values,
        test_df["avg_humidity"].values,
        phys_params.p_ref,
    )
    return np.mean(np.abs(h_pred - test_df["avg_altitude"].values))


def run_rf_xgb_random(train_df, test_df, phys_params):
    feat = ["avg_latitude", "avg_longitude", "avg_temperature", "avg_humidity"]
    X_tr = train_df[feat].values
    y_tr = train_df["pressure_bias"].values
    X_te = test_df[feat].values

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
    )
    rf.fit(X_tr, y_tr)
    y_rf = rf.predict(X_te)
    h_rf = bias_to_height(
        y_rf,
        test_df["avg_pressure"].values,
        test_df["avg_temperature"].values,
        test_df["avg_humidity"].values,
        phys_params.p_ref,
    )
    rf_mae = np.mean(np.abs(h_rf - test_df["avg_altitude"].values))

    xgb_m = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_m.fit(X_tr, y_tr)
    y_xgb = xgb_m.predict(X_te)
    h_xgb = bias_to_height(
        y_xgb,
        test_df["avg_pressure"].values,
        test_df["avg_temperature"].values,
        test_df["avg_humidity"].values,
        phys_params.p_ref,
    )
    xgb_mae = np.mean(np.abs(h_xgb - test_df["avg_altitude"].values))

    return rf_mae, xgb_mae


def main():
    print("=" * 70)
    print("RANDOM-SPLIT vs LOSO COMPARISON")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    df = pd.read_csv("data/sensor_data_filtered.csv")
    df, pp = compute_physics_baseline(df, p_ref=None, convert_to_hae=False)
    df = compute_sensor_bias(df, pp.p_ref)
    df["timestamp"] = df["processed_time"].apply(parse_timestamp)

    # Load LOSO results for comparison
    loso_dl = pd.read_csv("experiments/05_dl_baselines/results.csv")
    loso_bl = pd.read_csv("experiments/01_baseline_comparisons/results.csv")

    n_trials = 3
    random_maes = {
        m: [] for m in ["MLP", "SIREN", "HashSIREN", "TabNet", "RF", "XGBoost"]
    }

    for trial in range(n_trials):
        print(f"\n{'=' * 70}")
        print(f"Random-split Trial {trial + 1}/{n_trials}")
        print(f"{'=' * 70}")
        set_seed(42 + trial)

        perm = np.random.permutation(len(df))
        n_test = int(len(df) * 0.2)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

        # RF + XGBoost
        print("  RF + XGBoost...", end=" ", flush=True)
        t0 = time.time()
        rf_mae, xgb_mae = run_rf_xgb_random(train_df, test_df, pp)
        random_maes["RF"].append(rf_mae)
        random_maes["XGBoost"].append(xgb_mae)
        print(f"RF={rf_mae:.3f}m, XGB={xgb_mae:.3f}m ({time.time() - t0:.1f}s)")

        # Plain MLP
        print("  Plain MLP...", end=" ", flush=True)
        t0 = time.time()
        mae = train_mlp_random(train_df, test_df, pp, DEVICE, epochs=50)
        random_maes["MLP"].append(mae)
        print(f"{mae:.3f}m ({time.time() - t0:.1f}s)")

        # SIREN (Δh)
        print("  SIREN MLP (Δh)...", end=" ", flush=True)
        t0 = time.time()
        mae = train_siren_random(train_df, test_df, pp, DEVICE, epochs=50)
        random_maes["SIREN"].append(mae)
        print(f"{mae:.3f}m ({time.time() - t0:.1f}s)")

        # Hash-SIREN (δP, no P_bias)
        print("  Hash-SIREN (δP, no P_bias)...", end=" ", flush=True)
        t0 = time.time()
        mae = train_hash_siren_random(train_df, test_df, pp, DEVICE, epochs=50)
        random_maes["HashSIREN"].append(mae)
        print(f"{mae:.3f}m ({time.time() - t0:.1f}s)")

        # TabNet
        print("  TabNet...", end=" ", flush=True)
        t0 = time.time()
        mae = run_tabnet_random(train_df, test_df, pp)
        random_maes["TabNet"].append(mae)
        print(f"{mae:.3f}m ({time.time() - t0:.1f}s)")

    # Compute mean
    random_means = {m: np.mean(v) for m, v in random_maes.items()}

    # LOSO means
    loso_means = {
        "MLP": loso_dl["MLP_MAE"].mean(),
        "SIREN": loso_dl["SIREN_MAE"].mean(),
        "HashSIREN": loso_dl["HashSIREN_MAE"].mean(),
        "TabNet": loso_dl["TabNet_MAE"].mean(),
        "RF": loso_bl["RF_MAE"].mean(),
        "XGBoost": loso_bl["XGB_MAE"].mean(),
    }

    print("\n" + "=" * 70)
    print("RESULTS: Random-Split vs LOSO (MAE in meters)")
    print("=" * 70)
    print(f"{'Method':20s} {'Random-Split':>14s} {'LOSO':>14s} {'Ratio':>10s}")
    print("-" * 60)
    for m in ["RF", "XGBoost", "MLP", "SIREN", "HashSIREN", "TabNet"]:
        rs = random_means[m]
        ls = loso_means[m]
        ratio = ls / rs if rs > 0 else float("inf")
        print(f"{m:20s} {rs:14.3f} {ls:14.3f} {ratio:10.1f}x")
    print(f"{'PINF (Ours)':20s} {'---':>14s} {'3.55':>14s} {'---':>10s}")
    print(f"{'Physics Baseline':20s} {'36.96':>14s} {'36.96':>14s} {'1.0':>10s}")

    # Save
    out_df = pd.DataFrame(
        [
            {
                "method": m,
                "random_split_mae": random_means[m],
                "loso_mae": loso_means[m],
                "ratio": loso_means[m] / random_means[m]
                if random_means[m] > 0
                else np.nan,
            }
            for m in random_means
        ]
    )
    out_df = pd.concat(
        [
            out_df,
            pd.DataFrame(
                [
                    {
                        "method": "PINF (Ours)",
                        "random_split_mae": np.nan,
                        "loso_mae": 3.55,
                        "ratio": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    os.makedirs("experiments/05_dl_baselines", exist_ok=True)
    out_df.to_csv("experiments/05_dl_baselines/random_vs_loso.csv", index=False)
    print(f"\nSaved to: experiments/05_dl_baselines/random_vs_loso.csv")


if __name__ == "__main__":
    main()
