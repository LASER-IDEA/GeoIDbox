import argparse
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
try:
    import wandb
except ImportError:  # 轻量依赖，可选
    wandb = None

from height_field_project.config import TrainingConfig, save_config
from height_field_project.physics_baseline import fit_barometric_baseline
from height_field_project.neural_field import ResidualNeuralField
from height_field_project.era5_utils import enrich_with_era5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray | None = None,
        h_phys_raw: np.ndarray | None = None,
        p_raw: np.ndarray | None = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if w is None:
            self.w = torch.ones_like(self.y)
        else:
            self.w = torch.tensor(w, dtype=torch.float32)
        # raw values for physics residual
        self.h_phys_raw = torch.tensor(h_phys_raw if h_phys_raw is not None else np.zeros_like(y), dtype=torch.float32)
        self.p_raw = torch.tensor(p_raw if p_raw is not None else np.zeros_like(y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.w[idx], self.h_phys_raw[idx], self.p_raw[idx]


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # 确保 week_seq 存在
    if "week_seq" not in df.columns:
        df["week_seq"] = 0
    feature_cols = [
        "avg_latitude",
        "avg_longitude",
        "h_phys_m",
        "avg_temperature",
        "avg_humidity",
        "avg_pressure",
        "week_seq",
    ]
    # 可选 ERA5 大尺度特征
    optional_cols = ["era5_tv1000_k", "era5_tv900_k", "era5_lapse_1000_900"]
    for col in optional_cols:
        if col in df.columns and df[col].notna().any():
            feature_cols.append(col)
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"缺少特征列 {col}")
    return df[feature_cols], feature_cols


def build_pseudo_points(df: pd.DataFrame, count: int) -> pd.DataFrame:
    lat_min, lat_max = df["avg_latitude"].min(), df["avg_latitude"].max()
    lon_min, lon_max = df["avg_longitude"].min(), df["avg_longitude"].max()
    h_med = df["h_phys_m"].median()
    temp_med = df["avg_temperature"].median()
    hum_med = df["avg_humidity"].median()
    p_med = df["avg_pressure"].median()
    week_mode = df["week_seq"].mode().iloc[0] if not df["week_seq"].empty else 0

    data = {
        "avg_latitude": np.random.uniform(lat_min, lat_max, size=count),
        "avg_longitude": np.random.uniform(lon_min, lon_max, size=count),
        "h_phys_m": np.full(count, h_med),
        "avg_temperature": np.full(count, temp_med),
        "avg_humidity": np.full(count, hum_med),
        "avg_pressure": np.full(count, p_med),
        "week_seq": np.full(count, week_mode),
        "residual": np.zeros(count),
        "is_pseudo": np.ones(count, dtype=int),
    }

    # If ERA5 features exist in df, fill with medians
    for col in ["era5_tv1000_k", "era5_tv900_k", "era5_lapse_1000_900"]:
        if col in df.columns:
            data[col] = np.full(count, df[col].median())

    pseudo = pd.DataFrame(data)
    return pseudo


def main(args: argparse.Namespace) -> None:
    cfg = TrainingConfig(
        input_csv=args.input_csv,
        artifacts_dir=args.artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        fourier_L=args.fourier_L,
        dropout=args.dropout,
        pseudo_ratio=args.pseudo_ratio,
        pseudo_weight=args.pseudo_weight,
        huber_delta=args.huber_delta,
        seed=args.seed,
        lambda_phys=args.lambda_phys,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    run = None
    if args.wandb_project and wandb is not None:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "input_csv": cfg.input_csv,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "hidden_dim": cfg.hidden_dim,
                "depth": cfg.depth,
                "fourier_L": cfg.fourier_L,
                "dropout": cfg.dropout,
                "pseudo_ratio": cfg.pseudo_ratio,
                "pseudo_weight": cfg.pseudo_weight,
                "huber_delta": cfg.huber_delta,
                "seed": cfg.seed,
            },
        )
    elif args.wandb_project and wandb is None:
        print("wandb 未安装，跳过线上日志；可 `pip install wandb` 启用。")

    df = pd.read_csv(cfg.input_csv)
    if args.era5_nc is not None:
        if os.path.exists(args.era5_nc):
            print(f"Enriching with ERA5: {args.era5_nc}")
            df = enrich_with_era5(df, args.era5_nc)
        else:
            print(f"WARN: ERA5 file not found: {args.era5_nc}, skip enrichment.")
    df, baseline_params = fit_barometric_baseline(df)
    df["residual"] = df["avg_altitude"] - df["h_phys_m"]
    features_df, feature_cols = prepare_features(df)

    # Drop rows with NaNs in features to avoid scaler issues
    mask_valid = features_df.notna().all(axis=1)
    if not mask_valid.all():
        dropped = (~mask_valid).sum()
        print(f"Dropping {dropped} rows with NaNs in features")
    features_df = features_df[mask_valid]
    df = df.loc[features_df.index]
    if len(features_df) == 0:
        raise ValueError("No valid rows remain after dropping NaNs. Check ERA5 coverage or run without --era5_nc.")

    target = df["residual"].values
    h_phys_raw_all = df["h_phys_m"].values
    p_raw_all = df["avg_pressure"].values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(features_df.values)
    y_scaled = y_scaler.fit_transform(target.reshape(-1, 1)).flatten()

    # 拆分真实数据
    X_train, X_tmp, y_train, y_tmp, hp_train, hp_tmp, p_train, p_tmp = train_test_split(
        X_scaled,
        y_scaled,
        h_phys_raw_all,
        p_raw_all,
        test_size=cfg.val_ratio + cfg.test_ratio,
        random_state=cfg.seed,
    )
    val_size = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    X_val, X_test, y_val, y_test, hp_val, hp_test, p_val, p_test = train_test_split(
        X_tmp, y_tmp, hp_tmp, p_tmp, test_size=val_size, random_state=cfg.seed
    )

    # 伪点仅用于训练集
    X_train_real, y_train_real = X_train, y_train
    if cfg.pseudo_ratio > 0:
        pseudo_df = build_pseudo_points(df, int(len(df) * cfg.pseudo_ratio))
        pseudo_features, _ = prepare_features(pseudo_df)
        X_pseudo = x_scaler.transform(pseudo_features.values)
        y_pseudo = y_scaler.transform(pseudo_df["residual"].values.reshape(-1, 1)).flatten()
        w_pseudo = np.full(len(y_pseudo), cfg.pseudo_weight, dtype=np.float32)
        hp_pseudo = pseudo_df["h_phys_m"].values
        p_pseudo = pseudo_df["avg_pressure"].values
        X_train = np.concatenate([X_train_real, X_pseudo], axis=0)
        y_train = np.concatenate([y_train_real, y_pseudo], axis=0)
        w_train = np.concatenate([np.ones(len(X_train_real), dtype=np.float32), w_pseudo], axis=0)
        hp_train = np.concatenate([hp_train, hp_pseudo], axis=0)
        p_train = np.concatenate([p_train, p_pseudo], axis=0)
    else:
        w_train = np.ones(len(X_train), dtype=np.float32)
        hp_train = hp_train
        p_train = p_train

    train_ds = ResidualDataset(X_train, y_train, w_train, hp_train, p_train)
    val_ds = ResidualDataset(X_val, y_val, np.ones(len(X_val), dtype=np.float32), hp_val, p_val)
    test_ds = ResidualDataset(X_test, y_test, np.ones(len(X_test), dtype=np.float32), hp_test, p_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = ResidualNeuralField(
        in_dim=X_scaled.shape[1],
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        fourier_L=cfg.fourier_L,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.HuberLoss(delta=cfg.huber_delta)
    y_scale = float(y_scaler.scale_[0])
    y_mean = float(y_scaler.mean_[0])
    Hs = baseline_params["Hs_m"]
    P0 = baseline_params["P0_Pa"]

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        total = 0.0
        count = 0
        with torch.set_grad_enabled(train):
            for xb, yb, wb, hp_raw, p_raw in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                hp_raw = hp_raw.to(device)
                p_raw = p_raw.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss = (loss * wb).mean()

                # Physics residual: ln(p_obs) vs ln(p_hat) from predicted height
                # y_real = pred * scale + mean
                pred_real = pred * y_scale + y_mean
                h_pred = hp_raw + pred_real
                ln_p_hat = np.log(P0) - h_pred / Hs
                ln_p_obs = torch.log(p_raw)
                phys_res = ln_p_obs - ln_p_hat
                phys_loss = (phys_res ** 2).mean()
                loss = loss + cfg.lambda_phys * phys_loss

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total += loss.item() * len(xb)
                count += len(xb)
        return total / max(count, 1)

    best_val = float("inf")
    patience = 30
    no_improve = 0

    for epoch in range(cfg.epochs):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg.artifacts_dir, "model.pt"))
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0:
            print(f"[{epoch+1:04d}] train {train_loss:.4f} | val {val_loss:.4f}")
        if run:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 评估
    model.load_state_dict(torch.load(os.path.join(cfg.artifacts_dir, "model.pt"), map_location=device))
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for xb, yb, _, _, _ in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            gts.append(yb.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    preds_real = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    gts_real = y_scaler.inverse_transform(gts.reshape(-1, 1)).flatten()
    rmse = np.sqrt(np.mean((preds_real - gts_real) ** 2))
    mae = np.mean(np.abs(preds_real - gts_real))
    print(f"Test RMSE: {rmse:.3f} m | MAE: {mae:.3f} m")
    if run:
        wandb.log({"test_rmse_m": rmse, "test_mae_m": mae, "best_val": best_val})

    # 保存 scaler 与特征
    with open(os.path.join(cfg.artifacts_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(
            {
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "feature_cols": feature_cols,
                "baseline_params": baseline_params,
            },
            f,
        )
    save_config(cfg, os.path.join(cfg.artifacts_dir, "config.json"))
    print(f"Artifacts saved to {cfg.artifacts_dir}")
    if run:
        wandb.save(os.path.join(cfg.artifacts_dir, "model.pt"))
        wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train residual neural field for height conversion")
    p.add_argument("--input_csv", type=str, default="sensor_data_clean_stable.csv")
    p.add_argument("--artifacts_dir", type=str, default=os.path.join("height_field_project", "artifacts"))
    p.add_argument("--era5_nc", type=str, default=None, help="optional ERA5 pressure-level NetCDF for macro met features")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--fourier_L", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pseudo_ratio", type=float, default=1.0)
    p.add_argument("--pseudo_weight", type=float, default=0.5)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default='GeoBox', help="wandb project name; leave None to disable logging")
    p.add_argument("--wandb_run_name", type=str, default=None, help="optional wandb run name")
    p.add_argument("--lambda_phys", type=float, default=0.1, help="weight for physics residual loss")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
