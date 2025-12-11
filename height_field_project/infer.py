import argparse
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from height_field_project.config import load_config
from height_field_project.neural_field import ResidualNeuralField
from height_field_project.physics_baseline import fit_barometric_baseline


def load_artifacts(artifacts_dir: str) -> Tuple[ResidualNeuralField, dict, dict]:
    with open(os.path.join(artifacts_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    cfg = load_config(os.path.join(artifacts_dir, "config.json"))
    model = ResidualNeuralField(
        in_dim=len(scalers["feature_cols"]),
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        fourier_L=cfg.fourier_L,
        dropout=cfg.dropout,
    )
    state = torch.load(os.path.join(artifacts_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, scalers, cfg


def prepare_points_from_csv(
    csv_path: str,
    scalers: dict,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    df, _ = fit_barometric_baseline(df)
    if "week_seq" not in df.columns:
        df["week_seq"] = 0
    feature_cols = scalers["feature_cols"]
    features = df[feature_cols].values
    X = scalers["x_scaler"].transform(features)
    return df, X


def build_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    height_m: float,
    res: int,
    template_df: pd.DataFrame,
    scalers: dict,
) -> Tuple[pd.DataFrame, np.ndarray]:
    lat_grid = np.linspace(lat_min, lat_max, res)
    lon_grid = np.linspace(lon_min, lon_max, res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    n = res * res
    temp_med = template_df["avg_temperature"].median()
    hum_med = template_df["avg_humidity"].median()
    p_med = template_df["avg_pressure"].median()
    week_mode = template_df["week_seq"].mode().iloc[0] if "week_seq" in template_df.columns else 0
    df = pd.DataFrame(
        {
            "avg_latitude": lat_mesh.flatten(),
            "avg_longitude": lon_mesh.flatten(),
            "h_phys_m": np.full(n, height_m),
            "avg_temperature": np.full(n, temp_med),
            "avg_humidity": np.full(n, hum_med),
            "avg_pressure": np.full(n, p_med),
            "week_seq": np.full(n, week_mode),
        }
    )
    X = scalers["x_scaler"].transform(df[scalers["feature_cols"]].values)
    return df, X


def predict(model: ResidualNeuralField, X: np.ndarray, scalers: dict, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        mean, std = model.predict_mc(x_tensor, samples=samples)
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()
    mean_real = scalers["y_scaler"].inverse_transform(mean.reshape(-1, 1)).flatten()
    std_real = std * scalers["y_scaler"].scale_[0]
    return mean_real, std_real


def main(args: argparse.Namespace) -> None:
    model, scalers, cfg = load_artifacts(args.artifacts_dir)

    if args.input_csv:
        df, X = prepare_points_from_csv(args.input_csv, scalers)
    else:
        # 需要模板数据估计温湿度/气压
        template_df = pd.read_csv(cfg.input_csv)
        df, _ = fit_barometric_baseline(template_df)
        df, X = build_grid(
            lat_min=args.grid_bbox[0],
            lat_max=args.grid_bbox[1],
            lon_min=args.grid_bbox[2],
            lon_max=args.grid_bbox[3],
            height_m=args.grid_height,
            res=args.grid_res,
            template_df=df,
            scalers=scalers,
        )

    mean_res, std_res = predict(model, X, scalers, samples=args.samples)
    df["residual_mean"] = mean_res
    df["residual_std"] = std_res
    df["h_pred_mean"] = df["h_phys_m"] + df["residual_mean"]
    df["h_pred_std"] = df["residual_std"]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference for residual neural field")
    p.add_argument("--artifacts_dir", type=str, default=os.path.join("height_field_project", "artifacts"))
    p.add_argument("--input_csv", type=str, default=None, help="CSV with measurements; if None use grid mode")
    p.add_argument("--grid_bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"), default=None)
    p.add_argument("--grid_res", type=int, default=80)
    p.add_argument("--grid_height", type=float, default=150.0, help="Fixed h_phys value when generating grid slice")
    p.add_argument("--samples", type=int, default=20, help="MC dropout samples")
    p.add_argument("--out_csv", type=str, default=os.path.join("height_field_project", "artifacts", "predictions.csv"))
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not args.input_csv and args.grid_bbox is None:
        raise SystemExit("Either --input_csv or --grid_bbox must be provided.")
    main(args)
