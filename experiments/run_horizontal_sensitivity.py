"""
Figure 3: Horizontal-Position Sensitivity Analysis

For each LOSO fold, perturb the held-out sensor's horizontal coordinates
with Gaussian noise at various magnitudes and measure the MAE change.

Perturbation levels: 0, 1, 2, 5, 10, 20, 50, 100, 500, 1000 m (3 repeats each)
Output: experiments/figures/additional/fig3_horizontal_sensitivity.{pdf,png}
        experiments/figures/additional/horizontal_sensitivity_results.csv
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from height_field_project.train_generalized_with_bias import (
    BiasAwarePINN,
    compute_sensor_bias,
    BiasAwarePINNDataset,
)
from height_field_project.neural_field_pinn_generalized import (
    GeneralizedPressureCorrectionPINN,
)
from height_field_project.physics_baseline import compute_physics_baseline
from height_field_project.train_pinn import parse_timestamp

R_DRY_AIR = 287.05
G_STANDARD = 9.80665

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
    }
)

PALETTE = {
    "deep_blue": "#313695",
    "mid_blue": "#4393C3",
    "light_blue": "#74ADD1",
    "yellow": "#FEE090",
    "orange": "#F46D43",
    "red": "#D62728",
    "dark_red": "#A50026",
    "bg_axes": "#E8EFF6",
    "grid": "#B0BFCC",
    "text": "#1A1A2E",
    "green": "#1A9641",
}

CKPT_DIR = "height_field_project/loso_curriculum_results"
DATA_CSV = "data/sensor_data_filtered.csv"
OUT_DIR = "experiments/figures/additional"

LEVELS = [0, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
N_REPS = 3


def predict_height(model, lat, lon, batch, phys):
    with torch.no_grad():
        dp = model(
            lat,
            lon,
            batch["z"],
            batch["t"],
            batch["temperature"],
            batch["humidity"],
            batch["pressure_bias"],
        )
        p_c = batch["p_obs"] + dp
        tc = batch["temperature"]
        es = 610.94 * torch.exp(17.625 * tc / (tc + 243.04))
        e = (batch["humidity"] / 100.0) * es
        r = 0.62198 * e / (p_c - e)
        tv = (tc + 273.15) * (1 + 0.608 * r)
        H = R_DRY_AIR * tv / G_STANDARD
        h_pred = H * torch.log(phys.p_ref / p_c)
    return h_pred


def main():
    print("=" * 68)
    np.random.seed(42)
    torch.manual_seed(42)
    print("HORIZONTAL POSITION SENSITIVITY ANALYSIS")
    print(f"Perturbation levels: {LEVELS} m")
    print(f"Repeats per level: {N_REPS}")
    print("=" * 68)

    df = pd.read_csv(DATA_CSV, parse_dates=["processed_time"])
    df, pp = compute_physics_baseline(
        df, p_ref=None, t_ref_method="mean", convert_to_hae=False
    )
    df = compute_sensor_bias(df, pp.p_ref)
    df["timestamp"] = df["processed_time"].apply(parse_timestamp)
    sensors = df["uid"].unique()
    results = []

    # 诊断：hash 编码最大分辨率对应的理论网格尺度（经纬度方向）
    max_res = 1024
    lat_cell_m = 111320.0 * (180.0 / max_res)
    lon_cell_m = 111320.0 * (360.0 / max_res) * np.cos(
        np.radians(df["avg_latitude"].mean())
    )
    print(
        f"Approx. finest hash cell size: lat≈{lat_cell_m/1000:.1f} km, "
        f"lon≈{lon_cell_m/1000:.1f} km"
    )

    for fold in range(8):
        ckpt = torch.load(
            f"{CKPT_DIR}/model_curriculum_fold{fold}.pt",
            map_location="cpu",
            weights_only=False,
        )
        base = GeneralizedPressureCorrectionPINN(
            hash_levels=ckpt["args"]["hash_levels"],
            hash_features=ckpt["args"]["hash_features"],
            hidden_dim=ckpt["args"]["hidden_dim"],
            n_hidden_layers=ckpt["args"]["n_hidden_layers"],
            temporal_freqs=ckpt["args"]["temporal_freqs"],
            dropout=ckpt["args"]["dropout"],
            use_siren=ckpt["args"]["use_siren"],
        )
        model = BiasAwarePINN(base, bias_dim=ckpt["args"]["bias_dim"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        phys = ckpt["phys_params"]

        test_df = df[df["uid"] == sensors[fold]].copy()
        n = len(test_df)
        lat_m = test_df["avg_latitude"].mean()

        ds = BiasAwarePINNDataset(
            lat=test_df["avg_latitude"].values,
            lon=test_df["avg_longitude"].values,
            z=test_df["avg_altitude"].values,
            t=test_df["timestamp"].values,
            temperature=test_df["avg_temperature"].values,
            humidity=test_df["avg_humidity"].values,
            pressure_bias=test_df["pressure_bias"].values,
            sensor_id=np.zeros(n, dtype=np.int64),
            p_obs=test_df["avg_pressure"].values,
            h_gnss=test_df["avg_altitude"].values,
            h_phys=test_df["h_phys_hae"].values,
        )
        loader = DataLoader(ds, batch_size=n, shuffle=False)
        batch = next(iter(loader))
        h_base = predict_height(model, batch["lat"], batch["lon"], batch, phys)

        for level in LEVELS:
            n_reps = 1 if level == 0 else N_REPS
            for rep in range(n_reps):
                lat = batch["lat"].clone()
                lon = batch["lon"].clone()
                if level > 0:
                    angle = np.random.uniform(0, 2 * np.pi, n)
                    lat += torch.from_numpy(level * np.sin(angle) / 111320.0).float()
                    lon += torch.from_numpy(
                        level * np.cos(angle) / (111320.0 * np.cos(np.radians(lat_m)))
                    ).float()
                h_pred = predict_height(model, lat, lon, batch, phys)
                mae = torch.mean(torch.abs(h_pred - batch["h_gnss"])).item()
                drift = torch.mean(torch.abs(h_pred - h_base)).item()
                drift_p95 = torch.quantile(torch.abs(h_pred - h_base), 0.95).item()
                results.append(
                    {
                        "fold": fold,
                        "sensor": sensors[fold][-6:],
                        "perturbation_m": level,
                        "repeat": rep,
                        "mae": mae,
                        "pred_drift_mae": drift,
                        "pred_drift_p95": drift_p95,
                    }
                )

            lvl = [
                r["mae"]
                for r in results
                if r["fold"] == fold and r["perturbation_m"] == level
            ]
            lvl_drift = [
                r["pred_drift_mae"]
                for r in results
                if r["fold"] == fold and r["perturbation_m"] == level
            ]
            print(
                f"Fold {fold}, {level}m: MAE={np.mean(lvl):.3f}, "
                f"drift={np.mean(lvl_drift):.3e} m ({np.mean(lvl_drift)*1000:.3e} mm)"
            )

    rdf = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    rdf.to_csv(f"{OUT_DIR}/horizontal_sensitivity_results.csv", index=False)

    # ── Plot ──
    print("\n[Fig 3] Plotting...")
    summary = (
        rdf.groupby("perturbation_m")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            drift_mean=("pred_drift_mae", "mean"),
            drift_std=("pred_drift_mae", "std"),
            drift_p95_mean=("pred_drift_p95", "mean"),
        )
        .reset_index()
    )
    levels = summary["perturbation_m"].values
    means = summary["mae_mean"].values
    stds = summary["mae_std"].fillna(0.0).values
    drift_means = summary["drift_mean"].values
    levels_plot = levels.astype(float).copy()
    levels_plot[levels_plot == 0] = 0.5

    fig, ax = plt.subplots(figsize=(11, 7))
    ax2 = ax.twinx()
    ax.set_facecolor(PALETTE["bg_axes"])
    ax.grid(
        True, linestyle=":", linewidth=0.5, color=PALETTE["grid"], alpha=0.9, zorder=1
    )
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(PALETTE["grid"])
        s.set_linewidth(0.7)

    ax.errorbar(
        levels_plot,
        means,
        yerr=stds,
        fmt="o-",
        color=PALETTE["deep_blue"],
        linewidth=2.5,
        markersize=10,
        capsize=6,
        capthick=2,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=4,
        label="MAE vs GNSS",
    )

    drift_means_mm = drift_means * 1000.0
    ax2.plot(
        levels_plot,
        drift_means_mm,
        "s--",
        color=PALETTE["orange"],
        linewidth=2.0,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
        label="Prediction drift |h_pert-h_base| (mm)",
    )

    for _, row in summary.iterrows():
        x_plot = 0.5 if row["perturbation_m"] == 0 else row["perturbation_m"]
        ax.text(
            x_plot,
            row["mae_mean"] + row["mae_std"] + 0.15,
            f"{row['mae_mean']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["deep_blue"],
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.axvspan(0.5, 1.5, alpha=0.12, color=PALETTE["green"], zorder=1)
    ax.text(
        0.75,
        0.5,
        "GNSS CEP\n(~1.5 m)",
        ha="center",
        fontsize=9,
        color=PALETTE["green"],
        fontweight="bold",
        alpha=0.9,
    )

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks(levels_plot)
    ax.set_xticklabels([str(int(v)) for v in levels])
    ax.set_xlabel(
        "Horizontal Position Perturbation (m, log scale)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("MAE vs GNSS (m)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Prediction Drift (mm)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Sensitivity of Altitude Conversion to Horizontal-Position Uncertainty\n"
        "(8-Fold LOSO × 3 Repeats, 0 m to 50 km Perturbation)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=10,
        framealpha=0.92,
        edgecolor=PALETTE["grid"],
        loc="upper left",
    )

    ax.text(
        0.02,
        0.98,
        "Note: MAE can stay flat while\n"
        "predictions still drift at mm-scale.\n"
        "Tiny drift is expected from coarse\n"
        "global hash-grid quantization.",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        fontweight="bold",
        color=PALETTE["deep_blue"],
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["grid"], alpha=0.92),
    )
    ax.set_ylim(0, max(means) * 1.5)

    for ext in ("pdf", "png"):
        fig.savefig(
            f"{OUT_DIR}/fig3_horizontal_sensitivity.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    print(f"  Saved: fig3_horizontal_sensitivity.pdf/png")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(
            f"  {row['perturbation_m']:>6.0f} m: "
            f"MAE={row['mae_mean']:.3f} ± {row['mae_std']:.3f} m, "
            f"drift={row['drift_mean']:.3e} m ({row['drift_mean']*1000:.3e} mm), "
            f"drift_p95={row['drift_p95_mean']:.3e} m"
        )
    baseline = summary[summary["perturbation_m"] == 0]["mae_mean"].values[0]
    print(f"\n  Baseline: {baseline:.3f} m")
    mae_1km = summary[summary["perturbation_m"] == 1000]["mae_mean"].values[0]
    print(f"  Degradation at 1 km: {(mae_1km / baseline - 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
