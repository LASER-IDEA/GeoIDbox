"""
生成 6 张补充图例，用于丰富论文和 rebuttal：
  Fig 1: LOSO per-fold performance line plot
  Fig 2: Height discrepancy boxplot (raw baro / physics / PINF vs GNSS)
  Fig 4: Error vs distance to nearest training sensor
  Fig 5: Learned residual decomposition time series
  Fig 6: Risk-coverage curve

输出目录: experiments/figures/additional/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    "purple": "#7B3294",
}

CKPT_DIR = "height_field_project/loso_curriculum_results"
DATA_CSV = "data/sensor_data_filtered.csv"
OUT_DIR = "experiments/figures/additional"
NEIGHBOR_RADIUS_M = 200.0
TARGET_SENSOR_UID = "20240911193733A012843A9994605977"
SENSOR_ID_MAP = {
    "20240606181851A641973A1878250224": "4197",
    "20240606185609A190219A4811437779": "9021",
    "20240606201439A160695A3816948226": "6069",
    "20240911193046A806593A5642508217": "0659",
    "20240911193519A117375A6331369164": "1737",
    "20240911193733A012843A9994605977": "1284",
    "20240911194312A389747A0782527426": "8974",
    "20240911194957A179458A3827373510": "7945",
}


def save_fig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
    print(f"  Saved: {name}")
    plt.close(fig)


def setup_ax(ax):
    ax.set_facecolor(PALETTE["bg_axes"])
    ax.grid(
        True, linestyle=":", linewidth=0.5, color=PALETTE["grid"], alpha=0.9, zorder=1
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
        spine.set_linewidth(0.7)


def load_model_and_params(fold):
    ckpt_path = os.path.join(CKPT_DIR, f"model_curriculum_fold{fold}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    phys_params = ckpt["phys_params"]
    args_dict = ckpt["args"]
    base_model = GeneralizedPressureCorrectionPINN(
        hash_levels=args_dict["hash_levels"],
        hash_features=args_dict["hash_features"],
        hidden_dim=args_dict["hidden_dim"],
        n_hidden_layers=args_dict["n_hidden_layers"],
        temporal_freqs=args_dict["temporal_freqs"],
        dropout=args_dict["dropout"],
        use_siren=args_dict["use_siren"],
    )
    model = BiasAwarePINN(base_model, bias_dim=args_dict["bias_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, phys_params, args_dict


def run_inference(model, loader, phys_params, device="cpu"):
    model.eval()
    all_preds, all_gnss, all_alt = [], [], []
    with torch.no_grad():
        for batch in loader:
            delta_p = model(
                batch["lat"].to(device),
                batch["lon"].to(device),
                batch["z"].to(device),
                batch["t"].to(device),
                batch["temperature"].to(device),
                batch["humidity"].to(device),
                batch["pressure_bias"].to(device),
            )
            p_corrected = batch["p_obs"].to(device) + delta_p
            t_c = batch["temperature"].to(device)
            e_sat = 610.94 * torch.exp(17.625 * t_c / (t_c + 243.04))
            e = (batch["humidity"].to(device) / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_c + 273.15) * (1 + 0.608 * r)
            H = R_DRY_AIR * t_v / G_STANDARD
            h_pred = H * torch.log(phys_params.p_ref / p_corrected)
            all_preds.append(h_pred.cpu().numpy())
            all_gnss.append(batch["h_gnss"].numpy())
            all_alt.append(batch["z"].numpy())
    preds = np.concatenate(all_preds)
    gnss = np.concatenate(all_gnss)
    return preds, gnss


def compute_raw_baro_height(df, p_ref):
    t_c = df["avg_temperature"].values
    rh = df["avg_humidity"].values
    p = df["avg_pressure"].values
    e_sat = 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))
    e = (rh / 100.0) * e_sat
    rv = 0.62198 * e / (p - e)
    t_v = (t_c + 273.15) * (1 + 0.608 * rv)
    H = R_DRY_AIR * t_v / G_STANDARD
    h_baro = H * np.log(p_ref / p)
    return h_baro


def prepare_data_and_models():
    print("Loading data...")
    df = pd.read_csv(DATA_CSV, parse_dates=["processed_time"])
    df, phys_params_global = compute_physics_baseline(
        df, p_ref=None, t_ref_method="mean", convert_to_hae=False
    )
    df = compute_sensor_bias(df, phys_params_global.p_ref)
    df["timestamp"] = df["processed_time"].apply(parse_timestamp)
    return df, phys_params_global


def run_fold_inference(df, phys_params_global, fold):
    sensors = df["uid"].unique()
    held_out_sensor = sensors[fold % len(sensors)]
    test_df = df[df["uid"] == held_out_sensor].copy()
    model, phys_params, args_dict = load_model_and_params(fold)

    ds_test = BiasAwarePINNDataset(
        lat=test_df["avg_latitude"].values,
        lon=test_df["avg_longitude"].values,
        z=test_df["avg_altitude"].values,
        t=test_df["timestamp"].values,
        temperature=test_df["avg_temperature"].values,
        humidity=test_df["avg_humidity"].values,
        pressure_bias=test_df["pressure_bias"].values,
        sensor_id=np.zeros(len(test_df), dtype=np.int64),
        p_obs=test_df["avg_pressure"].values,
        h_gnss=test_df["avg_altitude"].values,
        h_phys=test_df["h_phys_hae"].values,
    )
    loader = DataLoader(ds_test, batch_size=2048, shuffle=False)
    preds, gnss = run_inference(model, loader, phys_params)

    h_phys = test_df["h_phys_hae"].values
    h_baro = compute_raw_baro_height(test_df, phys_params.p_ref)
    times = test_df["processed_time"].values

    return {
        "sensor": held_out_sensor,
        "preds": preds,
        "gnss": gnss,
        "h_phys": h_phys,
        "h_baro": h_baro,
        "times": times,
        "mae": np.mean(np.abs(preds - gnss)),
    }


def compute_fold_neighbor_stats(df, n_folds=8, radius_m=50.0):
    sensors = df["uid"].unique()
    locs = (
        df.groupby("uid")
        .agg(
            {
                "avg_latitude": "mean",
                "avg_longitude": "mean",
                "avg_altitude": "mean",
            }
        )
        .reset_index()
        .set_index("uid")
    )

    def distance_m(uid_a, uid_b):
        a = locs.loc[uid_a]
        b = locs.loc[uid_b]
        dlat = (b["avg_latitude"] - a["avg_latitude"]) * 111320.0
        dlon = (
            (b["avg_longitude"] - a["avg_longitude"])
            * 111320.0
            * np.cos(np.radians(a["avg_latitude"]))
        )
        dxy = np.sqrt(dlat**2 + dlon**2)
        dz = b["avg_altitude"] - a["avg_altitude"]
        return np.sqrt(dxy**2 + dz**2)

    fold_stats = []
    for fold in range(n_folds):
        held_uid = sensors[fold % len(sensors)]
        dists = [
            distance_m(held_uid, other_uid)
            for other_uid in sensors
            if other_uid != held_uid
        ]
        dists = np.array(dists, dtype=float)
        fold_stats.append(
            {
                "fold": fold,
                "sensor": held_uid,
                "neighbor_count": int(np.sum(dists <= radius_m)),
                "nearest_dist_m": float(np.min(dists)) if len(dists) else float("nan"),
            }
        )
    return fold_stats


def iqr_trim(values, whisker=1.5):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    low = q1 - whisker * iqr
    high = q3 + whisker * iqr
    mask = (values >= low) & (values <= high)
    return values[mask], mask


def select_fold_for_sensor(fold_data, target_uid):
    for fold, fd in fold_data.items():
        if fd["sensor"] == target_uid:
            return fold
    for fold, fd in fold_data.items():
        if str(fd["sensor"]).endswith(str(target_uid)[-6:]):
            return fold
    return min(fold_data, key=lambda f: fold_data[f]["mae"])


# ════════════════════════════════════════════════════════════════════════════
# Fig 1: LOSO per-fold performance line plot
# ════════════════════════════════════════════════════════════════════════════


def plot_loso_per_fold():
    print("\n[Fig 1] LOSO per-fold performance...")
    pinf = pd.read_csv("height_field_project/loso_curriculum_results/loso_summary.csv")
    base = pd.read_csv("experiments/01_baseline_comparisons/results.csv")
    dl = pd.read_csv("experiments/05_dl_baselines/results.csv")

    folds = np.arange(8)
    pinf_mae = pinf["mae"].values
    phys_mae = pinf["baseline_mae"].values
    rf_mae = base["RF_MAE"].values
    siren_mae = dl["SIREN_MAE"].values

    df = pd.read_csv(DATA_CSV)
    fold_stats = compute_fold_neighbor_stats(
        df, n_folds=len(folds), radius_m=NEIGHBOR_RADIUS_M
    )
    nearest = np.array([s["nearest_dist_m"] for s in fold_stats], dtype=float)

    # 按最近邻距离升序排序
    sort_idx = np.argsort(nearest)
    folds_sorted = folds[sort_idx]
    sensors = [s["sensor"] for s in fold_stats]
    labels = [
        f"{SENSOR_ID_MAP.get(sensors[f], sensors[f][-6:])} ({nearest[f]:.0f}m)"
        for f in folds_sorted
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    setup_ax(ax)
    x = np.arange(8)
    offset = 0.2

    lines = [
        (phys_mae[sort_idx], PALETTE["yellow"], "Physics Baseline", "s", 7),
        (rf_mae[sort_idx], PALETTE["orange"], "Random Forest", "^", 7),
        (siren_mae[sort_idx], PALETTE["purple"], "SIREN MLP", "D", 7),
        (pinf_mae[sort_idx], PALETTE["deep_blue"], "PINF (Ours)", "o", 9),
    ]

    for vals, color, label, marker, ms in lines:
        ax.plot(x, vals, "-", color=color, linewidth=2, alpha=0.85, zorder=3)
        ax.scatter(
            x,
            vals,
            marker=marker,
            s=ms**2,
            color=color,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
            label=label,
        )
        for xi, v in zip(x, vals):
            ax.text(
                xi,
                v + 1.5,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    # Mean markers on right
    for vals, color, label, marker, ms in lines:
        mean_v = np.mean(vals)
        ax.scatter(
            [8.5],
            [mean_v],
            marker=marker,
            s=ms**2 * 1.5,
            color=color,
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )
        ax.text(
            8.8,
            mean_v,
            f"{mean_v:.1f}",
            fontsize=9,
            va="center",
            color=color,
            fontweight="bold",
        )

    ax.axvline(7.5, color=PALETTE["grid"], linewidth=1.5, linestyle="--", alpha=0.6)
    ax.text(
        8.5,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 45,
        "Mean",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["text"],
    )

    ax.set_xticks(list(x) + [8.5])
    ax.set_xticklabels(labels + ["Mean"], fontsize=9)
    ax.set_ylabel("MAE (m)", fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "Held-out Sensor UUID (short), sorted by distance to nearest training sensor",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Fold-wise LOSO MAE across held-out GeoBox sensors",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=10, framealpha=0.92, edgecolor=PALETTE["grid"], loc="upper left")
    ax.set_ylim(0, max(phys_mae[sort_idx]) * 1.3)

    save_fig(fig, "fig1_loso_per_fold")


# ════════════════════════════════════════════════════════════════════════════
# Fig 2: Height discrepancy boxplot
# ════════════════════════════════════════════════════════════════════════════


def plot_height_discrepancy(fold_data_list):
    print("\n[Fig 2] Height discrepancy boxplot...")

    baro_errs, phys_errs, pinf_errs = [], [], []
    for fd in fold_data_list:
        baro_errs.extend(fd["h_baro"] - fd["gnss"])
        phys_errs.extend(fd["h_phys"] - fd["gnss"])
        pinf_errs.extend(fd["preds"] - fd["gnss"])
    baro_errs = np.array(baro_errs)
    phys_errs = np.array(phys_errs)
    pinf_errs = np.array(pinf_errs)

    baro_trim, baro_mask = iqr_trim(baro_errs)
    phys_trim, phys_mask = iqr_trim(phys_errs)
    pinf_trim, pinf_mask = iqr_trim(pinf_errs)

    fig, ax = plt.subplots(figsize=(10, 7))
    setup_ax(ax)

    data = [baro_trim, phys_trim, pinf_trim]
    colors = [PALETTE["orange"], PALETTE["mid_blue"], PALETTE["deep_blue"]]
    labels = [
        f"Raw Barometric\n(MAE={np.mean(np.abs(baro_errs)):.1f} m)",
        f"Physics Baseline\n(MAE={np.mean(np.abs(phys_errs)):.1f} m)",
        f"PINF (Ours)\n(MAE={np.mean(np.abs(pinf_errs)):.1f} m)",
    ]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color=PALETTE["text"], linewidth=1.2),
        capprops=dict(color=PALETTE["text"], linewidth=1.2),
        flierprops=dict(
            marker="o", markerfacecolor=PALETTE["light_blue"], markersize=2, alpha=0.3
        ),
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(
        0,
        color=PALETTE["green"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Zero error (perfect)",
    )

    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel(
        "Height Discrepancy vs. GNSS Reference (m)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "Vertical Reference Inconsistency: Raw Barometric vs. Physics vs. PINF\n"
        "(IQR-trimmed boxplot; All 8 LOSO Folds, N={:,} samples)".format(
            len(baro_errs)
        ),
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=10, framealpha=0.92, edgecolor=PALETTE["grid"])

    # Statistics text
    stats_text = ""
    for name, raw_errs, trim_errs in zip(
        ["Baro", "Phys", "PINF"],
        [baro_errs, phys_errs, pinf_errs],
        [baro_trim, phys_trim, pinf_trim],
    ):
        mae = np.mean(np.abs(raw_errs))
        med = np.median(trim_errs)
        p95 = np.percentile(np.abs(raw_errs), 95)
        kept = len(trim_errs) / len(raw_errs) * 100
        stats_text += (
            f"{name}: MAE={mae:.1f}m, Med(trim)={med:.1f}m, "
            f"P95={p95:.1f}m, Kept={kept:.1f}%\n"
        )
    ax.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["grid"], alpha=0.92),
    )

    save_fig(fig, "fig2_height_discrepancy_boxplot")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4: Error vs distance to nearest training sensor
# ════════════════════════════════════════════════════════════════════════════


def plot_error_vs_distance():
    print("\n[Fig 4] Error vs distance to nearest training sensor...")
    pinf = pd.read_csv("height_field_project/loso_curriculum_results/loso_summary.csv")
    base = pd.read_csv("experiments/01_baseline_comparisons/results.csv")
    dl = pd.read_csv("experiments/05_dl_baselines/results.csv")

    df = pd.read_csv(DATA_CSV)
    df["uid_short"] = df["uid"].str[-6:]
    locs = (
        df.groupby("uid_short")
        .agg(
            {
                "avg_latitude": "mean",
                "avg_longitude": "mean",
                "avg_altitude": "mean",
            }
        )
        .reset_index()
    )

    sensors = df["uid"].unique()
    uid_shorts = [s[-6:] for s in sensors]

    def nearest_train_dist(fold):
        held_short = uid_shorts[fold % len(sensors)]
        held_loc = locs[locs["uid_short"] == held_short].iloc[0]
        min_d = float("inf")
        for i in range(len(sensors)):
            if i == fold % len(sensors):
                continue
            tr_short = uid_shorts[i]
            tr_loc = locs[locs["uid_short"] == tr_short].iloc[0]
            dlat = (tr_loc["avg_latitude"] - held_loc["avg_latitude"]) * 111320
            dlon = (
                (tr_loc["avg_longitude"] - held_loc["avg_longitude"])
                * 111320
                * np.cos(np.radians(held_loc["avg_latitude"]))
            )
            dxy = np.sqrt(dlat**2 + dlon**2)
            dz = tr_loc["avg_altitude"] - held_loc["avg_altitude"]
            d = np.sqrt(dxy**2 + dz**2)
            if d < min_d:
                min_d = d
        return min_d

    distances = np.array([nearest_train_dist(f) for f in range(8)])

    fig, ax = plt.subplots(figsize=(10, 7))
    setup_ax(ax)

    methods = [
        (pinf["mae"].values, PALETTE["deep_blue"], "PINF (Ours)", "o", 80),
        (base["RF_MAE"].values, PALETTE["orange"], "Random Forest", "^", 60),
        (dl["SIREN_MAE"].values, PALETTE["purple"], "SIREN MLP", "D", 60),
        (pinf["baseline_mae"].values, PALETTE["yellow"], "Physics Baseline", "s", 60),
    ]

    for maes, color, label, marker, size in methods:
        ax.scatter(
            distances,
            maes,
            marker=marker,
            s=size,
            color=color,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
            label=label,
            alpha=0.9,
        )
        # Trend line
        if len(distances) > 2:
            z = np.polyfit(distances, maes, 1)
            p = np.poly1d(z)
            x_line = np.linspace(distances.min() * 0.9, distances.max() * 1.1, 50)
            ax.plot(x_line, p(x_line), "--", color=color, linewidth=1.5, alpha=0.5)

    ax.set_xlabel(
        "3D Distance to Nearest Training Sensor (m)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("MAE (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Spatial Generalization: Conversion Error vs. Distance to Training Network\n(8-Fold LOSO, each point = one held-out sensor)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=10, framealpha=0.92, edgecolor=PALETTE["grid"])

    # Correlation
    # corr = np.corrcoef(distances, pinf["mae"].values)[0, 1]
    # ax.text(
    #     0.02,
    #     0.98,
    #     f"PINF r = {corr:.3f}",
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     va="top",
    #     color=PALETTE["deep_blue"],
    #     fontweight="bold",
    #     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["grid"], alpha=0.9),
    # )

    save_fig(fig, "fig4_error_vs_distance")


# ════════════════════════════════════════════════════════════════════════════
# Fig 5: Residual decomposition time series
# ════════════════════════════════════════════════════════════════════════════


def plot_residual_decomposition(fold_data, fold=0):
    print(f"\n[Fig 5] Residual decomposition (Fold {fold})...")
    fd = fold_data[fold]

    gnss = fd["gnss"]
    h_phys = fd["h_phys"]
    preds = fd["preds"]
    times = pd.to_datetime(fd["times"])
    n = len(gnss)

    # Subsample for readability if too many points
    if n > 5000:
        idx = np.linspace(0, n - 1, 5000, dtype=int)
        times = times[idx]
        gnss = gnss[idx]
        h_phys = h_phys[idx]
        preds = preds[idx]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True
    )
    setup_ax(ax1)
    setup_ax(ax2)

    # Upper: height time series
    ax1.plot(
        times,
        gnss,
        ".",
        markersize=1.5,
        alpha=0.4,
        color=PALETTE["text"],
        label="GNSS Reference",
        zorder=3,
    )
    ax1.plot(
        times,
        h_phys,
        ".",
        markersize=1.5,
        alpha=0.4,
        color=PALETTE["mid_blue"],
        label="Physics Baseline",
        zorder=3,
    )
    ax1.plot(
        times,
        preds,
        ".",
        markersize=1.5,
        alpha=0.4,
        color=PALETTE["red"],
        label="PINF Prediction",
        zorder=3,
    )
    ax1.set_ylabel("Height (m)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Physics-Informed Residual Correction (Fold {fold}, Held-out Sensor)",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    ax1.legend(
        fontsize=9,
        markerscale=6,
        framealpha=0.92,
        edgecolor=PALETTE["grid"],
        loc="upper right",
    )

    # Stats
    phys_mae = np.mean(np.abs(h_phys - gnss))
    pinf_mae = np.mean(np.abs(preds - gnss))
    ax1.text(
        0.02,
        0.02,
        f"Physics MAE: {phys_mae:.1f} m\nPINF MAE: {pinf_mae:.2f} m",
        transform=ax1.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["grid"], alpha=0.9),
    )

    # Lower: residuals
    phys_residual = h_phys - gnss
    pinf_residual = preds - gnss
    ax2.plot(
        times,
        phys_residual,
        ".",
        markersize=1.5,
        alpha=0.4,
        color=PALETTE["mid_blue"],
        label="Physics Residual",
        zorder=3,
    )
    ax2.plot(
        times,
        pinf_residual,
        ".",
        markersize=1.5,
        alpha=0.5,
        color=PALETTE["red"],
        label="PINF Residual",
        zorder=3,
    )
    ax2.axhline(0, color=PALETTE["green"], linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Error (m)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax2.legend(
        fontsize=9,
        markerscale=6,
        framealpha=0.92,
        edgecolor=PALETTE["grid"],
        loc="upper right",
    )

    plt.tight_layout(h_pad=1.5)
    save_fig(fig, "fig5_residual_decomposition")


# ════════════════════════════════════════════════════════════════════════════
# Fig 6: Risk-coverage curve
# ════════════════════════════════════════════════════════════════════════════


def run_mc_dropout(model, loader, phys_params, n_samples=30, device="cpu"):
    model.train()
    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            fold_preds = []
            for batch in loader:
                delta_p = model(
                    batch["lat"].to(device),
                    batch["lon"].to(device),
                    batch["z"].to(device),
                    batch["t"].to(device),
                    batch["temperature"].to(device),
                    batch["humidity"].to(device),
                    batch["pressure_bias"].to(device),
                )
                p_corrected = batch["p_obs"].to(device) + delta_p
                t_c = batch["temperature"].to(device)
                e_sat = 610.94 * torch.exp(17.625 * t_c / (t_c + 243.04))
                e = (batch["humidity"].to(device) / 100.0) * e_sat
                r = 0.62198 * e / (p_corrected - e)
                t_v = (t_c + 273.15) * (1 + 0.608 * r)
                H = R_DRY_AIR * t_v / G_STANDARD
                h_pred = H * torch.log(phys_params.p_ref / p_corrected)
                fold_preds.append(h_pred.cpu().numpy())
            all_preds.append(np.concatenate(fold_preds))
    return np.array(all_preds)


def plot_risk_coverage(fold_data_list, df, phys_params_global):
    print("\n[Fig 6] Risk-coverage curve...")
    sensors = df["uid"].unique()

    all_uncertainties = []
    all_abs_errors = []

    for fold in range(8):
        print(f"  MC Dropout fold {fold}...")
        held_out_sensor = sensors[fold % len(sensors)]
        test_df = df[df["uid"] == held_out_sensor].copy()
        model, phys_params, args_dict = load_model_and_params(fold)

        ds_test = BiasAwarePINNDataset(
            lat=test_df["avg_latitude"].values,
            lon=test_df["avg_longitude"].values,
            z=test_df["avg_altitude"].values,
            t=test_df["timestamp"].values,
            temperature=test_df["avg_temperature"].values,
            humidity=test_df["avg_humidity"].values,
            pressure_bias=test_df["pressure_bias"].values,
            sensor_id=np.zeros(len(test_df), dtype=np.int64),
            p_obs=test_df["avg_pressure"].values,
            h_gnss=test_df["avg_altitude"].values,
            h_phys=test_df["h_phys_hae"].values,
        )
        loader = DataLoader(ds_test, batch_size=2048, shuffle=False)

        mc_preds = run_mc_dropout(model, loader, phys_params, n_samples=20)
        mean_pred = mc_preds.mean(axis=0)
        std_pred = mc_preds.std(axis=0)
        gnss = test_df["avg_altitude"].values
        abs_err = np.abs(mean_pred - gnss)

        all_uncertainties.extend(std_pred)
        all_abs_errors.extend(abs_err)

    all_uncertainties = np.array(all_uncertainties)
    all_abs_errors = np.array(all_abs_errors)

    # Sort by uncertainty
    sort_idx = np.argsort(all_uncertainties)
    sorted_errors = all_abs_errors[sort_idx]
    n_total = len(sorted_errors)

    # Compute risk-coverage
    coverages = np.linspace(0.01, 1.0, 100)
    maes = []
    p95s = []
    maxes = []
    for cov in coverages:
        n_keep = max(1, int(cov * n_total))
        subset = sorted_errors[:n_keep]
        maes.append(np.mean(subset))
        p95s.append(np.percentile(subset, 95))
        maxes.append(np.max(subset))

    fig, ax = plt.subplots(figsize=(10, 7))
    setup_ax(ax)

    ax.plot(
        coverages * 100,
        maes,
        "-",
        color=PALETTE["deep_blue"],
        linewidth=2.5,
        label="MAE",
        zorder=3,
    )
    ax.plot(
        coverages * 100,
        p95s,
        "--",
        color=PALETTE["orange"],
        linewidth=2,
        label="P95 Error",
        zorder=3,
    )
    ax.plot(
        coverages * 100,
        maxes,
        ":",
        color=PALETTE["red"],
        linewidth=1.5,
        alpha=0.7,
        label="Max Error",
        zorder=3,
    )

    ax.set_xlabel(
        "Coverage (%) — Samples Retained (sorted by uncertainty)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Error (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Risk-Coverage Analysis: Uncertainty-Based Sample Filtering\n(MC Dropout, 20 samples, All 8 LOSO Folds)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(fontsize=10, framealpha=0.92, edgecolor=PALETTE["grid"])
    ax.set_xlim(0, 100)

    # Annotation
    mae_100 = maes[-1]
    mae_90 = maes[int(0.9 * len(maes)) - 1]
    ax.annotate(
        f"MAE at 100%: {mae_100:.2f} m\nMAE at 90%: {mae_90:.2f} m\nReduction: {(1 - mae_90 / mae_100) * 100:.1f}%",
        xy=(90, mae_90),
        xytext=(60, mae_90 + 5),
        fontsize=9,
        fontweight="bold",
        color=PALETTE["deep_blue"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["deep_blue"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["grid"], alpha=0.9),
    )

    save_fig(fig, "fig6_risk_coverage")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 68)
    print("GENERATING 6 ADDITIONAL FIGURES")
    print("=" * 68)

    # ── Fig 1: no inference needed ──
    plot_loso_per_fold()

    # ── Fig 4: no inference needed ──
    plot_error_vs_distance()

    # ── Figures needing inference: load data + models ──
    df, phys_params_global = prepare_data_and_models()

    print("\nRunning inference for all 8 folds...")
    fold_data = {}
    for fold in range(8):
        print(f"  Fold {fold}...")
        fold_data[fold] = run_fold_inference(df, phys_params_global, fold)
        print(f"    MAE: {fold_data[fold]['mae']:.3f} m")

    # ── Fig 2 ──
    plot_height_discrepancy([fold_data[f] for f in range(8)])

    # ── Fig 5 (use fold 0) ──
    target_fold = select_fold_for_sensor(fold_data, TARGET_SENSOR_UID)
    print(
        f"\nUsing sensor for Fig 5: {fold_data[target_fold]['sensor']} "
        f"(fold {target_fold})"
    )
    plot_residual_decomposition(fold_data, fold=target_fold)

    # ── Fig 6 ──
    plot_risk_coverage(fold_data, df, phys_params_global)

    print("\n" + "=" * 68)
    print("ALL ADDITIONAL FIGURES GENERATED")
    print(f"  Output: {OUT_DIR}/")
    print("=" * 68)


if __name__ == "__main__":
    main()
