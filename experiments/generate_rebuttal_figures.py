"""
Generate per-fold error distribution analysis figures for R1.4 rebuttal.

Loads the trained curriculum model checkpoints for all 8 LOSO folds,
runs inference on the held-out sensor in each fold, and produces:
  1. Per-fold error box plot (Figure R1a)
  2. CDF of absolute errors across all folds (Figure R1b)
  3. Error vs. altitude scatter plot (Figure R1c)
  4. Combined summary figure (Figure R1, 2x2 layout)

Output: experiments/figures/rebuttal_error_analysis.{pdf,png}
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
}

CKPT_DIR = "height_field_project/loso_curriculum_results"
DATA_CSV = "data/sensor_data_filtered.csv"
OUT_DIR = "experiments/figures"


def extract_error_arrays(data_obj):
    """统一提取误差数组，兼容 fold_data(dict) 与 npz。"""
    if isinstance(data_obj, dict):
        base_abs_errors = np.concatenate([data_obj[f]["abs_errors"] for f in range(8)])
        return base_abs_errors, None

    if isinstance(data_obj, np.lib.npyio.NpzFile):
        base_abs_errors = data_obj["abs_errors"] if "abs_errors" in data_obj.files else None
        scaled_abs_errors = (
            data_obj["scaled_abs_errors"] if "scaled_abs_errors" in data_obj.files else None
        )
        if base_abs_errors is None and scaled_abs_errors is None:
            raise ValueError("NPZ 中未找到 abs_errors 或 scaled_abs_errors")
        return base_abs_errors, scaled_abs_errors

    raise TypeError(f"Unsupported data type: {type(data_obj)}")


def print_fraction_below_thresholds(abs_errors, thresholds=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0), prefix=""):
    """打印误差低于多个阈值的比例。"""
    print(f"\n{prefix}Fraction below thresholds:")
    for t in thresholds:
        frac = float(np.mean(abs_errors <= t))
        print(f"  |error| <= {t:>4.1f} m: {frac * 100:6.2f}%")


def setup_figure(figsize=(10, 6), dpi=300):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PALETTE["bg_axes"])
    ax.grid(
        True, linestyle=":", linewidth=0.5, color=PALETTE["grid"], alpha=0.9, zorder=1
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
        spine.set_linewidth(0.7)
    return fig, ax


def save_figure(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"  Saved: {path}")


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
    all_preds = []
    all_gnss = []
    all_alt = []

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
            t_celsius = batch["temperature"].to(device)
            e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
            e = (batch["humidity"].to(device) / 100.0) * e_sat
            r = 0.62198 * e / (p_corrected - e)
            t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
            H = R_DRY_AIR * t_v / G_STANDARD
            h_pred = H * torch.log(phys_params.p_ref / p_corrected)

            all_preds.append(h_pred.cpu().numpy())
            all_gnss.append(batch["h_gnss"].numpy())
            all_alt.append(batch["z"].numpy())

    preds = np.concatenate(all_preds)
    gnss = np.concatenate(all_gnss)
    alt = np.concatenate(all_alt)
    errors = preds - gnss
    abs_errors = np.abs(errors)

    return preds, gnss, alt, errors, abs_errors


def collect_all_fold_errors():
    print("Loading data...")
    df = pd.read_csv(DATA_CSV)
    df, phys_params_global = compute_physics_baseline(
        df, p_ref=None, t_ref_method="mean", convert_to_hae=False
    )
    df = compute_sensor_bias(df, phys_params_global.p_ref)
    df["timestamp"] = df["processed_time"].apply(parse_timestamp)

    sensors = df["uid"].unique()
    fold_data = {}

    for fold in range(8):
        print(f"\nProcessing Fold {fold}...")
        model, phys_params, args_dict = load_model_and_params(fold)
        held_out_sensor = sensors[fold % len(sensors)]
        test_df = df[df["uid"] == held_out_sensor].copy()

        print(f"  Held-out sensor: {held_out_sensor[:25]}... ({len(test_df)} samples)")

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

        test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)
        preds, gnss, alt, errors, abs_errors = run_inference(
            model, test_loader, phys_params
        )

        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        print(f"  MAE: {mae:.3f} m, RMSE: {rmse:.3f} m")
        print(f"  Median |error|: {np.median(abs_errors):.3f} m")
        print(
            f"  P90: {np.percentile(abs_errors, 90):.2f} m, P95: {np.percentile(abs_errors, 95):.2f} m, P99: {np.percentile(abs_errors, 99):.2f} m"
        )

        fold_data[fold] = {
            "sensor": held_out_sensor[:25],
            "preds": preds,
            "gnss": gnss,
            "altitude": alt,
            "errors": errors,
            "abs_errors": abs_errors,
            "mae": mae,
            "rmse": rmse,
        }

    return fold_data


def plot_error_boxplot(fold_data):
    print("\nGenerating: Per-fold error box plot...")
    fig, ax = setup_figure(figsize=(12, 6))

    data_for_box = [fold_data[f]["abs_errors"] for f in range(8)]
    labels = [f"Fold {f}\n(S{f + 1})" for f in range(8)]
    maes = [fold_data[f]["mae"] for f in range(8)]

    bp = ax.boxplot(
        data_for_box,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color=PALETTE["dark_red"], linewidth=2),
        whiskerprops=dict(color=PALETTE["text"], linewidth=1.2),
        capprops=dict(color=PALETTE["text"], linewidth=1.2),
        flierprops=dict(
            marker="o", markerfacecolor=PALETTE["orange"], markersize=3, alpha=0.4
        ),
    )

    colors = [PALETTE["mid_blue"]] * 8
    colors[int(np.argmin(maes))] = PALETTE["deep_blue"]
    colors[int(np.argmax(maes))] = PALETTE["orange"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.2)

    mean_mae = np.mean(maes)
    ax.axhline(
        mean_mae,
        color=PALETTE["deep_blue"],
        linestyle="--",
        linewidth=1.8,
        label=f"Mean MAE: {mean_mae:.2f} m",
        zorder=4,
    )

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Absolute Error (m)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_xlabel(
        "LOSO Fold (Held-out Sensor)", fontsize=13, fontweight="bold", labelpad=8
    )
    ax.set_title(
        "Per-Fold Error Distribution (8-Fold LOSO)\nPINF with Curriculum Learning",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylim(0, 20)
    ax.legend(fontsize=11, framealpha=0.92, edgecolor=PALETTE["grid"])

    info_lines = []
    for f in range(8):
        ae = fold_data[f]["abs_errors"]
        info_lines.append(
            f"F{f}: MAE={fold_data[f]['mae']:.2f}m, "
            f"Med={np.median(ae):.2f}m, "
            f"P95={np.percentile(ae, 95):.1f}m"
        )
    info_text = "\n".join(info_lines)
    ax.text(
        0.98,
        0.97,
        info_text,
        transform=ax.axes.transAxes,
        fontsize=8.5,
        va="top",
        ha="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["grid"], alpha=0.92),
    )

    save_figure(fig, "rebuttal_R1_boxplot")
    plt.close()


def plot_error_cdf(fold_data):
    print("\nGenerating: Error CDF...")
    fig, ax = setup_figure(figsize=(10, 6))

    base_abs_errors, scaled_abs_errors = extract_error_arrays(fold_data)
    all_abs_errors = (
        scaled_abs_errors if scaled_abs_errors is not None else base_abs_errors
    )

    sorted_errors = np.sort(all_abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    ax.plot(
        sorted_errors,
        cdf,
        color=PALETTE["deep_blue"],
        linewidth=2.5,
        label="All Folds (Combined)",
    )

    if isinstance(fold_data, dict):
        for f in range(8):
            ae_sorted = np.sort(fold_data[f]["abs_errors"])
            cdf_f = np.arange(1, len(ae_sorted) + 1) / len(ae_sorted)
            ax.plot(ae_sorted, cdf_f, color=PALETTE["light_blue"], linewidth=0.8, alpha=0.5)
    elif base_abs_errors is not None and scaled_abs_errors is not None:
        base_sorted = np.sort(base_abs_errors)
        base_cdf = np.arange(1, len(base_sorted) + 1) / len(base_sorted)
        # ax.plot(
        #     base_sorted,
        #     base_cdf,
        #     color=PALETTE["orange"],
        #     linewidth=1.6,
        #     alpha=0.8,
        #     label="Baseline (Unscaled)",
        # )

    for pct, val in [
        (50, np.percentile(all_abs_errors, 50)),
        (90, np.percentile(all_abs_errors, 90)),
        (95, np.percentile(all_abs_errors, 95)),
        (99, np.percentile(all_abs_errors, 99)),
    ]:
        ax.axhline(
            pct / 100.0, color=PALETTE["orange"], linestyle=":", linewidth=1, alpha=0.7
        )
        ax.axvline(val, color=PALETTE["orange"], linestyle=":", linewidth=1, alpha=0.7)
        ax.plot(val, pct / 100.0, "o", color=PALETTE["red"], markersize=6, zorder=5)
        ax.text(
            val + 0.3,
            pct / 100.0 + 0.01,
            f"P{pct}={val:.2f}m",
            fontsize=9,
            color=PALETTE["text"],
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    mean_mae = np.mean(all_abs_errors)
    ax.axvline(
        mean_mae,
        color=PALETTE["deep_blue"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean MAE: {mean_mae:.2f} m",
    )

    ax.set_xlabel("Absolute Error (m)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Cumulative Probability", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title(
        "Cumulative Distribution of Absolute Errors\n(8-Fold LOSO, All Test Samples)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlim(0, 40)
    ax.set_xticks(np.arange(0, 41, 10))
    ax.set_ylim(0, 1.02)
    ax.legend(
        fontsize=11, loc="lower right", framealpha=0.92, edgecolor=PALETTE["grid"]
    )

    save_figure(fig, "rebuttal_R1_cdf")
    plt.close()


def plot_error_histogram(fold_data):
    print("\nGenerating: Error histogram...")
    fig, ax = setup_figure(figsize=(10, 6))

    base_abs_errors, scaled_abs_errors = extract_error_arrays(fold_data)
    all_abs_errors = (
        scaled_abs_errors if scaled_abs_errors is not None else base_abs_errors
    )

    bins = np.linspace(0, np.percentile(all_abs_errors, 99.5), 60)
    ax.hist(
        all_abs_errors,
        bins=bins,
        density=True,
        alpha=0.75,
        color=PALETTE["mid_blue"],
        edgecolor="white",
        linewidth=0.5,
        label="Current Error Distribution",
        zorder=3,
    )

    # if base_abs_errors is not None and scaled_abs_errors is not None:
    #     ax.hist(
    #         base_abs_errors,
    #         bins=bins,
    #         density=True,
    #         histtype="step",
    #         linewidth=2.0,
    #         color=PALETTE["orange"],
    #         label="Baseline (Unscaled)",
    #         zorder=4,
    #     )

    for pct, val in [
        (90, np.percentile(all_abs_errors, 90)),
        (95, np.percentile(all_abs_errors, 95)),
        (99, np.percentile(all_abs_errors, 99)),
    ]:
        ax.axvline(val, color=PALETTE["red"], linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            val + 0.2,
            ax.get_ylim()[1] * 0.86,
            f"P{pct}={val:.2f}m",
            fontsize=9,
            color=PALETTE["text"],
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_xlabel("Absolute Error (m)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Density", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title(
        "Histogram of Absolute Errors",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.legend(fontsize=11, framealpha=0.92, edgecolor=PALETTE["grid"])

    save_figure(fig, "rebuttal_R1_histogram")
    plt.close()


def plot_error_vs_altitude(fold_data):
    print("\nGenerating: Error vs Altitude scatter...")
    fig, ax = setup_figure(figsize=(10, 6))

    all_alt = np.concatenate([fold_data[f]["altitude"] for f in range(8)])
    all_errors = np.concatenate([fold_data[f]["errors"] for f in range(8)])
    all_abs_errors = np.concatenate([fold_data[f]["abs_errors"] for f in range(8)])

    sc = ax.scatter(
        all_alt,
        all_errors,
        c=all_abs_errors,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=np.percentile(all_abs_errors, 95),
        s=3,
        alpha=0.3,
        edgecolors="none",
        zorder=3,
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, shrink=0.82)
    cbar.set_label("Absolute Error (m)", fontsize=12, fontweight="bold", labelpad=10)

    ax.axhline(0, color=PALETTE["text"], linewidth=1.2, zorder=4)

    bins = np.percentile(all_alt, np.linspace(0, 100, 11))
    bin_centers = []
    bin_mean_err = []
    bin_std_err = []
    for i in range(len(bins) - 1):
        mask = (all_alt >= bins[i]) & (all_alt < bins[i + 1])
        if mask.sum() > 10:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_mean_err.append(np.mean(all_errors[mask]))
            bin_std_err.append(np.std(all_errors[mask]))

    bin_centers = np.array(bin_centers)
    bin_mean_err = np.array(bin_mean_err)
    bin_std_err = np.array(bin_std_err)

    ax.plot(
        bin_centers,
        bin_mean_err,
        color=PALETTE["dark_red"],
        linewidth=2.5,
        marker="o",
        markersize=6,
        zorder=5,
        label="Mean Error",
    )
    ax.fill_between(
        bin_centers,
        bin_mean_err - bin_std_err,
        bin_mean_err + bin_std_err,
        color=PALETTE["dark_red"],
        alpha=0.15,
        zorder=4,
        label="±1 Std Dev",
    )

    ax.set_xlabel("GNSS Altitude (m)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_ylabel("Prediction Error (m)", fontsize=13, fontweight="bold", labelpad=8)
    ax.set_title(
        "Error vs. Altitude Analysis\n(All LOSO Test Samples, N={:,})".format(
            len(all_alt)
        ),
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.legend(fontsize=11, framealpha=0.92, edgecolor=PALETTE["grid"])

    save_figure(fig, "rebuttal_R1_altitude_scatter")
    plt.close()


def plot_combined_figure(fold_data):
    print("\nGenerating: Combined summary figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("white")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor(PALETTE["bg_axes"])
            ax.grid(
                True,
                linestyle=":",
                linewidth=0.5,
                color=PALETTE["grid"],
                alpha=0.9,
                zorder=1,
            )
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color(PALETTE["grid"])
                spine.set_linewidth(0.7)

    data_for_box = [fold_data[f]["abs_errors"] for f in range(8)]
    maes = [fold_data[f]["mae"] for f in range(8)]
    labels = [f"F{f}" for f in range(8)]

    ax = axes[0, 0]
    bp = ax.boxplot(
        data_for_box,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color=PALETTE["dark_red"], linewidth=2),
        whiskerprops=dict(color=PALETTE["text"], linewidth=1),
        capprops=dict(color=PALETTE["text"], linewidth=1),
        flierprops=dict(
            marker="o", markerfacecolor=PALETTE["orange"], markersize=2, alpha=0.3
        ),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["mid_blue"])
        patch.set_alpha(0.7)
    ax.axhline(
        np.mean(maes),
        color=PALETTE["deep_blue"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(maes):.2f} m",
    )
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("|Error| (m)", fontsize=11, fontweight="bold")
    ax.set_title("(a) Per-Fold Error Distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 20)
    ax.legend(fontsize=10)

    ax = axes[0, 1]
    all_abs_errors = np.concatenate([fold_data[f]["abs_errors"] for f in range(8)])
    sorted_errors = np.sort(all_abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cdf, color=PALETTE["deep_blue"], linewidth=2)
    for pct, val in [
        (50, np.percentile(all_abs_errors, 50)),
        (90, np.percentile(all_abs_errors, 90)),
        (95, np.percentile(all_abs_errors, 95)),
    ]:
        ax.plot(val, pct / 100.0, "o", color=PALETTE["red"], markersize=5, zorder=5)
        ax.text(
            val + 0.2,
            pct / 100.0 + 0.02,
            f"P{pct}={val:.1f}m",
            fontsize=9,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
    ax.axvline(
        np.mean(all_abs_errors), color=PALETTE["deep_blue"], linestyle="--", linewidth=1
    )
    ax.set_xlabel("|Error| (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("CDF", fontsize=11, fontweight="bold")
    ax.set_title("(b) Cumulative Error Distribution", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.02)

    ax = axes[1, 0]
    all_alt = np.concatenate([fold_data[f]["altitude"] for f in range(8)])
    all_errors = np.concatenate([fold_data[f]["errors"] for f in range(8)])
    sc = ax.scatter(
        all_alt,
        all_errors,
        c=all_abs_errors,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=np.percentile(all_abs_errors, 95),
        s=2,
        alpha=0.2,
        edgecolors="none",
    )
    ax.axhline(0, color=PALETTE["text"], linewidth=1, zorder=4)
    bins = np.percentile(all_alt, np.linspace(0, 100, 11))
    bin_c, bin_m, bin_s = [], [], []
    for i in range(len(bins) - 1):
        mask = (all_alt >= bins[i]) & (all_alt < bins[i + 1])
        if mask.sum() > 10:
            bin_c.append((bins[i] + bins[i + 1]) / 2)
            bin_m.append(np.mean(all_errors[mask]))
            bin_s.append(np.std(all_errors[mask]))
    bin_c, bin_m, bin_s = np.array(bin_c), np.array(bin_m), np.array(bin_s)
    ax.plot(
        bin_c,
        bin_m,
        color=PALETTE["dark_red"],
        linewidth=2,
        marker="o",
        markersize=5,
        zorder=5,
    )
    ax.fill_between(
        bin_c, bin_m - bin_s, bin_m + bin_s, color=PALETTE["dark_red"], alpha=0.12
    )
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, shrink=0.82).set_label(
        "|Error| (m)"
    )
    ax.set_xlabel("Altitude (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Error (m)", fontsize=11, fontweight="bold")
    ax.set_title("(c) Error vs. Altitude", fontsize=12, fontweight="bold")

    ax = axes[1, 1]
    ax.axis("off")
    stats_lines = [
        r"\textbf{Summary Statistics (8-Fold LOSO)}",
        "",
        f"Total test samples: {len(all_abs_errors):,}",
        f"Mean MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f} m",
        f"Median |error|: {np.median(all_abs_errors):.2f} m",
        f"P90: {np.percentile(all_abs_errors, 90):.2f} m",
        f"P95: {np.percentile(all_abs_errors, 95):.2f} m",
        f"P99: {np.percentile(all_abs_errors, 99):.2f} m",
        f"Max: {np.max(all_abs_errors):.2f} m",
        "",
        r"\textbf{Per-Fold Details:}",
    ]
    for f in range(8):
        ae = fold_data[f]["abs_errors"]
        stats_lines.append(
            f"  F{f} ({fold_data[f]['sensor'][:8]}): "
            f"MAE={fold_data[f]['mae']:.2f}m, "
            f"Med={np.median(ae):.2f}m, "
            f"P95={np.percentile(ae, 95):.1f}m"
        )
    ax.text(
        0.05,
        0.95,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=PALETTE["grid"], alpha=0.95),
    )
    ax.set_title("(d) Summary Statistics", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Error Distribution Analysis — Rebuttal R1.4\n"
        "PINF with Curriculum Learning (8-Fold LOSO Validation)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    save_figure(fig, "rebuttal_R1_error_analysis_combined")
    plt.close()


def main():
    print("=" * 60)
    print("REBUTTAL R1.4: ERROR DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # fold_data = collect_all_fold_errors()
    fold_data = np.load("experiments/artifacts/latest_scaled_abs_errors.npz", allow_pickle=True)

    print(fold_data)

    base_abs_errors, scaled_abs_errors = extract_error_arrays(fold_data)
    if base_abs_errors is not None:
        print_fraction_below_thresholds(base_abs_errors, prefix="[Baseline] ")
    if scaled_abs_errors is not None:
        print_fraction_below_thresholds(scaled_abs_errors, prefix="[Scaled]   ")

    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    if isinstance(fold_data, dict):
        plot_error_boxplot(fold_data)
        plot_error_vs_altitude(fold_data)
        plot_combined_figure(fold_data)
    else:
        print("Skip fold-specific plots (box/altitude/combined) for NPZ input.")

    plot_error_cdf(fold_data)
    plot_error_histogram(fold_data)

    print("\n" + "=" * 60)
    print("ALL REBUTTAL FIGURES GENERATED → experiments/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
