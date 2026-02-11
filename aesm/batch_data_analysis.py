import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

LOG_DIR = "px4log/px4_logs_parsed"
OUTPUT_FILE = "sensor_uncertainty_stats.json"
BIAS_WINDOW = 50  # samples for per-flight offset removal


def _compute_baro_altitude(pressure_series):
    pressure = np.asarray(pressure_series, dtype=float)
    pressure = pressure[np.isfinite(pressure)]
    if pressure.size == 0:
        return None
    pressure_hpa = pressure / 100.0 if np.nanmean(pressure) > 5000 else pressure
    altitude = 44330 * (1 - (pressure_hpa / 1013.25) ** 0.1903)
    altitude = altitude[np.isfinite(altitude)]
    return altitude if altitude.size > 0 else None


def analyze_single_log(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None

        required_cols = {"pressure", "altitude_ellipsoid_m"}
        if not required_cols.issubset(df.columns):
            return None

        df = df.dropna(subset=list(required_cols))
        if df.empty or len(df) < 100:
            return None

        gnss_alt = df["altitude_ellipsoid_m"].values.astype(float)
        baro_alt = _compute_baro_altitude(df["pressure"].values)
        if baro_alt is None:
            return None

        min_len = min(len(gnss_alt), len(baro_alt))
        gnss_alt = gnss_alt[:min_len]
        baro_alt = baro_alt[:min_len]

        window = max(10, min(BIAS_WINDOW, min_len))
        bias = np.mean(baro_alt[:window] - gnss_alt[:window])
        baro_aligned = baro_alt - bias

        error = baro_aligned - gnss_alt

        if "timestamp_sample_gps" in df.columns:
            t = df["timestamp_sample_gps"].values[:min_len] / 1e6
        else:
            t = np.arange(min_len) * 0.1
        t = t - t[0]

        epv = df["epv"].values[:min_len] if "epv" in df.columns else np.array([])
        eph = df["eph"].values[:min_len] if "eph" in df.columns else np.array([])
        epv = epv[np.isfinite(epv) & (epv > 0)]
        eph = eph[np.isfinite(eph) & (eph > 0)]

        return {
            "errors": error,
            "errors_abs": np.abs(error),
            "epv": epv,
            "eph": eph
        }
    except Exception as exc:
        print(f"Error processing {filepath}: {exc}")
        return None


def _fit_histogram(ax, data, title, xlabel, color, precision=3, max_range=None):
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        ax.text(0.5, 0.5, "No Data Available", ha="center", va="center")
        return None, None, None

    lower = np.percentile(arr, 0.5)
    upper = np.percentile(arr, 99.5)
    trimmed = arr[(arr >= lower) & (arr <= upper)]
    if max_range is not None:
        trimmed = trimmed[trimmed <= max_range]
    if trimmed.size == 0:
        ax.text(0.5, 0.5, "No Data Available", ha="center", va="center")
        return None, None, None

    sns.histplot(trimmed, bins=80, stat="probability", kde=False, color=color, alpha=0.65, ax=ax)
    mu, sigma = norm.fit(trimmed)
    x = np.linspace(*ax.get_xlim(), 1000)
    fmt = f"{{:.{precision}f}}"
    ax.plot(x, norm.pdf(x, mu, sigma), "k", linewidth=2,
            label=f"$\mu={fmt.format(mu)}$ m\n$\sigma={fmt.format(sigma)}$ m")
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Probability Density", fontsize=15)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    limit = max_range if max_range is not None else trimmed.max()
    ax.set_xlim(0, limit * 1.05)
    ymax = ax.get_ylim()[1]
    return mu, sigma, ymax


def plot_results(baro_abs_errors, epv_values):
    if not baro_abs_errors and len(epv_values) == 0:
        print("No data to plot.")
        return {}

    sns.set_theme(style="ticks", font_scale=1.8)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    baro_arr = np.asarray(baro_abs_errors, dtype=float)
    baro_arr = baro_arr[np.isfinite(baro_arr)]
    baro_plot_limit = float(np.percentile(baro_arr, 95)) if baro_arr.size else None

    mu_err, sigma_err, ymax_err = _fit_histogram(
        axes[0],
        baro_abs_errors,
        "Barometer Absolute Height Error",
        "|Baro Height - GNSS Height| (m)",
        "#4C72B0",
        precision=3,
        max_range=baro_plot_limit
    )

    mu_epv = sigma_epv = ymax_epv = None
    epv_arr = np.asarray(epv_values, dtype=float)
    epv_arr = epv_arr[np.isfinite(epv_arr) & (epv_arr > 0)]
    if epv_arr.size > 0:
        mu_epv, sigma_epv, ymax_epv = _fit_histogram(
            axes[1],
            epv_arr,
            "GNSS Reported Vertical Accuracy (EPV)",
            "EPV (m)",
            "#55A868",
            precision=2,
            max_range=float(np.percentile(epv_arr, 99))
        )
    else:
        axes[1].text(0.5, 0.5, "No EPV Samples", ha="center", va="center")

    y_limits = [val for val in (ymax_err, ymax_epv) if val]
    if y_limits:
        y_max = max(y_limits) * 1.05
        if ymax_err:
            axes[0].set_ylim(0, y_max)
        if ymax_epv:
            axes[1].set_ylim(0, y_max)

    sns.despine()
    plt.tight_layout()
    plt.savefig("height_uncertainty_model.png", dpi=300)
    print("Saved plot to height_uncertainty_model.png")

    err_abs = np.asarray(baro_abs_errors, dtype=float)
    err_abs = err_abs[np.isfinite(err_abs)]
    stats = {
        "samples_total": int(err_abs.size),
        "abs_error_mean": float(np.mean(err_abs)) if err_abs.size else None,
        "abs_error_std": float(np.std(err_abs)) if err_abs.size else None,
        "abs_error_rms": float(np.sqrt(np.mean(err_abs ** 2))) if err_abs.size else None,
        "abs_error_fit_mu": float(mu_err) if mu_err is not None else None,
        "abs_error_fit_sigma": float(sigma_err) if sigma_err is not None else None,
        "epv_mean": float(np.mean(epv_arr)) if epv_arr.size else None,
        "epv_std": float(np.std(epv_arr)) if epv_arr.size else None,
        "epv_fit_mu": float(mu_epv) if mu_epv is not None else None,
        "epv_fit_sigma": float(sigma_epv) if sigma_epv is not None else None
    }
    return stats


def batch_process():
    log_files = glob.glob(os.path.join(LOG_DIR, "*.csv"))
    all_error_abs = []
    all_epv = []
    all_eph = []

    print(f"Analyzing {len(log_files)} logs from {LOG_DIR}...")
    valid_logs = 0

    for path in log_files:
        result = analyze_single_log(path)
        if not result:
            continue
        if result["errors_abs"].size == 0:
            continue
        all_error_abs.extend(result["errors_abs"])
        if result["epv"].size > 0:
            all_epv.extend(result["epv"])
        if result["eph"].size > 0:
            all_eph.extend(result["eph"])
        valid_logs += 1

    if valid_logs == 0:
        print("No valid logs processed.")
        return

    print(f"Processed {valid_logs} logs.")
    stats = plot_results(all_error_abs, all_epv)
    stats.update({
        "logs_used": valid_logs,
        "data_source": f"{len(all_error_abs)} aligned samples from RTK-qualified flights",
        "epv_samples": len(all_epv),
        "eph_mean": float(np.mean(all_eph)) if len(all_eph) else None,
        "eph_std": float(np.std(all_eph)) if len(all_eph) else None
    })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(stats, f, indent=4)

    print("\nUncertainty Statistics:")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    batch_process()
