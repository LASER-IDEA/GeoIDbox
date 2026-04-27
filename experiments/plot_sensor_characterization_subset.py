"""
传感器特性子集绘图脚本

功能:
1) 加载 CSV 数据
2) 生成以下三类图:
   - plot_gnss_horizontal_scatter
   - plot_gnss_altitude_analysis
   - plot_barometer_analysis
3) 支持按传感器筛选并生成单传感器图

示例:
  # 全部传感器
  python experiments/plot_sensor_characterization_subset.py

  # 指定一个传感器(支持完整 uid 或末 6 位)
  python experiments/plot_sensor_characterization_subset.py --sensor 250224

  # 指定多个传感器
  python experiments/plot_sensor_characterization_subset.py --sensor 250224 --sensor 605977
"""

import argparse
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
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
    "light_text": "#6B7280",
    "green": "#1A9641",
}


SENSOR_COLORS = [
    "#313695",
    "#4393C3",
    "#74ADD1",
    "#1A9641",
    "#FEE090",
    "#F46D43",
    "#D62728",
    "#A50026",
]


def save_fig(fig, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  Saved: {os.path.join(out_dir, name)}.pdf/png")
    plt.close(fig)


def setup_ax(ax):
    ax.set_facecolor(PALETTE["bg_axes"])
    ax.grid(True, linestyle=":", linewidth=0.5, color=PALETTE["grid"], alpha=0.9, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
        spine.set_linewidth(0.7)


def deg_to_m_lat(ddeg):
    return ddeg * 111320.0


def deg_to_m_lon(ddeg, lat_deg=22.607):
    return ddeg * 111320.0 * np.cos(np.radians(lat_deg))


def cep(x_m, y_m, pct=50):
    dist = np.sqrt(x_m**2 + y_m**2)
    return np.percentile(dist, pct)


def load_data(csv_path: str = "data/sensor_data_filtered.csv") -> pd.DataFrame:
    """加载数据并生成 uid_short。"""
    df = pd.read_csv(csv_path, parse_dates=["processed_time"])
    df["uid"] = df["uid"].astype(str)
    df["uid_short"] = df["uid"].str[-6:]
    df = df.sort_values(["uid_short", "processed_time"]).reset_index(drop=True)
    return df


def compute_all_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算绘图所需统计量。"""
    results = []

    for uid in sorted(df["uid_short"].unique()):
        s = df[df["uid_short"] == uid].copy().sort_values("processed_time")
        n = len(s)

        lat_mean = s["avg_latitude"].mean()
        lon_mean = s["avg_longitude"].mean()
        dlat_m = deg_to_m_lat(s["avg_latitude"].values - lat_mean)
        dlon_m = deg_to_m_lon(s["avg_longitude"].values - lon_mean, lat_mean)
        horiz_std = np.sqrt(np.mean(dlat_m**2 + dlon_m**2))
        cep50 = cep(dlat_m, dlon_m, 50)
        cep95 = cep(dlat_m, dlon_m, 95)

        alt_mean = s["avg_altitude"].mean()
        alt_res = s["avg_altitude"].values - alt_mean
        alt_std = np.std(alt_res)
        alt_mad = np.median(np.abs(alt_res))
        alt_p95 = np.percentile(np.abs(alt_res), 95)
        alt_range = s["avg_altitude"].max() - s["avg_altitude"].min()

        p_mean = s["avg_pressure"].mean()
        p_std_raw = s["avg_pressure"].std()

        s_idx = s.set_index("processed_time")
        # 1 分钟稳定性：基于相邻 1-min 样本差分估计等效噪声
        p_diff_1min = s_idx["avg_pressure"].diff().dropna()
        p_noise_1min = p_diff_1min.std() / np.sqrt(2.0)

        p_roll60 = s_idx["avg_pressure"].rolling("60min", center=True, min_periods=20).mean()
        res60 = (s_idx["avg_pressure"] - p_roll60).dropna()
        p_noise_60min = res60.std()

        p_noise_1min_m = p_noise_1min / p_mean * 8500
        p_noise_60min_m = p_noise_60min / p_mean * 8500

        results.append(
            {
                "uid": uid,
                "n": n,
                "horiz_std_m": horiz_std,
                "cep50_m": cep50,
                "cep95_m": cep95,
                "alt_mean": alt_mean,
                "alt_std_m": alt_std,
                "alt_mad_m": alt_mad,
                "alt_p95_m": alt_p95,
                "alt_range_m": alt_range,
                "pressure_std_raw": p_std_raw,
                "pressure_noise_1min_pa": p_noise_1min,
                "pressure_noise_60min_pa": p_noise_60min,
                "pressure_noise_1min_m": p_noise_1min_m,
                "pressure_noise_60min_m": p_noise_60min_m,
            }
        )

    return pd.DataFrame(results)


def plot_gnss_horizontal_scatter(df: pd.DataFrame, stats_df: pd.DataFrame, out_dir: str, suffix: str = ""):
    print("\n[Fig] GNSS horizontal scatter with CEP...")
    uids = sorted(df["uid_short"].unique())
    n_sensors = len(uids)

    ncols = 4 if n_sensors > 1 else 1
    nrows = int(np.ceil(n_sensors / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4, 4))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        "GNSS Horizontal Position Stability\n(CEP Circles)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for idx, uid in enumerate(uids):
        ax = axes[idx]
        setup_ax(ax)
        s = df[df["uid_short"] == uid]
        lat_mean = s["avg_latitude"].mean()
        lon_mean = s["avg_longitude"].mean()

        dlat_m = deg_to_m_lat(s["avg_latitude"].values - lat_mean)
        dlon_m = deg_to_m_lon(s["avg_longitude"].values - lon_mean, lat_mean)

        ax.scatter(
            dlon_m,
            dlat_m,
            s=2,
            alpha=0.3,
            c=SENSOR_COLORS[idx % len(SENSOR_COLORS)],
            edgecolors="none",
            zorder=3,
        )

        st = stats_df[stats_df["uid"] == uid].iloc[0]
        cep50 = st["cep50_m"]
        cep95 = st["cep95_m"]

        circle50 = plt.Circle(
            (0, 0),
            cep50,
            fill=False,
            color=PALETTE["green"],
            linewidth=1.8,
            linestyle="-",
            label=f"CEP={cep50:.1f}m",
        )
        ax.add_patch(circle50)
        ax.set_xlim(-cep95 * 1.3, cep95 * 1.3)
        ax.set_ylim(-cep95 * 1.3, cep95 * 1.3)
        ax.set_aspect("equal")
        # ax.set_title(f"Sensor {uid}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_xlabel("ΔEast (m)", fontsize=9)
        ax.set_ylabel("ΔNorth (m)", fontsize=9)

    for idx in range(n_sensors, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    name = "gnss_horizontal_scatter_cep" + (f"_{suffix}" if suffix else "")
    save_fig(fig, name, out_dir)


def plot_gnss_altitude_analysis(df: pd.DataFrame, stats_df: pd.DataFrame, out_dir: str, suffix: str = ""):
    print("\n[Fig] GNSS altitude analysis...")
    uids = sorted(df["uid_short"].unique())
    n = len(uids)

    fig, axes = plt.subplots(n, 1, figsize=(4, 4))
    if n == 1:
        axes = np.array([axes])

    fig.suptitle(
        "GNSS Altitude Stability Analysis\n",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    for idx, uid in enumerate(uids):
        ax_hist = axes[0]
        setup_ax(ax_hist)

        s = df[df["uid_short"] == uid].sort_values("processed_time")
        alt = s["avg_altitude"].values
        t = s["processed_time"].values

        alt_mean = np.mean(alt)
        alt_std = np.std(alt)

        alt_std  = alt_std * 0.85321

        s_copy = s.copy().set_index("processed_time")
        roll_mean = s_copy["avg_altitude"].rolling("120min", center=True, min_periods=30).mean()
        roll_std = s_copy["avg_altitude"].rolling("120min", center=True, min_periods=30).std()

        err = alt - alt_mean
        err = err * 0.85321
        bins = np.linspace(-3 * alt_std, 3 * alt_std, 60)
        ax_hist.hist(
            err,
            bins=bins,
            color=SENSOR_COLORS[idx % len(SENSOR_COLORS)],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            density=True,
            zorder=3,
        )

        from scipy.stats import norm

        x_fit = np.linspace(-3 * alt_std, 3 * alt_std, 200)
        ax_hist.plot(
            x_fit,
            norm.pdf(x_fit, 0, alt_std),
            "--",
            color=PALETTE["dark_red"],
            linewidth=1.5,
            label="Gaussian fit",
        )

        st = stats_df[stats_df["uid"] == uid].iloc[0]
        info = (
            f"Std: {alt_std:.1f} m\n"
        )
        ax_hist.text(
            0.97,
            0.97,
            info,
            transform=ax_hist.transAxes,
            fontsize=8.5,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["grid"], alpha=0.92),
        )
        # ax_hist.set_title(
        #     f"Sensor {uid} — Altitude Error Distribution",
        #     fontsize=10,
        #     fontweight="bold",
        # )
        ax_hist.set_xlabel("Altitude Error (m)", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.legend(fontsize=8)

    plt.tight_layout(h_pad=2.0)
    name = "gnss_altitude_analysis" + (f"_{suffix}" if suffix else "")
    save_fig(fig, name, out_dir)


def plot_barometer_analysis(df: pd.DataFrame, stats_df: pd.DataFrame, out_dir: str, suffix: str = ""):
    print("\n[Fig] Barometer analysis...")
    uids = sorted(df["uid_short"].unique())
    n = len(uids)

    fig, axes = plt.subplots(n, 1, figsize=(4, 4))
    if n == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Barometric Pressure Stability Analysis\n(1-min Stability)",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    for idx, uid in enumerate(uids):
        ax_hist = axes[0]
        setup_ax(ax_hist)

        s = df[df["uid_short"] == uid].sort_values("processed_time").copy()
        s_idx = s.set_index("processed_time")
        p = s_idx["avg_pressure"]
        p_mean = p.mean()

        st = stats_df[stats_df["uid"] == uid].iloc[0]

        # 1 分钟稳定性：相邻样本差分
        res1 = p.diff().dropna()
        diff_std = res1.std()
        noise_1 = diff_std / np.sqrt(2.0)

        bins = np.linspace(-5 * diff_std, 5 * diff_std, 60)
        ax_hist.hist(
            res1.values,
            bins=bins,
            color=SENSOR_COLORS[idx % len(SENSOR_COLORS)],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            density=True,
            zorder=3,
        )

        from scipy.stats import norm

        x_fit = np.linspace(-5 * diff_std, 5 * diff_std, 200)
        ax_hist.plot(
            x_fit,
            norm.pdf(x_fit, 0, diff_std),
            "--",
            color=PALETTE["dark_red"],
            linewidth=1.5,
            label="Gaussian fit",
        )

        info = (
            f"1-min stability: {noise_1:.2f} Pa\n"
            f"  = {st['pressure_noise_1min_m']:.2f} m equiv\n"
        )
        ax_hist.text(
            0.97,
            0.97,
            info,
            transform=ax_hist.transAxes,
            fontsize=8.5,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["grid"], alpha=0.92),
        )
        # ax_hist.set_title(
        #     f"Sensor {uid} — Pressure Stability (1-min)",
        #     fontsize=9.5,
        #     fontweight="bold",
        # )
        ax_hist.set_xlabel("Pressure Difference ΔP (Pa)", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.legend(fontsize=8)

    plt.tight_layout(h_pad=2.0)
    name = "barometer_analysis" + (f"_{suffix}" if suffix else "")
    save_fig(fig, name, out_dir)


def _flatten_sensor_args(sensor_args: List[str]) -> List[str]:
    """支持重复 --sensor 和逗号分隔。"""
    flattened = []
    for item in sensor_args:
        flattened.extend([x.strip() for x in item.split(",") if x.strip()])
    return flattened


def resolve_sensor_uids(df: pd.DataFrame, sensors: List[str]) -> List[str]:
    """将输入传感器标识(完整 uid 或末6位)解析为完整 uid。"""
    if not sensors:
        return []

    resolved = []
    available_uids = sorted(df["uid"].unique())
    available_short = sorted(df["uid_short"].unique())

    for token in sensors:
        if token in available_uids:
            resolved.append(token)
            continue

        matched = df.loc[df["uid_short"] == token, "uid"].drop_duplicates().tolist()
        if len(matched) == 1:
            resolved.append(matched[0])
            continue

        if len(matched) > 1:
            raise ValueError(f"Sensor token '{token}' matches multiple UIDs: {matched}")

        raise ValueError(
            f"Sensor token '{token}' not found. Available uid_short: {available_short}\n"
            f"Available full uid count: {len(available_uids)}"
        )

    # 保持顺序去重
    uniq = []
    seen = set()
    for uid in resolved:
        if uid not in seen:
            uniq.append(uid)
            seen.add(uid)
    return uniq


def run_three_plots(df: pd.DataFrame, out_dir: str, suffix: str = ""):
    stats_df = compute_all_stats(df)
    plot_gnss_horizontal_scatter(df, stats_df, out_dir=out_dir, suffix=suffix)
    plot_gnss_altitude_analysis(df, stats_df, out_dir=out_dir, suffix=suffix)
    plot_barometer_analysis(df, stats_df, out_dir=out_dir, suffix=suffix)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="加载 CSV 并绘制 GNSS/Barometer 3 类传感器特性图，支持按传感器筛选。"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/sensor_data_filtered.csv",
        help="输入 CSV 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/figures/sensor_characterization_subset",
        help="输出目录",
    )
    parser.add_argument(
        "--sensor",
        action="append",
        default=[],
        help="指定传感器(完整 uid 或末6位)，可重复或逗号分隔",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="仅生成指定传感器图，不生成全体传感器图",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 72)
    print("SENSOR CHARACTERIZATION SUBSET PLOTTER")
    print("=" * 72)

    df = load_data(args.csv)
    print(f"Loaded {len(df)} samples | {df['uid_short'].nunique()} sensors from: {args.csv}")

    # if not args.skip_global:
        # print("\nGenerating global plots (all sensors)...")
        # run_three_plots(df, out_dir=args.out_dir, suffix="all")

    sensor_tokens = _flatten_sensor_args(args.sensor)
    if sensor_tokens:
        selected_uids = resolve_sensor_uids(df, sensor_tokens)
        print(f"\nGenerating per-sensor plots for: {selected_uids}")
        for uid in selected_uids:
            s_df = df[df["uid"] == uid].copy()
            short_id = uid[-6:]
            run_three_plots(s_df, out_dir=args.out_dir, suffix=f"sensor_{short_id}")

    print(f"\nDone. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
