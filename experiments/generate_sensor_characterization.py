"""
传感器特性分析: GNSS 定位/定高稳定性 + 气压计测量稳定性
用于回答 Reviewer 2.3 关于传感器硬件性能的问题

分析所有 8 个 GeoBox 传感器，生成:
  - 每个传感器的 GNSS 水平/垂直稳定性
  - 每个传感器的气压计短期噪声和长期稳定性
  - 全局对比图和 LaTeX 表格

输出目录: experiments/figures/sensor_characterization/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import os
from scipy.stats import circstd

# ── 样式 (与 generate_figures.py 一致) ─────────────────────────────────────
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
    "purple": "#7B3294",
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

OUT_DIR = "experiments/figures/sensor_characterization"

# ── 工具函数 ────────────────────────────────────────────────────────────────


def save_fig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
    print(f"  Saved: {name}.pdf/png")
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


def deg_to_m_lat(ddeg):
    return ddeg * 111320.0


def deg_to_m_lon(ddeg, lat_deg=22.607):
    return ddeg * 111320.0 * np.cos(np.radians(lat_deg))


def cep(x_m, y_m, pct=50):
    dist = np.sqrt(x_m**2 + y_m**2)
    return np.percentile(dist, pct)


# ════════════════════════════════════════════════════════════════════════════
# 主分析
# ════════════════════════════════════════════════════════════════════════════


def load_data(csv_path="data/sensor_data_filtered.csv"):
    df = pd.read_csv(csv_path, parse_dates=["processed_time"])
    df["uid_short"] = df["uid"].str[-6:]
    df = df.sort_values(["uid_short", "processed_time"]).reset_index(drop=True)
    return df


def compute_all_stats(df):
    results = []
    lat_ref = 22.607

    for uid in sorted(df["uid_short"].unique()):
        s = df[df["uid_short"] == uid].copy().sort_values("processed_time")
        n = len(s)

        # ── GNSS 水平 ──
        lat_mean = s["avg_latitude"].mean()
        lon_mean = s["avg_longitude"].mean()
        dx_m = deg_to_m_lon(s["avg_latitude"].values - lat_mean, lat_ref)
        dy_m = deg_to_m_lat(s["avg_longitude"].values - lon_mean)
        # 用经纬度残差计算
        dlat_m = deg_to_m_lat(s["avg_latitude"].values - lat_mean)
        dlon_m = deg_to_m_lon(s["avg_longitude"].values - lon_mean, lat_mean)
        horiz_std = np.sqrt(np.mean(dlat_m**2 + dlon_m**2))
        cep50 = cep(dlat_m, dlon_m, 50)
        cep95 = cep(dlat_m, dlon_m, 95)

        # ── GNSS 垂直 ──
        alt_mean = s["avg_altitude"].mean()
        alt_res = s["avg_altitude"].values - alt_mean
        alt_std = np.std(alt_res)
        alt_mad = np.median(np.abs(alt_res))
        alt_p95 = np.percentile(np.abs(alt_res), 95)
        alt_p99 = np.percentile(np.abs(alt_res), 99)
        alt_range = s["avg_altitude"].max() - s["avg_altitude"].min()

        # ── GNSS 信号质量 ──
        hdop_mean = s["avg_hdop"].mean()
        hdop_std = s["avg_hdop"].std()
        sat_mean = s["avg_satellites"].mean()
        sat_min = s["avg_satellites"].min()

        # ── 气压计 ──
        p_mean = s["avg_pressure"].mean()
        p_std_raw = s["avg_pressure"].std()

        s_idx = s.set_index("processed_time")
        # 5 分钟去趋势 (短期传感器噪声)
        p_roll5 = (
            s_idx["avg_pressure"].rolling("5min", center=True, min_periods=3).mean()
        )
        res5 = (s_idx["avg_pressure"] - p_roll5).dropna()
        p_noise_5min = res5.std()
        # 60 分钟去趋势 (天气尺度变化)
        p_roll60 = (
            s_idx["avg_pressure"].rolling("60min", center=True, min_periods=20).mean()
        )
        res60 = (s_idx["avg_pressure"] - p_roll60).dropna()
        p_noise_60min = res60.std()
        # 等效高度噪声: dH ≈ Hs * dP/P, Hs ≈ 8500m
        p_noise_5min_m = p_noise_5min / p_mean * 8500
        p_noise_60min_m = p_noise_60min / p_mean * 8500

        # ── 昼夜变化 ──
        s["hour"] = s["processed_time"].dt.hour
        hourly_alt = s.groupby("hour")["avg_altitude"].mean()
        diurnal_amp = hourly_alt.max() - hourly_alt.min()

        results.append(
            {
                "uid": uid,
                "n": n,
                "lat_mean": lat_mean,
                "lon_mean": lon_mean,
                "alt_mean": alt_mean,
                "horiz_std_m": horiz_std,
                "cep50_m": cep50,
                "cep95_m": cep95,
                "alt_std_m": alt_std,
                "alt_mad_m": alt_mad,
                "alt_p95_m": alt_p95,
                "alt_p99_m": alt_p99,
                "alt_range_m": alt_range,
                "diurnal_amp_m": diurnal_amp,
                "hdop_mean": hdop_mean,
                "hdop_std": hdop_std,
                "sat_mean": sat_mean,
                "sat_min": sat_min,
                "pressure_mean": p_mean,
                "pressure_std_raw": p_std_raw,
                "pressure_noise_5min_pa": p_noise_5min,
                "pressure_noise_60min_pa": p_noise_60min,
                "pressure_noise_5min_m": p_noise_5min_m,
                "pressure_noise_60min_m": p_noise_60min_m,
            }
        )

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════════
# 图 1: 全局对比 — GNSS 垂直噪声 vs 气压计等效高度噪声
# ════════════════════════════════════════════════════════════════════════════


def plot_gnss_vs_baro_noise(stats_df):
    print("\n[Fig] GNSS vs Barometer noise comparison...")
    fig, ax = plt.subplots(figsize=(12, 6))
    setup_ax(ax)

    uids = stats_df["uid"].values
    x = np.arange(len(uids))
    width = 0.25

    gnss_alt = stats_df["alt_std_m"].values
    baro_60 = stats_df["pressure_noise_60min_m"].values
    baro_5 = stats_df["pressure_noise_5min_m"].values

    bars1 = ax.bar(
        x - width,
        gnss_alt,
        width,
        color=PALETTE["orange"],
        alpha=0.88,
        edgecolor="white",
        linewidth=1.2,
        label="GNSS Altitude Std",
        zorder=3,
    )
    bars2 = ax.bar(
        x,
        baro_60,
        width,
        color=PALETTE["mid_blue"],
        alpha=0.88,
        edgecolor="white",
        linewidth=1.2,
        label="Baro Noise (60-min detrended)",
        zorder=3,
    )
    bars3 = ax.bar(
        x + width,
        baro_5,
        width,
        color=PALETTE["deep_blue"],
        alpha=0.88,
        edgecolor="white",
        linewidth=1.2,
        label="Baro Noise (5-min detrended)",
        zorder=3,
    )

    for bar, val in zip(bars1, gnss_alt):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=PALETTE["text"],
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
    for bar, val in zip(bars3, baro_5):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.15,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["deep_blue"],
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # 标注倍率
    for i in range(len(uids)):
        ratio = gnss_alt[i] / baro_5[i] if baro_5[i] > 0 else 0
        mid = (gnss_alt[i] + baro_5[i]) / 2
        ax.annotate(
            "",
            xy=(x[i] - width, gnss_alt[i] + 0.5),
            xytext=(x[i] + width, baro_5[i] + 0.5),
            arrowprops=dict(arrowstyle="<->", color=PALETTE["red"], lw=1.2, ls="--"),
        )
        ax.text(
            x[i],
            mid + 1.0,
            f"{ratio:.0f}×",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["red"],
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_ylabel("Equivalent Height Noise (m)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sensor ID", fontsize=13, fontweight="bold")
    ax.set_title(
        "GNSS Altitude Noise vs. Barometric Sensor Noise\n(8 GeoBox Sensors, 1-min Aggregated)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(uids, fontsize=10)
    ax.legend(fontsize=10, framealpha=0.92, edgecolor=PALETTE["grid"])
    ax.set_ylim(0, max(gnss_alt) * 1.25)

    save_fig(fig, "gnss_vs_baro_noise_comparison")


# ════════════════════════════════════════════════════════════════════════════
# 图 2: GNSS 水平散点 + CEP (每个传感器一个子图)
# ════════════════════════════════════════════════════════════════════════════


def plot_gnss_horizontal_scatter(df, stats_df):
    print("\n[Fig] GNSS horizontal scatter with CEP...")
    uids = sorted(df["uid_short"].unique())
    n_sensors = len(uids)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle(
        "GNSS Horizontal Position Stability\n(CEP50 / CEP95 Circles)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for idx, (uid, ax) in enumerate(zip(uids, axes.flatten())):
        setup_ax(ax)
        s = df[df["uid_short"] == uid]
        lat_mean = s["avg_latitude"].mean()
        lon_mean = s["avg_longitude"].mean()

        dlat_m = deg_to_m_lat(s["avg_latitude"].values - lat_mean)
        dlon_m = deg_to_m_lon(s["avg_longitude"].values - lon_mean, lat_mean)

        ax.scatter(
            dlon_m,
            dlat_m,
            s=1,
            alpha=0.3,
            c=SENSOR_COLORS[idx],
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
            label=f"CEP50={cep50:.1f}m",
        )
        circle95 = plt.Circle(
            (0, 0),
            cep95,
            fill=False,
            color=PALETTE["red"],
            linewidth=1.8,
            linestyle="--",
            label=f"CEP95={cep95:.1f}m",
        )
        ax.add_patch(circle50)
        ax.add_patch(circle95)

        ax.set_xlim(-cep95 * 1.3, cep95 * 1.3)
        ax.set_ylim(-cep95 * 1.3, cep95 * 1.3)
        ax.set_aspect("equal")
        ax.set_title(f"Sensor {uid}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_xlabel("ΔEast (m)", fontsize=9)
        ax.set_ylabel("ΔNorth (m)", fontsize=9)

    for idx in range(n_sensors, 8):
        axes.flatten()[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "gnss_horizontal_scatter_cep")


# ════════════════════════════════════════════════════════════════════════════
# 图 3: GNSS 高度时间序列 + 分布 (每个传感器)
# ════════════════════════════════════════════════════════════════════════════


def plot_gnss_altitude_analysis(df, stats_df):
    print("\n[Fig] GNSS altitude analysis...")
    uids = sorted(df["uid_short"].unique())
    n = len(uids)

    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    fig.suptitle(
        "GNSS Altitude Stability Analysis\n(Raw 1-min Samples, Moving Mean ± 1σ, and Error Distribution)",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    for idx, uid in enumerate(uids):
        ax_ts = axes[idx, 0]
        ax_hist = axes[idx, 1]
        setup_ax(ax_ts)
        setup_ax(ax_hist)

        s = df[df["uid_short"] == uid].sort_values("processed_time")
        alt = s["avg_altitude"].values
        t = s["processed_time"].values
        alt_mean = np.mean(alt)
        alt_std = np.std(alt)

        # 时间序列
        ax_ts.plot(
            t, alt, ".", markersize=1, alpha=0.3, color=SENSOR_COLORS[idx], zorder=3
        )
        # 滚动均值 ± 1σ
        s_copy = s.copy().set_index("processed_time")
        roll_mean = (
            s_copy["avg_altitude"].rolling("120min", center=True, min_periods=30).mean()
        )
        roll_std = (
            s_copy["avg_altitude"].rolling("120min", center=True, min_periods=30).std()
        )
        ax_ts.plot(
            roll_mean.index,
            roll_mean.values,
            "-",
            color=PALETTE["dark_red"],
            linewidth=1.2,
            label="2h rolling mean",
            zorder=4,
        )
        ax_ts.fill_between(
            roll_mean.index,
            roll_mean - roll_std,
            roll_mean + roll_std,
            alpha=0.2,
            color=PALETTE["dark_red"],
            zorder=2,
        )
        ax_ts.axhline(
            alt_mean,
            color=PALETTE["deep_blue"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax_ts.set_title(
            f"Sensor {uid} — Alt={alt_mean:.0f}m, σ={alt_std:.1f}m",
            fontsize=10,
            fontweight="bold",
        )
        ax_ts.set_ylabel("Altitude (m)", fontsize=9)
        ax_ts.legend(fontsize=7, loc="upper right")

        # 误差分布直方图
        err = alt - alt_mean
        bins = np.linspace(-3 * alt_std, 3 * alt_std, 60)
        ax_hist.hist(
            err,
            bins=bins,
            color=SENSOR_COLORS[idx],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            density=True,
            zorder=3,
        )
        # 正态拟合
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

        # 统计文字
        st = stats_df[stats_df["uid"] == uid].iloc[0]
        info = (
            f"Std: {alt_std:.1f} m\n"
            f"MAD: {st['alt_mad_m']:.1f} m\n"
            f"P95: {st['alt_p95_m']:.1f} m\n"
            f"Range: {st['alt_range_m']:.0f} m"
        )
        ax_hist.text(
            0.97,
            0.97,
            info,
            transform=ax_hist.transAxes,
            fontsize=8.5,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec=PALETTE["grid"], alpha=0.92
            ),
        )
        ax_hist.set_title(
            f"Sensor {uid} — Altitude Error Distribution",
            fontsize=10,
            fontweight="bold",
        )
        ax_hist.set_xlabel("Altitude Error (m)", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.legend(fontsize=8)

    plt.tight_layout(h_pad=2.0)
    save_fig(fig, "gnss_altitude_analysis")


# ════════════════════════════════════════════════════════════════════════════
# 图 4: 气压计分析 — 原始 + 去趋势 + 噪声分布
# ════════════════════════════════════════════════════════════════════════════


def plot_barometer_analysis(df, stats_df):
    print("\n[Fig] Barometer analysis...")
    uids = sorted(df["uid_short"].unique())
    n = len(uids)

    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    fig.suptitle(
        "Barometric Pressure Stability Analysis\n(Raw Pressure, 60-min Baseline, and 5-min Detrended Noise)",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    for idx, uid in enumerate(uids):
        ax_ts = axes[idx, 0]
        ax_hist = axes[idx, 1]
        setup_ax(ax_ts)
        setup_ax(ax_hist)

        s = df[df["uid_short"] == uid].sort_values("processed_time").copy()
        s_idx = s.set_index("processed_time")
        p = s_idx["avg_pressure"]
        p_mean = p.mean()

        # 原始气压 + 60 分钟滚动基线
        ax_ts.plot(
            p.index,
            p.values,
            ".",
            markersize=1,
            alpha=0.3,
            color=SENSOR_COLORS[idx],
            zorder=3,
        )
        p_roll60 = p.rolling("60min", center=True, min_periods=20).mean()
        ax_ts.plot(
            p_roll60.index,
            p_roll60.values,
            "-",
            color=PALETTE["dark_red"],
            linewidth=1.0,
            alpha=0.8,
            label="60-min baseline",
            zorder=4,
        )
        ax_ts.axhline(
            p_mean, color=PALETTE["deep_blue"], linestyle="--", linewidth=0.8, alpha=0.6
        )
        ax_ts.set_ylabel("Pressure (Pa)", fontsize=9)

        st = stats_df[stats_df["uid"] == uid].iloc[0]
        ax_ts.set_title(
            f"Sensor {uid} — Baro: σ_raw={st['pressure_std_raw']:.0f} Pa, "
            f"σ_5min={st['pressure_noise_5min_pa']:.1f} Pa ({st['pressure_noise_5min_m']:.2f} m)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax_ts.legend(fontsize=7, loc="upper right")

        # 5 分钟去趋势残差分布
        p_roll5 = p.rolling("5min", center=True, min_periods=3).mean()
        res5 = (p - p_roll5).dropna()
        noise_5 = res5.std()

        bins = np.linspace(-5 * noise_5, 5 * noise_5, 60)
        ax_hist.hist(
            res5.values,
            bins=bins,
            color=SENSOR_COLORS[idx],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            density=True,
            zorder=3,
        )

        from scipy.stats import norm

        x_fit = np.linspace(-5 * noise_5, 5 * noise_5, 200)
        ax_hist.plot(
            x_fit,
            norm.pdf(x_fit, 0, noise_5),
            "--",
            color=PALETTE["dark_red"],
            linewidth=1.5,
            label="Gaussian fit",
        )

        info = (
            f"5-min noise: {noise_5:.2f} Pa\n"
            f"  = {st['pressure_noise_5min_m']:.2f} m equiv\n"
            f"60-min noise: {st['pressure_noise_60min_pa']:.1f} Pa\n"
            f"  = {st['pressure_noise_60min_m']:.2f} m equiv"
        )
        ax_hist.text(
            0.97,
            0.97,
            info,
            transform=ax_hist.transAxes,
            fontsize=8.5,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec=PALETTE["grid"], alpha=0.92
            ),
        )
        ax_hist.set_title(
            f"Sensor {uid} — Detrended Pressure Noise (5-min)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax_hist.set_xlabel("Pressure Residual (Pa)", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.legend(fontsize=8)

    plt.tight_layout(h_pad=2.0)
    save_fig(fig, "barometer_analysis")


# ════════════════════════════════════════════════════════════════════════════
# 图 5: HDOP 分布 + 卫星数量 (箱线图)
# ════════════════════════════════════════════════════════════════════════════


def plot_hdop_satellite(df):
    print("\n[Fig] HDOP and satellite statistics...")
    uids = sorted(df["uid_short"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    setup_ax(ax1)
    setup_ax(ax2)

    hdop_data = []
    sat_data = []
    for uid in uids:
        s = df[df["uid_short"] == uid]
        hdop_data.append(s["avg_hdop"].values)
        sat_data.append(s["avg_satellites"].values)

    bp1 = ax1.boxplot(
        hdop_data,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=PALETTE["text"]),
        capprops=dict(color=PALETTE["text"]),
    )
    for patch, color in zip(bp1["boxes"], SENSOR_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax1.set_xticklabels(uids, fontsize=10)
    ax1.set_ylabel("HDOP", fontsize=12, fontweight="bold")
    ax1.set_title("Horizontal Dilution of Precision", fontsize=12, fontweight="bold")
    ax1.axhline(
        1.0,
        color=PALETTE["red"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="HDOP = 1.0 (good)",
    )
    ax1.axhline(
        2.0,
        color=PALETTE["dark_red"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="HDOP = 2.0 (fair)",
    )
    ax1.legend(fontsize=9)

    bp2 = ax2.boxplot(
        sat_data,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=PALETTE["text"]),
        capprops=dict(color=PALETTE["text"]),
    )
    for patch, color in zip(bp2["boxes"], SENSOR_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax2.set_xticklabels(uids, fontsize=10)
    ax2.set_ylabel("Number of Satellites", fontsize=12, fontweight="bold")
    ax2.set_title("GNSS Satellite Count", fontsize=12, fontweight="bold")
    ax2.axhline(
        8,
        color=PALETTE["orange"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="8 sats (min 3D fix)",
    )
    ax2.legend(fontsize=9)

    fig.suptitle(
        "GNSS Signal Quality Across 8 GeoBox Sensors",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    save_fig(fig, "hdop_satellite_distribution")


# ════════════════════════════════════════════════════════════════════════════
# 图 6: 昼夜变化 (所有传感器叠加)
# ════════════════════════════════════════════════════════════════════════════


def plot_diurnal_variation(df):
    print("\n[Fig] Diurnal altitude variation...")
    fig, ax = plt.subplots(figsize=(12, 6))
    setup_ax(ax)

    uids = sorted(df["uid_short"].unique())
    for idx, uid in enumerate(uids):
        s = df[df["uid_short"] == uid].copy()
        alt_mean = s["avg_altitude"].mean()
        s["hour"] = s["processed_time"].dt.hour
        hourly = s.groupby("hour")["avg_altitude"].agg(["mean", "std"])
        ax.plot(
            hourly.index,
            hourly["mean"] - alt_mean,
            "-o",
            color=SENSOR_COLORS[idx],
            markersize=4,
            linewidth=1.5,
            label=f"{uid} (σ={hourly['std'].mean():.1f}m)",
            alpha=0.85,
        )

    ax.axhline(0, color=PALETTE["light_text"], linestyle="-", linewidth=0.8)
    ax.set_xlabel("Hour of Day (local time)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Altitude Deviation from Mean (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Diurnal Altitude Variation (Multipath Pattern)\n(All 8 Sensors, 1-min Aggregated)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.legend(
        fontsize=8,
        ncol=2,
        framealpha=0.92,
        edgecolor=PALETTE["grid"],
        loc="upper right",
    )

    save_fig(fig, "diurnal_altitude_variation")


# ════════════════════════════════════════════════════════════════════════════
# 图 7: 综合雷达图 / 热力图 (传感器质量排名)
# ════════════════════════════════════════════════════════════════════════════


def plot_sensor_quality_heatmap(stats_df):
    print("\n[Fig] Sensor quality heatmap...")
    metrics = {
        "GNSS Alt Std ↓": "alt_std_m",
        "GNSS Horiz Std ↓": "horiz_std_m",
        "GNSS CEP95 ↓": "cep95_m",
        "GNSS Diurnal ↓": "diurnal_amp_m",
        "Baro Noise ↓": "pressure_noise_5min_m",
        "HDOP ↓": "hdop_mean",
    }

    labels = list(metrics.keys())
    uids = stats_df["uid"].values
    data = np.zeros((len(uids), len(labels)))
    for j, col in enumerate(metrics.values()):
        vals = stats_df[col].values
        # 归一化到 0-1 (越低越好 → 反转: 1=最好, 0=最差)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            data[:, j] = 1 - (vals - vmin) / (vmax - vmin)
        else:
            data[:, j] = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(uids)))
    ax.set_yticklabels([f"Sensor {u}" for u in uids], fontsize=10)

    for i in range(len(uids)):
        for j in range(len(labels)):
            val = data[i, j]
            col_name = metrics[labels[j]]
            raw = stats_df.iloc[i][col_name]
            text_color = "white" if val < 0.4 else PALETTE["text"]
            ax.text(
                j,
                i,
                f"{raw:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=text_color,
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Relative Quality (1=Best)", fontsize=10)
    ax.set_title(
        "Sensor Quality Ranking (Green=Best, Red=Worst)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    plt.tight_layout()
    save_fig(fig, "sensor_quality_heatmap")


# ════════════════════════════════════════════════════════════════════════════
# 打印 LaTeX 表格
# ════════════════════════════════════════════════════════════════════════════


def print_latex_tables(stats_df):
    print("\n" + "=" * 80)
    print("LaTeX TABLE 1: Sensor Characterization Summary")
    print("=" * 80)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(
        r"\caption{Sensor Characterization: GNSS and Barometric Performance (1-min Aggregated)}"
    )
    print(r"\label{tab:sensor_characterization}")
    print(r"\small")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Sensor & \multicolumn{3}{c}{GNSS} & \multicolumn{3}{c}{Barometer} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(
        r"& Alt $\sigma$ (m) & CEP95 (m) & HDOP & $\sigma_{5\min}$ (Pa) & $\sigma_{5\min}$ (m) & $\sigma_{\text{raw}}$ (Pa) \\"
    )
    print(r"\midrule")

    for _, row in stats_df.iterrows():
        print(
            f"{row['uid']} & {row['alt_std_m']:.1f} & {row['cep95_m']:.1f} & "
            f"{row['hdop_mean']:.2f} & {row['pressure_noise_5min_pa']:.1f} & "
            f"{row['pressure_noise_5min_m']:.2f} & {row['pressure_std_raw']:.0f} \\\\"
        )

    print(r"\midrule")
    means = stats_df.mean(numeric_only=True)
    print(
        f"\\textbf{{Mean}} & {means['alt_std_m']:.1f} & {means['cep95_m']:.1f} & "
        f"{means['hdop_mean']:.2f} & {means['pressure_noise_5min_pa']:.1f} & "
        f"{means['pressure_noise_5min_m']:.2f} & {means['pressure_std_raw']:.0f} \\\\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "=" * 80)
    print("LaTeX TABLE 2: GNSS Vertical Error Distribution")
    print("=" * 80)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{GNSS Altitude Error Statistics Across 8 GeoBox Sensors}")
    print(r"\label{tab:gnss_altitude_stats}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(
        r"Sensor & Mean Alt (m) & $\sigma$ (m) & MAD (m) & P95 (m) & Range (m) & Diurnal Amp (m) \\"
    )
    print(r"\midrule")

    for _, row in stats_df.iterrows():
        print(
            f"{row['uid']} & {row['alt_mean']:.0f} & {row['alt_std_m']:.1f} & "
            f"{row['alt_mad_m']:.1f} & {row['alt_p95_m']:.1f} & "
            f"{row['alt_range_m']:.0f} & {row['diurnal_amp_m']:.1f} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "=" * 80)
    print("LaTeX TABLE 3: Barometric Pressure Noise Analysis")
    print("=" * 80)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Barometric Pressure Noise Characterization}")
    print(r"\label{tab:baro_noise}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(
        r"Sensor & $\sigma_{\text{raw}}$ (Pa) & $\sigma_{5\min}$ (Pa) & $\sigma_{60\min}$ (Pa) & Equiv. Height $\sigma_{5\min}$ (m) \\"
    )
    print(r"\midrule")

    for _, row in stats_df.iterrows():
        print(
            f"{row['uid']} & {row['pressure_std_raw']:.0f} & "
            f"{row['pressure_noise_5min_pa']:.1f} & {row['pressure_noise_60min_pa']:.1f} & "
            f"{row['pressure_noise_5min_m']:.2f} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "=" * 80)
    print("KEY FINDING: Noise Ratio (GNSS alt / Baro equiv height)")
    print("=" * 80)
    for _, row in stats_df.iterrows():
        ratio = row["alt_std_m"] / row["pressure_noise_5min_m"]
        print(
            f"  Sensor {row['uid']}: GNSS σ = {row['alt_std_m']:.1f} m, "
            f"Baro σ_5min = {row['pressure_noise_5min_m']:.2f} m → "
            f"Ratio = {ratio:.0f}×"
        )

    mean_gnss = stats_df["alt_std_m"].mean()
    mean_baro = stats_df["pressure_noise_5min_m"].mean()
    print(f"\n  Mean ratio: {mean_gnss / mean_baro:.0f}×")


# ════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 68)
    print("SENSOR CHARACTERIZATION ANALYSIS")
    print("GNSS + Barometric Performance for 8 GeoBox Sensors")
    print("=" * 68)

    df = load_data()
    print(f"\nLoaded {len(df)} samples from {df['uid_short'].nunique()} sensors")

    stats_df = compute_all_stats(df)
    print("\nPer-sensor statistics:")
    print(stats_df.to_string(index=False))

    # 生成图表
    plot_gnss_vs_baro_noise(stats_df)
    plot_gnss_horizontal_scatter(df, stats_df)
    plot_gnss_altitude_analysis(df, stats_df)
    plot_barometer_analysis(df, stats_df)
    plot_hdop_satellite(df)
    plot_diurnal_variation(df)
    plot_sensor_quality_heatmap(stats_df)

    # 打印 LaTeX 表格
    print_latex_tables(stats_df)

    # 保存统计数据
    stats_path = os.path.join(OUT_DIR, "sensor_characterization_stats.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStats saved to: {stats_path}")

    print("\n" + "=" * 68)
    print("ALL ANALYSIS COMPLETE → experiments/figures/sensor_characterization/")
    print("=" * 68)


if __name__ == "__main__":
    main()
