#!/usr/bin/env python3
"""
Generate per-UID visual summaries for the weekly sensor analysis outputs.

The script expects data/analyze_weekly_sensor_data.py to have been executed so
that per-week CSV/JSON artifacts exist under
data/weekly_sensor_data/analysis_outputs/<week>/.

Outputs:
  data/reports/figures/per_uid_horizontal_stability.png
  data/reports/figures/per_uid_dpdt_peaks.png
  data/reports/figures/per_uid_tp_slope.png
  data/reports/figures/macro_events_per_week.png
  data/reports/figures/per_uid_event_counts.png
plus helper CSVs:
  data/reports/figures/per_uid_metrics.csv
  data/reports/figures/per_uid_event_counts.csv
  data/reports/figures/active_weeks_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = DATA_ROOT / "weekly_sensor_data" / "analysis_outputs"
FIG_DIR = DATA_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_summary() -> pd.DataFrame:
    summary_path = ANALYSIS_DIR / "summary.json"
    summary = pd.read_json(summary_path)
    summary["week_dt"] = pd.to_datetime(summary["week"])
    active = summary[summary["skipped"] != True].copy()  # noqa: E712
    active.sort_values("week_dt", inplace=True)
    return active


def load_static(week: str) -> pd.DataFrame:
    return pd.read_csv(ANALYSIS_DIR / week / "static_truth.csv")


def compile_per_uid_metrics(active: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in active.iterrows():
        static = load_static(row["week"]).copy()
        if static.empty:
            continue
        static["week"] = row["week"]
        static["week_dt"] = row["week_dt"]
        frames.append(static)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def uid_color_map(uids: pd.Index) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {uid: cmap(i % cmap.N) for i, uid in enumerate(sorted(uids))}


def plot_per_uid_metric(per_uid: pd.DataFrame, value_col: str, ylabel: str, title: str, output_name: str) -> None:
    if per_uid.empty or value_col not in per_uid.columns:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = uid_color_map(per_uid["uid"].unique())
    for uid, group in per_uid.groupby("uid"):
        ax.plot(
            group["week_dt"],
            group[value_col],
            marker="o",
            label=uid,
            color=colors[uid],
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.savefig(FIG_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_macro_event_bars(active: pd.DataFrame) -> None:
    if active.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(active))
    ax.bar(x, active["macro_event_count"], color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(active["week"], rotation=35, ha="right")
    ax.set_ylabel("Macro events (top 5)")
    ax.set_title("Macro dp/dt events per week")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "macro_events_per_week.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_per_uid_event_counts(active: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in active.iterrows():
        counts = row.get("per_uid_event_counts")
        if isinstance(counts, dict):
            for uid, count in counts.items():
                records.append({"week": row["week"], "uid": uid, "count": count})
    return pd.DataFrame(records)


def plot_per_uid_event_counts(event_df: pd.DataFrame, active: pd.DataFrame) -> None:
    if event_df.empty:
        return
    pivot = (
        event_df.pivot(index="uid", columns="week", values="count")
        .reindex(columns=active["week"])
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_xlabel("Week")
    ax.set_ylabel("UID")
    ax.set_title("Per-UID dp/dt excursions (> threshold) per week")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_uid_event_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    active = load_summary()
    if active.empty:
        raise SystemExit("No analyzed weeks found.")

    per_uid = compile_per_uid_metrics(active)
    if per_uid.empty:
        raise SystemExit("No per-UID metrics available.")

    per_uid.to_csv(FIG_DIR / "per_uid_metrics.csv", index=False)
    active.to_csv(FIG_DIR / "active_weeks_summary.csv", index=False)

    plot_per_uid_metric(
        per_uid,
        value_col="horizontal_std_m",
        ylabel="Horizontal σ (m)",
        title="Per-UID coordinate stability",
        output_name="per_uid_horizontal_stability.png",
    )
    plot_per_uid_metric(
        per_uid,
        value_col="dpdt_peak_pa_per_min",
        ylabel="Peak |dp/dt| (Pa/min)",
        title="Per-UID dp/dt extremes",
        output_name="per_uid_dpdt_peaks.png",
    )
    plot_per_uid_metric(
        per_uid,
        value_col="tp_slope_pa_per_c",
        ylabel="ΔP/ΔT slope (Pa/°C)",
        title="Per-UID temperature-pressure coupling",
        output_name="per_uid_tp_slope.png",
    )

    plot_macro_event_bars(active)

    event_df = build_per_uid_event_counts(active)
    if not event_df.empty:
        event_df.to_csv(FIG_DIR / "per_uid_event_counts.csv", index=False)
        plot_per_uid_event_counts(event_df, active)

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
