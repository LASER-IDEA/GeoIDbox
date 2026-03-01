#!/usr/bin/env python3
"""
Batch analysis for weekly height-box sensor data focusing on per-UID metrics.

The script scans the 1-minute aggregated weekly CSVs, filters out weeks with
<= 5 active UIDs, and for the remaining weeks performs:

1. Static truth recovery: per-UID coordinate/altitude stability metrics.
2. dp/dt event tagging: per-UID extrema plus macro (multi-UID) events only.
3. Temperature-pressure coupling: ΔT vs ΔP regression per UID.

Outputs:
  data/weekly_sensor_data/analysis_outputs/<week>/
    - static_truth.csv
    - pressure_events.json
    - tp_coupling_per_uid.csv
    - metadata.json
  data/reports/weekly_sensor_insights.md    (human-readable summary)
  data/weekly_sensor_data/analysis_outputs/summary.json
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DATA_ROOT = Path(__file__).resolve().parent
AGG_WEEKLY_DIR = DATA_ROOT / "weekly_sensor_data_1min"
OUTPUT_DIR = DATA_ROOT / "weekly_sensor_data" / "analysis_outputs"
REPORTS_DIR = DATA_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis hyper-parameters
PRESSURE_JUMP_THRESHOLD_PA = 15.0  # dp/dt threshold (Pa/min) for event detection
MACRO_EVENT_RATIO = 0.6            # >= 60% of boxes => macro event
MAX_EVENTS_PER_CATEGORY = 5
EARTH_RADIUS_M = 6_371_000
METERS_PER_DEG_LAT = 111_320.0


def extract_week_label(csv_path: Path) -> str:
    """Derive week label (YYYY-MM-DD) from filename."""
    stem = csv_path.stem  # e.g. sensor_data_week_2025-10-27_1min_agg
    prefix = "sensor_data_week_"
    suffix = "_1min_agg"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        raise ValueError(f"Unexpected weekly file name: {csv_path.name}")
    return stem[len(prefix) : -len(suffix)]


def meters_per_deg_lon(lat_deg: float) -> float:
    return METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters."""
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return EARTH_RADIUS_M * c


def cluster_radius(uids: Iterable[str], coords: Dict[str, Tuple[float, float]]) -> float:
    """Return max distance from centroid for the provided UID cluster."""
    pts = [coords[uid] for uid in uids if uid in coords]
    if not pts:
        return float("nan")
    if len(pts) == 1:
        return 0.0
    mean_lat = sum(p[0] for p in pts) / len(pts)
    mean_lon = sum(p[1] for p in pts) / len(pts)
    return max(haversine_m(lat, lon, mean_lat, mean_lon) for lat, lon in pts)


def compute_macro_events(
    dp: pd.DataFrame,
    coords: Dict[str, Tuple[float, float]],
    uid_count: int,
) -> List[dict]:
    """Return multi-UID macro events based on simultaneous dp/dt excursions."""
    if dp.empty:
        return []

    macro_min = max(3, math.ceil(uid_count * MACRO_EVENT_RATIO))
    macro_events: List[dict] = []

    for ts, row in dp.iterrows():
        if row.isna().all():
            continue

        for direction, subset in (
            ("drop", row[row < -PRESSURE_JUMP_THRESHOLD_PA]),
            ("rise", row[row > PRESSURE_JUMP_THRESHOLD_PA]),
        ):
            if len(subset) < macro_min:
                continue
            affected = subset.index.tolist()
            severity = float(subset.abs().mean())
            radius = cluster_radius(affected, coords)
            event = {
                "time": ts.isoformat(),
                "type": direction,
                "affected_count": len(affected),
                "uids": sorted(affected, key=lambda u: abs(subset[u]), reverse=True),
                "mean_delta_pa": severity,
                "max_delta_pa": float(subset.abs().max()),
                "cluster_radius_m": radius,
            }
            macro_events.append(event)

    macro_events.sort(key=lambda e: e["mean_delta_pa"], reverse=True)
    return macro_events[:MAX_EVENTS_PER_CATEGORY]


def compute_per_uid_events(dp: pd.DataFrame) -> Dict[str, List[dict]]:
    """Return per-UID dp/dt extremes."""
    per_uid: Dict[str, List[dict]] = {}
    if dp.empty:
        return per_uid

    for uid in dp.columns:
        series = dp[uid].dropna()
        hits = series[series.abs() > PRESSURE_JUMP_THRESHOLD_PA]
        if hits.empty:
            continue
        top_idx = hits.abs().sort_values(ascending=False).head(MAX_EVENTS_PER_CATEGORY).index
        events = []
        for ts in top_idx:
            val = series.loc[ts]
            events.append(
                {
                    "time": ts.isoformat(),
                    "type": "rise" if val > 0 else "drop",
                    "delta_pa": float(val),
                }
            )
        per_uid[uid] = events
    return per_uid


def compute_tp_coupling_per_uid(
    pressure_pivot: pd.DataFrame, temp_pivot: pd.DataFrame
) -> pd.DataFrame:
    """Return per-UID ΔT/ΔP coupling metrics (no cross-UID aggregation)."""

    def _fit(series_temp: pd.Series, series_press: pd.Series) -> Tuple[float, float]:
        aligned = pd.concat([series_temp, series_press], axis=1, keys=["temp", "pressure"]).dropna()
        if len(aligned) < 2:
            return float("nan"), float("nan")
        x = (aligned["temp"] - aligned["temp"].mean()).to_numpy()
        y = (aligned["pressure"] - aligned["pressure"].mean()).to_numpy()
        xx = float(np.dot(x, x))
        if xx == 0.0:
            return float("nan"), float("nan")
        slope = float(np.dot(x, y) / xx)
        corr = float(np.corrcoef(x, y)[0, 1]) if len(aligned) > 2 else float("nan")
        return slope, corr

    records = []
    for uid in sorted(set(pressure_pivot.columns) & set(temp_pivot.columns)):
        slope, corr = _fit(temp_pivot[uid], pressure_pivot[uid])
        records.append(
            {
                "uid": uid,
                "tp_slope_pa_per_c": slope,
                "tp_corr": corr,
                "tp_r2": corr ** 2 if not math.isnan(corr) else float("nan"),
            }
        )

    return pd.DataFrame(records)


def summarize_static_truth(static_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived spatial stability metrics to the per-UID frame."""
    if static_df.empty:
        return static_df
    static_df = static_df.copy()
    static_df["lat_std_m"] = static_df["std_lat"].fillna(0.0) * METERS_PER_DEG_LAT
    static_df["lon_std_m"] = static_df.apply(
        lambda row: (row["std_lon"] if not pd.isna(row["std_lon"]) else 0.0)
        * meters_per_deg_lon(row["mean_lat"]),
        axis=1,
    )
    static_df["horizontal_std_m"] = (static_df["lat_std_m"] ** 2 + static_df["lon_std_m"] ** 2) ** 0.5
    return static_df


def process_week(csv_path: Path) -> dict:
    week_label = extract_week_label(csv_path)
    df = pd.read_csv(csv_path, parse_dates=["processed_time"])
    df = df.sort_values("processed_time").reset_index(drop=True)

    uid_count = df["uid"].nunique()
    base_summary = {
        "week": week_label,
        "uid_count": int(uid_count),
        "total_minutes": int(len(df)),
        "raw_sample_estimate": int(df["record_count"].sum()),
    }

    if uid_count <= 5:
        base_summary["skipped"] = True
        return base_summary

    pressure_pivot = df.pivot(index="processed_time", columns="uid", values="avg_pressure")
    temp_pivot = df.pivot(index="processed_time", columns="uid", values="avg_temperature")
    dp = pressure_pivot.diff()

    static_df = (
        df.groupby("uid")
        .agg(
            sample_minutes=("processed_time", "count"),
            raw_samples=("record_count", "sum"),
            mean_lat=("avg_latitude", "mean"),
            std_lat=("avg_latitude", "std"),
            mean_lon=("avg_longitude", "mean"),
            std_lon=("avg_longitude", "std"),
            mean_alt=("avg_altitude", "mean"),
            std_alt=("avg_altitude", "std"),
            mean_height=("avg_height", "mean"),
            std_height=("avg_height", "std"),
            mean_pressure=("avg_pressure", "mean"),
            std_pressure=("avg_pressure", "std"),
            mean_temp=("avg_temperature", "mean"),
            std_temp=("avg_temperature", "std"),
            mean_sat=("avg_satellites", "mean"),
            mean_hdop=("avg_hdop", "mean"),
        )
        .reset_index()
    )
    static_df = summarize_static_truth(static_df)

    coords_lookup = {row.uid: (row.mean_lat, row.mean_lon) for row in static_df.itertuples(index=False)}
    macro_events = compute_macro_events(dp, coords_lookup, uid_count)
    per_uid_events = compute_per_uid_events(dp)

    max_dpdt = dp.abs().max()
    dpdt_peak = max_dpdt.rename("dpdt_peak_pa_per_min").reset_index().rename(columns={"index": "uid"})
    static_df = static_df.merge(dpdt_peak, on="uid", how="left")

    tp_per_uid = compute_tp_coupling_per_uid(pressure_pivot, temp_pivot)
    static_df = static_df.merge(tp_per_uid, on="uid", how="left")

    week_dir = OUTPUT_DIR / week_label
    week_dir.mkdir(parents=True, exist_ok=True)
    static_out = week_dir / "static_truth.csv"
    events_out = week_dir / "pressure_events.json"
    tp_uid_out = week_dir / "tp_coupling_per_uid.csv"
    meta_out = week_dir / "metadata.json"

    static_df.to_csv(static_out, index=False)
    with open(events_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "macro_events": macro_events,
                "per_uid_events": per_uid_events,
                "dpdt_threshold_pa_per_min": PRESSURE_JUMP_THRESHOLD_PA,
                "macro_ratio": MACRO_EVENT_RATIO,
            },
            f,
            indent=2,
        )
    tp_per_uid.to_csv(tp_uid_out, index=False)

    duration_hours = (df["processed_time"].max() - df["processed_time"].min()).total_seconds() / 3600.0
    best_anchor_row = static_df.nsmallest(1, "horizontal_std_m").iloc[0]

    week_summary = {
        **base_summary,
        "skipped": False,
        "duration_hours": duration_hours,
        "macro_event_count": len(macro_events),
        "best_anchor_uid": best_anchor_row.uid,
        "best_anchor_horizontal_std_m": float(best_anchor_row.horizontal_std_m),
        "per_uid_event_counts": {uid: len(events) for uid, events in per_uid_events.items()},
    }

    if macro_events:
        week_summary["strongest_macro_event"] = macro_events[0]

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(week_summary, f, indent=2)

    return week_summary


def build_markdown_report(summaries: List[dict]) -> str:
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    lines = [
        "# Weekly Sensor Insights",
        "",
        f"_Generated on {timestamp}_",
        "",
    ]
    for summary in summaries:
        if summary.get("skipped"):
            lines.append(f"- Week {summary['week']} skipped (only {summary['uid_count']} active UID).")
            continue

        lines.append(f"## Week {summary['week']}")
        lines.append(
            f"- Active UIDs: {summary['uid_count']} covering {summary['duration_hours']:.1f} hours "
            f"({summary['raw_sample_estimate']:,} raw samples)."
        )
        if summary["macro_event_count"]:
            lines.append(f"- Macro events detected: {summary['macro_event_count']}.")
            if summary.get("strongest_macro_event"):
                event = summary["strongest_macro_event"]
                lines.append(
                    f"  * Strongest macro {event['type']} at {event['time']} "
                    f"({event['affected_count']} UIDs, mean Δp {event['mean_delta_pa']:.1f} Pa)."
                )
        else:
            lines.append("- Macro events detected: none.")

        lines.append(
            f"- Strongest anchor: {summary['best_anchor_uid']} "
            f"(horizontal σ ≈ {summary['best_anchor_horizontal_std_m']:.2f} m)."
        )

        per_uid_events = summary.get("per_uid_event_counts") or {}
        if per_uid_events:
            busiest_uid, busiest_count = max(per_uid_events.items(), key=lambda kv: kv[1])
            lines.append(
                f"- Most dp/dt excursions (> {PRESSURE_JUMP_THRESHOLD_PA:.0f} Pa/min): "
                f"{busiest_uid} with {busiest_count} events (see per-UID catalog)."
            )
        else:
            lines.append(f"- No dp/dt excursions above {PRESSURE_JUMP_THRESHOLD_PA:.0f} Pa/min.")

        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    weekly_files = sorted(AGG_WEEKLY_DIR.glob("sensor_data_week_*_1min_agg.csv"))
    if not weekly_files:
        raise FileNotFoundError(f"No weekly files found in {AGG_WEEKLY_DIR}")

    summaries = [process_week(path) for path in weekly_files]

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    report_text = build_markdown_report(summaries)
    report_path = REPORTS_DIR / "weekly_sensor_insights.md"
    report_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
