import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_grid(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    lats = np.sort(df["avg_latitude"].unique())
    lons = np.sort(df["avg_longitude"].unique())
    lat_n, lon_n = len(lats), len(lons)

    def to_grid(col: str) -> np.ndarray:
        return df[col].values.reshape(lat_n, lon_n)

    mean_grid = to_grid("h_pred_mean")
    std_grid = to_grid("h_pred_std")
    return lats, lons, mean_grid, std_grid


def plot_contour(lats: np.ndarray, lons: np.ndarray, grid: np.ndarray, title: str, out_path: str) -> None:
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    plt.figure(figsize=(8, 6))
    cs = plt.contour(lon_mesh, lat_mesh, grid, levels=20, colors="k", linewidths=0.6)
    cf = plt.contourf(lon_mesh, lat_mesh, grid, levels=30, cmap="RdBu_r")
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    cbar = plt.colorbar(cf)
    cbar.set_label("Height (m)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(lats: np.ndarray, lons: np.ndarray, grid: np.ndarray, title: str, out_path: str) -> None:
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("Std (m)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize grid_slice.csv from infer")
    parser.add_argument("--csv", type=str, default=os.path.join("height_field_project", "artifacts", "grid_slice.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join("height_field_project", "artifacts"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lats, lons, mean_grid, std_grid = load_grid(args.csv)

    mean_path = os.path.join(args.out_dir, "grid_contour_mean.png")
    std_path = os.path.join(args.out_dir, "grid_heatmap_std.png")

    plot_contour(lats, lons, mean_grid, "Predicted Height (mean)", mean_path)
    plot_heatmap(lats, lons, std_grid, "Uncertainty (std)", std_path)

    print(f"Saved: {mean_path}")
    print(f"Saved: {std_path}")


if __name__ == "__main__":
    main()
