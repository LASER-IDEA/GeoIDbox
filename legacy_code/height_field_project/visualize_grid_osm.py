import argparse
import os

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.colormap import linear

matplotlib.use("Agg")


def load_grid(csv_path: str):
    df = pd.read_csv(csv_path)
    lats = np.sort(df["avg_latitude"].unique())
    lons = np.sort(df["avg_longitude"].unique())
    lat_n, lon_n = len(lats), len(lons)

    def to_grid(col: str) -> np.ndarray:
        return df[col].values.reshape(lat_n, lon_n)

    mean_grid = to_grid("h_pred_mean")
    std_grid = to_grid("h_pred_std")
    return df, lats, lons, mean_grid, std_grid


def save_raster(grid: np.ndarray, lats: np.ndarray, lons: np.ndarray, cmap: str, label: str, out_path: str):
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    plt.figure(figsize=(6, 6))
    im = plt.imshow(grid, origin="lower", extent=extent, cmap=cmap, aspect="auto", alpha=0.8)
    cbar = plt.colorbar(im)
    cbar.set_label(label)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Overlay grid_slice.csv onto OSM with mean & std layers")
    parser.add_argument("--csv", type=str, default=os.path.join("height_field_project", "artifacts", "grid_slice.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join("height_field_project", "artifacts"))
    parser.add_argument("--mean_png", type=str, default="grid_mean_overlay.png")
    parser.add_argument("--std_png", type=str, default="grid_std_overlay.png")
    parser.add_argument("--out_html", type=str, default="grid_osm.html")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, lats, lons, mean_grid, std_grid = load_grid(args.csv)

    mean_png_path = os.path.join(args.out_dir, args.mean_png)
    std_png_path = os.path.join(args.out_dir, args.std_png)
    out_html = os.path.join(args.out_dir, args.out_html)

    save_raster(mean_grid, lats, lons, cmap="RdBu_r", label="Height (m)", out_path=mean_png_path)
    save_raster(std_grid, lats, lons, cmap="viridis", label="Std (m)", out_path=std_png_path)

    bounds = [[lats.min(), lons.min()], [lats.max(), lons.max()]]
    center = [df["avg_latitude"].mean(), df["avg_longitude"].mean()]

    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    mean_overlay = folium.raster_layers.ImageOverlay(
        name="Predicted Height (mean)",
        image=mean_png_path,
        bounds=bounds,
        opacity=0.7,
        interactive=True,
        cross_origin=False,
    )
    std_overlay = folium.raster_layers.ImageOverlay(
        name="Uncertainty (std)",
        image=std_png_path,
        bounds=bounds,
        opacity=0.7,
        interactive=True,
        cross_origin=False,
    )
    mean_overlay.add_to(m)
    std_overlay.add_to(m)

    # Colorbars as legends
    mean_min, mean_max = np.percentile(mean_grid, [2, 98])
    std_min, std_max = np.percentile(std_grid, [2, 98])
    mean_cmap = linear.RdBu_11.scale(mean_min, mean_max)
    std_cmap = linear.viridis.scale(std_min, std_max)
    mean_cmap.caption = "Predicted Height (m)"
    std_cmap.caption = "Std (m)"
    mean_cmap.add_to(m)
    std_cmap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    print(f"Saved map: {out_html}")


if __name__ == "__main__":
    main()
