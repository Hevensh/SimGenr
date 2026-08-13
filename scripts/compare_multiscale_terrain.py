from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.core.config import load_world_config
from world_generator.core.random_state import build_rng_registry
from world_generator.hydrology.hydrology_generator import generate_hydrology
from world_generator.terrain.derivatives import derive_terrain_features
from world_generator.terrain.terrain_generator import generate_terrain_base


def _terrain(config, grid):
    rng = build_rng_registry(config.seed).generator("terrain")
    return generate_terrain_base(grid, config.terrain, rng)


def _plot_map(ax, values, title, cmap="terrain", vmin=None, vmax=None):
    image = ax.imshow(values, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunk-consistent multiscale terrain generation.")
    parser.add_argument("--config", default="configs/small_debug.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/terrain_v2_comparison_seed42")
    args = parser.parse_args()

    config = load_world_config(args.config)
    config = replace(config, seed=args.seed)
    terrain_config = replace(config.terrain, algorithm="multiscale_v2")
    hydrology_config = replace(config.hydrology, algorithm="conditioned_v2")
    config = replace(config, terrain=terrain_config, hydrology=hydrology_config)
    large_grid = replace(
        config.world,
        height=128,
        width=128,
        cell_size_km=1.0,
        origin_x_km=0.0,
        origin_y_km=0.0,
    )
    small_grid = replace(large_grid, height=64, width=64)
    terrain_64 = _terrain(config, small_grid)
    terrain_128 = _terrain(config, large_grid)

    tile_rows = []
    for origin_y in (0.0, 64.0):
        tile_columns = []
        for origin_x in (0.0, 64.0):
            tile_grid = replace(
                small_grid,
                origin_x_km=origin_x,
                origin_y_km=origin_y,
            )
            tile_columns.append(_terrain(config, tile_grid).elevation)
        tile_rows.append(np.concatenate(tile_columns, axis=1))
    stitched = np.concatenate(tile_rows, axis=0)
    difference = np.abs(stitched - terrain_128.elevation)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    vmin = float(min(terrain_64.elevation.min(), terrain_128.elevation.min()))
    vmax = float(max(terrain_64.elevation.max(), terrain_128.elevation.max()))
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.8), constrained_layout=True)
    image = _plot_map(axes[0, 0], terrain_64.elevation, "64 x 64 km crop", vmin=vmin, vmax=vmax)
    _plot_map(axes[0, 1], terrain_128.elevation, "128 x 128 km world", vmin=vmin, vmax=vmax)
    _plot_map(axes[1, 0], stitched, "Four independently sampled 64 km tiles", vmin=vmin, vmax=vmax)
    diff_image = _plot_map(
        axes[1, 1],
        difference,
        "Absolute stitching difference",
        cmap="magma",
        vmin=0.0,
        vmax=max(float(difference.max()), 1e-6),
    )
    figure.colorbar(
        image,
        ax=[axes[0, 0], axes[0, 1], axes[1, 0]],
        shrink=0.78,
        label="Elevation (m)",
    )
    figure.colorbar(diff_image, ax=axes[1, 1], shrink=0.72, label="Absolute difference (m)")
    figure.suptitle(f"Multiscale terrain chunk consistency, seed {args.seed}", fontsize=15)
    figure.savefig(output / "terrain_chunk_consistency.png", dpi=180)
    plt.close(figure)

    features_64 = derive_terrain_features(terrain_64, small_grid)
    features_128 = derive_terrain_features(terrain_128, large_grid)
    hydrology_64 = generate_hydrology(features_64, small_grid, hydrology_config)
    hydrology_128 = generate_hydrology(features_128, large_grid, hydrology_config)

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), constrained_layout=True)
    for ax, features, hydrology, title in (
        (axes[0], features_64, hydrology_64, "64 x 64 km"),
        (axes[1], features_128, hydrology_128, "128 x 128 km"),
    ):
        ax.imshow(features.elevation, cmap="terrain", origin="upper", alpha=0.92)
        river = np.ma.masked_where(~hydrology.river, hydrology.river)
        lake = np.ma.masked_where(~hydrology.lake, hydrology.lake)
        ax.imshow(river, cmap="Blues", origin="upper", alpha=0.88, vmin=0.0, vmax=1.0)
        ax.imshow(lake, cmap="winter", origin="upper", alpha=0.92, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
    figure.suptitle(
        "Hydrology extracted from conditioned DEM and physical catchment area",
        fontsize=15,
    )
    figure.savefig(output / "hydrology_scale_comparison.png", dpi=180)
    plt.close(figure)

    np.savez_compressed(
        output / "comparison_metrics.npz",
        maximum_stitching_difference_m=np.asarray(float(difference.max())),
        river_cells_64=np.asarray(int(hydrology_64.river.sum())),
        river_cells_128=np.asarray(int(hydrology_128.river.sum())),
        lake_cells_64=np.asarray(int(hydrology_64.lake.sum())),
        lake_cells_128=np.asarray(int(hydrology_128.lake.sum())),
    )
    metrics = {
        "maximum_stitching_difference_m": float(difference.max()),
        "elevation_min_64_m": float(terrain_64.elevation.min()),
        "elevation_max_64_m": float(terrain_64.elevation.max()),
        "elevation_relief_64_m": float(np.ptp(terrain_64.elevation)),
        "elevation_min_128_m": float(terrain_128.elevation.min()),
        "elevation_max_128_m": float(terrain_128.elevation.max()),
        "elevation_relief_128_m": float(np.ptp(terrain_128.elevation)),
        "river_cells_64": int(hydrology_64.river.sum()),
        "river_cells_128": int(hydrology_128.river.sum()),
        "lake_cells_64": int(hydrology_64.lake.sum()),
        "lake_cells_128": int(hydrology_128.lake.sum()),
    }
    (output / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"maximum stitching difference: {float(difference.max()):.6f} m")
    print(json.dumps(metrics, indent=2))
    print(f"saved comparison to: {output}")


if __name__ == "__main__":
    main()
