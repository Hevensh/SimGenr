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

from world_generator.city.city_generator import _effective_developable_area_km2, generate_initial_cities
from world_generator.climate.climate_generator import generate_climate_baseline
from world_generator.core.config import load_world_config
from world_generator.core.random_state import build_rng_registry
from world_generator.hydrology.hydrology_generator import generate_hydrology
from world_generator.land.land_generator import generate_static_land
from world_generator.terrain.derivatives import derive_terrain_features
from world_generator.terrain.terrain_generator import generate_terrain_base


def _generate_city_world(config, grid):
    rngs = build_rng_registry(config.seed)
    terrain_base = generate_terrain_base(grid, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(terrain_base, grid)
    hydrology = generate_hydrology(terrain, grid, config.hydrology)
    land = generate_static_land(terrain, hydrology, grid, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        grid,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        grid,
        config.city,
        rngs.generator("evolution"),
    )
    return terrain, hydrology, land, city


def _draw_city_world(ax, terrain, hydrology, city, title: str) -> None:
    ax.imshow(terrain.elevation, origin="upper", cmap="terrain", alpha=0.92)
    urban = np.ma.masked_where(city.urban_density <= 0.03, city.urban_density)
    ax.imshow(urban, origin="upper", cmap="inferno", alpha=0.72, vmin=0.0, vmax=1.0)
    water = hydrology.river | hydrology.lake
    ax.imshow(np.ma.masked_where(~water, water), origin="upper", cmap="Blues", alpha=0.82, vmin=0.0, vmax=1.0)
    if city.cities:
        populations = np.asarray([item.population for item in city.cities], dtype=np.float64)
        marker_sizes = 24.0 + 92.0 * np.sqrt(populations / max(float(populations.max()), 1.0))
        ax.scatter(
            [item.col for item in city.cities],
            [item.row for item in city.cities],
            s=marker_sizes,
            marker="o",
            facecolor="#fff3b0",
            edgecolor="#20252b",
            linewidth=0.65,
            zorder=5,
        )
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


def _metrics(grid, land, hydrology, city) -> dict[str, float | int]:
    populations = np.asarray([item.population for item in city.cities], dtype=np.float64)
    return {
        "width_km": float(grid.width * grid.cell_size_km),
        "height_km": float(grid.height * grid.cell_size_km),
        "gross_area_km2": float(grid.width * grid.height * grid.cell_size_km**2),
        "effective_developable_area_km2": _effective_developable_area_km2(land, hydrology, grid),
        "city_count": len(city.cities),
        "total_population": float(populations.sum()) if populations.size else 0.0,
        "largest_city_population": float(populations.max()) if populations.size else 0.0,
        "urbanized_area_km2": float(city.urban_mask.sum() * grid.cell_size_km**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scale-aware city generation at 64 km and 128 km extents.")
    parser.add_argument("--config", default="configs/small_debug.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/city_scaling_comparison_seed42")
    args = parser.parse_args()

    config = load_world_config(args.config)
    config = replace(config, seed=args.seed, city=replace(config.city, scaling_mode="scale_aware"))
    large_grid = replace(
        config.world,
        height=128,
        width=128,
        cell_size_km=1.0,
        origin_x_km=0.0,
        origin_y_km=0.0,
    )
    small_grid = replace(large_grid, height=64, width=64)
    world_64 = _generate_city_world(config, small_grid)
    world_128 = _generate_city_world(config, large_grid)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.2), constrained_layout=True)
    _draw_city_world(axes[0], world_64[0], world_64[1], world_64[3], "64 x 64 km")
    _draw_city_world(axes[1], world_128[0], world_128[1], world_128[3], "128 x 128 km")
    figure.savefig(output / "city_scale_comparison.png", dpi=190, facecolor="white")
    plt.close(figure)

    metrics = {
        "seed": args.seed,
        "scaling_mode": config.city.scaling_mode,
        "world_64": _metrics(small_grid, world_64[2], world_64[1], world_64[3]),
        "world_128": _metrics(large_grid, world_128[2], world_128[1], world_128[3]),
    }
    (output / "city_scale_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved comparison to: {output}")


if __name__ == "__main__":
    main()
