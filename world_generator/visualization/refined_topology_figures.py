from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import add_deduped_legend, draw_elevation_with_water_overlay, draw_grid_edge_lines, save_single_map
from world_generator.visualization.land_use_figures import _building_texture_overlay


REFINED_TOPOLOGY_FILES = [
    "refined_topology_overview.png",
    "transit_bus_candidates.png",
    "segmented_line_overlay.png",
    "refined_line_route_map.png",
]


def save_refined_topology_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_map(static_maps["refined_line_route_map"], output_dir / "refined_line_route_map.png", "Refined line route density", "magma", vmin=0.0, vmax=1.0)
    _save_segmented_line_overlay(static_maps, output_dir / "segmented_line_overlay.png")
    _save_transit_bus_candidates(static_maps, output_dir / "transit_bus_candidates.png")
    _save_refined_topology_overview(static_maps, output_dir / "refined_topology_overview.png")
    return REFINED_TOPOLOGY_FILES


def _save_refined_topology_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    image = draw_elevation_with_water_overlay(ax, static_maps)
    _draw_routes(ax, static_maps)
    _draw_buses(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("Refined topology over terrain/hydrology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    image = ax.imshow(static_maps["refined_line_route_map"], cmap="magma", origin="upper", vmin=0.0, vmax=1.0)
    _draw_buses(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("Transit buses over refined route density")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_segmented_line_overlay(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = draw_elevation_with_water_overlay(ax, static_maps)
    _draw_built_environment_texture(ax, static_maps)
    _draw_routes(ax, static_maps)
    _draw_buses(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("Segmented line routes with transit buses")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_built_environment_texture(ax: object, static_maps: dict[str, np.ndarray]) -> None:
    required = {
        "land_use_zone",
        "load_density_base",
        "economic_activity",
        "residential",
        "commercial",
        "industrial",
        "agriculture",
        "park_green",
    }
    if not required.issubset(static_maps):
        return
    overlay = _building_texture_overlay(
        static_maps["land_use_zone"],
        static_maps["load_density_base"],
        static_maps["economic_activity"],
        static_maps["residential"],
        static_maps["commercial"],
        static_maps["industrial"],
        static_maps["agriculture"],
        static_maps["park_green"],
        scale=3,
    )
    height, width = static_maps["land_use_zone"].shape
    ax.imshow(
        overlay,
        origin="upper",
        extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        interpolation="nearest",
        zorder=1,
    )


def _save_transit_bus_candidates(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(static_maps["line_route_map"], cmap="magma", origin="upper", vmin=0.0, vmax=1.0)
    _draw_buses(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("Transit buses over initial route density")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_routes(ax: object, static_maps: dict[str, np.ndarray]) -> None:
    if "refined_grid_buses" in static_maps and "refined_grid_edges" in static_maps:
        draw_grid_edge_lines(
            ax,
            static_maps["refined_grid_buses"],
            static_maps["refined_grid_edges"],
            linewidth=1.55,
            alpha=0.74,
        )
        return
    route = static_maps["refined_line_route_map"]
    overlay = np.zeros((*route.shape, 4), dtype=np.float32)
    overlay[route > 0.0] = [0.04, 0.04, 0.04, 0.66]
    overlay[route > 0.5] = [0.0, 0.0, 0.0, 0.84]
    ax.imshow(overlay, origin="upper")


def _draw_buses(ax: object, static_maps: dict[str, np.ndarray]) -> None:
    _scatter_mask(ax, static_maps["load_bus_map"] >= 0, "o", "#ff4fa3", 36, "load bus")
    source_mask = static_maps["source_bus_map"] >= 0
    _scatter_mask(ax, source_mask & (static_maps["wind_candidate_map"] >= 0), "^", "#38d0ff", 42, "wind bus")
    _scatter_mask(ax, source_mask & (static_maps["pv_candidate_map"] >= 0), "s", "#ffcf33", 38, "solar bus")
    _scatter_mask(ax, static_maps["thermal_bus_map"] >= 0, "D", "#f25f2c", 40, "thermal bus")
    _scatter_mask(ax, static_maps["transit_bus_map"] >= 0, "P", "#f7f7f7", 58, "transit bus")


def _scatter_mask(ax: object, mask: np.ndarray, marker: str, color: str, size: int, label: str) -> None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return
    ax.scatter(cols, rows, s=size, c=color, edgecolors="black", linewidths=0.65, marker=marker, label=label, zorder=5)
