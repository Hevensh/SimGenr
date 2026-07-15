from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import add_deduped_legend, draw_elevation_with_water_overlay, draw_grid_edge_lines, save_single_map


GRID_TOPOLOGY_FILES = [
    "grid_topology_overview.png",
    "routing_cost.png",
    "line_route_map.png",
    "line_routes_overlay.png",
]


def save_grid_topology_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_map(static_maps["routing_cost"], output_dir / "routing_cost.png", "Line routing cost", "inferno", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["line_route_map"], output_dir / "line_route_map.png", "Line route density", "magma", vmin=0.0, vmax=1.0)
    _save_line_routes_overlay(static_maps, output_dir / "line_routes_overlay.png")
    _save_grid_topology_overview(static_maps, output_dir / "grid_topology_overview.png")
    return GRID_TOPOLOGY_FILES


def _save_grid_topology_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], static_maps["routing_cost"], "Line routing cost", "inferno"),
        (axes[0, 1], static_maps["line_route_map"], "Line route density", "magma"),
    ]
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        _draw_bus_markers(ax, static_maps)

    ax = axes[1, 0]
    image = draw_elevation_with_water_overlay(ax, static_maps)
    _draw_line_routes(ax, static_maps)
    _draw_bus_markers(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("Grid topology over terrain/hydrology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    image = ax.imshow(static_maps["load_node_density"], cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
    _draw_line_routes(ax, static_maps)
    _draw_bus_markers(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("Grid topology over load density")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_line_routes_overlay(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = draw_elevation_with_water_overlay(ax, static_maps)
    _draw_line_routes(ax, static_maps)
    _draw_bus_markers(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("Initial line routes over terrain/hydrology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_line_routes(ax: object, static_maps: dict[str, np.ndarray]) -> None:
    if "grid_buses" in static_maps and "grid_edges" in static_maps:
        draw_grid_edge_lines(ax, static_maps["grid_buses"], static_maps["grid_edges"])
        return
    route = static_maps["line_route_map"]
    overlay = np.zeros((*route.shape, 4), dtype=np.float32)
    mask = route > 0.0
    overlay[mask] = [0.05, 0.05, 0.05, 0.65]
    heavy = route > 0.5
    overlay[heavy] = [0.0, 0.0, 0.0, 0.82]
    ax.imshow(overlay, origin="upper")


def _draw_bus_markers(ax: object, static_maps: dict[str, np.ndarray]) -> None:
    _scatter_mask(ax, static_maps["load_bus_map"] >= 0, "o", "#ff4fa3", "load bus", 38)
    source_mask = static_maps["source_bus_map"] >= 0
    _scatter_mask(ax, source_mask & (static_maps["wind_candidate_map"] >= 0), "^", "#38d0ff", "wind bus", 44)
    _scatter_mask(ax, source_mask & (static_maps["pv_candidate_map"] >= 0), "s", "#ffcf33", "solar bus", 40)
    _scatter_mask(ax, static_maps["thermal_bus_map"] >= 0, "D", "#f25f2c", "thermal bus", 42)


def _scatter_mask(ax: object, mask: np.ndarray, marker: str, color: str, label: str, size: int) -> None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return
    ax.scatter(cols, rows, s=size, c=color, edgecolors="black", linewidths=0.5, marker=marker, label=label, zorder=5)
