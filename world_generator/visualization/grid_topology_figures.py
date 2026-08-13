from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import (
    add_deduped_legend,
    draw_built_environment_texture,
    draw_elevation_with_water_overlay,
    draw_grid_edge_lines,
    save_single_map,
)


GRID_TOPOLOGY_FILES = [
    "grid_topology_overview.png",
    "routing_cost.png",
    "line_route_map.png",
    "line_routes_overlay.png",
]


def save_grid_topology_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_map(static_maps["routing_cost"], output_dir / "routing_cost.png", "Line routing cost", "inferno", vmin=0.0, vmax=1.0)
    route_map = static_maps.get("refined_line_route_map", static_maps["line_route_map"])
    save_single_map(route_map, output_dir / "line_route_map.png", "A* line route grid", "magma", vmin=0.0, vmax=1.0)
    _save_line_routes_overlay(static_maps, output_dir / "line_routes_overlay.png")
    _save_grid_topology_overview(static_maps, output_dir / "grid_topology_overview.png")
    return GRID_TOPOLOGY_FILES


def _save_grid_topology_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    ax = axes[0, 0]
    image = draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps)
    _draw_routes(ax, static_maps)
    _draw_buses(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("A* grid topology over terrain/hydrology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    panels = [
        (axes[0, 1], static_maps["routing_cost"], "Line routing cost", "inferno"),
        (
            axes[1, 0],
            static_maps.get("refined_line_route_map", static_maps["line_route_map"]),
            "A* line route grid",
            "magma",
        ),
        (axes[1, 1], static_maps["load_node_density"], "Grid topology over load density", "inferno"),
    ]
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
        if ax is axes[1, 1]:
            _draw_routes(
                ax,
                static_maps,
                color="#22c7df",
                redundant_color="#22c7df",
                unified_legend_label="transmission line",
            )
        _draw_buses(ax, static_maps)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    add_deduped_legend(axes[1, 1], static_maps)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_line_routes_overlay(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps)
    _draw_routes(ax, static_maps)
    _draw_buses(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_title("A* transmission paths over terrain/hydrology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_routes(
    ax: object,
    static_maps: dict[str, np.ndarray],
    *,
    color: str = "#151515",
    redundant_color: str = "#006f8f",
    unified_legend_label: str | None = None,
) -> None:
    buses = static_maps.get("refined_grid_buses", static_maps.get("grid_buses"))
    edges = static_maps.get("refined_grid_edges", static_maps.get("grid_edges"))
    if buses is None or edges is None:
        return
    edge_paths = static_maps.get("refined_grid_edge_paths", static_maps.get("grid_edge_paths"))
    draw_grid_edge_lines(
        ax,
        buses,
        edges,
        color=color,
        redundant_color=redundant_color,
        linewidth=1.55,
        alpha=0.74,
        edge_paths=edge_paths,
        unified_legend_label=unified_legend_label,
    )


def _draw_buses(ax: object, static_maps: dict[str, np.ndarray]) -> None:
    _scatter_mask(ax, static_maps["load_bus_map"] >= 0, "o", "#ff4fa3", 36, "load bus")
    source_mask = static_maps["source_bus_map"] >= 0
    _scatter_mask(ax, source_mask & (static_maps["wind_candidate_map"] >= 0), "^", "#38d0ff", 42, "wind bus")
    _scatter_mask(ax, source_mask & (static_maps["pv_candidate_map"] >= 0), "s", "#ffcf33", 38, "solar bus")
    _scatter_mask(ax, static_maps["thermal_bus_map"] >= 0, "D", "#f25f2c", 40, "thermal bus")


def _scatter_mask(ax: object, mask: np.ndarray, marker: str, color: str, size: int, label: str) -> None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return
    ax.scatter(cols, rows, s=size, c=color, edgecolors="black", linewidths=0.5, marker=marker, label=label, zorder=5)
