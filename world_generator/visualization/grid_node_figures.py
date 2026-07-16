from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import draw_built_environment_texture, draw_elevation_with_water_overlay, save_single_map


GRID_NODE_FILES = [
    "grid_node_overview.png",
    "bus_site_overview.png",
    "thermal_suitability.png",
    "thermal_externality.png",
    "load_bus_overlay.png",
    "source_bus_overlay.png",
]


def save_grid_node_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_map(static_maps["thermal_suitability"], output_dir / "thermal_suitability.png", "Thermal bus suitability", "YlOrBr", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["thermal_externality"], output_dir / "thermal_externality.png", "Thermal externality score", "Reds", vmin=0.0, vmax=1.0)
    _save_bus_overlay(static_maps, output_dir / "bus_site_overview.png", "Grid bus site overview", {"load", "source", "thermal"})
    _save_bus_overlay(static_maps, output_dir / "load_bus_overlay.png", "Load buses over terrain/hydrology", {"load"})
    _save_bus_overlay(static_maps, output_dir / "source_bus_overlay.png", "Source buses over terrain/hydrology", {"source", "thermal"})
    _save_grid_node_overview(static_maps, output_dir / "grid_node_overview.png")
    return GRID_NODE_FILES


def _save_grid_node_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], static_maps["thermal_suitability"], "Thermal bus suitability", "YlOrBr"),
        (axes[0, 1], static_maps["thermal_externality"], "Thermal externality score", "Reds"),
    ]
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        _draw_bus_markers(ax, static_maps, {"load", "source", "thermal"})

    ax = axes[1, 0]
    image = ax.imshow(static_maps["load_node_density"], cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_title("Load density with load buses")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _draw_bus_markers(ax, static_maps, {"load"})

    ax = axes[1, 1]
    image = draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps)
    ax.set_title("Grid buses over terrain/hydrology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _draw_bus_markers(ax, static_maps, {"load", "source", "thermal"})
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_bus_overlay(static_maps: dict[str, np.ndarray], path: Path, title: str, groups: set[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps)
    _draw_bus_markers(ax, static_maps, groups)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_bus_markers(ax: object, static_maps: dict[str, np.ndarray], groups: set[str]) -> None:
    if "load" in groups:
        _scatter_mask(ax, static_maps["load_bus_map"] >= 0, "o", "#ff4fa3", "load bus", 52)
    if "source" in groups:
        source_mask = static_maps["source_bus_map"] >= 0
        wind_mask = source_mask & (static_maps["wind_candidate_map"] >= 0)
        pv_mask = source_mask & (static_maps["pv_candidate_map"] >= 0)
        _scatter_mask(ax, wind_mask, "^", "#38d0ff", "wind bus", 58)
        _scatter_mask(ax, pv_mask, "s", "#ffcf33", "solar bus", 52)
    if "thermal" in groups:
        _scatter_mask(ax, static_maps["thermal_bus_map"] >= 0, "D", "#f25f2c", "thermal bus", 54)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.78)


def _scatter_mask(ax: object, mask: np.ndarray, marker: str, color: str, label: str, size: int) -> None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return
    ax.scatter(cols, rows, s=size, c=color, edgecolors="black", linewidths=0.55, marker=marker, label=label)
