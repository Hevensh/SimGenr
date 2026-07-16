from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import draw_built_environment_texture, draw_elevation_with_water_overlay, save_single_map


ENERGY_FILES = [
    "energy_candidate_overview.png",
    "wind_suitability.png",
    "pv_suitability.png",
    "load_node_density.png",
    "candidate_sites_overlay.png",
]


def save_energy_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_map(static_maps["wind_suitability"], output_dir / "wind_suitability.png", "Wind suitability", "YlGnBu", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["pv_suitability"], output_dir / "pv_suitability.png", "Solar suitability", "YlOrRd", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["load_node_density"], output_dir / "load_node_density.png", "Load node density", "inferno", vmin=0.0, vmax=1.0)
    _save_candidate_sites_overlay(static_maps, output_dir / "candidate_sites_overlay.png")
    _save_energy_candidate_overview(static_maps, output_dir / "energy_candidate_overview.png")
    return ENERGY_FILES


def _save_energy_candidate_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], static_maps["wind_suitability"], "Wind suitability", "YlGnBu"),
        (axes[0, 1], static_maps["pv_suitability"], "Solar suitability", "YlOrRd"),
        (axes[1, 0], static_maps["load_node_density"], "Load node density", "inferno"),
    ]
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        _draw_candidates(ax, static_maps)

    ax = axes[1, 1]
    image = draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps)
    _draw_candidates(ax, static_maps)
    ax.set_title("Candidate sites over terrain/hydrology")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_candidate_sites_overlay(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps)
    _draw_candidates(ax, static_maps)
    ax.set_title("Energy and load candidate sites")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_candidates(ax: object, static_maps: dict[str, np.ndarray]) -> None:
    load_map = static_maps["load_candidate_map"]
    wind_mask = static_maps["wind_candidate_map"] >= 0
    pv_mask = static_maps["pv_candidate_map"] >= 0
    load_mask = load_map >= 0
    _scatter_mask(ax, wind_mask, "^", "#38d0ff", "wind")
    _scatter_mask(ax, pv_mask, "s", "#ffcf33", "solar")
    _scatter_mask(ax, load_mask, "o", "#ff4fa3", "load")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.78)


def _scatter_mask(ax: object, mask: np.ndarray, marker: str, color: str, label: str) -> None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return
    ax.scatter(cols, rows, s=46, c=color, edgecolors="black", linewidths=0.55, marker=marker, label=label)
