from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import draw_land_cover, save_land_cover, save_single_map


STATIC_LAND_FILES = [
    "static_land_overview.png",
    "land_cover.png",
    "vegetation.png",
    "protected.png",
    "buildability.png",
    "terrain_cost.png",
]


def save_static_land_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_land_cover(static_maps["land_cover"], output_dir / "land_cover.png")
    save_single_map(static_maps["vegetation"], output_dir / "vegetation.png", "Vegetation", "Greens")
    save_single_map(static_maps["protected"], output_dir / "protected.png", "Protected area", "Purples")
    save_single_map(static_maps["buildability"], output_dir / "buildability.png", "Buildability", "YlGn")
    save_single_map(static_maps["terrain_cost"], output_dir / "terrain_cost.png", "Terrain construction cost", "inferno")
    _save_static_land_overview(static_maps, output_dir / "static_land_overview.png")
    return STATIC_LAND_FILES


def _save_static_land_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    draw_land_cover(axes[0, 0], static_maps["land_cover"], fig)

    panels = [
        (axes[0, 1], static_maps["vegetation"], "Vegetation", "Greens"),
        (axes[1, 0], static_maps["buildability"], "Buildability", "YlGn"),
        (axes[1, 1], static_maps["terrain_cost"], "Terrain construction cost", "inferno"),
    ]
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)
