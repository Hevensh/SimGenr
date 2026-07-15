from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import land_terrain_cmap, save_single_map


TERRAIN_FILES = [
    "terrain_overview.png",
    "elevation.png",
    "slope.png",
    "roughness.png",
    "curvature.png",
]


def save_terrain_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_cmap = land_terrain_cmap()
    save_single_map(static_maps["elevation"], output_dir / "elevation.png", "Elevation (m)", terrain_cmap)
    save_single_map(static_maps["slope"], output_dir / "slope.png", "Slope (m/m)", "magma")
    save_single_map(static_maps["roughness"], output_dir / "roughness.png", "Local roughness", "viridis")
    save_single_map(static_maps["curvature"], output_dir / "curvature.png", "Curvature", "coolwarm")
    _save_terrain_overview(static_maps, output_dir / "terrain_overview.png", plt)
    return TERRAIN_FILES


def _save_terrain_overview(static_maps: dict[str, np.ndarray], path: Path, plt: object) -> None:
    terrain_cmap = land_terrain_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = [
        ("elevation", "Elevation (m)", terrain_cmap),
        ("slope", "Slope (m/m)", "magma"),
        ("roughness", "Local roughness", "viridis"),
        ("curvature", "Curvature", "coolwarm"),
    ]
    for ax, (name, title, cmap) in zip(axes.flat, panels):
        image = ax.imshow(static_maps[name], cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)
