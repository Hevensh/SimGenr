from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import city_centers_from_id_map, draw_city_centers, save_single_map


CITY_FILES = [
    "city_overview.png",
    "city_suitability.png",
    "urban_core_suitability.png",
    "waterfront_amenity.png",
    "population_density.png",
    "economic_activity.png",
    "urban_density.png",
]


def save_city_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_map(static_maps["city_suitability"], output_dir / "city_suitability.png", "City suitability", "YlGn")
    save_single_map(
        static_maps["urban_core_suitability"],
        output_dir / "urban_core_suitability.png",
        "Urban core suitability",
        "YlGn",
    )
    save_single_map(
        static_maps["waterfront_amenity"],
        output_dir / "waterfront_amenity.png",
        "Waterfront amenity",
        "PuBuGn",
    )
    save_single_map(
        static_maps["population_density"],
        output_dir / "population_density.png",
        "Population density",
        "inferno",
    )
    save_single_map(
        static_maps["economic_activity"],
        output_dir / "economic_activity.png",
        "Economic activity",
        "magma",
    )
    save_single_map(static_maps["urban_density"], output_dir / "urban_density.png", "Urban density", "Reds")
    _save_city_overview(static_maps, output_dir / "city_overview.png")
    return CITY_FILES


def _save_city_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], static_maps["city_suitability"], "City suitability", "YlGn"),
        (axes[0, 1], static_maps["urban_core_suitability"], "Urban core suitability", "YlGn"),
        (axes[0, 2], static_maps["waterfront_amenity"], "Waterfront amenity", "PuBuGn"),
        (axes[1, 0], static_maps["population_density"], "Population density", "inferno"),
        (axes[1, 1], static_maps["economic_activity"], "Economic activity", "magma"),
        (axes[1, 2], static_maps["urban_density"], "Urban density", "Reds"),
    ]
    centers = city_centers_from_id_map(static_maps["city_id_map"], static_maps["urban_density"])
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        draw_city_centers(ax, centers)
    fig.savefig(path, dpi=180)
    plt.close(fig)
