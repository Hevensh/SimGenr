from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import draw_elevation_with_water_overlay, save_elevation_with_water_overlay, save_single_map


HYDROLOGY_FILES = [
    "hydrology_overview.png",
    "hydrology_elevation.png",
    "flow_accumulation.png",
    "distance_to_water.png",
    "flood_risk.png",
]


def save_hydrology_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_elevation_with_water_overlay(
        static_maps,
        output_dir / "hydrology_elevation.png",
        "Hydrology elevation with rivers/lakes",
    )
    save_single_map(
        np.log1p(static_maps["flow_accumulation"]),
        output_dir / "flow_accumulation.png",
        "Log flow accumulation",
        "Blues",
    )
    save_single_map(
        static_maps["distance_to_water"],
        output_dir / "distance_to_water.png",
        "Distance to water (km)",
        "cividis",
    )
    save_single_map(static_maps["flood_risk"], output_dir / "flood_risk.png", "Flood risk", "YlOrRd")
    if "river" in static_maps and "lake" in static_maps:
        _save_hydrology_overview(static_maps, output_dir / "hydrology_overview.png")
    return HYDROLOGY_FILES


def _save_hydrology_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    ax = axes[0, 0]
    image = draw_elevation_with_water_overlay(ax, static_maps)
    ax.set_title("Elevation with rivers/lakes")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    panels = [
        (axes[0, 1], np.log1p(static_maps["flow_accumulation"]), "Log flow accumulation", "Blues"),
        (axes[1, 0], static_maps["distance_to_water"], "Distance to water (km)", "cividis"),
        (axes[1, 1], static_maps["flood_risk"], "Flood risk", "YlOrRd"),
    ]
    for ax, values, title, cmap in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)
