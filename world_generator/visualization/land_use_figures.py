from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import (
    city_centers_from_id_map,
    draw_city_centers,
    draw_elevation_with_water_overlay,
    save_single_map,
)


LAND_USE_FILES = [
    "land_use_overview.png",
    "land_use_zone.png",
    "residential.png",
    "commercial.png",
    "industrial.png",
    "agriculture.png",
    "park_green.png",
    "load_density_base.png",
    "built_environment_overlay.png",
]

BUILT_ENVIRONMENT_TEXTURE_LEGEND = [
    ("residential", "warm yellow/orange sub-cells; medium density follows residential tendency and base load"),
    ("commercial", "bright pink/magenta sub-cells with pale highlights; densest pattern around high economic activity"),
    ("industrial", "blue/purple sub-cells with cold white accents; larger regular clusters around urban edges"),
    ("agriculture", "sparse warm yellow sub-cells; low-density rural activity outside cities"),
    ("park_green", "green sub-cells with cyan waterfront accents; sparse recreational or green-space lighting"),
]


def save_land_use_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_land_use_zone(static_maps["land_use_zone"], output_dir / "land_use_zone.png")
    save_single_map(static_maps["residential"], output_dir / "residential.png", "Residential tendency", "YlOrBr", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["commercial"], output_dir / "commercial.png", "Commercial tendency", "magma", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["industrial"], output_dir / "industrial.png", "Industrial tendency", "PuBu", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["agriculture"], output_dir / "agriculture.png", "Agriculture tendency", "YlGn", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["park_green"], output_dir / "park_green.png", "Park/green tendency", "Greens", vmin=0.0, vmax=1.0)
    save_single_map(static_maps["load_density_base"], output_dir / "load_density_base.png", "Base load density", "inferno", vmin=0.0, vmax=1.0)
    _save_land_use_overview(static_maps, output_dir / "land_use_overview.png")
    _save_built_environment_overlay(static_maps, output_dir / "built_environment_overlay.png")
    return LAND_USE_FILES


def _save_land_use_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], "land_use_zone", "Land use zones", None),
        (axes[0, 1], "load_density_base", "Base load density", "inferno"),
        (axes[0, 2], "commercial", "Commercial tendency", "magma"),
        (axes[1, 0], "residential", "Residential tendency", "YlOrBr"),
        (axes[1, 1], "industrial", "Industrial tendency", "PuBu"),
        (axes[1, 2], "park_green", "Park/green tendency", "Greens"),
    ]
    for ax, name, title, cmap in panels:
        if name == "land_use_zone":
            _draw_land_use_zone(ax, static_maps[name], fig)
        else:
            image = ax.imshow(static_maps[name], cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_land_use_zone(values: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _draw_land_use_zone(ax, values, fig)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_land_use_zone(ax: object, values: np.ndarray, fig: object) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = [
        "#202020",  # background
        "#ffd35a",  # residential
        "#ff4fa3",  # commercial
        "#7864ff",  # industrial
        "#9dd866",  # agriculture
        "#2ee68a",  # park/green
        "#8a62b8",  # protected
        "#008fbd",  # water
    ]
    labels = [
        "background",
        "residential",
        "commercial",
        "industrial",
        "agriculture",
        "park/green",
        "protected",
        "water",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1.0), cmap.N)
    image = ax.imshow(values, cmap=cmap, norm=norm, origin="upper")
    ax.set_title("Land use zones")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(8))
    cbar.ax.set_yticklabels(labels)


def _save_built_environment_overlay(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = draw_elevation_with_water_overlay(ax, static_maps)
    building_overlay = _building_texture_overlay(
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
        building_overlay,
        origin="upper",
        extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        interpolation="nearest",
    )
    centers = city_centers_from_id_map(static_maps["city_id_map"], static_maps["urban_density"])
    draw_city_centers(ax, centers)
    ax.set_title("Built environment and load zones")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _building_texture_overlay(
    land_use_zone: np.ndarray,
    load_density: np.ndarray,
    economic_activity: np.ndarray,
    residential: np.ndarray,
    commercial: np.ndarray,
    industrial: np.ndarray,
    agriculture: np.ndarray,
    park_green: np.ndarray,
    scale: int,
) -> np.ndarray:
    height, width = land_use_zone.shape
    overlay = np.zeros((height * scale, width * scale, 4), dtype=np.float32)
    pattern_order = [
        (1, 1),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    ]
    zone_specs = {
        1: (np.asarray([1.00, 0.76, 0.22], dtype=np.float32), 7.0, 0.28),  # residential
        2: (np.asarray([1.00, 0.34, 0.72], dtype=np.float32), 9.0, 0.42),  # commercial
        3: (np.asarray([0.48, 0.42, 1.00], dtype=np.float32), 6.0, 0.36),  # industrial
        4: (np.asarray([1.00, 0.86, 0.40], dtype=np.float32), 2.5, 0.18),  # agriculture
        5: (np.asarray([0.24, 1.00, 0.56], dtype=np.float32), 3.0, 0.20),  # park/green
    }
    commercial_accent = np.asarray([1.00, 0.92, 0.72], dtype=np.float32)
    waterfront_accent = np.asarray([0.20, 0.88, 1.00], dtype=np.float32)

    for row in range(height):
        for col in range(width):
            zone = int(land_use_zone[row, col])
            if zone not in zone_specs:
                continue
            tendency = {
                1: residential,
                2: commercial,
                3: industrial,
                4: agriculture,
                5: park_green,
            }[zone]
            strength = float(np.clip(tendency[row, col], 0.0, 1.0))
            load = float(np.clip(load_density[row, col], 0.0, 1.0))
            if strength <= 0.08 and load <= 0.08:
                continue
            economy = float(economic_activity[row, col])
            base_color, density_scale, alpha_base = zone_specs[zone]
            block_count = int(np.clip(np.ceil((0.55 * strength + 0.45 * load) * density_scale), 1, 9))
            if zone in (4, 5):
                block_count = min(block_count, 4)
            color = base_color.copy()
            if zone == 2:
                color = (1.0 - 0.35 * economy) * color + (0.35 * economy) * commercial_accent
            elif zone == 5:
                color = (1.0 - 0.28 * economy) * color + (0.28 * economy) * waterfront_accent
            alpha = np.clip(alpha_base + 0.42 * load + 0.20 * strength + 0.12 * economy, 0.0, 0.94)
            offset = (row * 17 + col * 31 + zone * 13) % len(pattern_order)
            ordered = pattern_order[offset:] + pattern_order[:offset]
            for index, (sub_r, sub_c) in enumerate(ordered[:block_count]):
                rr = row * scale + sub_r
                cc = col * scale + sub_c
                accent = (row * 11 + col * 7 + index * 5 + zone) % 9
                if accent == 0 and zone in (2, 5):
                    pixel_color = waterfront_accent if zone == 5 else commercial_accent
                elif accent == 1 and zone == 3:
                    pixel_color = np.asarray([0.78, 0.86, 1.00], dtype=np.float32)
                else:
                    pixel_color = color
                overlay[rr, cc, :3] = pixel_color
                overlay[rr, cc, 3] = alpha
    return overlay
