from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import save_single_map


CLIMATE_FILES = [
    "climate_overview.png",
    "mean_temperature.png",
    "annual_temperature_amplitude.png",
    "mean_humidity.png",
    "prevailing_wind_speed.png",
    "mean_precipitation.png",
    "mean_cloud.png",
    "mean_irradiance.png",
]


def save_climate_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wind_speed = np.hypot(static_maps["prevailing_wind_u"], static_maps["prevailing_wind_v"])
    save_single_map(static_maps["mean_temperature"], output_dir / "mean_temperature.png", "Mean temperature (C)", "coolwarm")
    save_single_map(
        static_maps["annual_temperature_amplitude"],
        output_dir / "annual_temperature_amplitude.png",
        "Annual temperature amplitude (C)",
        "YlOrBr",
    )
    save_single_map(static_maps["mean_humidity"], output_dir / "mean_humidity.png", "Mean humidity", "YlGnBu", vmin=0.0, vmax=1.0)
    _save_wind_map(
        wind_speed,
        static_maps["prevailing_wind_u"],
        static_maps["prevailing_wind_v"],
        static_maps["elevation"],
        output_dir / "prevailing_wind_speed.png",
    )
    save_single_map(
        static_maps["mean_precipitation"],
        output_dir / "mean_precipitation.png",
        "Mean precipitation (mm/year)",
        "Blues",
    )
    save_single_map(static_maps["mean_cloud"], output_dir / "mean_cloud.png", "Mean cloud", "Greys", vmin=0.0, vmax=1.0)
    save_single_map(
        static_maps["mean_irradiance"],
        output_dir / "mean_irradiance.png",
        "Mean irradiance (W/m2)",
        "YlOrRd",
    )
    _save_climate_overview(static_maps, output_dir / "climate_overview.png")
    return CLIMATE_FILES


def _save_climate_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    wind_speed = np.hypot(static_maps["prevailing_wind_u"], static_maps["prevailing_wind_v"])
    panels = [
        (axes[0, 0], static_maps["mean_temperature"], "Mean temperature (C)", "coolwarm", None, None),
        (axes[0, 1], static_maps["annual_temperature_amplitude"], "Annual amplitude (C)", "YlOrBr", None, None),
        (axes[0, 2], static_maps["mean_humidity"], "Mean humidity", "YlGnBu", 0.0, 1.0),
        (axes[1, 0], wind_speed, "Prevailing wind speed (m/s)", "viridis", None, None),
        (axes[1, 1], static_maps["mean_precipitation"], "Mean precipitation (mm/year)", "Blues", None, None),
        (axes[1, 2], static_maps["mean_irradiance"], "Mean irradiance (W/m2)", "YlOrRd", None, None),
    ]
    for ax, values, title, cmap, vmin, vmax in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _draw_wind_quiver(
        axes[1, 0],
        static_maps["prevailing_wind_u"],
        static_maps["prevailing_wind_v"],
    )
    _draw_elevation_contours(axes[1, 0], static_maps["elevation"])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_wind_map(
    wind_speed: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    elevation: np.ndarray,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(wind_speed, cmap="viridis", origin="upper")
    ax.set_title("Prevailing wind speed (m/s)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _draw_wind_quiver(ax, wind_u, wind_v)
    _draw_elevation_contours(ax, elevation)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_wind_quiver(ax: object, wind_u: np.ndarray, wind_v: np.ndarray) -> None:
    block_cells = 8
    vectors = _block_wind_vectors(wind_u, wind_v, block_cells)
    height, width = wind_u.shape
    for x0, y0, dx, dy, line_width in vectors:
        dx, dy = _fit_arrow_to_axes(x0, y0, dx, dy, width, height)
        ax.annotate(
            "",
            xy=(x0 + dx, y0 - dy),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "white",
                "alpha": 0.82,
                "linewidth": line_width,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
                "mutation_scale": 7.0 + 1.8 * line_width,
            },
        )


def _fit_arrow_to_axes(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    margin = 0.6
    end_x = x0 + dx
    end_y = y0 - dy
    scale = 1.0
    if end_x < margin and dx < 0.0:
        scale = min(scale, (x0 - margin) / -dx)
    if end_x > width - 1 - margin and dx > 0.0:
        scale = min(scale, (width - 1 - margin - x0) / dx)
    if end_y < margin and dy > 0.0:
        scale = min(scale, (y0 - margin) / dy)
    if end_y > height - 1 - margin and dy < 0.0:
        scale = min(scale, (height - 1 - margin - y0) / -dy)
    scale = max(float(scale), 0.15)
    return dx * scale, dy * scale


def _block_wind_vectors(wind_u: np.ndarray, wind_v: np.ndarray, block_cells: int) -> list[tuple[float, float, float, float, float]]:
    speed = np.hypot(wind_u, wind_v)
    min_speed = float(np.nanmin(speed))
    max_speed = max(float(np.nanmax(speed)), 1e-6)
    speed_span = max(max_speed - min_speed, 1e-6)

    centers = []
    local_stds = []
    height, width = wind_u.shape
    for row0 in range(0, height, block_cells):
        for col0 in range(0, width, block_cells):
            x0 = col0 + (min(block_cells, width - col0) - 1) / 2.0
            y0 = row0 + (min(block_cells, height - row0) - 1) / 2.0
            u_value, v_value, speed_values = _sample_four_point_wind(wind_u, wind_v, x0, y0)
            centers.append((x0, y0, u_value, v_value, speed_values))
            local_stds.append(float(np.nanstd(speed_values)))
    max_std = max(max(local_stds, default=0.0), 1e-6)

    vectors = []
    for x0, y0, u_value, v_value, speed_values in centers:
        vector_speed = float(np.hypot(u_value, v_value))
        if vector_speed < 1e-6:
            continue

        local_speed_mean = float(np.nanmean(speed_values))
        local_speed_std = float(np.nanstd(speed_values))
        direction_spread = 1.0 - vector_speed / max(local_speed_mean, 1e-6)
        dynamic_amount = np.clip(0.55 * local_speed_std / max_std + 0.45 * direction_spread, 0.0, 1.0)
        speed_level = np.clip((local_speed_mean - min_speed) / speed_span, 0.0, 1.0)
        length = block_cells * (0.26 + 0.56 * speed_level)
        line_width = 0.75 + 1.65 * dynamic_amount
        vectors.append((x0, y0, length * u_value / vector_speed, length * v_value / vector_speed, line_width))
    return vectors


def _sample_four_point_wind(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float, np.ndarray]:
    row0 = int(np.floor(y))
    col0 = int(np.floor(x))
    row1 = min(row0 + 1, wind_u.shape[0] - 1)
    col1 = min(col0 + 1, wind_u.shape[1] - 1)
    row0 = max(row0, 0)
    col0 = max(col0, 0)
    ty = np.clip(y - row0, 0.0, 1.0)
    tx = np.clip(x - col0, 0.0, 1.0)

    weights = np.asarray(
        [
            (1.0 - ty) * (1.0 - tx),
            (1.0 - ty) * tx,
            ty * (1.0 - tx),
            ty * tx,
        ],
        dtype=np.float32,
    )
    u_values = np.asarray(
        [wind_u[row0, col0], wind_u[row0, col1], wind_u[row1, col0], wind_u[row1, col1]],
        dtype=np.float32,
    )
    v_values = np.asarray(
        [wind_v[row0, col0], wind_v[row0, col1], wind_v[row1, col0], wind_v[row1, col1]],
        dtype=np.float32,
    )
    speed_values = np.hypot(u_values, v_values)
    return float(np.dot(weights, u_values)), float(np.dot(weights, v_values)), speed_values


def _draw_elevation_contours(ax: object, elevation: np.ndarray) -> None:
    levels = np.quantile(elevation, [0.25, 0.45, 0.65, 0.82])
    ax.contour(
        elevation,
        levels=levels,
        colors="black",
        linewidths=0.45,
        alpha=0.28,
        origin="upper",
    )
