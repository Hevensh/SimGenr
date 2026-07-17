from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.core.datatypes import WeatherStore
from world_generator.visualization.common import (
    draw_built_environment_texture,
    draw_elevation_with_water_overlay,
    format_hour_timestamp,
)


WEATHER_FILES = [
    "daily/daily_weather_overview.png",
    "daily/seasonal_temperature.png",
    "daily/seasonal_precipitation.png",
    "daily/seasonal_irradiance.png",
    "daily/annual_weather_summary.png",
]

HOURLY_WEATHER_FILES = [
    "hourly/hourly_weather_week_overview.png",
    "hourly/hourly_weather_fields.gif",
]


def save_weather_figures(
    weather: WeatherStore,
    output_dir: Path,
    hourly_weather: WeatherStore | None = None,
    static_maps: dict[str, np.ndarray] | None = None,
    *,
    render_hourly_gif: bool = True,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    _save_daily_weather_overview(weather, daily_dir / "daily_weather_overview.png")
    _save_seasonal_weather_field(weather, "temperature", daily_dir / "seasonal_temperature.png", "Temperature (C)", "coolwarm")
    _save_seasonal_weather_field(
        weather,
        "precipitation",
        daily_dir / "seasonal_precipitation.png",
        "Precipitation (mm/day)",
        "Blues",
    )
    _save_seasonal_weather_field(
        weather,
        "irradiance",
        daily_dir / "seasonal_irradiance.png",
        "Irradiance (W/m2)",
        "YlOrRd",
    )
    _save_annual_weather_summary(weather, daily_dir / "annual_weather_summary.png")
    files = list(WEATHER_FILES)
    if hourly_weather is not None:
        hourly_dir = output_dir / "hourly"
        hourly_dir.mkdir(parents=True, exist_ok=True)
        _save_hourly_weather_week_overview(hourly_weather, hourly_dir / "hourly_weather_week_overview.png")
        files.append("hourly/hourly_weather_week_overview.png")
        if static_maps is not None and render_hourly_gif:
            _save_hourly_weather_gif(static_maps, hourly_weather, hourly_dir / "hourly_weather_fields.gif")
            files.append("hourly/hourly_weather_fields.gif")
    return files


def _save_daily_weather_overview(weather: WeatherStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    channels = _weather_channel_index(weather)
    day_indices = _seasonal_day_indices(weather.dynamic.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(14, 13), constrained_layout=True)
    specs = [
        ("temperature", "Temperature (C)", "coolwarm"),
        ("cloud", "Cloud", "Greys"),
        ("precipitation", "Precipitation (mm/day)", "Blues"),
        ("irradiance", "Irradiance (W/m2)", "YlOrRd"),
    ]
    row_titles = ["Spring", "Summer", "Autumn", "Winter"]
    for row, day in enumerate(day_indices):
        for col, (channel, title, cmap) in enumerate(specs):
            ax = axes[row, col]
            values = weather.dynamic[day, channels[channel]]
            image = ax.imshow(values, cmap=cmap, origin="upper")
            ax.set_title(f"{row_titles[row]} day {int(weather.timestamps[day])}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_seasonal_weather_field(
    weather: WeatherStore,
    channel: str,
    path: Path,
    title: str,
    cmap: str,
) -> None:
    import matplotlib.pyplot as plt

    channels = _weather_channel_index(weather)
    day_indices = _seasonal_day_indices(weather.dynamic.shape[0])
    labels = ["Spring", "Summer", "Autumn", "Winter"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, day, label in zip(axes, day_indices, labels):
        image = ax.imshow(weather.dynamic[day, channels[channel]], cmap=cmap, origin="upper")
        ax.set_title(f"{label} day {int(weather.timestamps[day])}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_annual_weather_summary(weather: WeatherStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    channels = _weather_channel_index(weather)
    t = weather.timestamps
    series = {
        "Temperature (C)": weather.dynamic[:, channels["temperature"]].mean(axis=(1, 2)),
        "Humidity": weather.dynamic[:, channels["humidity"]].mean(axis=(1, 2)),
        "Cloud": weather.dynamic[:, channels["cloud"]].mean(axis=(1, 2)),
        "Precipitation (mm/day)": weather.dynamic[:, channels["precipitation"]].mean(axis=(1, 2)),
        "Irradiance (W/m2)": weather.dynamic[:, channels["irradiance"]].mean(axis=(1, 2)),
        "Wind speed (m/s)": weather.dynamic[:, channels["wind_speed"]].mean(axis=(1, 2)),
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (label, values) in zip(axes.flat, series.items()):
        ax.plot(t, values, linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_hourly_weather_week_overview(weather: WeatherStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    channels = {name: index for index, name in enumerate(weather.channel_names)}
    t = np.arange(weather.dynamic.shape[0])
    series = {
        "Temperature (C)": weather.dynamic[:, channels["temperature"]].mean(axis=(1, 2)),
        "Wind speed (m/s)": weather.dynamic[:, channels["wind_speed"]].mean(axis=(1, 2)),
        "Cloud": weather.dynamic[:, channels["cloud"]].mean(axis=(1, 2)),
        "Irradiance (W/m2)": weather.dynamic[:, channels["irradiance"]].mean(axis=(1, 2)),
        "Precipitation (mm/h)": weather.dynamic[:, channels["precipitation"]].mean(axis=(1, 2)),
        "Humidity": weather.dynamic[:, channels["humidity"]].mean(axis=(1, 2)),
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (label, values) in zip(axes.flat, series.items()):
        ax.plot(t, values, linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("Hour")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Hourly weather week, start day {weather.start_day_of_year}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_hourly_weather_gif(static_maps: dict[str, np.ndarray], weather: WeatherStore, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    channels = {name: index for index, name in enumerate(weather.channel_names)}
    specs = [
        ("temperature", "Temperature (C)", "coolwarm", "field"),
        ("cloud", "Cloud cover", "Greys_r", "cloud_overlay"),
        ("precipitation", "Precipitation (mm/h)", "PuBuGn", "field"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.1), constrained_layout=True)
    images = []
    for ax, (channel_name, title, cmap, mode) in zip(axes, specs):
        values = weather.dynamic[:, channels[channel_name]]
        if channel_name == "precipitation":
            vmin = 0.0
            vmax = float(np.nanpercentile(values, 99.7))
        else:
            vmin = float(np.nanpercentile(values, 2))
            vmax = float(np.nanpercentile(values, 98))
        if vmax <= vmin + 1e-6:
            vmax = vmin + 1.0
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if mode == "cloud_overlay":
            draw_elevation_with_water_overlay(ax, static_maps)
            draw_built_environment_texture(ax, static_maps, scale=3)
            image = ax.imshow(_cloud_rgba(values[0]), origin="upper", interpolation="nearest", zorder=3)
            colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap), ax=ax, shrink=0.76)
            colorbar.set_label("cloud")
            images.append((image, values, mode))
        else:
            image = ax.imshow(values[0], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
            fig.colorbar(image, ax=ax, shrink=0.76)
            images.append((image, values, mode))
    suptitle = fig.suptitle(format_hour_timestamp(int(weather.timestamps[0])))

    def update(frame: int) -> tuple[object, ...]:
        for image, values, mode in images:
            image.set_data(_cloud_rgba(values[frame]) if mode == "cloud_overlay" else values[frame])
        suptitle.set_text(format_hour_timestamp(int(weather.timestamps[frame])))
        return tuple(image for image, _, _ in images) + (suptitle,)

    animation = FuncAnimation(fig, update, frames=weather.dynamic.shape[0], interval=200, blit=False)
    animation.save(path, writer=PillowWriter(fps=5), dpi=120)
    plt.close(fig)


def _cloud_rgba(values: np.ndarray) -> np.ndarray:
    cloud = np.clip(values.astype(np.float32), 0.0, 1.0)
    visible_cloud = np.clip((cloud - 0.28) / 0.58, 0.0, 1.0)
    rgba = np.ones((*cloud.shape, 4), dtype=np.float32)
    rgba[..., :3] = 1.0
    rgba[..., 3] = np.clip(0.90 * visible_cloud**1.12, 0.0, 0.90)
    return rgba


def _weather_channel_index(weather: WeatherStore) -> dict[str, int]:
    return {name: index for index, name in enumerate(weather.channel_names)}


def _seasonal_day_indices(days: int) -> list[int]:
    anchors = [80, 172, 264, 355]
    return [min(max(day, 0), days - 1) for day in anchors]
