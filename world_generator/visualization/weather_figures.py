from __future__ import annotations

from pathlib import Path

from world_generator.core.datatypes import WeatherStore


WEATHER_FILES = [
    "daily_weather_overview.png",
    "seasonal_temperature.png",
    "seasonal_precipitation.png",
    "seasonal_irradiance.png",
    "annual_weather_summary.png",
]


def save_weather_figures(weather: WeatherStore, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_daily_weather_overview(weather, output_dir / "daily_weather_overview.png")
    _save_seasonal_weather_field(weather, "temperature", output_dir / "seasonal_temperature.png", "Temperature (C)", "coolwarm")
    _save_seasonal_weather_field(
        weather,
        "precipitation",
        output_dir / "seasonal_precipitation.png",
        "Precipitation (mm/day)",
        "Blues",
    )
    _save_seasonal_weather_field(
        weather,
        "irradiance",
        output_dir / "seasonal_irradiance.png",
        "Irradiance (W/m2)",
        "YlOrRd",
    )
    _save_annual_weather_summary(weather, output_dir / "annual_weather_summary.png")
    return WEATHER_FILES


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


def _weather_channel_index(weather: WeatherStore) -> dict[str, int]:
    return {name: index for index, name in enumerate(weather.channel_names)}


def _seasonal_day_indices(days: int) -> list[int]:
    anchors = [80, 172, 264, 355]
    return [min(max(day, 0), days - 1) for day in anchors]
