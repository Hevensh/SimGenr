from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.core.datatypes import SourceLoadForecastStore, WeatherStore


OPERATION_FILES = [
    "hourly_weather_week_overview.png",
    "source_load_timeseries.png",
    "bus_forecast_heatmap.png",
]


def save_operation_figures(hourly_weather: WeatherStore, forecast: SourceLoadForecastStore, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_hourly_weather_week_overview(hourly_weather, output_dir / "hourly_weather_week_overview.png")
    _save_source_load_timeseries(forecast, output_dir / "source_load_timeseries.png")
    _save_bus_forecast_heatmap(forecast, output_dir / "bus_forecast_heatmap.png")
    return OPERATION_FILES


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


def _save_source_load_timeseries(forecast: SourceLoadForecastStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(forecast.p_load_mw.shape[0])
    kinds = np.asarray(forecast.bus_kinds)
    load = forecast.p_load_mw.sum(axis=1)
    wind = forecast.p_gen_available_mw[:, kinds == "wind_bus"].sum(axis=1)
    solar = forecast.p_gen_available_mw[:, kinds == "pv_bus"].sum(axis=1)
    thermal_available = forecast.p_gen_available_mw[:, kinds == "thermal_bus"].sum(axis=1)
    thermal_scheduled = forecast.p_gen_scheduled_mw[:, kinds == "thermal_bus"].sum(axis=1)
    renewable_available = wind + solar

    fig, ax = plt.subplots(figsize=(15, 4.8), constrained_layout=True)
    ax.plot(t, load, color="#ff4fa3", linewidth=1.8, label="load")
    ax.plot(
        t,
        wind,
        color="#38d0ff",
        linewidth=1.1,
        linestyle="--",
        marker="o",
        markersize=2.0,
        markevery=6,
        alpha=0.9,
        label="wind available",
    )
    ax.plot(
        t,
        solar,
        color="#ffcf33",
        linewidth=1.1,
        linestyle="--",
        marker="s",
        markersize=2.0,
        markevery=6,
        alpha=0.9,
        label="solar available",
    )
    ax.plot(
        t,
        renewable_available,
        color="#1aa6b7",
        linewidth=1.35,
        label="renewable available",
    )
    ax.plot(t, thermal_scheduled, color="#f25f2c", linewidth=1.35, label="thermal scheduled")
    ax.plot(t, thermal_available, color="#f25f2c", linewidth=1.0, linestyle="--", alpha=0.55, label="thermal capacity")
    ax.set_title("Hourly source-load forecast")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_bus_forecast_heatmap(forecast: SourceLoadForecastStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    net_injection = forecast.p_gen_scheduled_mw - forecast.p_load_mw
    order = np.argsort(np.asarray(forecast.bus_kinds))
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(net_injection), 98))
    image = ax.imshow(net_injection[:, order].T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_title("Bus net injection forecast")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Bus, grouped by kind")
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="MW")
    fig.savefig(path, dpi=180)
    plt.close(fig)
