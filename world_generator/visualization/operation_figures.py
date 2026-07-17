from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.core.datatypes import SourceLoadForecastStore
from world_generator.visualization.common import format_hour_timestamp


OPERATION_FILES = [
    "source_load_timeseries.png",
    "bus_forecast_heatmap.png",
]


def save_operation_figures(
    forecast: SourceLoadForecastStore,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_source_load_timeseries(forecast, output_dir / "source_load_timeseries.png")
    _save_bus_forecast_heatmap(forecast, output_dir / "bus_forecast_heatmap.png")
    return OPERATION_FILES


def _save_source_load_timeseries(forecast: SourceLoadForecastStore, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter

    t = np.arange(forecast.p_load_mw.shape[0])
    kinds = np.asarray(forecast.bus_kinds)
    load = forecast.p_load_mw.sum(axis=1)
    wind = forecast.p_gen_available_mw[:, kinds == "wind_bus"].sum(axis=1)
    solar = forecast.p_gen_available_mw[:, kinds == "pv_bus"].sum(axis=1)
    thermal_available = forecast.p_gen_available_mw[:, kinds == "thermal_bus"].sum(axis=1)
    thermal_scheduled = forecast.p_gen_scheduled_mw[:, kinds == "thermal_bus"].sum(axis=1)
    renewable_available = wind + solar
    dispatchable_supply = thermal_available + renewable_available
    storage_needed = load > dispatchable_supply

    fig, ax = plt.subplots(figsize=(15, 4.8), constrained_layout=True)
    ax.fill_between(
        t,
        dispatchable_supply,
        load,
        where=storage_needed,
        color="#f7b7d8",
        alpha=0.38,
        interpolate=True,
        label="storage needed",
    )
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
    tick_positions = np.arange(0, forecast.p_load_mw.shape[0], 24)
    tick_labels = [format_hour_timestamp(int(forecast.timestamps[int(position)])).rsplit(" ", 1)[0] for position in tick_positions]
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.set_xlabel("Date")
    ax.set_ylabel("MW")
    ax.legend(loc="upper right", ncol=1)
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
