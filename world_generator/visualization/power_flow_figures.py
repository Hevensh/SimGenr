from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.core.datatypes import PowerFlowStore, WeatherStore
from world_generator.visualization.common import (
    add_deduped_legend,
    draw_built_environment_texture,
    draw_elevation_with_water_overlay,
    edge_capacity_multipliers,
    edge_line_xy,
    format_hour_timestamp,
    line_width_for_multiplier,
)
from world_generator.visualization.weather_figures import _cloud_rgba


POWER_FLOW_FILES = [
    "line_loading_peak.png",
    "line_loading_mean.png",
    "grid_operation_timeseries.png",
]


def save_power_flow_figures(
    static_maps: dict[str, np.ndarray],
    power_flow: PowerFlowStore,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_line_metric_map(
        static_maps,
        power_flow,
        np.nanmax(power_flow.line_loading_ratio, axis=0) if power_flow.line_loading_ratio.size else np.asarray([]),
        output_dir / "line_loading_peak.png",
        "Peak line loading",
        "peak loading",
    )
    _save_line_metric_map(
        static_maps,
        power_flow,
        np.nanmean(power_flow.line_loading_ratio, axis=0) if power_flow.line_loading_ratio.size else np.asarray([]),
        output_dir / "line_loading_mean.png",
        "Mean line loading",
        "mean loading",
    )
    _save_grid_operation_timeseries(power_flow, output_dir / "grid_operation_timeseries.png")
    return POWER_FLOW_FILES


def _save_line_metric_map(
    static_maps: dict[str, np.ndarray],
    power_flow: PowerFlowStore,
    metric: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    buses = np.atleast_2d(static_maps.get("refined_grid_buses", static_maps.get("grid_buses", np.empty((0, 0)))))
    edges = np.atleast_2d(static_maps.get("refined_grid_edges", static_maps.get("grid_edges", np.empty((0, 0)))))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps, scale=3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if metric.size and edges.size:
        edge_metric = _metric_by_edge_id(power_flow.branch_ids, metric, edges[:, 0].astype(int))
        capacity_multipliers = edge_capacity_multipliers(static_maps, edges[:, 0].astype(int))
        vmax = float(np.nanpercentile(edge_metric, 96)) if edge_metric.size else 1.0
        vmax = max(vmax, 0.05)
        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = _line_loading_cmap()
        _draw_metric_edges(
            ax,
            buses,
            edges,
            edge_metric,
            capacity_multipliers,
            norm,
            cmap,
            edge_paths=static_maps.get("refined_grid_edge_paths"),
        )
        edge_colors = np.asarray([cmap(norm(float(value))) for value in edge_metric])
        transit_styles = _transit_node_styles(edges, edge_colors, capacity_multipliers)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)
    else:
        transit_styles = {}

    _draw_operation_nodes(ax, buses, static_maps, transit_styles=transit_styles)
    add_deduped_legend(ax, static_maps, fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _metric_by_edge_id(branch_ids: np.ndarray, metric: np.ndarray, edge_ids: np.ndarray) -> np.ndarray:
    values_by_id = {int(edge_id): float(value) for edge_id, value in zip(branch_ids, metric)}
    return np.asarray([values_by_id.get(int(edge_id), np.nan) for edge_id in edge_ids], dtype=np.float32)


def _line_loading_cmap() -> object:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap("magma")
    return LinearSegmentedColormap.from_list(
        "magma_visible_low",
        base(np.linspace(0.28, 1.0, 256)),
    )


def _draw_metric_edges(
    ax: object,
    buses: np.ndarray,
    edges: np.ndarray,
    edge_metric: np.ndarray,
    capacity_multipliers: np.ndarray,
    norm: object,
    cmap: object,
    edge_paths: object | None = None,
) -> None:
    bus_by_id = {int(row[0]): row for row in buses if row.size >= 3}
    if not bus_by_id:
        return
    order = np.argsort(np.nan_to_num(edge_metric, nan=-1.0))
    for edge_index in order:
        edge = edges[edge_index]
        if edge.size < 3 or not np.isfinite(edge_metric[edge_index]):
            continue
        line = edge_line_xy(edge, bus_by_id, edge_paths)
        if line is None:
            continue
        xs, ys = line
        value = float(edge_metric[edge_index])
        color = cmap(norm(value))
        linewidth = line_width_for_multiplier(float(capacity_multipliers[edge_index]))
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth,
            alpha=0.82,
            solid_capstyle="round",
            zorder=3,
        )
    hot_index = int(np.nanargmax(edge_metric)) if np.isfinite(edge_metric).any() else None
    if hot_index is not None:
        edge = edges[hot_index]
        line = edge_line_xy(edge, bus_by_id, edge_paths)
        if line is not None:
            xs, ys = line
            ax.plot(
                xs,
                ys,
                color="#fff4a8",
                linewidth=line_width_for_multiplier(float(capacity_multipliers[hot_index])) + 0.7,
                alpha=0.95,
                zorder=4,
            )
def _transit_node_styles(
    edges: np.ndarray,
    edge_colors: np.ndarray,
    capacity_multipliers: np.ndarray,
) -> dict[int, tuple[tuple[float, float, float, float], float, float]]:
    colors_by_bus: dict[int, list[np.ndarray]] = {}
    multipliers_by_bus: dict[int, list[float]] = {}
    for index, edge in enumerate(edges):
        if edge.size < 3 or index >= len(edge_colors):
            continue
        for bus_id in (int(edge[1]), int(edge[2])):
            colors_by_bus.setdefault(bus_id, []).append(np.asarray(edge_colors[index], dtype=np.float64))
            multipliers_by_bus.setdefault(bus_id, []).append(float(capacity_multipliers[index]))
    styles = {}
    for bus_id, colors in colors_by_bus.items():
        mean_color = np.mean(np.stack(colors), axis=0)
        mean_multiplier = float(np.mean(multipliers_by_bus[bus_id]))
        marker_size = float(np.clip(34.0 * mean_multiplier, 23.9, 69.3))
        marker_edge_width = 0.30 * line_width_for_multiplier(mean_multiplier)
        styles[bus_id] = (tuple(float(value) for value in mean_color), marker_size, marker_edge_width)
    return styles


def _draw_operation_nodes(
    ax: object,
    buses: np.ndarray,
    static_maps: dict[str, np.ndarray],
    *,
    transit_styles: dict[int, tuple[tuple[float, float, float, float], float, float]] | None = None,
) -> dict[str, object]:
    if buses.size == 0:
        return
    show_ids = bool(static_maps.get("show_bus_ids", False))
    kinds = {
        "load": ("load_bus", "#ff4fa3", "o", 37.8),
        "wind": ("wind_bus", "#36d7ff", "^", 46.8),
        "solar": ("pv_bus", "#ffcf33", "s", 43.2),
        "thermal": ("thermal_bus", "#ff7a33", "D", 46.8),
        "transit": ("transit_bus", "#fff4a8", "P", 36.9),
    }
    bus_kind = _bus_kind_lookup(buses, static_maps)
    artists: dict[str, object] = {}
    for label, (kind, color, marker, size) in kinds.items():
        rows = [row for row in buses if bus_kind.get(int(row[0])) == kind]
        if not rows:
            continue
        values = np.asarray(rows)
        colors = color
        sizes: float | list[float] = size
        edge_widths: float | list[float] = 0.585
        if kind == "transit_bus" and transit_styles:
            fallback = ((1.0, 0.96, 0.66, 1.0), size, 0.585)
            colors = [transit_styles.get(int(row[0]), fallback)[0] for row in rows]
            sizes = [transit_styles.get(int(row[0]), fallback)[1] for row in rows]
            edge_widths = [transit_styles.get(int(row[0]), fallback)[2] for row in rows]
        artists[kind] = ax.scatter(
            values[:, 2],
            values[:, 1],
            s=sizes,
            c=colors,
            marker=marker,
            edgecolors="#151515",
            linewidths=edge_widths,
            alpha=0.94,
            label=label,
            zorder=6,
        )
    if show_ids:
        labels = static_maps.get("bus_labels", {})
        for row in buses:
            if row.size < 3:
                continue
            bus_id = int(row[0])
            label_text = labels.get(bus_id, str(bus_id)) if isinstance(labels, dict) else str(bus_id)
            ax.text(
                float(row[2]) + 0.35,
                float(row[1]) - 0.35,
                label_text,
                fontsize=5.6,
                color="#111111",
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "#333333",
                    "linewidth": 0.35,
                    "alpha": 0.78,
                },
                zorder=10,
            )
    return artists


def save_line_loading_gif(
    static_maps: dict[str, np.ndarray],
    power_flow: PowerFlowStore,
    hourly_weather: WeatherStore,
    path: Path,
    *,
    show_weather: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    buses = np.atleast_2d(static_maps.get("refined_grid_buses", np.empty((0, 0))))
    edges = np.atleast_2d(static_maps.get("refined_grid_edges", np.empty((0, 0))))
    if buses.size == 0 or edges.size == 0 or power_flow.line_loading_ratio.size == 0:
        return
    channels = {name: index for index, name in enumerate(hourly_weather.channel_names)}
    cloud = hourly_weather.dynamic[:, channels["cloud"]]
    frame_count = min(power_flow.line_loading_ratio.shape[0], cloud.shape[0])
    edge_ids = edges[:, 0].astype(int)
    capacity_multipliers = edge_capacity_multipliers(static_maps, edge_ids)
    bus_by_id = {int(row[0]): row for row in buses if row.size >= 3}
    edge_paths = static_maps.get("refined_grid_edge_paths")
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    cmap = _line_loading_cmap()

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps, scale=3)
    light_artist = None
    cloud_artist = None
    if show_weather:
        initial_hour = int(hourly_weather.timestamps[0]) % 24
        initial_shade = _twilight_shade_rgba(cloud[0].shape, initial_hour)
        light_artist = ax.imshow(
            initial_shade,
            origin="upper",
            interpolation="nearest",
            zorder=2.8,
        )
        cloud_artist = ax.imshow(
            _tinted_cloud_rgba(cloud[0], initial_shade[0, 0]),
            origin="upper",
            interpolation="nearest",
            zorder=3,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(format_hour_timestamp(int(hourly_weather.timestamps[0])))

    frame_metric = _metric_by_edge_id(power_flow.branch_ids, power_flow.line_loading_ratio[0], edge_ids)
    line_artists = []
    for edge_index, edge in enumerate(edges):
        line = edge_line_xy(edge, bus_by_id, edge_paths)
        if line is None:
            line_artists.append(None)
            continue
        xs, ys = line
        (artist,) = ax.plot(
            xs,
            ys,
            color=cmap(norm(float(frame_metric[edge_index]))),
            linewidth=line_width_for_multiplier(float(capacity_multipliers[edge_index])),
            alpha=0.90,
            solid_capstyle="round",
            zorder=4,
        )
        line_artists.append(artist)

    edge_colors = np.asarray([cmap(norm(float(value))) for value in frame_metric])
    transit_styles = _transit_node_styles(edges, edge_colors, capacity_multipliers)
    node_artists = _draw_operation_nodes(ax, buses, static_maps, transit_styles=transit_styles)
    transit_rows = [row for row in buses if _bus_kind_lookup(buses, static_maps).get(int(row[0])) == "transit_bus"]
    add_deduped_legend(ax, static_maps, fontsize=7)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("line loading")

    def update(frame: int) -> tuple[object, ...]:
        values = _metric_by_edge_id(power_flow.branch_ids, power_flow.line_loading_ratio[frame], edge_ids)
        colors = np.asarray([cmap(norm(float(value))) for value in values])
        changed: list[object] = [title]
        for artist, color in zip(line_artists, colors):
            if artist is not None:
                artist.set_color(color)
                changed.append(artist)
        if light_artist is not None and cloud_artist is not None:
            local_hour = int(hourly_weather.timestamps[frame]) % 24
            shade = _twilight_shade_rgba(cloud[frame].shape, local_hour)
            light_artist.set_data(shade)
            cloud_artist.set_data(_tinted_cloud_rgba(cloud[frame], shade[0, 0]))
            changed.extend((light_artist, cloud_artist))
        title.set_text(format_hour_timestamp(int(hourly_weather.timestamps[frame])))
        transit_artist = node_artists.get("transit_bus")
        if transit_artist is not None and transit_rows:
            styles = _transit_node_styles(edges, colors, capacity_multipliers)
            transit_artist.set_facecolors(
                [styles.get(int(row[0]), ((1.0, 0.96, 0.66, 1.0), 36.9, 0.585))[0] for row in transit_rows]
            )
            transit_artist.set_sizes(
                [styles.get(int(row[0]), ((1.0, 0.96, 0.66, 1.0), 36.9, 0.585))[1] for row in transit_rows]
            )
            transit_artist.set_linewidths(
                [styles.get(int(row[0]), ((1.0, 0.96, 0.66, 1.0), 36.9, 0.585))[2] for row in transit_rows]
            )
            changed.append(transit_artist)
        return tuple(changed)

    animation = FuncAnimation(fig, update, frames=frame_count, interval=200, blit=False)
    animation.save(path, writer=PillowWriter(fps=5), dpi=120)
    plt.close(fig)


def _twilight_shade_rgba(shape: tuple[int, int], local_hour: float) -> np.ndarray:
    night = np.asarray((0.18, 0.25, 0.34), dtype=np.float32)
    sunset = np.asarray((0.78, 0.30, 0.22), dtype=np.float32)
    hour = float(local_hour) % 24.0
    if 4.0 <= hour < 5.5:
        phase = _smoothstep((hour - 4.0) / 1.5)
        color = (1.0 - phase) * night + phase * sunset
        alpha = 0.50 - 0.16 * phase
    elif 5.5 <= hour < 7.0:
        phase = _smoothstep((hour - 5.5) / 1.5)
        color = sunset
        alpha = 0.34 * (1.0 - phase)
    elif 7.0 <= hour < 17.0:
        color = sunset
        alpha = 0.0
    elif 17.0 <= hour < 18.5:
        phase = _smoothstep((hour - 17.0) / 1.5)
        color = sunset
        alpha = 0.34 * phase
    elif 18.5 <= hour < 20.0:
        phase = _smoothstep((hour - 18.5) / 1.5)
        color = (1.0 - phase) * sunset + phase * night
        alpha = 0.34 + 0.16 * phase
    else:
        color = night
        alpha = 0.50
    rgba = np.empty((*shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = alpha
    return rgba


def _tinted_cloud_rgba(cloud: np.ndarray, shade: np.ndarray) -> np.ndarray:
    rgba = _cloud_rgba(cloud)
    tint_weight = np.clip(float(shade[3]) / 0.50, 0.0, 1.0) / 3.0
    rgba[..., :3] = (1.0 - tint_weight) * rgba[..., :3] + tint_weight * shade[:3]
    return rgba


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _bus_kind_lookup(buses: np.ndarray, static_maps: dict[str, np.ndarray]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    source_mask = static_maps.get("source_bus_map")
    wind_mask = static_maps.get("wind_candidate_map")
    pv_mask = static_maps.get("pv_candidate_map")
    thermal_mask = static_maps.get("thermal_bus_map")
    load_mask = static_maps.get("load_bus_map")
    transit_mask = static_maps.get("transit_bus_map")
    for row in buses:
        if row.size < 3:
            continue
        bus_id = int(row[0])
        rr = int(np.clip(round(float(row[1])), 0, static_maps["elevation"].shape[0] - 1))
        cc = int(np.clip(round(float(row[2])), 0, static_maps["elevation"].shape[1] - 1))
        if thermal_mask is not None and int(thermal_mask[rr, cc]) == bus_id:
            lookup[bus_id] = "thermal_bus"
        elif load_mask is not None and int(load_mask[rr, cc]) == bus_id:
            lookup[bus_id] = "load_bus"
        elif source_mask is not None and int(source_mask[rr, cc]) == bus_id:
            if wind_mask is not None and int(wind_mask[rr, cc]) >= 0:
                lookup[bus_id] = "wind_bus"
            elif pv_mask is not None and int(pv_mask[rr, cc]) >= 0:
                lookup[bus_id] = "pv_bus"
            else:
                lookup[bus_id] = "pv_bus"
        elif transit_mask is not None and int(transit_mask[rr, cc]) == bus_id:
            lookup[bus_id] = "transit_bus"
        else:
            lookup[bus_id] = "transit_bus"
    return lookup


def _save_grid_operation_timeseries(power_flow: PowerFlowStore, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter

    t = np.arange(power_flow.served_load_mw.shape[0])
    total_served = power_flow.served_load_mw.sum(axis=1)
    total_generation = power_flow.dispatched_generation_mw.sum(axis=1)
    unserved = power_flow.unserved_load_mw.sum(axis=1)
    curtailed = power_flow.curtailed_generation_mw.sum(axis=1)
    peak_loading = power_flow.line_loading_ratio.max(axis=1) if power_flow.line_loading_ratio.size else np.zeros_like(t, dtype=np.float32)
    congested_lines = (power_flow.line_loading_ratio > 0.80).sum(axis=1) if power_flow.line_loading_ratio.size else np.zeros_like(t)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(t, total_served, color="#ff4fa3", linewidth=1.6, label="served load")
    axes[0].plot(t, total_generation, color="#1aa6b7", linewidth=1.4, label="dispatched generation")
    axes[0].fill_between(t, total_served, total_served + unserved, color="#f7b7d8", alpha=0.38, label="unserved/storage needed")
    axes[0].fill_between(t, total_generation - curtailed, total_generation, color="#fff0a8", alpha=0.42, label="curtailed generation")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Grid operation balance")
    axes[0].legend(loc="upper right", ncol=1)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, peak_loading * 100.0, color="#9b2fae", linewidth=1.4, label="peak line loading")
    axes[1].bar(t, congested_lines, color="#ff9e33", alpha=0.32, label="lines > 80%")
    axes[1].axhline(100.0, color="#d62828", linewidth=1.0, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("% / count")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper right", ncol=1)
    axes[1].grid(True, alpha=0.25)

    tick_positions = np.arange(0, power_flow.served_load_mw.shape[0], 24)
    tick_labels = [format_hour_timestamp(int(power_flow.timestamps[int(position)])).rsplit(" ", 1)[0] for position in tick_positions]
    axes[1].xaxis.set_major_locator(FixedLocator(tick_positions))
    axes[1].xaxis.set_major_formatter(FixedFormatter(tick_labels))
    fig.savefig(path, dpi=180)
    plt.close(fig)
