from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.core.datatypes import (
    GridElectricalState,
    RefinedGridTopologyState,
    StorageDispatchStore,
    StorageNeedStore,
    StoragePlanStore,
    WeatherStore,
)
from world_generator.visualization.common import (
    add_deduped_legend,
    draw_built_environment_texture,
    draw_elevation_with_water_overlay,
    draw_grid_edge_lines,
    edge_capacity_multipliers,
    edge_line_xy,
    format_hour_timestamp,
    line_width_for_multiplier,
    save_animation_webp,
)
from world_generator.visualization.power_flow_figures import (
    DEFAULT_NODE_EDGE_WIDTH,
    DEFAULT_TRANSIT_MARKER_SIZE,
    _bus_kind_lookup,
    _draw_operation_nodes,
    _metric_by_edge_id,
    _transit_node_styles,
    _weather_overlay_rgba,
)


STORAGE_NEED_FILES = [
    "storage_need_overview.png",
    "storage_need_map.png",
    "storage_need_timeseries.png",
    "storage_site_plan.png",
    "storage_site_profiles.png",
]

STORAGE_DISPATCH_FILES = [
    "storage_dispatch_timeseries.png",
    "storage_soc_timeseries.png",
    "storage_network_relief.png",
    "capacity_expansion_plan.png",
]


def save_storage_need_figures(
    static_maps: dict[str, np.ndarray],
    storage: StorageNeedStore,
    plan: StoragePlanStore,
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_maps = _stage_maps(static_maps, topology, electrical)
    _save_storage_need_overview(stage_maps, storage, output_dir / "storage_need_overview.png")
    _save_storage_need_map(stage_maps, storage, output_dir / "storage_need_map.png")
    _save_storage_need_timeseries(storage, output_dir / "storage_need_timeseries.png")
    _save_storage_site_plan(stage_maps, storage, plan, output_dir / "storage_site_plan.png")
    _save_storage_site_profiles(plan, output_dir / "storage_site_profiles.png")
    return STORAGE_NEED_FILES


def save_storage_dispatch_figures(
    static_maps: dict[str, np.ndarray],
    dispatch: StorageDispatchStore,
    plan: StoragePlanStore,
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    output_dir: Path,
    hourly_weather: WeatherStore | None = None,
    *,
    render_animation: bool = True,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_maps = _stage_maps(static_maps, topology, electrical)
    _save_storage_dispatch_timeseries(dispatch, output_dir / "storage_dispatch_timeseries.png")
    _save_storage_soc_timeseries(dispatch, plan, output_dir / "storage_soc_timeseries.png")
    _save_storage_network_relief(stage_maps, dispatch, plan, output_dir / "storage_network_relief.png")
    _save_capacity_expansion_plan(stage_maps, dispatch, plan, output_dir / "capacity_expansion_plan.png")
    files = list(STORAGE_DISPATCH_FILES)
    if render_animation:
        _save_storage_dispatch_animation(
            stage_maps,
            dispatch,
            plan,
            output_dir / "storage_dispatch.webp",
        )
        files.append("storage_dispatch.webp")
        if hourly_weather is not None:
            _save_storage_dispatch_animation(
                stage_maps,
                dispatch,
                plan,
                output_dir / "storage_dispatch_with_weather.webp",
                hourly_weather=hourly_weather,
            )
            files.append("storage_dispatch_with_weather.webp")
    return files


def _stage_maps(
    static_maps: dict[str, np.ndarray],
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
) -> dict[str, object]:
    maps: dict[str, object] = dict(static_maps)
    maps |= topology.as_maps()
    maps |= topology.as_arrays()
    maps["refined_grid_edge_paths"] = {
        int(edge.edge_id): (edge.path_rows, edge.path_cols) for edge in topology.refined_edges
    }
    maps |= electrical.as_arrays()
    return maps


def _save_storage_need_overview(
    static_maps: dict[str, object],
    storage: StorageNeedStore,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.2), constrained_layout=True)
    panels = [
        (storage.need_score, "Storage need score", "need score"),
        (storage.suggested_power_mw, "Suggested storage power", "MW"),
        (storage.suggested_energy_mwh, "Suggested storage energy", "MWh"),
        (storage.congestion_exposure_hours, "Congestion exposure", "weighted hours"),
    ]
    for ax, (values, title, label) in zip(axes.ravel(), panels):
        _draw_storage_nodes(ax, static_maps, storage, values, title, label, fig)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_storage_need_map(
    static_maps: dict[str, object],
    storage: StorageNeedStore,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    _draw_network_base(ax, static_maps)
    image = ax.imshow(
        storage.need_score_map,
        origin="upper",
        cmap="magma",
        norm=Normalize(0.0, 1.0),
        alpha=np.clip(storage.need_score_map * 0.72, 0.0, 0.72),
        zorder=3,
    )
    sizes = _marker_sizes(storage.suggested_energy_mwh)
    scatter = ax.scatter(
        storage.cols,
        storage.rows,
        c=storage.need_score,
        s=sizes,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        edgecolors="#fff4b8",
        linewidths=0.8,
        zorder=5,
        label="storage need",
    )
    ax.set_title("Load-side storage need")
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("storage need score")
    if storage.bus_ids.size:
        ax.legend(handles=[scatter], labels=["load-side storage need"], loc="lower right", fontsize=8, framealpha=0.86)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_storage_nodes(
    ax: object,
    static_maps: dict[str, object],
    storage: StorageNeedStore,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    fig: object,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    _draw_network_base(ax, static_maps)
    vmax = max(float(np.nanmax(values, initial=0.0)), 1e-6)
    norm = Normalize(0.0, vmax)
    scatter = ax.scatter(
        storage.cols,
        storage.rows,
        c=values,
        s=_marker_sizes(storage.suggested_energy_mwh),
        cmap="magma",
        norm=norm,
        edgecolors="#fff4b8",
        linewidths=0.5,
        zorder=5,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    return scatter


def _draw_network_base(ax: object, static_maps: dict[str, object]) -> None:
    draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps, scale=3)
    buses = np.atleast_2d(static_maps.get("refined_grid_buses", np.empty((0, 0))))
    edges = np.atleast_2d(static_maps.get("refined_grid_edges", np.empty((0, 0))))
    draw_grid_edge_lines(
        ax,
        buses,
        edges,
        color="#5ec8d8",
        redundant_color="#5ec8d8",
        linewidth=0.8,
        alpha=0.38,
        edge_paths=static_maps.get("refined_grid_edge_paths"),
        unified_legend_label="transmission line",
    )


def _marker_sizes(energy_mwh: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_mwh, dtype=np.float32)
    maximum = float(np.max(energy, initial=0.0))
    normalized = energy / maximum if maximum > 1e-12 else np.zeros_like(energy)
    return 34.0 + 150.0 * np.sqrt(normalized)


def _save_storage_need_timeseries(storage: StorageNeedStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    hours = storage.timestamps.size
    t = np.arange(hours)
    aggregate = storage.support_requirement_mw.sum(axis=1)
    congestion = storage.congestion_support_mw.sum(axis=1)
    unserved = storage.unserved_load_mw.sum(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.6), sharex=True, constrained_layout=True)
    axes[0].plot(t, aggregate, color="#b5179e", linewidth=1.6, label="total support requirement")
    axes[0].plot(t, congestion, color="#4cc9f0", linewidth=1.15, label="congestion component")
    axes[0].plot(t, unserved, color="#f25f5c", linewidth=1.1, label="unserved component")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Hourly storage support requirement")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85)

    top = np.argsort(-storage.need_score)[: min(4, storage.bus_ids.size)]
    for index in top:
        axes[1].plot(
            t,
            storage.support_requirement_mw[:, index],
            linewidth=1.15,
            label=f"load bus {int(storage.bus_ids[index])}",
        )
    axes[1].set_ylabel("MW")
    axes[1].set_title("Highest-priority load regions")
    if top.size:
        axes[1].legend(loc="upper right", fontsize=8, framealpha=0.85)
    tick_positions = np.arange(0, hours, 24)
    tick_labels = [format_hour_timestamp(int(storage.timestamps[position])).rsplit(" ", 1)[0] for position in tick_positions]
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_xlabel("date")
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_storage_site_plan(
    static_maps: dict[str, object],
    storage: StorageNeedStore,
    plan: StoragePlanStore,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.4, 7.1), constrained_layout=True)
    _draw_network_base(ax, static_maps)
    colors = plt.get_cmap("tab10")(np.arange(max(len(plan.sites), 1)) % 10)
    load_index = {int(bus_id): index for index, bus_id in enumerate(storage.bus_ids)}
    for site in plan.sites:
        color = colors[site.site_id]
        for bus_id in site.covered_bus_ids:
            index = load_index[int(bus_id)]
            ax.plot(
                [storage.cols[index], site.col],
                [storage.rows[index], site.row],
                color=color,
                linewidth=0.7,
                linestyle="--",
                alpha=0.32,
                zorder=3,
            )
            ax.scatter(
                storage.cols[index],
                storage.rows[index],
                s=28,
                color=color,
                edgecolors="white",
                linewidths=0.45,
                alpha=0.82,
                zorder=4,
            )
        size = 100.0 + 180.0 * np.sqrt(site.energy_mwh / max(max((item.energy_mwh for item in plan.sites), default=1.0), 1e-6))
        ax.scatter(
            site.col,
            site.row,
            s=size,
            marker="H",
            color="#ffe082",
            edgecolors=color,
            linewidths=2.0,
            zorder=6,
        )
    ax.set_title("Storage attachment buses and service regions")
    ax.set_xticks([])
    ax.set_yticks([])
    if plan.sites:
        ax.legend(
            handles=[
                Line2D([0], [0], marker="H", color="none", markerfacecolor="#ffe082", markeredgecolor="#555555", markersize=10, label="storage at load bus"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#5aa9e6", markeredgecolor="white", markersize=6, label="served load bus"),
                Line2D([0], [0], color="#777777", linestyle="--", linewidth=0.8, label="service assignment"),
            ],
            loc="lower right",
            fontsize=8,
            framealpha=0.86,
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_storage_site_profiles(plan: StoragePlanStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.2), constrained_layout=True)
    t = np.arange(plan.timestamps.size)
    colors = plt.get_cmap("tab10")(np.arange(max(len(plan.sites), 1)) % 10)
    for site in plan.sites:
        axes[0].plot(
            t,
            plan.site_support_requirement_mw[:, site.site_id],
            color=colors[site.site_id],
            linewidth=1.25,
            label=f"storage {site.site_id + 1} at bus {site.bus_id}",
        )
    axes[0].set_title("Storage service-region support profiles")
    axes[0].set_ylabel("MW")
    if plan.sites:
        axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=2)
    axes[0].grid(alpha=0.18)

    positions = np.arange(len(plan.sites), dtype=np.float32)
    power = np.asarray([site.power_mw for site in plan.sites], dtype=np.float32)
    energy = np.asarray([site.energy_mwh for site in plan.sites], dtype=np.float32)
    width = 0.38
    axes[1].bar(positions - width / 2, power, width=width, color="#4cc9f0", label="power MW")
    energy_axis = axes[1].twinx()
    energy_axis.bar(positions + width / 2, energy, width=width, color="#ffb74d", label="energy MWh")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"S{site.site_id + 1}\nbus {site.bus_id}" for site in plan.sites])
    axes[1].set_ylabel("power (MW)")
    energy_axis.set_ylabel("energy (MWh)")
    axes[1].set_title("Suggested storage ratings")
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = energy_axis.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b, loc="upper right", fontsize=8, framealpha=0.85)
    axes[1].grid(axis="y", alpha=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _date_ticks(timestamps: np.ndarray) -> tuple[np.ndarray, list[str]]:
    positions = np.arange(0, timestamps.size, 24, dtype=np.int32)
    labels = [format_hour_timestamp(int(timestamps[index])).rsplit(" ", 1)[0] for index in positions]
    return positions, labels


def _save_storage_dispatch_timeseries(dispatch: StorageDispatchStore, path: Path) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(dispatch.timestamps.size)
    charge = dispatch.charge_mw.sum(axis=1)
    discharge = dispatch.discharge_mw.sum(axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(14.6, 8.4), sharex=True, constrained_layout=True)
    axes[0].plot(t, dispatch.total_load_mw, color="#ef476f", linewidth=1.45, label="load")
    axes[0].plot(t, dispatch.renewable_available_mw, color="#06d6a0", linewidth=1.15, label="renewable available")
    axes[0].plot(t, dispatch.scheduled_thermal_mw, color="#f78c6b", linewidth=1.25, label="thermal scheduled")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Generation and load after storage coordination")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.86)

    axes[1].fill_between(t, 0.0, discharge, color="#ffd166", alpha=0.78, label="storage discharge")
    axes[1].fill_between(t, 0.0, -charge, color="#4cc9f0", alpha=0.72, label="storage charge")
    emergency = dispatch.emergency_discharge_mw.sum(axis=1)
    if np.any(emergency > 1e-5):
        axes[1].plot(t, emergency, color="#ef476f", linewidth=1.2, label="fast reserve discharge")
    axes[1].axhline(0.0, color="#444444", linewidth=0.6)
    axes[1].set_ylabel("MW")
    axes[1].set_title("Storage power (positive discharge, negative charge)")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.86)

    axes[2].plot(t, dispatch.baseline_unserved_mw, color="#ef476f", linewidth=1.15, linestyle="--", label="unserved before")
    axes[2].plot(t, dispatch.dispatched_unserved_mw, color="#7b2cbf", linewidth=1.45, label="unserved after")
    axes[2].fill_between(
        t,
        dispatch.dispatched_unserved_mw,
        dispatch.baseline_unserved_mw,
        where=dispatch.baseline_unserved_mw >= dispatch.dispatched_unserved_mw,
        color="#80ed99",
        alpha=0.42,
        label="served by storage",
    )
    axes[2].set_ylabel("MW")
    axes[2].set_title("Unserved load reduction")
    axes[2].legend(loc="upper right", fontsize=8, framealpha=0.86)
    positions, labels = _date_ticks(dispatch.timestamps)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("date")
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_storage_soc_timeseries(
    dispatch: StorageDispatchStore,
    plan: StoragePlanStore,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14.2, 5.8), constrained_layout=True)
    t = np.arange(dispatch.timestamps.size + 1)
    colors = plt.get_cmap("tab10")(np.arange(max(len(plan.sites), 1)) % 10)
    for index, site in enumerate(plan.sites):
        capacity = max(float(site.energy_mwh), 1e-6)
        ax.plot(t, 100.0 * dispatch.soc_mwh[:, index] / capacity, color=colors[index], linewidth=1.5, label=f"S{site.site_id + 1}")
    if plan.sites:
        terminal_fraction = np.median(
            dispatch.cycle_boundary_soc_mwh
            / np.maximum(dispatch.site_energy_capacity_mwh, 1e-6)
        )
        ax.axhline(
            100.0 * float(terminal_fraction),
            color="#333333",
            linewidth=1.0,
            linestyle=":",
            label="cycle boundary SOC",
        )
    ax.axhspan(0.0, 100.0 * dispatch.minimum_soc_fraction, color="#ef476f", alpha=0.07)
    ax.axhspan(
        100.0 * dispatch.preferred_soc_lower_fraction,
        100.0 * dispatch.preferred_soc_upper_fraction,
        color="#80ed99",
        alpha=0.08,
    )
    ax.axhspan(100.0 * dispatch.maximum_soc_fraction, 100.0, color="#4cc9f0", alpha=0.06)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("SOC (%)")
    ax.set_title("Storage state of charge")
    positions, labels = _date_ticks(dispatch.timestamps)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("date")
    ax.grid(alpha=0.18)
    if plan.sites:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.86, ncol=min(len(plan.sites) + 1, 6))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_storage_network_relief(
    static_maps: dict[str, object],
    dispatch: StorageDispatchStore,
    plan: StoragePlanStore,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)
    baseline_peak = dispatch.baseline_line_loading_ratio.max(axis=0, initial=0.0)
    dispatched_peak = dispatch.dispatched_line_loading_ratio.max(axis=0, initial=0.0)
    vmax = max(float(np.max(baseline_peak, initial=0.0)), float(np.max(dispatched_peak, initial=0.0)), 1.0)
    norm = Normalize(0.0, vmax)
    cmap = _storage_loading_cmap()
    for ax, metric, title in (
        (axes[0], baseline_peak, "Peak line loading before storage"),
        (axes[1], dispatched_peak, "Peak line loading after storage"),
    ):
        _draw_network_base(ax, static_maps)
        _draw_loading_edges(ax, static_maps, dispatch.branch_ids, metric, norm, cmap)
        _draw_storage_site_markers(ax, plan, color="#ffe082")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("peak loading ratio")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_capacity_expansion_plan(
    static_maps: dict[str, object],
    dispatch: StorageDispatchStore,
    plan: StoragePlanStore,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(13.8, 7.0), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=(1.25, 1.0))
    map_ax = fig.add_subplot(grid[:, 0])
    power_ax = fig.add_subplot(grid[0, 1])
    network_ax = fig.add_subplot(grid[1, 1])
    _draw_network_base(map_ax, static_maps)

    buses = np.atleast_2d(static_maps.get("refined_grid_buses", np.empty((0, 0))))
    bus_by_id = {int(row[0]): row for row in buses if row.size >= 3}
    thermal_artists = []
    for bus_id, expansion in zip(dispatch.thermal_bus_ids, dispatch.thermal_capacity_expansion_mw):
        if expansion <= 1e-4 or int(bus_id) not in bus_by_id:
            continue
        row = bus_by_id[int(bus_id)]
        thermal_artists.append(
            map_ax.scatter(
                row[2],
                row[1],
                s=75.0 + 2.0 * float(expansion),
                marker="D",
                color="#ff7043",
                edgecolors="#4a1f14",
                linewidths=1.0,
                zorder=8,
            )
        )
    storage_artists = []
    for index, site in enumerate(plan.sites):
        expansion = float(dispatch.storage_power_expansion_mw[index])
        energy_expansion = float(dispatch.storage_energy_expansion_mwh[index])
        if max(expansion, energy_expansion) <= 1e-4:
            continue
        storage_artists.append(
            map_ax.scatter(
                site.col,
                site.row,
                s=72.0 + 3.0 * expansion + energy_expansion,
                marker="H",
                color="#ffe082",
                edgecolors="#6d5512",
                linewidths=1.0,
                zorder=8,
            )
        )
    edges = np.atleast_2d(static_maps.get("refined_grid_edges", np.empty((0, 0))))
    line_expansion_by_id = {
        int(branch_id): float(value)
        for branch_id, value in zip(dispatch.branch_ids, dispatch.line_capacity_expansion_mva)
    }
    for edge in edges:
        expansion = line_expansion_by_id.get(int(edge[0]), 0.0)
        if expansion <= 1e-4:
            continue
        line = edge_line_xy(edge, bus_by_id, static_maps.get("refined_grid_edge_paths"))
        if line is not None:
            map_ax.plot(*line, color="#e040fb", linewidth=1.5 + 0.015 * expansion, alpha=0.9, zorder=7)
    map_ax.set_title("Stage 14 capacity additions")
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    legend_handles = []
    if thermal_artists:
        legend_handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor="#ff7043", markeredgecolor="#4a1f14", label="thermal expansion"))
    if storage_artists:
        legend_handles.append(Line2D([0], [0], marker="H", color="none", markerfacecolor="#ffe082", markeredgecolor="#6d5512", label="storage expansion"))
    if np.any(dispatch.line_capacity_expansion_mva > 1e-4):
        legend_handles.append(Line2D([0], [0], color="#e040fb", linewidth=2.2, label="line reinforcement"))
    if legend_handles:
        map_ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.86)

    thermal_values = dispatch.thermal_capacity_expansion_mw
    storage_values = dispatch.storage_power_expansion_mw
    labels = [f"F{index + 1}" for index in range(dispatch.thermal_bus_ids.size)] + [f"S{site.site_id + 1}" for site in plan.sites]
    values = np.concatenate((thermal_values, storage_values))
    colors = ["#ff7043"] * thermal_values.size + ["#ffd166"] * storage_values.size
    positions = np.arange(values.size)
    power_ax.bar(positions, values, color=colors, edgecolor="#333333", linewidth=0.4)
    power_ax.set_xticks(positions)
    power_ax.set_xticklabels(labels, fontsize=8)
    power_ax.set_ylabel("added capacity (MW)")
    power_ax.set_title("Firm generation and storage power")
    power_ax.grid(axis="y", alpha=0.18)

    electrical_buses = np.atleast_2d(static_maps.get("electrical_buses", np.empty((0, 0))))
    planned_thermal = sum(
        float(row[2])
        for row in electrical_buses
        if row.size > 2 and int(row[0]) in {int(value) for value in dispatch.thermal_bus_ids}
    )
    electrical_branches = np.atleast_2d(static_maps.get("electrical_branches", np.empty((0, 0))))
    planned_line = float(electrical_branches[:, 8].sum()) if electrical_branches.size else 0.0
    additions = np.asarray(
        [
            dispatch.thermal_capacity_expansion_mw.sum(),
            dispatch.storage_power_expansion_mw.sum(),
            dispatch.storage_energy_expansion_mwh.sum(),
            dispatch.line_capacity_expansion_mva.sum(),
        ],
        dtype=np.float64,
    )
    bases = np.asarray(
        [
            planned_thermal - additions[0],
            dispatch.site_power_capacity_mw.sum() - additions[1],
            dispatch.site_energy_capacity_mwh.sum() - additions[2],
            planned_line - additions[3],
        ],
        dtype=np.float64,
    )
    relative_additions = 100.0 * additions / np.maximum(bases, 1e-6)
    network_ax.bar(
        np.arange(4),
        relative_additions,
        color=["#ff7043", "#ffd166", "#4cc9f0", "#e040fb"],
        edgecolor="#333333",
        linewidth=0.4,
    )
    network_ax.set_xticks(np.arange(4))
    network_ax.set_xticklabels(["thermal", "storage\npower", "storage\nenergy", "line"])
    network_ax.set_ylabel("increase from existing capacity (%)")
    network_ax.set_title("Relative capacity additions")
    network_ax.grid(axis="y", alpha=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_loading_edges(
    ax: object,
    static_maps: dict[str, object],
    branch_ids: np.ndarray,
    values: np.ndarray,
    norm: object,
    cmap: object,
) -> list[object]:
    buses = np.atleast_2d(static_maps.get("refined_grid_buses", np.empty((0, 0))))
    edges = np.atleast_2d(static_maps.get("refined_grid_edges", np.empty((0, 0))))
    bus_by_id = {int(row[0]): row for row in buses if row.size >= 3}
    value_by_id = {int(branch_id): float(value) for branch_id, value in zip(branch_ids, values)}
    artists = []
    for edge in edges:
        line = edge_line_xy(edge, bus_by_id, static_maps.get("refined_grid_edge_paths"))
        if line is None:
            continue
        value = value_by_id.get(int(edge[0]), 0.0)
        artists.extend(ax.plot(*line, color=cmap(norm(value)), linewidth=1.35, alpha=0.92, zorder=4))
    return artists


def _draw_storage_site_markers(ax: object, plan: StoragePlanStore, color: object) -> object | None:
    if not plan.sites:
        return None
    return ax.scatter(
        [site.col for site in plan.sites],
        [site.row for site in plan.sites],
        s=70,
        marker="H",
        c=color,
        edgecolors="#5d4b20",
        linewidths=0.8,
        zorder=7,
    )


def _save_storage_dispatch_animation(
    static_maps: dict[str, object],
    dispatch: StorageDispatchStore,
    plan: StoragePlanStore,
    path: Path,
    *,
    hourly_weather: WeatherStore | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if dispatch.timestamps.size == 0:
        return
    buses = np.atleast_2d(static_maps.get("refined_grid_buses", np.empty((0, 0))))
    edges = np.atleast_2d(static_maps.get("refined_grid_edges", np.empty((0, 0))))
    if buses.size == 0 or edges.size == 0:
        return
    edge_ids = edges[:, 0].astype(int)
    capacity_multipliers = edge_capacity_multipliers(static_maps, edge_ids)
    bus_by_id = {int(row[0]): row for row in buses if row.size >= 3}
    edge_paths = static_maps.get("refined_grid_edge_paths")
    bus_kind = _bus_kind_lookup(buses, static_maps)

    fig, ax = plt.subplots(figsize=(7.4, 6.3), constrained_layout=True)
    draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps, scale=3)
    weather_artist = None
    weather_overlays = None
    if hourly_weather is not None and "cloud" in hourly_weather.channel_names:
        channels = {name: index for index, name in enumerate(hourly_weather.channel_names)}
        cloud = hourly_weather.dynamic[:, channels["cloud"]]
        weather_by_timestamp = {
            int(timestamp): index for index, timestamp in enumerate(hourly_weather.timestamps)
        }
        weather_indices = [weather_by_timestamp.get(int(timestamp)) for timestamp in dispatch.timestamps]
        if all(index is not None for index in weather_indices):
            weather_overlays = np.stack(
                [
                    _weather_overlay_rgba(
                        cloud[int(index)],
                        int(dispatch.timestamps[frame]) % 24,
                    )
                    for frame, index in enumerate(weather_indices)
                ]
            )
            weather_artist = ax.imshow(
                weather_overlays[0],
                origin="upper",
                interpolation="nearest",
                zorder=3,
            )
    norm = Normalize(0.0, 1.0, clip=True)
    cmap = _storage_loading_cmap()
    frame_metric = _metric_by_edge_id(
        dispatch.branch_ids,
        dispatch.dispatched_line_loading_ratio[0],
        edge_ids,
    )
    line_artists: list[object | None] = []
    for edge_index, edge in enumerate(edges):
        line = edge_line_xy(edge, bus_by_id, edge_paths)
        if line is None:
            line_artists.append(None)
            continue
        (artist,) = ax.plot(
            *line,
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
    transit_rows = [row for row in buses if bus_kind.get(int(row[0])) == "transit_bus"]
    site_artist = None
    if plan.sites:
        initial_soc = dispatch.soc_mwh[0] / np.maximum(dispatch.site_energy_capacity_mwh, 1e-6)
        site_artist = ax.scatter(
            [site.col for site in plan.sites],
            [site.row for site in plan.sites],
            s=59.0,
            marker=(6, 0, 0),
            c=cmap(norm(initial_soc)),
            edgecolors="#151515",
            linewidths=1.45,
            alpha=0.98,
            label="storage (fill: SOC)",
            zorder=7,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    add_deduped_legend(ax, static_maps, fontsize=7)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("ratio (line loading / storage SOC)")

    def update(hour: int) -> list[object]:
        values = _metric_by_edge_id(
            dispatch.branch_ids,
            dispatch.dispatched_line_loading_ratio[hour],
            edge_ids,
        )
        colors = np.asarray([cmap(norm(float(value))) for value in values])
        changed = []
        if weather_artist is not None and weather_overlays is not None:
            weather_artist.set_data(weather_overlays[hour])
            changed.append(weather_artist)
        for artist, color in zip(line_artists, colors):
            if artist is not None:
                artist.set_color(color)
                changed.append(artist)
        transit_artist = node_artists.get("transit_bus")
        if transit_artist is not None and transit_rows:
            styles = _transit_node_styles(edges, colors, capacity_multipliers)
            fallback = (
                (1.0, 0.96, 0.66, 1.0),
                DEFAULT_TRANSIT_MARKER_SIZE,
                DEFAULT_NODE_EDGE_WIDTH,
            )
            transit_artist.set_facecolors([styles.get(int(row[0]), fallback)[0] for row in transit_rows])
            transit_artist.set_sizes([styles.get(int(row[0]), fallback)[1] for row in transit_rows])
            transit_artist.set_linewidths([styles.get(int(row[0]), fallback)[2] for row in transit_rows])
            changed.append(transit_artist)
        if site_artist is not None and plan.sites:
            soc_ratio = dispatch.soc_mwh[hour] / np.maximum(dispatch.site_energy_capacity_mwh, 1e-6)
            discharge = dispatch.discharge_mw[hour] + dispatch.emergency_discharge_mw[hour]
            charge = dispatch.charge_mw[hour]
            border_colors = [
                "#e53935" if discharge[index] > 1e-3 else "#1464d2" if charge[index] > 1e-3 else "#151515"
                for index in range(len(plan.sites))
            ]
            site_artist.set_facecolors(cmap(norm(soc_ratio)))
            site_artist.set_edgecolors(border_colors)
            changed.append(site_artist)
        total_capacity = max(sum(site.energy_mwh for site in plan.sites), 1e-6)
        soc_percent = 100.0 * float(dispatch.soc_mwh[hour].sum()) / total_capacity
        ax.set_title(f"{format_hour_timestamp(int(dispatch.timestamps[hour]))}   storage SOC {soc_percent:.0f}%")
        return changed

    animation = FuncAnimation(fig, update, frames=dispatch.timestamps.size, interval=200, blit=False)
    save_animation_webp(animation, path, fps=4, dpi=140)
    plt.close(fig)


def _storage_loading_cmap() -> object:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap("turbo")
    return LinearSegmentedColormap.from_list(
        "turbo_visible_high",
        base(np.linspace(0.18, 0.90, 256)),
    )
