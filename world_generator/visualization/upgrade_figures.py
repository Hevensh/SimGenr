from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.core.datatypes import GridUpgradePlanStore
from world_generator.visualization.common import (
    add_deduped_legend,
    draw_built_environment_texture,
    draw_elevation_with_water_overlay,
    edge_capacity_multipliers,
    edge_line_xy,
    edge_midpoint_xy,
    line_width_for_multiplier,
)
from world_generator.visualization.power_flow_figures import _draw_operation_nodes, _transit_node_styles


UPGRADE_FILES = [
    "upgrade_priority_map.png",
]


def save_upgrade_figures(
    static_maps: dict[str, np.ndarray],
    upgrade_plan: GridUpgradePlanStore,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_upgrade_priority_map(static_maps, upgrade_plan, output_dir / "upgrade_priority_map.png")
    return UPGRADE_FILES


def _save_upgrade_priority_map(static_maps: dict[str, np.ndarray], upgrade_plan: GridUpgradePlanStore, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    buses = np.atleast_2d(static_maps.get("refined_grid_buses", static_maps.get("grid_buses", np.empty((0, 0)))))
    edges = np.atleast_2d(static_maps.get("refined_grid_edges", static_maps.get("grid_edges", np.empty((0, 0)))))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps, scale=3)
    ax.set_title("Recommended grid upgrades")
    ax.set_xticks([])
    ax.set_yticks([])

    priority_by_edge = _values_by_edge_id(upgrade_plan.branch_ids, upgrade_plan.priority_score, edges[:, 0].astype(int))
    factor_by_edge = _values_by_edge_id(upgrade_plan.branch_ids, upgrade_plan.upgrade_factor, edges[:, 0].astype(int))
    capacity_multipliers = edge_capacity_multipliers(static_maps, edges[:, 0].astype(int))
    vmax = max(float(np.nanpercentile(priority_by_edge, 96)), 0.1) if priority_by_edge.size else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("inferno")
    bus_by_id = {int(row[0]): row for row in buses if row.size >= 3}
    edge_paths = static_maps.get("refined_grid_edge_paths")

    for edge_index in np.argsort(np.nan_to_num(priority_by_edge, nan=-1.0)):
        edge = edges[edge_index]
        line = edge_line_xy(edge, bus_by_id, edge_paths)
        if line is None:
            continue
        xs, ys = line
        priority = float(priority_by_edge[edge_index])
        factor = float(factor_by_edge[edge_index])
        linewidth = line_width_for_multiplier(float(capacity_multipliers[edge_index]))
        if factor <= 1.01:
            color = "#2f2f2f"
            alpha = 0.26
            zorder = 2
        else:
            color = cmap(norm(priority))
            alpha = 0.88
            zorder = 4
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )

    _highlight_top_upgrades(ax, buses, edges, priority_by_edge, factor_by_edge, edge_paths)
    edge_colors = np.asarray(
        [
            (0.184, 0.184, 0.184, 1.0) if float(factor) <= 1.01 else cmap(norm(float(priority)))
            for priority, factor in zip(priority_by_edge, factor_by_edge)
        ]
    )
    transit_styles = _transit_node_styles(edges, edge_colors, capacity_multipliers)
    _draw_operation_nodes(ax, buses, static_maps, transit_styles=transit_styles)
    add_deduped_legend(ax, static_maps, fontsize=7)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("upgrade priority")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _highlight_top_upgrades(
    ax: object,
    buses: np.ndarray,
    edges: np.ndarray,
    priority_by_edge: np.ndarray,
    factor_by_edge: np.ndarray,
    edge_paths: object | None = None,
) -> None:
    bus_by_id = {int(row[0]): row for row in buses if row.size >= 3}
    candidate_indices = np.where(factor_by_edge > 1.01)[0]
    if candidate_indices.size == 0:
        return
    top_indices = candidate_indices[np.argsort(-priority_by_edge[candidate_indices])[:5]]
    for rank, edge_index in enumerate(top_indices, start=1):
        edge = edges[edge_index]
        midpoint = edge_midpoint_xy(edge, bus_by_id, edge_paths)
        if midpoint is None:
            continue
        mid_col, mid_row = midpoint
        ax.text(
            mid_col,
            mid_row,
            str(rank),
            color="#111111",
            fontsize=7,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.18", "facecolor": "#fff4a8", "edgecolor": "#111111", "linewidth": 0.5},
            zorder=7,
        )


def _values_by_edge_id(source_ids: np.ndarray, source_values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    values_by_id = {int(edge_id): float(value) for edge_id, value in zip(source_ids, source_values)}
    return np.asarray([values_by_id.get(int(edge_id), np.nan) for edge_id in target_ids], dtype=np.float32)
