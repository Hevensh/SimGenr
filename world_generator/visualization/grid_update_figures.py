from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_generator.core.datatypes import WeatherStore
from world_generator.operation.grid_update_loop import GridUpdateLoopResult
from world_generator.visualization.power_flow_figures import save_line_loading_animation, save_power_flow_figures
from world_generator.visualization.upgrade_figures import save_upgrade_figures


def save_grid_update_loop(
    static_maps: dict[str, np.ndarray],
    update_loop: GridUpdateLoopResult,
    output_dir: Path,
    hourly_weather: WeatherStore | None = None,
    *,
    render_animation: bool = True,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    summary = update_loop.summary_dict()
    if update_loop.iterations:
        summary |= update_loop.iterations[-1].summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files.append("summary.json")

    if not update_loop.iterations:
        return files
    iteration = update_loop.iterations[-1]
    iteration_maps = _maps_with_iteration_edges(static_maps, iteration)
    files += save_power_flow_figures(iteration_maps, iteration.power_flow, output_dir)
    files += save_upgrade_figures(iteration_maps, iteration.upgrade_plan, output_dir)
    if hourly_weather is not None and render_animation:
        save_line_loading_animation(
            iteration_maps,
            iteration.power_flow,
            hourly_weather,
            output_dir / "line_loading_dynamic.webp",
            show_weather=False,
        )
        save_line_loading_animation(
            iteration_maps,
            iteration.power_flow,
            hourly_weather,
            output_dir / "line_loading_dynamic_with_weather.webp",
            show_weather=True,
        )
        files += ["line_loading_dynamic.webp", "line_loading_dynamic_with_weather.webp"]
    return files


def _maps_with_iteration_edges(
    static_maps: dict[str, np.ndarray],
    iteration: object,
) -> dict[str, np.ndarray]:
    maps = dict(static_maps)
    maps |= iteration.refined_topology.as_maps()
    maps |= iteration.refined_topology.as_arrays()
    maps["refined_grid_edge_paths"] = {
        int(edge.edge_id): (tuple(int(row) for row in edge.path_rows), tuple(int(col) for col in edge.path_cols))
        for edge in iteration.refined_topology.refined_edges
    }
    maps["electrical_branches"] = iteration.electrical.as_arrays()["electrical_branches"]
    return maps
