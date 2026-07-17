from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_generator.core.datatypes import WeatherStore
from world_generator.operation.grid_update_loop import GridUpdateLoopResult
from world_generator.visualization.power_flow_figures import save_line_loading_gif, save_power_flow_figures
from world_generator.visualization.upgrade_figures import save_upgrade_figures


def save_grid_update_loop(
    static_maps: dict[str, np.ndarray],
    update_loop: GridUpdateLoopResult,
    output_dir: Path,
    hourly_weather: WeatherStore | None = None,
    *,
    render_gif: bool = True,
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
    _save_iteration_payloads(iteration, output_dir)
    files += [
        "actions.json",
        "power_flow_hourly.npz",
        "grid_upgrade_plan.npz",
        "grid_electrical.npz",
        "refined_grid_topology.npz",
        "bus_index.json",
    ]
    files += save_power_flow_figures(iteration_maps, iteration.power_flow, output_dir)
    files += save_upgrade_figures(iteration_maps, iteration.upgrade_plan, output_dir)
    if hourly_weather is not None and render_gif:
        save_line_loading_gif(
            iteration_maps,
            iteration.power_flow,
            hourly_weather,
            output_dir / "line_loading_dynamic_no_cloud.gif",
            show_weather=False,
        )
        save_line_loading_gif(
            iteration_maps,
            iteration.power_flow,
            hourly_weather,
            output_dir / "line_loading_dynamic.gif",
            show_weather=True,
        )
        files += ["line_loading_dynamic_no_cloud.gif", "line_loading_dynamic.gif"]
    return files


def _save_iteration_payloads(iteration: object, output_dir: Path) -> None:
    labels = _stable_bus_labels(iteration.refined_topology.refined_buses)
    actions = [_action_with_labels(action, labels) for action in iteration.actions]
    (output_dir / "actions.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")
    (output_dir / "bus_index.json").write_text(
        json.dumps(
            [
                {
                    "label": labels[int(bus.bus_id)],
                    "bus_id": int(bus.bus_id),
                    "kind": bus.kind,
                    "row": int(bus.row),
                    "col": int(bus.col),
                    "source_kind": bus.source_kind,
                    "source_id": int(bus.source_id),
                }
                for bus in iteration.refined_topology.refined_buses
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "power_flow_hourly.npz", **iteration.power_flow.as_arrays())
    np.savez_compressed(output_dir / "grid_upgrade_plan.npz", **iteration.upgrade_plan.as_arrays())
    np.savez_compressed(output_dir / "grid_electrical.npz", **iteration.electrical.as_arrays())
    np.savez_compressed(output_dir / "refined_grid_topology.npz", **iteration.refined_topology.as_arrays())


def _maps_with_iteration_edges(
    static_maps: dict[str, np.ndarray],
    iteration: object,
) -> dict[str, np.ndarray]:
    maps = dict(static_maps)
    maps |= iteration.refined_topology.as_maps()
    maps |= iteration.refined_topology.as_arrays()
    maps["bus_labels"] = _stable_bus_labels(iteration.refined_topology.refined_buses)
    maps["refined_grid_edge_paths"] = {
        int(edge.edge_id): (tuple(int(row) for row in edge.path_rows), tuple(int(col) for col in edge.path_cols))
        for edge in iteration.refined_topology.refined_edges
    }
    maps["electrical_branches"] = iteration.electrical.as_arrays()["electrical_branches"]
    return maps


def _stable_bus_labels(buses: tuple[object, ...]) -> dict[int, str]:
    prefixes = {
        "pv_bus": "S",
        "wind_bus": "W",
        "load_bus": "L",
        "thermal_bus": "F",
        "transit_bus": "T",
    }
    labels: dict[int, str] = {}
    for kind, prefix in prefixes.items():
        rows = sorted((bus for bus in buses if bus.kind == kind), key=lambda item: int(item.bus_id))
        for index, bus in enumerate(rows, start=1):
            labels[int(bus.bus_id)] = f"{prefix}{index}"
    for bus in buses:
        labels.setdefault(int(bus.bus_id), f"B{int(bus.bus_id)}")
    return labels


def _action_with_labels(action: dict[str, object], labels: dict[int, str]) -> dict[str, object]:
    payload = dict(action)
    logical_edges = payload.get("logical_edges", [])
    if isinstance(logical_edges, list):
        payload["logical_edge_labels"] = [
            [labels.get(int(edge[0]), f"B{int(edge[0])}"), labels.get(int(edge[1]), f"B{int(edge[1])}")]
            for edge in logical_edges
            if isinstance(edge, list | tuple) and len(edge) == 2
        ]
    if "old_from_bus" in payload and "old_to_bus" in payload:
        old_from = int(payload["old_from_bus"])
        old_to = int(payload["old_to_bus"])
        payload["old_edge_labels"] = [labels.get(old_from, f"B{old_from}"), labels.get(old_to, f"B{old_to}")]
    return payload
