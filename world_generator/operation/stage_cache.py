from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_generator.core.output_layout import WorldDataLayout
from world_generator.core.datatypes import (
    BranchElectricalParam,
    BusElectricalParam,
    GridBus,
    GridEdge,
    GridElectricalState,
    PowerFlowStore,
    RefinedGridTopologyState,
    StoragePlanStore,
    StorageSite,
    WeatherStore,
)


def load_hourly_weather_checkpoint(output_dir: Path) -> WeatherStore:
    layout = WorldDataLayout(output_dir / "data")
    path = layout.existing(layout.weather, "hourly_weather_week.npz", layout.root / "hourly_weather_week.npz")
    if not path.exists():
        raise FileNotFoundError(f"Hourly weather checkpoint is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return WeatherStore(
            dynamic=payload["dynamic"].copy(),
            weather_class=payload["weather_class"].copy(),
            timestamps=payload["timestamps"].copy(),
            channel_names=tuple(str(value) for value in payload["channel_names"]),
            time_unit=str(payload["time_unit"]),
            start_day_of_year=int(payload["start_day_of_year"]),
            diagnostics={name.removeprefix("diagnostic__"): payload[name].copy() for name in payload.files if name.startswith("diagnostic__")},
            metadata=json.loads(str(payload["weather_metadata_json"])) if "weather_metadata_json" in payload else {},
            static_elevation_m=payload["static_elevation_m"].copy() if "static_elevation_m" in payload else None,
        )


def load_stage12_checkpoint(
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], RefinedGridTopologyState, GridElectricalState, PowerFlowStore]:
    layout = WorldDataLayout(output_dir / "data")
    static_maps_path = layout.existing(layout.topology, "static_maps.npz", layout.root / "static_maps.npz")
    stage_dir = layout.grid_update
    if not stage_dir.exists():
        legacy_data_stage = layout.root / "stage_12"
        stage_dir = legacy_data_stage if legacy_data_stage.exists() else output_dir / "figures" / "stage_12_grid_update"
    required = [
        static_maps_path,
        stage_dir / "refined_grid_topology.npz",
        stage_dir / "grid_electrical.npz",
        stage_dir / "power_flow_hourly.npz",
        stage_dir / "bus_index.json",
        stage_dir / "refined_grid_edges.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage 12 checkpoint is incomplete: " + ", ".join(missing))

    with np.load(static_maps_path, allow_pickle=False) as payload:
        static_maps = {name: payload[name] for name in payload.files}
    with np.load(stage_dir / "refined_grid_topology.npz", allow_pickle=False) as payload:
        bus_rows = payload["refined_grid_buses"]
        edge_rows = payload["refined_grid_edges"]
    bus_metadata = {
        int(item["bus_id"]): item
        for item in json.loads((stage_dir / "bus_index.json").read_text(encoding="utf-8"))
    }
    buses = tuple(_bus_from_row(row, bus_metadata.get(int(row[0]), {})) for row in np.atleast_2d(bus_rows))
    bus_by_id = {int(bus.bus_id): bus for bus in buses}
    path_metadata = _edge_path_metadata(stage_dir / "refined_grid_edges.json")
    edges = tuple(_edge_from_row(row, bus_by_id, path_metadata.get(int(row[0]))) for row in np.atleast_2d(edge_rows))
    shape = static_maps["elevation"].shape
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.asarray(static_maps.get("refined_line_route_map", np.zeros(shape)), dtype=np.float32),
        refined_grid_edge_map=np.asarray(static_maps.get("refined_grid_edge_map", np.full(shape, -1)), dtype=np.int32),
        transit_bus_map=np.asarray(static_maps.get("transit_bus_map", np.full(shape, -1)), dtype=np.int32),
        refined_buses=buses,
        refined_edges=edges,
    )

    with np.load(stage_dir / "grid_electrical.npz", allow_pickle=False) as payload:
        electrical = _electrical_from_arrays(payload["electrical_buses"], payload["electrical_branches"], bus_by_id)
    with np.load(stage_dir / "power_flow_hourly.npz", allow_pickle=False) as payload:
        power_flow = PowerFlowStore(
            timestamps=payload["timestamps"].copy(),
            bus_ids=payload["bus_ids"].copy(),
            branch_ids=payload["branch_ids"].copy(),
            bus_angle_rad=payload["bus_angle_rad"].copy(),
            bus_p_injection_mw=payload["bus_p_injection_mw"].copy(),
            served_load_mw=payload["served_load_mw"].copy(),
            dispatched_generation_mw=payload["dispatched_generation_mw"].copy(),
            unserved_load_mw=payload["unserved_load_mw"].copy(),
            curtailed_generation_mw=payload["curtailed_generation_mw"].copy(),
            line_flow_mw=payload["line_flow_mw"].copy(),
            line_loading_ratio=payload["line_loading_ratio"].copy(),
            slack_bus_id=int(payload["slack_bus_id"]),
        )
    return static_maps, topology, electrical, power_flow


def save_stage12_checkpoint(iteration: object, stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    labels = _stable_bus_labels(iteration.refined_topology.refined_buses)
    actions = [_action_with_labels(action, labels) for action in iteration.actions]
    (stage_dir / "actions.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")
    (stage_dir / "bus_index.json").write_text(
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
    np.savez_compressed(stage_dir / "power_flow_hourly.npz", **iteration.power_flow.as_arrays())
    np.savez_compressed(stage_dir / "grid_upgrade_plan.npz", **iteration.upgrade_plan.as_arrays())
    np.savez_compressed(stage_dir / "grid_electrical.npz", **iteration.electrical.as_arrays())
    np.savez_compressed(stage_dir / "refined_grid_topology.npz", **iteration.refined_topology.as_arrays())
    (stage_dir / "refined_grid_edges.json").write_text(
        json.dumps(iteration.refined_topology.edges_as_dicts(), indent=2),
        encoding="utf-8",
    )


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


def load_stage13_checkpoint(
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], RefinedGridTopologyState, GridElectricalState, PowerFlowStore, StoragePlanStore]:
    static_maps, topology, electrical, power_flow = load_stage12_checkpoint(output_dir)
    layout = WorldDataLayout(output_dir / "data")
    plan_path = layout.existing(layout.storage_planning, "storage_plan.npz", layout.root / "storage_plan.npz")
    metadata_path = layout.existing(layout.storage_planning, "storage_plan.json", layout.root / "storage_plan.json")
    if not plan_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Stage 13 checkpoint is incomplete: storage_plan.npz/json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    site_metadata = {int(item["site_id"]): item for item in metadata.get("sites", [])}
    with np.load(plan_path, allow_pickle=False) as payload:
        rows = np.asarray(payload["storage_sites"], dtype=np.float32).reshape(-1, 9)
        sites = tuple(_storage_site_from_row(row, site_metadata.get(int(row[0]), {})) for row in rows)
        plan = StoragePlanStore(
            timestamps=payload["timestamps"].copy(),
            load_bus_ids=payload["load_bus_ids"].copy(),
            assigned_site_ids=payload["assigned_site_ids"].copy(),
            site_support_requirement_mw=payload["site_support_requirement_mw"].copy(),
            sites=sites,
        )
    return static_maps, topology, electrical, power_flow, plan


def _storage_site_from_row(row: np.ndarray, metadata: dict[str, object]) -> StorageSite:
    return StorageSite(
        site_id=int(row[0]),
        bus_id=int(row[1]),
        row=int(row[2]),
        col=int(row[3]),
        covered_bus_ids=tuple(int(value) for value in metadata.get("covered_bus_ids", [])),
        power_mw=float(row[4]),
        energy_mwh=float(row[5]),
        initial_soc_mwh=float(row[6]),
        need_score=float(row[7]),
    )


def _bus_from_row(row: np.ndarray, metadata: dict[str, object]) -> GridBus:
    return GridBus(
        bus_id=int(row[0]),
        kind=str(metadata.get("kind", "transit_bus")),
        row=int(row[1]),
        col=int(row[2]),
        x=float(row[3]),
        y=float(row[4]),
        capacity_mw=float(row[5]),
        suitability=float(row[6]),
        externality_score=float(row[7]),
        source_kind=str(metadata.get("source_kind", "cached")),
        source_id=int(metadata.get("source_id", row[8])),
    )


def _edge_from_row(
    row: np.ndarray,
    bus_by_id: dict[int, GridBus],
    path: tuple[tuple[int, ...], tuple[int, ...]] | None,
) -> GridEdge:
    first, second = int(row[1]), int(row[2])
    if path is None:
        path = (
            (int(bus_by_id[first].row), int(bus_by_id[second].row)),
            (int(bus_by_id[first].col), int(bus_by_id[second].col)),
        )
    return GridEdge(
        edge_id=int(row[0]),
        from_bus=first,
        to_bus=second,
        length_km=float(row[3]),
        route_cost=float(row[4]),
        is_redundant=bool(row[5] >= 0.5),
        path_rows=path[0],
        path_cols=path[1],
    )


def _edge_path_metadata(path: Path) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    if not path.exists():
        return {}
    return {
        int(item["edge_id"]): (
            tuple(int(value) for value in item["path_rows"]),
            tuple(int(value) for value in item["path_cols"]),
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
    }


def _electrical_from_arrays(
    bus_rows: np.ndarray,
    branch_rows: np.ndarray,
    bus_by_id: dict[int, GridBus],
) -> GridElectricalState:
    buses = tuple(
        BusElectricalParam(
            bus_id=int(row[0]),
            kind=bus_by_id[int(row[0])].kind,
            nominal_kv=float(row[1]),
            p_capacity_mw=float(row[2]),
            q_capacity_mvar=float(row[3]),
            base_load_mw=float(row[4]),
            power_factor=float(row[5]),
            voltage_setpoint_pu=float(row[6]),
            control_mode="cached",
        )
        for row in np.atleast_2d(bus_rows)
    )
    branches = tuple(
        BranchElectricalParam(
            edge_id=int(row[0]),
            from_bus=int(row[1]),
            to_bus=int(row[2]),
            nominal_kv=float(row[3]),
            length_km=float(row[4]),
            r_ohm=float(row[5]),
            x_ohm=float(row[6]),
            b_us=float(row[7]),
            rate_mva=float(row[8]),
            is_redundant=bool(row[9] >= 0.5),
        )
        for row in np.atleast_2d(branch_rows)
    )
    return GridElectricalState(bus_params=buses, branch_params=branches)
