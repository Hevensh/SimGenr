from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_generator.core.output_layout import WorldDataLayout
from world_generator.core.contracts import interval_bounds_hours
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
    HydrologyTimeSeriesStore,
    SourceLoadForecastStore,
    StorageDispatchStore,
)


def load_asset_boundary_checkpoint(output_dir: Path, *, expected_timestamps: np.ndarray | None = None,
                                   expected_bus_ids: np.ndarray | None = None,
                                   expected_branch_ids: np.ndarray | None = None) -> dict[str, object]:
    """Require F's original/frozen design assets and check their content hashes."""
    from world_generator.core.contracts import entity_ids, interval_bounds_hours
    from world_generator.core.operation_contracts import PLANNING_MODES
    from world_generator.operation.asset_planning import asset_snapshot_sha256
    directory = output_dir / "data" / "planning"
    paths = [directory / name for name in ("initial_assets.npz","frozen_assets.npz","asset_boundary.json")]
    if not all(path.exists() for path in paths):
        raise ValueError("Checkpoint predates F asset boundaries; regenerate with --from-stage 1")
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    if not isinstance(metadata,dict) or metadata.get("schema_version") != "asset_planning_v1" or metadata.get("mode") not in PLANNING_MODES:
        raise ValueError("Unsupported F asset planning boundary metadata")
    required_metadata = {"initial_assets_sha256","frozen_assets_sha256","planning_input_hashes","planning_time_bounds_hours",
                         "operation_time_bounds_hours","planning_input_source","dispatch_foresight"}
    if not required_metadata.issubset(metadata) or not isinstance(metadata["planning_input_hashes"],dict):
        raise ValueError("Incomplete F asset planning information boundary")
    def check_bounds(value: object, name: str) -> np.ndarray:
        bounds = np.asarray(value,dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
            raise ValueError(f"{name} must contain finite increasing hour boundaries")
        return bounds
    operation_bounds = check_bounds(metadata["operation_time_bounds_hours"],"operation_time_bounds_hours")
    design_bounds = metadata["planning_time_bounds_hours"]
    if metadata["mode"] == "fixed_assets":
        if design_bounds is not None:
            raise ValueError("Fixed assets must not claim an operation-derived planning period")
    elif design_bounds is None:
        raise ValueError("Planning mode must declare its information time bounds")
    else:
        design_bounds = check_bounds(design_bounds,"planning_time_bounds_hours")
        if metadata["mode"] == "preplanned":
            if metadata.get("planning_time_axis") == "independent_design_climatology_not_operation_clock":
                available_at = metadata.get("planning_information_available_at_operation_hour")
                if not isinstance(available_at,(int,float)) or not np.isfinite(available_at) or available_at > operation_bounds[0]:
                    raise ValueError("Independent design information must be available before operation")
            elif design_bounds[1] > operation_bounds[0]:
                raise ValueError("Preplanned input on the operation clock must end before operation")
    if expected_timestamps is not None:
        expected = interval_bounds_hours(expected_timestamps,1.0,stamp_unit="hour")
        if not np.array_equal(operation_bounds,expected[[0,-1],[0,1]]):
            raise ValueError("Asset boundary operation period differs from current timestamps")
    if metadata["mode"] == "preplanned":
        input_paths = {name:directory / "design_input" / f"{name}.npz" for name in ("daily_weather","hourly_weather","source_load_forecast")}
    elif metadata["mode"] == "full_window_planning":
        input_paths = {"operation_source_used_as_design":output_dir / "data" / "stage_11_operation" / "source_load_forecast.npz"}
    else:
        input_paths = {}
    if set(metadata["planning_input_hashes"]) != set(input_paths):
        raise ValueError("Planning input hash identities differ from the declared asset mode")
    for name,path in input_paths.items():
        if not path.exists():
            raise ValueError(f"Planning input artifact is missing: {name}")
        with np.load(path,allow_pickle=False) as payload:
            inputs = {key:payload[key].copy() for key in payload.files}
        if asset_snapshot_sha256(inputs) != metadata["planning_input_hashes"][name]:
            raise ValueError(f"Planning input hash mismatch: {name}")
    required = {"bus_ids","bus_nameplate_capacity_mw","branch_ids","branch_capacity_mva","electrical_buses","electrical_branches",
                "storage_site_ids","storage_bus_ids","storage_power_mw","storage_energy_mwh","storage_initial_soc_mwh",
                "thermal_bus_ids","thermal_land_capacity_upper_bound_mw"}
    result = {"metadata":metadata}
    for label,path in zip(("initial_assets","frozen_assets"),paths[:2]):
        with np.load(path,allow_pickle=False) as payload:
            arrays = {name:payload[name].copy() for name in payload.files}
        if not required.issubset(arrays) or set(arrays)-required-{"refined_grid_buses"}:
            raise ValueError(f"Incomplete or unknown {label} snapshot fields")
        for name,value in arrays.items():
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"Asset snapshot {name} must contain finite numeric data")
        buses = entity_ids(arrays["bus_ids"],f"{label} bus_ids")
        branches = entity_ids(arrays["branch_ids"],f"{label} branch_ids")
        sites = entity_ids(arrays["storage_site_ids"],f"{label} storage_site_ids")
        thermals = entity_ids(arrays["thermal_bus_ids"],f"{label} thermal_bus_ids")
        for name,size in {"bus_nameplate_capacity_mw":len(buses),"branch_capacity_mva":len(branches),
                          "storage_bus_ids":len(sites),"storage_power_mw":len(sites),"storage_energy_mwh":len(sites),
                          "storage_initial_soc_mwh":len(sites),"thermal_land_capacity_upper_bound_mw":len(thermals)}.items():
            if arrays[name].shape != (size,) or np.any(arrays[name] < 0):
                raise ValueError(f"Asset {name} has invalid shape or negative values")
        if not set(arrays["storage_bus_ids"]).issubset(set(buses)) or not set(thermals).issubset(set(buses)):
            raise ValueError("Storage/thermal asset IDs must refer to snapshot buses")
        for name,count,ids in (("electrical_buses",len(buses),buses),("electrical_branches",len(branches),branches)):
            if arrays[name].ndim != 2 or len(arrays[name]) != count or arrays[name].shape[1] == 0 or not np.array_equal(arrays[name][:,0],ids):
                raise ValueError(f"{name} rows must match fixed entity IDs")
        if np.any(arrays["storage_initial_soc_mwh"] > arrays["storage_energy_mwh"] + 1e-6):
            raise ValueError("Asset initial SOC exceeds storage energy capacity")
        if asset_snapshot_sha256(arrays) != metadata[f"{label}_sha256"]:
            raise ValueError(f"Asset snapshot hash mismatch: {label}")
        result[label] = arrays
    frozen = result["frozen_assets"]
    if expected_bus_ids is not None and not np.array_equal(frozen["bus_ids"],expected_bus_ids):
        raise ValueError("Frozen asset bus IDs differ from current operation")
    if expected_branch_ids is not None and not np.array_equal(frozen["branch_ids"],expected_branch_ids):
        raise ValueError("Frozen asset branch IDs differ from current operation")
    return result


def load_operation_checkpoint(output_dir: Path, *, expected_timestamps: np.ndarray | None = None,
                              require_modern: bool = False) -> tuple[PowerFlowStore, StorageDispatchStore]:
    """Read persisted runtime topology and physical dispatch without rerolling outages."""
    layout = WorldDataLayout(output_dir / "data")
    with np.load(layout.storage_dispatch / "storage_dispatch_power_flow.npz",allow_pickle=False) as payload:
        power = PowerFlowStore.from_arrays({name:payload[name].copy() for name in payload.files})
    with np.load(layout.storage_dispatch / "storage_dispatch.npz",allow_pickle=False) as payload:
        storage = StorageDispatchStore.from_arrays({name:payload[name].copy() for name in payload.files})
    if require_modern and (not power.operation_arrays or not storage.operation_arrays):
        raise ValueError("Checkpoint lacks F runtime operation state; regenerate with --from-stage 1")
    if not np.array_equal(power.timestamps,storage.timestamps) or (expected_timestamps is not None and not np.array_equal(power.timestamps,expected_timestamps)):
        raise ValueError("Runtime operation checkpoints have inconsistent timestamps")
    if not np.array_equal(power.branch_ids,storage.branch_ids):
        raise ValueError("Runtime branch IDs differ between power flow and storage")
    if storage.operation_arrays and not np.array_equal(storage.operation_arrays["bus_ids"],power.bus_ids):
        raise ValueError("Runtime bus IDs differ between physical and storage accounts")
    return power, storage


def load_source_load_checkpoint(
    output_dir: Path, *, expected_timestamps: np.ndarray | None = None,
    expected_grid_shape: tuple[int, int] | None = None,
) -> SourceLoadForecastStore:
    """Load Stage11 exogenous power and, when declared, its complete E appendix."""
    from world_generator.core.source_load_contracts import SOURCE_LOAD_SCHEMA_VERSION, SOURCE_LOAD_MODE

    layout = WorldDataLayout(output_dir / "data")
    path = layout.existing(layout.operation, "source_load_forecast.npz", layout.root / "source_load_forecast.npz")
    with np.load(path, allow_pickle=False) as payload:
        store = SourceLoadForecastStore.from_arrays({name: payload[name].copy() for name in payload.files})
    metadata = json.loads(layout.metadata.read_text(encoding="utf-8")) if layout.metadata.exists() else {}
    declaration = metadata.get("source_load_appendix")
    if declaration is not None:
        if not isinstance(declaration, dict) or declaration.get("schema_version") != SOURCE_LOAD_SCHEMA_VERSION or declaration.get("mode") != SOURCE_LOAD_MODE or declaration.get("artifact") != "stage_11_operation/source_load_forecast.npz":
            raise ValueError("Unsupported source/load appendix declaration")
        if store.nameplate_capacity_mw is None:
            raise ValueError("Declared source/load appendix is missing from its checkpoint")
    if expected_timestamps is not None and not np.array_equal(store.timestamps, expected_timestamps):
        raise ValueError("Source/load checkpoint timestamps differ from hourly weather")
    if expected_grid_shape is not None and store.nameplate_capacity_mw is not None:
        if tuple(store.metadata["weather_grid_shape"]) != tuple(expected_grid_shape):
            raise ValueError("Source/load sampling grid differs from hourly weather")
    return store


def load_dynamic_hydrology_checkpoint(
    output_dir: Path, *, expected_timestamps: np.ndarray | None = None,
    expected_grid_shape: tuple[int, int] | None = None,
) -> HydrologyTimeSeriesStore | None:
    """Read declared D output; legacy/static worlds do not acquire fake flows."""
    from world_generator.core.hydrology_contracts import HYDROLOGY_MODE, HYDROLOGY_SCHEMA_VERSION

    layout = WorldDataLayout(output_dir / "data")
    metadata = json.loads(layout.metadata.read_text(encoding="utf-8")) if layout.metadata.exists() else {}
    declaration = metadata.get("dynamic_hydrology")
    path = layout.dynamic_hydrology / "hourly_hydrology.npz"
    if declaration is None:
        if path.exists():
            raise ValueError("Undeclared dynamic hydrology artifact in legacy metadata")
        return None
    if not isinstance(declaration, dict):
        raise ValueError("Dynamic hydrology declaration must be a mapping")
    mode = declaration.get("mode")
    if declaration.get("schema_version") != HYDROLOGY_SCHEMA_VERSION:
        raise ValueError("Unsupported dynamic hydrology metadata schema")
    if mode == "static_only":
        if path.exists() or declaration.get("artifact") is not None:
            raise ValueError("Static-only hydrology cannot include a dynamic artifact")
        return None
    if mode != HYDROLOGY_MODE or declaration.get("artifact") != "dynamic_hydrology/hourly_hydrology.npz":
        raise ValueError("Unsupported dynamic hydrology mode or artifact path")
    if not path.exists():
        raise FileNotFoundError(f"Dynamic hydrology checkpoint is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        store = HydrologyTimeSeriesStore.from_arrays({name: payload[name].copy() for name in payload.files})
    if expected_timestamps is not None and not np.array_equal(store.timestamps, expected_timestamps):
        raise ValueError("Dynamic hydrology timestamps do not match hourly weather")
    if expected_grid_shape is not None and store.states["soil_storage_mm"].shape[1:] != expected_grid_shape:
        raise ValueError("Dynamic hydrology grid does not match the static world")
    return store


def validate_weather_checkpoint_time(payload: object) -> None:
    """Preserve and validate an on-disk interval declaration before decoding."""
    unit = str(payload["time_unit"])
    bounds = interval_bounds_hours(payload["timestamps"], 1.0 if unit == "hour" else 24.0, stamp_unit=unit)
    if "time_bounds_hours" in payload:
        declared = np.asarray(payload["time_bounds_hours"])
        if declared.shape != bounds.shape or not np.allclose(declared, bounds, rtol=0, atol=1e-9):
            raise ValueError("Weather checkpoint interval bounds differ from its timestamps")


def load_hourly_weather_checkpoint(output_dir: Path) -> WeatherStore:
    layout = WorldDataLayout(output_dir / "data")
    path = layout.existing(layout.weather, "hourly_weather_week.npz", layout.root / "hourly_weather_week.npz")
    if not path.exists():
        raise FileNotFoundError(f"Hourly weather checkpoint is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        validate_weather_checkpoint_time(payload)
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
        power_flow = PowerFlowStore.from_arrays({name:payload[name].copy() for name in payload.files})
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
