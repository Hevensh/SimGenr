"""Asset information boundaries: fixed, independently preplanned, or oracle.

S: scenario design/controller. All dispatches may see their whole declared
operation window; fixed assets do not imply a causal online controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from world_generator.core.config import SourceLoadConfig, WeatherConfig, WorldConfig
from world_generator.core.datatypes import SourceLoadForecastStore, StoragePlanStore, StorageSite
from world_generator.core.random_state import build_rng_registry, derive_module_seed
from world_generator.operation.grid_update_loop import GridUpdateIteration, GridUpdateLoopResult, run_grid_update_loop
from world_generator.operation.grid_upgrade import build_grid_upgrade_plan
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.source_load_forecast import generate_source_load_forecast
from world_generator.operation.storage_dispatch import dispatch_storage_week
from world_generator.operation.storage_planning import analyze_storage_need, plan_storage_sites
from world_generator.weather.weather_generator import generate_daily_weather, generate_hourly_weather_week


@dataclass(frozen=True)
class AssetPlanningResult:
    update_loop: object
    storage_need: object
    storage_plan: object
    final_topology: object
    final_electrical: object
    final_power_flow: object
    runtime_source: SourceLoadForecastStore
    storage_dispatch: object
    dispatched_forecast: object
    dispatched_power_flow: object
    dispatched_electrical: object
    checkpoint_iteration: object
    metadata: dict[str, object]


def asset_snapshot_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Deterministic content hash, independent of NPZ ZIP timestamps."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name])
        if values.dtype.hasobject:
            raise ValueError("Asset hashes never accept pickle/object arrays")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def thermal_land_limits_from_ledger(ledger: np.ndarray | None, topology: object) -> dict[int, float]:
    """D reserved-footprint capacity limits, keyed by persistent thermal bus ID."""
    thermal = {int(bus.bus_id): float(bus.capacity_mw) for bus in topology.refined_buses if bus.kind == "thermal_bus"}
    if ledger is None:
        if thermal:
            raise ValueError("Asset planning requires the Stage 9 thermal land ledger")
        return {}
    values = np.asarray(ledger)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError("Thermal land ledger must be finite [project,9]")
    limits = {int(row[0]): float(row[7]) for row in values}
    if len(limits) != len(values) or set(limits) != set(thermal) or np.any(values[:, 0] != np.rint(values[:, 0])):
        raise ValueError("Thermal land ledger IDs must exactly match thermal assets")
    if any(limits[bus_id] < capacity - 1e-5 for bus_id, capacity in thermal.items()):
        raise ValueError("Initial thermal capacity already exceeds reserved land capacity")
    return limits


def restore_cached_thermal_land_boundary(topology: object, electrical: object, ledger: np.ndarray | None) -> tuple[object, object]:
    """Recover a land-saturated capacity rounded upward by float32 asset export.

    The authoritative D ledger retains float64 area-derived limits. This is
    restricted to a demonstrable half-ULP export round-off, not a dispatch
    feasibility tolerance: any other overshoot is rejected unchanged.
    """
    if ledger is None:
        thermal_land_limits_from_ledger(ledger, topology)
        return topology, electrical
    values = np.asarray(ledger)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError("Thermal land ledger must be finite [project,9]")
    limits = {int(row[0]): float(row[7]) for row in values}

    def recovered(bus_id: int, capacity: float) -> float:
        if bus_id not in limits:
            raise ValueError("Cached thermal asset is missing its land limit")
        limit = limits[bus_id]
        if capacity <= limit:
            return capacity
        rounded = np.float32(limit)
        half_ulp = 0.5 * abs(float(np.spacing(rounded)))
        if capacity != float(rounded) or capacity - limit > half_ulp:
            raise ValueError("Cached thermal capacity exceeds land limit beyond float32 export rounding")
        return limit

    buses = tuple(replace(bus, capacity_mw=recovered(int(bus.bus_id), float(bus.capacity_mw)))
                  if bus.kind == "thermal_bus" else bus for bus in topology.refined_buses)
    params = tuple(replace(bus, p_capacity_mw=recovered(int(bus.bus_id), float(bus.p_capacity_mw)))
                   if bus.kind == "thermal_bus" else bus for bus in electrical.bus_params)
    restored = replace(topology, refined_buses=buses)
    thermal_land_limits_from_ledger(ledger, restored)
    return restored, replace(electrical, bus_params=params)


def fixed_storage_plan(topology: object, timestamps: np.ndarray, config: WorldConfig) -> StoragePlanStore:
    """No weather input: explicit configured sites, or the default empty plan."""
    by_id = {int(bus.bus_id): bus for bus in topology.refined_buses}
    load_ids = np.asarray([bus.bus_id for bus in topology.refined_buses if bus.kind == "load_bus"], dtype=np.int32)
    sites = []
    assignment = np.full(len(load_ids), -1, dtype=np.int32)
    for index, item in enumerate(config.planning.fixed_storage_sites):
        bus_id = int(item["bus_id"])
        if bus_id not in by_id:
            raise ValueError(f"Configured storage bus {bus_id} is absent from fixed assets")
        fraction = float(item.get("initial_soc_fraction", config.storage.initial_soc_fraction))
        if not config.storage.minimum_soc_fraction <= fraction <= config.storage.maximum_soc_fraction:
            raise ValueError("Fixed storage initial SOC must fit configured operating bounds")
        site_id = int(item.get("site_id", index))
        covered = (bus_id,) if bus_id in load_ids else ()
        if covered:
            assignment[load_ids == bus_id] = site_id
        bus = by_id[bus_id]
        sites.append(StorageSite(site_id, bus_id, int(bus.row), int(bus.col), covered,
                                 float(item["power_mw"]), float(item["energy_mwh"]), fraction * float(item["energy_mwh"]), 0.0))
    return StoragePlanStore(np.asarray(timestamps).copy(), load_ids, assignment,
                            np.zeros((len(timestamps), len(sites)), dtype=np.float32), tuple(sites))


def operation_branch_mask(electrical: object, timestamps: np.ndarray, faults: tuple[dict[str, int], ...]) -> np.ndarray:
    """Faults change interval service status, never the underlying asset list."""
    by_id = {int(branch.edge_id): i for i, branch in enumerate(electrical.branch_params)}
    mask = np.ones((len(timestamps), len(by_id)), dtype=bool)
    for event in faults:
        start, end = int(event["start_offset_hours"]), int(event["start_offset_hours"] + event["duration_hours"])
        if event["branch_id"] not in by_id or start < 0 or end > len(timestamps):
            raise ValueError("Line fault ID/window is outside the declared operation assets/time axis")
        mask[start:end, by_id[int(event["branch_id"])]] = False
    return mask


def align_source_to_assets(source: SourceLoadForecastStore, topology: object, electrical: object) -> SourceLoadForecastStore:
    """Keep the operation clock; add zero transit columns and scale thermal A.

    The original Stage 11 artifact stays immutable. Thermal availability is
    rescaled by installed/original nameplate, preserving its hourly fraction.
    """
    original = {int(bus_id): i for i, bus_id in enumerate(source.bus_ids)}
    target = {int(bus.bus_id): i for i, bus in enumerate(topology.refined_buses)}
    shape = (len(source.timestamps), len(target))
    # Dispatch uses float64 installed capacities. A float32 in-place product
    # can round above the exact capacity and create false excess availability.
    arrays = [np.zeros(shape, dtype=np.float64) for _ in range(4)]
    incoming = (source.p_load_mw, source.p_gen_available_mw, source.p_gen_scheduled_mw, source.q_load_mvar)
    params = {int(param.bus_id): param for param in electrical.bus_params}
    for bus_id, position in original.items():
        if bus_id not in target:
            if any(np.any(values[:, position] != 0) for values in incoming):
                raise ValueError("Planning removed a bus carrying nonzero operation source/load")
            continue
        index = target[bus_id]
        for output, values in zip(arrays, incoming):
            output[:, index] = values[:, position]
        if source.bus_kinds[position] == "thermal_bus":
            installed = float(params[bus_id].p_capacity_mw)
            if source.nameplate_capacity_mw is not None:
                base = float(source.nameplate_capacity_mw[position])
            else:
                # Legacy source has no nameplate appendix. Preserve its
                # absolute availability; a window maximum is not nameplate.
                base = installed
            if base <= 0 and installed > 0:
                raise ValueError("Cannot infer availability of a zero-base thermal asset")
            if source.nameplate_capacity_mw is not None:
                available_fraction = np.asarray(incoming[1][:, position], dtype=np.float64) / base if base > 0 else np.zeros(shape[0])
                planned_fraction = np.asarray(incoming[2][:, position], dtype=np.float64) / base if base > 0 else np.zeros(shape[0])
                if np.any(available_fraction < 0) or np.any(available_fraction > 1):
                    raise ValueError("Original thermal availability exceeds its declared nameplate")
                arrays[1][:, index] = installed * available_fraction
                arrays[2][:, index] = installed * planned_fraction
    return SourceLoadForecastStore(source.timestamps.copy(), np.asarray(list(target), dtype=np.int32),
        tuple(bus.kind for bus in topology.refined_buses), *arrays, source.source_channels,
        data_semantics="synthetic_realization_rescaled_to_frozen_assets")


def _retime_storage_plan(plan: StoragePlanStore, timestamps: np.ndarray) -> StoragePlanStore:
    return replace(plan, timestamps=np.asarray(timestamps).copy(),
                   site_support_requirement_mw=np.zeros((len(timestamps), len(plan.sites)), dtype=np.float32))


def installed_assets(topology: object, electrical: object, storage_plan: StoragePlanStore, dispatch: object, config: WorldConfig) -> tuple[object, StoragePlanStore]:
    params = {int(param.bus_id): param for param in electrical.bus_params}
    installed_topology = replace(topology, refined_buses=tuple(
        replace(bus, capacity_mw=float(params[int(bus.bus_id)].p_capacity_mw)) if bus.kind == "thermal_bus" else bus
        for bus in topology.refined_buses))
    index = {int(site_id): i for i, site_id in enumerate(dispatch.site_ids)}
    sites = tuple(replace(site, power_mw=float(dispatch.site_power_capacity_mw[index[site.site_id]]),
                          energy_mwh=float(dispatch.site_energy_capacity_mwh[index[site.site_id]]),
                          initial_soc_mwh=(site.initial_soc_mwh / site.energy_mwh if site.energy_mwh > 0 else config.storage.initial_soc_fraction)
                          * float(dispatch.site_energy_capacity_mwh[index[site.site_id]])) for site in storage_plan.sites)
    return installed_topology, replace(storage_plan, sites=sites)


def storage_initial_boundary(plan: StoragePlanStore, config: WorldConfig) -> dict[int, float] | None:
    """Configured boundary inventory only; cyclic SOC remains a runtime optimum."""
    return None if config.storage.cyclic_state_of_charge else {int(site.site_id): float(site.initial_soc_mwh) for site in plan.sites}


def asset_snapshot(topology: object, electrical: object, plan: StoragePlanStore, land_limits: dict[int, float]) -> dict[str, np.ndarray]:
    # Canonical precision matches the existing Stage12/13 NPZ asset matrices,
    # so a cache round-trip does not create a false asset intervention.
    arrays = electrical.as_arrays()
    return {
        "bus_ids": np.asarray([bus.bus_id for bus in topology.refined_buses], dtype=np.int32),
        "bus_nameplate_capacity_mw": np.asarray([bus.capacity_mw for bus in topology.refined_buses], dtype=np.float32),
        "branch_ids": np.asarray([branch.edge_id for branch in electrical.branch_params], dtype=np.int32),
        "branch_capacity_mva": np.asarray([branch.rate_mva for branch in electrical.branch_params], dtype=np.float32),
        "electrical_buses": arrays["electrical_buses"], "electrical_branches": arrays["electrical_branches"],
        "refined_grid_buses": topology.as_arrays()["refined_grid_buses"],
        "storage_site_ids": np.asarray([site.site_id for site in plan.sites], dtype=np.int32),
        "storage_bus_ids": np.asarray([site.bus_id for site in plan.sites], dtype=np.int32),
        "storage_power_mw": np.asarray([site.power_mw for site in plan.sites], dtype=np.float32),
        "storage_energy_mwh": np.asarray([site.energy_mwh for site in plan.sites], dtype=np.float32),
        "storage_initial_soc_mwh": np.asarray([site.initial_soc_mwh for site in plan.sites], dtype=np.float32),
        "thermal_bus_ids": np.asarray(sorted(land_limits), dtype=np.int32),
        "thermal_land_capacity_upper_bound_mw": np.asarray([land_limits[i] for i in sorted(land_limits)], dtype=np.float32),
    }


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return asset_snapshot_sha256(arrays)


def _clock_bounds(timestamps: np.ndarray) -> list[float]:
    return [float(timestamps[0]), float(timestamps[-1] + 1)]


def run_asset_planning(*, config: WorldConfig, terrain: object, hydrology: object, climate: object,
                       land: object, land_use: object, topology: object, base_topology: object,
                       electrical: object, operation_source: SourceLoadForecastStore,
                       thermal_land_limits_mw: dict[int, float], output_dir: Path) -> AssetPlanningResult:
    """Run a genuine asset policy, then the operation interval on its own axis."""
    output_dir = Path(output_dir)
    mode = config.planning.mode
    initial_plan = fixed_storage_plan(topology, operation_source.timestamps, config)
    initial = asset_snapshot(topology, electrical, initial_plan, thermal_land_limits_mw)
    initial_hash = _save_arrays(output_dir / "initial_assets.npz", initial)
    input_hashes: dict[str, str] = {}
    planning_bounds = None
    planning_seed = None
    design_config_snapshot = None
    design_loop = None
    if mode == "fixed_assets":
        final_topology, final_electrical, plan = topology, electrical, initial_plan
        input_source = "explicit_static_assets_and_configured_storage_sites_no_weather_planning"
    else:
        if mode == "preplanned":
            planning_seed = derive_module_seed(config.seed, f"planning:{config.planning.design_seed}")
            registry = build_rng_registry(planning_seed)
            design_weather_config = WeatherConfig(days=config.planning.design_days, hourly_week_days=config.planning.design_days,
                start_day_of_year=config.planning.design_start_day_of_year, **config.planning.design_weather_overrides)
            design_daily = generate_daily_weather(terrain, hydrology, climate, config.world, design_weather_config, registry.generator("weather"))
            design_weather = generate_hourly_weather_week(design_daily, design_weather_config, registry.generator("weather_hourly"), grid=config.world)
            design_source_config = SourceLoadConfig(**config.planning.design_source_load_overrides)
            design_config_snapshot = {"weather": asdict(design_weather_config), "source_load": asdict(design_source_config)}
            design_source = generate_source_load_forecast(design_weather, topology, electrical, registry.generator("source_load"),
                config=design_source_config, land_use=land_use, grid=config.world)
            for name, arrays in (("daily_weather", design_daily.as_arrays()), ("hourly_weather", design_weather.as_arrays()), ("source_load_forecast", design_source.as_arrays())):
                input_hashes[name] = _save_arrays(output_dir / "design_input" / f"{name}.npz", arrays)
            input_source = "independent_generated_design_scenario_same_static_world_separate_rng_and_configs"
        else:
            design_source = operation_source
            input_hashes["operation_source_used_as_design"] = asset_snapshot_sha256(operation_source.as_arrays())
            input_source = "oracle_complete_operation_window_realized_source_load"
        planning_bounds = _clock_bounds(design_source.timestamps)
        design_loop = run_grid_update_loop(design_source, topology, base_topology, electrical, terrain, hydrology, land, config.world, config.power_grid)
        iteration = design_loop.iterations[-1]
        final_topology, final_electrical = iteration.refined_topology, iteration.electrical
        need = analyze_storage_need(final_topology, final_electrical, iteration.power_flow, config.world, config.storage)
        plan = plan_storage_sites(final_topology, need, config.storage)
        if mode == "preplanned":
            designed_source = align_source_to_assets(design_source, final_topology, final_electrical)
            design_dispatch = dispatch_storage_week(final_topology, final_electrical, iteration.power_flow, plan, config.storage,
                source_forecast=designed_source, thermal_land_limits_mw=thermal_land_limits_mw,
                initial_soc_mwh_by_site_id=storage_initial_boundary(plan, config))
            final_electrical = design_dispatch[3]
            final_topology, plan = installed_assets(final_topology, final_electrical, plan, design_dispatch[0], config)
            _save_arrays(output_dir / "design_result" / "storage_dispatch.npz", design_dispatch[0].as_arrays())
            _save_arrays(output_dir / "design_result" / "storage_need.npz", need.as_arrays())
            _save_arrays(output_dir / "design_result" / "storage_plan.npz", plan.as_arrays())
            _save_arrays(output_dir / "design_result" / "grid_electrical.npz", final_electrical.as_arrays())
    if mode != "full_window_planning":
        plan = _retime_storage_plan(plan, operation_source.timestamps)
    frozen_before = asset_snapshot(final_topology, final_electrical, plan, thermal_land_limits_mw)
    runtime_source = align_source_to_assets(operation_source, final_topology, final_electrical)
    service = operation_branch_mask(final_electrical, operation_source.timestamps, config.planning.line_faults)
    runtime_baseline = solve_dc_power_flow(runtime_source, final_topology, final_electrical, branch_in_service=service)
    # This is a diagnostic of operation vulnerability, not an input to fixed
    # or preplanned site choice. Their plan is already fixed above this line.
    storage_need = analyze_storage_need(final_topology, final_electrical, runtime_baseline, config.world, config.storage)
    runtime = dispatch_storage_week(final_topology, final_electrical, runtime_baseline, plan, config.storage,
        source_forecast=runtime_source, fixed_capacity=mode != "full_window_planning", branch_in_service=service,
        thermal_land_limits_mw=thermal_land_limits_mw, initial_soc_mwh_by_site_id=storage_initial_boundary(plan, config))
    installed_topology, installed_plan = installed_assets(final_topology, runtime[3], plan, runtime[0], config)
    frozen_after = asset_snapshot(installed_topology, runtime[3], installed_plan, thermal_land_limits_mw)
    if mode != "full_window_planning" and asset_snapshot_sha256(frozen_before) != asset_snapshot_sha256(frozen_after):
        raise ValueError("Fixed-capacity operation changed frozen assets")
    frozen_hash = _save_arrays(output_dir / "frozen_assets.npz", frozen_after)
    metadata = {
        "schema_version": "asset_planning_v1", "mode": mode,
        "initial_assets_sha256": initial_hash, "frozen_assets_sha256": frozen_hash,
        "planning_input_source": input_source, "planning_input_hashes": input_hashes,
        "planning_time_bounds_hours": planning_bounds, "operation_time_bounds_hours": _clock_bounds(operation_source.timestamps),
        "planning_time_axis": "independent_design_climatology_not_operation_clock" if mode == "preplanned" else "operation_clock" if mode == "full_window_planning" else None,
        "planning_information_available_at_operation_hour": float(operation_source.timestamps[0]) if mode != "full_window_planning" else float(operation_source.timestamps[-1] + 1.0),
        "derived_planning_seed": planning_seed, "design_seed_parameter": config.planning.design_seed if mode == "preplanned" else None,
        "design_config_snapshot": design_config_snapshot,
        "dispatch_foresight": "full_operation_window_perfect_foresight_dispatch",
        "initial_state_policy": "optimized_periodic_runtime" if config.storage.cyclic_state_of_charge else "fixed_prior_boundary",
        "storage_snapshot_initial_soc_semantics": "configured_prior_boundary_not_periodic_runtime_optimum",
        "asset_numeric_precision": "float32_matches_stage12_stage13_npz",
        "operation_capacity_policy": "oracle_joint_capacity_and_dispatch" if mode == "full_window_planning" else "frozen_capacities_no_expansion",
        "storage_need_use": "operation_diagnostic_not_asset_planning_input" if mode != "full_window_planning" else "oracle_design_and_operation",
        "faults": list(config.planning.line_faults), "fault_semantics": "temporary_branch_in_service_mask_asset_ids_unchanged",
    }
    (output_dir / "asset_boundary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    upgrade = build_grid_upgrade_plan(runtime_baseline, final_electrical, power_grid=config.power_grid)
    actions = () if design_loop is None else design_loop.iterations[-1].actions
    checkpoint = GridUpdateIteration(0 if mode == "fixed_assets" else 1, final_topology, final_electrical,
        runtime_baseline, upgrade, actions, runtime_baseline.summary_dict())
    loop = GridUpdateLoopResult(runtime_baseline.summary_dict(), (checkpoint,))
    return AssetPlanningResult(loop, storage_need, plan, final_topology, final_electrical, runtime_baseline,
        runtime_source, *runtime, checkpoint, metadata)
