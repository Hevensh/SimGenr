from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from world_generator.core.output_layout import WorldDataLayout
from world_generator.core.contracts import entity_ids
from world_generator.operation.stage_cache import load_stage12_checkpoint, load_dynamic_hydrology_checkpoint, load_source_load_checkpoint, load_asset_boundary_checkpoint, load_operation_checkpoint, validate_weather_checkpoint_time


SCHEMA_VERSION = "0.10.0"

STATIC_CONTINUOUS_CHANNELS = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "roughness",
    "curvature",
    "flow_accumulation",
    "water_depth",
    "hydrology_elevation",
    "distance_to_water",
    "vegetation",
    "mean_temperature",
    "annual_temperature_amplitude",
    "mean_humidity",
    "prevailing_wind_u",
    "prevailing_wind_v",
    "mean_precipitation",
    "mean_cloud",
    "mean_irradiance",
    "population_density",
    "economic_activity",
    "urban_density",
)

STATIC_CATEGORICAL_CHANNELS = (
    "flow_direction",
    "water_type",
    "watershed_id",
    "land_cover",
    "urban_mask",
    "city_id_map",
    "land_use_zone",
)

NODE_OPERATION_CHANNELS = (
    "p_load_mw",
    "p_gen_available_mw",
    "p_gen_scheduled_mw",
    "q_load_mvar",
    "bus_angle_rad",
    "bus_p_injection_mw",
    "served_load_mw",
    "dispatched_generation_mw",
    "unserved_load_mw",
    "curtailed_generation_mw",
)

LINE_OPERATION_CHANNELS = (
    "line_flow_mw",
    "line_loading_ratio",
    "baseline_line_loading_ratio",
)

NODE_TYPE_TO_ID = {
    "load_bus": 0,
    "wind_bus": 1,
    "pv_bus": 2,
    "thermal_bus": 3,
    "transit_bus": 4,
}

STATIC_UNITS = {
    "elevation": "m",
    "slope": "rise_over_run",
    "water_depth": "m",
    "hydrology_elevation": "m",
    "population_density": "persons/km2",
    "flow_accumulation": "upstream_cell_count",
    "catchment_area_km2": "km2",
    "distance_to_water": "km",
    "mean_temperature": "degC",
    "annual_temperature_amplitude": "degC",
    "prevailing_wind_u": "m/s",
    "prevailing_wind_v": "m/s",
    "mean_precipitation": "mm/year",
    "mean_irradiance": "W/m2",
}

OPERATION_UNITS = {
    "p_load_mw": "MW",
    "p_gen_available_mw": "MW",
    "exogenous_p_load_mw": "MW",
    "exogenous_p_gen_available_mw": "MW",
    "exogenous_p_renewable_available_mw": "MW",
    "p_gen_scheduled_mw": "MW",
    "q_load_mvar": "Mvar",
    "bus_angle_rad": "rad",
    "bus_p_injection_mw": "MW",
    "served_load_mw": "MW",
    "dispatched_generation_mw": "MW",
    "unserved_load_mw": "MW",
    "curtailed_generation_mw": "MW",
    "line_flow_mw": "MW",
    "line_loading_ratio": "ratio",
    "charge_mw": "MW",
    "discharge_mw": "MW",
    "emergency_discharge_mw": "MW",
    "soc_mwh": "MWh",
}


def build_dataset(
    world_dirs: Iterable[Path],
    output_dir: Path,
    *,
    partitions: dict[int, str] | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_root = output_dir / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    world_paths = [Path(world_dir) for world_dir in world_dirs]
    iterator = tqdm(
        world_paths,
        desc="Packaging dataset",
        unit="sample",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    samples = [package_world(world_dir, sample_root, partitions=partitions) for world_dir in iterator]
    split_counts = {
        split: sum(sample["partition"] == split for sample in samples)
        for split in ("train", "val", "test", "inspection")
        if any(sample["partition"] == split for sample in samples)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sample_unit": "one generated world with one configured hourly operation period",
        "split_policy": "split by world seed; never split hours from one world across partitions",
        "sample_count": len(samples),
        "split_counts": split_counts,
        "samples": samples,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "schema.json", dataset_schema())
    (output_dir / "README.md").write_text(_dataset_readme(), encoding="utf-8")
    (output_dir / "inspection_summary.md").write_text(_inspection_summary(output_dir, samples), encoding="utf-8")
    return manifest


def package_world(
    world_dir: Path,
    sample_root: Path,
    *,
    partitions: dict[int, str] | None = None,
) -> dict[str, object]:
    data_dir = world_dir / "data"
    layout = WorldDataLayout(data_dir)
    world_metadata = json.loads(layout.metadata.read_text(encoding="utf-8")) if layout.metadata.exists() else {}
    static_path = layout.existing(layout.topology, "static_maps.npz", data_dir / "static_maps.npz")
    weather_path = layout.existing(layout.weather, "hourly_weather_week.npz", data_dir / "hourly_weather_week.npz")
    forecast_path = layout.existing(
        layout.storage_dispatch,
        "storage_dispatch_forecast.npz",
        data_dir / "storage_dispatch_forecast.npz",
    )
    power_flow_path = layout.existing(
        layout.storage_dispatch,
        "storage_dispatch_power_flow.npz",
        data_dir / "storage_dispatch_power_flow.npz",
    )
    storage_path = layout.existing(
        layout.storage_dispatch,
        "storage_dispatch.npz",
        data_dir / "storage_dispatch.npz",
    )
    electrical_path = layout.existing(
        layout.storage_dispatch,
        "storage_dispatch_electrical.npz",
        data_dir / "storage_dispatch_electrical.npz",
    )
    exogenous_path = layout.existing(layout.operation, "source_load_forecast.npz", data_dir / "source_load_forecast.npz")
    required = [
        static_path,
        weather_path,
        forecast_path,
        power_flow_path,
        storage_path,
        electrical_path,
        layout.config_snapshot,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if _has_physical_units(world_metadata) and not exogenous_path.exists():
        missing.append(str(exogenous_path))
    if missing:
        raise FileNotFoundError("World output is incomplete: " + ", ".join(missing))

    static_source = _load_npz(static_path)
    weather = _load_npz(weather_path)
    validate_weather_checkpoint_time(weather)
    forecast = _load_npz(forecast_path)
    power_flow = _load_npz(power_flow_path)
    storage = _load_npz(storage_path)
    electrical = _load_npz(electrical_path)
    exogenous = _load_npz(exogenous_path) if exogenous_path.exists() else None
    _, topology, _, _ = load_stage12_checkpoint(world_dir)

    static_continuous = _stack_channels(static_source, STATIC_CONTINUOUS_CHANNELS, np.float32)
    static_categorical = _categorical_channels(static_source)
    static_payload = {
        "continuous": static_continuous,
        "categorical": static_categorical,
        "continuous_channels": np.asarray(STATIC_CONTINUOUS_CHANNELS),
        "categorical_channels": np.asarray(STATIC_CATEGORICAL_CHANNELS),
    }
    if "catchment_area_km2" in static_source:
        static_payload["catchment_area_km2"] = static_source["catchment_area_km2"].copy()
    land_payload = _land_accounting_payload(layout, static_source, world_metadata)
    hydrology_store = load_dynamic_hydrology_checkpoint(world_dir, expected_timestamps=weather["timestamps"],
                                                       expected_grid_shape=static_source["elevation"].shape)
    hydrology_payload = hydrology_store.as_arrays() if hydrology_store is not None else {}
    source_load_store = load_source_load_checkpoint(world_dir, expected_timestamps=weather["timestamps"], expected_grid_shape=static_source["elevation"].shape) if exogenous is not None else None
    source_load_payload = source_load_store.as_arrays() if source_load_store is not None and source_load_store.nameplate_capacity_mw is not None else {}
    operation_detail, power_flow_detail, asset_payload = {}, {}, {}
    boundary_path = world_dir / "data" / "planning" / "asset_boundary.json"
    if boundary_path.exists() or world_metadata.get("asset_planning") is not None:
        boundary = load_asset_boundary_checkpoint(world_dir, expected_timestamps=weather["timestamps"],
                    expected_bus_ids=forecast["bus_ids"],expected_branch_ids=power_flow["branch_ids"])
        modern_power, modern_storage = load_operation_checkpoint(world_dir,expected_timestamps=weather["timestamps"],require_modern=True)
        operation_detail, power_flow_detail = modern_storage.as_arrays(), modern_power.as_arrays()
        asset_payload = {f"{group}__{name}":value for group in ("initial_assets","frozen_assets") for name,value in boundary[group].items()}
        asset_payload["boundary_metadata_json"] = np.asarray(json.dumps(boundary["metadata"],sort_keys=True))
        world_metadata["asset_planning"] = boundary["metadata"]
    elif "operation_schema_version" in storage or "operation_schema_version" in power_flow:
        raise ValueError("F runtime outputs require an explicit asset planning boundary")
    dynamic_payload = {
        "timestamps": weather["timestamps"].astype(np.int32),
        "weather": weather["dynamic"].astype(np.float32),
        "weather_class": weather["weather_class"].astype(np.int16),
        "weather_channels": weather["channel_names"],
        "time_unit": weather["time_unit"],
        "start_day_of_year": weather["start_day_of_year"].astype(np.int32),
    }
    for name, values in weather.items():
        if name.startswith("diagnostic__"):
            if values.shape != weather["weather_class"].shape or not np.isfinite(values).all():
                raise ValueError(f"Weather diagnostic {name!r} must be finite [T,H,W]")
            dynamic_payload[name] = values.astype(np.float32)
        elif name in {"time_bounds_hours", "static_elevation_m"}:
            dynamic_payload[name] = values.copy()

    graph_payload, graph_metadata = _graph_payload(topology, electrical)
    graph_metadata["capacity_sources"] = {
        "node_features.capacity_mw":"Stage12 pre-operation topology design capacity; retained legacy values",
        "node_electrical.p_capacity_mw":"Stage14 electrical bus capacity; may include storage interface capacity",
        "final_asset_capacity":"asset_planning.frozen_assets__bus_nameplate_capacity_mw is authoritative for the declared frozen bus asset snapshot" if asset_payload else "legacy_no_frozen_asset_boundary",
        "physical_thermal_capacity":"operation_detail.op__thermal_installed_capacity_mw excludes storage; thermal_bus_ids define applicability" if operation_detail else "legacy_no_separate_physical_generator_account",
    }
    operation_payload = _operation_payload(forecast, power_flow, storage, exogenous=exogenous)

    validation = _validate_modalities(
        static_continuous,
        static_categorical,
        weather,
        forecast,
        power_flow,
        storage,
        graph_payload,
    )
    config_path = layout.config_snapshot
    provenance = _world_provenance(world_metadata)
    static_units = dict(STATIC_UNITS)
    if not _has_physical_units(provenance):
        # Repackaging old outputs cannot magically convert their normalized
        # density index to persons/km2. Keep the original data and its meaning.
        static_units["population_density"] = "legacy_normalized_index"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": world_dir.name,
        "seed": world_metadata.get("seed", _seed_from_world_id(world_dir.name)),
        **provenance,
        "source_world": str(world_dir.as_posix()),
        "config_sha256": _sha256(config_path),
        "grid_shape": list(static_continuous.shape[1:]),
        "hours": int(weather["dynamic"].shape[0]),
        "static": {
            "units": static_units,
            "continuous_channels": list(STATIC_CONTINUOUS_CHANNELS),
            "categorical_channels": list(STATIC_CATEGORICAL_CHANNELS),
            "continuous_statistics": _channel_statistics(static_continuous, STATIC_CONTINUOUS_CHANNELS),
        },
        "land": {
            "accounting_version": world_metadata.get("land_accounting_version", "legacy_no_land_guarantee"),
            "fields": list(land_payload),
            "fraction_support": "whole_cell; nine mutually exclusive land_use_fraction maps sum to one",
            "project_area_semantics": "exclusive envelope inside energy_reserve; project areas are nested, not additional top-level fractions or impervious cover",
            "display_compatibility": "legacy static categorical tensor retains mixed land_cover/land_use_zone display encoding; independent land__* arrays preserve actual cover and protection",
        },
        "hydrology": {
            **world_metadata.get("dynamic_hydrology", {"mode": "legacy_static_only", "artifact": None}),
            "fields": list(hydrology_payload),
            "state_support": "T+1 interval boundaries; all mm stores are whole-cell equivalent depths",
            "forcing_availability": "realized hourly precipitation; no implicit NWP or forecast availability",
        },
        "source_load": {
            **(source_load_store.metadata if source_load_payload else {"mode": "legacy_no_diagnostics"}),
            "fields": list(source_load_payload), "source_artifact": "stage_11_operation/source_load_forecast.npz",
            "semantics": "Exogenous requested load, generation availability and planned generation; no delivery or curtailment claim",
            "node_identity": "Original Stage11 bus_ids; independent of the final graph's node order",
            "window_accounting": "period_* are realized summaries, never static inputs; windows recompute them from their own intervals and use the preceding effective-temperature boundary",
        },
        "dynamic": {
            "weather_generation": json.loads(str(weather["weather_metadata_json"])) if "weather_metadata_json" in weather else {"generation_mode": "unspecified_legacy"},
            "optional_diagnostics": [name for name in dynamic_payload if name.startswith("diagnostic__")],
            "weather_channels": [str(value) for value in weather["channel_names"]],
            "weather_statistics": _channel_statistics(
                np.moveaxis(weather["dynamic"], 1, 0),
                tuple(str(value) for value in weather["channel_names"]),
            ),
        },
        "graph": graph_metadata,
        "asset_planning": world_metadata.get("asset_planning", {"mode":"legacy_full_window_unspecified"}),
        "operation_detail": {"mode":"operation_v1" if operation_detail else "legacy_no_appendix",
                             "fields":list(operation_detail),"semantics":"physical gross demand and generator output exclude storage; explicit island-local reserve accounts"},
        "operation": {
            "units": OPERATION_UNITS,
            "exogenous_available": exogenous is not None,
            "exogenous_source": "stage_11_operation/source_load_forecast.npz" if exogenous is not None else None,
            "node_dynamic_semantics": "Stage 14 dispatch: load includes charging; availability includes storage power",
            "bus_count": int(forecast["bus_ids"].size),
            "branch_count": int(power_flow["branch_ids"].size),
            "storage_site_count": int(storage["site_ids"].size),
            "peak_load_mw": float(np.max(storage["total_load_mw"])),
            "peak_line_loading_ratio": float(np.max(power_flow["line_loading_ratio"], initial=0.0)),
            "total_unserved_mwh": float(np.sum(power_flow["unserved_load_mw"])),
        },
        "validation": validation,
    }
    seed = metadata["seed"]
    sample_name = f"seed{seed}.npz" if seed is not None else f"{world_dir.name}.npz"
    sample_path = sample_root / sample_name
    payload = {
        **_prefix_payload("static", static_payload),
        **_prefix_payload("land", land_payload),
        **_prefix_payload("hydrology", hydrology_payload),
        **_prefix_payload("source_load", source_load_payload),
        **_prefix_payload("operation_detail", operation_detail),
        **_prefix_payload("power_flow_detail", power_flow_detail),
        **_prefix_payload("asset_planning", asset_payload),
        **_prefix_payload("dynamic", dynamic_payload),
        **_prefix_payload("graph", graph_payload),
        **_prefix_payload("operation", operation_payload),
        "metadata_json": np.asarray(json.dumps(metadata, separators=(",", ":"))),
        "config_yaml": np.asarray(config_path.read_text(encoding="utf-8")),
    }
    np.savez_compressed(sample_path, **payload)
    file_info = {
        "name": sample_name,
        "bytes": sample_path.stat().st_size,
        "sha256": _sha256(sample_path),
    }
    return {
        "sample_id": world_dir.name,
        "seed": seed,
        "partition": (partitions or {}).get(seed, "inspection"),
        "hours": metadata["hours"],
        "node_count": graph_metadata["node_count"],
        "edge_count": graph_metadata["edge_count"],
        "storage_site_count": metadata["operation"]["storage_site_count"],
        "peak_load_mw": metadata["operation"]["peak_load_mw"],
        "peak_line_loading_ratio": metadata["operation"]["peak_line_loading_ratio"],
        "file": file_info,
        "validation_passed": bool(validation["passed"]),
        "generator_version": provenance["generator_version"],
        "scenario_semantics": provenance["scenario_semantics"],
        "exogenous_available": exogenous is not None,
    }


def dataset_schema() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_file": {
            "format": "one compressed NPZ per world seed; keys use <group>__<field>",
            "static": {
                "continuous": "float32 [C_static,H,W]",
                "categorical": "int32 [C_categorical,H,W]",
            },
            "land": {
                "semantics": "Optional complete land_use_v1 appendix; static/background cover, planned whole-cell fractions and Stage8/9 project budgets. Never time-sliced.",
                "landform/land_cover_type": "integer [H,W]; separate geomorphology and potential background cover",
                "protected_mask/*_land_eligible": "bool [H,W]; hard identities and project exclusions",
                "allocatable_land_fraction/land_use_fraction_*": "float [H,W]; fractions of whole-cell area, use fractions sum to one",
                "*_area_km2": "float [H,W]; allocated or residual geometric budget",
                "energy_project_land_ledger/thermal_land_ledger": "float64 [N,9]; named columns preserve identity, reserved area and capacity",
                "*_project_area_by_cell_km2": "float64 [N,H,W]; project-by-cell reserved areas",
                "wind_candidates/pv_candidates/load_candidates": "legacy [N,7] tables, kept unchanged",
            },
            "dynamic": {
                "weather": "float32 [T,C_weather,H,W]",
                "weather_class": "int16 [T,H,W]",
                "diagnostic__*": "optional float32 [T,H,W]; units and statistic support in metadata.dynamic.weather_generation",
                "time_bounds_hours": "optional float64 [T,2], interval start/end in local solar hours",
                "static_elevation_m": "optional float32 [H,W], terrain elevation for moist-air diagnostics",
                "timestamps": "int32 [T] hours from start of year",
            },
            "hydrology": {
                "semantics": "Optional hydrology_v1 water account; explicit names and field_schema_json define time support",
                "state__*": "float64 [T+1,H,W] boundary states; state_time_hours includes terminal boundary",
                "flux__*": "float64 [T,H,W] interval accumulations, except discharge_m3_s interval mean",
                "static__*": "[H,W] fixed geometry/parameters; never time-sliced",
                "budget__*": "float64 [T] world-wide interval water accounts",
                "timestamps/time_bounds_hours/state_time_hours": "[T]/[T,2]/[T+1] hours in local solar convention",
                "absent": "Legacy or static_only worlds have no simulated dynamic hydrology arrays",
            },
            "source_load": {
                "semantics": "Optional complete source_load_v1 Stage11 exogenous appendix; original bus IDs are independent of final graph order",
                "static_assets": "nameplate_capacity_mw/reference_load_mw/weather_sample_row/weather_sample_col/capacity_factor_valid [N]",
                "power/energy/CF/diag__*": "Explicit schema fields [T,N]; energy is interval MWh, effective temperature is interval-end state",
                "initial_effective_temperature_c": "[N] preceding boundary; temporal windows take their own preceding state",
                "period_*_energy_mwh": "[N] realized whole-period audits or targets, never static input; each temporal window recomputes its own sums",
                "field_schema": "source_load_field_schema_json declares units and support; no shape-based classification",
                "absent": "Legacy worlds remain readable without fabricated capacities or diagnostics",
            },
            "operation_detail/power_flow_detail": {
                "semantics":"Complete operation_v1 Stores; explicit named support, no leading-dimension inference",
                "op__*":"Physical interval mean demand/generation/reserve and [T,N]/[T,E] runtime topology",
                "soc_mwh":"[T+1,S] boundary states; window initial and previous power use their own preceding boundary",
            },
            "asset_planning": {
                "initial_assets__/frozen_assets__":"Explicit fixed ID/capacity arrays and original/frozen asset snapshots",
                "boundary_metadata_json":"Mode, input hashes, information cutoff, separate planning/operation clocks and dispatch foresight",
                "window_use":"Separate design group; full-window planning assets remain oracle context, never relabeled issue-time inputs",
            },
            "graph": {
                "semantics": "only node and line are graph entities; A* paths are optional line geometry metadata",
                "node_id": "int32 [N]",
                "node_type": "int8 [N]",
                "node_features": "float32 [N,5]; capacity_mw retains Stage12 pre-operation design values; F authoritative frozen assets live in asset_planning",
                "node_source_id": "int32 [N] auxiliary source relationship",
                "node_electrical": "float32 [N,6]",
                "edge_id": "int32 [E]",
                "edge_index": "int32 [2,E]",
                "edge_bus_ids": "int32 [2,E]",
                "edge_features": "float32 [E,7]",
                "optional_line_geometry": {
                    "path_ptr": "int32 [E+1]",
                    "path_row/path_col": "int16 [sum_path_cells]",
                },
            },
            "operation": {
                "source_load_appendix": "Separate source_load group preserves Stage11 original node identity, nameplate/CF, diagnostics and interval/period MWh",
                "node_dynamic": "float32 [T,C_node,N]",
                "node_dynamic_semantics": "Stage 14 dispatch quantities; includes storage charging and inverter capacity",
                "exogenous_p_load_mw": "optional float32 [T,N]: Stage 11 requested demand, without storage charging",
                "exogenous_p_gen_available_mw": "optional float32 [T,N]: Stage 11 generator availability, without storage capacity or later expansion",
                "exogenous_p_renewable_available_mw": "optional float32 [T,N]: Stage 11 wind/PV availability, zero at other bus kinds",
                "exogenous_bus_present": "optional bool [N]: final bus existed in Stage 11",
                "line_dynamic": "float32 [T,C_line,E]",
                "storage time series": "float32 [T,S], except soc_mwh [T+1,S]",
            },
            "embedded_text": {
                "metadata_json": "JSON string scalar",
                "config_yaml": "YAML string scalar",
            },
        },
        "node_type_mapping": NODE_TYPE_TO_ID,
        "static_units": STATIC_UNITS,
        "operation_units": OPERATION_UNITS,
        "dispatch_semantics": "declared_per_world_separately_from_asset_planning_mode",
        "forecast_evaluation": "Future weather is realized weather, not a forecast available at issue time. Final topology and dispatch were planned using the operation period.",
    }


def _graph_payload(topology: object, electrical: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    buses = tuple(topology.refined_buses)
    edges = tuple(topology.refined_edges)
    electrical_buses = np.atleast_2d(electrical["electrical_buses"])
    electrical_branches = np.atleast_2d(electrical["electrical_branches"])
    bus_electrical = {int(row[0]): row for row in electrical_buses}
    branch_electrical = {int(row[0]): row for row in electrical_branches}

    node_id = np.asarray([bus.bus_id for bus in buses], dtype=np.int32)
    node_type = np.asarray([NODE_TYPE_TO_ID[bus.kind] for bus in buses], dtype=np.int8)
    node_features = np.asarray(
        [
            [
                bus.row,
                bus.col,
                bus.x,
                bus.y,
                bus.capacity_mw,
            ]
            for bus in buses
        ],
        dtype=np.float32,
    )
    node_electrical = np.asarray([bus_electrical[int(bus.bus_id)][1:7] for bus in buses], dtype=np.float32)
    edge_id = np.asarray([edge.edge_id for edge in edges], dtype=np.int32)
    node_position = {int(bus_id): index for index, bus_id in enumerate(node_id)}
    edge_bus_ids = np.asarray([[edge.from_bus, edge.to_bus] for edge in edges], dtype=np.int32).reshape(-1, 2).T
    edge_index = np.asarray(
        [[node_position[int(edge.from_bus)], node_position[int(edge.to_bus)]] for edge in edges],
        dtype=np.int32,
    ).reshape(-1, 2).T
    edge_features = np.asarray(
        [
            [
                edge.length_km,
                *branch_electrical[int(edge.edge_id)][3:9],
            ]
            for edge in edges
        ],
        dtype=np.float32,
    )
    path_lengths = np.asarray([len(edge.path_rows) for edge in edges], dtype=np.int32)
    path_ptr = np.concatenate((np.asarray([0], dtype=np.int32), np.cumsum(path_lengths, dtype=np.int32)))
    path_row = np.concatenate([np.asarray(edge.path_rows, dtype=np.int16) for edge in edges]) if edges else np.empty(0, dtype=np.int16)
    path_col = np.concatenate([np.asarray(edge.path_cols, dtype=np.int16) for edge in edges]) if edges else np.empty(0, dtype=np.int16)
    payload = {
        "node_id": node_id,
        "node_type": node_type,
        "node_features": node_features,
        "node_source_id": np.asarray([bus.source_id for bus in buses], dtype=np.int32),
        "node_electrical": node_electrical,
        "edge_id": edge_id,
        "edge_index": edge_index,
        "edge_bus_ids": edge_bus_ids,
        "edge_features": edge_features,
        "path_ptr": path_ptr,
        "path_row": path_row,
        "path_col": path_col,
    }
    metadata = {
        "node_count": len(buses),
        "edge_count": len(edges),
        "node_type_mapping": NODE_TYPE_TO_ID,
        "node_feature_names": [
            "row",
            "col",
            "x",
            "y",
            "capacity_mw",
        ],
        "node_electrical_names": [
            "nominal_kv",
            "p_capacity_mw",
            "q_capacity_mvar",
            "base_load_mw",
            "power_factor",
            "voltage_setpoint_pu",
        ],
        "edge_feature_names": [
            "route_length_km",
            "nominal_kv",
            "electrical_length_km",
            "r_ohm",
            "x_ohm",
            "b_us",
            "rate_mva",
        ],
    }
    return payload, metadata


def _operation_payload(
    forecast: dict[str, np.ndarray],
    power_flow: dict[str, np.ndarray],
    storage: dict[str, np.ndarray],
    *, exogenous: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    node_sources = forecast | power_flow
    line_sources = power_flow | storage
    payload = {
        "timestamps": forecast["timestamps"],
        "bus_ids": forecast["bus_ids"],
        "bus_kinds": forecast["bus_kinds"],
        "branch_ids": power_flow["branch_ids"],
        "slack_bus_id": power_flow["slack_bus_id"],
        "node_dynamic": np.stack(
            [np.asarray(node_sources[name], dtype=np.float32) for name in NODE_OPERATION_CHANNELS],
            axis=1,
        ),
        "node_dynamic_channels": np.asarray(NODE_OPERATION_CHANNELS),
        "line_dynamic": np.stack(
            [np.asarray(line_sources[name], dtype=np.float32) for name in LINE_OPERATION_CHANNELS],
            axis=1,
        ),
        "line_dynamic_channels": np.asarray(LINE_OPERATION_CHANNELS),
    }
    for key in (
        "site_ids",
        "site_bus_ids",
        "site_power_capacity_mw",
        "site_energy_capacity_mwh",
        "storage_power_expansion_mw",
        "storage_energy_expansion_mwh",
        "charge_mw",
        "discharge_mw",
        "emergency_discharge_mw",
        "soc_mwh",
        "target_soc_mwh",
        "cycle_boundary_soc_mwh",
        "minimum_soc_fraction",
        "maximum_soc_fraction",
        "preferred_soc_lower_fraction",
        "preferred_soc_upper_fraction",
        "total_load_mw",
        "renewable_available_mw",
        "baseline_thermal_mw",
        "scheduled_thermal_mw",
        "baseline_unserved_mw",
        "dispatched_unserved_mw",
        "baseline_curtailed_mw",
        "dispatched_curtailed_mw",
        "line_capacity_expansion_mva",
        "thermal_bus_ids",
        "thermal_capacity_expansion_mw",
    ):
        payload[key] = storage[key]
    if exogenous is not None:
        payload.update(_exogenous_payload(exogenous, forecast["bus_ids"], forecast["timestamps"]))
    return payload


def _exogenous_payload(source: dict[str, np.ndarray], final_bus_ids: np.ndarray, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    """Map pre-dispatch realizations by business ID, without inventing targets.

    New zero-injection transit buses receive zeros and a false presence mask.
    Removing a bus with nonzero original injection is rejected: otherwise a
    changed topology could silently destroy energy during dataset packaging.
    """
    if not np.array_equal(source["timestamps"], timestamps):
        raise ValueError("Stage 11 and final-operation timestamps differ")
    original_ids = entity_ids(source["bus_ids"], "exogenous bus_ids")
    target_ids = entity_ids(final_bus_ids, "final bus_ids")
    source_index = {int(bus_id): i for i, bus_id in enumerate(original_ids)}
    missing = ~np.isin(original_ids, target_ids)
    payload: dict[str, np.ndarray] = {}
    for name in ("p_load_mw", "p_gen_available_mw"):
        values = np.asarray(source[name], dtype=np.float32)
        if values.shape != (len(timestamps), len(original_ids)):
            raise ValueError(f"Stage 11 {name} has inconsistent shape")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Stage 11 {name} must be finite and nonnegative")
        if np.any(values[:, missing] != 0):
            raise ValueError("Final topology removed a bus with nonzero exogenous injection")
        mapped = np.zeros((len(timestamps), len(target_ids)), dtype=np.float32)
        for target_index, bus_id in enumerate(target_ids):
            if int(bus_id) in source_index:
                mapped[:, target_index] = values[:, source_index[int(bus_id)]]
        payload[f"exogenous_{name}"] = mapped
    kinds = np.asarray(source["bus_kinds"])
    if kinds.shape != original_ids.shape:
        raise ValueError("Stage 11 bus kinds must match bus IDs")
    renewable_mask = np.asarray([
        int(bus_id) in source_index and str(kinds[source_index[int(bus_id)]]) in {"wind_bus", "pv_bus"}
        for bus_id in target_ids
    ])
    payload["exogenous_p_renewable_available_mw"] = payload["exogenous_p_gen_available_mw"] * renewable_mask[None, :]
    payload["exogenous_bus_present"] = np.isin(target_ids, original_ids)
    return payload


def _has_physical_units(world_metadata: dict[str, object]) -> bool:
    # Explicit compatibility, never treat arbitrary future/legacy labels as SI.
    return world_metadata.get("generator_version") in {"physics_v3", "physics_v4"}


def _land_accounting_payload(layout: WorldDataLayout, static: dict[str, np.ndarray], metadata: dict[str, object]) -> dict[str, np.ndarray]:
    """Append C axes/ledgers without changing legacy model tensor channels."""
    version = metadata.get("land_accounting_version")
    has_new_fields = any(name in static for name in ("landform", "land_cover_type", "allocatable_land_fraction"))
    if version is None and not has_new_fields:
        return {}
    if version != "land_use_v1":
        raise ValueError("New land fields require the explicit land_use_v1 accounting version")
    maps = ("landform", "land_cover_type", "protected_mask", "allocatable_land_fraction",
            "energy_available_area_km2", "energy_wind_project_area_km2", "energy_pv_project_area_km2", "energy_unallocated_area_km2",
            "wind_land_eligible", "pv_land_eligible", "thermal_land_eligible", "thermal_allocated_area_km2", "energy_unallocated_after_thermal_area_km2")
    maps += tuple(f"land_use_fraction_{name}" for name in ("water", "wetland", "residential", "commercial", "industrial", "agriculture", "park_green", "natural", "energy_reserve"))
    missing = [name for name in maps if name not in static]
    if missing:
        raise ValueError(f"Incomplete modern land accounting maps: {missing}")
    shape = static["elevation"].shape
    for name in maps:
        if static[name].shape != shape or not np.isfinite(static[name]).all():
            raise ValueError(f"Land map {name} must be finite HxW")
    payload = {name: static[name].copy() for name in maps}
    candidates = _load_npz(layout.existing(layout.energy, "source_load_candidates.npz", layout.root / "source_load_candidates.npz"))
    nodes = _load_npz(layout.existing(layout.buses, "grid_nodes.npz", layout.root / "grid_nodes.npz"))
    for source, names in ((candidates, ("wind_candidates", "pv_candidates", "load_candidates", "energy_project_land_ledger", "energy_project_land_columns", "energy_project_area_by_cell_km2")),
                          (nodes, ("thermal_land_ledger", "thermal_land_columns", "thermal_project_area_by_cell_km2", "thermal_land_accounting_mode"))):
        for name in names:
            if name not in source:
                raise ValueError(f"Incomplete modern project land accounting: {name}")
            payload[name] = source[name].copy()
    return payload


def _world_provenance(world_metadata: dict[str, object]) -> dict[str, object]:
    planning = world_metadata.get("asset_planning", {})
    if not isinstance(planning,dict):
        raise ValueError("Asset planning metadata must be a mapping")
    mode = planning.get("mode","legacy_full_window_unspecified")
    assets_oracle = mode in {"full_window_planning","legacy_full_window_unspecified"}
    return {
        "generator_version": world_metadata.get("generator_version", "legacy_unspecified"),
        "scenario_semantics": world_metadata.get("scenario_semantics", "unspecified_legacy"),
        "time_convention": world_metadata.get("time_convention", "unspecified_legacy"),
        "execution_stage_order": world_metadata.get("execution_stage_order", []),
        "land_accounting_version": world_metadata.get("land_accounting_version", "legacy_no_land_guarantee"),
        "asset_planning_mode": mode,
        "asset_planning_uses_operation_future": assets_oracle,
        "dispatch_semantics": planning.get("dispatch_foresight","perfect_foresight_dispatch"),
        "forecast_feature_availability": "Future weather is realized. " + ("Assets use the full operation period and are oracle context. " if assets_oracle else "Assets are frozen before operation. ") + "Operation optimizer foresight is declared independently and is not implied by the asset mode.",
    }


def _validate_modalities(
    static_continuous: np.ndarray,
    static_categorical: np.ndarray,
    weather: dict[str, np.ndarray],
    forecast: dict[str, np.ndarray],
    power_flow: dict[str, np.ndarray],
    storage: dict[str, np.ndarray],
    graph: dict[str, np.ndarray],
) -> dict[str, object]:
    structural_checks = {
        "grid_shapes_match": bool(
            static_continuous.shape[1:]
            == static_categorical.shape[1:]
            == weather["dynamic"].shape[2:]
            == weather["weather_class"].shape[1:]
        ),
        "timestamps_match": bool(
            np.array_equal(weather["timestamps"], forecast["timestamps"])
            and np.array_equal(forecast["timestamps"], power_flow["timestamps"])
            and np.array_equal(power_flow["timestamps"], storage["timestamps"])
        ),
        "bus_ids_match": bool(
            np.array_equal(forecast["bus_ids"], power_flow["bus_ids"])
            and set(int(value) for value in forecast["bus_ids"])
            == set(int(value) for value in graph["node_id"])
        ),
        "branch_ids_match": bool(
            np.array_equal(power_flow["branch_ids"], storage["branch_ids"])
            and set(int(value) for value in power_flow["branch_ids"])
            == set(int(value) for value in graph["edge_id"])
        ),
        "storage_bus_ids_exist": bool(
            set(int(value) for value in storage["site_bus_ids"])
            <= set(int(value) for value in graph["node_id"])
        ),
        "soc_has_terminal_step": bool(storage["soc_mwh"].shape[0] == weather["timestamps"].size + 1),
        "finite_continuous_data": bool(
            np.isfinite(static_continuous).all()
            and np.isfinite(weather["dynamic"]).all()
            and np.isfinite(power_flow["line_loading_ratio"]).all()
            and np.isfinite(storage["soc_mwh"]).all()
        ),
        "nonnegative_line_loading": bool(np.min(power_flow["line_loading_ratio"], initial=0.0) >= -1e-6),
    }
    unserved = power_flow["unserved_load_mw"]
    diagnostics = {
        "total_unserved_mwh": float(np.sum(unserved)),
        "maximum_unserved_mw": float(np.max(unserved)),
        "unserved_is_numerical_residual": bool(np.max(unserved) <= 1e-4),
    }
    return {
        "passed": bool(all(structural_checks.values())),
        "structural_checks": structural_checks,
        "operation_diagnostics": diagnostics,
    }


def _stack_channels(source: dict[str, np.ndarray], names: tuple[str, ...], dtype: object) -> np.ndarray:
    missing = [name for name in names if name not in source]
    if missing:
        raise KeyError(f"Static channels are missing: {missing}")
    return np.stack([np.asarray(source[name], dtype=dtype) for name in names], axis=0)


def _categorical_channels(source: dict[str, np.ndarray]) -> np.ndarray:
    required = (
        "flow_direction",
        "river",
        "lake",
        "watershed_id",
        "land_cover",
        "urban_mask",
        "city_id_map",
        "land_use_zone",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise KeyError(f"Static categorical sources are missing: {missing}")
    water_type = np.zeros_like(source["river"], dtype=np.int32)
    water_type[np.asarray(source["river"], dtype=bool)] = 1
    water_type[np.asarray(source["lake"], dtype=bool)] = 2
    land_use_zone = np.asarray(source["land_use_zone"], dtype=np.int32).copy()
    land_use_zone[np.isin(land_use_zone, (6, 7))] = 0
    channels = {
        "flow_direction": source["flow_direction"],
        "water_type": water_type,
        "watershed_id": source["watershed_id"],
        "land_cover": source["land_cover"],
        "urban_mask": source["urban_mask"],
        "city_id_map": source["city_id_map"],
        "land_use_zone": land_use_zone,
    }
    return np.stack([np.asarray(channels[name], dtype=np.int32) for name in STATIC_CATEGORICAL_CHANNELS], axis=0)


def _channel_statistics(values: np.ndarray, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "min": float(np.min(values[index])),
            "max": float(np.max(values[index])),
            "mean": float(np.mean(values[index])),
            "std": float(np.std(values[index])),
        }
        for index, name in enumerate(names)
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _prefix_payload(group: str, payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{group}__{name}": np.asarray(values) for name, values in payload.items()}


def _seed_from_world_id(world_id: str) -> int | None:
    marker = "_seed"
    if marker not in world_id:
        return None
    try:
        return int(world_id.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _dataset_readme() -> str:
    return """# SimGenr dataset preview

Each sample is one generated world paired with its configured hourly operation period.

- `samples/seed<seed>.npz`: one complete world sample.
- `static__*`: continuous and categorical spatial channels.
- `dynamic__*`: hourly weather fields in TCHW layout.
- `graph__*`: final Stage 12 buses, branches, electrical attributes, and ragged A* paths.
- `operation__node_dynamic`: final Stage 14 source/load, power-flow and storage operation; demand includes charging.
- `operation__exogenous_*`: Stage 11 original demand and availability mapped to final bus IDs, when present.
- `metadata_json` and `config_yaml`: embedded metadata and generation configuration.

Rendered PNG/WebP figures are not packaged into training samples.

Dataset partitions must be assigned by world seed. Hours from one world must never be split across train,
validation, and test partitions.

Schema 0.6.0 preserves generator_version and scenario_semantics. Dispatch and final graph
are planned with perfect knowledge of the operation period. Future weather is realized
weather, not an issue-time forecast. Exogenous targets remove storage contamination but
do not by themselves make this dataset free of future-information leakage. Legacy files
without Stage 11 arrays remain readable; exogenous arrays are never fabricated from dispatch.
"""


def _inspection_summary(output_dir: Path, samples: list[dict[str, object]]) -> str:
    rows = []
    for sample in samples:
        rows.append(
            "| {sample_id} | {seed} | {split} | {nodes} | {edges} | {storage} | {load:.2f} | {loading:.3f} | {valid} |".format(
                sample_id=sample["sample_id"],
                seed=sample["seed"],
                split=sample["partition"],
                nodes=sample["node_count"],
                edges=sample["edge_count"],
                storage=sample["storage_site_count"],
                load=sample["peak_load_mw"],
                loading=sample["peak_line_loading_ratio"],
                valid="yes" if sample["validation_passed"] else "no",
            )
        )
    return """# Dataset inspection summary

| sample | seed | split | buses | branches | storage sites | peak load MW | peak line loading | valid |
|---|---:|:---:|---:|---:|---:|---:|---:|:---:|
{rows}

Samples are partitioned by world seed. Hours from one world remain in the same split.
""".format(rows="\n".join(rows))
