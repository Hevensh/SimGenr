from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from world_generator.core.output_layout import WorldDataLayout
from world_generator.operation.stage_cache import load_stage12_checkpoint


SCHEMA_VERSION = "0.5.0"

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
    "water_depth": "relative_depth",
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
        "sample_unit": "one generated world with one 168-hour operation week",
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
    if missing:
        raise FileNotFoundError("World output is incomplete: " + ", ".join(missing))

    static_source = _load_npz(static_path)
    weather = _load_npz(weather_path)
    forecast = _load_npz(forecast_path)
    power_flow = _load_npz(power_flow_path)
    storage = _load_npz(storage_path)
    electrical = _load_npz(electrical_path)
    _, topology, _, _ = load_stage12_checkpoint(world_dir)

    static_continuous = _stack_channels(static_source, STATIC_CONTINUOUS_CHANNELS, np.float32)
    static_categorical = _categorical_channels(static_source)
    static_payload = {
        "continuous": static_continuous,
        "categorical": static_categorical,
        "continuous_channels": np.asarray(STATIC_CONTINUOUS_CHANNELS),
        "categorical_channels": np.asarray(STATIC_CATEGORICAL_CHANNELS),
    }
    dynamic_payload = {
        "timestamps": weather["timestamps"].astype(np.int32),
        "weather": weather["dynamic"].astype(np.float32),
        "weather_class": weather["weather_class"].astype(np.int16),
        "weather_channels": weather["channel_names"],
        "time_unit": weather["time_unit"],
        "start_day_of_year": weather["start_day_of_year"].astype(np.int32),
    }

    graph_payload, graph_metadata = _graph_payload(topology, electrical)
    operation_payload = _operation_payload(forecast, power_flow, storage)

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
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": world_dir.name,
        "seed": _seed_from_world_id(world_dir.name),
        "source_world": str(world_dir.as_posix()),
        "config_sha256": _sha256(config_path),
        "grid_shape": list(static_continuous.shape[1:]),
        "hours": int(weather["dynamic"].shape[0]),
        "static": {
            "continuous_channels": list(STATIC_CONTINUOUS_CHANNELS),
            "categorical_channels": list(STATIC_CATEGORICAL_CHANNELS),
            "continuous_statistics": _channel_statistics(static_continuous, STATIC_CONTINUOUS_CHANNELS),
        },
        "dynamic": {
            "weather_channels": [str(value) for value in weather["channel_names"]],
            "weather_statistics": _channel_statistics(
                np.moveaxis(weather["dynamic"], 1, 0),
                tuple(str(value) for value in weather["channel_names"]),
            ),
        },
        "graph": graph_metadata,
        "operation": {
            "bus_count": int(forecast["bus_ids"].size),
            "branch_count": int(power_flow["branch_ids"].size),
            "storage_site_count": int(storage["site_ids"].size),
            "peak_load_mw": float(np.max(storage["total_load_mw"])),
            "peak_line_loading_ratio": float(np.max(power_flow["line_loading_ratio"])),
            "total_unserved_mwh": float(np.sum(power_flow["unserved_load_mw"])),
        },
        "validation": validation,
    }
    seed = metadata["seed"]
    sample_name = f"seed{seed}.npz" if seed is not None else f"{world_dir.name}.npz"
    sample_path = sample_root / sample_name
    legacy_sample_dir = sample_root / world_dir.name
    if legacy_sample_dir.is_dir():
        shutil.rmtree(legacy_sample_dir)
    payload = {
        **_prefix_payload("static", static_payload),
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
            "dynamic": {
                "weather": "float32 [T,C_weather,H,W]",
                "weather_class": "int16 [T,H,W]",
                "timestamps": "int32 [T] hours from start of year",
            },
            "graph": {
                "semantics": "only node and line are graph entities; A* paths are optional line geometry metadata",
                "node_id": "int32 [N]",
                "node_type": "int8 [N]",
                "node_features": "float32 [N,5]",
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
                "node_dynamic": "float32 [T,C_node,N]",
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
    edge_bus_ids = np.asarray([[edge.from_bus, edge.to_bus] for edge in edges], dtype=np.int32).T
    edge_index = np.asarray(
        [[node_position[int(edge.from_bus)], node_position[int(edge.to_bus)]] for edge in edges],
        dtype=np.int32,
    ).T
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
    path_row = np.concatenate([np.asarray(edge.path_rows, dtype=np.int16) for edge in edges])
    path_col = np.concatenate([np.asarray(edge.path_cols, dtype=np.int16) for edge in edges])
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
    return payload


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
        "nonnegative_line_loading": bool(np.min(power_flow["line_loading_ratio"]) >= -1e-6),
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

Each sample is one generated 64 x 64 world paired with one 168-hour operation week.

- `samples/seed<seed>.npz`: one complete world sample.
- `static__*`: continuous and categorical spatial channels.
- `dynamic__*`: hourly weather fields in TCHW layout.
- `graph__*`: final Stage 12 buses, branches, electrical attributes, and ragged A* paths.
- `operation__*`: final Stage 14 source/load, power-flow, and storage time series.
- `metadata_json` and `config_yaml`: embedded metadata and generation configuration.

Rendered PNG/WebP figures are not packaged into training samples.

Dataset partitions must be assigned by world seed. Hours from one world must never be split across train,
validation, and test partitions.
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
