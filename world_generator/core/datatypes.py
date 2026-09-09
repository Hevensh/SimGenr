from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np

from world_generator.core.contracts import interval_bounds_hours, validate_node_arrays, validate_weather_arrays


FloatMap = np.ndarray
BoolMap = np.ndarray


@dataclass(frozen=True)
class TerrainBase:
    elevation: FloatMap
    land_mask: BoolMap


@dataclass(frozen=True)
class TerrainFeatures:
    elevation: FloatMap
    slope: FloatMap
    aspect_sin: FloatMap
    aspect_cos: FloatMap
    roughness: FloatMap  # dimensionless 3x3 relief / (2 * grid spacing); not aerodynamic z0
    curvature: FloatMap

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "elevation": self.elevation,
            "slope": self.slope,
            "aspect_sin": self.aspect_sin,
            "aspect_cos": self.aspect_cos,
            "roughness": self.roughness,
            "curvature": self.curvature,
        }


@dataclass(frozen=True)
class HydrologyState:
    flow_direction: np.ndarray
    flow_accumulation: FloatMap
    river_centerline: BoolMap
    river: BoolMap
    lake: BoolMap
    water_depth: FloatMap
    hydrology_elevation: FloatMap
    watershed_id: np.ndarray
    distance_to_water: FloatMap
    flood_risk: FloatMap
    catchment_area_km2: FloatMap | None = None

    def as_maps(self) -> dict[str, np.ndarray]:
        maps = {
            "flow_direction": self.flow_direction,
            "flow_accumulation": self.flow_accumulation,
            "river_centerline": self.river_centerline,
            "river": self.river,
            "lake": self.lake,
            "water_depth": self.water_depth,
            "hydrology_elevation": self.hydrology_elevation,
            "watershed_id": self.watershed_id,
            "distance_to_water": self.distance_to_water,
            "flood_risk": self.flood_risk,
        }
        if self.catchment_area_km2 is not None:
            maps["catchment_area_km2"] = self.catchment_area_km2
        return maps


@dataclass(frozen=True)
class HydrologyTimeSeriesStore:
    """D: six explicit boundary states, interval fluxes and static water geometry."""

    timestamps: np.ndarray
    time_bounds_hours: np.ndarray
    state_time_hours: np.ndarray
    states: dict[str, np.ndarray]
    fluxes: dict[str, np.ndarray]
    static_maps: dict[str, np.ndarray]
    budgets: dict[str, np.ndarray]
    metadata: dict[str, object] = field(default_factory=dict)

    def as_arrays(self) -> dict[str, np.ndarray]:
        from world_generator.core.contracts import integer_labels
        from world_generator.core.hydrology_contracts import HYDROLOGY_GROUP_UNITS, HYDROLOGY_MODE, HYDROLOGY_SCHEMA_VERSION, hydrology_field_schema

        bounds = interval_bounds_hours(self.timestamps, 1.0)
        if self.time_bounds_hours.shape != bounds.shape or not np.allclose(self.time_bounds_hours, bounds, rtol=0, atol=1e-9):
            raise ValueError("Hydrology interval bounds must match consecutive hourly timestamps")
        state_time = np.r_[bounds[:, 0], bounds[-1, 1]]
        if self.state_time_hours.shape != state_time.shape or not np.allclose(self.state_time_hours, state_time, rtol=0, atol=1e-9):
            raise ValueError("Hydrology state_time_hours must have the T+1 interval boundaries")
        groups = {"state": self.states, "flux": self.fluxes, "static": self.static_maps, "budget": self.budgets}
        for group, values in groups.items():
            if set(values) != set(HYDROLOGY_GROUP_UNITS[group]):
                raise ValueError(f"Hydrology {group} fields must exactly match hydrology_v1")
        soil = np.asarray(self.states["soil_storage_mm"])
        if soil.ndim != 3 or min(soil.shape[1:]) < 1:
            raise ValueError("Hydrology states require T+1,H,W")
        shape, hours = soil.shape[1:], bounds.shape[0]
        expected = {"state": (hours + 1, *shape), "flux": (hours, *shape), "static": shape, "budget": (hours,)}
        arrays = {"timestamps": np.asarray(self.timestamps), "time_bounds_hours": np.asarray(self.time_bounds_hours),
                  "state_time_hours": np.asarray(self.state_time_hours), "schema_version": np.asarray(HYDROLOGY_SCHEMA_VERSION),
                  "mode": np.asarray(HYDROLOGY_MODE), "field_schema_json": np.asarray(json.dumps(hydrology_field_schema(), sort_keys=True)),
                  "hydrology_metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True, allow_nan=False))}
        signed = {"lake_water_level_m", "lake_bed_elevation_m", "lake_spill_elevation_m", "routing_receiver_flat_index", "cell_budget_residual_m3", "residual_m3"}
        for group, values in groups.items():
            for name, array in values.items():
                array = np.asarray(array)
                if array.shape != expected[group] or not np.isfinite(array).all():
                    raise ValueError(f"Hydrology {group} field {name} must be finite {expected[group]}")
                if name not in signed and np.any(array < 0):
                    raise ValueError(f"Hydrology {name} cannot be negative")
                if name.endswith("_fraction") and np.any(array > 1):
                    raise ValueError(f"Hydrology {name} cannot exceed one")
                if name in {"lake_id", "routing_receiver_flat_index"}: integer_labels(array, f"hydrology {name}")
                if name == "routing_receiver_flat_index" and np.any((array < -2) | (array >= np.prod(shape))):
                    raise ValueError("Hydrology receiver must be a valid cell index or -1/-2")
                if name == "closed_sink_mask" and not np.isin(array, [0, 1]).all():
                    raise ValueError("closed_sink_mask must be boolean")
                arrays[f"{group}__{name}"] = array
        return arrays

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "HydrologyTimeSeriesStore":
        from world_generator.core.hydrology_contracts import HYDROLOGY_GROUP_UNITS, HYDROLOGY_MODE, HYDROLOGY_SCHEMA_VERSION, hydrology_field_schema

        if str(arrays.get("schema_version")) != HYDROLOGY_SCHEMA_VERSION or str(arrays.get("mode")) != HYDROLOGY_MODE:
            raise ValueError("Unsupported or missing dynamic hydrology schema/mode")
        if json.loads(str(arrays.get("field_schema_json", "null"))) != hydrology_field_schema():
            raise ValueError("Hydrology field schema is missing or differs from hydrology_v1")
        expected = {f"{group}__{name}" for group, names in HYDROLOGY_GROUP_UNITS.items() for name in names}
        expected |= {"timestamps", "time_bounds_hours", "state_time_hours", "schema_version", "mode", "field_schema_json", "hydrology_metadata_json"}
        if set(arrays) != expected:
            raise ValueError("Hydrology checkpoint has missing or unrecognized fields")
        groups = [{name: np.asarray(arrays[f"{group}__{name}"]) for name in names} for group, names in HYDROLOGY_GROUP_UNITS.items()]
        store = cls(np.asarray(arrays["timestamps"]), np.asarray(arrays["time_bounds_hours"]), np.asarray(arrays["state_time_hours"]),
                    *groups, json.loads(str(arrays["hydrology_metadata_json"])))
        store.as_arrays()
        return store


@dataclass(frozen=True)
class StaticLandState:
    land_cover: np.ndarray
    vegetation: FloatMap
    protected: BoolMap
    buildability: FloatMap
    terrain_cost: FloatMap
    water_buffer: BoolMap
    landform: np.ndarray | None = None
    land_cover_type: np.ndarray | None = None
    allocatable_land_fraction: FloatMap | None = None

    def __post_init__(self) -> None:
        axes = (self.landform, self.land_cover_type, self.allocatable_land_fraction)
        if any(values is not None for values in axes):
            if not all(values is not None and values.shape == self.land_cover.shape for values in axes):
                raise ValueError("New land axes must all be present as HxW maps")

    @property
    def protected_mask(self) -> np.ndarray:
        return self.protected

    def as_maps(self) -> dict[str, np.ndarray]:
        maps = {
            "land_cover": self.land_cover,
            "vegetation": self.vegetation,
            "protected": self.protected,
            "buildability": self.buildability,
            "terrain_cost": self.terrain_cost,
            "water_buffer": self.water_buffer,
        }
        if self.landform is not None:
            maps.update(landform=self.landform, land_cover_type=self.land_cover_type,
                        protected_mask=self.protected_mask, allocatable_land_fraction=self.allocatable_land_fraction)
        return maps


@dataclass(frozen=True)
class ClimateBaseline:
    mean_temperature: FloatMap
    annual_temperature_amplitude: FloatMap
    mean_humidity: FloatMap
    prevailing_wind_u: FloatMap
    prevailing_wind_v: FloatMap
    mean_precipitation: FloatMap
    mean_cloud: FloatMap
    mean_irradiance: FloatMap

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "mean_temperature": self.mean_temperature,
            "annual_temperature_amplitude": self.annual_temperature_amplitude,
            "mean_humidity": self.mean_humidity,
            "prevailing_wind_u": self.prevailing_wind_u,
            "prevailing_wind_v": self.prevailing_wind_v,
            "mean_precipitation": self.mean_precipitation,
            "mean_cloud": self.mean_cloud,
            "mean_irradiance": self.mean_irradiance,
        }


@dataclass(frozen=True)
class WeatherStore:
    dynamic: np.ndarray
    weather_class: np.ndarray
    timestamps: np.ndarray
    channel_names: tuple[str, ...]
    time_unit: str = "day"
    start_day_of_year: int = 0
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    static_elevation_m: np.ndarray | None = None

    def as_arrays(self) -> dict[str, np.ndarray]:
        validate_weather_arrays(self.dynamic, self.weather_class, self.timestamps, self.channel_names, self.time_unit)
        arrays = {
            "dynamic": self.dynamic,
            "weather_class": self.weather_class,
            "timestamps": self.timestamps,
            "channel_names": np.asarray(self.channel_names),
            "time_unit": np.asarray(self.time_unit),
            "start_day_of_year": np.asarray(self.start_day_of_year, dtype=np.int32),
            "time_bounds_hours": interval_bounds_hours(self.timestamps, 1.0 if self.time_unit == "hour" else 24.0, stamp_unit=self.time_unit),
        }
        for name, values in self.diagnostics.items():
            values = np.asarray(values)
            if values.shape != self.weather_class.shape or not np.isfinite(values).all():
                raise ValueError(f"Weather diagnostic {name!r} must be finite [T,H,W]")
            arrays[f"diagnostic__{name}"] = values
        if self.static_elevation_m is not None:
            if self.static_elevation_m.shape != self.dynamic.shape[2:] or not np.isfinite(self.static_elevation_m).all():
                raise ValueError("Weather elevation must be finite [H,W]")
            arrays["static_elevation_m"] = self.static_elevation_m
        arrays["weather_metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True, allow_nan=False))
        return arrays


@dataclass(frozen=True)
class SourceLoadForecastStore:
    timestamps: np.ndarray
    bus_ids: np.ndarray
    bus_kinds: tuple[str, ...]
    p_load_mw: np.ndarray
    p_gen_available_mw: np.ndarray
    p_gen_scheduled_mw: np.ndarray
    q_load_mvar: np.ndarray
    source_channels: tuple[str, ...]
    data_semantics: str = "synthetic_realization"
    nameplate_capacity_mw: np.ndarray | None = None
    reference_load_mw: np.ndarray | None = None
    initial_effective_temperature_c: np.ndarray | None = None
    weather_sample_row: np.ndarray | None = None
    weather_sample_col: np.ndarray | None = None
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def as_arrays(self) -> dict[str, np.ndarray]:
        validate_node_arrays(self.timestamps, self.bus_ids, {
            "p_load_mw": self.p_load_mw, "p_gen_available_mw": self.p_gen_available_mw,
            "p_gen_scheduled_mw": self.p_gen_scheduled_mw, "q_load_mvar": self.q_load_mvar,
        })
        arrays = {
            "timestamps": self.timestamps,
            "bus_ids": self.bus_ids,
            "bus_kinds": np.asarray(self.bus_kinds),
            "p_load_mw": self.p_load_mw,
            "p_gen_available_mw": self.p_gen_available_mw,
            "p_gen_scheduled_mw": self.p_gen_scheduled_mw,
            "q_load_mvar": self.q_load_mvar,
            "source_channels": np.asarray(self.source_channels),
            "data_semantics": np.asarray(self.data_semantics),
            "time_bounds_hours": interval_bounds_hours(self.timestamps, 1.0),
        }
        optional = (self.nameplate_capacity_mw, self.reference_load_mw, self.initial_effective_temperature_c, self.weather_sample_row, self.weather_sample_col)
        if not any(value is not None for value in optional) and not self.diagnostics and not self.metadata:
            return arrays
        from world_generator.core.contracts import integer_labels, integrate_power_mwh
        from world_generator.core.source_load_contracts import (
            SOURCE_LOAD_SCHEMA_VERSION, SOURCE_LOAD_MODE, SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY,
            SOURCE_LOAD_ENERGY_POWER_FIELDS, SOURCE_LOAD_CF_POWER_FIELDS, source_load_field_schema,
        )
        if not all(value is not None for value in optional):
            raise ValueError("Source/load appendix requires all five node attribute arrays")
        count, hours = len(self.bus_ids), len(self.timestamps)
        kinds = np.asarray(self.bus_kinds)
        if kinds.shape != (count,):
            raise ValueError("Source/load bus kinds must match bus IDs")
        if self.metadata.get("schema_version") != SOURCE_LOAD_SCHEMA_VERSION or self.metadata.get("mode") != SOURCE_LOAD_MODE:
            raise ValueError("Modern source/load metadata must declare source_load_v1 exogenous_realization")
        if set(self.diagnostics) != set(SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY):
            raise ValueError("Modern source/load diagnostics must match the six-field source_load_v1 schema")
        expected_applicability = {name: list(values) for name, values in SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY.items()}
        if self.metadata.get("diagnostic_applicability") != expected_applicability:
            raise ValueError("Source/load diagnostic applicability must be explicit and match source_load_v1")
        grid_shape = np.asarray(self.metadata.get("weather_grid_shape", []))
        if grid_shape.shape != (2,) or np.any(integer_labels(grid_shape, "weather_grid_shape") <= 0):
            raise ValueError("Source/load metadata must identify the weather grid shape")
        for name in ("nameplate_capacity_mw", "reference_load_mw", "initial_effective_temperature_c", "weather_sample_row", "weather_sample_col"):
            values = np.asarray(getattr(self, name))
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"Source/load {name} must be finite [N]")
            if name != "initial_effective_temperature_c" and np.any(values < 0):
                raise ValueError(f"Source/load {name} must be nonnegative")
            arrays[name] = values
        for index, name in enumerate(("weather_sample_row", "weather_sample_col")):
            positions = integer_labels(arrays[name], name)
            if np.any(positions >= grid_shape[index]):
                raise ValueError("Source/load weather sample coordinates exceed the weather grid")
        load_mask = kinds == "load_bus"
        if np.any(arrays["reference_load_mw"][~load_mask] != 0) or np.any(arrays["initial_effective_temperature_c"][~load_mask] != 0):
            raise ValueError("Load reference and thermal initial state must be zero on non-load buses")
        for name, applies in SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY.items():
            values = np.asarray(self.diagnostics[name])
            if values.shape != (hours, count) or not np.isfinite(values).all():
                raise ValueError(f"Source/load diagnostic {name} must be finite [T,N]")
            if np.any(values[:, ~np.isin(kinds, applies)] != 0):
                raise ValueError(f"Source/load diagnostic {name} is nonzero outside applicable bus kinds")
            if name in {"hub_wind_speed_mps", "wind_air_density_kg_m3", "pv_poa_w_m2"} and np.any(values < 0):
                raise ValueError(f"Source/load diagnostic {name} must be nonnegative")
            arrays[f"diag__{name}"] = values
        valid = np.isin(kinds, ("wind_bus", "pv_bus", "thermal_bus")) & (arrays["nameplate_capacity_mw"] > 0)
        arrays["capacity_factor_valid"] = valid
        duration = arrays["time_bounds_hours"][:, 1] - arrays["time_bounds_hours"][:, 0]
        for name, power_name in SOURCE_LOAD_ENERGY_POWER_FIELDS.items():
            power = np.asarray(arrays[power_name], dtype=np.float64)
            arrays[name] = power * duration[:, None]
            arrays[f"period_{name}"] = integrate_power_mwh(power, duration)
        for name, power_name in SOURCE_LOAD_CF_POWER_FIELDS.items():
            arrays[name] = np.divide(arrays[power_name], arrays["nameplate_capacity_mw"][None, :],
                                     out=np.zeros((hours, count), dtype=float), where=valid[None, :])
        arrays["source_load_schema_version"] = np.asarray(SOURCE_LOAD_SCHEMA_VERSION)
        arrays["source_load_field_schema_json"] = np.asarray(json.dumps(source_load_field_schema(), sort_keys=True))
        arrays["source_load_metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True, allow_nan=False))
        return arrays

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "SourceLoadForecastStore":
        from world_generator.core.source_load_contracts import SOURCE_LOAD_SCHEMA_VERSION, source_load_field_schema

        base = ("timestamps", "bus_ids", "bus_kinds", "p_load_mw", "p_gen_available_mw", "p_gen_scheduled_mw", "q_load_mvar", "source_channels")
        if not set(base).issubset(arrays):
            raise ValueError("Source/load checkpoint is missing required power or identity fields")
        bounds = interval_bounds_hours(arrays["timestamps"], 1.0)
        if "time_bounds_hours" in arrays and (np.asarray(arrays["time_bounds_hours"]).shape != bounds.shape or not np.allclose(arrays["time_bounds_hours"], bounds, rtol=0, atol=1e-9)):
            raise ValueError("Source/load checkpoint has inconsistent interval bounds")
        kwargs = {}
        schema = source_load_field_schema()
        modern = any(name in arrays for name in ("source_load_schema_version", "source_load_metadata_json", "source_load_field_schema_json")) or any(name in arrays for name in schema) or any(name.startswith("diag__") for name in arrays)
        if modern:
            if str(arrays.get("source_load_schema_version")) != SOURCE_LOAD_SCHEMA_VERSION or json.loads(str(arrays.get("source_load_field_schema_json", "null"))) != schema:
                raise ValueError("Source/load appendix schema is missing or inconsistent")
            expected = set(base) | set(schema) | {"data_semantics", "time_bounds_hours", "source_load_schema_version", "source_load_field_schema_json", "source_load_metadata_json"}
            if set(arrays) != expected:
                raise ValueError("Source/load appendix is incomplete or has unknown fields")
            kwargs = {name: np.asarray(arrays[name]) for name in ("nameplate_capacity_mw", "reference_load_mw", "initial_effective_temperature_c", "weather_sample_row", "weather_sample_col")}
            kwargs["diagnostics"] = {name.removeprefix("diag__"): np.asarray(arrays[name]) for name in schema if name.startswith("diag__")}
            kwargs["metadata"] = json.loads(str(arrays["source_load_metadata_json"]))
        store = cls(np.asarray(arrays["timestamps"]), np.asarray(arrays["bus_ids"]), tuple(str(item) for item in arrays["bus_kinds"]),
                    *(np.asarray(arrays[name]) for name in ("p_load_mw", "p_gen_available_mw", "p_gen_scheduled_mw", "q_load_mvar")),
                    tuple(str(item) for item in arrays["source_channels"]), str(arrays.get("data_semantics", "synthetic_realization")), **kwargs)
        checked = store.as_arrays()
        if modern:
            for name in schema:
                if np.asarray(arrays[name]).shape != np.asarray(checked[name]).shape or not np.allclose(arrays[name], checked[name], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Source/load serialized accounting field {name} disagrees with its power basis")
        return store

    def summary_dict(self) -> dict[str, float | int | str | list[str]]:
        total_load = self.p_load_mw.sum(axis=1)
        total_available = self.p_gen_available_mw.sum(axis=1)
        total_scheduled = self.p_gen_scheduled_mw.sum(axis=1)
        return {
            "data_semantics": self.data_semantics,
            "hours": int(self.p_load_mw.shape[0]),
            "bus_count": int(self.p_load_mw.shape[1]),
            "bus_kinds": sorted(set(self.bus_kinds)),
            "peak_load_mw": float(np.max(total_load)),
            "mean_load_mw": float(np.mean(total_load)),
            "peak_available_generation_mw": float(np.max(total_available)),
            "mean_available_generation_mw": float(np.mean(total_available)),
            "peak_scheduled_generation_mw": float(np.max(total_scheduled)),
            "mean_scheduled_generation_mw": float(np.mean(total_scheduled)),
        }


@dataclass(frozen=True)
class PowerFlowStore:
    timestamps: np.ndarray
    bus_ids: np.ndarray
    branch_ids: np.ndarray
    bus_angle_rad: np.ndarray
    bus_p_injection_mw: np.ndarray
    served_load_mw: np.ndarray
    dispatched_generation_mw: np.ndarray
    unserved_load_mw: np.ndarray
    curtailed_generation_mw: np.ndarray
    line_flow_mw: np.ndarray
    line_loading_ratio: np.ndarray
    slack_bus_id: int

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "timestamps": self.timestamps,
            "bus_ids": self.bus_ids,
            "branch_ids": self.branch_ids,
            "bus_angle_rad": self.bus_angle_rad,
            "bus_p_injection_mw": self.bus_p_injection_mw,
            "served_load_mw": self.served_load_mw,
            "dispatched_generation_mw": self.dispatched_generation_mw,
            "unserved_load_mw": self.unserved_load_mw,
            "curtailed_generation_mw": self.curtailed_generation_mw,
            "line_flow_mw": self.line_flow_mw,
            "line_loading_ratio": self.line_loading_ratio,
            "slack_bus_id": np.asarray(self.slack_bus_id, dtype=np.int32),
        }

    def summary_dict(self) -> dict[str, float | int]:
        peak_loading = self.line_loading_ratio.max(axis=0) if self.line_loading_ratio.size else np.asarray([], dtype=np.float32)
        return {
            "hours": int(self.bus_p_injection_mw.shape[0]),
            "bus_count": int(self.bus_p_injection_mw.shape[1]),
            "branch_count": int(self.line_flow_mw.shape[1]),
            "slack_bus_id": int(self.slack_bus_id),
            "peak_line_loading_ratio": float(np.max(peak_loading)) if peak_loading.size else 0.0,
            "line_hours_over_80pct": int((self.line_loading_ratio > 0.80).sum()),
            "line_hours_over_100pct": int((self.line_loading_ratio > 1.00).sum()),
            "total_unserved_load_mwh": float(np.sum(self.unserved_load_mw)),
            "total_curtailed_generation_mwh": float(np.sum(self.curtailed_generation_mw)),
        }


@dataclass(frozen=True)
class StorageNeedStore:
    timestamps: np.ndarray
    bus_ids: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    support_requirement_mw: np.ndarray
    unserved_load_mw: np.ndarray
    congestion_support_mw: np.ndarray
    peak_support_mw: np.ndarray
    total_support_energy_mwh: np.ndarray
    max_event_energy_mwh: np.ndarray
    positive_ramp_p95_mw: np.ndarray
    congestion_exposure_hours: np.ndarray
    suggested_power_mw: np.ndarray
    suggested_energy_mwh: np.ndarray
    need_score: np.ndarray
    need_score_map: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "timestamps": self.timestamps,
            "bus_ids": self.bus_ids,
            "rows": self.rows,
            "cols": self.cols,
            "support_requirement_mw": self.support_requirement_mw,
            "unserved_load_mw": self.unserved_load_mw,
            "congestion_support_mw": self.congestion_support_mw,
            "peak_support_mw": self.peak_support_mw,
            "total_support_energy_mwh": self.total_support_energy_mwh,
            "max_event_energy_mwh": self.max_event_energy_mwh,
            "positive_ramp_p95_mw": self.positive_ramp_p95_mw,
            "congestion_exposure_hours": self.congestion_exposure_hours,
            "suggested_power_mw": self.suggested_power_mw,
            "suggested_energy_mwh": self.suggested_energy_mwh,
            "need_score": self.need_score,
            "need_score_map": self.need_score_map,
        }

    def as_dicts(self) -> list[dict[str, float | int]]:
        order = np.argsort(-self.need_score)
        return [
            {
                "bus_id": int(self.bus_ids[index]),
                "row": int(self.rows[index]),
                "col": int(self.cols[index]),
                "need_score": float(self.need_score[index]),
                "suggested_power_mw": float(self.suggested_power_mw[index]),
                "suggested_energy_mwh": float(self.suggested_energy_mwh[index]),
                "peak_support_mw": float(self.peak_support_mw[index]),
                "total_support_energy_mwh": float(self.total_support_energy_mwh[index]),
                "max_event_energy_mwh": float(self.max_event_energy_mwh[index]),
                "positive_ramp_p95_mw": float(self.positive_ramp_p95_mw[index]),
                "congestion_exposure_hours": float(self.congestion_exposure_hours[index]),
                "unserved_energy_mwh": float(self.unserved_load_mw[:, index].sum()),
            }
            for index in order
        ]

    def summary_dict(self) -> dict[str, float | int]:
        return {
            "load_bus_count": int(self.bus_ids.size),
            "hours": int(self.timestamps.size),
            "total_suggested_power_mw": float(self.suggested_power_mw.sum()),
            "total_suggested_energy_mwh": float(self.suggested_energy_mwh.sum()),
            "total_unserved_energy_mwh": float(self.unserved_load_mw.sum()),
            "total_congestion_support_mwh": float(self.congestion_support_mw.sum()),
            "peak_aggregate_support_mw": float(self.support_requirement_mw.sum(axis=1).max(initial=0.0)),
        }


@dataclass(frozen=True)
class StorageSite:
    site_id: int
    bus_id: int
    row: int
    col: int
    covered_bus_ids: tuple[int, ...]
    power_mw: float
    energy_mwh: float
    initial_soc_mwh: float
    need_score: float


@dataclass(frozen=True)
class StoragePlanStore:
    timestamps: np.ndarray
    load_bus_ids: np.ndarray
    assigned_site_ids: np.ndarray
    site_support_requirement_mw: np.ndarray
    sites: tuple[StorageSite, ...]

    def as_arrays(self) -> dict[str, np.ndarray]:
        site_rows = np.asarray(
            [
                [
                    site.site_id,
                    site.bus_id,
                    site.row,
                    site.col,
                    site.power_mw,
                    site.energy_mwh,
                    site.initial_soc_mwh,
                    site.need_score,
                    len(site.covered_bus_ids),
                ]
                for site in self.sites
            ],
            dtype=np.float32,
        ).reshape(-1, 9)
        return {
            "timestamps": self.timestamps,
            "load_bus_ids": self.load_bus_ids,
            "assigned_site_ids": self.assigned_site_ids,
            "site_support_requirement_mw": self.site_support_requirement_mw,
            "storage_sites": site_rows,
        }

    def as_dicts(self) -> list[dict[str, float | int | list[int]]]:
        return [
            {
                "site_id": site.site_id,
                "bus_id": site.bus_id,
                "row": site.row,
                "col": site.col,
                "covered_bus_ids": list(site.covered_bus_ids),
                "power_mw": site.power_mw,
                "energy_mwh": site.energy_mwh,
                "initial_soc_mwh": site.initial_soc_mwh,
                "need_score": site.need_score,
            }
            for site in self.sites
        ]

    def summary_dict(self) -> dict[str, float | int]:
        return {
            "site_count": len(self.sites),
            "covered_load_bus_count": int(np.sum(self.assigned_site_ids >= 0)),
            "total_power_mw": float(sum(site.power_mw for site in self.sites)),
            "total_energy_mwh": float(sum(site.energy_mwh for site in self.sites)),
            "total_initial_soc_mwh": float(sum(site.initial_soc_mwh for site in self.sites)),
        }


@dataclass(frozen=True)
class StorageDispatchStore:
    timestamps: np.ndarray
    site_ids: np.ndarray
    site_bus_ids: np.ndarray
    site_power_capacity_mw: np.ndarray
    site_energy_capacity_mwh: np.ndarray
    storage_power_expansion_mw: np.ndarray
    storage_energy_expansion_mwh: np.ndarray
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    emergency_discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    target_soc_mwh: np.ndarray
    cycle_boundary_soc_mwh: np.ndarray
    minimum_soc_fraction: float
    maximum_soc_fraction: float
    preferred_soc_lower_fraction: float
    preferred_soc_upper_fraction: float
    total_load_mw: np.ndarray
    renewable_available_mw: np.ndarray
    baseline_thermal_mw: np.ndarray
    scheduled_thermal_mw: np.ndarray
    baseline_unserved_mw: np.ndarray
    dispatched_unserved_mw: np.ndarray
    baseline_curtailed_mw: np.ndarray
    dispatched_curtailed_mw: np.ndarray
    branch_ids: np.ndarray
    line_capacity_expansion_mva: np.ndarray
    thermal_bus_ids: np.ndarray
    thermal_capacity_expansion_mw: np.ndarray
    baseline_line_loading_ratio: np.ndarray
    dispatched_line_loading_ratio: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "timestamps": self.timestamps,
            "site_ids": self.site_ids,
            "site_bus_ids": self.site_bus_ids,
            "site_power_capacity_mw": self.site_power_capacity_mw,
            "site_energy_capacity_mwh": self.site_energy_capacity_mwh,
            "storage_power_expansion_mw": self.storage_power_expansion_mw,
            "storage_energy_expansion_mwh": self.storage_energy_expansion_mwh,
            "charge_mw": self.charge_mw,
            "discharge_mw": self.discharge_mw,
            "emergency_discharge_mw": self.emergency_discharge_mw,
            "soc_mwh": self.soc_mwh,
            "target_soc_mwh": self.target_soc_mwh,
            "cycle_boundary_soc_mwh": self.cycle_boundary_soc_mwh,
            "minimum_soc_fraction": np.asarray(self.minimum_soc_fraction, dtype=np.float32),
            "maximum_soc_fraction": np.asarray(self.maximum_soc_fraction, dtype=np.float32),
            "preferred_soc_lower_fraction": np.asarray(self.preferred_soc_lower_fraction, dtype=np.float32),
            "preferred_soc_upper_fraction": np.asarray(self.preferred_soc_upper_fraction, dtype=np.float32),
            "total_load_mw": self.total_load_mw,
            "renewable_available_mw": self.renewable_available_mw,
            "baseline_thermal_mw": self.baseline_thermal_mw,
            "scheduled_thermal_mw": self.scheduled_thermal_mw,
            "baseline_unserved_mw": self.baseline_unserved_mw,
            "dispatched_unserved_mw": self.dispatched_unserved_mw,
            "baseline_curtailed_mw": self.baseline_curtailed_mw,
            "dispatched_curtailed_mw": self.dispatched_curtailed_mw,
            "branch_ids": self.branch_ids,
            "line_capacity_expansion_mva": self.line_capacity_expansion_mva,
            "thermal_bus_ids": self.thermal_bus_ids,
            "thermal_capacity_expansion_mw": self.thermal_capacity_expansion_mw,
            "baseline_line_loading_ratio": self.baseline_line_loading_ratio,
            "dispatched_line_loading_ratio": self.dispatched_line_loading_ratio,
        }

    def summary_dict(self) -> dict[str, float | int]:
        baseline_overload = int(np.sum(self.baseline_line_loading_ratio > 1.0))
        dispatched_overload = int(np.sum(self.dispatched_line_loading_ratio > 1.0))
        baseline_unserved = float(np.sum(self.baseline_unserved_mw))
        dispatched_unserved = float(np.sum(self.dispatched_unserved_mw))
        charged_energy = float(np.sum(self.charge_mw))
        discharged_energy = float(np.sum(self.discharge_mw))
        stored_energy_change = (
            float(np.sum(self.soc_mwh[-1]) - np.sum(self.soc_mwh[0]))
            if self.soc_mwh.size
            else 0.0
        )
        installed_energy = max(float(self.site_energy_capacity_mwh.sum()), 1e-6)
        baseline_thermal = float(np.sum(self.baseline_thermal_mw))
        scheduled_thermal = float(np.sum(self.scheduled_thermal_mw))
        return {
            "hours": int(self.timestamps.size),
            "site_count": int(self.site_ids.size),
            "charged_energy_mwh": charged_energy,
            "discharged_energy_mwh": discharged_energy,
            "fast_reserve_discharge_energy_mwh": float(np.sum(self.emergency_discharge_mw)),
            "fast_reserve_discharge_hours": int(np.sum(self.emergency_discharge_mw.sum(axis=1) > 1e-4)),
            "soc_time_in_preferred_band_fraction": float(
                np.mean(
                    (self.soc_mwh >= self.preferred_soc_lower_fraction * self.site_energy_capacity_mwh[None, :])
                    & (self.soc_mwh <= self.preferred_soc_upper_fraction * self.site_energy_capacity_mwh[None, :])
                )
            ) if self.soc_mwh.size else 0.0,
            "storage_conversion_loss_mwh": charged_energy - discharged_energy - stored_energy_change,
            "equivalent_full_cycles": discharged_energy / installed_energy,
            "baseline_thermal_energy_mwh": baseline_thermal,
            "scheduled_thermal_energy_mwh": scheduled_thermal,
            "additional_thermal_energy_mwh": scheduled_thermal - baseline_thermal,
            "baseline_unserved_energy_mwh": baseline_unserved,
            "dispatched_unserved_energy_mwh": dispatched_unserved,
            "unserved_energy_reduction_mwh": baseline_unserved - dispatched_unserved,
            "baseline_curtailed_energy_mwh": float(np.sum(self.baseline_curtailed_mw)),
            "dispatched_curtailed_energy_mwh": float(np.sum(self.dispatched_curtailed_mw)),
            "baseline_line_hours_over_100pct": baseline_overload,
            "dispatched_line_hours_over_100pct": dispatched_overload,
            "line_overload_hour_reduction": baseline_overload - dispatched_overload,
            "peak_baseline_line_loading_ratio": float(np.max(self.baseline_line_loading_ratio, initial=0.0)),
            "peak_dispatched_line_loading_ratio": float(np.max(self.dispatched_line_loading_ratio, initial=0.0)),
            "final_total_soc_mwh": float(np.sum(self.soc_mwh[-1])) if self.soc_mwh.size else 0.0,
            "initial_total_soc_mwh": float(np.sum(self.soc_mwh[0])) if self.soc_mwh.size else 0.0,
            "cycle_boundary_soc_mwh": float(np.sum(self.cycle_boundary_soc_mwh)),
            "cycle_boundary_mismatch_mwh": float(
                np.abs(self.soc_mwh[-1] - self.soc_mwh[0]).sum()
            ) if self.soc_mwh.size else 0.0,
            "added_thermal_capacity_mw": float(self.thermal_capacity_expansion_mw.sum()),
            "expanded_thermal_bus_count": int(np.sum(self.thermal_capacity_expansion_mw > 1e-4)),
            "added_storage_power_mw": float(self.storage_power_expansion_mw.sum()),
            "added_storage_energy_mwh": float(self.storage_energy_expansion_mwh.sum()),
            "added_line_capacity_mva": float(self.line_capacity_expansion_mva.sum()),
            "reinforced_line_count": int(np.sum(self.line_capacity_expansion_mva > 1e-4)),
        }

    def expansion_dict(self) -> dict[str, list[dict[str, float | int]]]:
        return {
            "thermal": [
                {"bus_id": int(bus_id), "added_capacity_mw": float(expansion)}
                for bus_id, expansion in zip(self.thermal_bus_ids, self.thermal_capacity_expansion_mw)
                if expansion > 1e-4
            ],
            "storage": [
                {
                    "site_id": int(site_id),
                    "bus_id": int(bus_id),
                    "added_power_mw": float(power),
                    "added_energy_mwh": float(energy),
                }
                for site_id, bus_id, power, energy in zip(
                    self.site_ids,
                    self.site_bus_ids,
                    self.storage_power_expansion_mw,
                    self.storage_energy_expansion_mwh,
                )
                if max(power, energy) > 1e-4
            ],
            "lines": [
                {"branch_id": int(branch_id), "added_capacity_mva": float(expansion)}
                for branch_id, expansion in zip(self.branch_ids, self.line_capacity_expansion_mva)
                if expansion > 1e-4
            ],
        }


@dataclass(frozen=True)
class GridUpgradePlanStore:
    branch_ids: np.ndarray
    current_rate_mva: np.ndarray
    recommended_rate_mva: np.ndarray
    upgrade_factor: np.ndarray
    peak_loading_ratio: np.ndarray
    p95_loading_ratio: np.ndarray
    hours_over_80pct: np.ndarray
    hours_over_100pct: np.ndarray
    overload_mwh_proxy: np.ndarray
    priority_score: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "branch_ids": self.branch_ids,
            "current_rate_mva": self.current_rate_mva,
            "recommended_rate_mva": self.recommended_rate_mva,
            "upgrade_factor": self.upgrade_factor,
            "peak_loading_ratio": self.peak_loading_ratio,
            "p95_loading_ratio": self.p95_loading_ratio,
            "hours_over_80pct": self.hours_over_80pct,
            "hours_over_100pct": self.hours_over_100pct,
            "overload_mwh_proxy": self.overload_mwh_proxy,
            "priority_score": self.priority_score,
        }

    def as_dicts(self) -> list[dict[str, float | int]]:
        rows = []
        order = np.argsort(-self.priority_score)
        for index in order:
            rows.append(
                {
                    "branch_id": int(self.branch_ids[index]),
                    "current_rate_mva": float(self.current_rate_mva[index]),
                    "recommended_rate_mva": float(self.recommended_rate_mva[index]),
                    "upgrade_factor": float(self.upgrade_factor[index]),
                    "peak_loading_ratio": float(self.peak_loading_ratio[index]),
                    "p95_loading_ratio": float(self.p95_loading_ratio[index]),
                    "hours_over_80pct": int(self.hours_over_80pct[index]),
                    "hours_over_100pct": int(self.hours_over_100pct[index]),
                    "overload_mwh_proxy": float(self.overload_mwh_proxy[index]),
                    "priority_score": float(self.priority_score[index]),
                }
            )
        return rows

    def summary_dict(self) -> dict[str, float | int]:
        needs_upgrade = self.upgrade_factor > 1.01
        return {
            "branch_count": int(self.branch_ids.size),
            "recommended_upgrade_count": int(needs_upgrade.sum()),
            "max_upgrade_factor": float(self.upgrade_factor.max()) if self.upgrade_factor.size else 1.0,
            "mean_upgrade_factor_for_recommended": float(self.upgrade_factor[needs_upgrade].mean()) if needs_upgrade.any() else 1.0,
            "total_added_rate_mva": float(np.maximum(self.recommended_rate_mva - self.current_rate_mva, 0.0).sum()),
            "top_priority_score": float(self.priority_score.max()) if self.priority_score.size else 0.0,
        }


@dataclass(frozen=True)
class CityNode:
    city_id: int
    row: int
    col: int
    x: float
    y: float
    population: float
    radius_km: float
    suitability: float


@dataclass(frozen=True)
class CityState:
    cities: tuple[CityNode, ...]
    city_suitability: FloatMap
    urban_core_suitability: FloatMap
    waterfront_amenity: FloatMap
    population_density: FloatMap  # persons / km^2; sum * cell area equals total population
    economic_activity: FloatMap
    urban_density: FloatMap
    urban_mask: BoolMap
    city_id_map: np.ndarray
    population_budget: dict[str, float | int] = field(default_factory=dict)

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "city_suitability": self.city_suitability,
            "urban_core_suitability": self.urban_core_suitability,
            "waterfront_amenity": self.waterfront_amenity,
            "population_density": self.population_density,
            "economic_activity": self.economic_activity,
            "urban_density": self.urban_density,
            "urban_mask": self.urban_mask,
            "city_id_map": self.city_id_map,
        }

    def cities_as_dicts(self) -> list[dict[str, float | int]]:
        return [
            {
                "city_id": city.city_id,
                "row": city.row,
                "col": city.col,
                "x": city.x,
                "y": city.y,
                "population": city.population,
                "radius_km": city.radius_km,
                "suitability": city.suitability,
            }
            for city in self.cities
        ]


@dataclass(frozen=True)
class LandUseState:
    residential: FloatMap
    commercial: FloatMap
    industrial: FloatMap
    agriculture: FloatMap
    park_green: FloatMap
    load_density_base: FloatMap
    land_use_zone: np.ndarray
    use_fractions: dict[str, np.ndarray] = field(default_factory=dict)

    def as_maps(self) -> dict[str, np.ndarray]:
        maps = {
            "residential": self.residential,
            "commercial": self.commercial,
            "industrial": self.industrial,
            "agriculture": self.agriculture,
            "park_green": self.park_green,
            "load_density_base": self.load_density_base,
            "land_use_zone": self.land_use_zone,
        }
        maps.update({f"land_use_fraction_{name}": values for name, values in self.use_fractions.items()})
        return maps


@dataclass(frozen=True)
class EnergyCandidate:
    candidate_id: int
    kind: str
    row: int
    col: int
    x: float
    y: float
    capacity_mw: float
    suitability: float


@dataclass(frozen=True)
class EnergyCandidateState:
    wind_suitability: FloatMap
    pv_suitability: FloatMap
    load_node_density: FloatMap
    wind_candidate_map: np.ndarray
    pv_candidate_map: np.ndarray
    source_candidate_map: np.ndarray
    load_candidate_map: np.ndarray
    wind_candidates: tuple[EnergyCandidate, ...]
    pv_candidates: tuple[EnergyCandidate, ...]
    load_candidates: tuple[EnergyCandidate, ...]
    land_accounting_maps: dict[str, np.ndarray] = field(default_factory=dict)
    project_land_ledger: np.ndarray | None = None
    project_area_by_cell_km2: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_project_land_arrays(self.project_land_ledger, self.project_area_by_cell_km2,
                                      self.wind_suitability.shape, "energy")

    def as_maps(self) -> dict[str, np.ndarray]:
        maps = {
            "wind_suitability": self.wind_suitability,
            "pv_suitability": self.pv_suitability,
            "load_node_density": self.load_node_density,
            "wind_candidate_map": self.wind_candidate_map,
            "pv_candidate_map": self.pv_candidate_map,
            "source_candidate_map": self.source_candidate_map,
            "load_candidate_map": self.load_candidate_map,
        }
        return maps | self.land_accounting_maps

    def candidates_as_dicts(self) -> dict[str, list[dict[str, float | int | str]]]:
        return {
            "wind": [_candidate_as_dict(item) for item in self.wind_candidates],
            "pv": [_candidate_as_dict(item) for item in self.pv_candidates],
            "load": [_candidate_as_dict(item) for item in self.load_candidates],
        }

    def candidates_as_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "wind_candidates": _candidates_to_array(self.wind_candidates),
            "pv_candidates": _candidates_to_array(self.pv_candidates),
            "load_candidates": _candidates_to_array(self.load_candidates),
        }
        if self.project_land_ledger is not None:
            arrays.update(energy_project_land_ledger=self.project_land_ledger,
                          energy_project_area_by_cell_km2=self.project_area_by_cell_km2,
                          energy_project_land_columns=np.asarray(ENERGY_PROJECT_LAND_COLUMNS))
        return arrays


ENERGY_PROJECT_LAND_COLUMNS = (
    "project_id", "technology_code", "candidate_id", "row", "col", "reserved_area_km2",
    "capacity_mw", "capacity_density_mw_km2", "capacity_equivalent_area_km2",
)


def _validate_project_land_arrays(ledger: np.ndarray | None, cube: np.ndarray | None,
                                  shape: tuple[int, ...], label: str) -> None:
    if (ledger is None) != (cube is None):
        raise ValueError(f"{label} project ledger and area cube must be supplied together")
    if ledger is None:
        return
    if ledger.ndim != 2 or ledger.shape[1] != 9 or cube.shape != (ledger.shape[0], *shape):
        raise ValueError(f"{label} project ledger must be Nx9 and area cube NxHxW")
    if not np.isfinite(ledger).all() or not np.isfinite(cube).all() or np.any(cube < 0):
        raise ValueError(f"{label} project ledger and area cube must be finite with nonnegative area")


def _candidate_as_dict(candidate: EnergyCandidate) -> dict[str, float | int | str]:
    return {
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "row": candidate.row,
        "col": candidate.col,
        "x": candidate.x,
        "y": candidate.y,
        "capacity_mw": candidate.capacity_mw,
        "suitability": candidate.suitability,
    }


def _candidates_to_array(candidates: tuple[EnergyCandidate, ...]) -> np.ndarray:
    rows = [
        [
            item.candidate_id,
            item.row,
            item.col,
            item.x,
            item.y,
            item.capacity_mw,
            item.suitability,
        ]
        for item in candidates
    ]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 7)


@dataclass(frozen=True)
class GridBus:
    bus_id: int
    kind: str
    row: int
    col: int
    x: float
    y: float
    capacity_mw: float
    suitability: float
    externality_score: float
    source_kind: str
    source_id: int


@dataclass(frozen=True)
class GridNodeState:
    thermal_suitability: FloatMap
    thermal_externality: FloatMap
    bus_site_map: np.ndarray
    load_bus_map: np.ndarray
    source_bus_map: np.ndarray
    thermal_bus_map: np.ndarray
    buses: tuple[GridBus, ...]
    land_accounting_maps: dict[str, np.ndarray] = field(default_factory=dict)
    thermal_project_land_ledger: np.ndarray | None = None
    thermal_project_area_by_cell_km2: np.ndarray | None = None
    thermal_land_accounting_mode: str = "legacy_no_land_guarantee"

    def __post_init__(self) -> None:
        _validate_project_land_arrays(self.thermal_project_land_ledger, self.thermal_project_area_by_cell_km2,
                                      self.thermal_suitability.shape, "thermal")
        if self.thermal_land_accounting_mode != "legacy_no_land_guarantee" and self.thermal_project_land_ledger is None:
            raise ValueError("Modern thermal land accounting requires a project ledger and area cube")

    def as_maps(self) -> dict[str, np.ndarray]:
        maps = {
            "thermal_suitability": self.thermal_suitability,
            "thermal_externality": self.thermal_externality,
            "bus_site_map": self.bus_site_map,
            "load_bus_map": self.load_bus_map,
            "source_bus_map": self.source_bus_map,
            "thermal_bus_map": self.thermal_bus_map,
        }
        return maps | self.land_accounting_maps

    def buses_as_dicts(self) -> list[dict[str, float | int | str]]:
        return [_bus_as_dict(item) for item in self.buses]

    def buses_as_arrays(self) -> dict[str, np.ndarray]:
        arrays = {"grid_buses": _buses_to_array(self.buses), "thermal_land_accounting_mode": np.asarray(self.thermal_land_accounting_mode)}
        if self.thermal_project_land_ledger is not None:
            arrays.update(thermal_land_ledger=self.thermal_project_land_ledger,
                          thermal_project_area_by_cell_km2=self.thermal_project_area_by_cell_km2,
                          thermal_land_columns=np.asarray(("bus_id", "row", "col", "reserved_area_km2", "capacity_mw", "capacity_density_mw_km2", "capacity_equivalent_area_km2", "land_capacity_upper_bound_mw", "requested_capacity_mw")))
        return arrays


def _bus_as_dict(bus: GridBus) -> dict[str, float | int | str]:
    return {
        "bus_id": bus.bus_id,
        "kind": bus.kind,
        "row": bus.row,
        "col": bus.col,
        "x": bus.x,
        "y": bus.y,
        "capacity_mw": bus.capacity_mw,
        "suitability": bus.suitability,
        "externality_score": bus.externality_score,
        "source_kind": bus.source_kind,
        "source_id": bus.source_id,
    }


def _buses_to_array(buses: tuple[GridBus, ...]) -> np.ndarray:
    rows = [
        [
            item.bus_id,
            item.row,
            item.col,
            item.x,
            item.y,
            item.capacity_mw,
            item.suitability,
            item.externality_score,
            item.source_id,
        ]
        for item in buses
    ]
    return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class GridEdge:
    edge_id: int
    from_bus: int
    to_bus: int
    length_km: float
    route_cost: float
    is_redundant: bool
    path_rows: tuple[int, ...]
    path_cols: tuple[int, ...]


@dataclass(frozen=True)
class GridTopologyState:
    routing_cost: FloatMap
    line_route_map: FloatMap
    grid_edge_map: np.ndarray
    edges: tuple[GridEdge, ...]

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "routing_cost": self.routing_cost,
            "line_route_map": self.line_route_map,
            "grid_edge_map": self.grid_edge_map,
        }

    def edges_as_dicts(self) -> list[dict[str, float | int | bool | list[int]]]:
        return [_edge_as_dict(item) for item in self.edges]

    def edges_as_arrays(self) -> dict[str, np.ndarray]:
        return {"grid_edges": _edges_to_array(self.edges)}


@dataclass(frozen=True)
class RefinedGridTopologyState:
    refined_line_route_map: FloatMap
    refined_grid_edge_map: np.ndarray
    transit_bus_map: np.ndarray
    refined_buses: tuple[GridBus, ...]
    refined_edges: tuple[GridEdge, ...]

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "refined_line_route_map": self.refined_line_route_map,
            "refined_grid_edge_map": self.refined_grid_edge_map,
            "transit_bus_map": self.transit_bus_map,
        }

    def buses_as_dicts(self) -> list[dict[str, float | int | str]]:
        return [_bus_as_dict(item) for item in self.refined_buses]

    def edges_as_dicts(self) -> list[dict[str, float | int | bool | list[int]]]:
        return [_edge_as_dict(item) for item in self.refined_edges]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "refined_grid_buses": _buses_to_array(self.refined_buses),
            "refined_grid_edges": _edges_to_array(self.refined_edges),
        }


@dataclass(frozen=True)
class BusElectricalParam:
    bus_id: int
    kind: str
    nominal_kv: float
    p_capacity_mw: float
    q_capacity_mvar: float
    base_load_mw: float
    power_factor: float
    voltage_setpoint_pu: float
    control_mode: str


@dataclass(frozen=True)
class BranchElectricalParam:
    edge_id: int
    from_bus: int
    to_bus: int
    nominal_kv: float
    length_km: float
    r_ohm: float
    x_ohm: float
    b_us: float
    rate_mva: float
    is_redundant: bool


@dataclass(frozen=True)
class GridElectricalState:
    bus_params: tuple[BusElectricalParam, ...]
    branch_params: tuple[BranchElectricalParam, ...]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "electrical_buses": _bus_electrical_to_array(self.bus_params),
            "electrical_branches": _branch_electrical_to_array(self.branch_params),
        }

    def as_dicts(self) -> dict[str, list[dict[str, float | int | bool | str]]]:
        return {
            "buses": [_bus_electrical_as_dict(item) for item in self.bus_params],
            "branches": [_branch_electrical_as_dict(item) for item in self.branch_params],
        }


def _edge_as_dict(edge: GridEdge) -> dict[str, float | int | bool | list[int]]:
    return {
        "edge_id": edge.edge_id,
        "from_bus": edge.from_bus,
        "to_bus": edge.to_bus,
        "length_km": edge.length_km,
        "route_cost": edge.route_cost,
        "is_redundant": edge.is_redundant,
        "path_rows": list(edge.path_rows),
        "path_cols": list(edge.path_cols),
    }


def _edges_to_array(edges: tuple[GridEdge, ...]) -> np.ndarray:
    rows = [
        [
            item.edge_id,
            item.from_bus,
            item.to_bus,
            item.length_km,
            item.route_cost,
            float(item.is_redundant),
        ]
        for item in edges
    ]
    return np.asarray(rows, dtype=np.float32)


def _bus_electrical_as_dict(bus: BusElectricalParam) -> dict[str, float | int | str]:
    return {
        "bus_id": bus.bus_id,
        "kind": bus.kind,
        "nominal_kv": bus.nominal_kv,
        "p_capacity_mw": bus.p_capacity_mw,
        "q_capacity_mvar": bus.q_capacity_mvar,
        "base_load_mw": bus.base_load_mw,
        "power_factor": bus.power_factor,
        "voltage_setpoint_pu": bus.voltage_setpoint_pu,
        "control_mode": bus.control_mode,
    }


def _branch_electrical_as_dict(branch: BranchElectricalParam) -> dict[str, float | int | bool]:
    return {
        "edge_id": branch.edge_id,
        "from_bus": branch.from_bus,
        "to_bus": branch.to_bus,
        "nominal_kv": branch.nominal_kv,
        "length_km": branch.length_km,
        "r_ohm": branch.r_ohm,
        "x_ohm": branch.x_ohm,
        "b_us": branch.b_us,
        "rate_mva": branch.rate_mva,
        "is_redundant": branch.is_redundant,
    }


def _bus_electrical_to_array(buses: tuple[BusElectricalParam, ...]) -> np.ndarray:
    rows = [
        [
            item.bus_id,
            item.nominal_kv,
            item.p_capacity_mw,
            item.q_capacity_mvar,
            item.base_load_mw,
            item.power_factor,
            item.voltage_setpoint_pu,
        ]
        for item in buses
    ]
    return np.asarray(rows, dtype=np.float32)


def _branch_electrical_to_array(branches: tuple[BranchElectricalParam, ...]) -> np.ndarray:
    rows = [
        [
            item.edge_id,
            item.from_bus,
            item.to_bus,
            item.nominal_kv,
            item.length_km,
            item.r_ohm,
            item.x_ohm,
            item.b_us,
            item.rate_mva,
            float(item.is_redundant),
        ]
        for item in branches
    ]
    return np.asarray(rows, dtype=np.float32)
