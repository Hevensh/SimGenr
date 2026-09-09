"""P: dimensional bookkeeping. Schema choices are S, not new physical laws.

All integrations use interval means/rates and explicit durations. Accumulated
rain depths are summed, never integrated a second time. No data are repaired.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from world_generator.core.hydrology_contracts import hydrology_field_schema

GENERATOR_VERSION = "physics_v4"
CONTRACT_VERSION = "1.0"


def positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def finite_array(values: np.ndarray, name: str, *, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        bad = tuple(np.argwhere(~np.isfinite(result))[0]) if result.ndim else ()
        raise ValueError(f"{name}: non-finite value at {bad}")
    if nonnegative and np.any(result < 0):
        bad = tuple(np.argwhere(result < 0)[0]) if result.ndim else ()
        raise ValueError(f"{name}: negative value at {bad}")
    return result


def cell_area_km2(cell_size_km: float) -> float:
    """P: square projected cells; geodesic areas need a different grid model."""
    return positive_finite(cell_size_km, "cell_size_km") ** 2


def integer_labels(values: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.dtype.kind not in "iuf" or not np.isfinite(labels).all():
        raise ValueError(f"{name} must contain finite integer labels")
    if labels.dtype.kind == "f" and (np.any(labels != np.rint(labels)) or np.any(np.abs(labels) > 2**53)):
        raise ValueError(f"{name} must contain exactly representable integer labels")
    if labels.size and (int(labels.min()) < -(2**63) or int(labels.max()) >= 2**63):
        raise ValueError(f"{name} exceeds signed 64-bit label range")
    return labels.astype(np.int64)


def entity_ids(values: np.ndarray, name: str = "bus_ids") -> np.ndarray:
    ids = integer_labels(values, name)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must be one-dimensional and unique")
    return ids


def depth_mm_to_volume_m3(depth_mm: np.ndarray, area_km2: np.ndarray | float) -> np.ndarray:
    depth = finite_array(depth_mm, "depth_mm", nonnegative=True)
    area = finite_array(area_km2, "area_km2", nonnegative=True)
    try:
        return depth * area * 1000.0
    except ValueError as error:
        raise ValueError(f"depth/area shapes cannot broadcast: {depth.shape}, {area.shape}") from error


def interval_bounds_hours(timestamps: np.ndarray, step_hours: float, *, stamp_unit: str = "hour") -> np.ndarray:
    dt = positive_finite(step_hours, "step_hours")
    stamps = finite_array(timestamps, "timestamps")
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError("timestamps must be a nonempty one-dimensional array")
    if stamp_unit not in {"hour", "day"}:
        raise ValueError(f"Unsupported timestamp unit: {stamp_unit}")
    starts = stamps * (24.0 if stamp_unit == "day" else 1.0)
    if starts.size > 1 and not np.allclose(np.diff(starts), dt, rtol=0, atol=1e-9):
        raise ValueError("timestamps do not match the declared consecutive interval duration")
    return np.column_stack((starts, starts + dt))


def integrate_power_mwh(power_mw: np.ndarray, step_hours: float | np.ndarray, *, axis: int = 0) -> np.ndarray:
    """P: signed interval-average MW times hours. Signed injection is allowed."""
    power = finite_array(power_mw, "power_mw")
    if power.ndim == 0:
        raise ValueError("power_mw must have a time axis")
    duration = finite_array(step_hours, "step_hours")
    if np.any(duration <= 0):
        raise ValueError("step_hours must be positive")
    if duration.ndim == 0:
        return np.sum(power, axis=axis, dtype=np.float64) * duration
    if duration.ndim != 1 or duration.size != power.shape[axis]:
        raise ValueError("interval durations must match the power time axis")
    shape = [1] * power.ndim
    shape[axis] = duration.size
    return np.sum(power * duration.reshape(shape), axis=axis, dtype=np.float64)


def aggregate_complete_days(values: np.ndarray, timestamps: np.ndarray, *, quantity_kind: str,
                            step_hours: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """P: integrate rates/means; sum interval accumulations; never average states.

    Returns (day indices, daily values). Incomplete/misaligned days raise rather
    than inventing missing hours. Input is [time,...], timestamps are hours.
    """
    array = finite_array(values, "values")
    bounds = interval_bounds_hours(timestamps, step_hours)
    if array.ndim == 0 or array.shape[0] != len(bounds):
        raise ValueError("values and timestamps have inconsistent time dimensions")
    if quantity_kind not in {"interval_mean", "rate", "accumulation"}:
        raise ValueError("daily aggregation requires interval_mean, rate or accumulation; states need boundary sampling")
    samples = 24.0 / step_hours
    if not np.isclose(samples, round(samples), rtol=0, atol=1e-9):
        raise ValueError("step_hours must divide 24 for aligned daily aggregation")
    count = int(round(samples))
    if len(bounds) % count or not np.isclose(bounds[0, 0] % 24, 0, atol=1e-9):
        raise ValueError("only complete midnight-aligned solar days can be aggregated")
    grouped = array.reshape((-1, count) + array.shape[1:])
    result = grouped.sum(axis=1, dtype=np.float64)
    if quantity_kind == "interval_mean":
        result /= count
    elif quantity_kind == "rate":
        result *= step_hours
    # Accepted sub-nanosecond timestamp noise must not truncate day 1 to day 0.
    return np.rint(bounds[::count, 0] / 24).astype(np.int64), result


def validate_weather_arrays(dynamic: np.ndarray, weather_class: np.ndarray, timestamps: np.ndarray,
                            channel_names: tuple[str, ...], time_unit: str) -> None:
    if time_unit not in {"hour", "day"}:
        raise ValueError("weather time_unit must be hour or day")
    values = finite_array(dynamic, "weather.dynamic")
    if values.ndim != 4 or values.shape[1] != len(channel_names):
        raise ValueError("weather.dynamic must be [time,channel,y,x] matching channel_names")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("weather channel names must be unique")
    if np.shape(weather_class) != (values.shape[0], *values.shape[2:]):
        raise ValueError("weather_class must be [time,y,x] matching dynamic")
    integer_labels(weather_class, "weather_class")
    if np.shape(timestamps) != (values.shape[0],):
        raise ValueError("weather timestamps do not match dynamic time dimension")
    interval_bounds_hours(timestamps, 1.0 if time_unit == "hour" else 24.0, stamp_unit=time_unit)


def validate_node_arrays(timestamps: np.ndarray, bus_ids: np.ndarray, arrays: Mapping[str, np.ndarray]) -> None:
    interval_bounds_hours(timestamps, 1.0)
    ids = entity_ids(bus_ids)
    expected = (len(timestamps), len(ids))
    for name, values in arrays.items():
        data = finite_array(values, name, nonnegative=True)
        if data.shape != expected:
            raise ValueError(f"{name}: expected [time,bus] {expected}, got {data.shape}")


@dataclass(frozen=True)
class FieldContract:
    unit: str
    quantity_kind: str
    spatial_support: str
    time_support: str
    source_module: str
    consumers: tuple[str, ...]
    relation_class: str
    semantic_layer: str


WEATHER_UNITS = {
    "temperature": "degC", "humidity": "1", "pressure": "hPa",
    "wind_u": "m/s", "wind_v": "m/s", "wind_speed": "m/s",
    "cloud": "1", "precipitation": "mm", "irradiance": "W/m2",
}


def field_contract_document(step_hours: float = 1.0) -> dict[str, object]:
    """S: versioned meanings of existing arrays; matrix columns are explicit.

    Prefix groups describe named arrays, not extra physical state. Additive keys
    preserve old NPZ readers. No units are inferred from numerical magnitudes.
    """
    positive_finite(step_hours, "step_hours")
    fields: dict[str, dict[str, object]] = {}

    def add(names: str, unit: str, kind: str, support: str, time: str,
            module: str, consumers: tuple[str, ...], relation: str, layer: str) -> None:
        contract = asdict(FieldContract(unit, kind, support, time, module, consumers, relation, layer))
        for name in names.split():
            fields[name] = dict(contract)

    add("elevation hydrology_elevation water_depth", "m", "state", "cell", "static", "terrain/hydrology", ("land", "climate", "routing"), "P", "static_design")
    add("slope roughness aspect_sin aspect_cos", "1", "diagnostic", "cell_with_declared_derivative_support", "static", "terrain", ("land", "climate", "routing"), "P", "static_design")
    add("curvature", "1/m", "diagnostic", "cell", "static", "terrain", ("hydrology",), "P", "static_design")
    add("flow_accumulation", "upstream_cell_count", "diagnostic", "upstream_catchment", "static", "hydrology", ("river_geometry",), "P", "static_D8_count_not_discharge")
    add("catchment_area_km2", "km2", "area", "upstream_catchment", "static", "hydrology", ("river_geometry", "validation", "dataset"), "P", "static_D8_area_not_discharge")
    add("distance_to_water", "km", "diagnostic", "cell", "static", "hydrology", ("city", "land"), "P", "static_design")
    add("population_density", "persons/km2", "density", "cell_area_mean", "static", "city", ("load", "land_use"), "S", "static_design")
    add("buildability terrain_cost vegetation flood_risk city_suitability urban_core_suitability waterfront_amenity urban_density economic_activity residential commercial industrial agriculture park_green load_density_base wind_suitability pv_suitability thermal_suitability thermal_externality load_node_density", "1", "score", "cell", "static", "land/city/energy", ("site_ranking", "display"), "S", "static_design")
    add("land_cover land_use_zone weather_class flow_direction watershed_id city_id_map", "1", "label", "cell", "static_or_interval_label", "respective_generator", ("display", "topology"), "S", "synthetic_truth")
    add("protected water_buffer river river_centerline lake urban_mask", "1", "mask", "cell", "static", "land/hydrology/city", ("site_constraints",), "S", "static_design")
    add("landform land_cover_type", "1", "label", "cell", "static_before_city_planning", "land", ("land_use", "dataset", "validation"), "S", "synthetic_potential_background_cover_and_geomorphology")
    add("protected_mask wind_land_eligible pv_land_eligible thermal_land_eligible", "1", "mask", "cell", "static_or_project_planning", "land/energy", ("hard_area_constraints", "dataset", "validation"), "S", "hard_land_identity_or_eligibility")
    add("allocatable_land_fraction " + " ".join(f"land_use_fraction_{name}" for name in ("water", "wetland", "residential", "commercial", "industrial", "agriculture", "park_green", "natural", "energy_reserve")), "1", "area_fraction", "whole_cell_area", "static_planning", "land/land_use", ("city", "energy", "dataset", "validation"), "S", "exclusive_land_budget")
    add("energy_available_area_km2 energy_wind_project_area_km2 energy_pv_project_area_km2 energy_unallocated_area_km2 thermal_allocated_area_km2 energy_unallocated_after_thermal_area_km2 energy_project_area_by_cell_km2 thermal_project_area_by_cell_km2", "km2", "allocated_area", "cell_or_project_by_cell", "static_planning", "energy/grid_nodes", ("capacity_bounds", "dataset", "validation"), "P", "exclusive_project_envelope_accounting")
    for name, spec in hydrology_field_schema().items():
        pure_accounting = (name.startswith("budget__") or name in {
            "state__lake_water_level_m", "state__lake_wetted_area_m2", "static__lake_capacity_m3",
            "flux__discharge_m3_s", "flux__actual_et_m3", "flux__cell_budget_residual_m3",
        })
        generation_relation = "P" if pure_accounting else "S"
        add(f"dynamic_hydrology.{name}", spec["unit"], spec["time_kind"], spec["spatial_support"],
            "explicit_hourly_intervals_and_T_plus_1_boundaries", "dynamic_hydrology",
            ("water_budget_validation", "dataset", "optional_future_consumers"),
            generation_relation,
            "optional_synthetic_water_account_not_flood_forecast")
        fields[f"dynamic_hydrology.{name}"].update(
            generation_relation_type=generation_relation, accounting_relation_type="P",
            generation_note=("Geometry, unit conversion, sum or balance under the prescribed model"
                             if pure_accounting else "Uncalibrated forcing or scenario closure: bounded soil bucket, linear reservoirs, prescribed geometry or shortwave PET; engineering simplification"),
        )
    for name, unit in WEATHER_UNITS.items():
        add(f"weather.{name}", unit, "accumulation" if name == "precipitation" else "interval_mean", "cell_area_representative", "explicit_interval_bounds_hours", "weather", ("source_load", "hydrology", "daily_aggregation"), "P" if name in {"wind_speed", "humidity", "pressure"} else "E", "exogenous_weather")
    add("weather.diagnostic__specific_humidity_kg_kg", "kg/kg moist air", "interval_representative_primitive", "cell", "explicit_interval_bounds_hours", "weather", ("moist_air_diagnosis", "source_load"), "S", "prescribed_moisture_forcing")
    add("weather.diagnostic__sea_level_pressure_hpa", "hPa", "interval_representative_primitive", "cell", "explicit_interval_bounds_hours", "weather", ("moist_air_diagnosis",), "S", "prescribed_synoptic_forcing")
    add("weather.diagnostic__air_density_kg_m3", "kg/m3", "diagnostic", "cell", "diagnose_hour_then_aggregate", "weather", ("source_load", "validation"), "P", "synthetic_weather_diagnostic")
    add("weather.diagnostic__specific_humidity_adjustment_kg_kg", "kg/kg moist air", "scenario_adjustment", "cell", "interval_representative_not_accumulated_water_depth", "weather", ("validation",), "S", "open_reservoir_saturation_adjustment_not_rain")
    add("weather.static_elevation_m", "m", "state", "cell", "static", "terrain", ("moist_air_diagnosis",), "P", "static_design")
    add("p_load_mw p_gen_available_mw", "MW", "interval_mean", "bus", "hour_interval", "artifact_dependent_see_overrides", ("planning", "dispatch"), "E", "artifact_dependent_see_overrides")
    add("p_gen_scheduled_mw", "MW", "interval_mean", "bus", "hour_interval", "planning_or_dispatch", ("power_flow",), "S", "planned_or_operated_power_by_artifact")
    add("q_load_mvar", "Mvar", "interval_mean", "bus", "hour_interval", "artifact_dependent_see_overrides", ("optional_AC",), "S", "artifact_dependent_see_overrides")
    add("bus_p_injection_mw served_load_mw dispatched_generation_mw unserved_load_mw curtailed_generation_mw", "MW", "interval_mean", "bus", "hour_interval", "power_flow", ("validation", "dataset"), "P", "operation_result")
    add("line_flow_mw", "MW", "interval_mean", "directed_branch_from_to", "hour_interval", "power_flow", ("validation", "dispatch"), "P", "operation_result")
    add("bus_angle_rad", "rad", "diagnostic", "bus_relative_to_island_reference", "hour_interval_DC", "power_flow", ("validation",), "P", "operation_result")
    add("line_loading_ratio", "1", "diagnostic", "branch", "hour_interval", "power_flow", ("validation",), "P", "operation_result")
    add("charge_mw discharge_mw emergency_discharge_mw", "MW", "interval_mean", "storage_site", "hour_interval", "storage_dispatch", ("SOC", "validation"), "P", "operation_result")
    add("soc_mwh", "MWh", "state", "storage_site", "T+1_interval_boundaries", "storage_dispatch", ("validation", "next_interval"), "P", "operation_result")
    add("site_power_capacity_mw storage_power_expansion_mw thermal_capacity_expansion_mw", "MW", "capacity", "asset", "fixed_for_declared_run", "asset_planning", ("dispatch", "validation"), "S", "planning_result")
    add("site_energy_capacity_mwh storage_energy_expansion_mwh cycle_boundary_soc_mwh target_soc_mwh", "MWh", "state_or_capacity_by_name", "storage_site", "declared_boundary_or_asset", "storage_planning", ("dispatch", "validation"), "S", "planning_result")
    artifact_overrides = {}
    for artifact, layer, module in (
        ("stage_11_operation/source_load_forecast.npz", "exogenous_source_load", "source_load"),
        ("stage_14_storage_dispatch/storage_dispatch_forecast.npz", "operation_result_with_storage_requests", "storage_dispatch"),
    ):
        artifact_overrides[artifact] = {
            name: dict(fields[name], semantic_layer=layer, source_module=module)
            for name in ("p_load_mw", "p_gen_available_mw", "q_load_mvar")
        }
    artifact_overrides["stage_14_storage_dispatch/storage_dispatch_forecast.npz"]["q_load_mvar"].update(
        semantic_layer="unsupported_DC_placeholder", quantity_kind="placeholder", note="Zero placeholder; reactive operation was not solved."
    )
    return {
        "contract_version": CONTRACT_VERSION, "generator_version": GENERATOR_VERSION,
        "step_hours": step_hours, "grid_area_rule": "square projected cell: cell_size_km**2 km2",
        "time_convention": "local_solar_time_365_day_climatology; timestamps are interval starts",
        "rain_aggregation": "sum depth_mm; 1000 * depth_mm * area_km2 = volume_m3",
        "power_aggregation": "interval_mean_MW * duration_h = energy_MWh",
        "weather_channel_order": "read channel_names; never assume a new channel index",
        "weather_daily_roles": {
            "daily_weather.npz": "conditional_driver_anchors; RH/p/rho diagnose_anchor_state",
            "hourly_weather_week.npz": "hourly_realization; generation_mode and daily_constraints in weather_metadata_json",
            "hourly_daily_summary.npz": "mean_of_hourly_diagnostics; precipitation_sum; never_rediagnose_from_daily_mean_primitives",
        },
        "fields": fields,
        "dynamic_hydrology_field_schema": hydrology_field_schema(),
        "artifact_field_overrides": artifact_overrides,
        "matrix_columns": {
            "electrical_buses": ["id:1", "nominal:kV", "P_capacity:MW", "Q_capacity:Mvar", "base_load:MW", "power_factor:1", "voltage_setpoint:pu"],
            "electrical_branches": ["id:1", "from_id:1", "to_id:1", "nominal:kV", "length:km", "r:ohm", "x:ohm", "b:microS", "rate:MVA", "redundant:bool"],
            "candidates": ["id:1", "row:index", "col:index", "x:normalized_0_1", "y:normalized_0_1", "capacity:MW", "suitability:1"],
            "energy_project_land_ledger": ["project_id:1", "technology_code:1", "candidate_id:1", "row:index", "col:index", "reserved_area:km2", "capacity:MW", "capacity_density:MW/km2", "capacity_equivalent_area:km2"],
            "thermal_land_ledger": ["bus_id:1", "row:index", "col:index", "reserved_area:km2", "capacity:MW", "capacity_density:MW/km2", "capacity_equivalent_area:km2", "land_capacity_upper_bound:MW", "requested_capacity:MW"],
        },
        "coordinate_convention": "legacy site x/y are normalized display coordinates; physical distances use row/col and configured cell_size_km, never x/y as km",
        "limitations": ["1 h generator; conversion helpers support explicit fractional durations", "DC reactive/voltage safety unsupported", "scores are not allocated area fractions", "legacy total-generation curtailment is not exclusively renewable curtailment"],
    }
