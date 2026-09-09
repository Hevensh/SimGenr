"""Validate exported physical identities without training, plotting or pytest.

Usage: python -B scripts/validate_world_physics.py outputs/small_debug_seed42
Writes physics_validation.json and physics_validation.md inside each world.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.core.config import load_world_config
from world_generator.core.output_layout import WorldDataLayout
from world_generator.core.contracts import integer_labels, entity_ids
from world_generator.weather.physics import extraterrestrial_hourly_irradiance, latitude_grid
from world_generator.weather.physics import diagnose_moist_air, saturation_vapor_pressure_hpa
from scripts.hydrology_validation import check_dynamic_hydrology
from scripts.source_load_validation import check_source_load_contracts
from scripts.operation_validation import check_operation_contracts, check_frozen_asset_contracts
from scripts.validation_checks import Checks, input_error_report, validation_context


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _positive_max(values: np.ndarray) -> float:
    return float(np.max(values, initial=0.0))


@validation_context(stage="B_weather",fields=["daily_weather.dynamic","hourly_weather.dynamic","hourly_daily_summary.dynamic"],engineering_simplification="Synthetic local-solar weather; hydrostatic and no-twilight identities, not empirical calibration")
def check_weather_contracts(checks: Checks, daily: dict[str, np.ndarray], hourly: dict[str, np.ndarray],
                            config: object, daily_summary: dict[str, np.ndarray] | None = None) -> None:
    """B: independently aggregate outputs and enforce only declared anchor constraints.

    A daily average of nonlinear diagnostics is not diagnosed from daily means.
    The primitive-state identities are checked on the hourly representative state.
    """
    meta = json.loads(str(hourly.get("weather_metadata_json", "{}")))
    mode = meta.get("generation_mode")
    if mode and daily_summary is None:
        raise ValueError("Hourly weather with declared primitives requires hourly_daily_summary.npz")
    reference = daily if daily_summary is None else daily_summary
    reference_indices = {int(value): index for index, value in enumerate(reference["timestamps"])}
    anchor_indices = {int(value): index for index, value in enumerate(daily["timestamps"])}
    channels = {str(name): i for i, name in enumerate(hourly["channel_names"])}
    reference_channels = {str(name): i for i, name in enumerate(reference["channel_names"])}
    anchor_channels = {str(name): i for i, name in enumerate(daily["channel_names"])}
    constraints = meta.get("daily_constraints", list(channels) if not mode else [])
    stamps = hourly["timestamps"].astype(np.int64)
    latitude = latitude_grid(config.world, hourly["dynamic"].shape[2:])
    errors: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    anchor_errors: dict[str, list[np.ndarray]] = {str(name): [] for name in constraints}
    night_errors, above_toa = [], []
    for day in np.unique(stamps // 24):
        positions = np.flatnonzero(stamps // 24 == day)
        complete = len(positions) == 24 and np.array_equal(stamps[positions] % 24, np.arange(24)) and int(day) in reference_indices
        checks.condition(f"complete_daily_alignment_{day}", complete)
        if not complete:
            continue
        block = hourly["dynamic"][positions].astype(np.float64)
        target = reference["dynamic"][reference_indices[int(day)]].astype(np.float64)
        anchor_day = int(day) if int(day) in anchor_indices else int(day) % 365
        for name, index in channels.items():
            if name not in reference_channels:
                checks.condition(f"daily_channel_{name}_exists", False)
                continue
            aggregate = block[:, index].sum(axis=0) if name == "precipitation" else block[:, index].mean(axis=0)
            errors[name].append(aggregate - target[reference_channels[name]])
            if mode and name in anchor_errors:
                if anchor_day not in anchor_indices or name not in anchor_channels:
                    checks.condition(f"daily_anchor_{name}_{day}_exists", False)
                else:
                    anchor_errors[name].append(aggregate - daily["dynamic"][anchor_indices[anchor_day], anchor_channels[name]])
        toa = extraterrestrial_hourly_irradiance(latitude, float(day))
        radiation = block[:, channels["irradiance"]]
        night_errors.append(radiation[toa <= 1e-8])
        above_toa.append(radiation - toa)
    for name, residuals in errors.items():
        checks.equal(f"daily_hourly_{name}", np.asarray(residuals), 2e-4,
                     "mm" if name == "precipitation" else "channel native unit",
                     "rain depth sums; interval means average; reference is realized daily summary when present")
    if mode:
        for name, residuals in anchor_errors.items():
            checks.equal(f"declared_anchor_{name}", np.asarray(residuals), 2e-4,
                         "mm" if name == "precipitation" else "channel native unit")
    checks.equal("nighttime_ghi_zero", np.concatenate(night_errors) if night_errors else np.zeros(0), 1e-6, "W/m2", "no-twilight solar model")
    checks.upper("irradiance_below_toa", np.asarray(above_toa), 0.0, 2e-4, "W/m2", "chosen model excludes lateral cloud-edge enhancement")
    for name in ("precipitation", "irradiance", "wind_speed"):
        checks.upper(f"hourly_{name}_nonnegative", -hourly["dynamic"][:, channels[name]], 0.0, 1e-6)
    for name in ("humidity", "cloud"):
        values = hourly["dynamic"][:, channels[name]]
        checks.upper(f"hourly_{name}_lower_bound", -values, 0.0, 1e-6)
        checks.upper(f"hourly_{name}_upper_bound", values, 1.0, 1e-6)
    checks.equal("wind_vector_magnitude", np.hypot(hourly["dynamic"][:, channels["wind_u"]], hourly["dynamic"][:, channels["wind_v"]]) - hourly["dynamic"][:, channels["wind_speed"]], 1e-5, "m/s")
    checks.upper("daily_wind_mean_triangle", np.hypot(reference["dynamic"][:, reference_channels["wind_u"]], reference["dynamic"][:, reference_channels["wind_v"]]), reference["dynamic"][:, reference_channels["wind_speed"]], 1e-5, "m/s")
    if mode:
        required = ("specific_humidity_kg_kg", "sea_level_pressure_hpa", "air_density_kg_m3")
        expected_shape = (len(stamps), *hourly["dynamic"].shape[2:])
        for name in required:
            key = f"diagnostic__{name}"
            if key not in hourly or hourly[key].shape != expected_shape:
                raise ValueError(f"{key} must be present with shape {expected_shape}")
        q = hourly["diagnostic__specific_humidity_kg_kg"].astype(np.float64)
        p0 = hourly["diagnostic__sea_level_pressure_hpa"].astype(np.float64)
        p = hourly["dynamic"][:, channels["pressure"]].astype(np.float64)
        t_c = hourly["dynamic"][:, channels["temperature"]].astype(np.float64)
        height = hourly.get("static_elevation_m")
        if height is None or height.shape != hourly["dynamic"].shape[2:]:
            raise ValueError("static_elevation_m is required with shape [H,W] for primitive pressure diagnostics")
        checks.upper("specific_humidity_nonnegative", -q, 0, 0, "kg/kg")
        checks.condition("specific_humidity_below_one", bool(np.all(q < 1)))
        checks.condition("positive_absolute_temperature_and_pressure", bool(np.all(t_c > -273.15) and np.all(p > 0) and np.all(p0 > 0)))
        # Selected hydrostatic-model consistency; independent gas-law residuals
        # below also test the exported pressure/RH/density without that kernel.
        _, expected_pressure, _ = diagnose_moist_air(t_c, q, height, p0)
        checks.equal("hydrostatic_pressure_from_primitives", p - expected_pressure, 2e-4, "hPa")
        epsilon = 287.05 / 461.5
        vapor_hpa = p * q / (epsilon + (1 - epsilon) * q)
        rh = vapor_hpa / saturation_vapor_pressure_hpa(t_c)
        checks.equal("relative_humidity_from_primitives", rh - hourly["dynamic"][:, channels["humidity"]], 2e-6, "1")
        rho = p * 100 / (287.05 * (t_c + 273.15) * (1 + (461.5 / 287.05 - 1) * q))
        checks.equal("moist_air_density_identity", rho - hourly["diagnostic__air_density_kg_m3"], 2e-6, "kg/m3")
        checks.condition("positive_moist_air_density", bool(np.all(hourly["diagnostic__air_density_kg_m3"] > 0)))


@validation_context(stage="C_land_city_energy",fields=["land_use_fraction_*","population_density","energy_project_land_ledger"],time_support="static_or_spatial_aggregate",engineering_simplification="Exclusive project envelopes and configured land allocation; area and population accounting")
def check_land_accounting(checks: Checks, static: dict[str, np.ndarray],
                          candidates: dict[str, np.ndarray], metadata: dict[str, object],
                          config: object) -> dict[str, float]:
    """C: independently close whole-cell use and exclusive project-area ledgers.

    Land-use fractions are dimensionless areas, not suitability probabilities.
    Project envelopes subdivide energy_reserve; they are not a tenth land use.
    """
    shape = (config.world.height, config.world.width)
    area = float(config.world.cell_size_km) ** 2

    def raster(name: str) -> np.ndarray:
        if name not in static or np.shape(static[name]) != shape:
            raise ValueError(f"Land accounting field {name} must have shape {shape}")
        values = np.asarray(static[name], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Land accounting field {name} must be finite")
        return values

    water = raster("river").astype(bool) | raster("lake").astype(bool)
    protected = raster("protected_mask")
    checks.equal("protected_mask_compatibility", protected - raster("protected"), 0)
    checks.condition("protected_mask_boolean", bool(np.isin(protected, [0, 1]).all()))
    cover = integer_labels(raster("land_cover_type"), "land_cover_type")
    form = integer_labels(raster("landform"), "landform")
    checks.condition("land_cover_type_supported", bool(np.isin(cover, [1, 2, 3, 4, 5]).all()))
    checks.condition("landform_supported", bool(np.isin(form, [1, 2, 3]).all()))
    checks.equal("land_cover_water_mask", (cover == 1).astype(float) - water, 0)
    names = ("water", "wetland", "residential", "commercial", "industrial", "agriculture",
             "park_green", "natural", "energy_reserve")
    fractions = {name: raster(f"land_use_fraction_{name}") for name in names}
    stacked = np.stack(list(fractions.values()))
    checks.upper("land_use_fraction_nonnegative", -stacked, 0, 1e-7, "1")
    checks.upper("land_use_fraction_upper_bound", stacked, 1, 1e-7, "1")
    checks.equal("land_use_fraction_cell_closure", stacked.sum(axis=0) - 1, 2e-7, "1")
    checks.equal("land_use_water_area", fractions["water"] - water, 1e-7, "1")
    checks.equal("land_use_wetland_area", fractions["wetland"] - (cover == 2), 1e-7, "1")
    allocatable = raster("allocatable_land_fraction")
    checks.upper("allocatable_fraction_nonnegative", -allocatable, 0, 0, "1")
    checks.upper("allocatable_fraction_upper_bound", allocatable, 1, 0, "1")
    blocked = water | protected.astype(bool) | (cover == 2)
    checks.equal("allocatable_land_hard_exclusions", np.where(blocked, allocatable, 0), 0, "1")
    developed = sum(fractions[name] for name in ("residential", "commercial", "industrial", "agriculture", "energy_reserve"))
    checks.upper("allocated_uses_within_hard_land_budget", developed, allocatable, 2e-7, "1")
    checks.equal("population_land_support", np.where(allocatable <= 0, raster("population_density"), 0), 1e-7, "persons/km2")

    available = raster("energy_available_area_km2")
    wind_area = raster("energy_wind_project_area_km2")
    pv_area = raster("energy_pv_project_area_km2")
    unused = raster("energy_unallocated_area_km2")
    for name, values in (("available", available), ("wind", wind_area), ("pv", pv_area), ("unused", unused)):
        checks.upper(f"energy_{name}_area_nonnegative", -values, 0, 1e-9, "km2")
    checks.equal("energy_available_from_reserved_use", available - fractions["energy_reserve"] * area, 2e-7 * area, "km2")
    checks.equal("energy_cell_area_budget", wind_area + pv_area + unused - available, 2e-7 * area, "km2")
    checks.equal("energy_project_hard_exclusions", np.where(blocked, wind_area + pv_area, 0), 1e-9, "km2")

    columns = ("project_id", "technology_code", "candidate_id", "row", "col", "reserved_area_km2",
               "capacity_mw", "capacity_density_mw_km2", "capacity_equivalent_area_km2")
    if tuple(str(x) for x in candidates.get("energy_project_land_columns", [])) != columns:
        raise ValueError("energy_project_land_columns does not match the declared C ledger schema")
    ledger = np.asarray(candidates["energy_project_land_ledger"], dtype=np.float64)
    if ledger.ndim != 2 or ledger.shape[1] != len(columns) or not np.isfinite(ledger).all():
        raise ValueError("energy_project_land_ledger must be finite [N,9]")
    cube = np.asarray(candidates["energy_project_area_by_cell_km2"], dtype=np.float64)
    if cube.shape != (len(ledger), *shape) or not np.isfinite(cube).all():
        raise ValueError("energy_project_area_by_cell_km2 must be finite [N,H,W]")
    entity_ids(ledger[:, 0], "energy project ids")
    integer_labels(ledger[:, 1:5], "energy project type/candidate/grid ids")
    checks.condition("energy_project_technology_codes", bool(np.isin(ledger[:, 1], [1, 2]).all()))
    checks.condition("energy_project_coordinates", bool(np.all((ledger[:, 3] >= 0) & (ledger[:, 3] < shape[0]) & (ledger[:, 4] >= 0) & (ledger[:, 4] < shape[1]))))
    checks.condition("energy_project_unique_candidate_keys", len(set(map(tuple, ledger[:, 1:3]))) == len(ledger))
    checks.upper("energy_project_cell_area_nonnegative", -cube, 0, 1e-9, "km2")
    checks.equal("energy_project_area_integration", cube.sum(axis=(1, 2)) - ledger[:, 5], 2e-6 * max(1, area), "km2")
    checks.upper("energy_projects_no_double_allocation", cube.sum(axis=0), available, 2e-7 * area, "km2")
    checks.condition("energy_project_density_positive", bool(np.all(ledger[:, 7] > 0)))
    checks.upper("energy_project_capacity_nonnegative", -ledger[:, 6], 0, 0, "MW")
    checks.upper("energy_project_capacity_area_bound", ledger[:, 6], ledger[:, 5] * ledger[:, 7], 2e-5, "MW")
    checks.equal("energy_capacity_equivalent_area_identity", ledger[:, 8] * ledger[:, 7] - ledger[:, 6], 2e-5, "MW")
    for code, kind, total_area in ((1, "wind", wind_area), (2, "pv", pv_area)):
        selected = ledger[:, 1] == code
        eligible = raster(f"{kind}_land_eligible")
        checks.condition(f"{kind}_land_eligible_boolean", bool(np.isin(eligible, [0, 1]).all()))
        checks.equal(f"{kind}_project_area_raster", cube[selected].sum(axis=0) - total_area, 2e-7 * area, "km2")
        checks.equal(f"{kind}_project_full_envelope_eligible", np.where(eligible > 0, 0, total_area), 1e-9, "km2")
        density = getattr(config.energy, f"{kind}_capacity_density_mw_km2")
        checks.equal(f"{kind}_project_config_density", ledger[selected, 7] - density, 1e-7, "MW/km2")
        rows = np.asarray(candidates[f"{kind}_candidates"])
        if rows.size == 0:
            rows = np.empty((0, 7))
        if rows.ndim != 2 or rows.shape[1] != 7:
            raise ValueError(f"{kind}_candidates must preserve the legacy [N,7] layout")
        ids = entity_ids(rows[:, 0], f"{kind} candidate ids")
        by_id = {int(row[0]): row for row in rows}
        checks.condition(f"{kind}_candidate_ledger_membership", set(ids) == set(ledger[selected, 2]))
        for row in ledger[selected]:
            key = int(row[2])
            if key in by_id:
                checks.equal(f"{kind}_candidate_{key}_ledger_identity", row[[3, 4, 6]] - by_id[key][[1, 2, 5]], 2e-5, "row,col,MW")

    budget = metadata.get("city_population_budget")
    if not isinstance(budget, dict):
        raise ValueError("city_population_budget is required for land_use_v1")
    city_total = sum(float(city["population"]) for city in metadata["cities"])
    effective_area = float(allocatable.sum() * area)
    if config.city.scaling_mode == "fixed":
        target_population = float(config.city.total_population)
    elif effective_area <= 0:
        target_population = 0.0
    else:
        # S: project scale-aware population prior, independently from the maps.
        target_population = float(config.city.total_population) * max(effective_area / config.city.reference_effective_area_km2, 1e-6) ** config.city.population_area_exponent
    tolerance = max(0.1, target_population * 2e-6)
    checks.equal("population_target_not_silently_dropped", np.asarray(city_total - target_population), tolerance, "persons")
    checks.equal("population_budget_declared_target", np.asarray(float(budget["target_population_persons"]) - target_population), tolerance, "persons")
    checks.equal("population_budget_declared_allocation", np.asarray(float(budget["allocated_population_persons"]) - city_total), tolerance, "persons")
    checks.equal("population_budget_configured_prior", np.asarray(float(budget["configured_population_persons"]) - config.city.total_population), tolerance, "persons")
    checks.equal("population_budget_land_area", np.asarray(float(budget["allocatable_land_area_km2"]) - effective_area), 2e-7 * max(1, effective_area), "km2")
    checks.condition("city_budget_placed_count", int(budget["placed_city_count"]) == len(metadata["cities"]))
    return {"domain_area_km2": float(np.prod(shape) * area), "allocatable_area_km2": effective_area,
            "energy_reserved_area_km2": float(available.sum()), "wind_project_area_km2": float(wind_area.sum()),
            "pv_project_area_km2": float(pv_area.sum()), "unallocated_energy_area_km2": float(unused.sum()),
            "target_population_persons": target_population}


@validation_context(stage="C_stage09_thermal_land",fields=["thermal_land_ledger","thermal_project_area_by_cell_km2"],time_support="static_or_spatial_aggregate",engineering_simplification="Configured thermal land density and exclusive envelopes")
def check_thermal_land(checks: Checks, static: dict[str, np.ndarray], nodes: dict[str, np.ndarray],
                       metadata: dict[str, object], config: object) -> dict[str, float]:
    """C: Stage 09 thermal footprints consume only the Stage 08 remaining area."""
    shape = (config.world.height, config.world.width)
    columns = ("bus_id", "row", "col", "reserved_area_km2", "capacity_mw", "capacity_density_mw_km2",
               "capacity_equivalent_area_km2", "land_capacity_upper_bound_mw", "requested_capacity_mw")
    if tuple(str(x) for x in nodes.get("thermal_land_columns", [])) != columns:
        raise ValueError("thermal_land_columns does not match the C Stage 09 schema")
    ledger = np.asarray(nodes["thermal_land_ledger"], dtype=np.float64)
    if ledger.ndim != 2 or ledger.shape[1] != 9 or not np.isfinite(ledger).all():
        raise ValueError("thermal_land_ledger must be finite [N,9]")
    cube = np.asarray(nodes["thermal_project_area_by_cell_km2"], dtype=np.float64)
    if cube.shape != (len(ledger), *shape) or not np.isfinite(cube).all():
        raise ValueError("thermal_project_area_by_cell_km2 must be finite [N,H,W]")
    maps = {}
    for name in ("energy_unallocated_area_km2", "thermal_allocated_area_km2", "energy_unallocated_after_thermal_area_km2", "thermal_land_eligible"):
        values = np.asarray(static[name], dtype=np.float64)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite [H,W]")
        maps[name] = values
    available, occupied, remaining, eligible = maps.values()
    area_tol = 2e-7 * config.world.cell_size_km ** 2
    ids = entity_ids(ledger[:, 0], "thermal land bus ids")
    integer_labels(ledger[:, 1:3], "thermal land grid coordinates")
    checks.condition("thermal_land_coordinates", bool(np.all((ledger[:, 1] >= 0) & (ledger[:, 1] < shape[0]) & (ledger[:, 2] >= 0) & (ledger[:, 2] < shape[1]))))
    checks.condition("thermal_land_eligible_boolean", bool(np.isin(eligible, [0, 1]).all()))
    checks.upper("thermal_project_cell_area_nonnegative", -cube, 0, 1e-9, "km2")
    checks.upper("thermal_unallocated_area_nonnegative", -remaining, 0, 1e-9, "km2")
    checks.equal("thermal_project_area_integration", cube.sum(axis=(1, 2)) - ledger[:, 3], 2e-6, "km2")
    checks.equal("thermal_project_area_raster", cube.sum(axis=0) - occupied, area_tol, "km2")
    checks.equal("thermal_stage09_remaining_area_budget", occupied + remaining - available, area_tol, "km2")
    checks.upper("thermal_cannot_reuse_wind_pv_land", cube.sum(axis=0), available, area_tol, "km2")
    checks.equal("thermal_project_full_envelope_eligible", np.where(eligible > 0, 0, occupied), 1e-9, "km2")
    checks.equal("thermal_project_hard_land_exclusions", np.where(static["allocatable_land_fraction"] > 0, 0, occupied), 1e-9, "km2")
    checks.condition("thermal_project_density_positive", bool(np.all(ledger[:, 5] > 0)))
    checks.equal("thermal_project_config_density", ledger[:, 5] - config.power_grid.thermal_capacity_density_mw_km2, 1e-7, "MW/km2")
    checks.equal("thermal_land_supported_capacity", ledger[:, 7] - ledger[:, 3] * ledger[:, 5], 2e-5, "MW")
    checks.equal("thermal_capacity_equivalent_area", ledger[:, 6] * ledger[:, 5] - ledger[:, 4], 2e-5, "MW")
    checks.upper("thermal_stage09_capacity_within_land", ledger[:, 4], ledger[:, 7], 2e-5, "MW")
    checks.upper("thermal_stage09_capacity_within_request", ledger[:, 4], ledger[:, 8], 2e-5, "MW")
    checks.upper("thermal_stage09_capacity_nonnegative", -ledger[:, 4], 0, 0, "MW")
    buses = {int(bus["bus_id"]): bus for bus in metadata["grid_buses"] if bus["kind"] == "thermal_bus"}
    checks.condition("thermal_land_bus_membership", set(ids) == set(buses))
    for row in ledger:
        key = int(row[0])
        if key in buses:
            bus = buses[key]
            checks.equal(f"thermal_bus_{key}_land_identity", row[[1, 2, 4]] - [bus["row"], bus["col"], bus["capacity_mw"]], 2e-5, "row,col,MW")
    return {"thermal_project_area_km2": float(occupied.sum()),
            "unallocated_after_thermal_area_km2": float(remaining.sum()),
            "thermal_stage09_capacity_mw": float(ledger[:, 4].sum()),
            "thermal_land_supported_capacity_mw": float(ledger[:, 7].sum())}


def validate_world(world_dir: Path) -> dict[str, object]:
    world_dir = world_dir.resolve()
    failure_marker = world_dir / "generation_failure.json"
    if failure_marker.exists():
        try:
            failure = json.loads(failure_marker.read_text(encoding="utf-8"))
        except (OSError,ValueError) as exc:
            return input_error_report(world_dir.name,ValueError(f"Unreadable generation_failure.json; old artifacts cannot be validated: {exc}"))
        if not isinstance(failure,dict):
            return input_error_report(world_dir.name,ValueError("Invalid generation_failure.json; old artifacts cannot be validated"))
        report = input_error_report(world_dir.name,ValueError("Latest generation failed; stale artifacts were not validated: " + str(failure.get("message","failure marker present"))))
        report["generation_failure"] = failure
        report["error_category"] = failure.get("category","INPUT_OR_GENERATION_ERROR")
        report["checks"][0].update(name="latest_generation_failed",stage=failure.get("stage","generation"),fields=["generation_failure.json"],error_category=report["error_category"])
        report["limitations"] = "Generation failure marker is authoritative until a successful rerun clears it. Existing data were not validated."
        return report
    layout = WorldDataLayout(world_dir / "data")
    config = load_world_config(layout.config_snapshot)
    metadata = json.loads(layout.metadata.read_text(encoding="utf-8"))
    paths = {
        "static": layout.existing(layout.topology, "static_maps.npz", layout.root / "static_maps.npz"),
        "daily": layout.existing(layout.weather, "daily_weather.npz", layout.root / "daily_weather.npz"),
        "hourly": layout.existing(layout.weather, "hourly_weather_week.npz", layout.root / "hourly_weather_week.npz"),
        "source": layout.existing(layout.operation, "source_load_forecast.npz", layout.root / "source_load_forecast.npz"),
        "storage": layout.existing(layout.storage_dispatch, "storage_dispatch.npz", layout.root / "storage_dispatch.npz"),
        "dispatch": layout.existing(layout.storage_dispatch, "storage_dispatch_forecast.npz", layout.root / "storage_dispatch_forecast.npz"),
        "flow": layout.existing(layout.storage_dispatch, "storage_dispatch_power_flow.npz", layout.root / "storage_dispatch_power_flow.npz"),
        "electrical": layout.existing(layout.storage_dispatch, "storage_dispatch_electrical.npz", layout.root / "storage_dispatch_electrical.npz"),
    }
    data = {name: _npz(path) for name, path in paths.items()}
    checks = Checks(max_failure_locations=config.validation.max_failure_locations)
    checks.set_context(stage="input_alignment",fields=["exported_artifacts"],relation_class="S",engineering_simplification="Input schema/finite-value checks; no physical feasibility claim")
    for artifact, arrays in data.items():
        invalid = [name for name, values in arrays.items() if values.dtype.kind in "fc" and not np.isfinite(values).all()]
        checks.condition(f"finite_{artifact}", not invalid, ", ".join(invalid),fields=[f"{artifact}.{name}" for name in (invalid or arrays.keys())])
    static, daily, hourly = data["static"], data["daily"], data["hourly"]
    source, storage, dispatch, flow, electrical = (data[key] for key in ("source", "storage", "dispatch", "flow", "electrical"))
    stamps = hourly["timestamps"].astype(np.int64)
    hours = len(stamps)
    checks.condition("nonempty_consecutive_hourly_timestamps", hours > 0 and bool(np.all(np.diff(stamps) == 1)))
    for name in ("source", "storage", "dispatch", "flow"):
        checks.condition(f"timestamps_match_{name}", np.array_equal(data[name]["timestamps"], stamps))

    checks.set_context(stage="C_static_population",fields=["population_density","cities.population"],relation_class="P",time_support="static_or_domain_aggregate",engineering_simplification="Area integral of whole-cell population density")
    population = float(np.sum(static["population_density"], dtype=np.float64) * config.world.cell_size_km**2)
    city_population = float(sum(city["population"] for city in metadata["cities"]))
    checks.equal("population_mass", np.asarray(population - city_population), max(0.1, city_population * 2e-6), "persons")
    checks.upper("population_nonnegative", -static["population_density"], 0.0, 1e-7, "persons/km2",axes=("row","col"))
    if "catchment_area_km2" in static:
        checks.equal("static_catchment_physical_area", static["catchment_area_km2"] - static["flow_accumulation"] * config.world.cell_size_km**2,
                     1e-6, "km2", "legacy upstream cell count converted to physical area; never interpreted as current discharge")

    land_summary = {}
    if metadata.get("land_accounting_version") == "land_use_v1":
        land_summary = check_land_accounting(checks, static, _npz(layout.energy / "source_load_candidates.npz"), metadata, config)
        land_summary.update(check_thermal_land(checks, static, _npz(layout.buses / "grid_nodes.npz"), metadata, config))
    else:
        checks.status("independent_land_area_accounting","NOT_RUN","Legacy world has no complete C land accounting declaration",stage="C_land_city_energy",fields=["land_accounting_version"])

    daily_summary_path = layout.weather / "hourly_daily_summary.npz"
    daily_summary = _npz(daily_summary_path) if daily_summary_path.exists() else None
    check_weather_contracts(checks, daily, hourly, config, daily_summary)

    hydrology_summary = {"mode": "legacy_static_unspecified"}
    if "dynamic_hydrology" in metadata:
        from world_generator.operation.stage_cache import load_dynamic_hydrology_checkpoint
        hydro = load_dynamic_hydrology_checkpoint(world_dir, expected_timestamps=hourly["timestamps"],
                                                   expected_grid_shape=(config.world.height, config.world.width))
        checks.condition("hydrology_mode_matches_configuration", (hydro is not None) == config.hydrology_dynamic.enabled,stage="D_dynamic_hydrology",fields=["dynamic_hydrology","hydrology_dynamic.enabled"],relation_class="S")
        hydrology_summary = {"mode": "static_only", "dynamic_water_balance": "NOT_RUN"}
        if hydro is not None:
            hydrology_summary = check_dynamic_hydrology(checks, hydro.as_arrays(), hourly, static, config)
        else:
            checks.status("dynamic_water_balance","NOT_RUN","Dynamic hydrology is disabled; static drainage geometry is not simulated flow",stage="D_dynamic_hydrology",fields=["hydrology_dynamic.enabled"])
    else:
        checks.status("dynamic_water_balance","NOT_RUN","Legacy output has no dynamic hydrology process",stage="D_dynamic_hydrology",fields=["dynamic_hydrology"])

    source_ids = source["bus_ids"].astype(int)
    source_kinds = np.asarray(source["bus_kinds"])
    original_buses = {int(bus["bus_id"]): bus for bus in metadata["refined_grid_buses"]}
    source_load_summary = {"mode": "legacy_without_source_load_v1"}
    if "source_load_schema_version" in source:
        from world_generator.core.datatypes import SourceLoadForecastStore
        SourceLoadForecastStore.from_arrays(source)
        source_load_summary = check_source_load_contracts(checks, source, hourly, config, metadata)
    checks.set_context(stage="E_source_load",fields=["source.p_load_mw","source.p_gen_available_mw"],relation_class="P",time_support="hour_interval_mean",engineering_simplification="Synthetic exogenous request and equipment availability; not delivered power")
    checks.upper("exogenous_load_nonnegative", -source["p_load_mw"], 0.0, 1e-6, "MW",axes=("time","bus"),timestamps=stamps)
    checks.upper("exogenous_availability_nonnegative", -source["p_gen_available_mw"], 0.0, 1e-6, "MW")
    capacity_factors: dict[str, float | None] = {}
    for kind in ("wind_bus", "pv_bus"):
        positions = np.flatnonzero(source_kinds == kind)
        capacities = np.asarray([original_buses[int(source_ids[index])]["capacity_mw"] for index in positions])
        availability = source["p_gen_available_mw"][:, positions]
        checks.upper(f"{kind}_ac_capacity", availability, capacities[None, :], 1e-4, "MW")
        capacity_factors[kind] = float(availability.sum() / (hours * capacities.sum())) if capacities.sum() > 0 else None

    checks.set_context(stage="F_lossless_DC",fields=["flow.line_flow_mw","flow.bus_p_injection_mw","dispatch.p_load_mw"],time_support="hour_interval_mean",engineering_simplification="Lossless DC network; no AC voltage, reactive power or frequency dynamics")
    bus_ids, branch_ids = flow["bus_ids"].astype(int), flow["branch_ids"].astype(int)
    bus_index = {int(bus_id): index for index, bus_id in enumerate(bus_ids)}
    checks.condition("unique_final_bus_ids", len(bus_index) == len(bus_ids))
    checks.condition("dispatch_flow_bus_order", np.array_equal(dispatch["bus_ids"], flow["bus_ids"]))
    # core.datatypes._branch_electrical_to_array exports ten columns:
    # id, from, to, nominal_kv, length_km, r, x, b, rate_mva, is_redundant.
    # Validate the stored layout rather than flattening/reshaping a different
    # schema, which can silently scramble rows when the element count divides.
    branch_rows = np.asarray(electrical["electrical_branches"])
    if branch_rows.size == 0:
        branch_rows = np.empty((0, 10))
    if branch_rows.ndim != 2 or branch_rows.shape[1] != 10:
        raise ValueError(f"Expected electrical_branches [E,10], got {branch_rows.shape}")
    branches = {int(row[0]): row for row in branch_rows}
    # _bus_electrical_to_array: id, kV, Pcap, Qcap, base_load, PF, Vset.
    bus_rows = np.asarray(electrical["electrical_buses"])
    if bus_rows.size == 0:
        bus_rows = np.empty((0, 7))
    if bus_rows.ndim != 2 or bus_rows.shape[1] != 7:
        raise ValueError(f"Expected electrical_buses [N,7], got {bus_rows.shape}")
    buses_electrical = {int(row[0]): row for row in bus_rows}
    incidence = np.zeros((len(bus_ids), len(branch_ids)))
    for edge_position, branch_id in enumerate(branch_ids):
        row = branches[int(branch_id)]
        incidence[bus_index[int(row[1])], edge_position] = 1.0
        incidence[bus_index[int(row[2])], edge_position] = -1.0
    checks.equal("kcl_incidence", flow["line_flow_mw"].astype(np.float64) @ incidence.T - flow["bus_p_injection_mw"], 2e-3, "MW",axes=("time","bus"),timestamps=stamps)
    checks.equal("kcl_global_balance", flow["bus_p_injection_mw"].sum(axis=1, dtype=np.float64), 2e-3, "MW",axes=("time",),timestamps=stamps)
    checks.equal("injection_matches_served_dispatch", flow["bus_p_injection_mw"] - (flow["dispatched_generation_mw"] - flow["served_load_mw"]), 2e-3, "MW")
    checks.equal("requested_load_equals_served_plus_unserved", dispatch["p_load_mw"] - flow["served_load_mw"] - flow["unserved_load_mw"], 2e-3, "MW", "includes requested storage charging; shortfall remains at its optimized bus")
    checks.upper("served_load_nonnegative", -flow["served_load_mw"], 0.0, 1e-6, "MW")
    checks.upper("unserved_load_nonnegative", -flow["unserved_load_mw"], 0.0, 1e-6, "MW")
    checks.upper("unserved_load_below_requested", flow["unserved_load_mw"], dispatch["p_load_mw"], 2e-3, "MW")
    if not config.storage.allow_load_shedding:
        checks.equal("strict_zero_unserved_load", flow["unserved_load_mw"], 2e-3, "MW")
    rates = np.asarray([branches[int(branch_id)][8] for branch_id in branch_ids])
    checks.upper("line_operating_limit", np.abs(flow["line_flow_mw"]), config.storage.line_operating_limit_ratio * rates[None, :], 2e-3, "MW",axes=("time","branch"),timestamps=stamps)

    checks.set_context(stage="F_storage",fields=["storage.charge_mw","storage.discharge_mw","storage.soc_mwh"],time_support="explicit_interval_or_boundary_by_check",engineering_simplification="Lumped battery efficiencies and configured inverter/C-rate/ramp limits")
    charge = storage["charge_mw"].astype(np.float64)
    discharge = storage["discharge_mw"].astype(np.float64)
    emergency = storage["emergency_discharge_mw"].astype(np.float64)
    soc = storage["soc_mwh"].astype(np.float64)
    energy_capacity = storage["site_energy_capacity_mwh"].astype(np.float64)
    power_capacity = storage["site_power_capacity_mw"].astype(np.float64)
    checks.condition("soc_terminal_boundary", soc.shape == (hours + 1, charge.shape[1]))
    # Exported discharge already includes emergency; never count it twice.
    checks.equal("soc_energy_conservation", np.diff(soc, axis=0) - config.storage.charge_efficiency * charge + discharge / config.storage.discharge_efficiency, 2e-3, "MWh",axes=("time","storage_site"),timestamps=stamps,time_support="hour_interval_energy_balance")
    checks.upper("soc_lower_bound", config.storage.minimum_soc_fraction * energy_capacity[None, :] - soc, 0.0, 2e-3, "MWh",axes=("state_time","storage_site"),timestamps=np.r_[stamps,stamps[-1]+1],time_support="T_plus_1_boundary_state")
    checks.upper("soc_upper_bound", soc, config.storage.maximum_soc_fraction * energy_capacity[None, :], 2e-3, "MWh",axes=("state_time","storage_site"),timestamps=np.r_[stamps,stamps[-1]+1],time_support="T_plus_1_boundary_state")
    if config.storage.cyclic_state_of_charge:
        checks.equal("soc_cycle_closure", soc[-1] - soc[0], 2e-3, "MWh")
    elif "op__initial_soc_mwh" in storage:
        checks.equal("soc_initial_state", soc[0] - storage["op__initial_soc_mwh"], 2e-3, "MWh")
    else:
        checks.equal("soc_initial_state", soc[0] - config.storage.initial_soc_fraction * energy_capacity, 2e-3, "MWh")
    checks.upper("storage_charge_nonnegative", -charge, 0.0, 1e-6, "MW")
    checks.upper("storage_discharge_nonnegative", -discharge, 0.0, 1e-6, "MW")
    checks.upper("emergency_part_of_total_discharge", emergency, discharge, 1e-5, "MW")
    checks.upper("storage_no_simultaneous_charge_discharge", np.minimum(charge, discharge), 0.0, 1e-4, "MW")
    checks.upper("storage_shared_inverter_capacity", charge + discharge, power_capacity[None, :], 2e-3, "MW")
    checks.upper("storage_normal_charge_c_rate", charge, config.storage.normal_dispatch_c_rate * energy_capacity[None, :], 2e-3, "MW")
    checks.upper("storage_normal_discharge_c_rate", discharge - emergency, config.storage.normal_dispatch_c_rate * energy_capacity[None, :], 2e-3, "MW")
    net_storage = discharge - charge
    storage_ramps = net_storage - np.roll(net_storage, 1, axis=0) if config.storage.cyclic_state_of_charge else np.diff(net_storage, axis=0)
    checks.upper("storage_ramp_including_emergency", np.abs(storage_ramps), config.storage.storage_power_ramp_fraction_per_hour * power_capacity[None, :], 2e-3, "MW/hour")

    checks.set_context(stage="F_thermal_operation",fields=["flow.dispatched_generation_mw","storage.thermal_bus_ids","electrical.electrical_buses"],time_support="hour_interval_or_ramp",engineering_simplification="Configured nameplate and ramp constraints; no unit commitment or combustion dynamics")
    thermal_ids = storage["thermal_bus_ids"].astype(int)
    thermal_positions = [bus_index[int(bus_id)] for bus_id in thermal_ids]
    thermal_capacity = np.asarray([buses_electrical[int(bus_id)][2] for bus_id in thermal_ids])
    thermal = flow["dispatched_generation_mw"][:, thermal_positions].astype(np.float64).copy()
    thermal_column = {int(bus_id): index for index, bus_id in enumerate(thermal_ids)}
    for site_index, bus_id in enumerate(storage["site_bus_ids"]):
        if int(bus_id) in thermal_column:
            thermal[:, thermal_column[int(bus_id)]] -= discharge[:, site_index]
    checks.upper("thermal_operating_limit", thermal, config.storage.thermal_operating_limit_ratio * thermal_capacity[None, :], 2e-3, "MW", "storage discharge at a thermal bus is subtracted before checking")
    thermal_ramps = thermal - np.roll(thermal, 1, axis=0) if config.storage.cyclic_state_of_charge else np.diff(thermal, axis=0)
    checks.upper("thermal_ramp", np.abs(thermal_ramps), config.storage.thermal_ramp_fraction_per_hour * thermal_capacity[None, :], 2e-3, "MW/hour")

    operation_summary = {"mode": "legacy_without_operation_v1"}
    if "operation_schema_version" in storage:
        operation_summary = check_operation_contracts(checks, storage, flow, electrical, source, config,
            _npz(layout.buses / "grid_nodes.npz").get("thermal_land_ledger"))
        operation_summary["asset_boundary"] = check_frozen_asset_contracts(checks, world_dir, storage, flow, electrical, config)

    for name,note in (
        ("AC_voltage_reactive_power","DC equations do not solve voltage magnitudes or reactive power"),
        ("frequency_stability","No inertia, governor or transient frequency process is simulated"),
        ("reserve_activation_network_deliverability","Island aggregate held reserve does not certify post-activation network flows"),
        ("empirical_regional_calibration","No observed regional data were fitted; scenario validity is not empirical calibration"),
    ):
        checks.status(name,"UNSUPPORTED",note,stage="validation_scope",fields=[name],relation_class="S",time_support="not_applicable_unsupported",engineering_simplification=note)
    validation_scope = checks.summary()
    return {
        "world": world_dir.name, "generator_version": metadata.get("generator_version", "unspecified"),
        "scenario_semantics": metadata.get("scenario_semantics", "unspecified"),
        "passed": validation_scope["passed"], "status":validation_scope["status"],"validation_scope":validation_scope,"checks": checks.rows,
        "summary": {"hours": hours, "population_persons": population, "city_population_persons": city_population,
                    "capacity_factors_descriptive_only": capacity_factors,
                    "peak_exogenous_load_mw": float(source["p_load_mw"].sum(axis=1).max(initial=0.0)),
                    "total_unserved_mwh": float(flow["unserved_load_mw"].sum()),
                    "total_curtailed_mwh": float(flow["curtailed_generation_mw"].sum()),
                    "storage_site_count": int(storage["site_ids"].size), "land_accounting": land_summary,
                    "dynamic_hydrology": hydrology_summary, "source_load": source_load_summary,
                    "operation": operation_summary},
        "limitations": "Checks validate exported physical identities and constraints, not empirical realism or forecast accuracy. A PASS can include explicitly reported unserved energy when load shedding is allowed; it does not imply supply adequacy. Capacity factors are descriptive only. Asset planning mode is declared separately from full-window dispatch foresight. Island reserve accounting does not certify reserve activation deliverability, AC voltage or frequency security.",
    }


def _safe_json(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    return value


def _markdown(report: dict[str, object]) -> str:
    rows = [f"# Physics validation: {report['world']}", "", f"Result: **{report.get('status','PASS' if report['passed'] else 'FAIL')}**", "",
            "Result covers evaluated supported constraints only; UNSUPPORTED and NOT_RUN are not physical certification.",""]
    if report.get("error_category"):
        rows.extend([f"Error category: **{report['error_category']}**",""])
    summary = report.get("summary", {})
    if summary:
        rows.extend([
            f"Unserved energy: **{summary['total_unserved_mwh']:.6g} MWh**; curtailed generation: **{summary['total_curtailed_mwh']:.6g} MWh**.",
            "A physical-constraint PASS does not imply zero supply shortfall.", "",
        ])
    rows.extend(["| Check | Result | Raw max / excess violation | Location / time | Stage |", "|---|---|---:|---|---|"])
    for check in report["checks"]:
        error,violation = check.get("max_residual"),check.get("max_violation")
        number = lambda value:f"{value:.6g}" if isinstance(value,(int,float)) else "—"
        error_text = f"{number(error)} / {number(violation)} {check.get('unit','')}"
        location = json.dumps({"raw":check.get("raw_max_location"),"violation":check.get("max_violation_location"),"time":check.get("time")},ensure_ascii=False)
        rows.append(f"| {check['name']} | {check.get('status','PASS' if check['passed'] else 'FAIL')} | {error_text} | {location} | {check.get('stage','unspecified')} |")
    rows.extend(["", str(report.get("limitations", "")), ""])
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    passed = True
    for candidate in args.world_dirs:
        world_dir = candidate.resolve()
        if not world_dir.is_relative_to(PROJECT_ROOT.parent) or not world_dir.is_dir():
            print(f"Refusing output outside the workspace or missing world directory: {world_dir}", file=sys.stderr)
            passed = False
            continue
        try:
            report = validate_world(world_dir)
        except (KeyError, ValueError, OSError, IndexError, TypeError) as exc:
            report = input_error_report(world_dir.name,exc)
        report = _safe_json(report)
        (world_dir / "physics_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (world_dir / "physics_validation.md").write_text(_markdown(report), encoding="utf-8")
        print(f"{world_dir.name}: {report.get('status','PASS' if report['passed'] else 'FAIL')} ({len(report['checks'])} checks; {report.get('validation_scope',{}).get('counts',{})})")
        passed &= bool(report["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
