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
from world_generator.weather.physics import extraterrestrial_hourly_irradiance, latitude_grid
from world_generator.weather.physics import diagnose_moist_air, saturation_vapor_pressure_hpa


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _positive_max(values: np.ndarray) -> float:
    return float(np.max(values, initial=0.0))


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def equal(self, name: str, residual: np.ndarray, tolerance: float, unit: str = "", note: str = "") -> None:
        error = _max_abs(np.asarray(residual, dtype=np.float64))
        self.rows.append({"name": name, "passed": bool(np.isfinite(error) and error <= tolerance),
                          "max_error": error, "tolerance": tolerance, "unit": unit, "note": note})

    def upper(self, name: str, values: np.ndarray, upper: np.ndarray | float, tolerance: float, unit: str = "", note: str = "") -> None:
        residual = np.asarray(values, dtype=np.float64) - upper
        error = _positive_max(residual)
        self.rows.append({"name": name, "passed": bool(np.isfinite(residual).all() and error <= tolerance),
                          "max_error": error, "tolerance": tolerance, "unit": unit, "note": note})

    def condition(self, name: str, passed: bool, note: str = "") -> None:
        self.rows.append({"name": name, "passed": bool(passed), "note": note})


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


def validate_world(world_dir: Path) -> dict[str, object]:
    world_dir = world_dir.resolve()
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
    checks = Checks()
    for artifact, arrays in data.items():
        invalid = [name for name, values in arrays.items() if values.dtype.kind in "fc" and not np.isfinite(values).all()]
        checks.condition(f"finite_{artifact}", not invalid, ", ".join(invalid))
    static, daily, hourly = data["static"], data["daily"], data["hourly"]
    source, storage, dispatch, flow, electrical = (data[key] for key in ("source", "storage", "dispatch", "flow", "electrical"))
    stamps = hourly["timestamps"].astype(np.int64)
    hours = len(stamps)
    checks.condition("nonempty_consecutive_hourly_timestamps", hours > 0 and bool(np.all(np.diff(stamps) == 1)))
    for name in ("source", "storage", "dispatch", "flow"):
        checks.condition(f"timestamps_match_{name}", np.array_equal(data[name]["timestamps"], stamps))

    population = float(np.sum(static["population_density"], dtype=np.float64) * config.world.cell_size_km**2)
    city_population = float(sum(city["population"] for city in metadata["cities"]))
    checks.equal("population_mass", np.asarray(population - city_population), max(0.1, city_population * 2e-6), "persons")
    checks.upper("population_nonnegative", -static["population_density"], 0.0, 1e-7, "persons/km2")

    daily_summary_path = layout.weather / "hourly_daily_summary.npz"
    daily_summary = _npz(daily_summary_path) if daily_summary_path.exists() else None
    check_weather_contracts(checks, daily, hourly, config, daily_summary)

    source_ids = source["bus_ids"].astype(int)
    source_kinds = np.asarray(source["bus_kinds"])
    original_buses = {int(bus["bus_id"]): bus for bus in metadata["refined_grid_buses"]}
    checks.upper("exogenous_load_nonnegative", -source["p_load_mw"], 0.0, 1e-6, "MW")
    checks.upper("exogenous_availability_nonnegative", -source["p_gen_available_mw"], 0.0, 1e-6, "MW")
    capacity_factors: dict[str, float | None] = {}
    for kind in ("wind_bus", "pv_bus"):
        positions = np.flatnonzero(source_kinds == kind)
        capacities = np.asarray([original_buses[int(source_ids[index])]["capacity_mw"] for index in positions])
        availability = source["p_gen_available_mw"][:, positions]
        checks.upper(f"{kind}_ac_capacity", availability, capacities[None, :], 1e-4, "MW")
        capacity_factors[kind] = float(availability.sum() / (hours * capacities.sum())) if capacities.sum() > 0 else None

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
    checks.equal("kcl_incidence", flow["line_flow_mw"].astype(np.float64) @ incidence.T - flow["bus_p_injection_mw"], 2e-3, "MW")
    checks.equal("kcl_global_balance", flow["bus_p_injection_mw"].sum(axis=1, dtype=np.float64), 2e-3, "MW")
    checks.equal("injection_matches_served_dispatch", flow["bus_p_injection_mw"] - (flow["dispatched_generation_mw"] - flow["served_load_mw"]), 2e-3, "MW")
    checks.equal("requested_load_equals_served_plus_unserved", dispatch["p_load_mw"] - flow["served_load_mw"] - flow["unserved_load_mw"], 2e-3, "MW", "includes requested storage charging; shortfall remains at its optimized bus")
    checks.upper("served_load_nonnegative", -flow["served_load_mw"], 0.0, 1e-6, "MW")
    checks.upper("unserved_load_nonnegative", -flow["unserved_load_mw"], 0.0, 1e-6, "MW")
    checks.upper("unserved_load_below_requested", flow["unserved_load_mw"], dispatch["p_load_mw"], 2e-3, "MW")
    if not config.storage.allow_load_shedding:
        checks.equal("strict_zero_unserved_load", flow["unserved_load_mw"], 2e-3, "MW")
    rates = np.asarray([branches[int(branch_id)][8] for branch_id in branch_ids])
    checks.upper("line_operating_limit", np.abs(flow["line_flow_mw"]), config.storage.line_operating_limit_ratio * rates[None, :], 2e-3, "MW")

    charge = storage["charge_mw"].astype(np.float64)
    discharge = storage["discharge_mw"].astype(np.float64)
    emergency = storage["emergency_discharge_mw"].astype(np.float64)
    soc = storage["soc_mwh"].astype(np.float64)
    energy_capacity = storage["site_energy_capacity_mwh"].astype(np.float64)
    power_capacity = storage["site_power_capacity_mw"].astype(np.float64)
    checks.condition("soc_terminal_boundary", soc.shape == (hours + 1, charge.shape[1]))
    # Exported discharge already includes emergency; never count it twice.
    checks.equal("soc_energy_conservation", np.diff(soc, axis=0) - config.storage.charge_efficiency * charge + discharge / config.storage.discharge_efficiency, 2e-3, "MWh")
    checks.upper("soc_lower_bound", config.storage.minimum_soc_fraction * energy_capacity[None, :] - soc, 0.0, 2e-3, "MWh")
    checks.upper("soc_upper_bound", soc, config.storage.maximum_soc_fraction * energy_capacity[None, :], 2e-3, "MWh")
    if config.storage.cyclic_state_of_charge:
        checks.equal("soc_cycle_closure", soc[-1] - soc[0], 2e-3, "MWh")
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

    return {
        "world": world_dir.name, "generator_version": metadata.get("generator_version", "unspecified"),
        "scenario_semantics": metadata.get("scenario_semantics", "unspecified"),
        "passed": all(row["passed"] for row in checks.rows), "checks": checks.rows,
        "summary": {"hours": hours, "population_persons": population, "city_population_persons": city_population,
                    "capacity_factors_descriptive_only": capacity_factors,
                    "peak_exogenous_load_mw": float(source["p_load_mw"].sum(axis=1).max(initial=0.0)),
                    "total_unserved_mwh": float(flow["unserved_load_mw"].sum()),
                    "total_curtailed_mwh": float(flow["curtailed_generation_mw"].sum()),
                    "storage_site_count": int(storage["site_ids"].size)},
        "limitations": "Checks validate exported physical identities and constraints, not empirical realism or forecast accuracy. A PASS can include explicitly reported unserved energy when load shedding is allowed; it does not imply supply adequacy. Capacity factors are descriptive only. Final graph and dispatch use perfect foresight.",
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
    rows = [f"# Physics validation: {report['world']}", "", f"Result: **{'PASS' if report['passed'] else 'FAIL'}**", ""]
    summary = report.get("summary", {})
    if summary:
        rows.extend([
            f"Unserved energy: **{summary['total_unserved_mwh']:.6g} MWh**; curtailed generation: **{summary['total_curtailed_mwh']:.6g} MWh**.",
            "A physical-constraint PASS does not imply zero supply shortfall.", "",
        ])
    rows.extend(["| Check | Result | Maximum error / violation | Tolerance |", "|---|---|---:|---:|"])
    for check in report["checks"]:
        error = check.get("max_error")
        error_text = f"{error:.6g} {check.get('unit', '')}" if isinstance(error, (int, float)) else "—"
        tolerance = check.get("tolerance", "—")
        rows.append(f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {error_text} | {tolerance} |")
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
        except (KeyError, ValueError, FileNotFoundError, IndexError) as exc:
            report = {"world": world_dir.name, "passed": False, "checks": [{"name": "read_and_align_inputs", "passed": False, "note": str(exc)}], "limitations": str(exc)}
        report = _safe_json(report)
        (world_dir / "physics_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (world_dir / "physics_validation.md").write_text(_markdown(report), encoding="utf-8")
        print(f"{world_dir.name}: {'PASS' if report['passed'] else 'FAIL'} ({len(report['checks'])} checks)")
        passed &= bool(report["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
