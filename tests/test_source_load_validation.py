"""E exported state/energy counterexamples with an analytic four-bus fixture."""
from dataclasses import replace

import numpy as np
import pytest

from scripts.source_load_validation import check_source_load_contracts
from scripts.validate_world_physics import Checks
from world_generator.core.config import SourceLoadConfig, WorldConfig, WorldGridConfig


def _example():
    cfg = replace(SourceLoadConfig(), wind_hub_height_m=10, wind_density_height_mode="surface_proxy", load_thermal_memory_hours=1,
                  load_initial_temperature_mode="configured", load_initial_temperature_c=20)
    config = replace(WorldConfig(), world=WorldGridConfig(width=4, height=1), source_load=cfg)
    ids = np.array([10, 20, 30, 40])
    kinds = np.array(["load_bus", "wind_bus", "pv_bus", "transit_bus"])
    nameplate = np.array([10., 100., 50., 0.])
    bounds = np.array([[0., 1.], [1., 2.]])
    weather = np.zeros((2, 3, 1, 4))
    weather[:, 0] = 20
    weather[1, 0, 0, 0] = 30
    weather[1, 1, 0, 1] = cfg.wind_rated_mps
    weather[1, 2] = 500
    hourly = {"timestamps": np.arange(2), "time_bounds_hours": bounds.copy(), "dynamic": weather,
              "channel_names": np.array(["temperature", "wind_speed", "irradiance"]),
              "diagnostic__air_density_kg_m3": np.full((2, 1, 4), 1.225)}
    source = {"timestamps": np.arange(2), "time_bounds_hours": bounds, "bus_ids": ids, "bus_kinds": kinds,
              "nameplate_capacity_mw": nameplate, "reference_load_mw": np.array([6.2, 0, 0, 0]),
              "initial_effective_temperature_c": np.array([20., 0, 0, 0]),
              "weather_sample_row": np.zeros(4, dtype=int), "weather_sample_col": np.arange(4),
              "p_load_mw": np.array([[10., 0, 0, 0], [12., 0, 0, 0]]),
              "p_gen_available_mw": np.array([[0., 0, 0, 0], [0., 92., 24.3648, 0]]),
              "p_gen_scheduled_mw": np.array([[0., 0, 0, 0], [0., 92., 24.3648, 0]]),
              "capacity_factor_valid": np.array([False, True, True, False])}
    for name in ("hub_wind_speed_mps", "wind_air_density_kg_m3", "pv_poa_w_m2", "pv_module_temperature_c", "load_effective_temperature_c", "load_log_residual"):
        source["diag__" + name] = np.zeros((2, 4))
    source["diag__hub_wind_speed_mps"][1, 1] = cfg.wind_rated_mps
    source["diag__wind_air_density_kg_m3"][:, 1] = 1.225
    source["diag__pv_poa_w_m2"][1, 2] = 500
    source["diag__pv_module_temperature_c"][:, 2] = [20, 40]
    source["diag__load_effective_temperature_c"][:, 0] = [20, 30 - 10 / np.e]
    for stem, power in (("requested_load", "p_load_mw"), ("available_generation", "p_gen_available_mw"), ("planned_generation", "p_gen_scheduled_mw")):
        source[stem + "_energy_mwh"] = source[power].copy()
        source["period_" + stem + "_energy_mwh"] = source[power].sum(axis=0)
    for stem, power in (("available", "p_gen_available_mw"), ("planned", "p_gen_scheduled_mw")):
        source[stem + "_capacity_factor"] = np.divide(source[power], nameplate[None, :], out=np.zeros((2, 4)), where=source["capacity_factor_valid"][None, :])
    metadata = {"refined_grid_buses": [{"bus_id": int(key), "kind": kinds[i], "row": 0, "col": i, "capacity_mw": nameplate[i]} for i, key in enumerate(ids)]}
    return source, hourly, config, metadata


def _run(args):
    checks = Checks()
    summary = check_source_load_contracts(checks, *args)
    return {row["name"]: row for row in checks.rows}, summary


def test_exogenous_energy_and_heat_state_close_without_clipping_requested_load():
    rows, summary = _run(_example())
    assert all(row["passed"] for row in rows.values()), [r for r in rows.values() if not r["passed"]]
    assert summary["requested_load_mwh"] == 22
    assert summary["max_requested_load_design_utilization"] == 1.2
    assert summary["requested_load_exceeds_design_count"] == 1


@pytest.mark.parametrize("case,check", [
    ("energy", "interval_energy_requested_load"), ("period", "period_energy_available_generation"),
    ("capacity_factor", "available_capacity_factor"), ("cf_mask", "capacity_factor_valid_mask"),
    ("wrong_bus_weather", "weather_location_static_anchor"), ("density", "shared_weather_density"),
    ("wind", "wind_engineering_curve"), ("pv_temperature", "pv_module_temperature"),
    ("pv_power", "pv_ac_conversion"), ("heat_recursion", "thermal_memory_recursion"),
    ("heat_initial", "thermal_initial_configuration"), ("schedule", "schedule_below_available"),
])
def test_corrupted_source_export_names_the_failed_transfer(case, check):
    args = _example()
    source = args[0]
    key, location, delta = {
        "energy": ("requested_load_energy_mwh", (0, 0), 3600),
        "period": ("period_available_generation_energy_mwh", (1,), 1),
        "capacity_factor": ("available_capacity_factor", (1, 1), .1),
        "cf_mask": ("capacity_factor_valid", (0,), True),
        "wrong_bus_weather": ("weather_sample_col", (1,), 1),
        "density": ("diag__wind_air_density_kg_m3", (1, 1), .1),
        "wind": ("p_gen_available_mw", (1, 1), -1),
        "pv_temperature": ("diag__pv_module_temperature_c", (1, 2), 1),
        "pv_power": ("p_gen_available_mw", (1, 2), 1),
        "heat_recursion": ("diag__load_effective_temperature_c", (1, 0), 1),
        "heat_initial": ("initial_effective_temperature_c", (0,), 1),
        "schedule": ("p_gen_scheduled_mw", (1, 1), 10),
    }[case]
    source[key][location] += delta
    rows, _ = _run(args)
    assert not rows["source_load_" + check]["passed"]


@pytest.mark.parametrize("case", ["fractional_position", "outside_grid", "missing_interval", "negative_nameplate"])
def test_invalid_spatial_temporal_and_capacity_contracts_fail_explicitly(case):
    args = _example()
    source = args[0]
    if case == "fractional_position": source["weather_sample_col"] = np.array([0., 1.5, 2, 3])
    elif case == "outside_grid": source["weather_sample_row"][1] = 1
    elif case == "missing_interval": source["time_bounds_hours"] = source["time_bounds_hours"][:1]
    else: source["nameplate_capacity_mw"][1] = -1
    with pytest.raises(ValueError):
        _run(args)
