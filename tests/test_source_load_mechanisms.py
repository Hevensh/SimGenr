"""E mechanism counterfactuals: fixed assets, clock and residual random seed."""
from dataclasses import replace

import numpy as np
import pytest

from world_generator.core.config import SourceLoadConfig, WorldGridConfig
from world_generator.core.contracts import integrate_power_mwh
from world_generator.core.datatypes import GridBus, RefinedGridTopologyState, WeatherStore
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.operation.source_load_forecast import (
    _effective_temperature, _thermal_locality_weights, _validate_config,
    _wind_available, _wind_conditions, generate_source_load_forecast,
)
from world_generator.weather.physics import diagnose_moist_air, extraterrestrial_hourly_irradiance
from world_generator.weather.weather_generator import WEATHER_CHANNELS


def _case(temperature=30.0, q=.005, hours=48):
    grid = WorldGridConfig(height=1, width=4, cell_size_km=1.)
    stamps = 172 * 24 + np.arange(hours)
    values = np.zeros((hours, 9, 1, 4), dtype=np.float32)
    values[:, 0] = 6.; values[:, 2] = 6.
    values[:, 3] = np.broadcast_to(np.asarray(temperature), (hours,))[:, None, None]
    humidity = np.full((hours, 1, 4), q, dtype=np.float32)
    pressure0 = np.full_like(humidity, 1013.25)
    rh, pressure, density = diagnose_moist_air(values[:, 3], humidity, np.zeros((1, 4)), pressure0)
    values[:, 4] = rh; values[:, 5] = pressure
    values[:, 6] = .3
    for day in np.unique(stamps // 24):
        indices = np.flatnonzero(stamps // 24 == day)
        ghi = .5 * extraterrestrial_hourly_irradiance(np.array(35.), float(day))
        values[indices, 8] = ghi[stamps[indices] % 24, None, None]
    weather = WeatherStore(values, np.zeros((hours, 1, 4), dtype=int), stamps, WEATHER_CHANNELS, "hour", 172,
                           diagnostics={"specific_humidity_kg_kg": humidity, "sea_level_pressure_hpa": pressure0, "air_density_kg_m3": density.astype(np.float32)})
    kinds = ("load_bus", "wind_bus", "pv_bus", "thermal_bus")
    buses = tuple(GridBus(i, kind, 0, i, i / 4, 0., 100., .5, 0., kind, i) for i, kind in enumerate(kinds))
    empty = np.zeros((1, 4))
    topology = RefinedGridTopologyState(empty, empty, empty, buses, ())
    return weather, topology, build_grid_electrical(topology), grid


def _generate(case, config=None, seed=42):
    weather, topology, electrical, grid = case
    return generate_source_load_forecast(weather, topology, electrical, np.random.default_rng(seed), config=config, grid=grid)


@pytest.mark.parametrize("seed", [42, 123])
def test_humidity_reaches_wind_through_same_weather_density(seed):
    dry = _case(q=0.)
    moist = _case(q=.02)
    first, second = _generate(dry, seed=seed), _generate(moist, seed=seed)
    assert np.all(second.diagnostics["wind_air_density_kg_m3"][:, 1] < first.diagnostics["wind_air_density_kg_m3"][:, 1])
    assert np.all(second.p_gen_available_mw[:, 1] < first.p_gen_available_mw[:, 1])
    np.testing.assert_array_equal(first.diagnostics["load_log_residual"], second.diagnostics["load_log_residual"])
    np.testing.assert_array_equal(first.p_load_mw, second.p_load_mw)
    assert second.metadata["wind_density_source_by_bus_id"]["1"] == "weather_moist_air_diagnostic"
    # Changing legacy T while holding authoritative surface rho/p fixed must
    # not trigger a hidden dry-air recomputation in the wind conversion.
    config = SourceLoadConfig()
    speed, rho, p = np.array([6.]), np.array([1.1]), np.array([1000.])
    cold = _wind_available(speed, 100., temperature=np.array([0.]), pressure_hpa=p, air_density_kg_m3=rho, config=config)
    hot = _wind_available(speed, 100., temperature=np.array([40.]), pressure_hpa=p, air_density_kg_m3=rho, config=config)
    np.testing.assert_array_equal(cold, hot)


@pytest.mark.parametrize("density_factor", [.5, 1., 2.])
def test_physical_wind_safety_and_continuity_at_nominal_rated(density_factor):
    config = replace(SourceLoadConfig(), wind_reference_height_m=90., wind_density_height_mode="surface_proxy", wind_system_loss_fraction=0.)
    epsilon = 1e-4
    speeds = np.array([3.-epsilon, 3., 3.+epsilon, 11.4-epsilon, 11.4, 11.4+epsilon, 25.-epsilon, 25.])
    rho = np.full_like(speeds, density_factor * config.wind_reference_density_kg_m3)
    power = _wind_available(speeds, 100., air_density_kg_m3=rho, config=config)
    np.testing.assert_array_equal(power[[0, 1, 7]], 0.)
    assert 0 < power[2] < .01
    assert np.ptp(power[3:6]) < .02  # No density-dependent artificial rated jump.
    assert np.all((power >= 0) & (power <= 100))
    if density_factor == .5:
        assert power[4] == pytest.approx(50., rel=1e-6)
    elif density_factor == 1.:
        assert power[4] == pytest.approx(100., rel=1e-6)


def test_surface_pressure_to_hub_uses_full_agl_not_wind_reference_height():
    config = SourceLoadConfig()
    wind, p, rho = np.array([6.]), np.array([1000.]), np.array([1.1])
    first = _wind_conditions(wind, 100., pressure_hpa=p, air_density_kg_m3=rho, config=config)
    second = _wind_conditions(wind, 100., pressure_hpa=p, air_density_kg_m3=rho, config=replace(config, wind_reference_height_m=50.))
    expected = rho * np.exp(-9.80665 * rho * config.wind_hub_height_m / (100 * p))
    np.testing.assert_allclose(first[1], expected, rtol=1e-12)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0][0] > second[0][0]


def test_legacy_density_fallback_is_declared_and_can_be_disabled():
    weather, topology, electrical, grid = _case()
    legacy = replace(weather, diagnostics={})
    result = _generate((legacy, topology, electrical, grid))
    assert result.metadata["wind_density_source_by_bus_id"]["1"] == "legacy_dry_air_from_surface_pressure_temperature"
    with pytest.raises(ValueError, match="fallback is disabled"):
        _generate((legacy, topology, electrical, grid), replace(SourceLoadConfig(), wind_density_fallback="error"))
    no_pressure_names = tuple(name for name in legacy.channel_names if name != "pressure")
    no_pressure = replace(legacy, dynamic=np.delete(legacy.dynamic, 5, axis=1), channel_names=no_pressure_names)
    result = _generate((no_pressure, topology, electrical, grid))
    assert result.metadata["wind_density_source_by_bus_id"]["1"] == "legacy_reference_density_at_hub"


def test_fixed_ghi_heat_reduces_pv_and_has_causal_load_response_with_shared_rng():
    temperatures = np.r_[np.full(24, 20.), np.full(24, 40.)]
    baseline, heat = _generate(_case(20.)), _generate(_case(temperatures))
    np.testing.assert_array_equal(baseline.timestamps, heat.timestamps)
    np.testing.assert_array_equal(baseline.diagnostics["load_log_residual"], heat.diagnostics["load_log_residual"])
    np.testing.assert_array_equal(baseline.p_load_mw[:24], heat.p_load_mw[:24])
    assert np.all(heat.p_load_mw[24:, 0] > baseline.p_load_mw[24:, 0])
    active = baseline.diagnostics["pv_poa_w_m2"][24:, 2] > 0
    assert np.all(heat.p_gen_available_mw[24:, 2][active] < baseline.p_gen_available_mw[24:, 2][active])
    np.testing.assert_array_equal(heat.p_gen_available_mw[:, 2][heat.diagnostics["pv_poa_w_m2"][:, 2] == 0], 0.)


def test_cloud_has_no_second_pv_attenuation_and_rain_is_not_required():
    weather, topology, electrical, grid = _case()
    baseline = _generate((weather, topology, electrical, grid))
    values = weather.dynamic.copy(); values[:, 6] = 1.; values[:, 7] = 0.
    cloudy = _generate((replace(weather, dynamic=values), topology, electrical, grid))
    np.testing.assert_array_equal(baseline.p_gen_available_mw[:, 2], cloudy.p_gen_available_mw[:, 2])
    values[:, 8] *= .3
    dimmed = _generate((replace(weather, dynamic=values), topology, electrical, grid))
    active = baseline.p_gen_available_mw[:, 2] > 0
    assert np.all(dimmed.p_gen_available_mw[:, 2][active] < baseline.p_gen_available_mw[:, 2][active])


def test_explicit_initial_thermal_state_updates_first_interval_and_keeps_demand():
    config = replace(SourceLoadConfig(), load_initial_temperature_mode="configured", load_initial_temperature_c=20., load_residual_std_fraction=0.)
    result = _generate(_case(45.), config)
    assert result.initial_effective_temperature_c[0] == 20.
    expected_first = np.exp(-1 / 8) * 20 + (1 - np.exp(-1 / 8)) * 45
    assert result.diagnostics["load_effective_temperature_c"][0, 0] == pytest.approx(expected_first, rel=1e-6)
    assert result.p_load_mw[:, 0].max() > result.nameplate_capacity_mw[0]
    half = _effective_temperature(np.array([45., 45.]), 8., initial_temperature_c=20., step_hours=.5)
    hour = _effective_temperature(np.array([45.]), 8., initial_temperature_c=20.)
    np.testing.assert_allclose(half[-1], hour[0], rtol=1e-12)
    prefix = _effective_temperature(np.full(8, 20.), 8., initial_temperature_c=10.)
    with_future = _effective_temperature(np.r_[np.full(8, 20.), np.full(5, 50.)], 8., initial_temperature_c=10.)
    np.testing.assert_array_equal(prefix, with_future[:8])


def test_thermal_locality_is_half_at_declared_km_for_two_resolutions():
    _, topology, _, _ = _case()
    original = list(topology.refined_buses)
    original[3] = replace(original[3], col=0)
    original[0] = replace(original[0], col=12)
    first = _thermal_locality_weights(tuple(original), [3], [0], [], half_distance_km=24., cell_size_km=2.)[0]
    refined = tuple(replace(bus, col=2 * bus.col) for bus in original)
    second = _thermal_locality_weights(refined, [3], [0], [], half_distance_km=24., cell_size_km=1.)[0]
    np.testing.assert_allclose(first, .5, rtol=1e-12)
    np.testing.assert_array_equal(first, second)


def test_energy_capacity_factors_and_zero_nameplate_masks():
    result = _generate(_case(hours=24))
    arrays = result.as_arrays()
    assert integrate_power_mwh(np.full(24, 10.), 1.) == 240.
    assert integrate_power_mwh(np.full(24, 10.), .5) == 120.
    # Store owns the energy field contract; these are direct independent sums.
    np.testing.assert_allclose(arrays["available_generation_energy_mwh"], result.p_gen_available_mw, rtol=1e-6)
    np.testing.assert_allclose(arrays["period_available_generation_energy_mwh"], result.p_gen_available_mw.sum(axis=0, dtype=float), rtol=1e-6)
    np.testing.assert_array_equal(arrays["capacity_factor_valid"], [False, True, True, True])
    np.testing.assert_allclose(arrays["available_capacity_factor"][:, 1:], result.p_gen_available_mw[:, 1:] / 100., rtol=1e-6)
    weather, topology, electrical, grid = _case(hours=24)
    topology = replace(topology, refined_buses=tuple(replace(bus, capacity_mw=0.) if bus.bus_id == 1 else bus for bus in topology.refined_buses))
    zero = _generate((weather, topology, build_grid_electrical(topology), grid)).as_arrays()
    assert not zero["capacity_factor_valid"][1]
    np.testing.assert_array_equal(zero["available_capacity_factor"][:, 1], 0.)


@pytest.mark.parametrize("field,value", [("row", -1), ("col", 4), ("row", .5), ("row", np.nan), ("bus_id", -1), ("bus_id", .5), ("capacity_mw", -1), ("capacity_mw", np.nan)])
def test_invalid_asset_and_sampling_inputs_fail(field, value):
    weather, topology, electrical, grid = _case()
    buses = list(topology.refined_buses); buses[0] = replace(buses[0], **{field: value})
    with pytest.raises(ValueError):
        _generate((weather, replace(topology, refined_buses=tuple(buses)), electrical, grid))


@pytest.mark.parametrize("problem", ["duplicate_ids", "missing_electrical_id", "bad_density", "bad_diagnostic_shape", "bad_clock", "negative_ghi", "negative_wind", "capacity_mismatch"])
def test_other_invalid_weather_and_asset_contracts_fail(problem):
    weather, topology, electrical, grid = _case()
    if problem == "duplicate_ids":
        buses = list(topology.refined_buses); buses[1] = replace(buses[1], bus_id=0); topology = replace(topology, refined_buses=tuple(buses))
    elif problem == "missing_electrical_id":
        electrical = replace(electrical, bus_params=electrical.bus_params[:-1])
    elif problem in {"bad_density", "bad_diagnostic_shape"}:
        diagnostics = dict(weather.diagnostics)
        diagnostics["air_density_kg_m3"] = np.full((48, 1, 4), -1.) if problem == "bad_density" else np.ones((47, 1, 4))
        weather = replace(weather, diagnostics=diagnostics)
    elif problem == "bad_clock":
        weather = replace(weather, timestamps=weather.timestamps + np.arange(48))
    elif problem in {"negative_ghi", "negative_wind"}:
        values = weather.dynamic.copy(); values[:, 8 if problem == "negative_ghi" else 2] = -1; weather = replace(weather, dynamic=values)
    elif problem == "capacity_mismatch":
        parameters = list(electrical.bus_params); parameters[1] = replace(parameters[1], p_capacity_mw=90.); electrical = replace(electrical, bus_params=tuple(parameters))
    with pytest.raises(ValueError):
        _generate((weather, topology, electrical, grid))


def test_explicit_grid_shape_must_match_and_legacy_grid_assumption_is_declared():
    weather, topology, electrical, grid = _case()
    with pytest.raises(ValueError, match="grid shape"):
        _generate((weather, topology, electrical, replace(grid, height=30, width=30)))
    legacy = generate_source_load_forecast(weather, topology, electrical, np.random.default_rng(42))
    assert legacy.metadata["grid_source"] == "legacy_default_grid_assumption"
    assert legacy.metadata["weather_grid_shape"] == [1, 4]


@pytest.mark.parametrize("kwargs", [
    {"wind_reference_density_kg_m3": 0.}, {"wind_density_height_mode": "unknown"},
    {"wind_density_fallback": "silent"}, {"pv_module_height_m": -1.},
    {"load_initial_temperature_mode": "future_mean"}, {"load_initial_temperature_c": float("nan")},
    {"thermal_dispatch_half_distance_km": 0.}, {"pv_temperature_coefficient_per_c": .004},
])
def test_new_mechanism_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        SourceLoadConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"wind_reference_density_kg_m3": -1.}, {"load_initial_temperature_mode": "future_mean"}])
def test_world_config_read_rejects_source_parameters_before_generation(monkeypatch, kwargs):
    import world_generator.core.config as configuration

    # Isolate parsing from file IO; constructor validation must run in loader.
    monkeypatch.setattr(configuration, "_load_mapping", lambda path: {"source_load": kwargs})
    with pytest.raises(ValueError):
        configuration.load_world_config("unused_config_mapping.yaml")
