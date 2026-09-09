"""Physical invariants and statistical contracts of synthetic source/load data."""
from dataclasses import replace

import numpy as np
import pytest

from world_generator.core.config import SourceLoadConfig, WorldGridConfig
from world_generator.core.datatypes import GridBus, RefinedGridTopologyState, WeatherStore
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.operation.source_load_forecast import (
    _calendar_features, _effective_temperature, _load_profile,
    _plane_of_array_irradiance, _solar_available, _spatial_load_residuals,
    _validate_config, _wind_available, generate_source_load_forecast,
)
from world_generator.weather.physics import extraterrestrial_hourly_irradiance


def test_wind_thresholds_capacity_and_cubic_region():
    config = replace(SourceLoadConfig(), wind_reference_height_m=90.0, wind_system_loss_fraction=0.0)
    speed = np.asarray([0.0, 2.9, 3.0, 7.0, 11.4, 20.0, 25.0, 35.0])
    power = _wind_available(speed, 10.0, config=config)
    assert np.all(power[[0, 1, 2, 6, 7]] == 0.0)
    np.testing.assert_allclose(power[[4, 5]], 10.0)
    assert power[3] == pytest.approx(10.0 * (7.0**3 - 3.0**3) / (11.4**3 - 3.0**3))
    assert (power >= 0).all() and (power <= 10).all()


def test_wind_hub_height_and_air_density_have_physical_effects():
    config = SourceLoadConfig()
    wind = np.full(4, 6.0)
    unscaled = _wind_available(wind, 20.0, config=replace(config, wind_reference_height_m=90.0))
    scaled = _wind_available(wind, 20.0, config=config)
    assert np.all(scaled > unscaled)
    dense = _wind_available(wind, 20.0, temperature=np.full(4, 15.0), pressure_hpa=np.full(4, 1013.25), config=config)
    thin = _wind_available(wind, 20.0, temperature=np.full(4, 15.0), pressure_hpa=np.full(4, 750.0), config=config)
    assert np.all(dense > thin)
    shutdown = _wind_available(np.full(4, 30.0), 20.0, temperature=np.full(4, 40.0), pressure_hpa=np.full(4, 500.0), config=config)
    assert np.all(shutdown == 0.0)


def test_horizontal_pv_transposition_conserves_ghi_and_night_is_zero():
    config = replace(SourceLoadConfig(), pv_tilt_degrees=0.0)
    stamps = np.arange(24) + 172 * 24
    ghi = 0.6 * extraterrestrial_hourly_irradiance(np.asarray(35.0), 172.0)
    poa = _plane_of_array_irradiance(ghi, stamps, 35.0, config)
    np.testing.assert_allclose(poa, ghi, rtol=1e-6, atol=1e-5)
    assert poa[0] == 0.0 and poa[-1] == 0.0
    assert np.isfinite(_plane_of_array_irradiance(ghi, stamps, 35.0, SourceLoadConfig())).all()


def test_tilt_orientation_changes_direct_sun_collection():
    stamps = np.arange(24) + 0 * 24
    ghi = 0.65 * extraterrestrial_hourly_irradiance(np.asarray(35.0), 0.0)
    south = _plane_of_array_irradiance(ghi, stamps, 35.0, SourceLoadConfig())
    north = _plane_of_array_irradiance(ghi, stamps, 35.0, replace(SourceLoadConfig(), pv_azimuth_degrees=0.0))
    assert south.sum() > north.sum()


def test_pv_temperature_wind_night_and_ac_nameplate():
    poa = np.asarray([0.0, 500.0, 800.0])
    cold = _solar_available(poa, np.full(3, 0.0), 10.0, wind_speed=np.ones(3))
    hot = _solar_available(poa, np.full(3, 40.0), 10.0, wind_speed=np.ones(3))
    windy = _solar_available(poa, np.full(3, 40.0), 10.0, wind_speed=np.full(3, 8.0))
    assert hot[0] == 0.0
    assert np.all(cold[1:] > hot[1:])
    assert np.all(windy[1:] > hot[1:])
    clipped = _solar_available(np.full(3, 1500.0), np.zeros(3), 10.0, wind_speed=np.full(3, 10.0))
    np.testing.assert_allclose(clipped, 10.0)


def test_calendar_uses_real_offset_and_day_type():
    hours, weekday = _calendar_features(np.arange(48) + 2 * 24 + 12, "2025-01-01")
    assert hours[0] == 12 and weekday[0] == 4  # Friday, not reset-to-midnight Monday.
    assert weekday[12] == 5
    with pytest.raises(ValueError, match="consecutive"):
        _calendar_features(np.asarray([0, 2]), "2025-01-01")


def test_commercial_weekend_and_reference_weekly_load():
    config = replace(SourceLoadConfig(), load_residual_std_fraction=0.0)
    weather = np.full((7 * 24, 1), 20.0)
    hour, day = _calendar_features(np.arange(168), "2025-01-06")
    profile = _load_profile(weather, {"temperature": 0}, 100.0, 0.5, np.random.default_rng(1),
                            config=config, hour_of_day=hour, weekday=day,
                            sector_weights=np.asarray([0.0, 1.0, 0.0]))
    assert profile.mean() == pytest.approx(100.0, rel=1e-6)
    assert profile.reshape(7, 24)[5:].mean() < profile.reshape(7, 24)[:5].mean()


def test_temperature_response_is_two_sided_and_causal():
    config = replace(SourceLoadConfig(), load_residual_std_fraction=0.0)
    series = []
    for temperature in [0.0, 20.0, 35.0, 45.0]:
        series.append(_load_profile(np.full((168, 1), temperature), {"temperature": 0}, 100.0,
                                   0.5, np.random.default_rng(1), config=config).mean())
    assert series[0] > series[1] and series[3] > series[2] > series[1]
    temperature = np.r_[np.full(24, 20.0), np.full(24, 40.0)]
    effective = _effective_temperature(temperature, 8.0)
    np.testing.assert_allclose(effective[:24], 20.0)
    assert 20.0 < effective[24] < effective[-1] < 40.0


def test_residual_stationary_variance_time_and_spatial_correlation():
    config = SourceLoadConfig()
    residual = _spatial_load_residuals(50000, np.asarray([[0, 0], [1, 0], [200, 0]]), np.random.default_rng(42), config)
    np.testing.assert_allclose(residual.std(axis=0), config.load_residual_std_fraction, rtol=0.05)
    correlation = np.corrcoef(residual.T)
    assert correlation[0, 1] > 0.45 and abs(correlation[0, 2]) < 0.06
    assert np.corrcoef(residual[1:, 0], residual[:-1, 0])[0, 1] == pytest.approx(config.load_residual_ar1, abs=0.02)
    np.testing.assert_allclose(np.exp(residual - config.load_residual_std_fraction**2 / 2).mean(axis=0), 1.0, atol=0.002)


def test_source_load_store_is_reproducible_and_obeys_units():
    hours = 48
    dynamic = np.zeros((hours, 4, 1, 4), dtype=np.float32)
    dynamic[:, 0] = 20.0
    dynamic[:, 1] = 6.0
    dynamic[:, 3] = 1013.25
    solar = 0.6 * extraterrestrial_hourly_irradiance(np.asarray(35.0), 0.0)
    dynamic[:, 2] = np.tile(solar, 2)[:, None, None]
    weather = WeatherStore(dynamic, np.zeros((hours, 1, 4)), np.arange(hours),
                           ("temperature", "wind_speed", "irradiance", "pressure"), time_unit="hour")
    kinds = ("load_bus", "wind_bus", "pv_bus", "thermal_bus")
    buses = tuple(GridBus(i, kind, 0, i, 2.0 * i, 0.0, 100.0, 0.5, 0.0, kind, i) for i, kind in enumerate(kinds))
    empty = np.zeros((1, 4))
    topology = RefinedGridTopologyState(empty, empty, empty, buses, ())
    electrical = build_grid_electrical(topology)
    first = generate_source_load_forecast(weather, topology, electrical, np.random.default_rng(9), grid=WorldGridConfig(height=1, width=4))
    second = generate_source_load_forecast(weather, topology, electrical, np.random.default_rng(9), grid=WorldGridConfig(height=1, width=4))
    np.testing.assert_array_equal(first.p_load_mw, second.p_load_mw)
    assert first.data_semantics == "synthetic_realization"
    assert np.isfinite(first.p_gen_available_mw).all()
    assert (first.p_gen_available_mw <= 100.0).all()
    assert (first.p_gen_scheduled_mw <= first.p_gen_available_mw + 1e-5).all()
    np.testing.assert_allclose(first.q_load_mvar[:, 0] / first.p_load_mw[:, 0], np.tan(np.arccos(0.94)), rtol=1e-6)


def test_source_load_entry_uses_physical_grid_distance_not_normalized_xy():
    hours = 12000
    dynamic = np.zeros((hours, 3, 1, 2), dtype=np.float32)
    dynamic[:, 0] = 20.0
    weather = WeatherStore(dynamic, np.zeros((hours, 1, 2)), np.arange(hours),
                           ("temperature", "wind_speed", "irradiance"), time_unit="hour")
    buses = tuple(GridBus(i, "load_bus", 0, i, float(i), 0.0, 100.0, 0.5, 0.0, "load_bus", i) for i in range(2))
    empty = np.zeros((1, 2))
    topology = RefinedGridTopologyState(empty, empty, empty, buses, ())
    electrical = build_grid_electrical(topology)
    config = replace(SourceLoadConfig(), load_common_variance_fraction=0.9, load_residual_ar1=0.0)
    grid = WorldGridConfig(height=1, width=2, cell_size_km=1.0)

    def generate(current_topology, current_grid, current_config=config):
        return generate_source_load_forecast(weather, current_topology, electrical, np.random.default_rng(41),
                                             config=current_config, grid=current_grid).p_load_mw

    near = generate(topology, grid)
    far = generate(topology, replace(grid, cell_size_km=100.0))
    reference = generate(topology, grid, replace(config, load_residual_std_fraction=0.0))
    near_residual = np.log(near.astype(np.float64) / reference)
    far_residual = np.log(far.astype(np.float64) / reference)
    near_correlation = np.corrcoef(near_residual.T)[0, 1]
    far_correlation = np.corrcoef(far_residual.T)[0, 1]
    assert near_correlation == pytest.approx(0.9 * np.exp(-1.0 / 20.0), abs=0.03)
    assert far_correlation == pytest.approx(0.9 * np.exp(-100.0 / 20.0), abs=0.03)
    # Legacy display coordinates and a common physical origin translation
    # must not change distances or draw a different stochastic realization.
    changed_xy = replace(topology, refined_buses=tuple(replace(bus, x=1.0-bus.x, y=0.7) for bus in buses))
    np.testing.assert_array_equal(generate(changed_xy, grid), near)
    np.testing.assert_array_equal(generate(topology, replace(grid, origin_x_km=300.0, origin_y_km=-70.0)), near)


@pytest.mark.parametrize("field,value", [
    ("wind_rated_mps", 2.0), ("load_residual_ar1", 1.0),
    ("load_spatial_correlation_km", 0.0), ("pv_inverter_efficiency", 1.1),
    ("load_common_variance_fraction", -0.1), ("load_residual_std_fraction", float("nan")),
])
def test_nonphysical_config_rejected(field, value):
    with pytest.raises(ValueError):
        _validate_config(replace(SourceLoadConfig(), **{field: value}))
