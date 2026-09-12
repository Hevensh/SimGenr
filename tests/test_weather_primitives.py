"""B: counterexamples and contracts, not snapshots of one stochastic output."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from test_weather_physics import _fixture
from world_generator.core.config import WeatherConfig
from world_generator.core.datatypes import WeatherStore
from world_generator.dataset.loader import _slice_weather
from world_generator.operation.stage_cache import load_hourly_weather_checkpoint
from world_generator.weather.physics import diagnose_moist_air, saturation_vapor_pressure_hpa, specific_humidity_from_relative_humidity
from world_generator.weather.random_fields import displacement_cells, gaussian_field, temporal_retention
from world_generator.weather.weather_generator import (
    _advect_and_innovate, _bounded_weighted_total, _condition_cloud_radiation,
    aggregate_daily_weather, generate_daily_weather, generate_hourly_weather_week,
)


@pytest.mark.parametrize("seed", [42, 123])
def test_hourly_primitives_diagnose_rh_pressure_density_and_aggregate(seed):
    terrain, water, climate, grid = _fixture((4, 5))
    terrain = replace(terrain, elevation=np.full((4, 5), 1800.0, dtype=np.float32))
    config = WeatherConfig(days=8, hourly_week_days=7)
    daily = generate_daily_weather(terrain, water, climate, grid, config, np.random.default_rng(seed))
    hourly = generate_hourly_weather_week(daily, config, np.random.default_rng(seed + 1), grid=grid)
    q = hourly.diagnostics["specific_humidity_kg_kg"]
    p0 = hourly.diagnostics["sea_level_pressure_hpa"]
    rh, pressure, density = diagnose_moist_air(hourly.dynamic[:, 3], q, terrain.elevation, p0)
    np.testing.assert_allclose(hourly.dynamic[:, 4], rh, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(hourly.dynamic[:, 5], pressure, rtol=2e-6)
    np.testing.assert_allclose(hourly.diagnostics["air_density_kg_m3"], density, rtol=2e-6)
    assert np.all((q >= 0) & (rh <= 1.000001))
    assert np.all(hourly.diagnostics["specific_humidity_adjustment_kg_kg"] <= 1e-9)
    summary = aggregate_daily_weather(hourly)
    np.testing.assert_allclose(summary.diagnostics["air_density_kg_m3"], density.reshape(7, 24, 4, 5).mean(axis=1), rtol=2e-6)
    daily_rh_at_mean_state, _, _ = diagnose_moist_air(summary.dynamic[:, 3], summary.diagnostics["specific_humidity_kg_kg"], terrain.elevation, summary.diagnostics["sea_level_pressure_hpa"])
    assert np.max(np.abs(summary.dynamic[:, 4] - daily_rh_at_mean_state)) > 0.001
    assert np.all(summary.dynamic[:, 2] >= np.hypot(summary.dynamic[:, 0], summary.dynamic[:, 1]) - 1e-6)
    directions = np.arctan2(hourly.dynamic[:24, 1], hourly.dynamic[:24, 0])
    assert np.std(np.unwrap(directions, axis=0), axis=0).max() > 0.03
    arrays = hourly.as_arrays()
    np.testing.assert_array_equal(arrays["time_bounds_hours"][:, 1] - arrays["time_bounds_hours"][:, 0], 1)
    assert json.loads(str(arrays["weather_metadata_json"]))["statistic_role"] == "hourly_realization"
    assert hourly.metadata["hourly_window_selection"] == "seeded_random_contiguous_window_from_daily_anchors"
    assert hourly.metadata["hourly_window_start_day"] == hourly.timestamps[0] // 24


def test_moist_air_ideal_gas_and_humidity_are_independently_checkable():
    temperature, q, z, p0 = np.array([20., -10.]), np.array([.01, .001]), np.array([0., 2500.]), np.array([1013.25, 1000.])
    rh, p, rho = diagnose_moist_air(temperature, q, z, p0)
    vapor = rh * saturation_vapor_pressure_hpa(temperature)
    independent_rho = ((p - vapor) * 100 / 287.05 + vapor * 100 / 461.5) / (temperature + 273.15)
    np.testing.assert_allclose(rho, independent_rho, rtol=1e-12)
    assert p[1] < p[0]
    assert rho[1] < rho[0]


@pytest.mark.parametrize("temperature", [-300., -273.15, -91., 80.])
def test_moist_air_rejects_unsupported_temperature_without_silent_clipping(temperature):
    with pytest.raises(ValueError, match="numerical domain"):
        specific_humidity_from_relative_humidity(temperature, .5, 1013.25)
    with pytest.raises(ValueError, match="numerical domain"):
        diagnose_moist_air(temperature, .005, 0, 1013.25)
    assert np.isfinite(specific_humidity_from_relative_humidity(65., .5, 1013.25))


def test_conditioned_reversing_wind_has_zero_vector_mean_but_positive_mean_speed():
    terrain, water, climate, grid = _fixture((1, 1))
    config = WeatherConfig(days=1, hourly_week_days=1, hourly_generation_mode="daily_conditioned", hourly_wind_variability=0, hourly_wind_direction_std_degrees=0)
    daily = generate_daily_weather(terrain, water, climate, grid, config, np.random.default_rng(42))
    values = daily.dynamic.copy()
    values[:, :2] = 0
    values[:, 2] = 5
    hourly = generate_hourly_weather_week(replace(daily, dynamic=values), config, np.random.default_rng(123), grid=grid)
    np.testing.assert_allclose(hourly.dynamic[:12, 0], 5, atol=1e-5)
    np.testing.assert_allclose(hourly.dynamic[12:, 0], -5, atol=1e-5)
    np.testing.assert_allclose(hourly.dynamic[:, :2].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(hourly.dynamic[:, 2].mean(axis=0), 5, atol=1e-6)
    summary = aggregate_daily_weather(hourly)
    for name in hourly.metadata["daily_constraints"]:
        index = daily.channel_names.index(name)
        np.testing.assert_allclose(summary.dynamic[:, index], values[:, index], rtol=2e-6, atol=2e-5)


def test_clear_cloud_and_dimmed_ghi_are_jointly_infeasible():
    clear = np.full((24, 1, 1), 200.0)
    with pytest.raises(ValueError, match="jointly infeasible"):
        _condition_cloud_radiation(np.zeros_like(clear), np.zeros((1, 1)), np.full((1, 1), 20.), clear, .68)
    with pytest.raises(ValueError, match="jointly infeasible"):
        _condition_cloud_radiation(np.zeros_like(clear), np.zeros((1, 1)), np.ones((1, 1)), clear * 0, .68)


def test_cloud_radiation_joint_projection_conserves_both_feasible_moments():
    clear = np.maximum(np.sin(np.pi * (np.arange(24) - 6) / 12), 0)[:, None, None] * 600
    cloud = np.full_like(clear, .5)
    requested = (clear * (1 - .68 * cloud)).mean(axis=0) * .85
    projected = _condition_cloud_radiation(cloud, np.array([[.5]]), requested, clear, .68)
    np.testing.assert_allclose(projected.mean(axis=0), .5, atol=1e-12)
    np.testing.assert_allclose((clear * (1 - .68 * projected)).mean(axis=0), requested, atol=1e-10)
    assert np.all((projected >= 0) & (projected <= 1))


def test_weighted_allocation_does_not_count_unreachable_capacity():
    weights = np.array([1., 0.])[:, None, None]
    capacity = np.array([1., 100.])[:, None, None]
    with pytest.raises(ValueError, match="hourly bounds"):
        _bounded_weighted_total(weights, np.array([[2.]]), capacity)
    np.testing.assert_allclose(_bounded_weighted_total(weights, np.array([[1.]]), capacity).sum(axis=0), 1)


def test_physical_units_scale_displacement_and_memory_not_grid_count():
    np.testing.assert_allclose(displacement_cells(4, 2, .5, 2), [-.5, 1])
    np.testing.assert_allclose(displacement_cells(4, 2, 1, 1), [-2, 4])
    assert temporal_retention(.5, 8) ** 2 == pytest.approx(temporal_retention(1, 8))
    # Equal L/dx gives the same sample; map resolution is explicit, not steps.
    first = gaussian_field((5, 6), np.random.default_rng(9), 10, 2)
    second = gaussian_field((5, 6), np.random.default_rng(9), 20, 4)
    np.testing.assert_array_equal(first, second)


def test_open_inflow_never_wraps_or_copies_the_outflow_edge():
    values = np.zeros((3, 4)); values[:, -1] = 1e6
    config = WeatherConfig(spatial_boundary="open", spatial_scale_mode="legacy", synoptic_shift_cells_per_day=1, temporal_scale_mode="legacy", advection_rho=.999)
    result = _advect_and_innovate(values, 1, 0, config, np.random.default_rng(5))
    assert np.abs(result[:, 0]).max() < 10
    assert np.abs(result[:, 1:]).max() < 10


@pytest.mark.parametrize("override", [
    {"hourly_memory_hours": 0}, {"innovation_correlation_length_km": -1},
    {"spatial_boundary": "nearest"}, {"temporal_scale_mode": "magic"},
    {"hourly_generation_mode": "lock_everything"}, {"irradiance_cloud_sensitivity": 1.1},
    {"synoptic_advection_speed_km_per_hour": float("nan")}, {"days": 2.5},
])
def test_invalid_physical_scale_configuration_fails(override):
    with pytest.raises(ValueError):
        WeatherConfig(**override)


@pytest.mark.parametrize("probability", [0., 1.])
def test_all_dry_and_all_wet_probability_endpoints(probability):
    terrain, water, climate, grid = _fixture((2, 2))
    config = WeatherConfig(days=5, wet_day_probability=probability)
    daily = generate_daily_weather(terrain, water, climate, grid, config, np.random.default_rng(42))
    np.testing.assert_array_equal(daily.dynamic[:, 7] > 0, probability == 1)
    assert np.isfinite(daily.dynamic).all()


def test_weather_config_keeps_legacy_positional_field_order():
    config = WeatherConfig(12, .5, .2)
    assert config.days == 12 and config.wet_day_probability == .5 and config.wet_day_persistence == .2


def test_daily_summary_rejects_half_hour_offset_and_reproducibility_uses_seed():
    terrain, water, climate, grid = _fixture((2, 2))
    config = WeatherConfig(days=1, hourly_week_days=1)
    first = generate_daily_weather(terrain, water, climate, grid, config, np.random.default_rng(42))
    second = generate_daily_weather(terrain, water, climate, grid, config, np.random.default_rng(42))
    np.testing.assert_array_equal(first.dynamic, second.dynamic)
    one = generate_hourly_weather_week(first, config, np.random.default_rng(123), grid=grid)
    two = generate_hourly_weather_week(second, config, np.random.default_rng(123), grid=grid)
    np.testing.assert_array_equal(one.dynamic, two.dynamic)
    for name in one.diagnostics:
        np.testing.assert_array_equal(one.diagnostics[name], two.diagnostics[name])
    with pytest.raises(ValueError, match="midnight-aligned"):
        aggregate_daily_weather(replace(one, timestamps=one.timestamps.astype(float) + .5))


def test_weather_cache_and_temporal_diagnostics_roundtrip_inside_workspace():
    terrain, water, climate, grid = _fixture((2, 3))
    config = WeatherConfig(days=1, hourly_week_days=1)
    daily = generate_daily_weather(terrain, water, climate, grid, config, np.random.default_rng(42))
    hourly = generate_hourly_weather_week(daily, config, np.random.default_rng(123), grid=grid)
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory); (root / "data").mkdir()
        np.savez_compressed(root / "data" / "hourly_weather_week.npz", **hourly.as_arrays())
        loaded = load_hourly_weather_checkpoint(root)
        assert loaded.metadata == hourly.metadata
        for name in hourly.diagnostics:
            np.testing.assert_array_equal(loaded.diagnostics[name], hourly.diagnostics[name])
        np.testing.assert_array_equal(loaded.static_elevation_m, hourly.static_elevation_m)
    dynamic = hourly.as_arrays(); dynamic["weather"] = dynamic.pop("dynamic")
    window = _slice_weather(dynamic, 3, 7, 24)
    np.testing.assert_array_equal(window["diagnostic__specific_humidity_kg_kg"], hourly.diagnostics["specific_humidity_kg_kg"][3:7])
    assert window["time_bounds_hours"].shape == (4, 2)
    assert "static_elevation_m" not in window
