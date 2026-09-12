from dataclasses import replace

import numpy as np
import pytest

from world_generator.climate.climate_generator import _wind_exposure, generate_climate_baseline
from world_generator.core.config import ClimateConfig, WeatherConfig, WorldGridConfig
from world_generator.core.datatypes import ClimateBaseline, HydrologyState, TerrainFeatures
from world_generator.weather.physics import extraterrestrial_hourly_irradiance, latitude_grid, saturation_vapor_pressure_hpa, solar_direction, surface_pressure_hpa
from world_generator.weather.weather_generator import _gaussian_field, aggregate_daily_weather, generate_daily_weather, generate_hourly_weather_week


def _fixture(shape=(6, 7)):
    zeros = np.zeros(shape, dtype=np.float32)
    terrain = TerrainFeatures(zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())
    hydrology = HydrologyState(zeros.astype(int), zeros.copy(), zeros.astype(bool), zeros.astype(bool), zeros.astype(bool), zeros.copy(), zeros.copy(), zeros.astype(int), zeros + 10.0, zeros.copy())
    climate = ClimateBaseline(zeros + 18.0, zeros + 10.0, zeros + 0.60, zeros + 4.0, zeros + 3.0, zeros + 800.0, zeros + 0.45, zeros + 180.0)
    return terrain, hydrology, climate, WorldGridConfig(height=shape[0], width=shape[1])


def test_solar_geometry_daylength_hemisphere_and_hour_integrals():
    north = extraterrestrial_hourly_irradiance(np.array(50.0), 172)
    winter = extraterrestrial_hourly_irradiance(np.array(50.0), 355)
    south = extraterrestrial_hourly_irradiance(np.array(-50.0), 172)
    assert np.count_nonzero(north) > np.count_nonzero(winter)
    assert north.mean() > 2.0 * winter.mean()
    assert south.mean() < north.mean()
    assert np.all(extraterrestrial_hourly_irradiance(np.array(89.0), 355) == 0)
    assert np.all(extraterrestrial_hourly_irradiance(np.array(89.0), 172) > 0)
    # Independent high-resolution integration checks hourly sunrise bin handling.
    times = (np.arange(24 * 600) + 0.5) / 600.0
    cosine = np.asarray([solar_direction(np.array(50.0), 172, hour)[2] for hour in times])
    distance = 1.0 + 0.033 * np.cos(2.0 * np.pi * 173.0 / 365.0)
    numerical = np.maximum(cosine, 0).reshape(24, 600).mean(axis=1) * 1366.6666666667 * distance
    np.testing.assert_allclose(north, numerical, atol=2e-4)


def test_pressure_decreases_hydrostatically_and_rh_uses_saturation():
    height = np.array([0.0, 1800.0, 3000.0, 5000.0])
    pressure = surface_pressure_hpa(height, 15.0 - 0.0065 * height, np.zeros(4), 1013.25)
    assert np.all(np.diff(pressure) < 0)
    # Standard-atmosphere benchmarks, with layer-mean approximation tolerance.
    np.testing.assert_allclose(pressure[[0, 2, 3]], [1013.25, 701.1, 540.2], rtol=0.003)
    np.testing.assert_allclose(saturation_vapor_pressure_hpa(np.array([0.0, 20.0])), [6.108, 23.38], rtol=0.001)


def test_hourly_conserves_declared_daily_constraints_and_diagnoses_daily_means():
    terrain, hydrology, climate, grid = _fixture()
    config = WeatherConfig(days=12, hourly_week_days=7, start_day_of_year=358)
    daily = generate_daily_weather(terrain, hydrology, climate, grid, config, np.random.default_rng(10))
    hourly = generate_hourly_weather_week(daily, config, np.random.default_rng(21), grid=grid)
    indices = {name: i for i, name in enumerate(daily.channel_names)}
    summary = aggregate_daily_weather(hourly)
    start = int(hourly.timestamps[0] // 24 - daily.timestamps[0])
    for day in range(config.hourly_week_days):
        values = hourly.dynamic[day * 24:(day + 1) * 24].astype(float)
        for name, index in indices.items():
            aggregate = values[:, index].sum(axis=0) if name == "precipitation" else values[:, index].mean(axis=0)
            np.testing.assert_allclose(aggregate, summary.dynamic[day, index], rtol=2e-6, atol=2e-5)
            if name in hourly.metadata["daily_constraints"]:
                np.testing.assert_allclose(aggregate, daily.dynamic[start + day, index], rtol=2e-6, atol=2e-5)
        toa = extraterrestrial_hourly_irradiance(latitude_grid(grid, terrain.elevation.shape), hourly.timestamps[day * 24] // 24)
        assert np.all(values[:, indices["irradiance"]][toa == 0] == 0)
        assert np.all(values[:, indices["irradiance"]] <= toa + 1e-4)
    np.testing.assert_allclose(np.hypot(hourly.dynamic[:, 0], hourly.dynamic[:, 1]), hourly.dynamic[:, 2], rtol=2e-6)
    assert np.all((hourly.dynamic[:, indices["humidity"]] >= 0) & (hourly.dynamic[:, indices["humidity"]] <= 1))
    assert np.isfinite(hourly.dynamic).all()
    # Low-variance temperature/humidity relationship should remain inverse.
    assert np.corrcoef(hourly.dynamic[:24, 3, 2, 2], hourly.dynamic[:24, 4, 2, 2])[0, 1] < -0.95


def test_rainfall_has_stationary_markov_occurrence_and_expected_annual_budget():
    terrain, hydrology, climate, grid = _fixture((8, 8))
    config = WeatherConfig(days=365 * 3, innovation_smoothing_steps=0)
    weather = generate_daily_weather(terrain, hydrology, climate, grid, config, np.random.default_rng(91))
    rain = weather.dynamic[:, 7]
    wet = rain > 0
    assert abs(wet.mean() - config.wet_day_probability) < 0.025
    observed_ww = wet[1:][wet[:-1]].mean()
    expected_ww = config.wet_day_probability + config.wet_day_persistence * (1 - config.wet_day_probability)
    assert abs(observed_ww - expected_ww) < 0.025
    assert abs(float(rain.sum(axis=0).mean() / 3.0) / 800.0 - 1) < 0.06
    assert (rain == 0).mean() > 0.6
    assert np.percentile(rain[wet], 95) > 2.0 * rain[wet].mean()


def test_noise_keeps_gaussian_variance_at_edges_and_spatial_correlation():
    rng = np.random.default_rng(71)
    samples = np.asarray([_gaussian_field((8, 8), rng, 3) for _ in range(1200)])
    assert np.max(np.abs(samples.mean(axis=0))) < 0.1
    assert np.max(np.abs(samples.var(axis=0) - 1.0)) < 0.16
    assert np.corrcoef(samples[:, 3, 3], samples[:, 3, 4])[0, 1] > 0.7


def test_climate_wind_convention_and_upslope_sign():
    terrain, hydrology, _, grid = _fixture()
    config = ClimateConfig(prevailing_wind_degrees=270.0, wind_direction_noise_degrees=0.0)
    climate = generate_climate_baseline(terrain, hydrology, grid, config, np.random.default_rng(1))
    assert np.all(climate.prevailing_wind_u > 0)
    np.testing.assert_allclose(climate.prevailing_wind_v, 0, atol=1e-6)
    ramp = replace(terrain, elevation=np.broadcast_to(np.arange(7) * 100.0, (6, 7)).copy())
    lift = _wind_exposure(ramp, np.ones((6, 7)), np.zeros((6, 7)), 2.0)
    assert np.all(lift > 0)
    np.testing.assert_allclose(_wind_exposure(ramp, -np.ones((6, 7)), np.zeros((6, 7)), 2.0), -lift)
    assert np.all(latitude_grid(grid, (6, 7))[0] > latitude_grid(grid, (6, 7))[-1])


def test_invalid_daily_radiation_and_latitude_are_rejected():
    terrain, hydrology, climate, grid = _fixture()
    config = WeatherConfig(days=1, hourly_week_days=1)
    daily = generate_daily_weather(terrain, hydrology, climate, grid, config, np.random.default_rng(1))
    invalid = daily.dynamic.copy()
    invalid[:, 8] = 2000.0
    with pytest.raises(ValueError, match="physical hourly bounds"):
        generate_hourly_weather_week(replace(daily, dynamic=invalid), config, np.random.default_rng(1), grid=grid)
    with pytest.raises(ValueError, match="beyond a pole"):
        latitude_grid(replace(grid, latitude_center_degrees=90.0), (6, 7))


@pytest.mark.parametrize("channel,value,message", [
    ("precipitation", -1.0, "nonnegative"),
    ("cloud", 1.5, "\\[0, 1\\]"),
    ("humidity", -0.1, "\\[0, 1\\]"),
    ("pressure", 0.0, "positive"),
    ("temperature", -300.0, "absolute zero"),
    ("wind_speed", 0.0, "wind_speed must be >= hypot"),
    ("wind_speed", -1.0, "nonnegative"),
    ("irradiance", -1.0, "nonnegative"),
    ("temperature", np.nan, "finite"),
])
def test_disaggregation_rejects_invalid_external_daily_physics(channel, value, message):
    terrain, hydrology, climate, grid = _fixture()
    config = WeatherConfig(days=1, hourly_week_days=1)
    daily = generate_daily_weather(terrain, hydrology, climate, grid, config, np.random.default_rng(1))
    invalid = daily.dynamic.copy()
    invalid[:, daily.channel_names.index(channel), 0, 0] = value
    with pytest.raises(ValueError, match=message):
        generate_hourly_weather_week(replace(daily, dynamic=invalid), config, np.random.default_rng(1), grid=grid)
