"""Conditional stochastic weather with conservative hourly disaggregation.

This is a scenario generator, not numerical weather prediction. Rain occurrence
precedes other variables. Empirical assumptions: docs/research_weather.md.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.ndimage import convolve1d, shift
from scipy.special import ndtr
from scipy.stats import gamma

from world_generator.core.config import WeatherConfig, WorldGridConfig
from world_generator.core.datatypes import ClimateBaseline, HydrologyState, TerrainFeatures, WeatherStore
from world_generator.weather.physics import clear_sky_transmissivity, extraterrestrial_hourly_irradiance, latitude_grid, saturation_vapor_pressure_hpa, surface_pressure_hpa


WEATHER_CHANNELS = (
    "wind_u", "wind_v", "wind_speed", "temperature", "humidity",
    "pressure", "cloud", "precipitation", "irradiance",
)


def generate_daily_weather(terrain: TerrainFeatures, hydrology: HydrologyState, climate: ClimateBaseline, grid: WorldGridConfig, config: WeatherConfig, rng: np.random.Generator) -> WeatherStore:
    """Daily means; precipitation is daily total water equivalent in mm."""
    del hydrology  # Its influence is already in the climate baseline.
    days = int(config.days)
    if days < 1:
        raise ValueError("Weather days must be positive")
    wet_probability = float(config.wet_day_probability)
    persistence = float(config.wet_day_persistence)
    gamma_shape = float(config.precipitation_gamma_shape)
    if not 0.0 < wet_probability < 1.0:
        raise ValueError("wet_day_probability must be in (0, 1)")
    if not 0.0 <= persistence < 1.0 or gamma_shape <= 0.0:
        raise ValueError("wet_day_persistence must be in [0, 1), gamma shape must be positive")
    shape = terrain.elevation.shape
    latitude = latitude_grid(grid, shape)
    hemisphere = np.where(latitude >= 0.0, 1.0, -1.0)
    clear_transmission = clear_sky_transmissivity(terrain.elevation)
    dynamic = np.zeros((days, len(WEATHER_CHANNELS), *shape), dtype=np.float32)
    weather_class = np.zeros((days, *shape), dtype=np.int16)
    timestamps = np.arange(days, dtype=np.int32) + int(config.start_day_of_year)
    storm = _gaussian_field(shape, rng, config.innovation_smoothing_steps)
    thermal = _gaussian_field(shape, rng, config.innovation_smoothing_steps)
    moisture = _gaussian_field(shape, rng, config.innovation_smoothing_steps)
    wet = ndtr(_gaussian_field(shape, rng, config.innovation_smoothing_steps)) < wet_probability
    wind_base_speed = np.hypot(climate.prevailing_wind_u, climate.prevailing_wind_v)
    safe_speed = np.maximum(wind_base_speed, 1e-9)
    mean_u = float(np.mean(climate.prevailing_wind_u / safe_speed))
    mean_v = float(np.mean(climate.prevailing_wind_v / safe_speed))
    # Constant occurrence probability has a known stationary distribution.
    # Seasonality in conditional amounts preserves expected annual mm.
    p_dry_to_wet = wet_probability * (1.0 - persistence)
    p_wet_to_wet = wet_probability + persistence * (1.0 - wet_probability)
    for t, timestamp in enumerate(timestamps):
        doy = int(timestamp) % 365
        storm = _advect_and_innovate(storm, mean_u, mean_v, config, rng)
        thermal = _advect_and_innovate(thermal, mean_u, mean_v, config, rng)
        moisture = _advect_and_innovate(moisture, mean_u, mean_v, config, rng)
        occurrence = ndtr(_gaussian_field(shape, rng, config.innovation_smoothing_steps))
        wet = occurrence < np.where(wet, p_wet_to_wet, p_dry_to_wet)
        # Gaussian copula yields correlated gamma marginals. No realization
        # min/max normalization and no artificial fixed annual rainfall sum.
        amount_uniform = np.clip(ndtr(_gaussian_field(shape, rng, config.innovation_smoothing_steps)), 1e-10, 1.0 - 1e-10)
        amount_multiplier = gamma.ppf(amount_uniform, a=gamma_shape) / gamma_shape
        wet_season = 1.0 + 0.35 * hemisphere * np.cos(2.0 * np.pi * (doy - 112.0) / 365.0)
        precipitation = wet * amount_multiplier * climate.mean_precipitation * wet_season / (365.0 * wet_probability)
        seasonal_temperature = climate.mean_temperature + climate.annual_temperature_amplitude * hemisphere * _seasonal_curve(doy, 172.0 + config.temperature_seasonal_lag_days, config)
        wet_anomaly = wet.astype(float) - wet_probability
        temperature = seasonal_temperature + config.temperature_synoptic_c * (0.8 * thermal - 0.3 * storm) - 1.5 * wet_anomaly
        # Vapor pressure first, then RH via temperature-dependent saturation.
        actual_vapor = climate.mean_humidity * saturation_vapor_pressure_hpa(seasonal_temperature)
        actual_vapor *= np.exp(config.humidity_synoptic_weight * (0.7 * moisture + 0.3 * storm) + 0.16 * wet_anomaly)
        humidity = np.clip(actual_vapor / saturation_vapor_pressure_hpa(temperature), 0.01, 1.0)
        cloud = np.clip(climate.mean_cloud + config.cloud_synoptic_weight * (0.55 * storm + 0.25 * moisture) + 0.38 * wet_anomaly, 0.0, 1.0)
        wind_sigma = max(float(config.wind_synoptic_weight), 0.0)
        wind_factor = np.exp(wind_sigma * storm - 0.5 * wind_sigma**2)
        wind_speed = wind_base_speed * wind_factor
        wind_u = climate.prevailing_wind_u * wind_factor
        wind_v = climate.prevailing_wind_v * wind_factor
        sea_level_pressure = np.clip(config.pressure_base_hpa - config.pressure_synoptic_hpa * (0.7 * storm + 0.3 * wet_anomaly), 850.0, 1080.0)
        pressure = surface_pressure_hpa(terrain.elevation, temperature, humidity, sea_level_pressure)
        toa_daily = extraterrestrial_hourly_irradiance(latitude, doy).mean(axis=0)
        cloud_transmission = np.clip(1.0 - config.irradiance_cloud_sensitivity * cloud, 0.0, 1.0)
        irradiance = toa_daily * clear_transmission * cloud_transmission
        for index, values in enumerate((wind_u, wind_v, wind_speed, temperature, humidity, pressure, cloud, precipitation, irradiance)):
            dynamic[t, index] = values
        weather_class[t] = _classify_weather(cloud, precipitation, wind_speed)
    return WeatherStore(dynamic, weather_class, timestamps, WEATHER_CHANNELS, "day", int(config.start_day_of_year))


def generate_hourly_weather_week_from_baseline(terrain: TerrainFeatures, hydrology: HydrologyState, climate: ClimateBaseline, grid: WorldGridConfig, config: WeatherConfig, rng: np.random.Generator) -> WeatherStore:
    """Compatibility entry point using the same daily→hourly construction.

    For paired datasets use generate_hourly_weather_week on the stored daily
    series. This wrapper generates only the configured consecutive days.
    """
    local_config = replace(config, days=max(1, int(config.hourly_week_days)))
    daily = generate_daily_weather(terrain, hydrology, climate, grid, local_config, rng)
    return generate_hourly_weather_week(daily, local_config, rng, grid=grid)


def generate_hourly_weather_week(daily_weather: WeatherStore, config: WeatherConfig, rng: np.random.Generator, *, grid: WorldGridConfig | None = None) -> WeatherStore:
    """Conserve each cell's supplied daily aggregates in consecutive days.

    GHI is an hourly mean W/m²; rain is mm accumulated in the hour. Temperature,
    RH, pressure, cloud and wind retain daily means. Wind direction stays fixed
    within a day so both vector and speed means can be conserved. Local solar
    hours describe intervals [h,h+1), not instantaneous noon samples.
    """
    if daily_weather.time_unit != "day" or daily_weather.dynamic.ndim != 4 or daily_weather.dynamic.shape[0] < 1:
        raise ValueError("Hourly disaggregation requires a nonempty daily WeatherStore")
    if not np.all(np.isfinite(daily_weather.dynamic)):
        raise ValueError("Daily weather must contain finite physical values")
    if daily_weather.timestamps.shape != (daily_weather.dynamic.shape[0],) or not np.all(np.isfinite(daily_weather.timestamps)):
        raise ValueError("Daily weather timestamps must match the number of days and be finite")
    if not np.all(daily_weather.timestamps == np.rint(daily_weather.timestamps)):
        raise ValueError("Daily weather timestamps must be integer day indices")
    if not np.all(np.diff(daily_weather.timestamps) == 1):
        raise ValueError("Daily weather timestamps must be consecutive")
    week_days = min(max(int(config.hourly_week_days), 1), daily_weather.dynamic.shape[0])
    start_index = int(rng.integers(0, daily_weather.dynamic.shape[0] - week_days + 1))
    shape = daily_weather.dynamic.shape[2:]
    if grid is None:
        grid = WorldGridConfig(height=shape[0], width=shape[1])
    latitude = latitude_grid(grid, shape)
    channels = {name: index for index, name in enumerate(daily_weather.channel_names)}
    if len(channels) != len(daily_weather.channel_names) or len(channels) != daily_weather.dynamic.shape[1]:
        raise ValueError("Daily weather channel names must be unique and match the channel dimension")
    if not set(WEATHER_CHANNELS).issubset(channels):
        raise ValueError("Daily weather is missing required channels")
    for name in ("precipitation", "irradiance", "wind_speed"):
        if np.any(daily_weather.dynamic[:, channels[name]] < 0):
            raise ValueError(f"Daily {name} must be nonnegative")
    for name in ("humidity", "cloud"):
        values = daily_weather.dynamic[:, channels[name]]
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"Daily {name} must lie in [0, 1]")
    if np.any(daily_weather.dynamic[:, channels["pressure"]] <= 0):
        raise ValueError("Daily pressure must be positive")
    if np.any(daily_weather.dynamic[:, channels["temperature"]] <= -273.15):
        raise ValueError("Daily temperature must exceed absolute zero")
    vector_speed = np.hypot(daily_weather.dynamic[:, channels["wind_u"]], daily_weather.dynamic[:, channels["wind_v"]])
    if not np.allclose(vector_speed, daily_weather.dynamic[:, channels["wind_speed"]], rtol=1e-5, atol=1e-6):
        raise ValueError("Fixed-direction disaggregation requires daily wind_speed = hypot(wind_u, wind_v)")
    dynamic = np.zeros((week_days * 24, len(channels), *shape), dtype=np.float32)
    weather_class = np.zeros((week_days * 24, *shape), dtype=np.int16)
    start_day = int(daily_weather.timestamps[start_index])
    timestamps = start_day * 24 + np.arange(week_days * 24, dtype=np.int32)
    cloud_state = _gaussian_field(shape, rng, config.innovation_smoothing_steps)
    wind_state = _gaussian_field(shape, rng, config.innovation_smoothing_steps)
    temperature_state = _gaussian_field(shape, rng, config.innovation_smoothing_steps)
    solar_hours = (np.arange(24) + 0.5)[:, None, None]
    for day_index in range(week_days):
        target = daily_weather.dynamic[start_index + day_index].astype(np.float64)
        cloud_noise, wind_noise, temperature_noise = [], [], []
        for _ in range(24):
            cloud_state = 0.88 * cloud_state + np.sqrt(1.0 - 0.88**2) * _gaussian_field(shape, rng, config.innovation_smoothing_steps)
            wind_state = 0.88 * wind_state + np.sqrt(1.0 - 0.88**2) * _gaussian_field(shape, rng, config.innovation_smoothing_steps)
            temperature_state = 0.88 * temperature_state + np.sqrt(1.0 - 0.88**2) * _gaussian_field(shape, rng, config.innovation_smoothing_steps)
            cloud_noise.append(cloud_state)
            wind_noise.append(wind_state)
            temperature_noise.append(temperature_state)
        cloud_noise = np.asarray(cloud_noise)
        cloud = _bounded_mean(target[channels["cloud"]] + config.hourly_cloud_variability * cloud_noise, target[channels["cloud"]])
        wind_factor = np.exp(config.hourly_wind_variability * np.asarray(wind_noise))
        wind_factor /= wind_factor.mean(axis=0)
        wind_u = target[channels["wind_u"]] * wind_factor
        wind_v = target[channels["wind_v"]] * wind_factor
        wind_speed = np.hypot(wind_u, wind_v)
        diurnal_amplitude = config.hourly_temperature_diurnal_c * (1.25 - 0.65 * target[channels["cloud"]])
        diurnal = diurnal_amplitude * np.cos(2.0 * np.pi * (solar_hours - 15.0) / 24.0)
        temperature_delta = diurnal + config.hourly_temperature_noise_c * np.asarray(temperature_noise) - 0.6 * (cloud - target[channels["cloud"]])
        temperature_delta -= temperature_delta.mean(axis=0)
        temperature = target[channels["temperature"]] + temperature_delta
        # Solve a single daily vapor pressure to preserve mean RH, including
        # saturation clipping, while increasing RH in cooler hours.
        inverse_saturation = 1.0 / saturation_vapor_pressure_hpa(temperature)
        humidity = _bounded_weighted_total(inverse_saturation, 24.0 * target[channels["humidity"]], np.ones_like(temperature))
        pressure_noise = -0.3 * config.pressure_synoptic_hpa * cloud_noise
        pressure = target[channels["pressure"]] + pressure_noise - pressure_noise.mean(axis=0)
        # Conditional wet-hour cluster, no extra unbudgeted rain component.
        center = float(rng.uniform(0.0, 24.0))
        duration = float(rng.uniform(1.5, 4.5))
        pulse = np.exp(-0.5 * ((solar_hours - center) / duration)**2)
        pulse = np.where(np.abs(solar_hours - center) <= 2.0 * duration, pulse, 0.0)
        rain_weights = pulse * np.exp(np.clip(config.hourly_precipitation_burstiness * cloud_noise, -8.0, 8.0)) * (0.2 + cloud)
        precipitation = target[channels["precipitation"]] * rain_weights / rain_weights.sum(axis=0)
        toa = extraterrestrial_hourly_irradiance(latitude, start_day + day_index)
        radiation_weights = toa * np.clip(1.0 - config.irradiance_cloud_sensitivity * cloud, 0.02, 1.0)
        irradiance = _bounded_weighted_total(radiation_weights, 24.0 * target[channels["irradiance"]], toa)
        day_slice = slice(day_index * 24, (day_index + 1) * 24)
        dynamic[day_slice] = target[None, ...]
        for name, values in zip(WEATHER_CHANNELS, (wind_u, wind_v, wind_speed, temperature, humidity, pressure, cloud, precipitation, irradiance)):
            dynamic[day_slice, channels[name]] = values
        weather_class[day_slice] = _classify_weather(cloud, precipitation * 24.0, wind_speed)
    return WeatherStore(dynamic, weather_class, timestamps, daily_weather.channel_names, "hour", start_day)


def _bounded_mean(values: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
    lower = -np.max(np.abs(values), axis=0) - 2.0
    upper = -lower
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        mean = np.clip(values + middle, 0.0, 1.0).mean(axis=0)
        lower = np.where(mean < target_mean, middle, lower)
        upper = np.where(mean >= target_mean, middle, upper)
    return np.clip(values + 0.5 * (lower + upper), 0.0, 1.0)


def _bounded_weighted_total(weights: np.ndarray, target_total: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Proportional allocation with a per-hour physical upper bound."""
    if np.any(target_total < -1e-7) or np.any(target_total > capacity.sum(axis=0) + 1e-3):
        raise ValueError("Daily aggregate cannot fit the physical hourly bounds")
    available = (weights > 0.0) & (capacity > 0.0)
    safe_weights = np.where(available, np.maximum(weights, 1e-15), 0.0)
    lower = np.zeros_like(target_total)
    upper = np.max(np.divide(capacity, safe_weights, out=np.zeros_like(capacity), where=available), axis=0)
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        total = np.minimum(middle * safe_weights, capacity).sum(axis=0)
        lower = np.where(total < target_total, middle, lower)
        upper = np.where(total >= target_total, middle, upper)
    return np.minimum(0.5 * (lower + upper) * safe_weights, capacity)


@lru_cache(maxsize=32)
def _noise_kernel_and_scale(shape: tuple[int, int], steps: int) -> tuple[np.ndarray, np.ndarray]:
    # Exact Gaussian marginal variance of separable reflection convolution,
    # including edges and tiny grids. Independent of the realized field.
    sigma = max(float(steps) / 2.0, 0.5)
    radius = int(np.ceil(3.0 * sigma)) if steps > 0 else 0
    coordinates = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinates / sigma)**2)
    kernel /= kernel.sum()
    variances = []
    for size in shape:
        operator = convolve1d(np.eye(size), kernel, axis=0, mode="reflect")
        variances.append(np.sum(operator**2, axis=1))
    scale = np.sqrt(variances[0][:, None] * variances[1][None, :])
    return kernel, scale


def _gaussian_field(shape: tuple[int, int], rng: np.random.Generator, steps: int) -> np.ndarray:
    kernel, scale = _noise_kernel_and_scale(shape, max(int(steps), 0))
    field = rng.standard_normal(shape)
    for axis in (0, 1):
        field = convolve1d(field, kernel, axis=axis, mode="reflect")
    return field / scale


def _advect_and_innovate(values: np.ndarray, mean_u: float, mean_v: float, config: WeatherConfig, rng: np.random.Generator) -> np.ndarray:
    # Integer translation preserves marginal variance; stochastic rounding
    # preserves fractional mean velocity. Edge values do not wrap the world.
    displacement = np.array([-mean_v, mean_u]) * config.synoptic_shift_cells_per_day
    integer_shift = np.floor(displacement) + (rng.random(2) < (displacement - np.floor(displacement)))
    advected = shift(values, integer_shift, order=0, mode="nearest", prefilter=False)
    rho = float(np.clip(config.advection_rho, 0.0, 0.9999))
    return rho * advected + np.sqrt(1.0 - rho**2) * _gaussian_field(values.shape, rng, config.innovation_smoothing_steps)


def _seasonal_curve(day_of_year: float, peak_day: float, config: WeatherConfig) -> float:
    angle = 2.0 * np.pi * (day_of_year - peak_day) / 365.0
    value = np.cos(angle) + config.seasonal_second_harmonic * np.sin(2.0 * angle - 0.55) + config.seasonal_third_harmonic * np.cos(3.0 * angle + 0.8)
    return float(value / (1.0 + abs(config.seasonal_second_harmonic) + abs(config.seasonal_third_harmonic)))


def _classify_weather(cloud: np.ndarray, precipitation: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    """Display classes; hourly rainfall uses its equivalent daily rate."""
    weather = np.zeros(cloud.shape, dtype=np.int16)
    weather[cloud > 0.55] = 1
    weather[precipitation > 1.0] = 2
    weather[precipitation > 6.0] = 3
    weather[wind_speed > 9.0] = 4
    return weather
