from __future__ import annotations

import numpy as np

from world_generator.core.config import WeatherConfig, WorldGridConfig
from world_generator.core.datatypes import ClimateBaseline, HydrologyState, TerrainFeatures, WeatherStore


WEATHER_CHANNELS = (
    "wind_u",
    "wind_v",
    "wind_speed",
    "temperature",
    "humidity",
    "pressure",
    "cloud",
    "precipitation",
    "irradiance",
)


def generate_daily_weather(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    climate: ClimateBaseline,
    grid: WorldGridConfig,
    config: WeatherConfig,
    rng: np.random.Generator,
) -> WeatherStore:
    del grid
    days = int(config.days)
    shape = terrain.elevation.shape
    dynamic = np.zeros((days, len(WEATHER_CHANNELS), *shape), dtype=np.float32)
    weather_class = np.zeros((days, *shape), dtype=np.int16)
    timestamps = np.arange(days, dtype=np.int32) + int(config.start_day_of_year)

    moisture = _smooth_noise(shape, rng, config.innovation_smoothing_steps)
    storm = _smooth_noise(shape, rng, config.innovation_smoothing_steps)
    temperature_wave = _smooth_noise(shape, rng, config.innovation_smoothing_steps)
    elevation_n = _normalize01(terrain.elevation)
    roughness_n = _normalize01(terrain.roughness)
    wind_base_speed = np.hypot(climate.prevailing_wind_u, climate.prevailing_wind_v)
    wind_base_speed = np.maximum(wind_base_speed, 0.2)
    wind_dir_u = climate.prevailing_wind_u / wind_base_speed
    wind_dir_v = climate.prevailing_wind_v / wind_base_speed
    water_wetness = np.exp(-hydrology.distance_to_water / 8.0)
    oro = _orographic_lift(terrain, wind_dir_u, wind_dir_v)
    low_frequency_anomaly = _seasonal_low_frequency_anomaly(days, rng)

    for t in range(days):
        day_of_year = (config.start_day_of_year + t) % 365
        mean_u = float(np.nanmean(wind_dir_u))
        mean_v = float(np.nanmean(wind_dir_v))
        moisture = _advect_and_innovate(moisture, mean_u, mean_v, config, rng)
        storm = _advect_and_innovate(storm, mean_u, mean_v, config, rng)
        temperature_wave = _advect_and_innovate(temperature_wave, mean_u, mean_v, config, rng)

        temperature_season = _seasonal_curve(
            day_of_year,
            peak_day=172.0 + config.temperature_seasonal_lag_days,
            config=config,
        )
        solar_curve = _seasonal_curve(
            day_of_year,
            peak_day=172.0 + config.solar_seasonal_lag_days,
            config=config,
        )
        wet_season = 0.5 + 0.5 * _seasonal_curve(day_of_year, peak_day=112.0, config=config)
        solar_season = 0.88 + 0.20 * (0.5 + 0.5 * solar_curve)
        low_frequency = low_frequency_anomaly[t]

        temperature = (
            climate.mean_temperature
            + climate.annual_temperature_amplitude * temperature_season
            + config.temperature_synoptic_c * (temperature_wave - 0.5)
            + config.low_frequency_temperature_c * low_frequency
            + 0.9 * water_wetness * (0.15 - temperature_season)
        )

        humidity = np.clip(
            climate.mean_humidity
            + config.humidity_synoptic_weight * (moisture - 0.5)
            + 0.09 * wet_season
            + 0.04 * water_wetness
            - 0.05 * elevation_n,
            0.03,
            0.99,
        )

        storm_intensity = np.clip(0.58 * storm + 0.32 * moisture + 0.10 * wet_season, 0.0, 1.0)
        cloud = np.clip(
            0.52 * climate.mean_cloud
            + config.cloud_synoptic_weight * storm_intensity
            + 0.24 * humidity
            + 0.08 * oro,
            0.02,
            0.98,
        )

        wind_factor = 1.0 + config.wind_synoptic_weight * (storm - 0.5) + 0.12 * roughness_n + 0.08 * oro
        wind_speed = np.clip(wind_base_speed * wind_factor, 0.1, None)
        wind_u = wind_dir_u * wind_speed
        wind_v = wind_dir_v * wind_speed

        precipitation = (climate.mean_precipitation / 365.0) * (
            0.22
            + config.precipitation_event_scale * np.maximum(storm_intensity - 0.48, 0.0)
            + 0.75 * oro
            + 0.25 * water_wetness
        )
        precipitation = np.clip(precipitation, 0.0, None)

        pressure = (
            config.pressure_base_hpa
            - 0.012 * terrain.elevation
            - config.pressure_synoptic_hpa * (storm - 0.5)
            + 1.8 * (0.5 - moisture)
        )

        rain_reduction = np.clip(precipitation / 18.0, 0.0, 1.0)
        irradiance = climate.mean_irradiance * solar_season * (1.0 + config.low_frequency_irradiance_weight * low_frequency) * (
            1.0 - config.irradiance_cloud_sensitivity * cloud
        )
        irradiance *= 1.0 - 0.22 * rain_reduction
        irradiance += 7.0 * elevation_n
        irradiance = np.clip(irradiance, 0.0, None)

        dynamic[t, 0] = wind_u.astype(np.float32)
        dynamic[t, 1] = wind_v.astype(np.float32)
        dynamic[t, 2] = wind_speed.astype(np.float32)
        dynamic[t, 3] = temperature.astype(np.float32)
        dynamic[t, 4] = humidity.astype(np.float32)
        dynamic[t, 5] = pressure.astype(np.float32)
        dynamic[t, 6] = cloud.astype(np.float32)
        dynamic[t, 7] = precipitation.astype(np.float32)
        dynamic[t, 8] = irradiance.astype(np.float32)
        weather_class[t] = _classify_weather(cloud, precipitation, wind_speed)

    return WeatherStore(
        dynamic=dynamic,
        weather_class=weather_class,
        timestamps=timestamps,
        channel_names=WEATHER_CHANNELS,
    )


def _advect_and_innovate(
    values: np.ndarray,
    mean_u: float,
    mean_v: float,
    config: WeatherConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    shift_col = int(np.rint(mean_u * config.synoptic_shift_cells_per_day))
    shift_row = int(np.rint(-mean_v * config.synoptic_shift_cells_per_day))
    advected = np.roll(values, shift=(shift_row, shift_col), axis=(0, 1))
    innovation = _smooth_noise(values.shape, rng, config.innovation_smoothing_steps)
    rho = np.clip(config.advection_rho, 0.0, 0.999)
    mixed = rho * advected + np.sqrt(1.0 - rho * rho) * innovation
    return _normalize01(mixed)


def _seasonal_curve(day_of_year: float, peak_day: float, config: WeatherConfig) -> float:
    angle = 2.0 * np.pi * (day_of_year - peak_day) / 365.0
    value = np.cos(angle)
    value += config.seasonal_second_harmonic * np.sin(2.0 * angle - 0.55)
    value += config.seasonal_third_harmonic * np.cos(3.0 * angle + 0.8)
    return float(np.clip(value / (1.0 + config.seasonal_second_harmonic + config.seasonal_third_harmonic), -1.0, 1.0))


def _seasonal_low_frequency_anomaly(days: int, rng: np.random.Generator) -> np.ndarray:
    anchors = max(8, int(np.ceil(days / 42)))
    values = rng.normal(0.0, 1.0, anchors).astype(np.float32)
    values = np.r_[values[-2:], values, values[:2]]
    for _ in range(3):
        values = (
            np.roll(values, 1)
            + 2.0 * values
            + np.roll(values, -1)
        ) / 4.0
    anchor_x = np.linspace(-2.0, days + 1.0, values.size, dtype=np.float32)
    day_x = np.arange(days, dtype=np.float32)
    anomaly = np.interp(day_x, anchor_x, values).astype(np.float32)
    anomaly -= float(np.mean(anomaly))
    std = float(np.std(anomaly))
    if std > 1e-6:
        anomaly /= std
    return np.clip(anomaly, -1.8, 1.8).astype(np.float32)


def _orographic_lift(terrain: TerrainFeatures, wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(terrain.elevation.astype(np.float32))
    lift = wind_u * grad_x + wind_v * grad_y
    return _normalize01(np.maximum(lift, 0.0))


def _classify_weather(cloud: np.ndarray, precipitation: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    weather = np.zeros(cloud.shape, dtype=np.int16)
    weather[cloud > 0.55] = 1
    weather[precipitation > 1.0] = 2
    weather[precipitation > 6.0] = 3
    weather[wind_speed > 9.0] = np.maximum(weather[wind_speed > 9.0], 4)
    return weather


def _smooth_noise(shape: tuple[int, int], rng: np.random.Generator, steps: int) -> np.ndarray:
    noise = rng.random(shape, dtype=np.float32)
    for _ in range(max(steps, 0)):
        padded = np.pad(noise, 1, mode="wrap")
        noise = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + 2.0 * padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 10.0
    return _normalize01(noise)


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)
