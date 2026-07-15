from __future__ import annotations

import numpy as np

from world_generator.core.config import ClimateConfig, WorldGridConfig
from world_generator.core.coordinates import normalized_grid
from world_generator.core.datatypes import ClimateBaseline, HydrologyState, TerrainFeatures


def generate_climate_baseline(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    grid: WorldGridConfig,
    config: ClimateConfig,
    rng: np.random.Generator,
) -> ClimateBaseline:
    height, width = terrain.elevation.shape
    x, y = normalized_grid(height, width)
    elevation_km = terrain.elevation / 1000.0
    elevation_n = _normalize01(terrain.elevation)
    slope_n = _normalize01(terrain.slope)
    water_influence = np.exp(-hydrology.distance_to_water / max(config.water_moderation_km, 1e-6))
    broad_noise = _smooth_noise((height, width), rng, steps=5)

    latitude_temperature = config.base_temperature_c - config.latitude_temperature_gradient_c * y
    terrain_cooling = config.lapse_rate_c_per_km * elevation_km
    water_temperature_bonus = 1.8 * water_influence
    mean_temperature = latitude_temperature - terrain_cooling + water_temperature_bonus
    mean_temperature += config.climate_noise_weight * 2.0 * (broad_noise - 0.5)

    annual_amplitude = (
        config.annual_amplitude_c
        + 3.0 * np.abs(y - 0.5)
        + 1.5 * elevation_n
        - config.coastal_amplitude_reduction_c * water_influence
    )
    annual_amplitude = np.clip(annual_amplitude, 3.0, None)

    wind_rad = np.deg2rad(config.prevailing_wind_degrees)
    base_u = np.cos(wind_rad)
    base_v = -np.sin(wind_rad)
    wind_dir_u, wind_dir_v = _terrain_steered_wind_direction(
        terrain,
        base_u,
        base_v,
        config,
        rng,
    )
    wind_exposure = _wind_exposure(terrain, wind_dir_u, wind_dir_v)
    wind_speed = config.prevailing_wind_speed_mps * (
        1.0
        + config.wind_speed_terrain_factor * (0.65 * slope_n + 0.35 * elevation_n)
        + 0.10 * (broad_noise - 0.5)
    )
    wind_speed = np.clip(wind_speed, 0.5, None)
    prevailing_wind_u = wind_speed * wind_dir_u
    prevailing_wind_v = wind_speed * wind_dir_v

    water_humidity = np.exp(-hydrology.distance_to_water / max(config.humidity_water_decay_km, 1e-6))
    humidity = (
        0.34
        + 0.34 * water_humidity
        + 0.14 * (1.0 - elevation_n)
        + 0.10 * wind_exposure
        + 0.08 * broad_noise
    )
    mean_humidity = np.clip(humidity, 0.05, 0.98)

    precipitation = config.precipitation_base_mm_year * (
        0.70
        + 0.42 * mean_humidity
        + config.orographic_precipitation_factor * wind_exposure * (0.35 + slope_n)
        - config.rain_shadow_factor * (1.0 - wind_exposure) * slope_n
    )
    precipitation = np.clip(precipitation, 80.0, None)

    mean_cloud = np.clip(
        0.18 + 0.48 * mean_humidity + 0.22 * _normalize01(precipitation) + 0.06 * broad_noise,
        0.05,
        0.92,
    )
    latitude_solar = 1.0 - 0.22 * y
    mean_irradiance = config.irradiance_base_w_m2 * latitude_solar * (1.0 - 0.52 * mean_cloud)
    mean_irradiance += 18.0 * elevation_n
    mean_irradiance = np.clip(mean_irradiance, 35.0, None)

    return ClimateBaseline(
        mean_temperature=mean_temperature.astype(np.float32),
        annual_temperature_amplitude=annual_amplitude.astype(np.float32),
        mean_humidity=mean_humidity.astype(np.float32),
        prevailing_wind_u=prevailing_wind_u.astype(np.float32),
        prevailing_wind_v=prevailing_wind_v.astype(np.float32),
        mean_precipitation=precipitation.astype(np.float32),
        mean_cloud=mean_cloud.astype(np.float32),
        mean_irradiance=mean_irradiance.astype(np.float32),
    )


def _terrain_steered_wind_direction(
    terrain: TerrainFeatures,
    base_u: float,
    base_v: float,
    config: ClimateConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(terrain.elevation.astype(np.float32))
    grad_mag = np.hypot(grad_x, grad_y)
    safe_mag = np.maximum(grad_mag, 1e-6)
    tangent_u = -grad_y / safe_mag
    tangent_v = grad_x / safe_mag
    tangent_alignment = base_u * tangent_u + base_v * tangent_v
    channel_u = np.where(tangent_alignment >= 0.0, tangent_u, -tangent_u)
    channel_v = np.where(tangent_alignment >= 0.0, tangent_v, -tangent_v)

    angle_noise = _smooth_noise(
        terrain.elevation.shape,
        rng,
        steps=config.wind_direction_smoothing_steps,
    )
    angle = np.deg2rad(config.wind_direction_noise_degrees) * (2.0 * angle_noise - 1.0)
    noisy_u = base_u * np.cos(angle) - base_v * np.sin(angle)
    noisy_v = base_u * np.sin(angle) + base_v * np.cos(angle)

    slope_influence = _normalize01(grad_mag)
    steering = np.clip(
        config.wind_direction_terrain_steering * (0.25 + 0.75 * slope_influence),
        0.0,
        0.98,
    )
    wind_u = (1.0 - steering) * noisy_u + steering * channel_u
    wind_v = (1.0 - steering) * noisy_v + steering * channel_v
    norm = np.maximum(np.hypot(wind_u, wind_v), 1e-6)
    return (wind_u / norm).astype(np.float32), (wind_v / norm).astype(np.float32)


def _wind_exposure(terrain: TerrainFeatures, wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(terrain.elevation.astype(np.float32))
    upslope_into_wind = -(grad_x * wind_u + grad_y * wind_v)
    return _normalize01(upslope_into_wind)


def _smooth_noise(shape: tuple[int, int], rng: np.random.Generator, steps: int) -> np.ndarray:
    noise = rng.random(shape, dtype=np.float32)
    for _ in range(max(steps, 0)):
        padded = np.pad(noise, 1, mode="edge")
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
