from __future__ import annotations

import numpy as np

from world_generator.core.config import ClimateConfig, WorldGridConfig
from world_generator.core.datatypes import ClimateBaseline, HydrologyState, TerrainFeatures
from world_generator.weather.physics import clear_sky_transmissivity, extraterrestrial_hourly_irradiance, latitude_grid


def generate_climate_baseline(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    grid: WorldGridConfig,
    config: ClimateConfig,
    rng: np.random.Generator,
) -> ClimateBaseline:
    height, width = terrain.elevation.shape
    latitude = latitude_grid(grid, (height, width))
    elevation_km = terrain.elevation / 1000.0
    elevation_n = np.clip(terrain.elevation / 4000.0, 0.0, 1.0)
    slope_n = np.clip(terrain.slope / 0.35, 0.0, 1.0)
    water_influence = np.exp(-hydrology.distance_to_water / max(config.water_moderation_km, 1e-6))
    broad_noise = _smooth_noise((height, width), rng, steps=5)

    # Base temperature is the equatorial sea-level reference; the gradient is
    # equator-to-pole, not an arbitrary full change across every small map.
    latitude_temperature = config.base_temperature_c - config.latitude_temperature_gradient_c * np.abs(latitude) / 90.0
    terrain_cooling = config.lapse_rate_c_per_km * elevation_km
    mean_temperature = latitude_temperature - terrain_cooling
    mean_temperature += config.climate_noise_weight * 2.0 * (broad_noise - 0.5)

    annual_amplitude = (
        config.annual_amplitude_c
        + 3.0 * np.abs(latitude) / 90.0
        + 1.5 * elevation_n
        - config.coastal_amplitude_reduction_c * water_influence
    )
    annual_amplitude = np.clip(annual_amplitude, 3.0, None)

    wind_rad = np.deg2rad(config.prevailing_wind_degrees)
    # Meteorological direction: where the wind comes FROM, clockwise from N.
    # u is eastward and v northward; raster rows increase toward the south.
    base_u = -np.sin(wind_rad)
    base_v = -np.cos(wind_rad)
    wind_dir_u, wind_dir_v = _terrain_steered_wind_direction(
        terrain,
        base_u,
        base_v,
        config,
        rng,
    )
    wind_exposure = _wind_exposure(terrain, wind_dir_u, wind_dir_v, grid.cell_size_km)
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
        + 0.10 * np.maximum(wind_exposure, 0.0)
        + 0.08 * broad_noise
    )
    mean_humidity = np.clip(humidity, 0.05, 0.98)

    precipitation = config.precipitation_base_mm_year * (
        0.70
        + 0.42 * mean_humidity
        + config.orographic_precipitation_factor * np.maximum(wind_exposure, 0.0)
        - config.rain_shadow_factor * np.maximum(-wind_exposure, 0.0)
    )
    precipitation = np.clip(precipitation, 0.0, None)

    mean_cloud = np.clip(
        0.18 + 0.48 * mean_humidity + 0.22 * _normalize01(precipitation) + 0.06 * broad_noise,
        0.05,
        0.92,
    )
    annual_toa = np.zeros((height, width), dtype=np.float64)
    for day in range(365):
        annual_toa += extraterrestrial_hourly_irradiance(latitude, day).mean(axis=0) / 365.0
    # Annual mean GHI, W/m² including nighttime. A scenario cloud attenuation
    # coefficient is an empirical prior, not a radiative-transfer calculation.
    mean_irradiance = annual_toa * clear_sky_transmissivity(terrain.elevation) * (1.0 - 0.68 * mean_cloud)

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
    grad_y, grad_x = _elevation_gradient(terrain.elevation)
    grad_mag = np.hypot(grad_x, grad_y)
    safe_mag = np.maximum(grad_mag, 1e-6)
    tangent_u = grad_y / safe_mag
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

    slope_influence = np.clip(terrain.slope / 0.35, 0.0, 1.0)
    steering = np.clip(
        0.5 * config.wind_direction_terrain_steering * slope_influence,
        0.0,
        0.70,
    )
    wind_u = (1.0 - steering) * noisy_u + steering * channel_u
    wind_v = (1.0 - steering) * noisy_v + steering * channel_v
    norm = np.maximum(np.hypot(wind_u, wind_v), 1e-6)
    return (wind_u / norm).astype(np.float32), (wind_v / norm).astype(np.float32)


def _wind_exposure(terrain: TerrainFeatures, wind_u: np.ndarray, wind_v: np.ndarray, cell_size_km: float = 2.0) -> np.ndarray:
    """Signed U·∇h proxy: positive windward, negative leeward, zero flat.

    The 0.1 slope scale is a configurable-model prior documented in the report;
    this limited proxy is not the full Smith--Barstad linear cloud model.
    """
    grad_row, grad_col = _elevation_gradient(terrain.elevation)
    along_wind_slope = (grad_col * wind_u - grad_row * wind_v) / (1000.0 * cell_size_km)
    return np.tanh(along_wind_slope / 0.1)


def _elevation_gradient(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = elevation.astype(np.float64)
    gradient_row = np.gradient(values, axis=0) if values.shape[0] > 1 else np.zeros_like(values)
    gradient_col = np.gradient(values, axis=1) if values.shape[1] > 1 else np.zeros_like(values)
    return gradient_row, gradient_col


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
