from __future__ import annotations

import numpy as np

from world_generator.core.config import LandConfig, WorldGridConfig
from world_generator.core.datatypes import HydrologyState, StaticLandState, TerrainFeatures


LAND_COVER = {
    "plain": 1,
    "hill": 2,
    "mountain": 3,
    "water": 4,
    "wetland": 5,
    "protected": 6,
}


def generate_static_land(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    grid: WorldGridConfig,
    config: LandConfig,
    rng: np.random.Generator,
) -> StaticLandState:
    water = hydrology.river | hydrology.lake
    water_buffer = hydrology.distance_to_water <= config.water_buffer_km
    elevation_n = _normalize01(terrain.elevation)
    slope_n = _normalize01(terrain.slope)
    roughness_n = _normalize01(terrain.roughness)
    flood = np.clip(hydrology.flood_risk, 0.0, 1.0)

    vegetation_noise = _smooth_noise(terrain.elevation.shape, rng, steps=3)
    vegetation = (
        0.34 * (1.0 - slope_n)
        + 0.24 * (1.0 - elevation_n)
        + 0.17 * np.exp(-hydrology.distance_to_water / 6.0)
        + config.vegetation_noise_weight * vegetation_noise
    )
    vegetation = _normalize01(vegetation)
    vegetation[water] = 0.0

    protected_score = (
        0.36 * elevation_n
        + 0.22 * roughness_n
        + 0.20 * vegetation
        + 0.14 * np.exp(-hydrology.distance_to_water / max(config.water_buffer_km, 1e-6))
        + 0.08 * _protected_patch_field(terrain.elevation.shape, rng, config.random_patch_count)
    )
    protected_score[water] = 0.0
    protected_threshold = np.quantile(protected_score, 1.0 - np.clip(config.protected_fraction, 0.0, 0.8))
    protected = (protected_score >= protected_threshold) & ~water

    terrain_cost = (
        0.36 * slope_n
        + 0.24 * roughness_n
        + 0.16 * flood
        + 0.12 * water_buffer.astype(np.float32)
        + 0.12 * protected.astype(np.float32)
    )
    terrain_cost = _normalize01(terrain_cost)
    terrain_cost[water] = 1.0

    buildability = 1.0 - terrain_cost
    buildability *= _steep_slope_multiplier(terrain.slope, config.steep_slope_threshold)
    buildability[flood >= config.high_flood_threshold] *= 0.45
    buildability[water | protected] = 0.0
    buildability = np.clip(buildability, 0.0, 1.0).astype(np.float32)

    land_cover = _classify_land_cover(
        elevation_n=elevation_n,
        slope=terrain.slope,
        water=water,
        water_buffer=water_buffer,
        protected=protected,
        flood=flood,
        config=config,
    )

    return StaticLandState(
        land_cover=land_cover.astype(np.int16),
        vegetation=vegetation.astype(np.float32),
        protected=protected,
        buildability=buildability,
        terrain_cost=terrain_cost.astype(np.float32),
        water_buffer=water_buffer,
    )


def _classify_land_cover(
    elevation_n: np.ndarray,
    slope: np.ndarray,
    water: np.ndarray,
    water_buffer: np.ndarray,
    protected: np.ndarray,
    flood: np.ndarray,
    config: LandConfig,
) -> np.ndarray:
    land_cover = np.full(elevation_n.shape, LAND_COVER["plain"], dtype=np.int16)
    land_cover[(elevation_n > 0.46) | (slope > config.steep_slope_threshold * 0.55)] = LAND_COVER["hill"]
    land_cover[(elevation_n > 0.70) | (slope > config.steep_slope_threshold)] = LAND_COVER["mountain"]
    land_cover[water_buffer & (flood > 0.45) & ~water] = LAND_COVER["wetland"]
    land_cover[protected] = LAND_COVER["protected"]
    land_cover[water] = LAND_COVER["water"]
    return land_cover


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


def _protected_patch_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    patch_count: int,
) -> np.ndarray:
    height, width = shape
    rows, cols = np.indices(shape)
    field = np.zeros(shape, dtype=np.float32)
    for _ in range(max(patch_count, 0)):
        row = rng.uniform(0, height - 1)
        col = rng.uniform(0, width - 1)
        radius = rng.uniform(5.0, 13.0)
        field += np.exp(-(((rows - row) ** 2 + (cols - col) ** 2) / (2.0 * radius * radius)))
    return _normalize01(field)


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _steep_slope_multiplier(slope: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.0:
        return np.full_like(slope, 0.35, dtype=np.float32)
    lower = 0.75 * float(threshold)
    upper = 1.25 * float(threshold)
    progress = np.clip((slope - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    eased = 0.5 - 0.5 * np.cos(np.pi * progress)
    return (1.0 - 0.65 * eased).astype(np.float32)
