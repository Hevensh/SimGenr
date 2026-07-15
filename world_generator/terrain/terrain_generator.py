from __future__ import annotations

import numpy as np

from world_generator.core.config import TerrainConfig, WorldGridConfig
from world_generator.core.coordinates import normalized_grid
from world_generator.core.datatypes import TerrainBase


def generate_terrain_base(
    grid: WorldGridConfig,
    terrain: TerrainConfig,
    rng: np.random.Generator,
) -> TerrainBase:
    height, width = grid.height, grid.width
    xx, yy = normalized_grid(height, width)

    base = _fbm_value_noise(height, width, terrain.octaves, rng)
    ridges = _mountain_ridge_field(xx, yy, terrain.mountain_ridges, rng)
    basins = _basin_field(xx, yy, terrain.basins, rng)
    boundary = _boundary_falloff(xx, yy, terrain.boundary_falloff)

    shaped = base
    shaped = (1.0 - terrain.plainness) * shaped + terrain.plainness * _soften_lowlands(base)
    shaped = shaped + terrain.ridge_strength * ridges
    shaped = shaped - terrain.basin_strength * basins
    shaped = shaped * boundary
    shaped = _normalize01(shaped)

    elevation = terrain.elevation_min_m + shaped * (
        terrain.elevation_max_m - terrain.elevation_min_m
    )
    elevation = elevation.astype(np.float32)
    land_mask = np.ones((height, width), dtype=bool)
    return TerrainBase(elevation=elevation, land_mask=land_mask)


def _fbm_value_noise(
    height: int,
    width: int,
    octaves: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(max(octaves, 1)):
        coarse_h = min(2 ** (octave + 2) + 1, height)
        coarse_w = min(2 ** (octave + 2) + 1, width)
        coarse = rng.normal(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32)
        result += amplitude * _bilinear_resize(coarse, height, width)
        total_amp += amplitude
        amplitude *= 0.52
    return _normalize01(result / max(total_amp, 1e-6))


def _bilinear_resize(src: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = src.shape
    row_pos = np.linspace(0, src_h - 1, height)
    col_pos = np.linspace(0, src_w - 1, width)
    r0 = np.floor(row_pos).astype(np.int32)
    c0 = np.floor(col_pos).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, src_h - 1)
    c1 = np.clip(c0 + 1, 0, src_w - 1)
    rw = (row_pos - r0).astype(np.float32)
    cw = (col_pos - c0).astype(np.float32)

    top = (1.0 - cw)[None, :] * src[r0[:, None], c0[None, :]] + cw[None, :] * src[
        r0[:, None], c1[None, :]
    ]
    bottom = (1.0 - cw)[None, :] * src[r1[:, None], c0[None, :]] + cw[None, :] * src[
        r1[:, None], c1[None, :]
    ]
    return ((1.0 - rw)[:, None] * top + rw[:, None] * bottom).astype(np.float32)


def _mountain_ridge_field(
    xx: np.ndarray,
    yy: np.ndarray,
    ridge_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    field = np.zeros_like(xx, dtype=np.float32)
    for _ in range(max(ridge_count, 0)):
        center = rng.uniform(0.15, 0.85, size=2)
        angle = rng.uniform(0.0, np.pi)
        normal = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)
        distance = np.abs((xx - center[0]) * normal[0] + (yy - center[1]) * normal[1])
        width = rng.uniform(0.035, 0.09)
        length_axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        along = np.abs((xx - center[0]) * length_axis[0] + (yy - center[1]) * length_axis[1])
        ridge = np.exp(-(distance / width) ** 2) * np.exp(-(along / 0.58) ** 4)
        field += ridge.astype(np.float32)
    return _normalize01(field)


def _basin_field(
    xx: np.ndarray,
    yy: np.ndarray,
    basin_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    field = np.zeros_like(xx, dtype=np.float32)
    for _ in range(max(basin_count, 0)):
        center = rng.uniform(0.18, 0.82, size=2)
        sx = rng.uniform(0.10, 0.24)
        sy = rng.uniform(0.10, 0.24)
        basin = np.exp(-(((xx - center[0]) / sx) ** 2 + ((yy - center[1]) / sy) ** 2))
        field += basin.astype(np.float32)
    return _normalize01(field)


def _boundary_falloff(xx: np.ndarray, yy: np.ndarray, strength: float) -> np.ndarray:
    distance = np.minimum.reduce([xx, yy, 1.0 - xx, 1.0 - yy])
    if strength <= 0:
        return np.ones_like(xx, dtype=np.float32)
    edge = np.clip(distance / max(strength, 1e-6), 0.0, 1.0)
    return (0.72 + 0.28 * edge).astype(np.float32)


def _soften_lowlands(values: np.ndarray) -> np.ndarray:
    centered = np.clip(values, 0.0, 1.0)
    return np.where(centered < 0.55, 0.55 * (centered / 0.55) ** 1.75, centered)


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)
