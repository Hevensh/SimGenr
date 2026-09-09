from __future__ import annotations

import numpy as np

from world_generator.core.config import WorldGridConfig
from world_generator.core.datatypes import TerrainBase, TerrainFeatures


def derive_terrain_features(
    base: TerrainBase,
    grid: WorldGridConfig,
) -> TerrainFeatures:
    elevation = base.elevation.astype(np.float32)
    cell_m = float(grid.cell_size_km) * 1000.0
    if cell_m <= 0.0 or not np.isfinite(elevation).all():
        raise ValueError("Terrain derivatives require finite elevation and positive cell size")
    grad_y = np.gradient(elevation, cell_m, axis=0) if elevation.shape[0] > 1 else np.zeros_like(elevation)
    grad_x = np.gradient(elevation, cell_m, axis=1) if elevation.shape[1] > 1 else np.zeros_like(elevation)
    slope = np.hypot(grad_x, grad_y).astype(np.float32)
    aspect = np.arctan2(-grad_y, -grad_x)
    aspect_sin = np.sin(aspect).astype(np.float32)
    aspect_cos = np.cos(aspect).astype(np.float32)
    # Dimensionless local relief over the 3x3 window's two-cell baseline.
    # Unlike min-max scaling this is unchanged by distant mountain outliers.
    roughness = (_local_range(elevation, radius=1) / (2.0 * cell_m)).astype(np.float32)
    curvature = _laplacian(elevation, cell_m).astype(np.float32)
    return TerrainFeatures(
        elevation=elevation,
        slope=slope,
        aspect_sin=aspect_sin,
        aspect_cos=aspect_cos,
        roughness=roughness,
        curvature=curvature,
    )


def _local_range(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    windows = []
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            windows.append(padded[dr : dr + values.shape[0], dc : dc + values.shape[1]])
    stack = np.stack(windows, axis=0)
    local = stack.max(axis=0) - stack.min(axis=0)
    return local.astype(np.float32)


def _laplacian(values: np.ndarray, cell_m: float) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * center
    )
    scale = max(cell_m * cell_m, 1.0)
    return lap / scale
