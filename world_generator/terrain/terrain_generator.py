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
    if grid.height < 1 or grid.width < 1 or grid.cell_size_km <= 0.0:
        raise ValueError("Terrain requires a nonempty grid and positive cell size")
    if terrain.algorithm == "multiscale_v2":
        return _generate_multiscale_terrain(grid, terrain, rng)
    if terrain.algorithm != "legacy":
        raise ValueError(f"Unknown terrain algorithm: {terrain.algorithm}")

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

    lowland_min = float(terrain.lowland_elevation_min_m)
    lowland_max = float(terrain.lowland_elevation_max_m)
    if lowland_max < lowland_min:
        raise ValueError("Terrain lowland elevation maximum must not be below its minimum")
    lowland_elevation = float(rng.uniform(lowland_min, lowland_max))
    relief = _sample_positive_relief(terrain, rng)
    elevation = lowland_elevation + shaped * relief
    elevation = elevation.astype(np.float32)
    land_mask = np.ones((height, width), dtype=bool)
    return TerrainBase(elevation=elevation, land_mask=land_mask)


def _generate_multiscale_terrain(
    grid: WorldGridConfig,
    terrain: TerrainConfig,
    rng: np.random.Generator,
) -> TerrainBase:
    height, width = grid.height, grid.width
    cell_size = float(grid.cell_size_km)
    x = float(grid.origin_x_km) + (np.arange(width, dtype=np.float64) + 0.5) * cell_size
    y = float(grid.origin_y_km) + (np.arange(height, dtype=np.float64) + 0.5) * cell_size
    xx, yy = np.meshgrid(x, y)

    noise_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    warp_scale = max(float(terrain.domain_warp_wavelength_km), cell_size * 2.0)
    warp_amp = float(terrain.domain_warp_amplitude_km)
    warp_x = warp_amp * _value_noise_world(xx, yy, warp_scale, noise_seed + 101)
    warp_y = warp_amp * _value_noise_world(xx, yy, warp_scale, noise_seed + 211)
    warped_x = xx + warp_x
    warped_y = yy + warp_y

    continent_scale = max(float(terrain.continent_wavelength_km), cell_size * 4.0)
    continent = (
        0.70 * _value_noise_world(warped_x, warped_y, continent_scale, noise_seed + 307)
        + 0.30 * _value_noise_world(warped_x, warped_y, continent_scale * 0.5, noise_seed + 401)
    )
    erosion = _value_noise_world(
        warped_x,
        warped_y,
        max(float(terrain.erosion_wavelength_km), cell_size * 4.0),
        noise_seed + 503,
    )
    ridge_scale = max(float(terrain.ridge_wavelength_km), cell_size * 4.0)
    ridge_primary = 1.0 - np.abs(_value_noise_world(warped_x, warped_y, ridge_scale, noise_seed + 601))
    ridge_secondary = 1.0 - np.abs(
        _value_noise_world(warped_x, warped_y, ridge_scale * 0.5, noise_seed + 701)
    )
    ridges = np.clip(0.68 * ridge_primary**2.2 + 0.32 * ridge_secondary**2.0, 0.0, 1.0)
    detail = _world_fbm(
        warped_x,
        warped_y,
        max(float(terrain.detail_wavelength_km), cell_size * 2.0),
        max(int(terrain.octaves), 1),
        cell_size,
        noise_seed + 809,
    )

    continent01 = np.clip(0.5 + 0.5 * continent, 0.0, 1.0)
    erosion01 = np.clip(0.5 + 0.5 * erosion, 0.0, 1.0)
    detail01 = np.clip(0.5 + 0.5 * detail, 0.0, 1.0)
    continental_relief = _smoothstep(np.clip((continent01 - 0.30) / 0.40, 0.0, 1.0))
    mountain_gate = _smoothstep(np.clip((continent01 - 0.36) / 0.34, 0.0, 1.0))
    uplift = ridges * (0.24 + 0.76 * (1.0 - erosion01)) * mountain_gate
    eroded_basins = np.maximum(erosion01 - 0.58, 0.0) * (1.0 - ridges)
    shaped = (
        0.04
        + 0.56 * continental_relief
        + 1.25 * float(terrain.ridge_strength) * uplift
        + 0.12 * detail01
        - float(terrain.basin_strength) * eroded_basins
    )
    # Fixed calibration preserves world-coordinate consistency while allowing
    # typical 64-128 km windows to use most of the configured relief range.
    shaped = np.clip((shaped - 0.04) / 0.62, 0.0, 1.0).astype(np.float32)

    lowland_min = float(terrain.lowland_elevation_min_m)
    lowland_max = float(terrain.lowland_elevation_max_m)
    if lowland_max < lowland_min:
        raise ValueError("Terrain lowland elevation maximum must not be below its minimum")
    lowland_elevation = float(rng.uniform(lowland_min, lowland_max))
    relief = _sample_positive_relief(terrain, rng)
    elevation = (lowland_elevation + shaped * relief).astype(np.float32)
    return TerrainBase(elevation=elevation, land_mask=np.ones((height, width), dtype=bool))


def _world_fbm(
    xx: np.ndarray,
    yy: np.ndarray,
    base_wavelength_km: float,
    octaves: int,
    cell_size_km: float,
    seed: int,
) -> np.ndarray:
    result = np.zeros_like(xx, dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    wavelength = float(base_wavelength_km)
    for octave in range(max(octaves, 1)):
        if wavelength < cell_size_km * 2.0:
            break
        result += amplitude * _value_noise_world(xx, yy, wavelength, seed + octave * 977)
        total_amplitude += amplitude
        amplitude *= 0.52
        wavelength *= 0.5
    if total_amplitude <= 0.0:
        return np.zeros_like(xx, dtype=np.float32)
    return (result / total_amplitude).astype(np.float32)


def _value_noise_world(
    xx: np.ndarray,
    yy: np.ndarray,
    wavelength_km: float,
    seed: int,
) -> np.ndarray:
    wavelength = max(float(wavelength_km), 1e-6)
    angle = ((seed % 4093) / 4093.0 - 0.5) * np.pi
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    sx = (cos_angle * xx - sin_angle * yy) / wavelength
    sy = (sin_angle * xx + cos_angle * yy) / wavelength
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    dx = sx - x0
    dy = sy - y0
    tx = _quintic(dx)
    ty = _quintic(dy)
    n00 = _gradient_dot(x0, y0, dx, dy, seed)
    n10 = _gradient_dot(x0 + 1, y0, dx - 1.0, dy, seed)
    n01 = _gradient_dot(x0, y0 + 1, dx, dy - 1.0, seed)
    n11 = _gradient_dot(x0 + 1, y0 + 1, dx - 1.0, dy - 1.0, seed)
    nx0 = n00 + tx * (n10 - n00)
    nx1 = n01 + tx * (n11 - n01)
    return np.clip(1.75 * (nx0 + ty * (nx1 - nx0)), -1.0, 1.0).astype(np.float32)


def _gradient_dot(
    ix: np.ndarray,
    iy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    seed: int,
) -> np.ndarray:
    gx = _lattice_hash(ix, iy, seed + 17).astype(np.float64)
    gy = _lattice_hash(ix, iy, seed + 43).astype(np.float64)
    norm = np.maximum(np.hypot(gx, gy), 1e-9)
    return gx / norm * dx + gy / norm * dy


def _lattice_hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(ix, dtype=np.int64).view(np.uint64)
    y = np.asarray(iy, dtype=np.int64).view(np.uint64)
    value = x * np.uint64(0x9E3779B185EBCA87)
    value ^= y * np.uint64(0xC2B2AE3D27D4EB4F)
    value ^= np.uint64(seed & ((1 << 64) - 1))
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    unit = (value >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
    return (2.0 * unit - 1.0).astype(np.float32)


def _quintic(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values**3 * (values * (values * 6.0 - 15.0) + 10.0)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _sample_positive_relief(terrain: TerrainConfig, rng: np.random.Generator) -> float:
    mean = float(terrain.relief_mean_m)
    std = float(terrain.relief_std_m)
    if mean <= 0.0:
        raise ValueError("Terrain relief mean must be positive")
    if std < 0.0:
        raise ValueError("Terrain relief standard deviation must be non-negative")
    if std == 0.0:
        return mean
    while True:
        relief = float(rng.normal(mean, std))
        if relief > 0.0:
            return relief


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
