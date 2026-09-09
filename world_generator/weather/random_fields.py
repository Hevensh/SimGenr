"""Gaussian scenario fields with declared physical correlation lengths.

L is the approximately e-folding covariance distance in km on a well-resolved
interior grid: convolution sigma = L / (2 dx). Finite grids and boundaries
alter that covariance; each cell still has unit Gaussian marginal variance.
"""
from functools import lru_cache

import numpy as np
from scipy.ndimage import convolve1d


@lru_cache(maxsize=64)
def gaussian_kernel_and_scale(shape: tuple[int, int], sigma_cells: float, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    if boundary not in {"open", "reflect", "periodic"}:
        raise ValueError("Field boundary must be open, reflect or periodic")
    if not np.isfinite(sigma_cells) or sigma_cells < 0:
        raise ValueError("Gaussian sigma must be finite and nonnegative")
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Gaussian field needs a nonempty two-dimensional shape")
    mode = "wrap" if boundary == "periodic" else "reflect"
    radius = int(np.ceil(3 * sigma_cells)) if sigma_cells > 0 else 0
    coordinates = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinates / max(sigma_cells, 1e-12)) ** 2)
    kernel /= kernel.sum()
    variances = []
    for size in shape:
        operator = convolve1d(np.eye(size), kernel, axis=0, mode=mode)
        variances.append(np.sum(operator ** 2, axis=1))
    return kernel, np.sqrt(variances[0][:, None] * variances[1][None, :])


def gaussian_field(shape: tuple[int, int], rng: np.random.Generator, correlation_length_km: float, cell_size_km: float, boundary: str = "reflect") -> np.ndarray:
    if not np.isfinite(cell_size_km) or cell_size_km <= 0 or not np.isfinite(correlation_length_km) or correlation_length_km <= 0:
        raise ValueError("Physical correlation length and cell size must be positive and finite")
    kernel, scale = gaussian_kernel_and_scale(shape, correlation_length_km / (2.0 * cell_size_km), boundary)
    values = rng.standard_normal(shape)
    for axis in (0, 1):
        values = convolve1d(values, kernel, axis=axis, mode="wrap" if boundary == "periodic" else "reflect")
    return values / scale


def temporal_retention(dt_hours: float, memory_hours: float) -> float:
    """AR(1) retention exp(-dt/tau), supporting non-hourly helper calls."""
    if not np.isfinite(dt_hours) or dt_hours <= 0 or not np.isfinite(memory_hours) or memory_hours <= 0:
        raise ValueError("Time step and memory time must be positive and finite")
    return float(np.exp(-dt_hours / memory_hours))


def displacement_cells(u_km_per_hour: float, v_km_per_hour: float, dt_hours: float, cell_size_km: float) -> np.ndarray:
    """East/north velocity to south-row/east-column displacement."""
    if not all(np.isfinite(value) for value in (u_km_per_hour, v_km_per_hour, dt_hours, cell_size_km)) or dt_hours <= 0 or cell_size_km <= 0:
        raise ValueError("Velocity must be finite and time/grid spacing positive")
    return np.asarray([-v_km_per_hour, u_km_per_hour]) * dt_hours / cell_size_km
