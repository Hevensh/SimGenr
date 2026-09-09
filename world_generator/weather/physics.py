"""Small physical kernels; time is local solar time and DOY is zero based.

FAO-56 equations 11, 23--25 and 39 underpin humidity and solar geometry.
These are approximate geometrical (no refraction) solar positions, not NREL SPA.
"""
from __future__ import annotations

import numpy as np

from world_generator.core.config import WorldGridConfig


def latitude_grid(grid: WorldGridConfig, shape: tuple[int, int]) -> np.ndarray:
    """Rows increase southward; distances use a spherical local projection."""
    if grid.cell_size_km <= 0:
        raise ValueError("cell_size_km must be positive")
    latitude = grid.latitude_center_degrees + (
        (shape[0] - 1) / 2.0 - np.arange(shape[0])
    ) * grid.cell_size_km / 111.195
    if np.any(np.abs(latitude) > 90.0):
        raise ValueError("Grid latitude extends beyond a pole")
    return np.broadcast_to(latitude[:, None], shape).copy()


def _solar_declination(day_of_year: float) -> float:
    return float(0.409 * np.sin(2.0 * np.pi * ((day_of_year % 365) + 1.0) / 365.0 - 1.39))


def solar_direction(latitude_degrees: np.ndarray, day_of_year: float, solar_hour: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solar unit vector (east, north, up), including negative up at night."""
    latitude = np.deg2rad(np.asarray(latitude_degrees, dtype=np.float64))
    delta = _solar_declination(day_of_year)
    angle = np.pi * (solar_hour - 12.0) / 12.0
    east = np.broadcast_to(-np.cos(delta) * np.sin(angle), latitude.shape)
    north = np.cos(latitude) * np.sin(delta) - np.sin(latitude) * np.cos(delta) * np.cos(angle)
    up = np.sin(latitude) * np.sin(delta) + np.cos(latitude) * np.cos(delta) * np.cos(angle)
    return east, north, up


def solar_cosine_zenith(latitude_degrees: np.ndarray, day_of_year: float, solar_hour: float) -> np.ndarray:
    return solar_direction(latitude_degrees, day_of_year, solar_hour)[2]


def extraterrestrial_hourly_irradiance(latitude_degrees: np.ndarray, day_of_year: float) -> np.ndarray:
    """Analytically integrate TOA horizontal irradiance over each solar hour.

    Returns W/m² with shape (24, *latitude.shape). Night bins are exactly zero;
    sunrise/sunset bins contain their positive daylight integral. Polar day and
    night are handled by limiting the sunset hour angle.
    """
    latitude = np.deg2rad(np.asarray(latitude_degrees, dtype=np.float64))
    delta = _solar_declination(day_of_year)
    a = np.sin(latitude) * np.sin(delta)
    b = np.cos(latitude) * np.cos(delta)
    sunset = np.arccos(np.clip(-a / np.maximum(b, 1e-15), -1.0, 1.0))
    distance_factor = 1.0 + 0.033 * np.cos(2.0 * np.pi * ((day_of_year % 365) + 1.0) / 365.0)
    bounds = np.arange(25, dtype=np.float64).reshape((25,) + (1,) * latitude.ndim)
    angle = (bounds - 12.0) * np.pi / 12.0
    lower = np.maximum(angle[:-1], -sunset)
    upper = np.minimum(angle[1:], sunset)
    integral = a * (upper - lower) + b * (np.sin(upper) - np.sin(lower))
    return np.where(upper > lower, np.maximum(integral, 0.0), 0.0) * (1366.6666666667 * distance_factor * 12.0 / np.pi)


def clear_sky_transmissivity(elevation_m: np.ndarray) -> np.ndarray:
    """FAO-56 eq. 37; capped below TOA outside its ordinary elevation range."""
    return np.clip(0.75 + 2.0e-5 * np.asarray(elevation_m), 0.0, 0.95)


def saturation_vapor_pressure_hpa(temperature_c: np.ndarray) -> np.ndarray:
    """FAO-56 saturation vapor pressure over liquid water, in hPa."""
    temperature = np.clip(np.asarray(temperature_c, dtype=np.float64), -90.0, 65.0)
    return 6.108 * np.exp(17.27 * temperature / (temperature + 237.3))


def surface_pressure_hpa(elevation_m: np.ndarray, temperature_c: np.ndarray, relative_humidity: np.ndarray, sea_level_pressure_hpa: np.ndarray | float = 1013.25) -> np.ndarray:
    """Hypsometric pressure using an approximate layer-mean virtual temperature.

    The unobserved layer is approximated with a 6.5 K/km lapse rate and the
    surface mixing ratio. This is a hydrostatic approximation, not a sounding.
    """
    height = np.asarray(elevation_m, dtype=np.float64)
    temperature = np.asarray(temperature_c, dtype=np.float64) + 273.15
    p0 = np.asarray(sea_level_pressure_hpa, dtype=np.float64)
    if np.any(p0 <= 0) or np.any(temperature <= 0):
        raise ValueError("Pressure and absolute temperature must be positive")
    layer_temperature = np.maximum(temperature + 0.00325 * height, 150.0)
    dry_pressure = p0 * np.exp(-9.80665 * height / (287.05 * layer_temperature))
    vapor = np.minimum(np.clip(relative_humidity, 0.0, 1.0) * saturation_vapor_pressure_hpa(temperature_c), 0.2 * dry_pressure)
    specific_humidity = 0.622 * vapor / (dry_pressure - 0.378 * vapor)
    virtual_temperature = layer_temperature * (1.0 + 0.61 * specific_humidity)
    return p0 * np.exp(-9.80665 * height / (287.05 * virtual_temperature))
