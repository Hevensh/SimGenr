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


def specific_humidity_from_relative_humidity(temperature_c: np.ndarray, relative_humidity: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    """Specific humidity kg water / kg moist air; all pressures in hPa.

    Uses the same liquid-water Tetens saturation convention as the generator.
    No hidden supersaturation clipping: invalid thermodynamic inputs fail.
    """
    temperature, humidity, pressure = np.broadcast_arrays(temperature_c, relative_humidity, pressure_hpa)
    _validate_moist_air_temperature(temperature)
    if not all(np.isfinite(value).all() for value in (temperature, humidity, pressure)):
        raise ValueError("Moist-air inputs must be finite")
    if np.any((humidity < 0) | (humidity > 1)) or np.any(pressure <= 0):
        raise ValueError("Relative humidity must be [0,1] and pressure positive")
    vapor = humidity * saturation_vapor_pressure_hpa(temperature)
    if np.any(vapor >= pressure):
        raise ValueError("Vapor pressure must be below total pressure")
    epsilon = 287.05 / 461.5
    return epsilon * vapor / (pressure - (1.0 - epsilon) * vapor)


def diagnose_moist_air(temperature_c: np.ndarray, specific_humidity_kg_kg: np.ndarray, elevation_m: np.ndarray, sea_level_pressure_hpa: np.ndarray | float = 1013.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RH fraction, surface hPa and density kg/m3 from T/q/z/p0.

    P: ideal moist-air gas law, q = water mass / moist-air mass. E: the
    hydrostatic layer uses surface q and T_surface + 0.00325*z K. This is
    a reduced column, not a sounding. RH may exceed 1: callers must declare
    an explicit saturation treatment rather than clipping this diagnostic.
    """
    _validate_moist_air_temperature(temperature_c)
    temperature, humidity, height, p0 = np.broadcast_arrays(
        np.asarray(temperature_c, dtype=np.float64) + 273.15,
        np.asarray(specific_humidity_kg_kg, dtype=np.float64),
        np.asarray(elevation_m, dtype=np.float64),
        np.asarray(sea_level_pressure_hpa, dtype=np.float64),
    )
    if not all(np.isfinite(value).all() for value in (temperature, humidity, height, p0)):
        raise ValueError("Moist-air primitives must be finite")
    if np.any(temperature <= 0) or np.any(p0 <= 0) or np.any((humidity < 0) | (humidity >= 1)):
        raise ValueError("T and p0 must be positive; specific humidity must be [0,1)")
    virtual_factor = 1.0 + (461.5 / 287.05 - 1.0) * humidity
    layer_temperature = np.maximum(temperature + 0.00325 * height, 150.0)
    pressure = p0 * np.exp(-9.80665 * height / (287.05 * layer_temperature * virtual_factor))
    epsilon = 287.05 / 461.5
    vapor = pressure * humidity / (epsilon + (1.0 - epsilon) * humidity)
    relative_humidity = vapor / saturation_vapor_pressure_hpa(temperature - 273.15)
    density = 100.0 * pressure / (287.05 * temperature * virtual_factor)
    return relative_humidity, pressure, density


def _validate_moist_air_temperature(temperature_c: np.ndarray) -> None:
    """Numerical domain of our liquid-water approximation, not accuracy bounds.

    The legacy saturation helper clips its input; new thermodynamic kernels
    reject unsupported primitives so T=80 C cannot silently reuse T=65 C.
    Supercooled liquid water is a convention here, not mixed-phase physics.
    """
    values = np.asarray(temperature_c, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < -90.0) | (values > 65.0)):
        raise ValueError("Moist-air temperature must be finite and within the supported [-90,65] degC numerical domain")
