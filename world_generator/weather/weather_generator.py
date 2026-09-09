"""Conditional stochastic weather with conservative hourly disaggregation.

This is a scenario generator, not numerical weather prediction. Rain occurrence
precedes other variables. Empirical assumptions: docs/research_weather.md.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.ndimage import convolve1d, shift
from scipy.special import ndtr
from scipy.stats import gamma

from world_generator.core.config import WeatherConfig, WorldGridConfig
from world_generator.core.contracts import aggregate_complete_days
from world_generator.core.datatypes import ClimateBaseline, HydrologyState, TerrainFeatures, WeatherStore
from world_generator.weather.physics import clear_sky_transmissivity, diagnose_moist_air, extraterrestrial_hourly_irradiance, latitude_grid, saturation_vapor_pressure_hpa, specific_humidity_from_relative_humidity, surface_pressure_hpa
from world_generator.weather.random_fields import displacement_cells, gaussian_field, temporal_retention


WEATHER_CHANNELS = (
    "wind_u", "wind_v", "wind_speed", "temperature", "humidity",
    "pressure", "cloud", "precipitation", "irradiance",
)


def generate_daily_weather(terrain: TerrainFeatures, hydrology: HydrologyState, climate: ClimateBaseline, grid: WorldGridConfig, config: WeatherConfig, rng: np.random.Generator) -> WeatherStore:
    """Daily driver anchors; rain is mm/day, RH/p diagnose the anchor state.

    These are conditional scenario inputs, not means of a hidden hourly truth.
    Use aggregate_daily_weather for realized means of generated hourly values.
    """
    del hydrology  # Its influence is already in the climate baseline.
    days = int(config.days)
    if days < 1:
        raise ValueError("Weather days must be positive")
    wet_probability = float(config.wet_day_probability)
    persistence = float(config.wet_day_persistence)
    gamma_shape = float(config.precipitation_gamma_shape)
    if not 0.0 <= wet_probability <= 1.0:
        raise ValueError("wet_day_probability must be in [0, 1]")
    if not 0.0 <= persistence < 1.0 or gamma_shape <= 0.0:
        raise ValueError("wet_day_persistence must be in [0, 1), gamma shape must be positive")
    shape = terrain.elevation.shape
    latitude = latitude_grid(grid, shape)
    hemisphere = np.where(latitude >= 0.0, 1.0, -1.0)
    clear_transmission = clear_sky_transmissivity(terrain.elevation)
    dynamic = np.zeros((days, len(WEATHER_CHANNELS), *shape), dtype=np.float32)
    weather_class = np.zeros((days, *shape), dtype=np.int16)
    diagnostics = {name: np.zeros((days, *shape), dtype=np.float32) for name in ("specific_humidity_kg_kg", "sea_level_pressure_hpa", "air_density_kg_m3")}
    timestamps = np.arange(days, dtype=np.int32) + int(config.start_day_of_year)
    storm = _scenario_field(shape, rng, config, grid)
    thermal = _scenario_field(shape, rng, config, grid)
    moisture = _scenario_field(shape, rng, config, grid)
    wet = ndtr(_scenario_field(shape, rng, config, grid)) < wet_probability
    wind_base_speed = np.hypot(climate.prevailing_wind_u, climate.prevailing_wind_v)
    safe_speed = np.maximum(wind_base_speed, 1e-9)
    mean_u = float(np.mean(climate.prevailing_wind_u / safe_speed))
    mean_v = float(np.mean(climate.prevailing_wind_v / safe_speed))
    # Constant occurrence probability has a known stationary distribution.
    # Seasonality in conditional amounts preserves expected annual mm.
    p_dry_to_wet = wet_probability * (1.0 - persistence)
    p_wet_to_wet = wet_probability + persistence * (1.0 - wet_probability)
    for t, timestamp in enumerate(timestamps):
        doy = int(timestamp) % 365
        storm = _advect_and_innovate(storm, mean_u, mean_v, config, rng, grid=grid)
        thermal = _advect_and_innovate(thermal, mean_u, mean_v, config, rng, grid=grid)
        moisture = _advect_and_innovate(moisture, mean_u, mean_v, config, rng, grid=grid)
        occurrence = ndtr(_scenario_field(shape, rng, config, grid))
        wet = occurrence < np.where(wet, p_wet_to_wet, p_dry_to_wet)
        # Gaussian copula yields correlated gamma marginals. No realization
        # min/max normalization and no artificial fixed annual rainfall sum.
        amount_uniform = np.clip(ndtr(_scenario_field(shape, rng, config, grid)), 1e-10, 1.0 - 1e-10)
        amount_multiplier = gamma.ppf(amount_uniform, a=gamma_shape) / gamma_shape
        wet_season = 1.0 + 0.35 * hemisphere * np.cos(2.0 * np.pi * (doy - 112.0) / 365.0)
        precipitation = (wet * amount_multiplier * climate.mean_precipitation * wet_season / (365.0 * wet_probability)
                         if wet_probability > 0 else np.zeros(shape, dtype=np.float64))
        seasonal_temperature = climate.mean_temperature + climate.annual_temperature_amplitude * hemisphere * _seasonal_curve(doy, 172.0 + config.temperature_seasonal_lag_days, config)
        wet_anomaly = wet.astype(float) - wet_probability
        temperature = seasonal_temperature + config.temperature_synoptic_c * (0.8 * thermal - 0.3 * storm) - 1.5 * wet_anomaly
        # Vapor pressure first, then RH via temperature-dependent saturation.
        actual_vapor = climate.mean_humidity * saturation_vapor_pressure_hpa(seasonal_temperature)
        actual_vapor *= np.exp(config.humidity_synoptic_weight * (0.7 * moisture + 0.3 * storm) + 0.16 * wet_anomaly)
        humidity = np.clip(actual_vapor / saturation_vapor_pressure_hpa(temperature), 0.01, 1.0)
        cloud = np.clip(climate.mean_cloud + config.cloud_synoptic_weight * (0.55 * storm + 0.25 * moisture) + 0.38 * wet_anomaly, 0.0, 1.0)
        wind_sigma = max(float(config.wind_synoptic_weight), 0.0)
        wind_factor = np.exp(wind_sigma * storm - 0.5 * wind_sigma**2)
        wind_speed = wind_base_speed * wind_factor
        wind_u = climate.prevailing_wind_u * wind_factor
        wind_v = climate.prevailing_wind_v * wind_factor
        sea_level_pressure = np.clip(config.pressure_base_hpa - config.pressure_synoptic_hpa * (0.7 * storm + 0.3 * wet_anomaly), 850.0, 1080.0)
        pressure = surface_pressure_hpa(terrain.elevation, temperature, humidity, sea_level_pressure)
        specific_humidity = specific_humidity_from_relative_humidity(temperature, humidity, pressure)
        specific_humidity, humidity, pressure, density = _bounded_moist_air(temperature, specific_humidity, terrain.elevation, sea_level_pressure)
        for name, values in zip(diagnostics, (specific_humidity, sea_level_pressure, density)):
            diagnostics[name][t] = values
        toa_daily = extraterrestrial_hourly_irradiance(latitude, doy).mean(axis=0)
        cloud_transmission = np.clip(1.0 - config.irradiance_cloud_sensitivity * cloud, 0.0, 1.0)
        irradiance = toa_daily * clear_transmission * cloud_transmission
        for index, values in enumerate((wind_u, wind_v, wind_speed, temperature, humidity, pressure, cloud, precipitation, irradiance)):
            dynamic[t, index] = values
        weather_class[t] = _classify_weather(cloud, precipitation, wind_speed)
    return WeatherStore(dynamic, weather_class, timestamps, WEATHER_CHANNELS, "day", int(config.start_day_of_year), diagnostics, {
        **_weather_metadata(config, grid), "statistic_role": "daily_driver_anchors",
        "diagnostic_support": "diagnostics_of_daily_anchor_state_not_means_of_hourly_diagnostics",
    }, terrain.elevation.copy())


def generate_hourly_weather_week_from_baseline(terrain: TerrainFeatures, hydrology: HydrologyState, climate: ClimateBaseline, grid: WorldGridConfig, config: WeatherConfig, rng: np.random.Generator) -> WeatherStore:
    """Compatibility entry point using the same daily→hourly construction.

    For paired datasets use generate_hourly_weather_week on the stored daily
    series. This wrapper generates only the configured consecutive days.
    """
    local_config = replace(config, days=max(1, int(config.hourly_week_days)))
    daily = generate_daily_weather(terrain, hydrology, climate, grid, local_config, rng)
    return generate_hourly_weather_week(daily, local_config, rng, grid=grid)


def generate_hourly_weather_week(daily_weather: WeatherStore, config: WeatherConfig, rng: np.random.Generator, *, grid: WorldGridConfig | None = None) -> WeatherStore:
    """Hourly primitive/diagnostic weather from declared daily constraints.

    Primitive mode conditions T, vector wind, cloud and rain; scalar wind,
    RH/p and GHI daily statistics follow the hourly mechanism. Conditioned
    mode additionally enforces scalar wind and jointly feasible cloud/GHI.
    Neither mode independently locks daily RH or pressure. Solar intervals
    are [h,h+1), GHI is W/m² mean and rain is accumulated mm in that interval.
    """
    if daily_weather.time_unit != "day" or daily_weather.dynamic.ndim != 4 or daily_weather.dynamic.shape[0] < 1:
        raise ValueError("Hourly disaggregation requires a nonempty daily WeatherStore")
    if not np.all(np.isfinite(daily_weather.dynamic)):
        raise ValueError("Daily weather must contain finite physical values")
    if daily_weather.timestamps.shape != (daily_weather.dynamic.shape[0],) or not np.all(np.isfinite(daily_weather.timestamps)):
        raise ValueError("Daily weather timestamps must match the number of days and be finite")
    if not np.all(daily_weather.timestamps == np.rint(daily_weather.timestamps)):
        raise ValueError("Daily weather timestamps must be integer day indices")
    if not np.all(np.diff(daily_weather.timestamps) == 1):
        raise ValueError("Daily weather timestamps must be consecutive")
    daily_weather.as_arrays()  # Also validate optional diagnostic shapes and labels.
    week_days = min(max(int(config.hourly_week_days), 1), daily_weather.dynamic.shape[0])
    start_index = int(rng.integers(0, daily_weather.dynamic.shape[0] - week_days + 1))
    shape = daily_weather.dynamic.shape[2:]
    if grid is None:
        grid = WorldGridConfig(height=shape[0], width=shape[1])
    latitude = latitude_grid(grid, shape)
    channels = {name: index for index, name in enumerate(daily_weather.channel_names)}
    if len(channels) != len(daily_weather.channel_names) or len(channels) != daily_weather.dynamic.shape[1]:
        raise ValueError("Daily weather channel names must be unique and match the channel dimension")
    if not set(WEATHER_CHANNELS).issubset(channels):
        raise ValueError("Daily weather is missing required channels")
    for name in ("precipitation", "irradiance", "wind_speed"):
        if np.any(daily_weather.dynamic[:, channels[name]] < 0):
            raise ValueError(f"Daily {name} must be nonnegative")
    for name in ("humidity", "cloud"):
        values = daily_weather.dynamic[:, channels[name]]
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"Daily {name} must lie in [0, 1]")
    if np.any(daily_weather.dynamic[:, channels["pressure"]] <= 0):
        raise ValueError("Daily pressure must be positive")
    if np.any(daily_weather.dynamic[:, channels["temperature"]] <= -273.15):
        raise ValueError("Daily temperature must exceed absolute zero")
    vector_speed = np.hypot(daily_weather.dynamic[:, channels["wind_u"]], daily_weather.dynamic[:, channels["wind_v"]])
    scalar_speed = daily_weather.dynamic[:, channels["wind_speed"]]
    if np.any(scalar_speed < vector_speed - (1e-6 + 1e-5 * vector_speed)):
        raise ValueError("Daily mean wind_speed must be >= hypot(mean wind_u, mean wind_v)")
    dynamic = np.zeros((week_days * 24, len(channels), *shape), dtype=np.float32)
    weather_class = np.zeros((week_days * 24, *shape), dtype=np.int16)
    start_day = int(daily_weather.timestamps[start_index])
    timestamps = start_day * 24 + np.arange(week_days * 24, dtype=np.int32)
    cloud_state = _scenario_field(shape, rng, config, grid)
    wind_state = _scenario_field(shape, rng, config, grid)
    temperature_state = _scenario_field(shape, rng, config, grid)
    elevation = np.zeros(shape) if daily_weather.static_elevation_m is None else daily_weather.static_elevation_m
    diagnostics = {name: np.zeros((week_days * 24, *shape), dtype=np.float32) for name in (
        "specific_humidity_kg_kg", "sea_level_pressure_hpa", "air_density_kg_m3", "specific_humidity_adjustment_kg_kg",
    )}
    solar_hours = (np.arange(24) + 0.5)[:, None, None]
    for day_index in range(week_days):
        target = daily_weather.dynamic[start_index + day_index].astype(np.float64)
        cloud_noise, wind_noise, temperature_noise = [], [], []
        for _ in range(24):
            rho = temporal_retention(1.0, config.hourly_memory_hours) if config.temporal_scale_mode == "physical" else 0.88
            cloud_state = rho * cloud_state + np.sqrt(1.0 - rho**2) * _scenario_field(shape, rng, config, grid)
            wind_state = rho * wind_state + np.sqrt(1.0 - rho**2) * _scenario_field(shape, rng, config, grid)
            temperature_state = rho * temperature_state + np.sqrt(1.0 - rho**2) * _scenario_field(shape, rng, config, grid)
            cloud_noise.append(cloud_state)
            wind_noise.append(wind_state)
            temperature_noise.append(temperature_state)
        cloud_noise = np.asarray(cloud_noise)
        cloud = _bounded_mean(target[channels["cloud"]] + config.hourly_cloud_variability * cloud_noise, target[channels["cloud"]])
        toa = extraterrestrial_hourly_irradiance(latitude, start_day + day_index)
        if np.any(target[channels["irradiance"]] > toa.mean(axis=0) + 1e-3):
            raise ValueError("Daily aggregate cannot fit the model physical hourly bounds (no side illumination)")
        clear_ghi = toa * clear_sky_transmissivity(elevation)
        if config.hourly_generation_mode == "daily_conditioned":
            cloud = _condition_cloud_radiation(cloud, target[channels["cloud"]], target[channels["irradiance"]], clear_ghi, config.irradiance_cloud_sensitivity)
        wind_u, wind_v = _hourly_wind(target[channels["wind_u"]], target[channels["wind_v"]], target[channels["wind_speed"]], np.asarray(wind_noise), np.asarray(temperature_noise), config)
        wind_speed = np.hypot(wind_u, wind_v)
        diurnal_amplitude = config.hourly_temperature_diurnal_c * (1.25 - 0.65 * target[channels["cloud"]])
        diurnal = diurnal_amplitude * np.cos(2.0 * np.pi * (solar_hours - 15.0) / 24.0)
        temperature_delta = diurnal + config.hourly_temperature_noise_c * np.asarray(temperature_noise) - 0.6 * (cloud - target[channels["cloud"]])
        temperature_delta -= temperature_delta.mean(axis=0)
        temperature = target[channels["temperature"]] + temperature_delta
        anchor_index = start_index + day_index
        q_anchor = daily_weather.diagnostics.get("specific_humidity_kg_kg")
        q_requested = (specific_humidity_from_relative_humidity(target[channels["temperature"]], target[channels["humidity"]], target[channels["pressure"]]) if q_anchor is None else q_anchor[anchor_index])
        p0_anchor = daily_weather.diagnostics.get("sea_level_pressure_hpa")
        if p0_anchor is None:
            # Explicit inverse of the same reduced hydrostatic column. Old
            # stores without elevation use z=0, recorded in metadata.
            _, pressure_at_unit_p0, _ = diagnose_moist_air(target[channels["temperature"]], q_requested, elevation, 1.0)
            p0_day = target[channels["pressure"]] / pressure_at_unit_p0
        else:
            p0_day = p0_anchor[anchor_index]
        pressure_noise = -0.3 * config.pressure_synoptic_hpa * cloud_noise
        sea_level_pressure = p0_day + pressure_noise - pressure_noise.mean(axis=0)
        q, humidity, pressure, density = _bounded_moist_air(temperature, np.broadcast_to(q_requested, temperature.shape), elevation, sea_level_pressure)
        # Conditional wet-hour cluster, no extra unbudgeted rain component.
        center = float(rng.uniform(0.0, 24.0))
        duration = float(rng.uniform(1.5, 4.5))
        pulse = np.exp(-0.5 * ((solar_hours - center) / duration)**2)
        pulse = np.where(np.abs(solar_hours - center) <= 2.0 * duration, pulse, 0.0)
        rain_weights = pulse * np.exp(np.clip(config.hourly_precipitation_burstiness * cloud_noise, -8.0, 8.0)) * (0.2 + cloud)
        precipitation = target[channels["precipitation"]] * rain_weights / rain_weights.sum(axis=0)
        irradiance = clear_ghi * (1.0 - config.irradiance_cloud_sensitivity * cloud)
        day_slice = slice(day_index * 24, (day_index + 1) * 24)
        dynamic[day_slice] = target[None, ...]
        for name, values in zip(WEATHER_CHANNELS, (wind_u, wind_v, wind_speed, temperature, humidity, pressure, cloud, precipitation, irradiance)):
            dynamic[day_slice, channels[name]] = values
        weather_class[day_slice] = _classify_weather(cloud, precipitation * 24.0, wind_speed)
        for name, values in zip(diagnostics, (q, sea_level_pressure, density, q - q_requested)):
            diagnostics[name][day_slice] = values
    hard_constraints = ["temperature", "wind_u", "wind_v", "cloud", "precipitation"]
    if config.hourly_generation_mode == "daily_conditioned":
        hard_constraints.extend(["wind_speed", "irradiance"])
    realized = dynamic.astype(np.float64).reshape(week_days, 24, len(channels), *shape).mean(axis=1)
    realized[:, channels["precipitation"]] *= 24.0
    residuals = np.max(np.abs(realized - daily_weather.dynamic[start_index:start_index + week_days]), axis=(0, 2, 3))
    return WeatherStore(dynamic, weather_class, timestamps, daily_weather.channel_names, "hour", start_day, diagnostics, {
        **_weather_metadata(config, grid), "statistic_role": "hourly_realization",
        "hourly_window_selection": "seeded_random_contiguous_window_from_daily_anchors",
        "hourly_window_start_day": start_day,
        "diagnostic_support": "diagnosed_from_each_hour_primitive_state",
        "daily_constraints": hard_constraints,
        "daily_anchor_residual_max_abs": {name: float(residuals[index]) for name, index in channels.items()},
        "elevation_source": "daily_store" if daily_weather.static_elevation_m is not None else "legacy_assumed_sea_level",
    }, np.asarray(elevation, dtype=np.float32).copy())


def aggregate_daily_weather(hourly_weather: WeatherStore) -> WeatherStore:
    """Aggregate complete solar days; diagnostics are averaged AFTER diagnosis.

    In general mean RH != RH(mean T, mean q, mean p). Rain is summed, all
    other channels and additional diagnostics are interval-weighted means
    (all intervals here are one hour). Incomplete or nonaligned days fail.
    """
    hourly_weather.as_arrays()
    stamps = hourly_weather.timestamps
    if hourly_weather.time_unit != "hour":
        raise ValueError("Daily aggregation requires complete midnight-aligned hourly days")
    day_indices, mean = aggregate_complete_days(hourly_weather.dynamic, stamps, quantity_kind="interval_mean")
    days = len(day_indices)
    channels = {name: index for index, name in enumerate(hourly_weather.channel_names)}
    mean[:, channels["precipitation"]] *= 24.0
    diagnostics = {name: values.astype(np.float64).reshape(days, 24, *values.shape[1:]).mean(axis=1).astype(np.float32) for name, values in hourly_weather.diagnostics.items()}
    classes = _classify_weather(mean[:, channels["cloud"]], mean[:, channels["precipitation"]], mean[:, channels["wind_speed"]])
    return WeatherStore(mean.astype(np.float32), classes, day_indices.astype(np.int32), hourly_weather.channel_names, "day", int(day_indices[0]), diagnostics, {
        **hourly_weather.metadata, "statistic_role": "daily_aggregates_of_hourly_realization",
        "diagnostic_support": "mean_of_hourly_diagnostics_not_diagnostics_of_daily_mean_state",
    }, hourly_weather.static_elevation_m)


def _weather_metadata(config: WeatherConfig, grid: WorldGridConfig) -> dict[str, object]:
    return {
        "generation_mode": config.hourly_generation_mode,
        "cloud_radiation_projection": "joint_two_moment_feasibility_then_convex_projection" if config.hourly_generation_mode == "daily_conditioned" else "none_hourly_radiation_follows_cloud",
        "spatial_scale_mode": config.spatial_scale_mode,
        "innovation_correlation_length_km": config.innovation_correlation_length_km if config.spatial_scale_mode == "physical" else None,
        "legacy_innovation_smoothing_steps": config.innovation_smoothing_steps if config.spatial_scale_mode == "legacy" else None,
        "cell_size_km": grid.cell_size_km, "spatial_boundary": config.spatial_boundary,
        "innovation_filter_boundary": "periodic" if config.spatial_boundary == "periodic" and config.spatial_scale_mode == "physical" else "reflect",
        "advection_boundary": config.spatial_boundary,
        "zero_wet_probability_semantics": "explicit_dry_scenario_overrides_climate_annual_rainfall" if config.wet_day_probability == 0 else "not_applicable",
        "temporal_scale_mode": config.temporal_scale_mode,
        "synoptic_memory_hours": config.synoptic_memory_hours if config.temporal_scale_mode == "physical" else None,
        "hourly_memory_hours": config.hourly_memory_hours if config.temporal_scale_mode == "physical" else None,
        "synoptic_advection_speed_km_per_hour": config.synoptic_advection_speed_km_per_hour if config.spatial_scale_mode == "physical" else None,
        "precipitation_statistic": "interval_accumulation_mm", "other_channels_statistic": "interval_mean",
        "solar_model": "plane_parallel_cloud_attenuation_no_side_illumination_no_twilight",
        "moisture_model": "prescribed_q_with_reported_saturation_adjustment_not_closed_column_water",
        "saturation_convention": "Tetens_over_liquid_water_not_ice_microphysics",
        "diagnostic_units": {"specific_humidity_kg_kg": "kg/kg moist air", "sea_level_pressure_hpa": "hPa", "air_density_kg_m3": "kg/m3", "specific_humidity_adjustment_kg_kg": "kg/kg moist air"},
    }


def _bounded_moist_air(temperature: np.ndarray, q_requested: np.ndarray, elevation: np.ndarray, p0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Explicit saturation adjustment of prescribed water-vapor forcing.

    Removed q is recorded, not called conserved cloud condensate or rain.
    A future prognostic water column must replace this open-reservoir rule.
    """
    q = np.broadcast_to(q_requested, np.broadcast_shapes(np.shape(temperature), np.shape(q_requested))).astype(np.float64).copy()
    rh, pressure, density = diagnose_moist_air(temperature, q, elevation, p0)
    supersaturated = rh > 1.0
    if np.any(supersaturated):
        lower, upper = np.zeros_like(q), q.copy()
        for _ in range(32):
            middle = 0.5 * (lower + upper)
            middle_rh, _, _ = diagnose_moist_air(temperature, middle, elevation, p0)
            lower = np.where(middle_rh <= 1.0, middle, lower)
            upper = np.where(middle_rh > 1.0, middle, upper)
        q = np.where(supersaturated, lower, q)
        rh, pressure, density = diagnose_moist_air(temperature, q, elevation, p0)
    return q, rh, pressure, density


def _hourly_wind(mean_u: np.ndarray, mean_v: np.ndarray, mean_speed: np.ndarray, along_noise: np.ndarray, turning_noise: np.ndarray, config: WeatherConfig) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean vector perturbations; optional feasible scalar-mean solve."""
    vector_norm = np.hypot(mean_u, mean_v)
    direction_u = np.divide(mean_u, vector_norm, out=np.ones_like(mean_u), where=vector_norm > 1e-12)
    direction_v = np.divide(mean_v, vector_norm, out=np.zeros_like(mean_v), where=vector_norm > 1e-12)
    along = along_noise - along_noise.mean(axis=0)
    turning = turning_noise - turning_noise.mean(axis=0)
    base = np.maximum(mean_speed, vector_norm)
    delta_u = base * (config.hourly_wind_variability * along * direction_u - np.deg2rad(config.hourly_wind_direction_std_degrees) * turning * direction_v)
    delta_v = base * (config.hourly_wind_variability * along * direction_v + np.deg2rad(config.hourly_wind_direction_std_degrees) * turning * direction_u)
    if config.hourly_generation_mode == "primitive_hourly":
        return mean_u + delta_u, mean_v + delta_v
    equal = np.isclose(mean_speed, vector_norm, rtol=1e-5, atol=1e-6)
    # Equality in the triangle inequality requires a single orientation.
    factor = np.exp(np.clip(config.hourly_wind_variability * along, -30.0, 30.0))
    factor /= factor.mean(axis=0)
    amplitude = np.hypot(delta_u, delta_v).mean(axis=0)
    # Configuring no variation cannot erase a requested nonzero turning gap.
    fallback = (np.arange(24)[:, None, None] < 12).astype(float) * 2.0 - 1.0
    delta_u = np.where(amplitude < 1e-12, fallback * direction_u, delta_u)
    delta_v = np.where(amplitude < 1e-12, fallback * direction_v, delta_v)
    amplitude = np.hypot(delta_u, delta_v).mean(axis=0)
    lower = np.zeros_like(mean_speed)
    upper = (mean_speed + vector_norm + 1.0) / np.maximum(amplitude, 1e-12)
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        speed = np.hypot(mean_u + middle * delta_u, mean_v + middle * delta_v).mean(axis=0)
        lower = np.where(speed < mean_speed, middle, lower)
        upper = np.where(speed >= mean_speed, middle, upper)
    scale = 0.5 * (lower + upper)
    return np.where(equal, mean_u * factor, mean_u + scale * delta_u), np.where(equal, mean_v * factor, mean_v + scale * delta_v)


def _condition_cloud_radiation(cloud: np.ndarray, cloud_mean: np.ndarray, ghi_mean: np.ndarray, clear_ghi: np.ndarray, sensitivity: float) -> np.ndarray:
    """Two moment constraints with box bounds under GHI=clear*(1-k*cloud).

    Sorting the 24 solar weights constructs exact extrema at fixed mean
    cloud. A convex mixture with the stochastic cloud satisfies any feasible
    radiation target. This is declared conditional projection, not a cloud
    water process. Zero cloud / dimmed GHI and polar-night GHI are rejected.
    """
    if sensitivity == 0:
        if not np.allclose(ghi_mean, clear_ghi.mean(axis=0), rtol=2e-6, atol=2e-5):
            raise ValueError("Cloud/GHI daily constraints are jointly infeasible with zero attenuation")
        return cloud
    order = np.argsort(clear_ghi, axis=0, kind="stable")
    mass = 24.0 * cloud_mean
    sorted_low = np.clip(mass[None] - np.arange(24)[:, None, None], 0.0, 1.0)
    sorted_high = sorted_low[::-1]
    low, high = np.zeros_like(cloud), np.zeros_like(cloud)
    np.put_along_axis(low, order, sorted_low, axis=0)
    np.put_along_axis(high, order, sorted_high, axis=0)
    minimum = (low * clear_ghi).sum(axis=0)
    maximum = (high * clear_ghi).sum(axis=0)
    required = (clear_ghi.sum(axis=0) - 24.0 * ghi_mean) / sensitivity
    tolerance = 2e-3 + 2e-6 * clear_ghi.sum(axis=0)
    if np.any((required < minimum - tolerance) | (required > maximum + tolerance)):
        raise ValueError("Cloud/GHI daily constraints are jointly infeasible in the plane-parallel model")
    required = np.clip(required, minimum, maximum)
    current = (cloud * clear_ghi).sum(axis=0)
    endpoint = np.where((required >= current)[None], high, low)
    end_weight = (endpoint * clear_ghi).sum(axis=0)
    alpha = np.divide(required - current, end_weight - current, out=np.zeros_like(current), where=np.abs(end_weight - current) > 1e-12)
    return cloud + np.clip(alpha, 0.0, 1.0)[None] * (endpoint - cloud)


def _bounded_mean(values: np.ndarray, target_mean: np.ndarray) -> np.ndarray:
    lower = -np.max(np.abs(values), axis=0) - 2.0
    upper = -lower
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        mean = np.clip(values + middle, 0.0, 1.0).mean(axis=0)
        lower = np.where(mean < target_mean, middle, lower)
        upper = np.where(mean >= target_mean, middle, upper)
    return np.clip(values + 0.5 * (lower + upper), 0.0, 1.0)


def _bounded_weighted_total(weights: np.ndarray, target_total: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Proportional allocation only on positive-weight, positive-capacity support."""
    if not all(np.isfinite(values).all() for values in (weights, target_total, capacity)) or np.any(weights < 0) or np.any(capacity < 0):
        raise ValueError("Allocation inputs must be finite and nonnegative")
    available = (weights > 0.0) & (capacity > 0.0)
    if np.any(target_total < -1e-7) or np.any(target_total > np.where(available, capacity, 0.0).sum(axis=0) + 1e-3):
        raise ValueError("Daily aggregate cannot fit the physical hourly bounds")
    safe_weights = np.where(available, np.maximum(weights, 1e-15), 0.0)
    lower = np.zeros_like(target_total)
    upper = np.max(np.divide(capacity, safe_weights, out=np.zeros_like(capacity), where=available), axis=0)
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        total = np.minimum(middle * safe_weights, capacity).sum(axis=0)
        lower = np.where(total < target_total, middle, lower)
        upper = np.where(total >= target_total, middle, upper)
    return np.minimum(0.5 * (lower + upper) * safe_weights, capacity)


@lru_cache(maxsize=32)
def _noise_kernel_and_scale(shape: tuple[int, int], steps: int) -> tuple[np.ndarray, np.ndarray]:
    # Exact Gaussian marginal variance of separable reflection convolution,
    # including edges and tiny grids. Independent of the realized field.
    sigma = max(float(steps) / 2.0, 0.5)
    radius = int(np.ceil(3.0 * sigma)) if steps > 0 else 0
    coordinates = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinates / sigma)**2)
    kernel /= kernel.sum()
    variances = []
    for size in shape:
        operator = convolve1d(np.eye(size), kernel, axis=0, mode="reflect")
        variances.append(np.sum(operator**2, axis=1))
    scale = np.sqrt(variances[0][:, None] * variances[1][None, :])
    return kernel, scale


def _gaussian_field(shape: tuple[int, int], rng: np.random.Generator, steps: int) -> np.ndarray:
    kernel, scale = _noise_kernel_and_scale(shape, max(int(steps), 0))
    field = rng.standard_normal(shape)
    for axis in (0, 1):
        field = convolve1d(field, kernel, axis=axis, mode="reflect")
    return field / scale


def _scenario_field(shape: tuple[int, int], rng: np.random.Generator, config: WeatherConfig, grid: WorldGridConfig) -> np.ndarray:
    if config.spatial_scale_mode == "legacy":
        return _gaussian_field(shape, rng, config.innovation_smoothing_steps)
    return gaussian_field(shape, rng, config.innovation_correlation_length_km, grid.cell_size_km, config.spatial_boundary)


def _advect_and_innovate(values: np.ndarray, mean_u: float, mean_v: float, config: WeatherConfig, rng: np.random.Generator, *, grid: WorldGridConfig | None = None, dt_hours: float = 24.0) -> np.ndarray:
    # Integer translation preserves marginal variance; stochastic rounding
    # preserves fractional mean velocity. Edge values do not wrap the world.
    grid = grid or WorldGridConfig(height=values.shape[0], width=values.shape[1])
    displacement = (displacement_cells(mean_u * config.synoptic_advection_speed_km_per_hour, mean_v * config.synoptic_advection_speed_km_per_hour, dt_hours, grid.cell_size_km)
                    if config.spatial_scale_mode == "physical" else np.array([-mean_v, mean_u]) * config.synoptic_shift_cells_per_day * dt_hours / 24.0)
    integer_shift = np.floor(displacement) + (rng.random(2) < (displacement - np.floor(displacement)))
    if config.spatial_boundary == "periodic":
        advected = np.roll(values, integer_shift.astype(int), axis=(0, 1))
    elif config.spatial_boundary == "reflect":
        advected = shift(values, integer_shift, order=0, mode="reflect", prefilter=False)
    else:
        advected = shift(values, integer_shift, order=0, mode="constant", cval=np.nan, prefilter=False)
        # Open boundary: fresh scenario inflow, never the opposite edge.
        inflow = _scenario_field(values.shape, rng, config, grid)
        advected = np.where(np.isfinite(advected), advected, inflow)
    rho = temporal_retention(dt_hours, config.synoptic_memory_hours) if config.temporal_scale_mode == "physical" else config.advection_rho ** (dt_hours / 24.0)
    return rho * advected + np.sqrt(1.0 - rho**2) * _scenario_field(values.shape, rng, config, grid)


def _seasonal_curve(day_of_year: float, peak_day: float, config: WeatherConfig) -> float:
    angle = 2.0 * np.pi * (day_of_year - peak_day) / 365.0
    value = np.cos(angle) + config.seasonal_second_harmonic * np.sin(2.0 * angle - 0.55) + config.seasonal_third_harmonic * np.cos(3.0 * angle + 0.8)
    return float(value / (1.0 + abs(config.seasonal_second_harmonic) + abs(config.seasonal_third_harmonic)))


def _classify_weather(cloud: np.ndarray, precipitation: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    """Display classes; hourly rainfall uses its equivalent daily rate."""
    weather = np.zeros(cloud.shape, dtype=np.int16)
    weather[cloud > 0.55] = 1
    weather[precipitation > 1.0] = 2
    weather[precipitation > 6.0] = 3
    weather[wind_speed > 9.0] = 4
    return weather
