from __future__ import annotations

import numpy as np

from world_generator.core.config import WeatherConfig, WorldGridConfig
from world_generator.core.datatypes import ClimateBaseline, HydrologyState, TerrainFeatures, WeatherStore


WEATHER_CHANNELS = (
    "wind_u",
    "wind_v",
    "wind_speed",
    "temperature",
    "humidity",
    "pressure",
    "cloud",
    "precipitation",
    "irradiance",
)


def generate_daily_weather(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    climate: ClimateBaseline,
    grid: WorldGridConfig,
    config: WeatherConfig,
    rng: np.random.Generator,
) -> WeatherStore:
    del grid
    days = int(config.days)
    shape = terrain.elevation.shape
    dynamic = np.zeros((days, len(WEATHER_CHANNELS), *shape), dtype=np.float32)
    weather_class = np.zeros((days, *shape), dtype=np.int16)
    timestamps = np.arange(days, dtype=np.int32) + int(config.start_day_of_year)

    moisture = _smooth_noise(shape, rng, config.innovation_smoothing_steps)
    storm = _smooth_noise(shape, rng, config.innovation_smoothing_steps)
    temperature_wave = _smooth_noise(shape, rng, config.innovation_smoothing_steps)
    elevation_n = _normalize01(terrain.elevation)
    roughness_n = _normalize01(terrain.roughness)
    wind_base_speed = np.hypot(climate.prevailing_wind_u, climate.prevailing_wind_v)
    wind_base_speed = np.maximum(wind_base_speed, 0.2)
    wind_dir_u = climate.prevailing_wind_u / wind_base_speed
    wind_dir_v = climate.prevailing_wind_v / wind_base_speed
    water_wetness = np.exp(-hydrology.distance_to_water / 8.0)
    oro = _orographic_lift(terrain, wind_dir_u, wind_dir_v)
    low_frequency_anomaly = _seasonal_low_frequency_anomaly(days, rng)

    for t in range(days):
        day_of_year = (config.start_day_of_year + t) % 365
        mean_u = float(np.nanmean(wind_dir_u))
        mean_v = float(np.nanmean(wind_dir_v))
        moisture = _advect_and_innovate(moisture, mean_u, mean_v, config, rng)
        storm = _advect_and_innovate(storm, mean_u, mean_v, config, rng)
        temperature_wave = _advect_and_innovate(temperature_wave, mean_u, mean_v, config, rng)

        temperature_season = _seasonal_curve(
            day_of_year,
            peak_day=172.0 + config.temperature_seasonal_lag_days,
            config=config,
        )
        solar_curve = _seasonal_curve(
            day_of_year,
            peak_day=172.0 + config.solar_seasonal_lag_days,
            config=config,
        )
        wet_season = 0.5 + 0.5 * _seasonal_curve(day_of_year, peak_day=112.0, config=config)
        solar_season = 0.88 + 0.20 * (0.5 + 0.5 * solar_curve)
        low_frequency = low_frequency_anomaly[t]

        temperature = (
            climate.mean_temperature
            + climate.annual_temperature_amplitude * temperature_season
            + config.temperature_synoptic_c * (temperature_wave - 0.5)
            + config.low_frequency_temperature_c * low_frequency
            + 0.9 * water_wetness * (0.15 - temperature_season)
        )

        humidity = np.clip(
            climate.mean_humidity
            + config.humidity_synoptic_weight * (moisture - 0.5)
            + 0.09 * wet_season
            + 0.04 * water_wetness
            - 0.05 * elevation_n,
            0.03,
            0.99,
        )

        storm_intensity = np.clip(0.58 * storm + 0.32 * moisture + 0.10 * wet_season, 0.0, 1.0)
        cloud = np.clip(
            0.52 * climate.mean_cloud
            + config.cloud_synoptic_weight * storm_intensity
            + 0.24 * humidity
            + 0.08 * oro,
            0.02,
            0.98,
        )

        wind_factor = 1.0 + config.wind_synoptic_weight * (storm - 0.5) + 0.12 * roughness_n + 0.08 * oro
        wind_speed = np.clip(wind_base_speed * wind_factor, 0.1, None)
        wind_u = wind_dir_u * wind_speed
        wind_v = wind_dir_v * wind_speed

        precipitation = (climate.mean_precipitation / 365.0) * (
            0.22
            + config.precipitation_event_scale * np.maximum(storm_intensity - 0.48, 0.0)
            + 0.75 * oro
            + 0.25 * water_wetness
        )
        precipitation = np.clip(precipitation, 0.0, None)

        pressure = (
            config.pressure_base_hpa
            - 0.012 * terrain.elevation
            - config.pressure_synoptic_hpa * (storm - 0.5)
            + 1.8 * (0.5 - moisture)
        )

        rain_reduction = np.clip(precipitation / 18.0, 0.0, 1.0)
        irradiance = climate.mean_irradiance * solar_season * (1.0 + config.low_frequency_irradiance_weight * low_frequency) * (
            1.0 - config.irradiance_cloud_sensitivity * cloud
        )
        irradiance *= 1.0 - 0.22 * rain_reduction
        irradiance += 7.0 * elevation_n
        irradiance = np.clip(irradiance, 0.0, None)

        dynamic[t, 0] = wind_u.astype(np.float32)
        dynamic[t, 1] = wind_v.astype(np.float32)
        dynamic[t, 2] = wind_speed.astype(np.float32)
        dynamic[t, 3] = temperature.astype(np.float32)
        dynamic[t, 4] = humidity.astype(np.float32)
        dynamic[t, 5] = pressure.astype(np.float32)
        dynamic[t, 6] = cloud.astype(np.float32)
        dynamic[t, 7] = precipitation.astype(np.float32)
        dynamic[t, 8] = irradiance.astype(np.float32)
        weather_class[t] = _classify_weather(cloud, precipitation, wind_speed)

    return WeatherStore(
        dynamic=dynamic,
        weather_class=weather_class,
        timestamps=timestamps,
        channel_names=WEATHER_CHANNELS,
        time_unit="day",
        start_day_of_year=int(config.start_day_of_year),
    )


def generate_hourly_weather_week_from_baseline(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    climate: ClimateBaseline,
    grid: WorldGridConfig,
    config: WeatherConfig,
    rng: np.random.Generator,
) -> WeatherStore:
    week_days = int(np.clip(config.hourly_week_days, 1, 52))
    hours = week_days * 24
    shape = terrain.elevation.shape
    dynamic = np.zeros((hours, len(WEATHER_CHANNELS), *shape), dtype=np.float32)
    weather_class = np.zeros((hours, *shape), dtype=np.int16)
    start_day = _random_calendar_start_day(rng, int(config.start_day_of_year))
    timestamps = np.arange(hours, dtype=np.int32) + start_day * 24

    rows, cols = np.indices(shape, dtype=np.float32)
    elevation_n = _normalize01(terrain.elevation)
    roughness_n = _normalize01(terrain.roughness)
    water_wetness = np.exp(-hydrology.distance_to_water / max(10.0, grid.cell_size_km)).astype(np.float32)
    wind_base_speed = np.maximum(np.hypot(climate.prevailing_wind_u, climate.prevailing_wind_v), 0.2)
    wind_dir_u = climate.prevailing_wind_u / wind_base_speed
    wind_dir_v = climate.prevailing_wind_v / wind_base_speed
    oro = _orographic_lift(terrain, wind_dir_u, wind_dir_v)
    cloud_systems = _initial_cloud_systems(terrain, climate, hydrology, grid, config, rng)
    wind_noise = _hourly_correlated_scalar_noise(hours, rng, config.hourly_wind_variability)
    temperature_noise = _hourly_correlated_scalar_noise(hours, rng, config.hourly_temperature_noise_c)
    pressure_wave = _hourly_correlated_scalar_noise(hours, rng, 1.0)
    daily_amplitude = rng.uniform(0.78, 1.22, size=week_days).astype(np.float32)
    daily_peak_shift = rng.normal(0.0, 1.4, size=week_days).astype(np.float32)
    rain_pulses = [_cloud_rain_pulse(hours, rng, config.hourly_storm_event_rate) for _ in cloud_systems]
    cloud_cooling_memory = np.zeros(shape, dtype=np.float32)

    for hour_index in range(hours):
        day_of_year = (start_day + hour_index / 24.0) % 365.0
        local_hour = hour_index % 24
        local_day = hour_index // 24
        temperature_season = _seasonal_curve(
            day_of_year,
            peak_day=172.0 + config.temperature_seasonal_lag_days,
            config=config,
        )
        solar_season = 0.88 + 0.20 * (0.5 + 0.5 * _seasonal_curve(day_of_year, peak_day=172.0 + config.solar_seasonal_lag_days, config=config))
        wet_season = 0.5 + 0.5 * _seasonal_curve(day_of_year, peak_day=112.0, config=config)
        wind_factor = np.clip(1.0 + wind_noise[hour_index] + 0.10 * roughness_n + 0.06 * oro, 0.25, None)
        wind_speed = np.clip(wind_base_speed * wind_factor, 0.05, None)
        wind_u = wind_dir_u * wind_speed
        wind_v = wind_dir_v * wind_speed

        cloud, rain_core = _render_hourly_cloud_systems(
            cloud_systems,
            rain_pulses,
            hour_index,
            rows,
            cols,
            wind_dir_u,
            wind_dir_v,
            terrain,
            climate,
            hydrology,
            grid,
            config,
        )
        background_cloud = np.clip(0.02 + 0.12 * climate.mean_cloud + 0.05 * water_wetness + 0.04 * oro, 0.0, 0.28)
        cloud = np.clip(np.maximum(background_cloud, cloud) + 0.08 * wet_season * rain_core, 0.0, 1.0)
        humidity = np.clip(
            climate.mean_humidity
            + 0.10 * wet_season
            + 0.08 * water_wetness
            + 0.18 * cloud
            - 0.06 * elevation_n,
            0.02,
            0.99,
        )
        precipitation = (
            config.hourly_precipitation_burstiness
            * rain_core
            * np.maximum(cloud - 0.36, 0.0) ** 1.35
            * (0.35 + 0.65 * humidity)
        )
        precipitation += 0.012 * climate.mean_precipitation / 365.0 * np.maximum(cloud - 0.62, 0.0)
        precipitation = np.clip(precipitation, 0.0, None)
        cloud_shading = np.clip(0.72 * cloud**1.35 + 0.28 * rain_core, 0.0, 1.0)
        cloud_cooling_memory = 0.78 * cloud_cooling_memory + 0.22 * cloud_shading
        temperature_diurnal = -np.cos(2.0 * np.pi * (local_hour - 5.0 - daily_peak_shift[local_day]) / 24.0)
        temperature = (
            climate.mean_temperature
            + climate.annual_temperature_amplitude * temperature_season
            + config.hourly_temperature_diurnal_c * daily_amplitude[local_day] * temperature_diurnal
            + temperature_noise[hour_index]
            - 2.15 * cloud_cooling_memory
            - 0.75 * rain_core
            - 0.006 * terrain.elevation
        )
        pressure = (
            config.pressure_base_hpa
            - 0.012 * terrain.elevation
            - config.pressure_synoptic_hpa * (rain_core - 0.20)
            + 1.2 * pressure_wave[hour_index]
        )
        solar = _solar_hour_factor(local_hour + 0.25 * daily_peak_shift[local_day])
        cloud_transmission = np.clip(1.0 - config.irradiance_cloud_sensitivity * cloud_shading**1.10, 0.06, 1.0)
        clear_sky_peak = np.clip(climate.mean_irradiance * 5.2, 0.0, 980.0)
        irradiance = np.clip(
            clear_sky_peak
            * solar_season
            * solar
            * cloud_transmission
            * (1.0 - 0.18 * np.clip(precipitation / 3.0, 0.0, 1.0)),
            0.0,
            None,
        )

        dynamic[hour_index, 0] = wind_u.astype(np.float32)
        dynamic[hour_index, 1] = wind_v.astype(np.float32)
        dynamic[hour_index, 2] = wind_speed.astype(np.float32)
        dynamic[hour_index, 3] = temperature.astype(np.float32)
        dynamic[hour_index, 4] = humidity.astype(np.float32)
        dynamic[hour_index, 5] = pressure.astype(np.float32)
        dynamic[hour_index, 6] = cloud.astype(np.float32)
        dynamic[hour_index, 7] = precipitation.astype(np.float32)
        dynamic[hour_index, 8] = irradiance.astype(np.float32)
        weather_class[hour_index] = _classify_weather(cloud, precipitation, wind_speed)

    return WeatherStore(
        dynamic=dynamic,
        weather_class=weather_class,
        timestamps=timestamps,
        channel_names=WEATHER_CHANNELS,
        time_unit="hour",
        start_day_of_year=start_day,
    )


def generate_hourly_weather_week(
    daily_weather: WeatherStore,
    config: WeatherConfig,
    rng: np.random.Generator,
) -> WeatherStore:
    week_days = int(np.clip(config.hourly_week_days, 1, max(1, daily_weather.dynamic.shape[0])))
    start_index = int(rng.integers(0, max(daily_weather.dynamic.shape[0] - week_days + 1, 1)))
    hours = week_days * 24
    shape = daily_weather.dynamic.shape[2:]
    dynamic = np.zeros((hours, len(daily_weather.channel_names), *shape), dtype=np.float32)
    weather_class = np.zeros((hours, *shape), dtype=np.int16)
    timestamps = np.arange(hours, dtype=np.int32) + int(daily_weather.timestamps[start_index]) * 24
    channels = {name: index for index, name in enumerate(daily_weather.channel_names)}
    wind_noise = _hourly_correlated_noise(hours, shape, rng, config.hourly_wind_variability)
    cloud_noise = _hourly_correlated_noise(hours, shape, rng, config.hourly_cloud_variability)
    storm_system = _hourly_weather_system(hours, shape, rng)
    temperature_noise = _hourly_correlated_scalar_noise(hours, rng, config.hourly_temperature_noise_c)
    cloud_synoptic = _hourly_correlated_scalar_noise(hours, rng, 0.035)
    daily_amplitude = rng.uniform(0.70, 1.25, size=week_days).astype(np.float32)
    daily_peak_shift = rng.normal(0.0, 1.7, size=week_days).astype(np.float32)
    rain_event_gate = _rain_event_gate(hours, rng, config.hourly_storm_event_rate)
    first_base = daily_weather.dynamic[start_index]
    cloud_state = first_base[channels["cloud"]].astype(np.float32).copy()
    humidity_state = first_base[channels["humidity"]].astype(np.float32).copy()
    storm_state = _smooth_noise(shape, rng, 3)
    temperature_memory = np.zeros(shape, dtype=np.float32)

    for hour_index in range(hours):
        local_hour = hour_index % 24
        local_day = hour_index // 24
        day_a = min(start_index + local_day, daily_weather.dynamic.shape[0] - 1)
        day_b = min(day_a + 1, daily_weather.dynamic.shape[0] - 1)
        frac = local_hour / 24.0
        target = (1.0 - frac) * daily_weather.dynamic[day_a] + frac * daily_weather.dynamic[day_b]

        solar = _solar_hour_factor(local_hour + 0.25 * daily_peak_shift[local_day])
        temperature_diurnal = -np.cos(2.0 * np.pi * (local_hour - 5.0 - daily_peak_shift[local_day]) / 24.0)
        base_speed = np.maximum(target[channels["wind_speed"]], 0.1)
        wind_speed = np.clip(target[channels["wind_speed"]] * (1.0 + wind_noise[hour_index]), 0.05, None)
        wind_u = target[channels["wind_u"]] / base_speed * wind_speed
        wind_v = target[channels["wind_v"]] / base_speed * wind_speed

        mean_u = float(np.nanmean(wind_u / np.maximum(wind_speed, 0.1)))
        mean_v = float(np.nanmean(wind_v / np.maximum(wind_speed, 0.1)))
        cloud_advected = _hourly_advect_field(cloud_state, mean_u, mean_v, config)
        humidity_advected = _hourly_advect_field(humidity_state, mean_u, mean_v, config)
        storm_advected = _hourly_advect_field(storm_state, mean_u, mean_v, config)
        storm_state = np.clip(0.72 * storm_advected + 0.20 * storm_system[hour_index] + 0.08 * _smooth_noise(shape, rng, 2), 0.0, 1.0)
        humidity_target = target[channels["humidity"]]
        cloud_target = target[channels["cloud"]]
        humidity = np.clip(
            0.76 * humidity_advected
            + 0.18 * humidity_target
            + 0.035 * (storm_state - 0.5)
            + 0.12 * cloud_noise[hour_index],
            0.02,
            0.99,
        )
        cloud = np.clip(
            0.70 * cloud_advected
            + 0.20 * cloud_target
            + 0.18 * (storm_state - 0.5)
            + 0.12 * (humidity - 0.55)
            + cloud_synoptic[hour_index]
            + 0.24 * cloud_noise[hour_index],
            0.0,
            1.0,
        )
        cloud_state = cloud.astype(np.float32)
        humidity_state = humidity.astype(np.float32)
        temperature_memory = 0.82 * temperature_memory + 0.18 * temperature_noise[hour_index]
        temperature = (
            target[channels["temperature"]]
            + config.hourly_temperature_diurnal_c * daily_amplitude[local_day] * temperature_diurnal
            + temperature_memory
            - 1.35 * np.maximum(cloud - cloud_target, 0.0)
        )

        wetness = np.clip(0.35 * humidity + 0.65 * cloud, 0.0, 1.0)
        event_intensity = np.maximum(storm_state + 0.45 * cloud_noise[hour_index], 0.0)
        event_intensity = np.clip(event_intensity * rain_event_gate[hour_index], 0.0, 1.0)
        precipitation = target[channels["precipitation"]] / 24.0 * (0.10 + 0.50 * wetness)
        precipitation += config.hourly_precipitation_burstiness * event_intensity * np.maximum(cloud - 0.48, 0.0) ** 1.4
        precipitation = np.clip(precipitation, 0.0, None)
        humidity = np.clip(humidity + 0.050 * event_intensity - 0.020 * temperature_diurnal, 0.02, 0.99)
        humidity_state = humidity.astype(np.float32)
        irradiance = np.clip(
            target[channels["irradiance"]] * solar * (1.0 - 0.74 * cloud**1.25) * (1.0 - 0.18 * np.clip(precipitation / 4.0, 0.0, 1.0)),
            0.0,
            None,
        )

        dynamic[hour_index] = target.astype(np.float32)
        dynamic[hour_index, channels["wind_u"]] = wind_u.astype(np.float32)
        dynamic[hour_index, channels["wind_v"]] = wind_v.astype(np.float32)
        dynamic[hour_index, channels["wind_speed"]] = wind_speed.astype(np.float32)
        dynamic[hour_index, channels["temperature"]] = temperature.astype(np.float32)
        dynamic[hour_index, channels["humidity"]] = humidity.astype(np.float32)
        dynamic[hour_index, channels["cloud"]] = cloud.astype(np.float32)
        dynamic[hour_index, channels["precipitation"]] = precipitation.astype(np.float32)
        dynamic[hour_index, channels["irradiance"]] = irradiance.astype(np.float32)
        weather_class[hour_index] = _classify_weather(cloud, precipitation * 24.0, wind_speed)

    return WeatherStore(
        dynamic=dynamic,
        weather_class=weather_class,
        timestamps=timestamps,
        channel_names=daily_weather.channel_names,
        time_unit="hour",
        start_day_of_year=int(daily_weather.timestamps[start_index]),
    )


def _hourly_advect_field(values: np.ndarray, mean_u: float, mean_v: float, config: WeatherConfig) -> np.ndarray:
    shift_col = mean_u * config.synoptic_shift_cells_per_day / 24.0
    shift_row = -mean_v * config.synoptic_shift_cells_per_day / 24.0
    return _shift_bilinear_wrap(values, shift_row, shift_col)


def _random_calendar_start_day(rng: np.random.Generator, base_day_of_year: int = 0) -> int:
    month_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    month = int(rng.integers(0, len(month_days)))
    day_in_month = int(rng.integers(0, month_days[month]))
    return int((base_day_of_year + sum(month_days[:month]) + day_in_month) % 365)


def _initial_cloud_systems(
    terrain: TerrainFeatures,
    climate: ClimateBaseline,
    hydrology: HydrologyState,
    grid: WorldGridConfig,
    config: WeatherConfig,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    shape = climate.mean_cloud.shape
    water_wetness = np.exp(-hydrology.distance_to_water / max(12.0, grid.cell_size_km))
    spawn_weight = np.clip(0.18 + 0.50 * climate.mean_humidity + 0.22 * climate.mean_cloud + 0.20 * water_wetness, 0.0, None)
    flat_weight = spawn_weight.ravel().astype(np.float64)
    flat_weight /= max(float(flat_weight.sum()), 1e-12)
    flat_indices = rng.choice(flat_weight.size, size=max(config.hourly_cloud_system_count, 1), replace=True, p=flat_weight)
    systems: list[dict[str, object]] = []
    wind_speed = np.maximum(np.hypot(climate.prevailing_wind_u, climate.prevailing_wind_v), 0.2)
    cloudlet_count = max(int(config.hourly_cloudlet_count), 1)
    for index, flat_index in enumerate(np.atleast_1d(flat_indices)):
        row, col = np.unravel_index(int(flat_index), shape)
        radius_km = float(rng.uniform(config.hourly_cloud_radius_min_km, config.hourly_cloud_radius_max_km))
        local_humidity = float(climate.mean_humidity[row, col])
        local_cloud = float(climate.mean_cloud[row, col])
        amplitude = float(np.clip(rng.uniform(0.34, 0.82) + 0.18 * local_humidity + 0.10 * local_cloud, 0.20, 0.95))
        is_raining = rng.random() < np.clip(config.hourly_raining_cloud_fraction + 0.24 * local_humidity + 0.10 * local_cloud, 0.05, 0.88)
        base_speed_cells = config.hourly_cloud_motion_km_per_hour / max(grid.cell_size_km, 1e-6)
        wind_scale = float(np.clip(wind_speed[row, col] / max(float(np.nanmean(wind_speed)), 0.1), 0.55, 1.65))
        speed_cells = base_speed_cells * wind_scale * float(rng.uniform(0.72, 1.32))
        dir_u = float(climate.prevailing_wind_u[row, col] / wind_speed[row, col])
        dir_v = float(climate.prevailing_wind_v[row, col] / wind_speed[row, col])
        offsets = rng.normal(0.0, 0.42, size=(cloudlet_count, 2)).astype(np.float32)
        offsets[0] = 0.0
        offset_norm = np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-6)
        offsets *= np.minimum(offset_norm, 1.15) / offset_norm
        radius_cells = radius_km / max(grid.cell_size_km, 1e-6)
        local_elevation = float(terrain.elevation[row, col])
        systems.append(
            {
                "row": float(row) + rng.uniform(-1.0, 1.0),
                "col": float(col) + rng.uniform(-1.0, 1.0),
                "radius_cells": radius_cells,
                "amplitude": amplitude,
                "rain_intensity": float(rng.uniform(0.45, 1.35) if is_raining else 0.0),
                "speed_cells": speed_cells,
                "dir_u": dir_u,
                "dir_v": dir_v,
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                "wobble": float(rng.uniform(0.25, 1.15)),
                "birth": float(rng.integers(-24, 72)) if index >= 3 else float(rng.integers(-48, 12)),
                "lifetime": float(rng.integers(54, 150)),
                "cloudlet_offset_rows": offsets[:, 0] * radius_cells,
                "cloudlet_offset_cols": offsets[:, 1] * radius_cells,
                "cloudlet_radius_scale": rng.uniform(0.24, 0.52, size=cloudlet_count).astype(np.float32),
                "cloudlet_strength": rng.uniform(0.50, 1.05, size=cloudlet_count).astype(np.float32),
                "cloudlet_altitude_m": (
                    local_elevation
                    + rng.uniform(650.0, 1850.0, size=cloudlet_count)
                    + 260.0 * rng.normal(0.0, 1.0, size=cloudlet_count)
                ).astype(np.float32),
                "cloudlet_cohesion": rng.uniform(0.18, 0.52, size=cloudlet_count).astype(np.float32),
                "cloudlet_phase": rng.uniform(0.0, 2.0 * np.pi, size=cloudlet_count).astype(np.float32),
            }
        )
    return systems


def _render_hourly_cloud_systems(
    systems: list[dict[str, object]],
    rain_pulses: list[np.ndarray],
    hour_index: int,
    rows: np.ndarray,
    cols: np.ndarray,
    wind_dir_u: np.ndarray,
    wind_dir_v: np.ndarray,
    terrain: TerrainFeatures,
    climate: ClimateBaseline,
    hydrology: HydrologyState,
    grid: WorldGridConfig,
    config: WeatherConfig,
) -> tuple[np.ndarray, np.ndarray]:
    del wind_dir_u, wind_dir_v, config
    height, width = rows.shape
    water_wetness = np.exp(-hydrology.distance_to_water / max(12.0, grid.cell_size_km))
    cloud = np.zeros((height, width), dtype=np.float32)
    rain_core = np.zeros((height, width), dtype=np.float32)
    moisture_gain = np.clip(0.80 + 0.38 * climate.mean_humidity + 0.18 * water_wetness, 0.45, 1.35)
    grad_y, grad_x = np.gradient(terrain.elevation.astype(np.float32))
    terrain_relief = _normalize01(terrain.elevation)
    for system, rain_pulse in zip(systems, rain_pulses):
        age = float(hour_index) - float(system["birth"])
        lifetime = float(system["lifetime"])
        if age < 0.0 or age > lifetime:
            continue
        life = np.sin(np.pi * age / max(lifetime, 1.0))
        if life <= 0.0:
            continue
        phase = float(system["phase"])
        center_row = (
            float(system["row"])
            - float(system["dir_v"]) * float(system["speed_cells"]) * age
            + float(system["wobble"]) * np.sin(0.07 * age + phase)
        ) % height
        center_col = (
            float(system["col"])
            + float(system["dir_u"]) * float(system["speed_cells"]) * age
            + float(system["wobble"]) * np.cos(0.06 * age + phase)
        ) % width
        offsets_r = np.asarray(system["cloudlet_offset_rows"], dtype=np.float32)
        offsets_c = np.asarray(system["cloudlet_offset_cols"], dtype=np.float32)
        radius_scales = np.asarray(system["cloudlet_radius_scale"], dtype=np.float32)
        strengths = np.asarray(system["cloudlet_strength"], dtype=np.float32)
        altitudes = np.asarray(system["cloudlet_altitude_m"], dtype=np.float32)
        cohesions = np.asarray(system["cloudlet_cohesion"], dtype=np.float32)
        phases = np.asarray(system["cloudlet_phase"], dtype=np.float32)
        system_radius = max(float(system["radius_cells"]), 1.0)
        for idx in range(offsets_r.size):
            attraction = 1.0 - cohesions[idx] * (1.0 - np.exp(-age / 36.0))
            split = np.sin(0.045 * age + phases[idx])
            cloudlet_row = (center_row + offsets_r[idx] * attraction + 0.55 * split * float(system["dir_u"])) % height
            cloudlet_col = (center_col + offsets_c[idx] * attraction - 0.55 * split * float(system["dir_v"])) % width
            nearest_row = int(np.clip(np.rint(cloudlet_row), 0, height - 1))
            nearest_col = int(np.clip(np.rint(cloudlet_col), 0, width - 1))
            peak_pressure = float(np.clip((terrain.elevation[nearest_row, nearest_col] + 420.0 - altitudes[idx]) / 900.0, 0.0, 1.0))
            if peak_pressure > 0.05:
                cloudlet_row = (cloudlet_row - 1.35 * peak_pressure * grad_y[nearest_row, nearest_col] / max(abs(grad_y[nearest_row, nearest_col]) + abs(grad_x[nearest_row, nearest_col]), 1e-6)) % height
                cloudlet_col = (cloudlet_col - 1.35 * peak_pressure * grad_x[nearest_row, nearest_col] / max(abs(grad_y[nearest_row, nearest_col]) + abs(grad_x[nearest_row, nearest_col]), 1e-6)) % width
            distance = _wrapped_distance(rows, cols, float(cloudlet_row), float(cloudlet_col), height, width)
            radius = max(system_radius * float(radius_scales[idx]) * (1.0 - 0.28 * peak_pressure), 0.85)
            core = np.exp(-0.5 * np.square(distance / radius)).astype(np.float32)
            fine = (
                0.76
                + 0.15 * np.sin(0.62 * rows + 0.31 * cols + 0.11 * age + phases[idx])
                + 0.09 * np.sin(0.21 * rows - 0.48 * cols + 0.07 * age + 1.7 * phases[idx])
            )
            altitude_clearance = np.clip((altitudes[idx] - terrain.elevation) / 1100.0, 0.05, 1.0)
            terrain_split = np.clip(1.0 - 0.48 * peak_pressure - 0.24 * terrain_relief * (1.0 - altitude_clearance), 0.05, 1.0)
            patch = float(system["amplitude"]) * float(strengths[idx]) * life * core * fine * moisture_gain * terrain_split
            cloud = np.maximum(cloud, patch.astype(np.float32))
            if float(system["rain_intensity"]) > 0.0 and idx % 3 != 1:
                rain_radius = max(radius * 0.72, 0.75)
                rain_blob = np.exp(-0.5 * np.square(distance / rain_radius)).astype(np.float32)
                rain_core = np.maximum(
                    rain_core,
                    rain_blob * float(system["rain_intensity"]) * float(rain_pulse[hour_index]) * life * float(strengths[idx]) * altitude_clearance,
                )
    return np.clip(cloud, 0.0, 1.0), np.clip(rain_core, 0.0, 1.0)


def _wrapped_distance(
    rows: np.ndarray,
    cols: np.ndarray,
    center_row: float,
    center_col: float,
    height: int,
    width: int,
) -> np.ndarray:
    dr = np.abs(rows - center_row)
    dc = np.abs(cols - center_col)
    dr = np.minimum(dr, height - dr)
    dc = np.minimum(dc, width - dc)
    return np.hypot(dr, dc)


def _cloud_rain_pulse(hours: int, rng: np.random.Generator, event_rate: float) -> np.ndarray:
    pulse = np.zeros(hours, dtype=np.float32)
    index = 0
    local_rate = float(np.clip(event_rate * rng.uniform(0.45, 1.25), 0.01, 0.60))
    while index < hours:
        if rng.random() < local_rate:
            duration = int(rng.integers(3, 14))
            peak = float(rng.uniform(0.45, 1.0))
            end = min(index + duration, hours)
            window = np.sin(np.linspace(0.0, np.pi, duration, dtype=np.float32)) * peak
            pulse[index:end] = np.maximum(pulse[index:end], window[: end - index])
            index += max(1, duration // 2)
        else:
            index += 1
    return pulse


def _shift_bilinear_wrap(values: np.ndarray, shift_row: float, shift_col: float) -> np.ndarray:
    height, width = values.shape
    rows, cols = np.indices(values.shape, dtype=np.float32)
    src_rows = (rows - shift_row) % height
    src_cols = (cols - shift_col) % width
    r0 = np.floor(src_rows).astype(np.int32)
    c0 = np.floor(src_cols).astype(np.int32)
    r1 = (r0 + 1) % height
    c1 = (c0 + 1) % width
    wr = src_rows - r0
    wc = src_cols - c0
    top = (1.0 - wc) * values[r0, c0] + wc * values[r0, c1]
    bottom = (1.0 - wc) * values[r1, c0] + wc * values[r1, c1]
    return ((1.0 - wr) * top + wr * bottom).astype(np.float32)


def _advect_and_innovate(
    values: np.ndarray,
    mean_u: float,
    mean_v: float,
    config: WeatherConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    shift_col = int(np.rint(mean_u * config.synoptic_shift_cells_per_day))
    shift_row = int(np.rint(-mean_v * config.synoptic_shift_cells_per_day))
    advected = np.roll(values, shift=(shift_row, shift_col), axis=(0, 1))
    innovation = _smooth_noise(values.shape, rng, config.innovation_smoothing_steps)
    rho = np.clip(config.advection_rho, 0.0, 0.999)
    mixed = rho * advected + np.sqrt(1.0 - rho * rho) * innovation
    return _normalize01(mixed)


def _seasonal_curve(day_of_year: float, peak_day: float, config: WeatherConfig) -> float:
    angle = 2.0 * np.pi * (day_of_year - peak_day) / 365.0
    value = np.cos(angle)
    value += config.seasonal_second_harmonic * np.sin(2.0 * angle - 0.55)
    value += config.seasonal_third_harmonic * np.cos(3.0 * angle + 0.8)
    return float(np.clip(value / (1.0 + config.seasonal_second_harmonic + config.seasonal_third_harmonic), -1.0, 1.0))


def _seasonal_low_frequency_anomaly(days: int, rng: np.random.Generator) -> np.ndarray:
    anchors = max(8, int(np.ceil(days / 42)))
    values = rng.normal(0.0, 1.0, anchors).astype(np.float32)
    values = np.r_[values[-2:], values, values[:2]]
    for _ in range(3):
        values = (
            np.roll(values, 1)
            + 2.0 * values
            + np.roll(values, -1)
        ) / 4.0
    anchor_x = np.linspace(-2.0, days + 1.0, values.size, dtype=np.float32)
    day_x = np.arange(days, dtype=np.float32)
    anomaly = np.interp(day_x, anchor_x, values).astype(np.float32)
    anomaly -= float(np.mean(anomaly))
    std = float(np.std(anomaly))
    if std > 1e-6:
        anomaly /= std
    return np.clip(anomaly, -1.8, 1.8).astype(np.float32)


def _hourly_correlated_noise(hours: int, shape: tuple[int, int], rng: np.random.Generator, scale: float) -> np.ndarray:
    values = rng.normal(0.0, scale, size=(hours, *shape)).astype(np.float32)
    for index in range(1, hours):
        values[index] = 0.62 * values[index - 1] + 0.38 * values[index]
    return values


def _hourly_correlated_scalar_noise(hours: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    values = rng.normal(0.0, scale, size=hours).astype(np.float32)
    for index in range(1, hours):
        values[index] = 0.58 * values[index - 1] + 0.42 * values[index]
    return values


def _hourly_weather_system(hours: int, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    coarse = rng.normal(0.0, 1.0, size=(hours, *shape)).astype(np.float32)
    for index in range(hours):
        coarse[index] = _smooth_signed_field(coarse[index], 4)
    for index in range(1, hours):
        coarse[index] = 0.80 * coarse[index - 1] + 0.20 * coarse[index]
    return np.clip(0.5 + 0.5 * coarse, 0.0, 1.0).astype(np.float32)


def _rain_event_gate(hours: int, rng: np.random.Generator, event_rate: float) -> np.ndarray:
    gate = np.zeros(hours, dtype=np.float32)
    index = 0
    while index < hours:
        if rng.random() < event_rate:
            duration = int(rng.integers(2, 9))
            peak = float(rng.uniform(0.45, 1.0))
            window = np.sin(np.linspace(0.0, np.pi, duration, dtype=np.float32)) * peak
            end = min(index + duration, hours)
            gate[index:end] = np.maximum(gate[index:end], window[: end - index])
            index += max(1, duration // 2)
        else:
            index += 1
    return gate


def _smooth_signed_field(values: np.ndarray, steps: int) -> np.ndarray:
    field = values.astype(np.float32)
    for _ in range(max(steps, 0)):
        padded = np.pad(field, 1, mode="wrap")
        field = (
            padded[:-2, 1:-1]
            + padded[1:-1, :-2]
            + 2.0 * padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, 1:-1]
        ) / 6.0
    std = float(np.std(field))
    if std > 1e-6:
        field = (field - float(np.mean(field))) / std
    return np.clip(field, -2.5, 2.5).astype(np.float32)


def _solar_hour_factor(local_hour: float) -> float:
    phase = np.sin(np.pi * (local_hour - 6.0) / 12.0)
    return float(np.maximum(phase, 0.0) ** 1.35)


def _orographic_lift(terrain: TerrainFeatures, wind_u: np.ndarray, wind_v: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(terrain.elevation.astype(np.float32))
    lift = wind_u * grad_x + wind_v * grad_y
    return _normalize01(np.maximum(lift, 0.0))


def _classify_weather(cloud: np.ndarray, precipitation: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    weather = np.zeros(cloud.shape, dtype=np.int16)
    weather[cloud > 0.55] = 1
    weather[precipitation > 1.0] = 2
    weather[precipitation > 6.0] = 3
    weather[wind_speed > 9.0] = np.maximum(weather[wind_speed > 9.0], 4)
    return weather


def _smooth_noise(shape: tuple[int, int], rng: np.random.Generator, steps: int) -> np.ndarray:
    noise = rng.random(shape, dtype=np.float32)
    for _ in range(max(steps, 0)):
        padded = np.pad(noise, 1, mode="wrap")
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
