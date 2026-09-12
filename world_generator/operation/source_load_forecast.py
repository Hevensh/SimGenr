from __future__ import annotations

import numpy as np

from world_generator.core.config import SourceLoadConfig, WorldGridConfig
from world_generator.core.contracts import entity_ids, integer_labels, positive_finite
from world_generator.core.datatypes import (
    GridElectricalState, LandUseState, RefinedGridTopologyState, SourceLoadForecastStore, WeatherStore,
)
from world_generator.weather.physics import (
    extraterrestrial_hourly_irradiance, latitude_grid, solar_direction,
)


def generate_source_load_forecast(
    hourly_weather: WeatherStore,
    refined_topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    rng: np.random.Generator,
    config: SourceLoadConfig | None = None,
    land_use: LandUseState | None = None,
    grid: WorldGridConfig | None = None,
) -> SourceLoadForecastStore:
    """Generate a synthetic realization conditional on *realized* hourly weather.

    The historical API name is retained for saved-stage compatibility. This is
    not an out-of-sample forecast: no forecast issue time or NWP is supplied.
    Defaults are transparent scenario assumptions, not locally fitted values.
    """
    config = config or SourceLoadConfig()
    grid_was_provided = grid is not None
    shape = hourly_weather.dynamic.shape[2:]
    _validate_config(config)
    if hourly_weather.time_unit != "hour":
        raise ValueError("Source/load generation requires hourly weather")
    if hourly_weather.dynamic.ndim != 4 or not np.isfinite(hourly_weather.dynamic).all():
        raise ValueError("Weather must be a finite [hour, channel, row, col] array")
    grid = grid or WorldGridConfig(height=shape[0], width=shape[1])
    if grid_was_provided and (grid.height, grid.width) != shape:
        raise ValueError("Explicit world grid height/width must match the weather grid shape")
    hourly_weather.as_arrays()
    positive_finite(grid.cell_size_km, "cell_size_km")
    buses = refined_topology.refined_buses
    ids = entity_ids(np.asarray([bus.bus_id for bus in buses]), "source/load bus_ids")
    electrical_ids = entity_ids(np.asarray([item.bus_id for item in electrical.bus_params]), "electrical bus_ids")
    if ids.size == 0 or np.any((ids < 0) | (ids > np.iinfo(np.int32).max)) or set(ids) != set(electrical_ids):
        raise ValueError("Source/load requires nonempty nonnegative bus IDs matching electrical IDs")
    rows = integer_labels(np.asarray([bus.row for bus in buses]), "weather sample row")
    cols = integer_labels(np.asarray([bus.col for bus in buses]), "weather sample col")
    if np.any((rows < 0) | (rows >= shape[0]) | (cols < 0) | (cols >= shape[1])):
        raise ValueError("Source/load bus coordinates must lie inside the weather grid; no clipping")
    bus_params = {item.bus_id: item for item in electrical.bus_params}
    _validate_assets(buses, bus_params)
    hours = hourly_weather.dynamic.shape[0]
    bus_count = len(buses)
    p_load = np.zeros((hours, bus_count), dtype=np.float32)
    p_available = np.zeros((hours, bus_count), dtype=np.float32)
    p_scheduled = np.zeros((hours, bus_count), dtype=np.float32)
    q_load = np.zeros((hours, bus_count), dtype=np.float32)
    diagnostics = {name: np.zeros((hours, bus_count), dtype=np.float32) for name in (
        "hub_wind_speed_mps", "wind_air_density_kg_m3", "pv_poa_w_m2",
        "pv_module_temperature_c", "load_effective_temperature_c", "load_log_residual",
    )}
    initial_temperature = np.zeros(bus_count, dtype=np.float32)
    density_sources: dict[str, str] = {}
    channels = {name: index for index, name in enumerate(hourly_weather.channel_names)}
    required = {"temperature", "wind_speed", "irradiance"}
    if not required.issubset(channels):
        raise ValueError(f"Missing weather channels: {sorted(required - channels.keys())}")
    _validate_weather_physics(hourly_weather, channels)
    if hourly_weather.timestamps.shape != (hours,):
        raise ValueError("Weather timestamps must match the hourly dimension")
    hour_of_day, weekday = _calendar_features(hourly_weather.timestamps, config.calendar_start_date)
    load_indices = sorted((i for i, bus in enumerate(buses) if bus.kind == "load_bus"), key=lambda i: int(buses[i].bus_id))
    # GridBus.x/y are legacy normalized map coordinates, not kilometres.
    # Derive physical cell-centre coordinates from row/col and cell size so a
    # 20 km correlation length does not couple almost the entire world alike.
    coords = np.asarray([
        (grid.origin_x_km + (buses[i].col + 0.5) * grid.cell_size_km,
         grid.origin_y_km + (buses[i].row + 0.5) * grid.cell_size_km)
        for i in load_indices
    ], dtype=np.float64).reshape(-1, 2)
    residuals = _spatial_load_residuals(hours, coords, rng, config)
    residual_by_bus = {index: residuals[:, i] for i, index in enumerate(load_indices)}
    latitudes = latitude_grid(grid, hourly_weather.dynamic.shape[2:])

    for bus_index, bus in enumerate(buses):
        row, col = int(rows[bus_index]), int(cols[bus_index])
        weather_at_bus = hourly_weather.dynamic[:, :, row, col]
        param = bus_params[bus.bus_id]
        if bus.kind == "load_bus":
            initial = _initial_load_temperature(weather_at_bus[:, channels["temperature"]], config)
            effective = _effective_temperature(weather_at_bus[:, channels["temperature"]], config.load_thermal_memory_hours, initial_temperature_c=initial)
            profile = _load_profile(
                weather_at_bus, channels, param.base_load_mw, bus.suitability, rng,
                config=config, hour_of_day=hour_of_day, weekday=weekday,
                residual=residual_by_bus[bus_index],
                sector_weights=_sector_weights(land_use, row, col, config),
                effective_temperature=effective,
            )
            initial_temperature[bus_index] = initial
            diagnostics["load_effective_temperature_c"][:, bus_index] = effective
            diagnostics["load_log_residual"][:, bus_index] = residual_by_bus[bus_index]
            p_load[:, bus_index] = profile
            q_load[:, bus_index] = profile * float(np.tan(np.arccos(param.power_factor)))
        elif bus.kind == "wind_bus":
            pressure = weather_at_bus[:, channels["pressure"]] if "pressure" in channels else None
            density = hourly_weather.diagnostics.get("air_density_kg_m3")
            hub_wind, hub_density, density_source = _wind_conditions(
                weather_at_bus[:, channels["wind_speed"]], bus.capacity_mw,
                temperature=weather_at_bus[:, channels["temperature"]],
                pressure_hpa=pressure, air_density_kg_m3=None if density is None else density[:, row, col], config=config,
            )
            p_available[:, bus_index] = _wind_power_curve(hub_wind, hub_density, bus.capacity_mw, config)
            diagnostics["hub_wind_speed_mps"][:, bus_index] = hub_wind
            diagnostics["wind_air_density_kg_m3"][:, bus_index] = hub_density
            density_sources[str(bus.bus_id)] = density_source
        elif bus.kind == "pv_bus":
            poa = _plane_of_array_irradiance(weather_at_bus[:, channels["irradiance"]], hourly_weather.timestamps, float(latitudes[row, col]), config)
            p_available[:, bus_index] = _solar_available(
                poa,
                weather_at_bus[:, channels["temperature"]],
                bus.capacity_mw,
                wind_speed=weather_at_bus[:, channels["wind_speed"]], config=config,
            )
            diagnostics["pv_poa_w_m2"][:, bus_index] = poa
            diagnostics["pv_module_temperature_c"][:, bus_index] = _pv_module_temperature(poa, weather_at_bus[:, channels["temperature"]], weather_at_bus[:, channels["wind_speed"]], config)
        elif bus.kind == "thermal_bus":
            p_available[:, bus_index] = float(max(bus.capacity_mw, 0.0))

    p_scheduled[:] = p_available
    _dispatch_thermal(p_load, p_available, p_scheduled, buses, grid=grid, config=config)
    return SourceLoadForecastStore(
        timestamps=hourly_weather.timestamps.copy(),
        bus_ids=np.asarray([bus.bus_id for bus in buses], dtype=np.int32),
        bus_kinds=tuple(bus.kind for bus in buses),
        p_load_mw=p_load,
        p_gen_available_mw=p_available,
        p_gen_scheduled_mw=p_scheduled,
        q_load_mvar=q_load,
        source_channels=("p_load_mw", "p_gen_available_mw", "p_gen_scheduled_mw", "q_load_mvar"),
        nameplate_capacity_mw=np.asarray([bus.capacity_mw for bus in buses], dtype=np.float32),
        reference_load_mw=np.asarray([bus_params[bus.bus_id].base_load_mw for bus in buses], dtype=np.float32),
        initial_effective_temperature_c=initial_temperature,
        weather_sample_row=rows, weather_sample_col=cols,
        diagnostics=diagnostics,
        metadata={
            "schema_version": "source_load_v1", "mode": "exogenous_realization",
            "grid_source": "configured_grid" if grid_was_provided else "legacy_default_grid_assumption",
            "cell_size_km": grid.cell_size_km, "latitude_center_degrees": grid.latitude_center_degrees,
            "weather_grid_shape": list(shape),
            "weather_sampling": "nearest_declared_cell_no_coordinate_clipping",
            "wind_density_source_by_bus_id": density_sources,
            "wind_density_height_mode": config.wind_density_height_mode,
            "wind_rated_speed_semantics": "nominal_at_reference_density; safety_thresholds_use_physical_hub_wind",
            "pv_radiation_chain": "GHI_already_contains_cloud_effect_then_POA_Faiman_AC; no_second_cloud_factor",
            "load_initial_temperature_mode": config.load_initial_temperature_mode,
            "load_initial_temperature_time_support": "boundary_before_first_hour; first_hour_mode_is_explicit_cold_start",
            "initial_temperature_applicability": ["load_bus"],
            "load_design_capacity_semantics": "design_peak_reference; requested_load_is_not_clipped",
            "random_residual_alignment": "sorted_load_bus_ids_same_time_axis_same_rng_for_paired_interventions",
            "calendar_semantics": "calendar_start_date_is_weekday_prior; solar_day_comes_from_weather_365_day_climatology",
            "diagnostic_applicability": {
                "hub_wind_speed_mps": ["wind_bus"], "wind_air_density_kg_m3": ["wind_bus"],
                "pv_poa_w_m2": ["pv_bus"], "pv_module_temperature_c": ["pv_bus"],
                "load_effective_temperature_c": ["load_bus"], "load_log_residual": ["load_bus"],
            },
            "nonapplicable_diagnostic_value": "zero_with_bus_kind_mask; not_a_measurement",
            "operation_results": "delivered_power_curtailment_unserved_load_are_downstream_F_results_not_generated_here",
        },
    )


def _load_profile(
    weather_at_bus: np.ndarray,
    channels: dict[str, int],
    base_load_mw: float,
    suitability: float,
    rng: np.random.Generator,
    *,
    config: SourceLoadConfig | None = None,
    hour_of_day: np.ndarray | None = None,
    weekday: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    sector_weights: np.ndarray | None = None,
    effective_temperature: np.ndarray | None = None,
) -> np.ndarray:
    """Reference weekly mean + sector schedules + causal degree-hour response.

    Suitability is a siting score, so it must not rescale an already assigned
    demand a second time. `base_load_mw` denotes non-HVAC reference mean demand.
    """
    del suitability
    config = config or SourceLoadConfig()
    hours = weather_at_bus.shape[0]
    if hour_of_day is None or weekday is None:
        hour_of_day, weekday = _calendar_features(np.arange(hours), config.calendar_start_date)
    weights = _sector_weights(None, 0, 0, config) if sector_weights is None else sector_weights
    weekly_hours = np.arange(168) % 24 + 0.5
    weekly_days = np.arange(168) // 24
    reference = _sector_schedules(weekly_hours, weekly_days).mean(axis=0)
    schedule = _sector_schedules(np.asarray(hour_of_day) + 0.5, np.asarray(weekday)) / reference
    profile = schedule @ weights
    raw_temperature = weather_at_bus[:, channels["temperature"]]
    temperature = (_effective_temperature(raw_temperature, config.load_thermal_memory_hours, initial_temperature_c=_initial_load_temperature(raw_temperature, config))
                   if effective_temperature is None else np.asarray(effective_temperature))
    if temperature.shape != (hours,) or not np.isfinite(temperature).all():
        raise ValueError("Effective load temperature must be finite and match the time axis")
    heating = np.maximum(config.load_heating_balance_c - temperature, 0.0)
    cooling = np.maximum(temperature - config.load_cooling_balance_c, 0.0)
    # Industry retains process load; building end uses carry the HVAC response.
    hvac_activity = schedule @ (weights * np.asarray([1.0, 1.0, 0.15]))
    profile += hvac_activity * (
        config.load_heating_sensitivity_per_c * heating
        + config.load_cooling_sensitivity_per_c * cooling
    )
    if residual is None:
        residual = _spatial_load_residuals(hours, np.zeros((1, 2)), rng, config)[:, 0]
    # Lognormal mean correction preserves E[demand | weather, calendar].
    multiplier = np.exp(residual - 0.5 * config.load_residual_std_fraction**2)
    return (max(float(base_load_mw), 0.0) * profile * multiplier).astype(np.float32)


def _calendar_features(timestamps: np.ndarray, calendar_start_date: str) -> tuple[np.ndarray, np.ndarray]:
    """Use the configured representative calendar with local solar-hour bins."""
    stamps = np.asarray(timestamps)
    if np.issubdtype(stamps.dtype, np.datetime64):
        absolute_hours = stamps.astype("datetime64[h]").astype(np.int64)
        if np.isnat(stamps).any():
            raise ValueError("Timestamps must not contain NaT")
    else:
        if not np.isfinite(stamps).all() or not np.equal(stamps, np.floor(stamps)).all():
            raise ValueError("Timestamps must be integer hours")
        start = np.datetime64(calendar_start_date, "h")
        if np.isnat(start):
            raise ValueError("calendar_start_date must be a valid date")
        absolute_hours = start.astype(np.int64) + stamps.astype(np.int64)
    if stamps.size > 1 and not np.all(np.diff(absolute_hours) == 1):
        raise ValueError("Timestamps must be consecutive hourly intervals")
    # Unix day zero was Thursday; Monday=0, Sunday=6.
    return absolute_hours % 24, (absolute_hours // 24 + 3) % 7


def _sector_schedules(hour: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    weekend = weekday >= 5
    morning = np.exp(-0.5 * (_cyclic_hour_distance(hour, np.where(weekend, 9.0, 7.5)) / 1.8) ** 2)
    evening = np.exp(-0.5 * (_cyclic_hour_distance(hour, 19.0) / 2.5) ** 2)
    business = 1.0 / (1.0 + np.exp(-(hour - 8.0))) / (1.0 + np.exp(hour - 18.0))
    residential = 0.58 + 0.26 * morning + 0.45 * evening + 0.08 * weekend
    commercial = 0.26 + 0.90 * business * np.where(weekend, 0.42, 1.0)
    industrial = np.where(weekend, 0.74, 0.94) + 0.10 * business
    return np.stack((residential, commercial, industrial), axis=-1)


def _sector_weights(land_use: LandUseState | None, row: int, col: int, config: SourceLoadConfig) -> np.ndarray:
    weights = np.asarray([
        config.load_residential_fraction, config.load_commercial_fraction, config.load_industrial_fraction,
    ], dtype=np.float64)
    if land_use is not None:
        # Land-use maps are relative intensities, not metered electricity shares.
        local = np.asarray([land_use.residential[row, col], land_use.commercial[row, col], land_use.industrial[row, col]])
        if np.any(local < 0.0) or not np.isfinite(local).all():
            raise ValueError("Land-use intensities must be finite and nonnegative")
        weighted = weights * local
        if weighted.sum() > 0:
            weights = weighted
    return weights / weights.sum()


def _initial_load_temperature(temperature: np.ndarray, config: SourceLoadConfig) -> float:
    return float(temperature[0] if config.load_initial_temperature_mode == "first_hour" else config.load_initial_temperature_c)


def _effective_temperature(temperature: np.ndarray, memory_hours: float, *, initial_temperature_c: float | None = None, step_hours: float = 1.0) -> np.ndarray:
    """E/S: first-order building-memory proxy, not a closed building heat budget.

    Initial state precedes the first interval. Every sample, including index 0,
    is the updated interval-end state. The main pipeline has one-hour steps.
    """
    values = np.asarray(temperature, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Thermal-memory input must be a finite nonempty time series")
    if not np.isfinite(memory_hours) or memory_hours < 0:
        raise ValueError("Thermal memory must be finite and nonnegative")
    positive_finite(step_hours, "thermal step_hours")
    previous = float(values[0] if initial_temperature_c is None else initial_temperature_c)
    if not np.isfinite(previous) or previous <= -273.15 or np.any(values <= -273.15):
        raise ValueError("Thermal temperatures must be finite and exceed absolute zero")
    effective = np.empty_like(values)
    rho = np.exp(-step_hours / memory_hours) if memory_hours > 0 else 0.0
    for index, value in enumerate(values):
        previous = rho * previous + (1.0 - rho) * value
        effective[index] = previous
    return effective


def _spatial_load_residuals(hours: int, coordinates_km: np.ndarray, rng: np.random.Generator, config: SourceLoadConfig) -> np.ndarray:
    """Stationary Gaussian AR(1) with a distance-based covariance and nugget."""
    count = len(coordinates_km)
    if hours == 0 or count == 0:
        return np.zeros((hours, count), dtype=np.float64)
    distance = np.linalg.norm(coordinates_km[:, None, :] - coordinates_km[None, :, :], axis=-1)
    common = config.load_common_variance_fraction
    covariance = common * np.exp(-distance / config.load_spatial_correlation_km) + (1.0 - common) * np.eye(count)
    # Eigendecomposition also supports coincident buses and common fraction=1.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    residual = rng.normal(size=(hours, count)) @ factor.T
    rho = config.load_residual_ar1
    innovation_scale = np.sqrt(1.0 - rho**2)
    for index in range(1, hours):
        residual[index] = rho * residual[index - 1] + innovation_scale * residual[index]
    return residual * config.load_residual_std_fraction


def _cyclic_hour_distance(hour_of_day: np.ndarray, center_hour: float | np.ndarray) -> np.ndarray:
    distance = np.abs(hour_of_day - center_hour)
    return np.minimum(distance, 24.0 - distance)


def _wind_available(
    wind_speed: np.ndarray, capacity_mw: float, *, temperature: np.ndarray | None = None,
    pressure_hpa: np.ndarray | None = None, config: SourceLoadConfig | None = None,
    air_density_kg_m3: np.ndarray | None = None,
) -> np.ndarray:
    """Reference-height wind -> hub wind -> continuous density-scaled curve.

    Threshold defaults refer to NREL's 5 MW turbine; the interpolating cubic
    is a reduced model, not its aeroelastic or manufacturer power curve.
    """
    config = config or SourceLoadConfig()
    hub_wind, density, _ = _wind_conditions(wind_speed, capacity_mw, temperature=temperature, pressure_hpa=pressure_hpa, air_density_kg_m3=air_density_kg_m3, config=config)
    return _wind_power_curve(hub_wind, density, capacity_mw, config)


def _wind_conditions(wind_speed: np.ndarray, capacity_mw: float, *, temperature: np.ndarray | None = None, pressure_hpa: np.ndarray | None = None, air_density_kg_m3: np.ndarray | None = None, config: SourceLoadConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """Sampled weather rho is authoritative. Only legacy input uses dry fallback.

    E: neutral power-law shear. P + engineering reduction: isothermal virtual
    temperature column gives rho_h=rho_s exp[-g*rho_s*z_h/(100*p_s)]. Surface
    pressure is at terrain level, so z_h is full AGL hub height, not z_h-z_wind.
    """
    _validate_config(config)
    _finite_nonnegative(np.asarray(capacity_mw), "wind capacity_mw")
    speed = np.asarray(wind_speed, dtype=np.float64)
    _finite_nonnegative(speed, "wind_speed")
    hub_wind = speed * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_exponent
    pressure = None if pressure_hpa is None else _matching_series(pressure_hpa, speed.shape, "pressure_hpa", positive=True)
    if air_density_kg_m3 is not None:
        density = _matching_series(air_density_kg_m3, speed.shape, "air_density_kg_m3", positive=True)
        source = "weather_moist_air_diagnostic"
    elif config.wind_density_fallback == "error":
        raise ValueError("Wind requires the weather air_density_kg_m3 diagnostic; legacy fallback is disabled")
    elif pressure is not None and temperature is not None:
        kelvin = _matching_series(temperature, speed.shape, "temperature") + 273.15
        if np.any(kelvin <= 0):
            raise ValueError("Wind temperature must exceed absolute zero; no temperature clipping")
        density = pressure * 100.0 / (287.05 * kelvin)
        source = "legacy_dry_air_from_surface_pressure_temperature"
    else:
        density = np.full_like(hub_wind, config.wind_reference_density_kg_m3)
        source = "legacy_reference_density_at_hub"
    if config.wind_density_height_mode == "isothermal_surface_to_hub" and source != "legacy_reference_density_at_hub":
        if pressure is None:
            raise ValueError("Surface-to-hub density requires surface pressure; choose surface_proxy explicitly if unavailable")
        density = density * np.exp(-9.80665 * density * config.wind_hub_height_m / (100.0 * pressure))
    return hub_wind, density, source


def _wind_power_curve(hub_wind: np.ndarray, density: np.ndarray, capacity_mw: float, config: SourceLoadConfig) -> np.ndarray:
    # E/S engineering curve, not an OEM aeroelastic model. Density scales the
    # cubic region continuously; rated wind is nominal at reference density.
    fraction = np.clip(
        (density / config.wind_reference_density_kg_m3) * (hub_wind**3 - config.wind_cut_in_mps**3)
        / (config.wind_rated_mps**3 - config.wind_cut_in_mps**3), 0.0, 1.0,
    )
    # Shutdown is a wind-speed safety threshold, not a density threshold.
    fraction = np.where((hub_wind > config.wind_cut_in_mps) & (hub_wind < config.wind_cut_out_mps), fraction, 0.0)
    return (float(capacity_mw) * (1.0 - config.wind_system_loss_fraction) * fraction).astype(np.float32)


def _plane_of_array_irradiance(ghi: np.ndarray, timestamps: np.ndarray, latitude: float, config: SourceLoadConfig) -> np.ndarray:
    """Erbs hourly partition and isotropic-sky transposition of hourly mean GHI.

    Twelve solar positions per bin retain sunrise/sunset partial hours. Beam
    normal irradiance is capped at extraterrestrial normal radiation; the
    excess is assigned to diffuse so horizontal-plane energy is conserved.
    """
    _finite_nonnegative(np.asarray(ghi), "GHI")
    stamps = np.asarray(timestamps)
    if np.shape(ghi) != stamps.shape or stamps.ndim != 1:
        raise ValueError("PV GHI and timestamps must share one time axis")
    if np.issubdtype(stamps.dtype, np.datetime64):
        day = stamps.astype("datetime64[D]")
        doy = (day - day.astype("datetime64[Y]")).astype(int)
        hod = (stamps.astype("datetime64[h]") - day).astype("timedelta64[h]").astype(int)
    else:
        doy, hod = stamps.astype(np.int64) // 24 % 365, stamps.astype(np.int64) % 24
    tilt, azimuth = np.deg2rad([config.pv_tilt_degrees, config.pv_azimuth_degrees])
    normal = np.asarray([np.sin(tilt) * np.sin(azimuth), np.sin(tilt) * np.cos(azimuth), np.cos(tilt)])
    poa = np.zeros_like(ghi, dtype=np.float64)
    for day_number in np.unique(doy):
        indices = np.flatnonzero(doy == day_number)
        hourly_toa = extraterrestrial_hourly_irradiance(np.asarray(latitude), float(day_number))
        extraterrestrial_normal = 1366.6666666667 * (1.0 + 0.033 * np.cos(2.0 * np.pi * (day_number + 1.0) / 365.0))
        for index in indices:
            horizontal = float(ghi[index])
            toa = float(hourly_toa[hod[index]])
            if horizontal == 0.0 or toa <= 1e-8:
                continue
            kt = np.clip(horizontal / toa, 0.0, 1.0)
            diffuse_fraction = (
                1.0 - 0.09 * kt if kt <= 0.22 else
                0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
                if kt <= 0.80 else 0.165
            )
            sky_up = []
            beam_incidence = []
            for sub_hour in (np.arange(12) + 0.5) / 12.0 + hod[index]:
                direction = np.asarray(solar_direction(np.asarray(latitude), float(day_number), float(sub_hour)))
                sky_up.append(max(float(direction[2]), 0.0))
                beam_incidence.append(max(float(normal @ direction), 0.0) if direction[2] > 0.0 else 0.0)
            mean_cos = float(np.mean(sky_up))
            if mean_cos <= 1e-8:
                poa[index] = horizontal * (1.0 + np.cos(tilt)) / 2.0
                continue
            diffuse = np.clip(diffuse_fraction, 0.0, 1.0) * horizontal
            dni = min((horizontal - diffuse) / mean_cos, extraterrestrial_normal)
            diffuse = horizontal - dni * mean_cos
            poa[index] = (
                dni * np.mean(beam_incidence) + diffuse * (1.0 + np.cos(tilt)) / 2.0
                + horizontal * config.pv_ground_albedo * (1.0 - np.cos(tilt)) / 2.0
            )
    return poa.astype(np.float32)


def _solar_available(
    irradiance: np.ndarray, temperature: np.ndarray, capacity_mw: float,
    *, wind_speed: np.ndarray | None = None, config: SourceLoadConfig | None = None,
) -> np.ndarray:
    """POA -> Faiman module temperature -> PVWatts DC -> clipped AC MW.

    Capacity is the grid-side AC nameplate. Faiman temperature approximates
    cell temperature for this reduced model; roof mounting needs new U values.
    """
    config = config or SourceLoadConfig()
    _finite_nonnegative(np.asarray(capacity_mw), "PV capacity_mw")
    poa = np.asarray(irradiance, dtype=np.float64)
    module_temperature = _pv_module_temperature(poa, temperature, wind_speed, config)
    dc = (
        float(capacity_mw) * config.pv_dc_ac_ratio * poa / 1000.0
        * np.maximum(1.0 + config.pv_temperature_coefficient_per_c * (module_temperature - 25.0), 0.0)
        * (1.0 - config.pv_system_loss_fraction)
    )
    return np.clip(dc * config.pv_inverter_efficiency, 0.0, float(capacity_mw)).astype(np.float32)


def _pv_module_temperature(poa: np.ndarray, temperature: np.ndarray, wind_speed: np.ndarray | None, config: SourceLoadConfig) -> np.ndarray:
    """E: Faiman heat-loss fit; S: module-height wind power-law approximation.

    POA already includes cloud effects present in GHI. No cloud factor enters
    this energy-conversion chain a second time; no full cell heat storage.
    """
    _validate_config(config)
    poa = np.asarray(poa, dtype=np.float64)
    _finite_nonnegative(poa, "POA")
    ambient = _matching_series(temperature, poa.shape, "PV ambient temperature")
    if np.any(ambient <= -273.15):
        raise ValueError("PV ambient temperature must exceed absolute zero")
    if wind_speed is None:
        module_wind = 1.0  # Preserved direct-helper legacy default, unused by entry.
    else:
        wind = _matching_series(wind_speed, poa.shape, "PV wind_speed")
        _finite_nonnegative(wind, "PV wind_speed")
        module_wind = wind * (config.pv_module_height_m / config.wind_reference_height_m) ** config.pv_wind_shear_exponent
    return ambient + poa / (config.pv_heat_loss_constant + config.pv_heat_loss_wind * module_wind)


def _finite_nonnegative(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


def _matching_series(values: np.ndarray, shape: tuple[int, ...], name: str, *, positive: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} must be finite, {'positive and ' if positive else ''}match the time axis")
    return result


def _validate_assets(buses: tuple[object, ...], parameters: dict[int, object]) -> None:
    for bus in buses:
        if bus.kind not in {"load_bus", "wind_bus", "pv_bus", "thermal_bus", "transit_bus"}:
            raise ValueError(f"Unsupported bus kind {bus.kind!r}")
        param = parameters[bus.bus_id]
        _finite_nonnegative(np.asarray([bus.capacity_mw, param.base_load_mw, param.q_capacity_mvar]), "asset capacity/reference load")
        if not np.isfinite(param.p_capacity_mw) or not np.isclose(abs(param.p_capacity_mw), bus.capacity_mw, rtol=1e-6, atol=1e-6):
            raise ValueError("Electrical and topology nameplate capacities must match")
        if param.kind != bus.kind or (bus.kind == "load_bus" and param.p_capacity_mw > 0) or (bus.kind != "load_bus" and param.p_capacity_mw < 0):
            raise ValueError("Electrical bus kind and capacity sign must match topology")
        if not np.isfinite(param.power_factor) or not 0 < param.power_factor <= 1:
            raise ValueError("Power factor must lie in (0,1]; no clipping")


def _validate_weather_physics(weather: WeatherStore, channels: dict[str, int]) -> None:
    if np.any(weather.dynamic[:, channels["temperature"]] <= -273.15):
        raise ValueError("Weather temperature must exceed absolute zero")
    for name in ("wind_speed", "irradiance", "precipitation"):
        if name in channels:
            _finite_nonnegative(weather.dynamic[:, channels[name]], name)
    for name in ("humidity", "cloud"):
        if name in channels and np.any((weather.dynamic[:, channels[name]] < 0) | (weather.dynamic[:, channels[name]] > 1)):
            raise ValueError(f"Weather {name} must lie in [0,1]")
    if "pressure" in channels and np.any(weather.dynamic[:, channels["pressure"]] <= 0):
        raise ValueError("Weather pressure must be positive")
    for name in ("air_density_kg_m3", "sea_level_pressure_hpa"):
        if name in weather.diagnostics and np.any(weather.diagnostics[name] <= 0):
            raise ValueError(f"Weather diagnostic {name} must be positive")
    if "specific_humidity_kg_kg" in weather.diagnostics:
        humidity = weather.diagnostics["specific_humidity_kg_kg"]
        if np.any((humidity < 0) | (humidity >= 1)):
            raise ValueError("Weather specific humidity must lie in [0,1)")


def _validate_config(config: SourceLoadConfig) -> None:
    """Compatibility wrapper; the configuration owns the only validator."""
    config.validate()


def _dispatch_thermal(
    p_load: np.ndarray,
    p_available: np.ndarray,
    p_scheduled: np.ndarray,
    buses: tuple[object, ...],
    *, grid: WorldGridConfig | None = None, config: SourceLoadConfig | None = None,
) -> None:
    bus_kinds = tuple(str(bus.kind) for bus in buses)
    thermal_indices = [index for index, kind in enumerate(bus_kinds) if kind == "thermal_bus"]
    load_indices = [index for index, kind in enumerate(bus_kinds) if kind == "load_bus"]
    renewable_indices = [index for index, kind in enumerate(bus_kinds) if kind in {"wind_bus", "pv_bus"}]
    p_scheduled[:, thermal_indices] = 0.0
    if not thermal_indices:
        return
    total_load = p_load.sum(axis=1)
    renewable = p_available[:, renewable_indices].sum(axis=1) if renewable_indices else np.zeros(p_load.shape[0], dtype=np.float32)
    residual = np.maximum(total_load - renewable, 0.0)
    thermal_capacity = p_available[:, thermal_indices]
    config = config or SourceLoadConfig()
    grid = grid or WorldGridConfig()
    locality = _thermal_locality_weights(buses, thermal_indices, load_indices, renewable_indices, half_distance_km=config.thermal_dispatch_half_distance_km, cell_size_km=grid.cell_size_km)
    for hour in range(p_load.shape[0]):
        local_load = locality[0] @ p_load[hour, load_indices] if load_indices else np.zeros(len(thermal_indices))
        local_renewable = (
            locality[1] @ p_available[hour, renewable_indices]
            if renewable_indices
            else np.zeros(len(thermal_indices))
        )
        local_need = np.maximum(local_load - local_renewable, 0.0)
        capacities = thermal_capacity[hour].astype(np.float64, copy=False)
        priority = capacities * (0.15 + 0.85 * local_need / max(float(local_need.max()), 1e-6))
        p_scheduled[hour, thermal_indices] = _allocate_with_capacity_limit(
            float(residual[hour]),
            capacities,
            priority,
        )


def _thermal_locality_weights(
    buses: tuple[object, ...],
    thermal_indices: list[int],
    load_indices: list[int],
    renewable_indices: list[int],
    half_distance_cells: float | None = None,
    *, half_distance_km: float = 24.0, cell_size_km: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    # S: optional old positional cell count is translated explicitly; main
    # dispatch always supplies physical km. Not an electrical-distance model.
    positive_finite(cell_size_km, "cell_size_km")
    distance_km = half_distance_km if half_distance_cells is None else half_distance_cells * cell_size_km
    decay = positive_finite(distance_km, "thermal half distance km") / np.log(2.0)
    thermal_coords = np.asarray([(buses[index].row, buses[index].col) for index in thermal_indices], dtype=np.float64) * cell_size_km

    def weights(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros((len(thermal_indices), 0), dtype=np.float64)
        coords = np.asarray([(buses[index].row, buses[index].col) for index in indices], dtype=np.float64) * cell_size_km
        distance = np.linalg.norm(thermal_coords[:, None, :] - coords[None, :, :], axis=2)
        return np.exp(-distance / decay)

    return weights(load_indices), weights(renewable_indices)


def _allocate_with_capacity_limit(target: float, capacities: np.ndarray, priority: np.ndarray) -> np.ndarray:
    allocation = np.zeros_like(capacities, dtype=np.float64)
    remaining = min(max(float(target), 0.0), float(np.sum(capacities)))
    available = capacities > 1e-9
    while remaining > 1e-7 and np.any(available):
        weights = np.where(available, np.maximum(priority, 1e-9), 0.0)
        proposal = remaining * weights / max(float(weights.sum()), 1e-9)
        headroom = np.maximum(capacities - allocation, 0.0)
        addition = np.minimum(proposal, headroom)
        allocation += addition
        remaining -= float(addition.sum())
        available &= headroom - addition > 1e-7
    return allocation.astype(np.float32)
