from __future__ import annotations

import numpy as np

from world_generator.core.config import SourceLoadConfig, WorldGridConfig
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
    grid = grid or WorldGridConfig()
    _validate_config(config)
    if hourly_weather.time_unit != "hour":
        raise ValueError("Source/load generation requires hourly weather")
    if hourly_weather.dynamic.ndim != 4 or not np.isfinite(hourly_weather.dynamic).all():
        raise ValueError("Weather must be a finite [hour, channel, row, col] array")
    buses = refined_topology.refined_buses
    bus_params = {item.bus_id: item for item in electrical.bus_params}
    hours = hourly_weather.dynamic.shape[0]
    bus_count = len(buses)
    p_load = np.zeros((hours, bus_count), dtype=np.float32)
    p_available = np.zeros((hours, bus_count), dtype=np.float32)
    p_scheduled = np.zeros((hours, bus_count), dtype=np.float32)
    q_load = np.zeros((hours, bus_count), dtype=np.float32)
    channels = {name: index for index, name in enumerate(hourly_weather.channel_names)}
    required = {"temperature", "wind_speed", "irradiance"}
    if not required.issubset(channels):
        raise ValueError(f"Missing weather channels: {sorted(required - channels.keys())}")
    if hourly_weather.timestamps.shape != (hours,):
        raise ValueError("Weather timestamps must match the hourly dimension")
    hour_of_day, weekday = _calendar_features(hourly_weather.timestamps, config.calendar_start_date)
    load_indices = [i for i, bus in enumerate(buses) if bus.kind == "load_bus"]
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
        row = int(np.clip(bus.row, 0, hourly_weather.dynamic.shape[2] - 1))
        col = int(np.clip(bus.col, 0, hourly_weather.dynamic.shape[3] - 1))
        weather_at_bus = hourly_weather.dynamic[:, :, row, col]
        param = bus_params[bus.bus_id]
        if bus.kind == "load_bus":
            profile = _load_profile(
                weather_at_bus, channels, param.base_load_mw, bus.suitability, rng,
                config=config, hour_of_day=hour_of_day, weekday=weekday,
                residual=residual_by_bus[bus_index],
                sector_weights=_sector_weights(land_use, row, col, config),
            )
            p_load[:, bus_index] = profile
            q_load[:, bus_index] = profile * float(np.tan(np.arccos(np.clip(param.power_factor, 0.1, 1.0))))
        elif bus.kind == "wind_bus":
            pressure = weather_at_bus[:, channels["pressure"]] if "pressure" in channels else None
            p_available[:, bus_index] = _wind_available(
                weather_at_bus[:, channels["wind_speed"]], bus.capacity_mw,
                temperature=weather_at_bus[:, channels["temperature"]],
                pressure_hpa=pressure, config=config,
            )
        elif bus.kind == "pv_bus":
            p_available[:, bus_index] = _solar_available(
                _plane_of_array_irradiance(
                    weather_at_bus[:, channels["irradiance"]], hourly_weather.timestamps,
                    float(latitudes[row, col]), config,
                ),
                weather_at_bus[:, channels["temperature"]],
                bus.capacity_mw,
                wind_speed=weather_at_bus[:, channels["wind_speed"]], config=config,
            )
        elif bus.kind == "thermal_bus":
            p_available[:, bus_index] = float(max(bus.capacity_mw, 0.0))

    p_scheduled[:] = p_available
    _dispatch_thermal(p_load, p_available, p_scheduled, buses)
    return SourceLoadForecastStore(
        timestamps=hourly_weather.timestamps.copy(),
        bus_ids=np.asarray([bus.bus_id for bus in buses], dtype=np.int32),
        bus_kinds=tuple(bus.kind for bus in buses),
        p_load_mw=p_load,
        p_gen_available_mw=p_available,
        p_gen_scheduled_mw=p_scheduled,
        q_load_mvar=q_load,
        source_channels=("p_load_mw", "p_gen_available_mw", "p_gen_scheduled_mw", "q_load_mvar"),
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
    temperature = _effective_temperature(weather_at_bus[:, channels["temperature"]], config.load_thermal_memory_hours)
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


def _effective_temperature(temperature: np.ndarray, memory_hours: float) -> np.ndarray:
    effective = np.asarray(temperature, dtype=np.float64).copy()
    rho = np.exp(-1.0 / memory_hours) if memory_hours > 0 else 0.0
    for index in range(1, effective.size):
        effective[index] = rho * effective[index - 1] + (1.0 - rho) * temperature[index]
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
) -> np.ndarray:
    """10 m wind -> hub wind -> density-adjusted cubic engineering power curve.

    Threshold defaults refer to NREL's 5 MW turbine; the interpolating cubic
    is a reduced model, not its aeroelastic or manufacturer power curve.
    """
    config = config or SourceLoadConfig()
    hub_wind = np.maximum(wind_speed, 0.0) * (config.wind_hub_height_m / config.wind_reference_height_m) ** config.wind_shear_exponent
    rho = np.ones_like(hub_wind, dtype=np.float64) * 1.225
    if temperature is not None and pressure_hpa is not None:
        kelvin = np.maximum(np.asarray(temperature) + 273.15, 150.0)
        hub_pressure_pa = np.asarray(pressure_hpa) * 100.0 * np.exp(-9.80665 * config.wind_hub_height_m / (287.05 * kelvin))
        rho = hub_pressure_pa / (287.05 * kelvin)
    equivalent_wind = hub_wind * np.cbrt(np.maximum(rho, 0.0) / 1.225)
    fraction = np.clip(
        (equivalent_wind**3 - config.wind_cut_in_mps**3)
        / (config.wind_rated_mps**3 - config.wind_cut_in_mps**3), 0.0, 1.0,
    )
    # Shutdown is a wind-speed safety threshold, not a density threshold.
    fraction = np.where((hub_wind >= config.wind_cut_in_mps) & (hub_wind < config.wind_cut_out_mps), fraction, 0.0)
    return (max(float(capacity_mw), 0.0) * (1.0 - config.wind_system_loss_fraction) * fraction).astype(np.float32)


def _plane_of_array_irradiance(ghi: np.ndarray, timestamps: np.ndarray, latitude: float, config: SourceLoadConfig) -> np.ndarray:
    """Erbs hourly partition and isotropic-sky transposition of hourly mean GHI.

    Twelve solar positions per bin retain sunrise/sunset partial hours. Beam
    normal irradiance is capped at extraterrestrial normal radiation; the
    excess is assigned to diffuse so horizontal-plane energy is conserved.
    """
    stamps = np.asarray(timestamps)
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
            horizontal = max(float(ghi[index]), 0.0)
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
    poa = np.maximum(np.asarray(irradiance), 0.0)
    # Approximate module-height wind at 2 m from the 10 m weather channel.
    module_wind = 1.0 if wind_speed is None else np.maximum(wind_speed, 0.0) * (2.0 / config.wind_reference_height_m) ** config.wind_shear_exponent
    module_temperature = np.asarray(temperature) + poa / (config.pv_heat_loss_constant + config.pv_heat_loss_wind * module_wind)
    dc = (
        max(float(capacity_mw), 0.0) * config.pv_dc_ac_ratio * poa / 1000.0
        * np.maximum(1.0 + config.pv_temperature_coefficient_per_c * (module_temperature - 25.0), 0.0)
        * (1.0 - config.pv_system_loss_fraction)
    )
    return np.clip(dc * config.pv_inverter_efficiency, 0.0, max(float(capacity_mw), 0.0)).astype(np.float32)


def _validate_config(config: SourceLoadConfig) -> None:
    values = vars(config)
    if any(not np.isfinite(value) for value in values.values() if isinstance(value, (int, float))):
        raise ValueError("Source/load parameters must be finite")
    if not 0 <= config.wind_cut_in_mps < config.wind_rated_mps < config.wind_cut_out_mps:
        raise ValueError("Wind thresholds must satisfy 0 <= cut-in < rated < cut-out")
    for name in ("wind_reference_height_m", "wind_hub_height_m", "pv_dc_ac_ratio", "pv_heat_loss_constant", "load_spatial_correlation_km"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("wind_system_loss_fraction", "pv_system_loss_fraction", "load_common_variance_fraction"):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if not 0 < config.pv_inverter_efficiency <= 1 or not 0 <= config.pv_ground_albedo <= 1:
        raise ValueError("PV efficiency and ground albedo must be physical fractions")
    if not 0 <= config.pv_tilt_degrees <= 90 or not 0 <= config.pv_azimuth_degrees < 360:
        raise ValueError("PV tilt must be 0..90 degrees and azimuth 0..<360 degrees")
    if not -1 < config.load_residual_ar1 < 1:
        raise ValueError("Load AR(1) coefficient must be strictly inside (-1, 1)")
    for name in ("pv_heat_loss_wind", "load_thermal_memory_hours", "load_residual_std_fraction", "load_heating_sensitivity_per_c", "load_cooling_sensitivity_per_c", "load_residential_fraction", "load_commercial_fraction", "load_industrial_fraction"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if config.load_heating_balance_c > config.load_cooling_balance_c:
        raise ValueError("Heating balance temperature must not exceed cooling balance temperature")
    if config.load_residential_fraction + config.load_commercial_fraction + config.load_industrial_fraction <= 0:
        raise ValueError("At least one load sector fraction must be positive")


def _dispatch_thermal(
    p_load: np.ndarray,
    p_available: np.ndarray,
    p_scheduled: np.ndarray,
    buses: tuple[object, ...],
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
    locality = _thermal_locality_weights(buses, thermal_indices, load_indices, renewable_indices)
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
    half_distance_cells: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    thermal_coords = np.asarray([(buses[index].row, buses[index].col) for index in thermal_indices], dtype=np.float64)
    decay = max(float(half_distance_cells) / np.log(2.0), 1e-6)

    def weights(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros((len(thermal_indices), 0), dtype=np.float64)
        coords = np.asarray([(buses[index].row, buses[index].col) for index in indices], dtype=np.float64)
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
