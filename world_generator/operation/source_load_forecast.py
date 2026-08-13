from __future__ import annotations

import numpy as np

from world_generator.core.datatypes import GridElectricalState, RefinedGridTopologyState, SourceLoadForecastStore, WeatherStore


def generate_source_load_forecast(
    hourly_weather: WeatherStore,
    refined_topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    rng: np.random.Generator,
) -> SourceLoadForecastStore:
    buses = refined_topology.refined_buses
    bus_params = {item.bus_id: item for item in electrical.bus_params}
    hours = hourly_weather.dynamic.shape[0]
    bus_count = len(buses)
    p_load = np.zeros((hours, bus_count), dtype=np.float32)
    p_available = np.zeros((hours, bus_count), dtype=np.float32)
    p_scheduled = np.zeros((hours, bus_count), dtype=np.float32)
    q_load = np.zeros((hours, bus_count), dtype=np.float32)
    channels = {name: index for index, name in enumerate(hourly_weather.channel_names)}

    for bus_index, bus in enumerate(buses):
        row = int(np.clip(bus.row, 0, hourly_weather.dynamic.shape[2] - 1))
        col = int(np.clip(bus.col, 0, hourly_weather.dynamic.shape[3] - 1))
        weather_at_bus = hourly_weather.dynamic[:, :, row, col]
        param = bus_params[bus.bus_id]
        if bus.kind == "load_bus":
            profile = _load_profile(weather_at_bus, channels, param.base_load_mw, bus.suitability, rng)
            p_load[:, bus_index] = profile
            q_load[:, bus_index] = profile * float(np.tan(np.arccos(np.clip(param.power_factor, 0.1, 1.0))))
        elif bus.kind == "wind_bus":
            p_available[:, bus_index] = _wind_available(weather_at_bus[:, channels["wind_speed"]], bus.capacity_mw)
        elif bus.kind == "pv_bus":
            p_available[:, bus_index] = _solar_available(
                weather_at_bus[:, channels["irradiance"]],
                weather_at_bus[:, channels["temperature"]],
                bus.capacity_mw,
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
) -> np.ndarray:
    hours = weather_at_bus.shape[0]
    hour_of_day = np.arange(hours, dtype=np.float32) % 24.0
    morning = np.exp(-(_cyclic_hour_distance(hour_of_day, 8.0) ** 2) / 18.0)
    evening = np.exp(-(_cyclic_hour_distance(hour_of_day, 19.0) ** 2) / 14.0)
    business = np.exp(-(_cyclic_hour_distance(hour_of_day, 14.0) ** 2) / 32.0)
    temperature = weather_at_bus[:, channels["temperature"]]
    cooling = np.clip((temperature - 24.0) / 12.0, 0.0, 1.2)
    heating = np.clip((12.0 - temperature) / 10.0, 0.0, 0.8)
    small_noise = _correlated_hourly_noise(hours, rng, 0.018)
    daily_scale = _smooth_daily_load_scale(hours, rng)
    profile = (
        0.64
        + 0.16 * morning
        + 0.24 * evening
        + 0.12 * business
        + 0.18 * cooling
        + 0.10 * heating
        + small_noise
    )
    profile *= daily_scale
    profile *= 0.92 + 0.18 * float(np.clip(suitability, 0.0, 1.0))
    return np.clip(base_load_mw * profile, 0.08 * base_load_mw, None).astype(np.float32)


def _cyclic_hour_distance(hour_of_day: np.ndarray, center_hour: float) -> np.ndarray:
    distance = np.abs(hour_of_day - float(center_hour))
    return np.minimum(distance, 24.0 - distance)


def _correlated_hourly_noise(hours: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=hours).astype(np.float32)
    for index in range(1, hours):
        noise[index] = 0.78 * noise[index - 1] + 0.22 * noise[index]
    return noise


def _smooth_daily_load_scale(hours: int, rng: np.random.Generator) -> np.ndarray:
    days = int(np.ceil(hours / 24.0))
    anchors = rng.normal(1.0, 0.025, size=days + 3).astype(np.float32)
    anchors = (np.roll(anchors, 1) + 2.0 * anchors + np.roll(anchors, -1)) / 4.0
    anchor_hours = (np.arange(days + 3, dtype=np.float32) - 1.0) * 24.0
    hour_axis = np.arange(hours, dtype=np.float32)
    return np.interp(hour_axis, anchor_hours, anchors).astype(np.float32)


def _wind_available(wind_speed: np.ndarray, capacity_mw: float) -> np.ndarray:
    cut_in = 3.0
    rated = 11.0
    cut_out = 25.0
    normalized = np.clip((wind_speed - cut_in) / (rated - cut_in), 0.0, 1.0)
    speed_sensitive = normalized**2.0
    fraction = np.where(wind_speed < cut_out, speed_sensitive, 0.0)
    return (float(capacity_mw) * np.clip(fraction, 0.0, 1.0)).astype(np.float32)


def _solar_available(irradiance: np.ndarray, temperature: np.ndarray, capacity_mw: float) -> np.ndarray:
    performance_ratio = 0.86
    irradiance_fraction = np.clip(irradiance / 1000.0, 0.0, 1.05)
    temperature_derate = np.clip(1.0 - 0.004 * np.maximum(temperature - 25.0, 0.0), 0.82, 1.0)
    return (float(capacity_mw) * irradiance_fraction * performance_ratio * temperature_derate).astype(np.float32)


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
