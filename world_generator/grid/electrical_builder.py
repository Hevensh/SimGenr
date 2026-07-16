from __future__ import annotations

import numpy as np

from world_generator.core.datatypes import (
    BranchElectricalParam,
    BusElectricalParam,
    GridElectricalState,
    GridBus,
    RefinedGridTopologyState,
)


GENERATOR_KINDS = {"wind_bus", "pv_bus", "thermal_bus"}


def build_grid_electrical(refined_topology: RefinedGridTopologyState) -> GridElectricalState:
    bus_by_id = {bus.bus_id: bus for bus in refined_topology.refined_buses}
    bus_params = tuple(_bus_param(bus) for bus in refined_topology.refined_buses)
    branch_params = tuple(_branch_param(edge, bus_by_id) for edge in refined_topology.refined_edges)
    return GridElectricalState(bus_params=bus_params, branch_params=branch_params)


def _bus_param(bus: GridBus) -> BusElectricalParam:
    nominal_kv = _bus_voltage_kv(bus)
    if bus.kind == "load_bus":
        p_capacity = -float(bus.capacity_mw)
        base_load = float(0.62 * bus.capacity_mw)
        power_factor = 0.94
        control_mode = "PQ_LOAD"
    elif bus.kind == "thermal_bus":
        p_capacity = float(bus.capacity_mw)
        base_load = 0.0
        power_factor = 0.90
        control_mode = "PV_GEN"
    elif bus.kind in {"wind_bus", "pv_bus"}:
        p_capacity = float(bus.capacity_mw)
        base_load = 0.0
        power_factor = 0.98
        control_mode = "PQ_RENEWABLE"
    else:
        p_capacity = 0.0
        base_load = 0.0
        power_factor = 1.0
        control_mode = "TRANSIT"

    q_capacity = abs(p_capacity) * float(np.tan(np.arccos(np.clip(power_factor, 0.1, 1.0))))
    return BusElectricalParam(
        bus_id=bus.bus_id,
        kind=bus.kind,
        nominal_kv=nominal_kv,
        p_capacity_mw=p_capacity,
        q_capacity_mvar=q_capacity,
        base_load_mw=base_load,
        power_factor=power_factor,
        voltage_setpoint_pu=1.02 if bus.kind in GENERATOR_KINDS else 1.0,
        control_mode=control_mode,
    )


def _branch_param(edge: object, bus_by_id: dict[int, GridBus]) -> BranchElectricalParam:
    from_bus = bus_by_id[int(edge.from_bus)]
    to_bus = bus_by_id[int(edge.to_bus)]
    nominal_kv = max(_bus_voltage_kv(from_bus), _bus_voltage_kv(to_bus), _line_voltage_kv(float(edge.length_km), from_bus, to_bus))
    r_per_km, x_per_km, b_per_km = _line_unit_params(nominal_kv)
    rating = _line_rating_mva(nominal_kv, from_bus, to_bus, bool(edge.is_redundant))
    return BranchElectricalParam(
        edge_id=int(edge.edge_id),
        from_bus=int(edge.from_bus),
        to_bus=int(edge.to_bus),
        nominal_kv=nominal_kv,
        length_km=float(edge.length_km),
        r_ohm=float(r_per_km * edge.length_km),
        x_ohm=float(x_per_km * edge.length_km),
        b_us=float(b_per_km * edge.length_km),
        rate_mva=rating,
        is_redundant=bool(edge.is_redundant),
    )


def _bus_voltage_kv(bus: GridBus) -> float:
    if bus.kind == "thermal_bus" or bus.capacity_mw >= 160.0:
        return 220.0
    if bus.kind in {"wind_bus", "pv_bus"} and bus.capacity_mw >= 90.0:
        return 220.0
    if bus.kind == "transit_bus":
        return 110.0
    return 110.0


def _line_voltage_kv(length_km: float, from_bus: GridBus, to_bus: GridBus) -> float:
    endpoint_capacity = max(float(from_bus.capacity_mw), float(to_bus.capacity_mw))
    if length_km >= 6.0 or endpoint_capacity >= 150.0:
        return 220.0
    return 110.0


def _line_unit_params(nominal_kv: float) -> tuple[float, float, float]:
    if nominal_kv >= 200.0:
        return 0.075, 0.32, 3.3
    return 0.12, 0.40, 2.4


def _line_rating_mva(nominal_kv: float, from_bus: GridBus, to_bus: GridBus, is_redundant: bool) -> float:
    base = 260.0 if nominal_kv >= 200.0 else 120.0
    endpoint_capacity = max(float(from_bus.capacity_mw), float(to_bus.capacity_mw))
    capacity_floor = 1.35 * endpoint_capacity if endpoint_capacity > 0.0 else 0.0
    redundancy_factor = 0.92 if is_redundant else 1.0
    return float(max(base, capacity_floor) * redundancy_factor)
