from __future__ import annotations

import numpy as np

from world_generator.core.datatypes import GridElectricalState, PowerFlowStore, RefinedGridTopologyState, SourceLoadForecastStore


BASE_MVA = 100.0


def solve_dc_power_flow(
    forecast: SourceLoadForecastStore,
    refined_topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
) -> PowerFlowStore:
    buses = refined_topology.refined_buses
    branches = electrical.branch_params
    hours = forecast.p_load_mw.shape[0]
    bus_ids = np.asarray([bus.bus_id for bus in buses], dtype=np.int32)
    branch_ids = np.asarray([branch.edge_id for branch in branches], dtype=np.int32)
    bus_index = {int(bus_id): index for index, bus_id in enumerate(bus_ids)}
    slack_index = _choose_slack_bus_index(refined_topology)

    susceptance = _branch_susceptance_mw_per_rad(branches)
    b_matrix = _build_b_matrix(branches, bus_index, susceptance, len(buses))
    non_slack = np.asarray([index for index in range(len(buses)) if index != slack_index], dtype=np.int32)
    reduced_b = b_matrix[np.ix_(non_slack, non_slack)]

    bus_angle = np.zeros((hours, len(buses)), dtype=np.float32)
    bus_injection = np.zeros((hours, len(buses)), dtype=np.float32)
    served_load = np.zeros_like(forecast.p_load_mw, dtype=np.float32)
    dispatched_generation = np.zeros_like(forecast.p_gen_scheduled_mw, dtype=np.float32)
    unserved_load = np.zeros_like(forecast.p_load_mw, dtype=np.float32)
    curtailed_generation = np.zeros_like(forecast.p_gen_scheduled_mw, dtype=np.float32)
    line_flow = np.zeros((hours, len(branches)), dtype=np.float32)

    bus_kinds = np.asarray(forecast.bus_kinds)
    renewable_mask = np.isin(bus_kinds, ("wind_bus", "pv_bus"))
    generation_mask = forecast.p_gen_scheduled_mw.max(axis=0) > 0.0

    for hour in range(hours):
        load = forecast.p_load_mw[hour].astype(np.float64, copy=True)
        generation = forecast.p_gen_scheduled_mw[hour].astype(np.float64, copy=True)
        load, generation, unserved, curtailed = _balance_dispatch(load, generation, renewable_mask, generation_mask)
        p_injection = generation - load
        p_injection[slack_index] -= p_injection.sum()

        theta = np.zeros(len(buses), dtype=np.float64)
        if non_slack.size:
            theta[non_slack] = _solve_reduced_angles(reduced_b, p_injection[non_slack])

        bus_angle[hour] = theta.astype(np.float32)
        bus_injection[hour] = p_injection.astype(np.float32)
        served_load[hour] = load.astype(np.float32)
        dispatched_generation[hour] = generation.astype(np.float32)
        unserved_load[hour] = unserved.astype(np.float32)
        curtailed_generation[hour] = curtailed.astype(np.float32)
        line_flow[hour] = _line_flows(branches, bus_index, susceptance, theta)

    rates = np.asarray([max(float(branch.rate_mva), 1e-6) for branch in branches], dtype=np.float32)
    line_loading = np.abs(line_flow) / rates[None, :] if len(branches) else np.zeros_like(line_flow)
    return PowerFlowStore(
        timestamps=forecast.timestamps.copy(),
        bus_ids=bus_ids,
        branch_ids=branch_ids,
        bus_angle_rad=bus_angle,
        bus_p_injection_mw=bus_injection,
        served_load_mw=served_load,
        dispatched_generation_mw=dispatched_generation,
        unserved_load_mw=unserved_load,
        curtailed_generation_mw=curtailed_generation,
        line_flow_mw=line_flow,
        line_loading_ratio=line_loading.astype(np.float32),
        slack_bus_id=int(bus_ids[slack_index]),
    )


def _choose_slack_bus_index(refined_topology: RefinedGridTopologyState) -> int:
    generator_priority = {"thermal_bus": 3, "pv_bus": 2, "wind_bus": 2, "transit_bus": 1, "load_bus": 0}
    scores = [
        (generator_priority.get(bus.kind, 0), float(bus.capacity_mw), -index)
        for index, bus in enumerate(refined_topology.refined_buses)
    ]
    return int(max(range(len(scores)), key=lambda index: scores[index]))


def _branch_susceptance_mw_per_rad(branches: tuple[object, ...]) -> np.ndarray:
    values = []
    for branch in branches:
        z_base_ohm = float(branch.nominal_kv) ** 2 / BASE_MVA
        x_pu = max(float(branch.x_ohm) / max(z_base_ohm, 1e-6), 1e-5)
        values.append(BASE_MVA / x_pu)
    return np.asarray(values, dtype=np.float64)


def _build_b_matrix(
    branches: tuple[object, ...],
    bus_index: dict[int, int],
    susceptance: np.ndarray,
    bus_count: int,
) -> np.ndarray:
    matrix = np.zeros((bus_count, bus_count), dtype=np.float64)
    for branch, branch_b in zip(branches, susceptance):
        i = bus_index[int(branch.from_bus)]
        j = bus_index[int(branch.to_bus)]
        matrix[i, i] += branch_b
        matrix[j, j] += branch_b
        matrix[i, j] -= branch_b
        matrix[j, i] -= branch_b
    return matrix


def _balance_dispatch(
    load: np.ndarray,
    generation: np.ndarray,
    renewable_mask: np.ndarray,
    generation_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    served_load = load.copy()
    dispatched = generation.copy()
    unserved = np.zeros_like(load)
    curtailed = np.zeros_like(generation)
    total_load = float(load.sum())
    total_generation = float(generation.sum())
    imbalance = total_generation - total_load

    if imbalance > 1e-6:
        remaining = imbalance
        renewable_generation = dispatched * renewable_mask
        curtailed_renewable = _proportional_reduction(renewable_generation, remaining)
        dispatched -= curtailed_renewable
        curtailed += curtailed_renewable
        remaining -= float(curtailed_renewable.sum())
        if remaining > 1e-6:
            generator_generation = dispatched * generation_mask
            curtailed_generator = _proportional_reduction(generator_generation, remaining)
            dispatched -= curtailed_generator
            curtailed += curtailed_generator
    elif imbalance < -1e-6 and total_load > 1e-6:
        unserved = _proportional_reduction(served_load, -imbalance)
        served_load -= unserved

    return served_load, dispatched, unserved, curtailed


def _proportional_reduction(values: np.ndarray, target: float) -> np.ndarray:
    total = float(values.sum())
    if total <= 1e-9 or target <= 1e-9:
        return np.zeros_like(values)
    amount = min(float(target), total)
    return values * (amount / total)


def _solve_reduced_angles(reduced_b: np.ndarray, p_injection: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(reduced_b, p_injection)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(reduced_b, p_injection, rcond=None)[0]


def _line_flows(
    branches: tuple[object, ...],
    bus_index: dict[int, int],
    susceptance: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    flows = np.zeros(len(branches), dtype=np.float32)
    for index, (branch, branch_b) in enumerate(zip(branches, susceptance)):
        i = bus_index[int(branch.from_bus)]
        j = bus_index[int(branch.to_bus)]
        flows[index] = float(branch_b * (theta[i] - theta[j]))
    return flows
