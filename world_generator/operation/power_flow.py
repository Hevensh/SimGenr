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
    if not buses or len(bus_index) != len(buses):
        raise ValueError("DC power flow requires nonempty, uniquely identified buses")
    if not np.array_equal(forecast.bus_ids, bus_ids):
        raise ValueError("Forecast bus order must match topology bus order")
    for values in (forecast.p_load_mw, forecast.p_gen_scheduled_mw):
        if values.shape != (hours, len(buses)) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Source/load power must be finite, nonnegative [time, bus] arrays")
    slack_index = _choose_slack_bus_index(refined_topology)

    susceptance = _branch_susceptance_mw_per_rad(branches)
    b_matrix = _build_b_matrix(branches, bus_index, susceptance, len(buses))
    components = _network_components(len(buses), branches, bus_index)

    bus_angle = np.zeros((hours, len(buses)), dtype=np.float64)
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
        unserved = np.zeros_like(load)
        curtailed = np.zeros_like(generation)
        # An island cannot import a fictitious supply from the global slack bus.
        for component in components:
            load[component], generation[component], unserved[component], curtailed[component] = _balance_dispatch(
                load[component], generation[component], renewable_mask[component], generation_mask[component]
            )
        p_injection = generation - load
        bus_injection[hour] = p_injection.astype(np.float32)
        served_load[hour] = load.astype(np.float32)
        dispatched_generation[hour] = generation.astype(np.float32)
        unserved_load[hour] = unserved.astype(np.float32)
        curtailed_generation[hour] = curtailed.astype(np.float32)

    # Factor/solve once per island for all hours, rather than once per snapshot.
    for component in components:
        reference = slack_index if slack_index in component else int(component[0])
        non_slack = component[component != reference]
        if non_slack.size:
            angles = _solve_reduced_angles(
                b_matrix[np.ix_(non_slack, non_slack)], bus_injection[:, non_slack].astype(np.float64).T
            ).T
            bus_angle[:, non_slack] = angles
    for index, (branch, branch_b) in enumerate(zip(branches, susceptance)):
        i, j = bus_index[int(branch.from_bus)], bus_index[int(branch.to_bus)]
        line_flow[:, index] = branch_b * (bus_angle[:, i].astype(np.float64) - bus_angle[:, j])

    rates = np.asarray([max(float(branch.rate_mva), 1e-6) for branch in branches], dtype=np.float32)
    line_loading = np.abs(line_flow) / rates[None, :] if len(branches) else np.zeros_like(line_flow)
    return PowerFlowStore(
        timestamps=forecast.timestamps.copy(),
        bus_ids=bus_ids,
        branch_ids=branch_ids,
        bus_angle_rad=bus_angle.astype(np.float32),
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
        if not np.isfinite(branch.x_ohm) or branch.x_ohm <= 0 or not np.isfinite(branch.nominal_kv) or branch.nominal_kv <= 0:
            raise ValueError("DC branches require positive finite reactance and nominal voltage")
        z_base_ohm = float(branch.nominal_kv) ** 2 / BASE_MVA
        x_pu = float(branch.x_ohm) / z_base_ohm
        values.append(BASE_MVA / x_pu)
    return np.asarray(values, dtype=np.float64)


def _network_components(bus_count: int, branches: tuple[object, ...], bus_index: dict[int, int]) -> list[np.ndarray]:
    """Connected components, including isolated buses, in deterministic order."""
    adjacency: list[list[int]] = [[] for _ in range(bus_count)]
    for branch in branches:
        i, j = bus_index[int(branch.from_bus)], bus_index[int(branch.to_bus)]
        if i == j:
            raise ValueError("Self-loop branches are not physical transmission lines")
        adjacency[i].append(j)
        adjacency[j].append(i)
    unseen = set(range(bus_count))
    components: list[np.ndarray] = []
    while unseen:
        stack = [min(unseen)]
        unseen.remove(stack[0])
        members = []
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(members), dtype=np.int32))
    return components


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
        # Economic screening: back down dispatchable units before spilling renewables.
        dispatchable_generation = dispatched * generation_mask * ~renewable_mask
        backed_down = _proportional_reduction(dispatchable_generation, remaining)
        dispatched -= backed_down
        curtailed += backed_down
        remaining -= float(backed_down.sum())
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
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular DC island: check topology and line reactances") from exc


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
