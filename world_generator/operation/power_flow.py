from __future__ import annotations

import numpy as np

from world_generator.core.contracts import entity_ids, interval_bounds_hours
from world_generator.core.datatypes import GridElectricalState, PowerFlowStore, RefinedGridTopologyState, SourceLoadForecastStore


BASE_MVA = 100.0


def solve_dc_power_flow(
    forecast: SourceLoadForecastStore,
    refined_topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    *, branch_in_service: np.ndarray | None = None, rebalance: bool = True,
) -> PowerFlowStore:
    buses = refined_topology.refined_buses
    branches = electrical.branch_params
    hours = forecast.p_load_mw.shape[0]
    bus_ids = entity_ids(np.asarray([bus.bus_id for bus in buses]), "DC bus_ids").astype(np.int32)
    branch_ids = entity_ids(np.asarray([branch.edge_id for branch in branches]), "DC branch_ids").astype(np.int32)
    interval_bounds_hours(forecast.timestamps, 1.0)
    bus_index = {int(bus_id): index for index, bus_id in enumerate(bus_ids)}
    if not buses or len(bus_index) != len(buses):
        raise ValueError("DC power flow requires nonempty, uniquely identified buses")
    if not np.array_equal(forecast.bus_ids, bus_ids):
        raise ValueError("Forecast bus order must match topology bus order")
    for values in (forecast.p_load_mw, forecast.p_gen_scheduled_mw, forecast.p_gen_available_mw):
        if values.shape != (hours, len(buses)) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Source/load power must be finite, nonnegative [time, bus] arrays")
    if np.any(forecast.p_gen_scheduled_mw > forecast.p_gen_available_mw + 1e-6):
        raise ValueError("Scheduled generation cannot exceed available generation")
    slack_index = _choose_slack_bus_index(refined_topology)

    susceptance = _branch_susceptance_mw_per_rad(branches)
    in_service = _branch_status(branch_in_service, hours, len(branches))
    patterns = {}
    for pattern in np.unique(in_service, axis=0):
        active = tuple(branch for branch, enabled in zip(branches, pattern) if enabled)
        patterns[pattern.tobytes()] = (
            _build_b_matrix(active, bus_index, susceptance[pattern], len(buses)),
            _network_components(len(buses), active, bus_index),
        )

    bus_angle = np.zeros((hours, len(buses)), dtype=np.float64)
    bus_injection = np.zeros((hours, len(buses)), dtype=np.float64)
    served_load = np.zeros_like(forecast.p_load_mw, dtype=np.float64)
    dispatched_generation = np.zeros_like(forecast.p_gen_scheduled_mw, dtype=np.float64)
    unserved_load = np.zeros_like(forecast.p_load_mw, dtype=np.float64)
    curtailed_generation = np.zeros_like(forecast.p_gen_scheduled_mw, dtype=np.float64)
    line_flow = np.zeros((hours, len(branches)), dtype=np.float64)
    island_id = np.zeros((hours, len(buses)), dtype=np.int64)

    bus_kinds = np.asarray(forecast.bus_kinds)
    renewable_mask = np.isin(bus_kinds, ("wind_bus", "pv_bus"))
    generation_mask = forecast.p_gen_scheduled_mw.max(axis=0) > 0.0

    for hour in range(hours):
        _, components = patterns[in_service[hour].tobytes()]
        load = forecast.p_load_mw[hour].astype(np.float64, copy=True)
        generation = forecast.p_gen_scheduled_mw[hour].astype(np.float64, copy=True)
        unserved = np.zeros_like(load)
        curtailed = np.zeros_like(generation)
        # An island cannot import a fictitious supply from the global slack bus.
        for component in components:
            island_id[hour, component] = int(bus_ids[component].min())
            if rebalance:
                load[component], generation[component], unserved[component], curtailed[component] = _balance_dispatch(
                    load[component], generation[component], renewable_mask[component], generation_mask[component]
                )
            elif abs(float((generation[component] - load[component]).sum())) > 1e-5 + 1e-10 * float(load[component].sum()):
                raise ValueError("Declared dispatch does not balance an active island")
        p_injection = generation - load
        bus_injection[hour] = p_injection
        served_load[hour] = load
        dispatched_generation[hour] = generation
        unserved_load[hour] = unserved
        curtailed_generation[hour] = curtailed

    # Factor/solve once per island for all hours, rather than once per snapshot.
    for pattern_key, (b_matrix, components) in patterns.items():
        selected_hours = np.flatnonzero([row.tobytes() == pattern_key for row in in_service])
        for component in components:
            reference = slack_index if slack_index in component else int(component[0])
            non_slack = component[component != reference]
            if non_slack.size:
                angles = _solve_reduced_angles(
                    b_matrix[np.ix_(non_slack, non_slack)], bus_injection[np.ix_(selected_hours, non_slack)].T
                ).T
                bus_angle[np.ix_(selected_hours, non_slack)] = angles
    for index, (branch, branch_b) in enumerate(zip(branches, susceptance)):
        i, j = bus_index[int(branch.from_bus)], bus_index[int(branch.to_bus)]
        line_flow[:, index] = in_service[:, index] * branch_b * (bus_angle[:, i] - bus_angle[:, j])

    rates = np.asarray([float(branch.rate_mva) for branch in branches], dtype=float)
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("DC branch ratings must be finite and positive")
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
        line_loading_ratio=line_loading,
        slack_bus_id=int(bus_ids[slack_index]),
        operation_arrays={
            "requested_load_mw": np.asarray(forecast.p_load_mw, dtype=float).copy(),
            "generation_available_mw": np.asarray(forecast.p_gen_available_mw, dtype=float).copy(),
            "renewable_curtailment_mw": (forecast.p_gen_available_mw - dispatched_generation) * renewable_mask,
            "thermal_backdown_mw": curtailed_generation * (bus_kinds == "thermal_bus"),
            "thermal_unused_available_mw": (forecast.p_gen_available_mw - dispatched_generation) * (bus_kinds == "thermal_bus"),
            "island_id": island_id, "branch_in_service": in_service,
        },
        operation_metadata={"schema_version": "operation_v1", "store_kind": "power_flow",
                            "network_model": "lossless_DC_fixed_voltage_magnitude; no_voltage_or_frequency_security",
                            "balance_scope": "each_active_island", "power_basis": "input_forecast_including_storage_if_present"},
    )


def _branch_status(values: np.ndarray | None, hours: int, count: int) -> np.ndarray:
    if values is None:
        return np.ones((hours, count), dtype=bool)
    values = np.asarray(values)
    if values.shape != (hours, count) or not np.isin(values, (0, 1)).all():
        raise ValueError("branch_in_service must be boolean [T,E] in persistent electrical branch order")
    return values.astype(bool)


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

    if imbalance > 0:
        remaining = imbalance
        # Economic screening: back down dispatchable units before spilling renewables.
        dispatchable_generation = dispatched * generation_mask * ~renewable_mask
        backed_down = _proportional_reduction(dispatchable_generation, remaining)
        dispatched -= backed_down
        curtailed += backed_down
        remaining -= float(backed_down.sum())
        if remaining > 0:
            generator_generation = dispatched * generation_mask
            curtailed_generator = _proportional_reduction(generator_generation, remaining)
            dispatched -= curtailed_generator
            curtailed += curtailed_generator
    elif imbalance < 0 and total_load > 0:
        unserved = _proportional_reduction(served_load, -imbalance)
        served_load -= unserved

    return served_load, dispatched, unserved, curtailed


def _proportional_reduction(values: np.ndarray, target: float) -> np.ndarray:
    total = float(values.sum())
    if total <= 0 or target <= 0:
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
