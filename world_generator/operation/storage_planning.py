from __future__ import annotations

from collections import deque

import numpy as np

from world_generator.core.config import StorageConfig, WorldGridConfig
from world_generator.core.datatypes import (
    GridElectricalState,
    PowerFlowStore,
    RefinedGridTopologyState,
    StoragePlanStore,
    StorageNeedStore,
    StorageSite,
)


def analyze_storage_need(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    power_flow: PowerFlowStore,
    grid: WorldGridConfig,
    config: StorageConfig,
) -> StorageNeedStore:
    buses = topology.refined_buses
    bus_by_id = {int(bus.bus_id): bus for bus in buses}
    flow_bus_index = {int(bus_id): index for index, bus_id in enumerate(power_flow.bus_ids)}
    load_buses = [bus for bus in buses if bus.kind == "load_bus" and int(bus.bus_id) in flow_bus_index]
    load_ids = np.asarray([int(bus.bus_id) for bus in load_buses], dtype=np.int32)
    rows = np.asarray([int(bus.row) for bus in load_buses], dtype=np.int16)
    cols = np.asarray([int(bus.col) for bus in load_buses], dtype=np.int16)
    hours = int(power_flow.timestamps.size)

    if not load_buses:
        empty_nodes = np.zeros(0, dtype=np.float32)
        empty_dynamic = np.zeros((hours, 0), dtype=np.float32)
        return StorageNeedStore(
            timestamps=power_flow.timestamps.copy(),
            bus_ids=load_ids,
            rows=rows,
            cols=cols,
            support_requirement_mw=empty_dynamic,
            unserved_load_mw=empty_dynamic.copy(),
            congestion_support_mw=empty_dynamic.copy(),
            peak_support_mw=empty_nodes,
            total_support_energy_mwh=empty_nodes.copy(),
            max_event_energy_mwh=empty_nodes.copy(),
            positive_ramp_p95_mw=empty_nodes.copy(),
            congestion_exposure_hours=empty_nodes.copy(),
            suggested_power_mw=empty_nodes.copy(),
            suggested_energy_mwh=empty_nodes.copy(),
            need_score=empty_nodes.copy(),
            need_score_map=np.zeros((grid.height, grid.width), dtype=np.float32),
        )

    load_flow_indices = np.asarray([flow_bus_index[int(bus_id)] for bus_id in load_ids], dtype=np.int32)
    load_profile = (
        power_flow.served_load_mw[:, load_flow_indices]
        + power_flow.unserved_load_mw[:, load_flow_indices]
    ).astype(np.float32)
    unserved = power_flow.unserved_load_mw[:, load_flow_indices].astype(np.float32, copy=True)
    congestion_support, congestion_hours = _allocate_congestion_support(
        topology,
        electrical,
        power_flow,
        load_ids,
        load_profile,
        config,
    )
    support = (unserved + congestion_support).astype(np.float32)

    peak_support = support.max(axis=0, initial=0.0).astype(np.float32)
    total_energy = support.sum(axis=0).astype(np.float32)
    max_event_energy = np.asarray(
        [_max_contiguous_energy(support[:, index]) for index in range(load_ids.size)],
        dtype=np.float32,
    )
    positive_ramp = np.maximum(np.diff(load_profile, axis=0, prepend=load_profile[:1]), 0.0)
    ramp_p95 = np.percentile(positive_ramp, 95, axis=0).astype(np.float32)
    suggested_power = np.asarray(
        [_active_percentile(support[:, index], 90.0) for index in range(load_ids.size)],
        dtype=np.float32,
    )
    suggested_power = np.maximum(suggested_power, 0.35 * peak_support).astype(np.float32)
    suggested_energy = np.where(
        suggested_power > 1e-6,
        np.minimum(
            np.maximum(max_event_energy / _deliverable_energy_fraction(config), suggested_power * max(config.minimum_duration_hours, 0.0)),
            suggested_power * max(config.maximum_duration_hours, config.minimum_duration_hours, 0.0),
        ),
        0.0,
    ).astype(np.float32)

    need_score = (
        0.35 * _normalize_nodes(suggested_power)
        + 0.35 * _normalize_nodes(suggested_energy)
        + 0.20 * _normalize_nodes(congestion_hours)
        + 0.10 * _normalize_nodes(ramp_p95)
    ).astype(np.float32)
    need_score = _normalize_nodes(need_score)
    need_map = _spread_node_metric(rows, cols, need_score, grid, config.map_spread_radius_km)

    return StorageNeedStore(
        timestamps=power_flow.timestamps.copy(),
        bus_ids=load_ids,
        rows=rows,
        cols=cols,
        support_requirement_mw=support,
        unserved_load_mw=unserved,
        congestion_support_mw=congestion_support,
        peak_support_mw=peak_support,
        total_support_energy_mwh=total_energy,
        max_event_energy_mwh=max_event_energy,
        positive_ramp_p95_mw=ramp_p95,
        congestion_exposure_hours=congestion_hours,
        suggested_power_mw=suggested_power,
        suggested_energy_mwh=suggested_energy,
        need_score=need_score,
        need_score_map=need_map,
    )


def plan_storage_sites(
    topology: RefinedGridTopologyState,
    storage_need: StorageNeedStore,
    config: StorageConfig,
) -> StoragePlanStore:
    load_count = int(storage_need.bus_ids.size)
    if load_count == 0:
        return StoragePlanStore(
            timestamps=storage_need.timestamps.copy(),
            load_bus_ids=storage_need.bus_ids.copy(),
            assigned_site_ids=np.zeros(0, dtype=np.int16),
            site_support_requirement_mw=np.zeros((storage_need.timestamps.size, 0), dtype=np.float32),
            sites=(),
        )

    adjacency = _bus_adjacency(topology)
    distances = _load_bus_distance_matrix(storage_need.bus_ids, adjacency)
    priority = (
        0.65 * storage_need.need_score
        + 0.35 * _normalize_nodes(storage_need.suggested_energy_mwh)
    ).astype(np.float32)
    eligible = (
        (storage_need.need_score >= float(config.site_min_need_score))
        & (storage_need.peak_support_mw > 1e-6)
    )
    if not np.any(eligible) and np.any(storage_need.peak_support_mw > 1e-6):
        eligible[int(np.argmax(priority))] = True

    selected = _select_storage_attachment_indices(priority, eligible, distances, config)
    assignments = _assign_load_buses_to_sites(
        storage_need.support_requirement_mw,
        selected,
        distances,
        int(config.site_max_service_hops),
    )
    profiles = np.zeros((storage_need.timestamps.size, len(selected)), dtype=np.float32)
    sites: list[StorageSite] = []
    reserve = max(float(config.capacity_reserve_margin), 1.0)
    for site_id, load_index in enumerate(selected):
        members = np.where(assignments == site_id)[0]
        if members.size:
            profiles[:, site_id] = storage_need.support_requirement_mw[:, members].sum(axis=1)
        profile = profiles[:, site_id]
        peak = float(np.max(profile, initial=0.0))
        base_power = max(_active_percentile(profile, 90.0), 0.35 * peak)
        power_mw = reserve * base_power
        event_energy = _max_contiguous_energy(profile)
        energy_mwh = max(
            reserve * event_energy / _deliverable_energy_fraction(config),
            power_mw * max(float(config.minimum_duration_hours), 0.0),
        )
        energy_mwh = min(
            energy_mwh,
            power_mw * max(float(config.maximum_duration_hours), float(config.minimum_duration_hours), 0.0),
        )
        member_score = float(np.max(storage_need.need_score[members], initial=0.0)) if members.size else 0.0
        sites.append(
            StorageSite(
                site_id=site_id,
                bus_id=int(storage_need.bus_ids[load_index]),
                row=int(storage_need.rows[load_index]),
                col=int(storage_need.cols[load_index]),
                covered_bus_ids=tuple(int(storage_need.bus_ids[index]) for index in members),
                power_mw=float(power_mw),
                energy_mwh=float(energy_mwh),
                initial_soc_mwh=float(energy_mwh * np.clip(config.initial_soc_fraction, 0.0, 1.0)),
                need_score=member_score,
            )
        )
    return StoragePlanStore(
        timestamps=storage_need.timestamps.copy(),
        load_bus_ids=storage_need.bus_ids.copy(),
        assigned_site_ids=assignments.astype(np.int16),
        site_support_requirement_mw=profiles,
        sites=tuple(sites),
    )


def _deliverable_energy_fraction(config: StorageConfig) -> float:
    """AC-deliverable energy per nameplate MWh over the allowed SOC range."""
    fraction = (config.maximum_soc_fraction - config.minimum_soc_fraction) * config.discharge_efficiency
    if not 0 < fraction <= 1:
        raise ValueError("Storage sizing requires a positive SOC range and physical discharge efficiency")
    return float(fraction)


def _select_storage_attachment_indices(
    priority: np.ndarray,
    eligible: np.ndarray,
    distances: np.ndarray,
    config: StorageConfig,
) -> list[int]:
    if not np.any(eligible) or config.site_max_count <= 0:
        return []
    decay = max(float(config.site_coverage_decay_hops), 1e-6)
    max_hops = max(int(config.site_max_service_hops), 0)
    coverage = np.exp(-distances / decay).astype(np.float32)
    coverage[~np.isfinite(distances) | (distances > max_hops)] = 0.0
    residual = np.where(eligible, priority, 0.0).astype(np.float32)
    initial_total = max(float(residual.sum()), 1e-6)
    selected: list[int] = []
    while len(selected) < int(config.site_max_count):
        utilities = coverage @ residual
        utilities *= 0.65 + 0.35 * priority
        if selected:
            utilities[np.asarray(selected, dtype=np.int32)] = -np.inf
        utilities[~eligible] = -np.inf
        candidate = int(np.argmax(utilities))
        if not np.isfinite(utilities[candidate]) or utilities[candidate] <= 1e-8:
            break
        selected.append(candidate)
        residual *= 1.0 - coverage[candidate]
        if float(residual.sum()) <= initial_total * max(float(config.site_min_residual_fraction), 0.0):
            break
    return selected


def _assign_load_buses_to_sites(
    support_profile: np.ndarray,
    selected: list[int],
    distances: np.ndarray,
    max_service_hops: int,
) -> np.ndarray:
    assignments = np.full(support_profile.shape[1], -1, dtype=np.int16)
    if not selected:
        return assignments
    active = np.max(support_profile, axis=0, initial=0.0) > 1e-6
    selected_distances = distances[np.asarray(selected, dtype=np.int32), :]
    nearest_site = np.argmin(selected_distances, axis=0)
    nearest_distance = selected_distances[nearest_site, np.arange(support_profile.shape[1])]
    covered = active & np.isfinite(nearest_distance) & (nearest_distance <= max(max_service_hops, 0))
    assignments[covered] = nearest_site[covered].astype(np.int16)
    return assignments


def _load_bus_distance_matrix(load_ids: np.ndarray, adjacency: dict[int, set[int]]) -> np.ndarray:
    count = int(load_ids.size)
    distances = np.full((count, count), np.inf, dtype=np.float32)
    for row, bus_id in enumerate(load_ids):
        hops = _hop_distances(int(bus_id), adjacency)
        distances[row] = np.asarray([hops.get(int(target), np.inf) for target in load_ids], dtype=np.float32)
    return distances


def _allocate_congestion_support(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    power_flow: PowerFlowStore,
    load_ids: np.ndarray,
    load_profile: np.ndarray,
    config: StorageConfig,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.zeros_like(load_profile, dtype=np.float32)
    exposure_hours = np.zeros(load_ids.size, dtype=np.float32)
    if not electrical.branch_params or load_ids.size == 0:
        return support, exposure_hours

    adjacency = _bus_adjacency(topology)
    mean_load = np.maximum(load_profile.mean(axis=0), 1e-3)
    receivers = {
        int(branch.from_bus) for branch in electrical.branch_params
    } | {
        int(branch.to_bus) for branch in electrical.branch_params
    }
    receiver_weights = {
        bus_id: _load_weights_from_receiver(
            bus_id,
            load_ids,
            mean_load,
            adjacency,
            config.congestion_decay_hops,
        )
        for bus_id in receivers
    }
    flow_index = {int(branch_id): index for index, branch_id in enumerate(power_flow.branch_ids)}
    trigger = float(np.clip(config.congestion_trigger_ratio, 0.0, None))
    for branch in electrical.branch_params:
        index = flow_index.get(int(branch.edge_id))
        if index is None:
            continue
        flow = power_flow.line_flow_mw[:, index].astype(np.float32, copy=False)
        threshold_mw = trigger * max(float(branch.rate_mva), 1e-6)
        excess = np.maximum(np.abs(flow) - threshold_mw, 0.0)
        active = excess > 1e-6
        if not np.any(active):
            continue
        if float(np.mean(active)) > float(np.clip(config.congestion_storage_max_active_fraction, 0.0, 1.0)):
            continue
        forward = flow >= 0.0
        for mask, receiver in (
            (active & forward, int(branch.to_bus)),
            (active & ~forward, int(branch.from_bus)),
        ):
            if not np.any(mask):
                continue
            weights = receiver_weights[receiver]
            support[mask] += excess[mask, None] * weights[None, :]
            exposure_hours += float(mask.sum()) * weights
    return support, exposure_hours


def _bus_adjacency(topology: RefinedGridTopologyState) -> dict[int, set[int]]:
    adjacency = {int(bus.bus_id): set() for bus in topology.refined_buses}
    for edge in topology.refined_edges:
        first, second = int(edge.from_bus), int(edge.to_bus)
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    return adjacency


def _load_weights_from_receiver(
    receiver: int,
    load_ids: np.ndarray,
    mean_load: np.ndarray,
    adjacency: dict[int, set[int]],
    decay_hops: float,
) -> np.ndarray:
    distances = _hop_distances(receiver, adjacency)
    decay = max(float(decay_hops), 1e-6)
    weights = np.asarray(
        [
            mean_load[index] * np.exp(-float(distances.get(int(bus_id), 1_000_000)) / decay)
            for index, bus_id in enumerate(load_ids)
        ],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 1e-12:
        weights = np.ones(load_ids.size, dtype=np.float64)
        total = float(weights.sum())
    return (weights / total).astype(np.float32)


def _hop_distances(start: int, adjacency: dict[int, set[int]]) -> dict[int, int]:
    distances = {int(start): 0}
    queue = deque([int(start)])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, ()):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _max_contiguous_energy(values: np.ndarray, threshold: float = 1e-6) -> float:
    best = 0.0
    current = 0.0
    for value in np.asarray(values, dtype=np.float64):
        if value > threshold:
            current += float(value)
            best = max(best, current)
        else:
            current = 0.0
    return best


def _active_percentile(values: np.ndarray, percentile: float) -> float:
    active = np.asarray(values, dtype=np.float64)
    active = active[active > 1e-6]
    return float(np.percentile(active, percentile)) if active.size else 0.0


def _normalize_nodes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmax = float(np.max(values, initial=0.0))
    return values / vmax if vmax > 1e-12 else np.zeros_like(values)


def _spread_node_metric(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    grid: WorldGridConfig,
    radius_km: float,
) -> np.ndarray:
    rr, cc = np.indices((grid.height, grid.width))
    radius_cells = max(float(radius_km) / max(grid.cell_size_km, 1e-6), 1e-6)
    result = np.zeros((grid.height, grid.width), dtype=np.float32)
    for row, col, value in zip(rows, cols, values):
        influence = float(value) * np.exp(-((rr - int(row)) ** 2 + (cc - int(col)) ** 2) / (2.0 * radius_cells**2))
        result = np.maximum(result, influence.astype(np.float32))
    return result
