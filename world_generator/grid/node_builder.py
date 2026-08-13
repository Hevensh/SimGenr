from __future__ import annotations

import numpy as np

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import (
    EnergyCandidate,
    EnergyCandidateState,
    GridBus,
    GridNodeState,
    HydrologyState,
    LandUseState,
    StaticLandState,
    TerrainFeatures,
)


def build_grid_nodes(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    land_use: LandUseState,
    energy: EnergyCandidateState,
    grid: WorldGridConfig,
    config: PowerGridConfig,
) -> GridNodeState:
    buses: list[GridBus] = []
    for candidate in energy.load_candidates:
        buses.append(_bus_from_candidate(len(buses), "load_bus", candidate, 0.0))
    for candidate in energy.wind_candidates:
        buses.append(_bus_from_candidate(len(buses), "wind_bus", candidate, 0.0))
    for candidate in energy.pv_candidates:
        buses.append(_bus_from_candidate(len(buses), "pv_bus", candidate, 0.0))

    thermal_suitability, thermal_externality = _thermal_suitability(
        terrain,
        hydrology,
        land,
        land_use,
        energy,
        grid,
        config,
    )
    thermal_candidates = _select_thermal_buses(
        thermal_suitability,
        thermal_externality,
        config,
        grid,
        start_bus_id=len(buses),
        existing_generation_buses=tuple(bus for bus in buses if bus.kind in {"wind_bus", "pv_bus"}),
    )
    thermal_candidates = _size_thermal_capacity_for_adequacy(buses, thermal_candidates, config)
    buses.extend(thermal_candidates)

    load_bus_map = _bus_map(thermal_suitability.shape, buses, {"load_bus"})
    source_bus_map = _bus_map(thermal_suitability.shape, buses, {"wind_bus", "pv_bus"})
    thermal_bus_map = _bus_map(thermal_suitability.shape, buses, {"thermal_bus"})
    bus_site_map = np.maximum.reduce([load_bus_map, source_bus_map, thermal_bus_map])

    return GridNodeState(
        thermal_suitability=thermal_suitability.astype(np.float32),
        thermal_externality=thermal_externality.astype(np.float32),
        bus_site_map=bus_site_map,
        load_bus_map=load_bus_map,
        source_bus_map=source_bus_map,
        thermal_bus_map=thermal_bus_map,
        buses=tuple(buses),
    )


def _size_thermal_capacity_for_adequacy(
    existing_buses: list[GridBus],
    thermal_buses: list[GridBus],
    config: PowerGridConfig,
) -> list[GridBus]:
    if not thermal_buses:
        return thermal_buses
    load_capacity = sum(float(bus.capacity_mw) for bus in existing_buses if bus.kind == "load_bus")
    wind_capacity = sum(float(bus.capacity_mw) for bus in existing_buses if bus.kind == "wind_bus")
    solar_capacity = sum(float(bus.capacity_mw) for bus in existing_buses if bus.kind == "pv_bus")
    target = max(
        float(config.thermal_adequacy_load_fraction) * load_capacity
        - float(config.thermal_wind_capacity_credit) * wind_capacity
        - float(config.thermal_solar_capacity_credit) * solar_capacity,
        0.0,
    )
    capacities = np.asarray([float(bus.capacity_mw) for bus in thermal_buses], dtype=np.float64)
    maximum = max(float(config.thermal_capacity_max_mw), float(config.thermal_capacity_min_mw))
    total_headroom = float(np.maximum(maximum - capacities, 0.0).sum())
    remaining = min(max(target - float(capacities.sum()), 0.0), total_headroom)
    while remaining > 1e-6:
        headroom = np.maximum(maximum - capacities, 0.0)
        active = headroom > 1e-6
        if not np.any(active):
            break
        weights = np.asarray([0.35 + 0.65 * bus.suitability for bus in thermal_buses], dtype=np.float64)
        weights = np.where(active, weights, 0.0)
        addition = np.minimum(remaining * weights / max(float(weights.sum()), 1e-9), headroom)
        delivered = float(addition.sum())
        capacities += addition
        remaining -= delivered
        if delivered <= 1e-9:
            break
    return [
        GridBus(
            bus_id=bus.bus_id,
            kind=bus.kind,
            row=bus.row,
            col=bus.col,
            x=bus.x,
            y=bus.y,
            capacity_mw=float(capacities[index]),
            suitability=bus.suitability,
            externality_score=bus.externality_score,
            source_kind=bus.source_kind,
            source_id=bus.source_id,
        )
        for index, bus in enumerate(thermal_buses)
    ]


def _thermal_suitability(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    land_use: LandUseState,
    energy: EnergyCandidateState,
    grid: WorldGridConfig,
    config: PowerGridConfig,
) -> tuple[np.ndarray, np.ndarray]:
    water = hydrology.river | hydrology.lake
    protected = land.protected.astype(bool)
    blocked = water | protected
    buildability = np.clip(land.buildability, 0.0, 1.0)
    slope_ok = 1.0 - _normalize01(terrain.slope)
    load_mask = energy.load_candidate_map >= 0
    residential_core = land_use.residential > 0.22

    load_distance = _distance_to_mask(load_mask, grid)
    residential_distance = _distance_to_mask(residential_core, grid)
    load_access = np.exp(-((load_distance - config.thermal_load_optimal_km) ** 2) / (2.0 * max(config.thermal_load_sigma_km, 1e-6) ** 2))
    residential_buffer = np.clip(residential_distance / max(config.thermal_residential_buffer_km, 1e-6), 0.0, 1.0)
    industrial_preference = np.clip(0.50 + 0.50 * land_use.industrial + 0.20 * land_use.agriculture, 0.0, 1.0)
    flood_safe = 1.0 - np.clip(hydrology.flood_risk, 0.0, 1.0)

    suitability = _normalize01(
        load_access
        * residential_buffer
        * industrial_preference
        * buildability
        * slope_ok
        * flood_safe
    )
    suitability[blocked] = 0.0
    suitability[land_use.residential > 0.45] *= 0.15
    suitability[energy.load_candidate_map >= 0] *= 0.40

    externality = np.clip(1.0 - residential_buffer, 0.0, 1.0)
    externality = np.maximum(externality, 0.35 * land_use.residential)
    externality[blocked] = 1.0
    return suitability, externality


def _select_thermal_buses(
    suitability: np.ndarray,
    externality: np.ndarray,
    config: PowerGridConfig,
    grid: WorldGridConfig,
    start_bus_id: int,
    existing_generation_buses: tuple[GridBus, ...],
) -> list[GridBus]:
    selected: list[GridBus] = []
    rows, cols = np.indices(suitability.shape)
    thermal_distance_cells = max(config.thermal_min_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    renewable_distance_cells = max(config.thermal_min_renewable_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    spread_radius_cells = max(config.thermal_spread_radius_km / max(grid.cell_size_km, 1e-6), thermal_distance_cells)
    while len(selected) < config.thermal_candidate_count:
        score_map = suitability - 0.18 * externality
        hard_mask = np.zeros(suitability.shape, dtype=bool)
        coverage = np.zeros(suitability.shape, dtype=np.float32)
        for item in selected:
            distance = np.hypot(rows - item.row, cols - item.col)
            coverage = np.maximum(coverage, np.exp(-np.square(distance / max(spread_radius_cells, 1e-6))))
            hard_mask |= distance < thermal_distance_cells
        for item in existing_generation_buses:
            distance = np.hypot(rows - item.row, cols - item.col)
            hard_mask |= distance < renewable_distance_cells
        if selected:
            score_map *= 1.0 - config.thermal_spread_penalty_weight * coverage
        score_map[hard_mask] = -1.0
        score_map[suitability <= 1e-6] = -1.0
        row, col = np.unravel_index(int(np.argmax(score_map)), suitability.shape)
        score = float(suitability[row, col])
        if score <= 1e-6 or float(score_map[row, col]) <= 0.0:
            break
        capacity = config.thermal_capacity_min_mw + (config.thermal_capacity_max_mw - config.thermal_capacity_min_mw) * np.sqrt(score)
        selected.append(
            GridBus(
                bus_id=start_bus_id + len(selected),
                kind="thermal_bus",
                row=int(row),
                col=int(col),
                x=float(col / max(grid.width - 1, 1)),
                y=float(row / max(grid.height - 1, 1)),
                capacity_mw=float(capacity),
                suitability=score,
                externality_score=float(externality[row, col]),
                source_kind="thermal",
                source_id=len(selected),
            )
        )
    return _relax_thermal_buses(suitability, externality, selected, existing_generation_buses, config, grid)


def _relax_thermal_buses(
    suitability: np.ndarray,
    externality: np.ndarray,
    buses: list[GridBus],
    existing_generation_buses: tuple[GridBus, ...],
    config: PowerGridConfig,
    grid: WorldGridConfig,
) -> list[GridBus]:
    if config.thermal_relax_iterations <= 0 or not buses:
        return buses
    thermal_distance_cells = max(config.thermal_min_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    renewable_distance_cells = max(config.thermal_min_renewable_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    search_radius = max(int(np.ceil(thermal_distance_cells * 0.55)), 2)
    relaxed = list(buses)
    for _ in range(config.thermal_relax_iterations):
        updated: list[GridBus] = []
        for index, bus in enumerate(relaxed):
            row, col, score = _best_thermal_cell(
                suitability,
                externality,
                bus.row,
                bus.col,
                tuple(updated) + tuple(relaxed[index + 1 :]),
                existing_generation_buses,
                search_radius,
                thermal_distance_cells,
                renewable_distance_cells,
                config.thermal_neighbor_repulsion_weight,
            )
            capacity = config.thermal_capacity_min_mw + (config.thermal_capacity_max_mw - config.thermal_capacity_min_mw) * np.sqrt(score)
            updated.append(
                GridBus(
                    bus_id=bus.bus_id,
                    kind=bus.kind,
                    row=row,
                    col=col,
                    x=float(col / max(grid.width - 1, 1)),
                    y=float(row / max(grid.height - 1, 1)),
                    capacity_mw=float(capacity),
                    suitability=float(score),
                    externality_score=float(externality[row, col]),
                    source_kind=bus.source_kind,
                    source_id=bus.source_id,
                )
            )
        relaxed = updated
    return relaxed


def _best_thermal_cell(
    suitability: np.ndarray,
    externality: np.ndarray,
    row: int,
    col: int,
    thermal_others: tuple[GridBus, ...],
    renewable_others: tuple[GridBus, ...],
    search_radius: int,
    thermal_distance_cells: float,
    renewable_distance_cells: float,
    repulsion_weight: float,
) -> tuple[int, int, float]:
    best_row = row
    best_col = col
    best_utility = float(suitability[row, col])
    best_score = -np.inf
    thermal_sigma = max(thermal_distance_cells, 1.0)
    renewable_sigma = max(renewable_distance_cells, 1.0)
    for rr in range(max(row - search_radius, 0), min(row + search_radius + 1, suitability.shape[0])):
        for cc in range(max(col - search_radius, 0), min(col + search_radius + 1, suitability.shape[1])):
            utility = float(suitability[rr, cc])
            if utility <= 1e-6:
                continue
            thermal_distances = [float(np.hypot(rr - item.row, cc - item.col)) for item in thermal_others]
            renewable_distances = [float(np.hypot(rr - item.row, cc - item.col)) for item in renewable_others]
            if any(distance < thermal_distance_cells for distance in thermal_distances):
                continue
            if any(distance < renewable_distance_cells for distance in renewable_distances):
                continue
            soft_repulsion = sum(np.exp(-((distance / thermal_sigma) ** 2)) for distance in thermal_distances)
            soft_repulsion += sum(np.exp(-((distance / renewable_sigma) ** 2)) for distance in renewable_distances)
            move_penalty = 0.025 * float(np.hypot(rr - row, cc - col)) / max(search_radius, 1)
            score = utility - 0.18 * float(externality[rr, cc]) - repulsion_weight * 0.45 * soft_repulsion - move_penalty
            if score > best_score:
                best_score = score
                best_row = int(rr)
                best_col = int(cc)
                best_utility = utility
    return best_row, best_col, best_utility


def _bus_from_candidate(bus_id: int, kind: str, candidate: EnergyCandidate, externality_score: float) -> GridBus:
    return GridBus(
        bus_id=bus_id,
        kind=kind,
        row=candidate.row,
        col=candidate.col,
        x=candidate.x,
        y=candidate.y,
        capacity_mw=candidate.capacity_mw,
        suitability=candidate.suitability,
        externality_score=externality_score,
        source_kind=candidate.kind,
        source_id=candidate.candidate_id,
    )


def _bus_map(shape: tuple[int, int], buses: list[GridBus], kinds: set[str]) -> np.ndarray:
    values = np.full(shape, -1, dtype=np.int16)
    for bus in buses:
        if bus.kind in kinds:
            values[bus.row, bus.col] = bus.bus_id
    return values


def _distance_to_mask(mask: np.ndarray, grid: WorldGridConfig) -> np.ndarray:
    rows, cols = np.indices(mask.shape)
    points = np.argwhere(mask)
    if points.size == 0:
        return np.full(mask.shape, max(mask.shape) * grid.cell_size_km, dtype=np.float32)
    distance_cells = np.full(mask.shape, np.inf, dtype=np.float32)
    for row, col in points:
        distance_cells = np.minimum(distance_cells, np.hypot(rows - row, cols - col))
    return (distance_cells * grid.cell_size_km).astype(np.float32)


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)
