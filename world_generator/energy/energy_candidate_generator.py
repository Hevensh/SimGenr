from __future__ import annotations

import numpy as np

from world_generator.core.config import EnergyConfig, WorldGridConfig
from world_generator.core.datatypes import (
    CityState,
    EnergyCandidate,
    EnergyCandidateState,
    HydrologyState,
    LandUseState,
    StaticLandState,
    TerrainFeatures,
)


def generate_energy_candidates(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    city: CityState,
    land_use: LandUseState,
    climate_maps: dict[str, np.ndarray],
    grid: WorldGridConfig,
    config: EnergyConfig,
) -> EnergyCandidateState:
    water = hydrology.river | hydrology.lake
    protected = land.protected.astype(bool)
    developable = np.clip(land.buildability, 0.0, 1.0)
    slope_n = _normalize01(terrain.slope)
    roughness_n = _normalize01(terrain.roughness)
    wind_speed = np.hypot(climate_maps["prevailing_wind_u"], climate_maps["prevailing_wind_v"])
    wind_n = _normalize01(wind_speed)
    irradiance_n = _normalize01(climate_maps["mean_irradiance"])
    cloud = np.clip(climate_maps["mean_cloud"], 0.0, 1.0)
    city_distance = _distance_to_mask(city.urban_density > 0.18, grid)
    wind_city_ramp = _distance_ramp(city_distance, config.wind_min_city_distance_km)
    wind_city_decay = _half_life_decay(
        np.maximum(city_distance - config.wind_min_city_distance_km, 0.0),
        config.wind_city_half_distance_km,
    )
    wind_city_ok = wind_city_ramp * wind_city_decay
    pv_city_ok = _distance_ramp(city_distance, config.pv_min_city_distance_km)
    local_flatness = np.clip(1.0 - slope_n, 0.0, 1.0)
    roughness_ok = np.clip(1.0 - roughness_n, 0.0, 1.0)
    wind_buildability = np.clip(
        config.wind_flatness_weight * local_flatness
        + config.wind_roughness_penalty_weight * roughness_ok,
        0.0,
        1.0,
    )

    source_allowed = developable.copy()
    source_allowed[water | protected] = 0.0
    sparse_land = np.clip(1.0 - 0.72 * city.urban_density - 0.18 * land_use.residential - 0.10 * land_use.commercial, 0.0, 1.0)

    wind_suitability = _normalize01(
        wind_n
        * wind_buildability
        * source_allowed
        * sparse_land
        * wind_city_ok
        * (1.0 - 0.55 * hydrology.flood_risk)
    )
    wind_suitability[water | protected] = 0.0

    pv_suitability = _normalize01(
        irradiance_n
        * (1.0 - 0.62 * cloud)
        * (1.0 - 0.82 * slope_n)
        * source_allowed
        * np.clip(0.58 + 0.42 * land_use.agriculture + 0.18 * sparse_land, 0.0, 1.0)
        * pv_city_ok
    )
    pv_suitability[water | protected] = 0.0
    wind_selection_utility = _local_source_utility(wind_suitability, config.wind_selection_radius_km, grid)
    pv_selection_utility = _local_source_utility(pv_suitability, config.pv_selection_radius_km, grid)

    load_node_density = _normalize01(land_use.load_density_base * (1.0 - 0.35 * hydrology.flood_risk))
    load_node_density[water | protected] = 0.0

    wind_candidates = _select_candidates(
        wind_suitability,
        wind_selection_utility,
        "wind",
        config.wind_candidate_count,
        config.wind_min_source_distance_km,
        config.wind_capacity_min_mw,
        config.wind_capacity_max_mw,
        grid,
        config,
    )
    wind_candidates = _relax_candidates(
        wind_suitability,
        wind_selection_utility,
        wind_candidates,
        (),
        config.wind_min_source_distance_km,
        config.wind_capacity_min_mw,
        config.wind_capacity_max_mw,
        grid,
        config.source_relax_iterations,
        config.source_neighbor_repulsion_weight,
        hard_min_distance_km=0.65 * config.wind_min_source_distance_km,
    )
    pv_candidates = _select_candidates(
        pv_suitability,
        pv_selection_utility,
        "pv",
        config.pv_candidate_count,
        config.min_source_distance_km,
        config.pv_capacity_min_mw,
        config.pv_capacity_max_mw,
        grid,
        config,
        avoid_candidates=wind_candidates,
        avoid_min_distance_km=config.min_cross_source_distance_km,
    )
    pv_candidates = _relax_candidates(
        pv_suitability,
        pv_selection_utility,
        pv_candidates,
        tuple(wind_candidates),
        config.min_source_distance_km,
        config.pv_capacity_min_mw,
        config.pv_capacity_max_mw,
        grid,
        config.source_relax_iterations,
        config.source_neighbor_repulsion_weight,
        hard_min_distance_km=0.65 * config.min_source_distance_km,
        avoid_distance_km=config.min_cross_source_distance_km,
    )
    wind_candidates = _apply_source_cluster_capacity(
        wind_candidates,
        wind_suitability,
        config.wind_capacity_min_mw,
        config.wind_capacity_max_mw,
        config.wind_cluster_radius_km,
        config.wind_cluster_capacity_max_multiplier,
        grid,
    )
    pv_candidates = _apply_source_cluster_capacity(
        pv_candidates,
        pv_suitability,
        config.pv_capacity_min_mw,
        config.pv_capacity_max_mw,
        config.pv_cluster_radius_km,
        config.pv_cluster_capacity_max_multiplier,
        grid,
    )
    load_candidates = _select_load_candidates(
        load_node_density,
        city,
        water | protected,
        config,
        grid,
    )
    wind_candidate_map = _candidate_map(wind_suitability.shape, wind_candidates)
    pv_candidate_map = _candidate_map(pv_suitability.shape, pv_candidates)
    source_candidate_map = np.maximum(wind_candidate_map, pv_candidate_map)
    load_candidate_map = _candidate_map(load_node_density.shape, load_candidates)

    return EnergyCandidateState(
        wind_suitability=wind_suitability.astype(np.float32),
        pv_suitability=pv_suitability.astype(np.float32),
        load_node_density=load_node_density.astype(np.float32),
        wind_candidate_map=wind_candidate_map,
        pv_candidate_map=pv_candidate_map,
        source_candidate_map=source_candidate_map,
        load_candidate_map=load_candidate_map,
        wind_candidates=tuple(wind_candidates),
        pv_candidates=tuple(pv_candidates),
        load_candidates=tuple(load_candidates),
    )


def _select_candidates(
    suitability: np.ndarray,
    utility_map: np.ndarray,
    kind: str,
    count: int,
    min_distance_km: float,
    capacity_min_mw: float,
    capacity_max_mw: float,
    grid: WorldGridConfig,
    config: EnergyConfig,
    selected: list[EnergyCandidate] | None = None,
    avoid_candidates: list[EnergyCandidate] | tuple[EnergyCandidate, ...] = (),
    avoid_min_distance_km: float | None = None,
) -> list[EnergyCandidate]:
    min_distance_cells = max(min_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    spread_radius_cells = max(config.source_spread_radius_km / max(grid.cell_size_km, 1e-6), min_distance_cells)
    avoid_min_distance_cells = min_distance_cells
    if avoid_min_distance_km is not None:
        avoid_min_distance_cells = max(avoid_min_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    selected = list(selected or [])
    avoid = list(avoid_candidates)
    rows, cols = np.indices(suitability.shape)
    while len(selected) < count:
        score_map = utility_map.copy()
        hard_mask = np.zeros(suitability.shape, dtype=bool)
        coverage = np.zeros(suitability.shape, dtype=np.float32)
        for item in selected:
            distance = np.hypot(rows - item.row, cols - item.col)
            coverage = np.maximum(coverage, np.exp(-np.square(distance / max(spread_radius_cells, 1e-6))))
            hard_mask |= distance < min_distance_cells
        for item in avoid:
            distance = np.hypot(rows - item.row, cols - item.col)
            hard_mask |= distance < avoid_min_distance_cells
        if selected:
            score_map *= 1.0 - config.source_spread_penalty_weight * coverage
        score_map[hard_mask] = -1.0
        score_map[(suitability <= 1e-6) | (utility_map <= 1e-6)] = -1.0
        row, col = np.unravel_index(int(np.argmax(score_map)), suitability.shape)
        score = float(suitability[row, col])
        if score <= 1e-6 or float(score_map[row, col]) <= 0.0:
            break
        selected.append(_make_candidate(len(selected), kind, row, col, score, capacity_min_mw, capacity_max_mw, grid))
    return selected


def _relax_candidates(
    suitability: np.ndarray,
    utility_map: np.ndarray,
    candidates: list[EnergyCandidate],
    avoid_candidates: tuple[EnergyCandidate, ...],
    min_distance_km: float,
    capacity_min_mw: float,
    capacity_max_mw: float,
    grid: WorldGridConfig,
    iterations: int,
    repulsion_weight: float,
    max_step_cells: int | None = None,
    hard_min_distance_km: float | None = None,
    avoid_distance_km: float | None = None,
) -> list[EnergyCandidate]:
    if iterations <= 0 or not candidates:
        return candidates
    min_distance_cells = max(min_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    avoid_distance_cells = min_distance_cells
    if avoid_distance_km is not None:
        avoid_distance_cells = max(avoid_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    search_radius = max(int(np.ceil(min_distance_cells * 0.55)), 2)
    if max_step_cells is not None:
        search_radius = max(int(max_step_cells), 1)
    hard_min_distance_cells = 1.0
    if hard_min_distance_km is not None:
        hard_min_distance_cells = max(hard_min_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    relaxed = list(candidates)
    for _ in range(iterations):
        updated: list[EnergyCandidate] = []
        for index, candidate in enumerate(relaxed):
            same_others = tuple(updated) + tuple(relaxed[index + 1 :])
            row, col, score = _best_relaxed_cell(
                suitability,
                utility_map,
                candidate.row,
                candidate.col,
                same_others,
                avoid_candidates,
                search_radius,
                min_distance_cells,
                avoid_distance_cells,
                repulsion_weight,
                hard_min_distance_cells,
            )
            updated.append(
                EnergyCandidate(
                    candidate_id=candidate.candidate_id,
                    kind=candidate.kind,
                    row=row,
                    col=col,
                    x=float(col / max(grid.width - 1, 1)),
                    y=float(row / max(grid.height - 1, 1)),
                    capacity_mw=float(capacity_min_mw + (capacity_max_mw - capacity_min_mw) * np.sqrt(score)),
                    suitability=float(score),
                )
            )
        relaxed = updated
    return relaxed


def _best_relaxed_cell(
    suitability: np.ndarray,
    utility_map: np.ndarray,
    row: int,
    col: int,
    same_others: tuple[EnergyCandidate, ...],
    avoid_candidates: tuple[EnergyCandidate, ...],
    search_radius: int,
    min_distance_cells: float,
    avoid_distance_cells: float,
    repulsion_weight: float,
    hard_min_distance_cells: float,
) -> tuple[int, int, float]:
    best_row = row
    best_col = col
    best_utility = float(suitability[row, col])
    best_score = -np.inf
    sigma = max(min_distance_cells, 1.0)
    for rr in range(max(row - search_radius, 0), min(row + search_radius + 1, suitability.shape[0])):
        for cc in range(max(col - search_radius, 0), min(col + search_radius + 1, suitability.shape[1])):
            center_score = float(suitability[rr, cc])
            utility = float(utility_map[rr, cc])
            if center_score <= 1e-6 or utility <= 1e-6:
                continue
            same_distances = [float(np.hypot(rr - item.row, cc - item.col)) for item in same_others]
            avoid_distances = [float(np.hypot(rr - item.row, cc - item.col)) for item in avoid_candidates]
            distances = same_distances + avoid_distances
            if any(distance < hard_min_distance_cells for distance in distances):
                continue
            soft_repulsion = sum(np.exp(-((distance / sigma) ** 2)) for distance in same_distances)
            soft_repulsion += sum(np.exp(-((distance / max(avoid_distance_cells, 1.0)) ** 2)) for distance in avoid_distances)
            hard_penalty = sum((max(min_distance_cells - distance, 0.0) / min_distance_cells) ** 2 for distance in same_distances)
            hard_penalty += sum((max(avoid_distance_cells - distance, 0.0) / avoid_distance_cells) ** 2 for distance in avoid_distances)
            move_penalty = 0.025 * float(np.hypot(rr - row, cc - col)) / max(search_radius, 1)
            score = utility - repulsion_weight * (0.45 * soft_repulsion + 2.0 * hard_penalty) - move_penalty
            if score > best_score:
                best_score = score
                best_row = int(rr)
                best_col = int(cc)
                best_utility = center_score
    return best_row, best_col, best_utility


def _make_candidate(
    candidate_id: int,
    kind: str,
    row: int,
    col: int,
    score: float,
    capacity_min_mw: float,
    capacity_max_mw: float,
    grid: WorldGridConfig,
) -> EnergyCandidate:
    capacity = capacity_min_mw + (capacity_max_mw - capacity_min_mw) * np.sqrt(score)
    return EnergyCandidate(
        candidate_id=candidate_id,
        kind=kind,
        row=int(row),
        col=int(col),
        x=float(col / max(grid.width - 1, 1)),
        y=float(row / max(grid.height - 1, 1)),
        capacity_mw=float(capacity),
        suitability=float(score),
    )


def _apply_source_cluster_capacity(
    candidates: list[EnergyCandidate],
    suitability: np.ndarray,
    capacity_min_mw: float,
    capacity_max_mw: float,
    cluster_radius_km: float,
    max_multiplier: float,
    grid: WorldGridConfig,
) -> list[EnergyCandidate]:
    if not candidates:
        return candidates
    radius_cells = max(cluster_radius_km / max(grid.cell_size_km, 1e-6), 0.0)
    updated: list[EnergyCandidate] = []
    for candidate in candidates:
        local_sum = _local_suitability_sum(suitability, candidate.row, candidate.col, radius_cells)
        center = max(float(suitability[candidate.row, candidate.col]), 1e-6)
        multiplier = float(np.clip(local_sum / center, 1.0, max_multiplier))
        base_capacity = capacity_min_mw + (capacity_max_mw - capacity_min_mw) * np.sqrt(candidate.suitability)
        updated.append(
            EnergyCandidate(
                candidate_id=candidate.candidate_id,
                kind=candidate.kind,
                row=candidate.row,
                col=candidate.col,
                x=candidate.x,
                y=candidate.y,
                capacity_mw=float(base_capacity * multiplier),
                suitability=candidate.suitability,
            )
        )
    return updated


def _local_source_utility(suitability: np.ndarray, cluster_radius_km: float, grid: WorldGridConfig) -> np.ndarray:
    radius_cells = max(cluster_radius_km / max(grid.cell_size_km, 1e-6), 0.0)
    if radius_cells <= 0.0:
        return suitability.copy()
    radius_int = int(np.ceil(radius_cells))
    utility = np.zeros_like(suitability, dtype=np.float32)
    for dr in range(-radius_int, radius_int + 1):
        for dc in range(-radius_int, radius_int + 1):
            distance = float(np.hypot(dr, dc))
            if distance > radius_cells:
                continue
            shifted = np.zeros_like(suitability, dtype=np.float32)
            src_r0 = max(-dr, 0)
            src_r1 = min(suitability.shape[0] - dr, suitability.shape[0])
            src_c0 = max(-dc, 0)
            src_c1 = min(suitability.shape[1] - dc, suitability.shape[1])
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc
            if src_r0 >= src_r1 or src_c0 >= src_c1:
                continue
            taper = 1.0 - 0.35 * distance / max(radius_cells, 1e-6)
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = suitability[src_r0:src_r1, src_c0:src_c1]
            utility += taper * shifted
    utility *= suitability > 1e-6
    return _normalize01(utility)


def _local_suitability_sum(suitability: np.ndarray, row: int, col: int, radius_cells: float) -> float:
    if radius_cells <= 0.0:
        return float(suitability[row, col])
    radius_int = int(np.ceil(radius_cells))
    total = 0.0
    for rr in range(max(row - radius_int, 0), min(row + radius_int + 1, suitability.shape[0])):
        for cc in range(max(col - radius_int, 0), min(col + radius_int + 1, suitability.shape[1])):
            distance = float(np.hypot(rr - row, cc - col))
            if distance > radius_cells:
                continue
            taper = 1.0 - 0.35 * distance / max(radius_cells, 1e-6)
            total += float(suitability[rr, cc]) * taper
    return total


def _select_load_candidates(
    load_density: np.ndarray,
    city: CityState,
    blocked: np.ndarray,
    config: EnergyConfig,
    grid: WorldGridConfig,
) -> list[EnergyCandidate]:
    selected: list[EnergyCandidate] = []
    min_distance_cells = max(config.min_load_node_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    for city_node in city.cities:
        if len(selected) >= config.load_node_count:
            break
        radius_cells = max(city_node.radius_km / max(grid.cell_size_km, 1e-6), min_distance_cells)
        row, col, score = _best_city_load_cell(
            load_density,
            city.city_id_map,
            blocked,
            city_node.city_id,
            city_node.row,
            city_node.col,
            radius_cells,
            selected,
            min_distance_cells,
        )
        if score <= 1e-6:
            continue
        selected.append(
            _make_candidate(
                len(selected),
                "load",
                row,
                col,
                score,
                config.load_capacity_min_mw,
                config.load_capacity_max_mw,
                grid,
            )
        )

    fill_candidates = _select_spread_load_candidates(
        load_density,
        selected,
        config,
        grid,
    )
    return fill_candidates[: config.load_node_count]


def _select_spread_load_candidates(
    load_density: np.ndarray,
    selected: list[EnergyCandidate],
    config: EnergyConfig,
    grid: WorldGridConfig,
) -> list[EnergyCandidate]:
    selected = list(selected)
    rows, cols = np.indices(load_density.shape)
    min_distance_cells = max(config.min_load_node_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    spread_radius_cells = max(config.load_spread_radius_km / max(grid.cell_size_km, 1e-6), min_distance_cells)
    while len(selected) < config.load_node_count:
        score = load_density.copy()
        if selected:
            nearest_distance = np.full(load_density.shape, np.inf, dtype=np.float32)
            hard_mask = np.zeros(load_density.shape, dtype=bool)
            for item in selected:
                distance = np.hypot(rows - item.row, cols - item.col)
                nearest_distance = np.minimum(nearest_distance, distance)
                hard_mask |= distance < min_distance_cells
            local_coverage = np.exp(-np.square(nearest_distance / max(spread_radius_cells, 1e-6)))
            score *= 1.0 - config.load_spread_penalty_weight * local_coverage
            score[hard_mask] = -1.0
        score[load_density <= 1e-6] = -1.0
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        if float(score[row, col]) <= 0.0:
            break
        selected.append(
            _make_candidate(
                len(selected),
                "load",
                int(row),
                int(col),
                float(load_density[row, col]),
                config.load_capacity_min_mw,
                config.load_capacity_max_mw,
                grid,
            )
        )
    return selected


def _best_city_load_cell(
    load_density: np.ndarray,
    city_id_map: np.ndarray,
    blocked: np.ndarray,
    city_id: int,
    city_row: int,
    city_col: int,
    radius_cells: float,
    avoid_candidates: list[EnergyCandidate],
    min_distance_cells: float,
) -> tuple[int, int, float]:
    rows, cols = np.indices(load_density.shape)
    local = np.hypot(rows - city_row, cols - city_col) <= radius_cells
    same_city = city_id_map == city_id
    mask = (same_city | local) & ~blocked
    if not mask.any():
        mask = ~blocked
    if avoid_candidates:
        avoid_mask = np.zeros_like(mask, dtype=bool)
        for item in avoid_candidates:
            avoid_mask |= np.hypot(rows - item.row, cols - item.col) < min_distance_cells
        mask &= ~avoid_mask
        if not mask.any():
            mask = ~blocked & ~avoid_mask
    score = load_density.copy()
    distance_penalty = np.clip(np.hypot(rows - city_row, cols - city_col) / max(radius_cells, 1e-6), 0.0, 1.0)
    score *= 1.0 - 0.18 * distance_penalty
    score[~mask] = -1.0
    row, col = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(row), int(col), float(load_density[row, col])


def _candidate_map(shape: tuple[int, int], *candidate_groups: list[EnergyCandidate]) -> np.ndarray:
    values = np.full(shape, -1, dtype=np.int16)
    for group in candidate_groups:
        for item in group:
            values[item.row, item.col] = item.candidate_id
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


def _distance_ramp(distance_km: np.ndarray, minimum_km: float) -> np.ndarray:
    if minimum_km <= 1e-6:
        return np.ones_like(distance_km, dtype=np.float32)
    return np.clip(distance_km / minimum_km, 0.0, 1.0).astype(np.float32)


def _half_life_decay(distance_km: np.ndarray, half_distance_km: float) -> np.ndarray:
    if half_distance_km <= 1e-6:
        return np.ones_like(distance_km, dtype=np.float32)
    return np.power(0.5, distance_km / half_distance_km).astype(np.float32)


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)
