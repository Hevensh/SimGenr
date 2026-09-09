from __future__ import annotations

import heapq
from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from world_generator.core.config import HydrologyConfig, WorldGridConfig
from world_generator.core.datatypes import HydrologyState, TerrainFeatures


D8_OFFSETS = np.array(
    [
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
    ],
    dtype=np.int32,
)

D8_DISTANCE = np.array([1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)])


def generate_hydrology(
    terrain: TerrainFeatures,
    grid: WorldGridConfig,
    config: HydrologyConfig,
) -> HydrologyState:
    if grid.cell_size_km <= 0.0 or not np.isfinite(terrain.elevation).all():
        raise ValueError("Hydrology requires finite elevations and positive cell size")
    if config.algorithm == "conditioned_v2":
        return _generate_conditioned_hydrology(terrain, grid, config)
    if config.algorithm != "legacy":
        raise ValueError(f"Unknown hydrology algorithm: {config.algorithm}")

    elevation = terrain.elevation
    flow_elevation = _smooth_for_flow(elevation, config.flow_smoothing_steps)
    flow_direction = _compute_d8_flow_direction(flow_elevation, grid.cell_size_km)
    flow_accumulation = _compute_flow_accumulation(flow_elevation, flow_direction)
    river_centerline = _extract_rivers(flow_accumulation, config.river_threshold_quantile)
    lake = _extract_lakes(flow_elevation, terrain.slope, flow_direction, flow_accumulation, river_centerline, config)
    river = _expand_rivers_by_accumulation(
        river_centerline,
        flow_accumulation,
        max_radius_cells=config.river_max_dilation_cells,
        min_strength=config.river_dilation_min_strength,
    )
    if config.lake_extra_dilation_cells > 0:
        lake = _dilate(lake, iterations=config.lake_extra_dilation_cells)
        max_elevation = float(np.quantile(flow_elevation, np.clip(config.lake_max_elevation_quantile, 0.0, 1.0)))
        lake &= (flow_elevation <= max_elevation) & (
            terrain.slope <= config.lake_slope_threshold * config.lake_expansion_slope_multiplier
        )
        lake = _remove_small_components(lake, min_cells=config.lake_min_cells)
    lake &= ~river
    water = river | lake
    water_depth = _assign_water_depth(river, lake, flow_accumulation, config)
    hydrology_elevation = _carve_hydrology_elevation(elevation, water_depth)
    distance_to_water = _distance_to_mask(water, grid.cell_size_km)
    watershed_id = _label_watersheds(flow_direction)
    flood_risk = _estimate_flood_risk(
        elevation=elevation,
        slope=terrain.slope,
        flow_accumulation=flow_accumulation,
        distance_to_water=distance_to_water,
        decay_km=config.flood_water_decay_km,
    )
    return HydrologyState(
        flow_direction=flow_direction.astype(np.int8),
        flow_accumulation=flow_accumulation.astype(np.float32),
        river_centerline=river_centerline,
        river=river,
        lake=lake,
        water_depth=water_depth.astype(np.float32),
        hydrology_elevation=hydrology_elevation.astype(np.float32),
        watershed_id=watershed_id.astype(np.int32),
        distance_to_water=distance_to_water.astype(np.float32),
        flood_risk=flood_risk.astype(np.float32),
    )


def _generate_conditioned_hydrology(
    terrain: TerrainFeatures,
    grid: WorldGridConfig,
    config: HydrologyConfig,
) -> HydrologyState:
    elevation = terrain.elevation.astype(np.float32)
    flow_elevation = _smooth_for_flow(elevation, config.flow_smoothing_steps)
    # Separate the physical spill surface from the infinitesimal routing slope.
    # Barnes et al. (2014): numerical flat resolution is not lake bathymetry.
    spill_elevation = _priority_flood_fill(flow_elevation, epsilon_m=0.0)
    conditioned_elevation = _priority_flood_fill(
        flow_elevation,
        epsilon_m=np.finfo(np.float64).eps,
    )
    depression_depth = np.maximum(spill_elevation - flow_elevation, 0.0)
    flow_direction = _compute_d8_flow_direction(conditioned_elevation, grid.cell_size_km)
    flow_accumulation = _compute_flow_accumulation(conditioned_elevation, flow_direction)
    catchment_area_km2 = flow_accumulation * float(grid.cell_size_km) ** 2
    river_centerline = (
        (catchment_area_km2 >= max(float(config.river_min_catchment_km2), 0.0))
    )
    lake = _extract_depression_lakes(
        depression_depth,
        river_centerline,
        cell_area_km2=float(grid.cell_size_km) ** 2,
        config=config,
    )
    river = _expand_rivers_by_catchment(
        river_centerline,
        catchment_area_km2,
        cell_size_km=float(grid.cell_size_km),
        max_radius_cells=config.river_max_dilation_cells,
        reference_catchment_km2=config.river_width_reference_catchment_km2,
        exponent=config.river_width_exponent,
        reference_width_m=config.river_width_reference_m,
    )
    river &= ~lake
    water = river | lake
    water_depth = _assign_water_depth(river, lake, flow_accumulation, config, catchment_area_km2)
    water_depth[lake] = depression_depth[lake]
    hydrology_elevation = _carve_hydrology_elevation(elevation, water_depth)
    # The original depression is the lake bed; it must not be excavated twice.
    hydrology_elevation[lake] = elevation[lake]
    distance_to_water = _distance_to_mask(water, grid.cell_size_km)
    watershed_id = _label_watersheds(flow_direction)
    flood_risk = _estimate_flood_risk(
        elevation=elevation,
        slope=terrain.slope,
        flow_accumulation=flow_accumulation,
        distance_to_water=distance_to_water,
        decay_km=config.flood_water_decay_km,
        flow_direction=flow_direction,
        drainage=water,
        drainage_elevation=np.where(lake, spill_elevation, elevation),
    )
    return HydrologyState(
        flow_direction=flow_direction.astype(np.int8),
        flow_accumulation=flow_accumulation.astype(np.float32),
        river_centerline=river_centerline,
        river=river,
        lake=lake,
        water_depth=water_depth.astype(np.float32),
        hydrology_elevation=hydrology_elevation.astype(np.float32),
        watershed_id=watershed_id.astype(np.int32),
        distance_to_water=distance_to_water.astype(np.float32),
        flood_risk=flood_risk.astype(np.float32),
    )


def _priority_flood_fill(elevation: np.ndarray, epsilon_m: float) -> np.ndarray:
    height, width = elevation.shape
    if height == 0 or width == 0:
        return elevation.astype(np.float32).copy()
    filled = elevation.astype(np.float64).copy()
    visited = np.zeros((height, width), dtype=bool)
    queue: list[tuple[float, int, int]] = []

    for row in range(height):
        for col in (0, width - 1):
            if not visited[row, col]:
                visited[row, col] = True
                heapq.heappush(queue, (float(filled[row, col]), row, col))
    for col in range(width):
        for row in (0, height - 1):
            if not visited[row, col]:
                visited[row, col] = True
                heapq.heappush(queue, (float(filled[row, col]), row, col))

    epsilon = max(float(epsilon_m), 0.0)
    while queue:
        current_height, row, col = heapq.heappop(queue)
        for dr, dc in D8_OFFSETS:
            rr = row + int(dr)
            cc = col + int(dc)
            if rr < 0 or rr >= height or cc < 0 or cc >= width or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            floor = current_height
            if epsilon > 0.0:
                floor = max(current_height + epsilon, float(np.nextafter(current_height, np.inf)))
            next_height = max(float(filled[rr, cc]), floor)
            filled[rr, cc] = next_height
            heapq.heappush(queue, (next_height, rr, cc))
    # Converting to float32 here erases flat-resolution increments.
    return filled


def _extract_depression_lakes(
    depression_depth: np.ndarray,
    river_centerline: np.ndarray,
    cell_area_km2: float,
    config: HydrologyConfig,
) -> np.ndarray:
    del river_centerline
    components, count = label(depression_depth > 1e-8, structure=np.ones((3, 3), dtype=np.int8))
    lake = np.zeros(depression_depth.shape, dtype=bool)
    for component_id in range(1, count + 1):
        component = components == component_id
        area = float(component.sum()) * cell_area_km2
        if (config.lake_min_area_km2 <= area <= config.lake_max_area_km2
                and float(depression_depth[component].max()) >= config.lake_min_depth_m):
            lake |= component
    # Keep lakes as continuous basins. River centerlines may pass underneath but
    # the rendered and categorical water surface remains the lake.
    return lake


def _filter_lake_components(mask: np.ndarray, min_cells: int, max_cells: int) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not mask[row, col]:
                continue
            stack = [(row, col)]
            component: list[tuple[int, int]] = []
            visited[row, col] = True
            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= height or nc < 0 or nc >= width:
                            continue
                        if visited[nr, nc] or not mask[nr, nc]:
                            continue
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if min_cells <= len(component) <= max_cells:
                for rr, cc in component:
                    kept[rr, cc] = True
    return kept


def _expand_rivers_by_catchment(
    river_centerline: np.ndarray,
    catchment_area_km2: np.ndarray,
    cell_size_km: float,
    max_radius_cells: int,
    reference_catchment_km2: float,
    exponent: float,
    reference_width_m: float = 30.0,
) -> np.ndarray:
    expanded = river_centerline.copy()
    if max_radius_cells <= 0 or not river_centerline.any():
        return expanded
    reference = max(float(reference_catchment_km2), cell_size_km * cell_size_km)
    power = max(float(exponent), 0.0)
    height, width = river_centerline.shape
    for row, col in np.argwhere(river_centerline):
        width_m = max(reference_width_m, 0.0) * (float(catchment_area_km2[row, col]) / reference) ** power
        radius = min(max_radius_cells, max(0, int(np.floor(0.5 * width_m / (1000.0 * cell_size_km)))))
        if radius <= 0:
            continue
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if np.hypot(dr, dc) > radius + 0.15:
                    continue
                rr = row + dr
                cc = col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    expanded[rr, cc] = True
    return expanded


def _compute_d8_flow_direction(elevation: np.ndarray, cell_size_km: float) -> np.ndarray:
    height, width = elevation.shape
    direction = np.full((height, width), -1, dtype=np.int8)
    for row in range(height):
        for col in range(width):
            best_dir = -1
            best_drop = 0.0
            current = float(elevation[row, col])
            for idx, (dr, dc) in enumerate(D8_OFFSETS):
                rr = row + int(dr)
                cc = col + int(dc)
                if rr < 0 or rr >= height or cc < 0 or cc >= width:
                    continue
                drop = (current - float(elevation[rr, cc])) / (D8_DISTANCE[idx] * max(1000.0 * cell_size_km, 1e-6))
                if drop > best_drop:
                    best_drop = drop
                    best_dir = idx
            direction[row, col] = best_dir
    return direction


def _smooth_for_flow(elevation: np.ndarray, steps: int) -> np.ndarray:
    smoothed = elevation.astype(np.float32).copy()
    for _ in range(max(steps, 0)):
        padded = np.pad(smoothed, 1, mode="edge")
        neighbor_sum = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        smoothed = 0.55 * smoothed + 0.45 * neighbor_sum / 8.0
    return smoothed.astype(np.float32)


def _compute_flow_accumulation(elevation: np.ndarray, flow_direction: np.ndarray) -> np.ndarray:
    height, width = elevation.shape
    accumulation = np.ones((height, width), dtype=np.float64)
    indegree = np.zeros((height, width), dtype=np.int32)
    for row, col in np.argwhere(flow_direction >= 0):
        direction = int(flow_direction[row, col])
        if direction >= len(D8_OFFSETS):
            raise ValueError("Invalid D8 flow direction")
        dr, dc = D8_OFFSETS[direction]
        rr, cc = row + int(dr), col + int(dc)
        if 0 <= rr < height and 0 <= cc < width:
            indegree[rr, cc] += 1
    queue = deque(map(tuple, np.argwhere(indegree == 0)))
    visited = 0
    while queue:
        row, col = queue.popleft()
        visited += 1
        direction = int(flow_direction[row, col])
        if direction < 0:
            continue
        dr, dc = D8_OFFSETS[direction]
        rr = row + int(dr)
        cc = col + int(dc)
        if 0 <= rr < height and 0 <= cc < width:
            accumulation[rr, cc] += accumulation[row, col]
            indegree[rr, cc] -= 1
            if indegree[rr, cc] == 0:
                queue.append((rr, cc))
    if visited != height * width:
        raise ValueError("Flow direction contains a cycle; accumulation is undefined")
    return accumulation


def _extract_rivers(flow_accumulation: np.ndarray, threshold_quantile: float) -> np.ndarray:
    threshold = float(np.quantile(flow_accumulation, np.clip(threshold_quantile, 0.0, 1.0)))
    river = flow_accumulation >= max(threshold, 2.0)
    return _thin_river_mask(river, flow_accumulation)


def _thin_river_mask(river: np.ndarray, flow_accumulation: np.ndarray) -> np.ndarray:
    thinned = river.copy()
    height, width = river.shape
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if not river[row, col]:
                continue
            local = flow_accumulation[row - 1 : row + 2, col - 1 : col + 2]
            if flow_accumulation[row, col] < np.max(local) * 0.55:
                thinned[row, col] = False
    return thinned


def _expand_rivers_by_accumulation(
    river_centerline: np.ndarray,
    flow_accumulation: np.ndarray,
    max_radius_cells: int,
    min_strength: float,
) -> np.ndarray:
    if max_radius_cells <= 0 or not river_centerline.any():
        return river_centerline.copy()
    expanded = river_centerline.copy()
    river_strength = _normalize01(np.log1p(flow_accumulation))
    height, width = river_centerline.shape
    for row, col in np.argwhere(river_centerline):
        if river_strength[row, col] < min_strength:
            continue
        local_strength = (river_strength[row, col] - min_strength) / max(1.0 - min_strength, 1e-6)
        radius = int(np.ceil(local_strength * max_radius_cells))
        radius = max(1, min(radius, max_radius_cells))
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if np.hypot(dr, dc) > radius + 0.15:
                    continue
                rr = row + dr
                cc = col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    expanded[rr, cc] = True
    return expanded


def _extract_lakes(
    elevation: np.ndarray,
    slope: np.ndarray,
    flow_direction: np.ndarray,
    flow_accumulation: np.ndarray,
    river: np.ndarray,
    config: HydrologyConfig,
) -> np.ndarray:
    max_elevation = float(np.quantile(elevation, np.clip(config.lake_max_elevation_quantile, 0.0, 1.0)))
    min_accumulation = float(
        np.quantile(flow_accumulation, np.clip(config.lake_min_accumulation_quantile, 0.0, 1.0))
    )
    sinks = flow_direction < 0
    lake_core = (
        sinks
        & (elevation <= max_elevation)
        & (slope <= config.lake_slope_threshold)
        & (flow_accumulation >= max(min_accumulation, 4.0))
    )
    lake = _dilate(lake_core, iterations=1)
    lake &= (elevation <= max_elevation) & (slope <= config.lake_slope_threshold * 1.5)
    lake = _remove_small_components(lake & ~river, min_cells=config.lake_min_cells)
    return lake


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for dr in range(3):
            for dc in range(3):
                expanded |= padded[dr : dr + result.shape[0], dc : dc + result.shape[1]]
        result = expanded
    return result


def _remove_small_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1:
        return mask
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not mask[row, col]:
                continue
            stack = [(row, col)]
            component: list[tuple[int, int]] = []
            visited[row, col] = True
            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= height or nc < 0 or nc >= width:
                            continue
                        if visited[nr, nc] or not mask[nr, nc]:
                            continue
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(component) >= min_cells:
                for rr, cc in component:
                    kept[rr, cc] = True
    return kept


def _assign_water_depth(
    river: np.ndarray,
    lake: np.ndarray,
    flow_accumulation: np.ndarray,
    config: HydrologyConfig,
    catchment_area_km2: np.ndarray | None = None,
) -> np.ndarray:
    depth = np.zeros_like(flow_accumulation, dtype=np.float32)
    if river.any():
        if catchment_area_km2 is not None:
            river_depth = np.clip(
                config.river_depth_min_m * (catchment_area_km2 / max(config.river_min_catchment_km2, 1e-9)) ** 0.2,
                config.river_depth_min_m, config.river_depth_max_m,
            )
        else:
            river_flow = np.log1p(flow_accumulation)
            river_strength = _normalize01(river_flow)
            river_depth = config.river_depth_min_m + river_strength * (
                config.river_depth_max_m - config.river_depth_min_m
            )
        depth[river] = river_depth[river]
    depth[lake] = np.maximum(depth[lake], float(config.lake_depth_m))
    return depth


def _carve_hydrology_elevation(elevation: np.ndarray, water_depth: np.ndarray) -> np.ndarray:
    hydrology_elevation = elevation.astype(np.float32).copy()
    water = water_depth > 0
    hydrology_elevation[water] -= water_depth[water]
    return hydrology_elevation


def _distance_to_mask(mask: np.ndarray, cell_size_km: float) -> np.ndarray:
    height, width = mask.shape
    if not mask.any():
        return np.full((height, width), np.inf, dtype=np.float32)
    return distance_transform_edt(~mask, sampling=float(cell_size_km)).astype(np.float32)


def _label_watersheds(flow_direction: np.ndarray) -> np.ndarray:
    height, width = flow_direction.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    outlet_to_id: dict[tuple[int, int], int] = {}

    for row in range(height):
        for col in range(width):
            path: list[tuple[int, int]] = []
            current = (row, col)
            seen: set[tuple[int, int]] = set()
            while True:
                rr, cc = current
                if labels[rr, cc] >= 0:
                    label = int(labels[rr, cc])
                    break
                if current in seen:
                    label = _get_outlet_label(current, outlet_to_id)
                    break
                seen.add(current)
                path.append(current)
                direction = int(flow_direction[rr, cc])
                if direction < 0:
                    label = _get_outlet_label(current, outlet_to_id)
                    break
                dr, dc = D8_OFFSETS[direction]
                nr = rr + int(dr)
                nc = cc + int(dc)
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    label = _get_outlet_label(current, outlet_to_id)
                    break
                current = (nr, nc)
            for rr, cc in path:
                labels[rr, cc] = label
    return labels


def _get_outlet_label(outlet: tuple[int, int], outlet_to_id: dict[tuple[int, int], int]) -> int:
    if outlet not in outlet_to_id:
        outlet_to_id[outlet] = len(outlet_to_id)
    return outlet_to_id[outlet]


def _estimate_flood_risk(
    elevation: np.ndarray,
    slope: np.ndarray,
    flow_accumulation: np.ndarray,
    distance_to_water: np.ndarray,
    decay_km: float,
    flow_direction: np.ndarray | None = None,
    drainage: np.ndarray | None = None,
    drainage_elevation: np.ndarray | None = None,
) -> np.ndarray:
    water_proximity = np.exp(-distance_to_water / max(decay_km, 1e-6))
    if flow_direction is not None and drainage is not None:
        # HAND is a terrain susceptibility proxy, not a return-period probability.
        hand = _height_above_drainage(elevation, flow_direction, drainage, drainage_elevation)
        risk = water_proximity * np.exp(-hand / 5.0)
        risk[drainage] = 1.0
        return np.clip(risk, 0.0, 1.0).astype(np.float32)
    low_slope = 1.0 - _normalize01(slope)
    low_elevation = 1.0 - _normalize01(elevation)
    flow_pressure = _normalize01(np.log1p(flow_accumulation))
    risk = 0.42 * water_proximity + 0.24 * low_slope + 0.18 * low_elevation + 0.16 * flow_pressure
    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def _height_above_drainage(
    elevation: np.ndarray, flow_direction: np.ndarray, drainage: np.ndarray,
    drainage_elevation: np.ndarray | None = None,
) -> np.ndarray:
    """Vertical distance to the first drainage cell along each D8 flow path."""
    height, width = elevation.shape
    reference = np.full(elevation.shape, np.nan, dtype=np.float64)
    surface = elevation if drainage_elevation is None else drainage_elevation
    reference[drainage] = surface[drainage]
    for row in range(height):
        for col in range(width):
            if not np.isnan(reference[row, col]):
                continue
            path: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            rr, cc = row, col
            while np.isnan(reference[rr, cc]):
                if (rr, cc) in seen:
                    raise ValueError("Flow direction contains a cycle")
                seen.add((rr, cc))
                path.append((rr, cc))
                direction = int(flow_direction[rr, cc])
                if direction < 0:
                    reference[rr, cc] = -np.inf
                    break
                dr, dc = D8_OFFSETS[direction]
                nr, nc = rr + int(dr), cc + int(dc)
                if not (0 <= nr < height and 0 <= nc < width):
                    reference[rr, cc] = -np.inf
                    break
                rr, cc = nr, nc
            value = reference[rr, cc]
            for pr, pc in path:
                reference[pr, pc] = value
    return np.maximum(elevation.astype(np.float64) - reference, 0.0)


def _normalize01(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax - vmin < 1e-12:
        return np.zeros_like(finite, dtype=np.float32)
    return ((finite - vmin) / (vmax - vmin)).astype(np.float32)
