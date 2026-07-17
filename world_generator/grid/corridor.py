from __future__ import annotations

import heapq

import numpy as np

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import BranchElectricalParam, GridBus, GridEdge, HydrologyState, StaticLandState, TerrainFeatures


def transit_risk_cost(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    config: PowerGridConfig,
) -> np.ndarray:
    water = hydrology.river | hydrology.lake
    slope_n = normalize01(terrain.slope)
    terrain_cost = np.clip(land.terrain_cost, 0.0, 1.0)
    cost = (
        1.0
        + config.transit_flood_penalty * np.clip(hydrology.flood_risk, 0.0, 1.0)
        + config.transit_water_penalty * water.astype(np.float32)
        + config.transit_protected_penalty * land.protected.astype(np.float32)
        + config.transit_slope_penalty * slope_n
        + config.transit_terrain_cost_penalty * terrain_cost
    )
    return np.clip(cost, 1e-3, None).astype(np.float32)


def risk_path(row0: int, col0: int, row1: int, col1: int, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = cost.shape
    start = (int(np.clip(row0, 0, shape[0] - 1)), int(np.clip(col0, 0, shape[1] - 1)))
    goal = (int(np.clip(row1, 0, shape[0] - 1)), int(np.clip(col1, 0, shape[1] - 1)))
    if start == goal:
        return np.asarray([start[0]], dtype=np.int16), np.asarray([start[1]], dtype=np.int16)

    frontier: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor, step_distance in neighbors(current, shape):
            step_cost = step_distance * (0.5 * float(cost[current]) + 0.5 * float(cost[neighbor]))
            new_cost = cost_so_far[current] + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(frontier, (priority, counter, neighbor))
                came_from[neighbor] = current
    if goal not in came_from:
        return line_cells(row0, col0, row1, col1)
    path = []
    current: tuple[int, int] | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    rows = np.asarray([item[0] for item in path], dtype=np.int16)
    cols = np.asarray([item[1] for item in path], dtype=np.int16)
    return rows, cols


def path_length_km(rows: np.ndarray, cols: np.ndarray, cell_size_km: float) -> float:
    if rows.size <= 1:
        return 0.0
    dr = np.diff(rows.astype(np.float32))
    dc = np.diff(cols.astype(np.float32))
    return float(np.hypot(dr, dc).sum() * cell_size_km)


def edge_maps(shape: tuple[int, int], edges: list[GridEdge]) -> tuple[np.ndarray, np.ndarray]:
    route_map = np.zeros(shape, dtype=np.float32)
    edge_map = np.full(shape, -1, dtype=np.int16)
    for edge in edges:
        for row, col in zip(edge.path_rows, edge.path_cols):
            route_map[row, col] += 1.0
            edge_map[row, col] = edge.edge_id
    if route_map.max() > 0.0:
        route_map /= route_map.max()
    return route_map, edge_map


def edge_geometry_conflicts(
    branch_a: BranchElectricalParam,
    branch_b: BranchElectricalParam,
    edge_by_id: dict[int, GridEdge],
    bus_by_id: dict[int, GridBus],
    *,
    near_radius_cells: int = 1,
) -> bool:
    if edge_geometry_crosses(branch_a, branch_b, edge_by_id, bus_by_id):
        return True
    edge_a = edge_by_id.get(int(branch_a.edge_id))
    edge_b = edge_by_id.get(int(branch_b.edge_id))
    if edge_a is None or edge_b is None:
        return False
    cells_a = set(zip(edge_a.path_rows, edge_a.path_cols))
    cells_b = set(zip(edge_b.path_rows, edge_b.path_cols))
    endpoint_cells = {
        (bus_by_id[int(branch_a.from_bus)].row, bus_by_id[int(branch_a.from_bus)].col),
        (bus_by_id[int(branch_a.to_bus)].row, bus_by_id[int(branch_a.to_bus)].col),
        (bus_by_id[int(branch_b.from_bus)].row, bus_by_id[int(branch_b.from_bus)].col),
        (bus_by_id[int(branch_b.to_bus)].row, bus_by_id[int(branch_b.to_bus)].col),
    }
    if (cells_a & cells_b) - endpoint_cells:
        return True
    filtered_a = cells_a - endpoint_cells
    filtered_b = cells_b - endpoint_cells
    for row, col in filtered_a:
        for drow in range(-near_radius_cells, near_radius_cells + 1):
            for dcol in range(-near_radius_cells, near_radius_cells + 1):
                if drow == 0 and dcol == 0:
                    continue
                if (row + drow, col + dcol) in filtered_b:
                    return True
    return False


def edge_geometry_crosses(
    branch_a: BranchElectricalParam,
    branch_b: BranchElectricalParam,
    edge_by_id: dict[int, GridEdge],
    bus_by_id: dict[int, GridBus],
) -> bool:
    points_a = edge_points(branch_a, edge_by_id, bus_by_id)
    points_b = edge_points(branch_b, edge_by_id, bus_by_id)
    if len(points_a) < 2 or len(points_b) < 2:
        return False
    for a0, a1 in zip(points_a[:-1], points_a[1:]):
        for b0, b1 in zip(points_b[:-1], points_b[1:]):
            if segments_cross(a0, a1, b0, b1):
                return True
    return False


def edge_points(
    branch: BranchElectricalParam,
    edge_by_id: dict[int, GridEdge],
    bus_by_id: dict[int, GridBus],
) -> list[tuple[float, float]]:
    edge = edge_by_id.get(int(branch.edge_id))
    if edge is not None and len(edge.path_rows) >= 2:
        return [(float(col), float(row)) for row, col in zip(edge.path_rows, edge.path_cols)]
    start = bus_by_id.get(int(branch.from_bus))
    end = bus_by_id.get(int(branch.to_bus))
    if start is None or end is None:
        return []
    return [(float(start.col), float(start.row)), (float(end.col), float(end.row))]


def segments_cross(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    return orientation(p1, p2, q1) * orientation(p1, p2, q2) < 0.0 and orientation(q1, q2, p1) * orientation(q1, q2, p2) < 0.0


def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def neighbors(node: tuple[int, int], shape: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    row, col = node
    values = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = row + dr
            cc = col + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                values.append(((rr, cc), float(np.hypot(dr, dc))))
    return values


def heuristic(node: tuple[int, int], goal: tuple[int, int]) -> float:
    return float(np.hypot(node[0] - goal[0], node[1] - goal[1]))


def line_cells(row0: int, col0: int, row1: int, col1: int) -> tuple[np.ndarray, np.ndarray]:
    steps = max(abs(row1 - row0), abs(col1 - col0), 1) + 1
    rows = np.rint(np.linspace(row0, row1, steps)).astype(np.int16)
    cols = np.rint(np.linspace(col0, col1, steps)).astype(np.int16)
    return rows, cols


def normalize01(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmax <= vmin + 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)
