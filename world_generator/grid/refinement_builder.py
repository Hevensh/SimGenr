from __future__ import annotations

import numpy as np

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import GridBus, GridEdge, GridNodeState, GridTopologyState, RefinedGridTopologyState


def refine_grid_topology(
    grid_nodes: GridNodeState,
    topology: GridTopologyState,
    grid: WorldGridConfig,
    config: PowerGridConfig,
) -> RefinedGridTopologyState:
    refined_buses = list(grid_nodes.buses)
    refined_edges: list[GridEdge] = []
    bus_by_id = {bus.bus_id: bus for bus in refined_buses}
    max_segment_cells = max(config.max_line_segment_km / max(grid.cell_size_km, 1e-6), 1.0)

    for edge in topology.edges:
        node_ids = [edge.from_bus]
        split_indices = _split_indices(edge.length_km, grid.cell_size_km, len(edge.path_rows), max_segment_cells)
        for index in split_indices:
            row = edge.path_rows[index]
            col = edge.path_cols[index]
            transit = GridBus(
                bus_id=len(refined_buses),
                kind="transit_bus",
                row=int(row),
                col=int(col),
                x=float(col / max(grid.width - 1, 1)),
                y=float(row / max(grid.height - 1, 1)),
                capacity_mw=0.0,
                suitability=1.0,
                externality_score=0.0,
                source_kind="transit",
                source_id=edge.edge_id,
            )
            refined_buses.append(transit)
            bus_by_id[transit.bus_id] = transit
            node_ids.append(transit.bus_id)
        node_ids.append(edge.to_bus)

        for from_bus, to_bus in zip(node_ids[:-1], node_ids[1:]):
            from_node = bus_by_id[from_bus]
            to_node = bus_by_id[to_bus]
            path_rows, path_cols = _line_cells(from_node.row, from_node.col, to_node.row, to_node.col)
            length_km = float(np.hypot(from_node.row - to_node.row, from_node.col - to_node.col) * grid.cell_size_km)
            local_cost = float(np.mean(topology.routing_cost[path_rows, path_cols]))
            refined_edges.append(
                GridEdge(
                    edge_id=len(refined_edges),
                    from_bus=from_bus,
                    to_bus=to_bus,
                    length_km=length_km,
                    route_cost=length_km * (1.0 + local_cost),
                    is_redundant=edge.is_redundant,
                    path_rows=tuple(int(v) for v in path_rows),
                    path_cols=tuple(int(v) for v in path_cols),
                )
            )

    refined_line_route_map, refined_grid_edge_map = _edge_maps(topology.routing_cost.shape, refined_edges)
    transit_bus_map = _transit_bus_map(topology.routing_cost.shape, refined_buses, original_count=len(grid_nodes.buses))
    return RefinedGridTopologyState(
        refined_line_route_map=refined_line_route_map.astype(np.float32),
        refined_grid_edge_map=refined_grid_edge_map.astype(np.int16),
        transit_bus_map=transit_bus_map.astype(np.int16),
        refined_buses=tuple(refined_buses),
        refined_edges=tuple(refined_edges),
    )


def _split_indices(length_km: float, cell_size_km: float, path_length: int, max_segment_cells: float) -> list[int]:
    length_cells = length_km / max(cell_size_km, 1e-6)
    if length_cells <= max_segment_cells:
        return []
    segment_count = int(np.ceil(length_cells / max_segment_cells))
    if segment_count <= 1:
        return []
    return [int(round(i * (path_length - 1) / segment_count)) for i in range(1, segment_count)]


def _edge_maps(shape: tuple[int, int], edges: list[GridEdge]) -> tuple[np.ndarray, np.ndarray]:
    route_map = np.zeros(shape, dtype=np.float32)
    edge_map = np.full(shape, -1, dtype=np.int16)
    for edge in edges:
        for row, col in zip(edge.path_rows, edge.path_cols):
            route_map[row, col] += 1.0
            edge_map[row, col] = edge.edge_id
    if route_map.max() > 0.0:
        route_map /= route_map.max()
    return route_map, edge_map


def _transit_bus_map(shape: tuple[int, int], buses: list[GridBus], original_count: int) -> np.ndarray:
    values = np.full(shape, -1, dtype=np.int16)
    for bus in buses[original_count:]:
        values[bus.row, bus.col] = bus.bus_id
    return values


def _line_cells(row0: int, col0: int, row1: int, col1: int) -> tuple[np.ndarray, np.ndarray]:
    steps = max(abs(row1 - row0), abs(col1 - col0), 1) + 1
    rows = np.rint(np.linspace(row0, row1, steps)).astype(np.int16)
    cols = np.rint(np.linspace(col0, col1, steps)).astype(np.int16)
    return rows, cols
