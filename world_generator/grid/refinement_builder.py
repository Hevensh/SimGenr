from __future__ import annotations

import numpy as np

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import (
    GridEdge,
    GridNodeState,
    GridTopologyState,
    HydrologyState,
    RefinedGridTopologyState,
    StaticLandState,
    TerrainFeatures,
)
from world_generator.grid.corridor import (
    edge_maps,
    path_length_km,
    risk_path,
    transit_risk_cost,
)


def refine_grid_topology(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid_nodes: GridNodeState,
    topology: GridTopologyState,
    grid: WorldGridConfig,
    config: PowerGridConfig,
) -> RefinedGridTopologyState:
    refined_buses = list(grid_nodes.buses)
    refined_edges: list[GridEdge] = []
    bus_by_id = {bus.bus_id: bus for bus in refined_buses}
    risk_cost = transit_risk_cost(terrain, hydrology, land, config)

    for edge in topology.edges:
        start_bus = bus_by_id[edge.from_bus]
        end_bus = bus_by_id[edge.to_bus]
        route_rows, route_cols = risk_path(start_bus.row, start_bus.col, end_bus.row, end_bus.col, risk_cost)
        route_length_km = path_length_km(route_rows, route_cols, grid.cell_size_km)
        local_cost = float(np.mean(risk_cost[route_rows, route_cols]))
        refined_edges.append(
            GridEdge(
                edge_id=len(refined_edges),
                from_bus=edge.from_bus,
                to_bus=edge.to_bus,
                length_km=route_length_km,
                route_cost=route_length_km * (1.0 + local_cost),
                is_redundant=edge.is_redundant,
                path_rows=tuple(int(v) for v in route_rows),
                path_cols=tuple(int(v) for v in route_cols),
            )
        )

    refined_line_route_map, refined_grid_edge_map = edge_maps(topology.routing_cost.shape, refined_edges)
    transit_bus_map = np.full(topology.routing_cost.shape, -1, dtype=np.int16)
    return RefinedGridTopologyState(
        refined_line_route_map=refined_line_route_map.astype(np.float32),
        refined_grid_edge_map=refined_grid_edge_map.astype(np.int16),
        transit_bus_map=transit_bus_map.astype(np.int16),
        refined_buses=tuple(refined_buses),
        refined_edges=tuple(refined_edges),
    )

