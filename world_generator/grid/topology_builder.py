from __future__ import annotations

import itertools

import numpy as np

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import GridBus, GridEdge, GridNodeState, GridTopologyState, HydrologyState, StaticLandState, TerrainFeatures
from world_generator.grid.corridor import line_cells, normalize01


def build_grid_topology(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid_nodes: GridNodeState,
    grid: WorldGridConfig,
    config: PowerGridConfig,
) -> GridTopologyState:
    routing_cost = _routing_cost_surface(terrain, hydrology, land, config)
    candidates = _candidate_edges(grid_nodes.buses, routing_cost, grid, config)
    tree_edges = _minimum_spanning_edges(candidates, len(grid_nodes.buses))
    redundant_edges = _redundant_edges(candidates, tree_edges, grid_nodes.buses, config)
    tree_edges, redundant_edges = _repair_crossing_edges(
        tree_edges,
        redundant_edges,
        grid_nodes.buses,
        routing_cost,
        grid,
    )
    redundant_edges = _prune_bad_redundant_edges(tree_edges, redundant_edges, grid_nodes.buses)
    redundant_edges = _top_up_redundant_edges(candidates, tree_edges, redundant_edges, grid_nodes.buses, config)
    edges = _make_edges(tree_edges, redundant_edges)
    line_route_map, grid_edge_map = _edge_maps(routing_cost.shape, edges)
    return GridTopologyState(
        routing_cost=routing_cost.astype(np.float32),
        line_route_map=line_route_map.astype(np.float32),
        grid_edge_map=grid_edge_map.astype(np.int16),
        edges=tuple(edges),
    )


def _routing_cost_surface(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    config: PowerGridConfig,
) -> np.ndarray:
    water = (hydrology.river | hydrology.lake).astype(np.float32)
    protected = land.protected.astype(np.float32)
    slope = normalize01(terrain.slope)
    terrain_cost = np.clip(land.terrain_cost, 0.0, 1.0)
    cost = (
        1.0
        + config.line_water_penalty * water
        + config.line_protected_penalty * protected
        + config.line_slope_penalty * slope
        + config.line_terrain_cost_penalty * terrain_cost
    )
    return normalize01(cost)


def _candidate_edges(
    buses: tuple[GridBus, ...],
    routing_cost: np.ndarray,
    grid: WorldGridConfig,
    config: PowerGridConfig,
) -> list[dict[str, object]]:
    candidates: dict[tuple[int, int], dict[str, object]] = {}
    coords = np.asarray([(bus.row, bus.col) for bus in buses], dtype=np.float32)
    for index, bus in enumerate(buses):
        distances = np.hypot(coords[:, 0] - bus.row, coords[:, 1] - bus.col)
        order = np.argsort(distances)
        for other_index in order[1 : config.candidate_knn + 1]:
            a, b = sorted((index, int(other_index)))
            if a == b or (a, b) in candidates:
                continue
            candidates[(a, b)] = _make_candidate_edge(buses[a], buses[b], routing_cost, grid)

    _add_load_cluster_generation_edges(candidates, buses, routing_cost, grid, config)

    if len(candidates) < max(len(buses) - 1, 1):
        for a, b in itertools.combinations(range(len(buses)), 2):
            key = (a, b)
            if key in candidates:
                continue
            candidates[key] = _make_candidate_edge(buses[a], buses[b], routing_cost, grid)
    return sorted(candidates.values(), key=lambda item: float(item["route_cost"]))


def _make_candidate_edge(
    bus_a: GridBus,
    bus_b: GridBus,
    routing_cost: np.ndarray,
    grid: WorldGridConfig,
    *,
    priority: str = "normal",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    path_rows, path_cols = line_cells(bus_a.row, bus_a.col, bus_b.row, bus_b.col)
    length_km = float(np.hypot(bus_a.row - bus_b.row, bus_a.col - bus_b.col) * grid.cell_size_km)
    route_cost = float(length_km * (1.0 + np.mean(routing_cost[path_rows, path_cols])))
    edge: dict[str, object] = {
        "from_bus": bus_a.bus_id,
        "to_bus": bus_b.bus_id,
        "length_km": length_km,
        "route_cost": route_cost,
        "path_rows": tuple(int(v) for v in path_rows),
        "path_cols": tuple(int(v) for v in path_cols),
        "priority": priority,
    }
    if metadata:
        edge.update(metadata)
    return edge


def _add_load_cluster_generation_edges(
    candidates: dict[tuple[int, int], dict[str, object]],
    buses: tuple[GridBus, ...],
    routing_cost: np.ndarray,
    grid: WorldGridConfig,
    config: PowerGridConfig,
) -> None:
    load_buses = [bus for bus in buses if bus.kind == "load_bus"]
    gen_buses = [bus for bus in buses if bus.kind in {"wind_bus", "pv_bus", "thermal_bus"}]
    load_clusters = _load_candidate_components(load_buses, candidates, config, grid)
    generation_clusters = _bus_clusters(gen_buses, config.generation_cluster_radius_km, grid)
    if not load_clusters or not generation_clusters:
        return

    for load_cluster_id, load_cluster in enumerate(load_clusters):
        if len(load_cluster) < config.load_cluster_min_loads:
            continue
        load_center = _cluster_center(load_cluster)
        ranked_generation_clusters = sorted(
            generation_clusters,
            key=lambda cluster: _center_distance(load_center, _cluster_center(cluster), grid),
        )
        for generation_cluster_id, generation_cluster in enumerate(ranked_generation_clusters[: config.load_cluster_generation_links]):
            for load_bus in load_cluster:
                best_edge: dict[str, object] | None = None
                best_key: tuple[int, int] | None = None
                for gen_bus in generation_cluster:
                    key = tuple(sorted((load_bus.bus_id, gen_bus.bus_id)))
                    edge = _make_candidate_edge(
                        load_bus,
                        gen_bus,
                        routing_cost,
                        grid,
                        priority="load_cluster_generation",
                        metadata={
                            "load_cluster_id": load_cluster_id,
                            "generation_cluster_id": generation_cluster_id,
                        },
                    )
                    if best_edge is None or float(edge["route_cost"]) < float(best_edge["route_cost"]):
                        best_edge = edge
                        best_key = key
                if best_edge is not None and best_key is not None:
                    candidates[best_key] = best_edge


def _load_candidate_components(
    load_buses: list[GridBus],
    candidates: dict[tuple[int, int], dict[str, object]],
    config: PowerGridConfig,
    grid: WorldGridConfig,
) -> list[list[GridBus]]:
    if not load_buses:
        return []
    load_by_id = {bus.bus_id: bus for bus in load_buses}
    max_link_km = max(config.load_cluster_radius_km, grid.cell_size_km)
    adjacency = {bus.bus_id: set() for bus in load_buses}
    for edge in candidates.values():
        from_bus = int(edge["from_bus"])
        to_bus = int(edge["to_bus"])
        if from_bus not in load_by_id or to_bus not in load_by_id:
            continue
        if float(edge["length_km"]) > max_link_km:
            continue
        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)

    clusters: list[list[GridBus]] = []
    remaining = set(adjacency)
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for neighbor in sorted(adjacency[current] & remaining):
                remaining.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        clusters.append([load_by_id[bus_id] for bus_id in sorted(component)])
    return clusters


def _bus_clusters(buses: list[GridBus], radius_km: float, grid: WorldGridConfig) -> list[list[GridBus]]:
    if not buses:
        return []
    radius_cells = max(radius_km / max(grid.cell_size_km, 1e-6), 1.0)
    remaining = set(range(len(buses)))
    clusters = []
    while remaining:
        start = remaining.pop()
        cluster_indices = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            nearby = [
                index
                for index in list(remaining)
                if np.hypot(buses[current].row - buses[index].row, buses[current].col - buses[index].col) <= radius_cells
            ]
            for index in nearby:
                remaining.remove(index)
                cluster_indices.add(index)
                frontier.append(index)
        clusters.append([buses[index] for index in sorted(cluster_indices)])
    return clusters


def _cluster_center(cluster: list[GridBus]) -> tuple[float, float]:
    return (
        float(np.mean([bus.row for bus in cluster])),
        float(np.mean([bus.col for bus in cluster])),
    )


def _center_distance(center_a: tuple[float, float], center_b: tuple[float, float], grid: WorldGridConfig) -> float:
    return float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]) * grid.cell_size_km)


def _minimum_spanning_edges(candidates: list[dict[str, object]], bus_count: int) -> list[dict[str, object]]:
    parent = list(range(bus_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree = []
    for edge in candidates:
        a = int(edge["from_bus"])
        b = int(edge["to_bus"])
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            continue
        parent[root_b] = root_a
        tree.append(edge)
        if len(tree) >= bus_count - 1:
            break
    return tree


def _redundant_edges(
    candidates: list[dict[str, object]],
    tree_edges: list[dict[str, object]],
    buses: tuple[GridBus, ...],
    config: PowerGridConfig,
) -> list[dict[str, object]]:
    tree_pairs = {_edge_pair(edge) for edge in tree_edges}
    tree_degree = {bus.bus_id: 0 for bus in buses}
    for edge in tree_edges:
        tree_degree[int(edge["from_bus"])] += 1
        tree_degree[int(edge["to_bus"])] += 1
    target = max(int(round(len(tree_edges) * config.redundancy_ratio)), 1)
    redundant = []
    available = [edge for edge in candidates if _edge_pair(edge) not in tree_pairs]
    if not available:
        return redundant
    seen_pairs = set(tree_pairs)
    max_route_cost = max(float(edge["route_cost"]) for edge in available)
    bus_by_id = {bus.bus_id: bus for bus in buses}
    external_degree = _external_generation_degree(tree_edges, bus_by_id)
    selected_edges = list(tree_edges)

    forced = [edge for edge in available if edge.get("priority") == "load_cluster_generation"]
    forced_edges = _select_forced_load_cluster_edges(
        forced,
        selected_edges,
        seen_pairs,
        bus_by_id,
        external_degree,
        config,
        max_route_cost,
    )
    for edge in forced_edges:
        redundant.append(edge)
        selected_edges.append(edge)
        seen_pairs.add(_edge_pair(edge))

    remaining = [edge for edge in available if _edge_pair(edge) not in seen_pairs]
    while remaining and len(redundant) < max(target, len(forced_edges)):
        edge = min(
            remaining,
            key=lambda item: _redundancy_score(item, bus_by_id, tree_degree, external_degree, config, max_route_cost),
        )
        pair = _edge_pair(edge)
        if pair in seen_pairs:
            remaining = [item for item in remaining if _edge_pair(item) != pair]
            continue
        redundant.append(edge)
        selected_edges.append(edge)
        seen_pairs.add(pair)
        _increment_external_generation_degree(edge, bus_by_id, external_degree)
        remaining = [item for item in remaining if _edge_pair(item) != pair]
    return redundant


def _select_forced_load_cluster_edges(
    forced_edges: list[dict[str, object]],
    selected_edges: list[dict[str, object]],
    seen_pairs: set[tuple[int, int]],
    bus_by_id: dict[int, GridBus],
    external_degree: dict[int, int],
    config: PowerGridConfig,
    max_route_cost: float,
) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for edge in forced_edges:
        if _edge_pair(edge) in seen_pairs:
            continue
        key = (int(edge.get("load_cluster_id", -1)), int(edge.get("generation_cluster_id", -1)))
        groups.setdefault(key, []).append(edge)

    selected: list[dict[str, object]] = []
    for key in sorted(groups):
        options = [edge for edge in groups[key] if _edge_pair(edge) not in seen_pairs]
        if not options:
            continue
        edge = min(
            options,
            key=lambda item: _forced_load_cluster_score(
                item,
                selected_edges + selected,
                bus_by_id,
                external_degree,
                config,
                max_route_cost,
            ),
        )
        selected.append(edge)
        seen_pairs.add(_edge_pair(edge))
        _increment_external_generation_degree(edge, bus_by_id, external_degree)
    return selected


def _forced_load_cluster_score(
    edge: dict[str, object],
    selected_edges: list[dict[str, object]],
    bus_by_id: dict[int, GridBus],
    external_degree: dict[int, int],
    config: PowerGridConfig,
    max_route_cost: float,
) -> float:
    from_bus = int(edge["from_bus"])
    to_bus = int(edge["to_bus"])
    route_cost = float(edge["route_cost"]) / max(max_route_cost, 1e-6)
    load_degree, generation_degree = _edge_load_generation_degrees(edge, bus_by_id, external_degree)
    degree_penalty = (
        config.load_cluster_external_degree_penalty * load_degree
        + config.generation_cluster_external_degree_penalty * generation_degree
    )
    return route_cost + degree_penalty


def _external_generation_degree(edges: list[dict[str, object]], bus_by_id: dict[int, GridBus]) -> dict[int, int]:
    values = {bus_id: 0 for bus_id in bus_by_id}
    for edge in edges:
        _increment_external_generation_degree(edge, bus_by_id, values)
    return values


def _increment_external_generation_degree(
    edge: dict[str, object],
    bus_by_id: dict[int, GridBus],
    values: dict[int, int],
) -> None:
    from_bus = int(edge["from_bus"])
    to_bus = int(edge["to_bus"])
    kinds = {bus_by_id[from_bus].kind, bus_by_id[to_bus].kind}
    if "load_bus" in kinds and ({"wind_bus", "pv_bus", "thermal_bus"} & kinds):
        values[from_bus] = values.get(from_bus, 0) + 1
        values[to_bus] = values.get(to_bus, 0) + 1


def _edge_load_generation_degrees(
    edge: dict[str, object],
    bus_by_id: dict[int, GridBus],
    external_degree: dict[int, int],
) -> tuple[int, int]:
    from_bus = int(edge["from_bus"])
    to_bus = int(edge["to_bus"])
    load_degree = 0
    generation_degree = 0
    for bus_id in (from_bus, to_bus):
        kind = bus_by_id[bus_id].kind
        if kind == "load_bus":
            load_degree += external_degree.get(bus_id, 0)
        elif kind in {"wind_bus", "pv_bus", "thermal_bus"}:
            generation_degree += external_degree.get(bus_id, 0)
    return load_degree, generation_degree


def _redundancy_score(
    edge: dict[str, object],
    bus_by_id: dict[int, GridBus],
    tree_degree: dict[int, int],
    external_degree: dict[int, int],
    config: PowerGridConfig,
    max_route_cost: float,
) -> float:
    from_bus = int(edge["from_bus"])
    to_bus = int(edge["to_bus"])
    pair_priority = _pair_priority(bus_by_id[from_bus].kind, bus_by_id[to_bus].kind)
    route_cost = float(edge["route_cost"]) / max(max_route_cost, 1e-6)
    length_penalty = float(edge["length_km"]) / max(float(edge["length_km"]) + 12.0, 1e-6)
    degree_bonus = 0.18 * float(tree_degree[from_bus] <= 1) + 0.18 * float(tree_degree[to_bus] <= 1)
    load_degree, generation_degree = _edge_load_generation_degrees(edge, bus_by_id, external_degree)
    external_degree_penalty = (
        0.35 * config.load_cluster_external_degree_penalty * load_degree
        + 0.35 * config.generation_cluster_external_degree_penalty * generation_degree
    )
    return route_cost + 0.35 * length_penalty + external_degree_penalty - 0.55 * pair_priority - degree_bonus


def _pair_priority(kind_a: str, kind_b: str) -> float:
    kinds = {kind_a, kind_b}
    if "load_bus" in kinds and ({"wind_bus", "pv_bus", "thermal_bus"} & kinds):
        return 1.0
    if kinds == {"load_bus"}:
        return 0.85
    if "thermal_bus" in kinds and ({"wind_bus", "pv_bus"} & kinds):
        return 0.55
    if kinds <= {"wind_bus", "pv_bus"}:
        return 0.35
    return 0.15


def _repair_crossing_edges(
    tree_edges: list[dict[str, object]],
    redundant_edges: list[dict[str, object]],
    buses: tuple[GridBus, ...],
    routing_cost: np.ndarray,
    grid: WorldGridConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(tree_edges) + len(redundant_edges) < 2:
        return tree_edges, redundant_edges
    bus_by_id = {bus.bus_id: bus for bus in buses}
    edges = list(tree_edges) + list(redundant_edges)

    for _ in range(2):
        changed = False
        for tree_index, redundant_index in itertools.product(range(len(tree_edges)), range(len(tree_edges), len(edges))):
            if not _edges_cross(edges[tree_index], edges[redundant_index]):
                continue
            replacement = _redundant_reconnection(
                edges,
                tree_index,
                redundant_index,
                bus_by_id,
                routing_cost,
                grid,
            )
            if replacement is None:
                continue
            edges[redundant_index] = replacement
            changed = True
            break
        if not changed:
            break
    return edges[: len(tree_edges)], edges[len(tree_edges) :]


def _redundant_reconnection(
    edges: list[dict[str, object]],
    tree_index: int,
    redundant_index: int,
    bus_by_id: dict[int, GridBus],
    routing_cost: np.ndarray,
    grid: WorldGridConfig,
) -> dict[str, object] | None:
    tree_edge = edges[tree_index]
    redundant_edge = edges[redundant_index]
    tree_endpoints = (int(tree_edge["from_bus"]), int(tree_edge["to_bus"]))
    redundant_endpoints = (int(redundant_edge["from_bus"]), int(redundant_edge["to_bus"]))
    existing_pairs = {_edge_pair(edge) for index, edge in enumerate(edges) if index != redundant_index}
    best_edge: dict[str, object] | None = None
    best_cost = float("inf")
    for from_bus in tree_endpoints:
        for to_bus in redundant_endpoints:
            if from_bus == to_bus:
                continue
            key = tuple(sorted((from_bus, to_bus)))
            if key in existing_pairs:
                continue
            edge = _make_candidate_edge(bus_by_id[from_bus], bus_by_id[to_bus], routing_cost, grid)
            other_edges = [item for index, item in enumerate(edges) if index != redundant_index]
            if any(_edges_cross(edge, other) for other in other_edges):
                continue
            if float(edge["route_cost"]) <= float(redundant_edge["route_cost"]) * 1.35 and float(edge["route_cost"]) < best_cost:
                best_cost = float(edge["route_cost"])
                best_edge = edge
    return best_edge


def _prune_bad_redundant_edges(
    tree_edges: list[dict[str, object]],
    redundant_edges: list[dict[str, object]],
    buses: tuple[GridBus, ...],
) -> list[dict[str, object]]:
    if not redundant_edges:
        return redundant_edges
    bus_by_id = {bus.bus_id: bus for bus in buses}
    all_pairs = {_edge_pair(edge) for edge in tree_edges + redundant_edges}
    selected_edges = list(tree_edges)
    kept = []
    for edge in redundant_edges:
        if _is_bad_redundant_edge(edge, all_pairs, bus_by_id) or any(_edges_cross(edge, selected) for selected in selected_edges):
            all_pairs.remove(_edge_pair(edge))
            continue
        kept.append(edge)
        selected_edges.append(edge)
    return kept


def _top_up_redundant_edges(
    candidates: list[dict[str, object]],
    tree_edges: list[dict[str, object]],
    redundant_edges: list[dict[str, object]],
    buses: tuple[GridBus, ...],
    config: PowerGridConfig,
) -> list[dict[str, object]]:
    target = max(int(round(len(tree_edges) * config.redundancy_ratio)), 1)
    if len(redundant_edges) >= target:
        return redundant_edges
    bus_by_id = {bus.bus_id: bus for bus in buses}
    tree_degree = {bus.bus_id: 0 for bus in buses}
    for edge in tree_edges:
        tree_degree[int(edge["from_bus"])] += 1
        tree_degree[int(edge["to_bus"])] += 1
    selected_edges = list(tree_edges) + list(redundant_edges)
    selected_pairs = {_edge_pair(edge) for edge in selected_edges}
    external_degree = _external_generation_degree(selected_edges, bus_by_id)
    available = [edge for edge in candidates if _edge_pair(edge) not in selected_pairs]
    if not available:
        return redundant_edges
    max_route_cost = max(float(edge["route_cost"]) for edge in available)

    while available and len(redundant_edges) < target:
        viable = [
            edge
            for edge in available
            if _edge_pair(edge) not in selected_pairs
            and not _is_bad_redundant_edge(edge, selected_pairs | {_edge_pair(edge)}, bus_by_id)
            and not any(_edges_cross(edge, selected) for selected in selected_edges)
        ]
        if not viable:
            break
        edge = min(
            viable,
            key=lambda item: _redundancy_score(item, bus_by_id, tree_degree, external_degree, config, max_route_cost),
        )
        redundant_edges.append(edge)
        selected_edges.append(edge)
        selected_pairs.add(_edge_pair(edge))
        _increment_external_generation_degree(edge, bus_by_id, external_degree)
        available = [item for item in available if _edge_pair(item) not in selected_pairs]
    return redundant_edges


def _is_bad_redundant_edge(
    edge: dict[str, object],
    edge_pairs: set[tuple[int, int]],
    bus_by_id: dict[int, GridBus],
) -> bool:
    from_bus = int(edge["from_bus"])
    to_bus = int(edge["to_bus"])
    return _has_too_acute_adjacent_angle(from_bus, to_bus, edge_pairs, bus_by_id) or _passes_too_close_to_bus(
        edge,
        bus_by_id,
    )


def _has_too_acute_adjacent_angle(
    from_bus: int,
    to_bus: int,
    edge_pairs: set[tuple[int, int]],
    bus_by_id: dict[int, GridBus],
) -> bool:
    for center_bus, edge_other_bus in ((from_bus, to_bus), (to_bus, from_bus)):
        for neighbor_bus in bus_by_id:
            if neighbor_bus in {center_bus, edge_other_bus}:
                continue
            if tuple(sorted((center_bus, neighbor_bus))) not in edge_pairs:
                continue
            if _angle_degrees(bus_by_id[center_bus], bus_by_id[edge_other_bus], bus_by_id[neighbor_bus]) < 20.0:
                return True
    return False


def _angle_degrees(center: GridBus, side_a: GridBus, side_b: GridBus) -> float:
    vec_a = np.asarray([side_a.row - center.row, side_a.col - center.col], dtype=np.float32)
    vec_b = np.asarray([side_b.row - center.row, side_b.col - center.col], dtype=np.float32)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _passes_too_close_to_bus(edge: dict[str, object], bus_by_id: dict[int, GridBus]) -> bool:
    from_bus = int(edge["from_bus"])
    to_bus = int(edge["to_bus"])
    start = bus_by_id[from_bus]
    end = bus_by_id[to_bus]
    for bus_id, bus in bus_by_id.items():
        if bus_id in {from_bus, to_bus}:
            continue
        if _point_segment_distance_cells(bus, start, end) < 1.5:
            return True
    return False


def _point_segment_distance_cells(point: GridBus, start: GridBus, end: GridBus) -> float:
    seg = np.asarray([end.row - start.row, end.col - start.col], dtype=np.float32)
    rel = np.asarray([point.row - start.row, point.col - start.col], dtype=np.float32)
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= 1e-6:
        return float(np.linalg.norm(rel))
    t = float(np.clip(np.dot(rel, seg) / seg_len_sq, 0.0, 1.0))
    nearest = np.asarray([start.row, start.col], dtype=np.float32) + t * seg
    point_xy = np.asarray([point.row, point.col], dtype=np.float32)
    return float(np.linalg.norm(point_xy - nearest))


def _edges_cross(edge_a: dict[str, object], edge_b: dict[str, object]) -> bool:
    a0 = int(edge_a["from_bus"])
    a1 = int(edge_a["to_bus"])
    b0 = int(edge_b["from_bus"])
    b1 = int(edge_b["to_bus"])
    if len({a0, a1, b0, b1}) < 4:
        return False
    p1 = _edge_endpoint(edge_a, first=True)
    p2 = _edge_endpoint(edge_a, first=False)
    q1 = _edge_endpoint(edge_b, first=True)
    q2 = _edge_endpoint(edge_b, first=False)
    return _segments_intersect(p1, p2, q1, q2)


def _edge_endpoint(edge: dict[str, object], *, first: bool) -> tuple[float, float]:
    rows = edge["path_rows"]
    cols = edge["path_cols"]
    if first:
        return (float(cols[0]), float(rows[0]))  # type: ignore[index]
    return (float(cols[-1]), float(rows[-1]))  # type: ignore[index]


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)
    return o1 * o2 < 0.0 and o3 * o4 < 0.0


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _make_edges(tree_edges: list[dict[str, object]], redundant_edges: list[dict[str, object]]) -> list[GridEdge]:
    edges = []
    for edge in tree_edges + redundant_edges:
        edges.append(
            GridEdge(
                edge_id=len(edges),
                from_bus=int(edge["from_bus"]),
                to_bus=int(edge["to_bus"]),
                length_km=float(edge["length_km"]),
                route_cost=float(edge["route_cost"]),
                is_redundant=len(edges) >= len(tree_edges),
                path_rows=edge["path_rows"],  # type: ignore[arg-type]
                path_cols=edge["path_cols"],  # type: ignore[arg-type]
            )
        )
    return edges


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


def _edge_pair(edge: dict[str, object]) -> tuple[int, int]:
    return tuple(sorted((int(edge["from_bus"]), int(edge["to_bus"]))))

