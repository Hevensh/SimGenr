from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import (
    BranchElectricalParam,
    BusElectricalParam,
    GridElectricalState,
    GridBus,
    GridEdge,
    GridTopologyState,
    GridUpgradePlanStore,
    HydrologyState,
    PowerFlowStore,
    RefinedGridTopologyState,
    SourceLoadForecastStore,
    StaticLandState,
    TerrainFeatures,
)
from world_generator.grid.corridor import (
    edge_geometry_conflicts,
    edge_geometry_crosses,
    path_length_km,
    risk_path,
    transit_risk_cost,
)
from world_generator.operation.grid_upgrade import build_grid_upgrade_plan
from world_generator.operation.power_flow import solve_dc_power_flow


@dataclass(frozen=True)
class GridUpdateIteration:
    iteration_index: int
    refined_topology: RefinedGridTopologyState
    electrical: GridElectricalState
    power_flow: PowerFlowStore
    upgrade_plan: GridUpgradePlanStore
    actions: tuple[dict[str, float | int | str], ...]
    summary: dict[str, float | int]


@dataclass(frozen=True)
class GridUpdateLoopResult:
    baseline_summary: dict[str, float | int]
    iterations: tuple[GridUpdateIteration, ...]

    def summary_dict(self) -> dict[str, float | int]:
        baseline = self.baseline_summary
        final = self.iterations[-1].summary if self.iterations else {}
        return {
            "iteration_count": int(len(self.iterations)),
            "baseline_line_hours_over_100pct": int(baseline.get("line_hours_over_100pct", 0)),
            "final_line_hours_over_100pct": int(final.get("line_hours_over_100pct", 0)),
            "baseline_peak_line_loading_ratio": float(baseline.get("peak_line_loading_ratio", 0.0)),
            "final_peak_line_loading_ratio": float(final.get("peak_line_loading_ratio", 0.0)),
            "total_actions": int(sum(len(item.actions) for item in self.iterations)),
        }


def run_grid_update_loop(
    forecast: SourceLoadForecastStore,
    refined_topology: RefinedGridTopologyState,
    base_topology: GridTopologyState,
    electrical: GridElectricalState,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
    *,
    actions_per_iteration: int = 3,
    min_upgrade_factor: float = 1.25,
) -> GridUpdateLoopResult:
    current_topology = refined_topology
    current_electrical = electrical
    current_power_flow = solve_dc_power_flow(forecast, current_topology, current_electrical)
    current_plan = build_grid_upgrade_plan(current_power_flow, current_electrical, power_grid=power_grid)
    baseline_summary = _iteration_summary(current_power_flow, current_plan, ())

    added_corridors: set[tuple[int, int]] = {
        _corridor_key(branch.from_bus, branch.to_bus) for branch in current_electrical.branch_params
    }
    added_corridors.update(_corridor_key(edge.from_bus, edge.to_bus) for edge in base_topology.edges)
    actions = _select_bypass_actions(
        current_plan,
        current_electrical,
        current_power_flow,
        forecast,
        current_topology,
        terrain,
        hydrology,
        land,
        grid,
        power_grid,
        added_corridors,
        actions_per_iteration=actions_per_iteration,
        min_upgrade_factor=min_upgrade_factor,
    )
    if actions:
        current_topology, current_electrical, actions = _apply_bypass_actions(
            current_topology,
            current_electrical,
            actions,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
    post_action_forecast = _forecast_for_topology(forecast, current_topology)
    post_action_power_flow = solve_dc_power_flow(post_action_forecast, current_topology, current_electrical)
    current_topology, current_electrical, merge_actions = _merge_collinear_branches(
        current_topology,
        current_electrical,
        terrain,
        hydrology,
        land,
        grid,
        power_grid,
        post_action_power_flow,
    )
    actions.extend(merge_actions)
    current_forecast = _forecast_for_topology(forecast, current_topology)
    current_power_flow = solve_dc_power_flow(current_forecast, current_topology, current_electrical)
    current_electrical, current_power_flow, downgrade_actions = _downgrade_low_utilization_lines(
        current_topology,
        current_electrical,
        current_forecast,
        current_power_flow,
        power_grid,
    )
    actions.extend(downgrade_actions)
    current_plan = build_grid_upgrade_plan(current_power_flow, current_electrical, power_grid=power_grid)
    final_iteration = GridUpdateIteration(
        iteration_index=1,
        refined_topology=current_topology,
        electrical=current_electrical,
        power_flow=current_power_flow,
        upgrade_plan=current_plan,
        actions=tuple(actions),
        summary=_iteration_summary(current_power_flow, current_plan, tuple(actions)),
    )
    return GridUpdateLoopResult(
        baseline_summary=baseline_summary,
        iterations=(final_iteration,),
    )


def _select_bypass_actions(
    plan: GridUpgradePlanStore,
    electrical: GridElectricalState,
    current_power_flow: PowerFlowStore,
    forecast: SourceLoadForecastStore,
    refined_topology: RefinedGridTopologyState,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
    existing_corridors: set[tuple[int, int]],
    *,
    actions_per_iteration: int,
    min_upgrade_factor: float,
) -> list[dict[str, float | int | str]]:
    order = np.argsort(-plan.priority_score)
    actions: list[dict[str, float | int | str]] = []
    trial_electrical = electrical
    trial_power_flow = current_power_flow
    trial_existing = set(existing_corridors)
    for index in order:
        if len(actions) >= actions_per_iteration:
            break
        branch_id = int(plan.branch_ids[index])
        factor = float(plan.upgrade_factor[index])
        branch_by_id = {branch.edge_id: branch for branch in trial_electrical.branch_params}
        source_branch = branch_by_id.get(branch_id)
        if source_branch is None or factor < min_upgrade_factor or float(plan.priority_score[index]) <= 0.0:
            continue
        adjacency = _branch_adjacency(trial_electrical)
        candidates = _update_candidates_for_branch(source_branch, adjacency, trial_existing, refined_topology)
        if not candidates:
            continue
        action = _best_trial_bypass_action(
            source_branch,
            candidates,
            trial_electrical,
            trial_power_flow,
            forecast,
            refined_topology,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
            trial_existing,
            factor,
            float(plan.priority_score[index]),
            float(plan.peak_loading_ratio[index]),
            int(plan.hours_over_100pct[index]),
        )
        if action is None:
            continue
        refined_topology, trial_electrical, applied = _apply_bypass_actions(
            refined_topology,
            trial_electrical,
            [action],
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
        if not applied:
            continue
        trial_power_flow = solve_dc_power_flow(forecast, refined_topology, trial_electrical)
        actions.extend(applied)
        for edge_from, edge_to in _action_corridors(applied[0]):
            trial_existing.add(_corridor_key(edge_from, edge_to))
    return actions


def _apply_bypass_actions(
    refined_topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    actions: list[dict[str, float | int | str]],
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
) -> tuple[RefinedGridTopologyState, GridElectricalState, list[dict[str, float | int | str]]]:
    buses: list[GridBus] = list(refined_topology.refined_buses)
    edges: list[GridEdge] = list(refined_topology.refined_edges)
    branches: list[BranchElectricalParam] = list(electrical.branch_params)
    bus_params: list[BusElectricalParam] = list(electrical.bus_params)
    next_branch_id = max((branch.edge_id for branch in branches), default=-1) + 1
    next_edge_id = max((edge.edge_id for edge in edges), default=-1) + 1
    applied_actions: list[dict[str, float | int | str]] = []
    for action in actions:
        result = _apply_single_bypass_action(
            buses,
            edges,
            bus_params,
            branches,
            action,
            next_edge_id,
            next_branch_id,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
        if result is None:
            continue
        buses, edges, bus_params, branches, next_edge_id, next_branch_id, applied = result
        applied_actions.append(applied)
    new_topology = _rebuild_refined_topology(refined_topology, buses, edges)
    new_electrical = GridElectricalState(bus_params=tuple(bus_params), branch_params=tuple(branches))
    return new_topology, new_electrical, applied_actions


def _apply_single_bypass_action(
    buses: list[GridBus],
    edges: list[GridEdge],
    bus_params: list[BusElectricalParam],
    branches: list[BranchElectricalParam],
    action: dict[str, float | int | str],
    next_edge_id: int,
    next_branch_id: int,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
) -> tuple[
    list[GridBus],
    list[GridEdge],
    list[BusElectricalParam],
    list[BranchElectricalParam],
    int,
    int,
    dict[str, float | int | str],
] | None:
    branch_by_id = {int(branch.edge_id): branch for branch in branches}
    template = branch_by_id.get(int(action["source_branch_id"]))
    if template is None:
        return None
    removed_branch_ids: set[int] = set()
    if action.get("action") == "swap_crossing_lines":
        removed_branch_ids.add(int(action["removed_branch_id"]))
        specs = [
            (int(action["edge1_from_bus"]), int(action["edge1_to_bus"]), "new_branch_ids_1"),
            (int(action["edge2_from_bus"]), int(action["edge2_to_bus"]), "new_branch_ids_2"),
        ]
    else:
        if action.get("action") == "reroute_overloaded_endpoint":
            removed_branch_ids.add(int(action["source_branch_id"]))
        specs = [(int(action["from_bus"]), int(action["to_bus"]), "new_branch_ids")]

    result = _apply_connection_specs(
        buses,
        edges,
        bus_params,
        branches,
        template,
        action,
        removed_branch_ids,
        specs,
        next_edge_id,
        next_branch_id,
        terrain,
        hydrology,
        land,
        grid,
        power_grid,
    )
    if result is None:
        return None

    new_crossed = _first_crossing_between_new_branches(result[3], result[1], result[0], result[6])
    if new_crossed is not None:
        first_branch, second_branch = new_crossed
        swapped_new = _apply_new_segment_crossing_swap(
            result[0],
            result[2],
            buses,
            edges,
            branches,
            template,
            action,
            removed_branch_ids,
            first_branch,
            second_branch,
            next_edge_id,
            next_branch_id,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
        if swapped_new is None:
            return None
        result = swapped_new
        result[5]["action"] = "segmented_swap_new_lines"
        result[5]["crossing_segment_1"] = [int(first_branch.from_bus), int(first_branch.to_bus)]
        result[5]["crossing_segment_2"] = [int(second_branch.from_bus), int(second_branch.to_bus)]
        result[5]["segmented_swap_from_action"] = str(action.get("action", "add_bypass_line"))

    crossed = _first_crossing_for_new_branches(result[3], result[1], result[0], result[6], removed_branch_ids)
    if crossed is not None:
        new_branch, crossed_branch, crossed_spec = crossed
        swapped = _apply_segmented_crossing_swap(
            buses,
            edges,
            bus_params,
            branches,
            template,
            action,
            removed_branch_ids,
            specs,
            crossed_spec,
            crossed_branch,
            next_edge_id,
            next_branch_id,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
        if swapped is None:
            return None
        result = swapped
        result[5]["action"] = "segmented_swap_crossing_lines"
        result[5]["crossing_segment_from_bus"] = int(new_branch.from_bus)
        result[5]["crossing_segment_to_bus"] = int(new_branch.to_bus)
        result[5]["removed_branch_id"] = int(crossed_branch.edge_id)
        result[5]["segmented_swap_from_action"] = str(action.get("action", "add_bypass_line"))

    return result[0], result[1], result[2], result[3], result[4][0], result[4][1], result[5]


def _apply_new_segment_crossing_swap(
    seeded_buses: list[GridBus],
    seeded_bus_params: list[BusElectricalParam],
    base_buses: list[GridBus],
    base_edges: list[GridEdge],
    base_branches: list[BranchElectricalParam],
    template: BranchElectricalParam,
    action: dict[str, float | int | str],
    removed_branch_ids: set[int],
    first_branch: BranchElectricalParam,
    second_branch: BranchElectricalParam,
    next_edge_id: int,
    next_branch_id: int,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
) -> tuple[
    list[GridBus],
    list[GridEdge],
    list[BusElectricalParam],
    list[BranchElectricalParam],
    tuple[int, int],
    dict[str, float | int | str],
    dict[int, tuple[int, int, str]],
] | None:
    a0, a1 = int(first_branch.from_bus), int(first_branch.to_bus)
    b0, b1 = int(second_branch.from_bus), int(second_branch.to_bus)
    options = [
        [(a0, b1, "new_branch_ids_1"), (b0, a1, "new_branch_ids_2")],
        [(a0, b0, "new_branch_ids_1"), (a1, b1, "new_branch_ids_2")],
    ]
    blocked = _existing_corridors_for_specs(base_branches, removed_branch_ids) | _blocked_corridors_from_action(action)
    best: tuple[float, tuple[
        list[GridBus],
        list[GridEdge],
        list[BusElectricalParam],
        list[BranchElectricalParam],
        tuple[int, int],
        dict[str, float | int | str],
        dict[int, tuple[int, int, str]],
    ]] | None = None
    for specs in options:
        if not _specs_are_usable(specs, blocked):
            continue
        result = _apply_connection_specs(
            seeded_buses,
            base_edges,
            seeded_bus_params,
            base_branches,
            template,
            action,
            removed_branch_ids,
            specs,
            next_edge_id,
            next_branch_id,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
        if result is None:
            continue
        if _first_crossing_between_new_branches(result[3], result[1], result[0], result[6]) is not None:
            continue
        if _first_crossing_for_new_branches(result[3], result[1], result[0], result[6], removed_branch_ids) is not None:
            continue
        score = float(result[5].get("length_km", 0.0))
        result[5]["swap_new_segment_edges"] = [[int(x), int(y)] for x, y, _ in specs]
        if best is None or score < best[0]:
            best = (score, result)
    return None if best is None else best[1]


def _apply_connection_specs(
    buses: list[GridBus],
    edges: list[GridEdge],
    bus_params: list[BusElectricalParam],
    branches: list[BranchElectricalParam],
    template: BranchElectricalParam,
    action: dict[str, float | int | str],
    removed_branch_ids: set[int],
    specs: list[tuple[int, int, str]],
    next_edge_id: int,
    next_branch_id: int,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
) -> tuple[
    list[GridBus],
    list[GridEdge],
    list[BusElectricalParam],
    list[BranchElectricalParam],
    tuple[int, int],
    dict[str, float | int | str],
    dict[int, tuple[int, int, str]],
] | None:
    trial_buses = list(buses)
    trial_edges = [edge for edge in edges if int(edge.edge_id) not in removed_branch_ids]
    trial_bus_params = list(bus_params)
    trial_branches = [branch for branch in branches if int(branch.edge_id) not in removed_branch_ids]
    applied = dict(action)
    if removed_branch_ids:
        applied["removed_branch_ids"] = [int(item) for item in sorted(removed_branch_ids)]
    all_new_ids: list[int] = []
    total_length = 0.0
    branch_to_spec: dict[int, tuple[int, int, str]] = {}
    for from_bus_id, to_bus_id, output_key in specs:
        result = _append_segmented_connection(
            trial_buses,
            trial_edges,
            trial_bus_params,
            trial_branches,
            template,
            action,
            next_edge_id,
            next_branch_id,
            from_bus_id,
            to_bus_id,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
        if result is None:
            return None
        next_edge_id, next_branch_id, new_ids, length_km = result
        for branch_id in new_ids:
            branch_to_spec[int(branch_id)] = (int(from_bus_id), int(to_bus_id), output_key)
        applied[output_key] = [int(item) for item in new_ids]
        all_new_ids.extend(new_ids)
        total_length += float(length_km)
    if _component_count(trial_buses, trial_branches) > _component_count(buses, branches):
        return None
    applied["new_branch_ids"] = [int(item) for item in all_new_ids]
    applied["new_branch_id"] = int(all_new_ids[0]) if all_new_ids else -1
    applied["logical_edges"] = [[int(a), int(b)] for a, b, _ in specs]
    applied["length_km"] = float(total_length)
    return trial_buses, trial_edges, trial_bus_params, trial_branches, (next_edge_id, next_branch_id), applied, branch_to_spec


def _apply_segmented_crossing_swap(
    buses: list[GridBus],
    edges: list[GridEdge],
    bus_params: list[BusElectricalParam],
    branches: list[BranchElectricalParam],
    template: BranchElectricalParam,
    action: dict[str, float | int | str],
    removed_branch_ids: set[int],
    specs: list[tuple[int, int, str]],
    crossed_spec: tuple[int, int, str],
    crossed_branch: BranchElectricalParam,
    next_edge_id: int,
    next_branch_id: int,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
) -> tuple[
    list[GridBus],
    list[GridEdge],
    list[BusElectricalParam],
    list[BranchElectricalParam],
    tuple[int, int],
    dict[str, float | int | str],
    dict[int, tuple[int, int, str]],
] | None:
    a, b, key = crossed_spec
    c = int(crossed_branch.from_bus)
    d = int(crossed_branch.to_bus)
    remaining = [spec for spec in specs if spec != crossed_spec]
    options = [
        [(a, d, key), (c, b, f"{key}_swap")],
        [(a, c, key), (d, b, f"{key}_swap")],
    ]
    blocked = _existing_corridors_for_specs(branches, removed_branch_ids | {int(crossed_branch.edge_id)}) | _blocked_corridors_from_action(action)
    best: tuple[float, tuple[
        list[GridBus],
        list[GridEdge],
        list[BusElectricalParam],
        list[BranchElectricalParam],
        tuple[int, int],
        dict[str, float | int | str],
        dict[int, tuple[int, int, str]],
    ]] | None = None
    for option in options:
        candidate_specs = remaining + option
        if not _specs_are_usable(candidate_specs, blocked):
            continue
        result = _apply_connection_specs(
            buses,
            edges,
            bus_params,
            branches,
            template,
            action,
            removed_branch_ids | {int(crossed_branch.edge_id)},
            candidate_specs,
            next_edge_id,
            next_branch_id,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
        )
        if result is None:
            continue
        if _first_crossing_for_new_branches(result[3], result[1], result[0], result[6], removed_branch_ids | {int(crossed_branch.edge_id)}) is not None:
            continue
        score = float(result[5].get("length_km", 0.0))
        result[5]["swap_crossed_branch_id"] = int(crossed_branch.edge_id)
        result[5]["swap_logical_edges"] = [[int(x), int(y)] for x, y, _ in candidate_specs]
        if best is None or score < best[0]:
            best = (score, result)
    return None if best is None else best[1]


def _append_segmented_connection(
    buses: list[GridBus],
    edges: list[GridEdge],
    bus_params: list[BusElectricalParam],
    branches: list[BranchElectricalParam],
    template: BranchElectricalParam,
    action: dict[str, float | int | str],
    next_edge_id: int,
    next_branch_id: int,
    from_bus_id: int,
    to_bus_id: int,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
) -> tuple[int, int, list[int], float] | None:
    bus_by_id = {bus.bus_id: bus for bus in buses}
    from_bus = bus_by_id.get(int(from_bus_id))
    to_bus = bus_by_id.get(int(to_bus_id))
    if from_bus is None or to_bus is None:
        return None
    risk_cost = transit_risk_cost(terrain, hydrology, land, power_grid)
    route_rows, route_cols = risk_path(from_bus.row, from_bus.col, to_bus.row, to_bus.col, risk_cost)
    route_length_km = path_length_km(route_rows, route_cols, grid.cell_size_km)
    local_cost = float(np.mean(risk_cost[route_rows, route_cols])) if route_rows.size else 0.0
    edges.append(
        GridEdge(
            edge_id=int(next_edge_id),
            from_bus=int(from_bus_id),
            to_bus=int(to_bus_id),
            length_km=float(route_length_km),
            route_cost=float(route_length_km * (1.0 + local_cost)),
            is_redundant=True,
            path_rows=tuple(int(v) for v in route_rows),
            path_cols=tuple(int(v) for v in route_cols),
        )
    )
    branch = _new_branch_from_template(
        template,
        action,
        next_branch_id,
        int(from_bus_id),
        int(to_bus_id),
        bus_by_id,
        grid,
        route_length_km,
    )
    if branch is None:
        return None
    branches.append(branch)
    return next_edge_id + 1, next_branch_id + 1, [int(next_branch_id)], float(route_length_km)


def _new_branch_from_template(
    template: BranchElectricalParam,
    action: dict[str, float | int | str],
    edge_id: int,
    from_bus_id: int,
    to_bus_id: int,
    bus_by_id: dict[int, object],
    grid: WorldGridConfig,
    length_km_override: float | None = None,
) -> BranchElectricalParam | None:
    from_bus = bus_by_id.get(from_bus_id)
    to_bus = bus_by_id.get(to_bus_id)
    if from_bus is None or to_bus is None:
        return None
    length_km = (
        float(length_km_override)
        if length_km_override is not None
        else float(np.hypot(from_bus.row - to_bus.row, from_bus.col - to_bus.col) * grid.cell_size_km)
    )
    r_per_km = float(action["source_r_ohm_per_km"])
    x_per_km = float(action["source_x_ohm_per_km"])
    b_per_km = float(action["source_b_us_per_km"])
    source_rate = float(action["source_rate_mva"])
    bypass_rate = float(action["bypass_rate_mva"])
    if not np.isfinite(source_rate) or not np.isfinite(bypass_rate) or min(source_rate, bypass_rate) <= 0.0:
        raise ValueError("Equivalent bypass circuits require positive finite source and target ratings")
    # Stage 12 uses a continuous parallel-circuit equivalent. Template per-km
    # values already contain its existing multiplier, so use the rating ratio
    # rather than applying the target multiplier a second time. Stage 14's
    # fixed-impedance thermal rerating is a separate planning approximation.
    impedance_scale = source_rate / bypass_rate
    return replace(
        template,
        edge_id=edge_id,
        from_bus=from_bus_id,
        to_bus=to_bus_id,
        nominal_kv=float(action["nominal_kv"]),
        length_km=length_km,
        r_ohm=max(r_per_km * length_km * impedance_scale, 1e-4),
        x_ohm=max(x_per_km * length_km * impedance_scale, 1e-4),
        b_us=max(b_per_km * length_km / impedance_scale, 1e-4),
        rate_mva=bypass_rate,
        is_redundant=True,
    )


def _rebuild_refined_topology(
    previous: RefinedGridTopologyState,
    buses: list[GridBus],
    edges: list[GridEdge],
) -> RefinedGridTopologyState:
    shape = previous.refined_line_route_map.shape
    route_map = np.zeros(shape, dtype=np.float32)
    edge_map = np.full(shape, -1, dtype=np.int16)
    for edge in edges:
        for row, col in zip(edge.path_rows, edge.path_cols):
            route_map[int(row), int(col)] += 1.0
            edge_map[int(row), int(col)] = int(edge.edge_id)
    if route_map.max() > 0.0:
        route_map /= route_map.max()
    transit_map = np.full(shape, -1, dtype=np.int16)
    for bus in buses:
        if bus.kind == "transit_bus":
            transit_map[int(bus.row), int(bus.col)] = int(bus.bus_id)
    return RefinedGridTopologyState(
        refined_line_route_map=route_map.astype(np.float32),
        refined_grid_edge_map=edge_map.astype(np.int16),
        transit_bus_map=transit_map.astype(np.int16),
        refined_buses=tuple(buses),
        refined_edges=tuple(edges),
    )


def _merge_collinear_branches(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
    power_flow: PowerFlowStore | None = None,
) -> tuple[RefinedGridTopologyState, GridElectricalState, list[dict[str, float | int | str]]]:
    current_topology = topology
    current_electrical = electrical
    all_actions: list[dict[str, float | int | str]] = []
    for merge_pass in range(8):
        current_topology, current_electrical, actions = _merge_collinear_branches_pass(
            current_topology,
            current_electrical,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
            power_flow,
        )
        if not actions:
            break
        for action in actions:
            action["merge_pass"] = merge_pass + 1
        all_actions.extend(actions)
    return current_topology, current_electrical, all_actions


def _merge_collinear_branches_pass(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
    power_flow: PowerFlowStore | None,
    *,
    allowed_branch_ids: frozenset[int] | None = None,
    allow_near_parallel: bool = True,
    routing_cost_override: np.ndarray | None = None,
) -> tuple[RefinedGridTopologyState, GridElectricalState, list[dict[str, float | int | str]]]:
    buses = list(topology.refined_buses)
    edges = list(topology.refined_edges)
    bus_params = list(electrical.bus_params)
    branches = list(electrical.branch_params)
    edge_by_id = {int(edge.edge_id): edge for edge in edges}
    branch_by_id = {int(branch.edge_id): branch for branch in branches}
    flow_by_branch = _flow_series_by_branch(power_flow)
    risk_cost = (
        np.asarray(routing_cost_override, dtype=np.float32)
        if routing_cost_override is not None
        else transit_risk_cost(terrain, hydrology, land, power_grid)
    )
    occupied = {(int(bus.row), int(bus.col)) for bus in buses}

    candidates: list[tuple[int, int, int, int, int, int, list[tuple[int, int]]]] = []
    internal_candidates: list[tuple[int, int, int, tuple[int, int], tuple[int, int], bool]] = []
    for first, second in combinations(branches, 2):
        pair_ids = frozenset((int(first.edge_id), int(second.edge_id)))
        if allowed_branch_ids is not None and pair_ids != allowed_branch_ids:
            continue
        first_edge = edge_by_id.get(int(first.edge_id))
        second_edge = edge_by_id.get(int(second.edge_id))
        if first_edge is None or second_edge is None:
            continue
        shared = {int(first.from_bus), int(first.to_bus)} & {int(second.from_bus), int(second.to_bus)}
        if not shared:
            internal_overlap = _longest_internal_path_overlap(first_edge, second_edge)
            if internal_overlap is not None:
                overlap, same_direction = internal_overlap
                if overlap[0] not in occupied and overlap[-1] not in occupied:
                    internal_candidates.append(
                        (
                            -len(overlap),
                            int(first.edge_id),
                            int(second.edge_id),
                            overlap[0],
                            overlap[-1],
                            same_direction,
                        )
                    )
        if len(shared) != 1:
            continue
        shared_bus = shared.pop()
        first_other = int(first.to_bus) if int(first.from_bus) == shared_bus else int(first.from_bus)
        second_other = int(second.to_bus) if int(second.from_bus) == shared_bus else int(second.from_bus)
        if first_other == second_other:
            continue
        common = _common_path_prefix(
            _edge_cells_from_bus(first_edge, shared_bus),
            _edge_cells_from_bus(second_edge, shared_bus),
        )
        available = [cell for cell in common[1:] if cell not in occupied]
        if not available:
            continue
        candidates.append(
            (
                -len(common),
                int(first.edge_id),
                int(second.edge_id),
                shared_bus,
                first_other,
                second_other,
                available,
            )
        )

    used_branches: set[int] = set()
    actions: list[dict[str, float | int | str]] = []
    next_bus_id = max((int(bus.bus_id) for bus in buses), default=-1) + 1
    next_edge_id = max(
        [int(edge.edge_id) for edge in edges] + [int(branch.edge_id) for branch in branches],
        default=-1,
    ) + 1
    bus_by_id = {int(bus.bus_id): bus for bus in buses}

    for _, first_id, second_id, shared_bus, first_other, second_other, cells in sorted(candidates):
        if first_id in used_branches or second_id in used_branches:
            continue
        first_branch = branch_by_id.get(first_id)
        second_branch = branch_by_id.get(second_id)
        if first_branch is None or second_branch is None:
            continue
        first_flow = flow_by_branch.get(first_id)
        second_flow = flow_by_branch.get(second_id)
        first_multiplier = _suitable_line_multiplier(first_flow, first_branch, power_grid)
        second_multiplier = _suitable_line_multiplier(second_flow, second_branch, power_grid)
        common_flow = _combined_flow_at_shared_bus(
            first_flow,
            first_branch,
            second_flow,
            second_branch,
            shared_bus,
        )
        common_nominal_kv = float(max(first_branch.nominal_kv, second_branch.nominal_kv))
        common_multiplier = (
            _suitable_line_multiplier_for_voltage(common_flow, common_nominal_kv, power_grid)
            if common_flow is not None
            else first_multiplier + second_multiplier
        )
        endpoint_weights = {
            shared_bus: common_multiplier,
            first_other: first_multiplier,
            second_other: second_multiplier,
        }
        junction_cell, weighted_cost = _best_weighted_junction_cell(
            cells,
            endpoint_weights,
            bus_by_id,
            risk_cost,
            (first_other, second_other),
        )
        if junction_cell is None:
            continue

        junction_row, junction_col = junction_cell
        _append_implicit_junction(
            next_bus_id,
            junction_cell,
            (first_branch, second_branch),
            buses,
            bus_params,
            bus_by_id,
            risk_cost,
            grid,
        )
        edges = [edge for edge in edges if int(edge.edge_id) not in {first_id, second_id}]
        branches = [branch for branch in branches if int(branch.edge_id) not in {first_id, second_id}]

        new_ids: list[int] = []
        segment_specs = [
            (shared_bus, next_bus_id, (first_branch, second_branch), common_multiplier),
            (next_bus_id, first_other, (first_branch,), first_multiplier),
            (next_bus_id, second_other, (second_branch,), second_multiplier),
        ]
        for from_bus, to_bus, contributors, multiplier in segment_specs:
            edge, branch = _equivalent_path_segment(
                next_edge_id,
                from_bus,
                to_bus,
                contributors,
                bus_by_id,
                risk_cost,
                grid,
                multiplier,
            )
            edges.append(edge)
            branches.append(branch)
            new_ids.append(next_edge_id)
            next_edge_id += 1

        actions.append(
            {
                "action": "merge_collinear_lines",
                "junction_bus_id": int(next_bus_id),
                "junction_row": int(junction_row),
                "junction_col": int(junction_col),
                "weighted_astar_cost": float(weighted_cost),
                "merged_branch_ids": [int(first_id), int(second_id)],
                "new_branch_ids": [int(item) for item in new_ids],
                "common_multiplier": float(common_multiplier),
                "first_branch_multiplier": float(first_multiplier),
                "second_branch_multiplier": float(second_multiplier),
                "common_rate_mva": float(_base_line_rate_mva(common_nominal_kv) * common_multiplier),
                "logical_edges": [
                    [int(shared_bus), int(next_bus_id)],
                    [int(next_bus_id), int(first_other)],
                    [int(next_bus_id), int(second_other)],
                ],
            }
        )
        occupied.add(junction_cell)
        used_branches.update((first_id, second_id))
        next_bus_id += 1

    for _, first_id, second_id, first_cell, second_cell, same_direction in sorted(internal_candidates):
        if first_id in used_branches or second_id in used_branches:
            continue
        first_branch = branch_by_id.get(first_id)
        second_branch = branch_by_id.get(second_id)
        first_edge = edge_by_id.get(first_id)
        second_edge = edge_by_id.get(second_id)
        if first_branch is None or second_branch is None or first_edge is None or second_edge is None:
            continue
        if first_cell in occupied or second_cell in occupied:
            continue

        first_flow = flow_by_branch.get(first_id)
        second_flow = flow_by_branch.get(second_id)
        first_multiplier = _suitable_line_multiplier(first_flow, first_branch, power_grid)
        second_multiplier = _suitable_line_multiplier(second_flow, second_branch, power_grid)
        common_flow = _combined_flow_on_internal_overlap(
            first_flow,
            first_branch,
            first_edge,
            second_flow,
            second_branch,
            second_edge,
            same_direction,
        )
        common_nominal_kv = float(max(first_branch.nominal_kv, second_branch.nominal_kv))
        common_multiplier = (
            _suitable_line_multiplier_for_voltage(common_flow, common_nominal_kv, power_grid)
            if common_flow is not None
            else first_multiplier + second_multiplier
        )
        first_junction_id = next_bus_id
        second_junction_id = next_bus_id + 1
        _append_implicit_junction(
            first_junction_id,
            first_cell,
            (first_branch, second_branch),
            buses,
            bus_params,
            bus_by_id,
            risk_cost,
            grid,
        )
        _append_implicit_junction(
            second_junction_id,
            second_cell,
            (first_branch, second_branch),
            buses,
            bus_params,
            bus_by_id,
            risk_cost,
            grid,
        )

        second_first_bus = int(second_edge.from_bus) if same_direction else int(second_edge.to_bus)
        second_last_bus = int(second_edge.to_bus) if same_direction else int(second_edge.from_bus)
        segment_specs = [
            (int(first_edge.from_bus), first_junction_id, (first_branch,), first_multiplier),
            (second_first_bus, first_junction_id, (second_branch,), second_multiplier),
            (first_junction_id, second_junction_id, (first_branch, second_branch), common_multiplier),
            (second_junction_id, int(first_edge.to_bus), (first_branch,), first_multiplier),
            (second_junction_id, second_last_bus, (second_branch,), second_multiplier),
        ]
        edges = [edge for edge in edges if int(edge.edge_id) not in {first_id, second_id}]
        branches = [branch for branch in branches if int(branch.edge_id) not in {first_id, second_id}]
        new_ids: list[int] = []
        for from_bus, to_bus, contributors, multiplier in segment_specs:
            edge, branch = _equivalent_path_segment(
                next_edge_id,
                from_bus,
                to_bus,
                contributors,
                bus_by_id,
                risk_cost,
                grid,
                multiplier,
            )
            edges.append(edge)
            branches.append(branch)
            new_ids.append(next_edge_id)
            next_edge_id += 1

        actions.append(
            {
                "action": "merge_internal_collinear_lines",
                "junction_bus_ids": [int(first_junction_id), int(second_junction_id)],
                "junction_cells": [[int(first_cell[0]), int(first_cell[1])], [int(second_cell[0]), int(second_cell[1])]],
                "merged_branch_ids": [int(first_id), int(second_id)],
                "new_branch_ids": [int(item) for item in new_ids],
                "common_multiplier": float(common_multiplier),
                "common_rate_mva": float(_base_line_rate_mva(common_nominal_kv) * common_multiplier),
                "logical_edges": [[int(item[0]), int(item[1])] for item in segment_specs],
            }
        )
        occupied.update((first_cell, second_cell))
        used_branches.update((first_id, second_id))
        next_bus_id += 2

    if not actions:
        if allow_near_parallel:
            return _merge_near_parallel_branches_pass(
                topology,
                electrical,
                terrain,
                hydrology,
                land,
                grid,
                power_grid,
                power_flow,
            )
        return topology, electrical, []
    return (
        _rebuild_refined_topology(topology, buses, edges),
        GridElectricalState(bus_params=tuple(bus_params), branch_params=tuple(branches)),
        actions,
    )


def _merge_near_parallel_branches_pass(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
    power_flow: PowerFlowStore | None,
) -> tuple[RefinedGridTopologyState, GridElectricalState, list[dict[str, float | int | str]]]:
    if power_grid.near_parallel_max_distance_km <= 0.0 or power_grid.near_parallel_min_length_km <= 0.0:
        return topology, electrical, []

    edge_by_id = {int(edge.edge_id): edge for edge in topology.refined_edges}
    bus_by_id = {int(bus.bus_id): bus for bus in topology.refined_buses}
    occupied = {(int(bus.row), int(bus.col)) for bus in topology.refined_buses}
    risk_cost = transit_risk_cost(terrain, hydrology, land, power_grid)
    max_distance_cells = power_grid.near_parallel_max_distance_km / max(grid.cell_size_km, 1e-6)
    candidates: list[dict[str, object]] = []

    for first_branch, second_branch in combinations(electrical.branch_params, 2):
        first_edge = edge_by_id.get(int(first_branch.edge_id))
        second_edge = edge_by_id.get(int(second_branch.edge_id))
        if first_edge is None or second_edge is None:
            continue
        shared = {int(first_branch.from_bus), int(first_branch.to_bus)} & {
            int(second_branch.from_bus),
            int(second_branch.to_bus),
        }
        if len(shared) > 1:
            continue

        first_cells = [(int(row), int(col)) for row, col in zip(first_edge.path_rows, first_edge.path_cols)]
        second_cells = [(int(row), int(col)) for row, col in zip(second_edge.path_rows, second_edge.path_cols)]
        if len(shared) == 1:
            shared_bus = next(iter(shared))
            first_cells = _edge_cells_from_bus(first_edge, shared_bus)
            second_cells = _edge_cells_from_bus(second_edge, shared_bus)
            match = _longest_near_parallel_match(
                first_cells,
                second_cells,
                max_distance_cells,
                power_grid.near_parallel_max_angle_deg,
                grid.cell_size_km,
                power_grid.near_parallel_min_length_km,
                allow_reverse=False,
                include_start=True,
            )
            if match is None:
                continue
            start_distance = max(
                _cell_path_length(first_cells[: int(match["first_start_index"]) + 1]),
                _cell_path_length(second_cells[: int(match["second_start_index"]) + 1]),
            ) * grid.cell_size_km
            if start_distance > power_grid.near_parallel_max_distance_km + grid.cell_size_km:
                continue
            corridor_cost = _near_merge_corridor_cost(
                risk_cost,
                first_cells,
                second_cells,
                power_grid.near_parallel_corridor_radius_km,
                grid.cell_size_km,
            )
            shared_node = bus_by_id[shared_bus]
            for junction in _near_merge_junction_choices(
                match["first_end_cell"],
                match["second_end_cell"],
                corridor_cost,
                occupied,
                max_distance_cells,
            ):
                first_snapped = _reroute_edge_via_shared_junction(
                    first_edge, shared_bus, junction, bus_by_id, corridor_cost, grid
                )
                second_snapped = _reroute_edge_via_shared_junction(
                    second_edge, shared_bus, junction, bus_by_id, corridor_cost, grid
                )
                common_rows, common_cols = risk_path(
                    shared_node.row, shared_node.col, junction[0], junction[1], corridor_cost
                )
                improvement = _near_merge_cost_improvement(
                    first_edge,
                    second_edge,
                    first_snapped,
                    second_snapped,
                    list(zip(common_rows.tolist(), common_cols.tolist())),
                    risk_cost,
                )
                if improvement + 1e-9 < power_grid.near_parallel_min_cost_improvement:
                    continue
                candidates.append(
                    {
                        "first_id": int(first_edge.edge_id),
                        "second_id": int(second_edge.edge_id),
                        "first_edge": first_snapped,
                        "second_edge": second_snapped,
                        "routing_cost": corridor_cost,
                        "improvement": float(improvement),
                        "matched_length_km": float(match["length_km"]),
                        "mean_distance_km": float(match["mean_distance_cells"]) * grid.cell_size_km,
                        "mode": "shared",
                    }
                )
                break
            continue

        match = _longest_near_parallel_match(
            first_cells,
            second_cells,
            max_distance_cells,
            power_grid.near_parallel_max_angle_deg,
            grid.cell_size_km,
            power_grid.near_parallel_min_length_km,
            allow_reverse=True,
        )
        if match is None:
            continue
        corridor_cost = _near_merge_corridor_cost(
            risk_cost,
            first_cells,
            second_cells,
            power_grid.near_parallel_corridor_radius_km,
            grid.cell_size_km,
        )
        start_choices = _near_merge_junction_choices(
            match["first_start_cell"],
            match["second_start_cell"],
            corridor_cost,
            occupied,
            max_distance_cells,
        )
        end_choices = _near_merge_junction_choices(
            match["first_end_cell"],
            match["second_end_cell"],
            corridor_cost,
            occupied,
            max_distance_cells,
        )
        best_internal: dict[str, object] | None = None
        for first_junction in start_choices:
            for second_junction in end_choices:
                if first_junction == second_junction:
                    continue
                common_rows, common_cols = risk_path(
                    first_junction[0], first_junction[1], second_junction[0], second_junction[1], corridor_cost
                )
                common_cells = list(zip(common_rows.tolist(), common_cols.tolist()))
                if _cell_path_length(common_cells) * grid.cell_size_km < power_grid.near_parallel_min_length_km:
                    continue
                first_snapped = _reroute_edge_via_internal_corridor(
                    first_edge,
                    first_junction,
                    second_junction,
                    True,
                    bus_by_id,
                    corridor_cost,
                    grid,
                )
                second_snapped = _reroute_edge_via_internal_corridor(
                    second_edge,
                    first_junction,
                    second_junction,
                    bool(match["same_direction"]),
                    bus_by_id,
                    corridor_cost,
                    grid,
                )
                improvement = _near_merge_cost_improvement(
                    first_edge,
                    second_edge,
                    first_snapped,
                    second_snapped,
                    common_cells,
                    risk_cost,
                )
                if improvement + 1e-9 < power_grid.near_parallel_min_cost_improvement:
                    continue
                proposal = {
                    "first_id": int(first_edge.edge_id),
                    "second_id": int(second_edge.edge_id),
                    "first_edge": first_snapped,
                    "second_edge": second_snapped,
                    "routing_cost": corridor_cost,
                    "improvement": float(improvement),
                    "matched_length_km": float(match["length_km"]),
                    "mean_distance_km": float(match["mean_distance_cells"]) * grid.cell_size_km,
                    "mode": "internal",
                }
                if best_internal is None or float(proposal["improvement"]) > float(best_internal["improvement"]):
                    best_internal = proposal
        if best_internal is not None:
            candidates.append(best_internal)

    candidates.sort(
        key=lambda item: (
            -float(item["improvement"]),
            -float(item["matched_length_km"]),
            float(item["mean_distance_km"]),
            int(item["first_id"]),
            int(item["second_id"]),
        )
    )
    for candidate in candidates:
        replacement_by_id = {
            int(candidate["first_id"]): candidate["first_edge"],
            int(candidate["second_id"]): candidate["second_edge"],
        }
        snapped_edges = tuple(
            replacement_by_id.get(int(edge.edge_id), edge) for edge in topology.refined_edges
        )
        snapped_topology = replace(topology, refined_edges=snapped_edges)
        merged_topology, merged_electrical, actions = _merge_collinear_branches_pass(
            snapped_topology,
            electrical,
            terrain,
            hydrology,
            land,
            grid,
            power_grid,
            power_flow,
            allowed_branch_ids=frozenset((int(candidate["first_id"]), int(candidate["second_id"]))),
            allow_near_parallel=False,
            routing_cost_override=np.asarray(candidate["routing_cost"], dtype=np.float32),
        )
        if not actions:
            continue
        action_name = (
            "merge_near_parallel_lines"
            if candidate["mode"] == "shared"
            else "merge_internal_near_parallel_lines"
        )
        for action in actions:
            action["action"] = action_name
            action["near_parallel_length_km"] = float(candidate["matched_length_km"])
            action["near_parallel_mean_distance_km"] = float(candidate["mean_distance_km"])
            action["corridor_cost_improvement"] = float(candidate["improvement"])
        return merged_topology, merged_electrical, actions
    return topology, electrical, []


def _longest_near_parallel_match(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
    max_distance_cells: float,
    max_angle_deg: float,
    cell_size_km: float,
    min_length_km: float,
    *,
    allow_reverse: bool,
    include_start: bool = False,
) -> dict[str, object] | None:
    if len(first) < 4 or len(second) < 4:
        return None
    cosine_limit = float(np.cos(np.deg2rad(max_angle_deg)))
    best: dict[str, object] | None = None
    orientations = ((False, second), (True, list(reversed(second)))) if allow_reverse else ((False, second),)
    first_start = 0 if include_start else 1
    for reversed_path, oriented_second in orientations:
        second_start = 0 if include_start else 1
        chains: list[list[tuple[int, int]]] = []
        current_chain: list[tuple[int, int]] = []
        for first_index in range(first_start, len(first) - 1):
            first_directions = _path_directions(first, first_index)
            compatible: list[tuple[float, int]] = []
            for second_index in range(second_start, len(oriented_second) - 1):
                distance = float(np.hypot(
                    first[first_index][0] - oriented_second[second_index][0],
                    first[first_index][1] - oriented_second[second_index][1],
                ))
                if distance > max_distance_cells + 1e-9:
                    continue
                second_directions = _path_directions(oriented_second, second_index)
                if max(
                    float(np.dot(first_direction, second_direction))
                    for first_direction in first_directions
                    for second_direction in second_directions
                ) < cosine_limit:
                    continue
                compatible.append((distance, second_index))
            if not compatible:
                if current_chain:
                    chains.append(current_chain)
                    current_chain = []
                continue
            _, second_index = min(compatible, key=lambda item: (item[0], item[1]))
            if current_chain and not (current_chain[-1][1] <= second_index <= current_chain[-1][1] + 2):
                chains.append(current_chain)
                current_chain = []
            current_chain.append((first_index, second_index))
        if current_chain:
            chains.append(current_chain)

        for chain in chains:
            if len(chain) < 2:
                continue
            first_length = _cell_path_length([first[first_index] for first_index, _ in chain])
            second_length = _cell_path_length([oriented_second[second_index] for _, second_index in chain])
            length_km = min(first_length, second_length) * cell_size_km
            distances = [
                float(np.hypot(
                    first[first_index][0] - oriented_second[second_index][0],
                    first[first_index][1] - oriented_second[second_index][1],
                ))
                for first_index, second_index in chain
            ]
            proposal = {
                "first_start_index": int(chain[0][0]),
                "second_start_index": int(chain[0][1]),
                "first_start_cell": first[chain[0][0]],
                "second_start_cell": oriented_second[chain[0][1]],
                "first_end_cell": first[chain[-1][0]],
                "second_end_cell": oriented_second[chain[-1][1]],
                "same_direction": not reversed_path,
                "length_km": float(length_km),
                "mean_distance_cells": float(np.mean(distances)),
            }
            if best is None or (
                float(proposal["length_km"]),
                -float(proposal["mean_distance_cells"]),
            ) > (
                float(best["length_km"]),
                -float(best["mean_distance_cells"]),
            ):
                best = proposal
    if best is None or float(best["length_km"]) + 1e-9 < min_length_km:
        return None
    return best


def _path_directions(path: list[tuple[int, int]], index: int) -> tuple[np.ndarray, ...]:
    current = np.asarray(path[index], dtype=np.float64)
    vectors = (
        current - np.asarray(path[max(index - 1, 0)], dtype=np.float64),
        np.asarray(path[min(index + 1, len(path) - 1)], dtype=np.float64) - current,
    )
    directions = []
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            directions.append(vector / norm)
    return tuple(directions)


def _cell_path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    values = np.asarray(path, dtype=np.float64)
    return float(np.hypot(np.diff(values[:, 0]), np.diff(values[:, 1])).sum())


def _near_merge_corridor_cost(
    risk_cost: np.ndarray,
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
    radius_km: float,
    cell_size_km: float,
) -> np.ndarray:
    radius = max(1, int(np.ceil(radius_km / max(cell_size_km, 1e-6))))
    allowed = np.zeros(risk_cost.shape, dtype=bool)
    for row, col in first + second:
        row0 = max(0, row - radius)
        row1 = min(risk_cost.shape[0], row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(risk_cost.shape[1], col + radius + 1)
        rr, cc = np.ogrid[row0:row1, col0:col1]
        allowed[row0:row1, col0:col1] |= (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    penalty = max(float(np.nanmax(risk_cost, initial=1.0)), 1.0) * 50.0
    return np.where(allowed, risk_cost, risk_cost + penalty).astype(np.float32)


def _near_merge_junction_choices(
    first: tuple[int, int],
    second: tuple[int, int],
    cost: np.ndarray,
    occupied: set[tuple[int, int]],
    max_distance_cells: float,
) -> list[tuple[int, int]]:
    first_cell = (int(first[0]), int(first[1]))
    second_cell = (int(second[0]), int(second[1]))
    midpoint = (
        int(round(0.5 * (first_cell[0] + second_cell[0]))),
        int(round(0.5 * (first_cell[1] + second_cell[1]))),
    )
    choices = {first_cell, second_cell, midpoint}
    valid = [
        cell
        for cell in choices
        if 0 <= cell[0] < cost.shape[0]
        and 0 <= cell[1] < cost.shape[1]
        and cell not in occupied
        and np.hypot(cell[0] - first_cell[0], cell[1] - first_cell[1]) <= max_distance_cells + 1e-9
        and np.hypot(cell[0] - second_cell[0], cell[1] - second_cell[1]) <= max_distance_cells + 1e-9
    ]
    return sorted(valid, key=lambda cell: (float(cost[cell]), abs(cell[0] - midpoint[0]) + abs(cell[1] - midpoint[1]), cell))


def _reroute_edge_via_shared_junction(
    edge: GridEdge,
    shared_bus: int,
    junction: tuple[int, int],
    bus_by_id: dict[int, GridBus],
    cost: np.ndarray,
    grid: WorldGridConfig,
) -> GridEdge:
    shared = bus_by_id[shared_bus]
    other_bus_id = int(edge.to_bus) if int(edge.from_bus) == shared_bus else int(edge.from_bus)
    other = bus_by_id[other_bus_id]
    common = _risk_path_cells((shared.row, shared.col), junction, cost)
    tail = _risk_path_cells(junction, (other.row, other.col), cost)
    cells = _join_cell_paths(common, tail)
    if int(edge.from_bus) != shared_bus:
        cells.reverse()
    return _replace_edge_path(edge, cells, cost, grid)


def _reroute_edge_via_internal_corridor(
    edge: GridEdge,
    first_junction: tuple[int, int],
    second_junction: tuple[int, int],
    same_direction: bool,
    bus_by_id: dict[int, GridBus],
    cost: np.ndarray,
    grid: WorldGridConfig,
) -> GridEdge:
    start = bus_by_id[int(edge.from_bus)]
    end = bus_by_id[int(edge.to_bus)]
    entry, exit_ = (first_junction, second_junction) if same_direction else (second_junction, first_junction)
    cells = _join_cell_paths(
        _risk_path_cells((start.row, start.col), entry, cost),
        _risk_path_cells(entry, exit_, cost),
        _risk_path_cells(exit_, (end.row, end.col), cost),
    )
    return _replace_edge_path(edge, cells, cost, grid)


def _risk_path_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    cost: np.ndarray,
) -> list[tuple[int, int]]:
    rows, cols = risk_path(start[0], start[1], end[0], end[1], cost)
    return [(int(row), int(col)) for row, col in zip(rows, cols)]


def _join_cell_paths(*paths: list[tuple[int, int]]) -> list[tuple[int, int]]:
    joined: list[tuple[int, int]] = []
    for path in paths:
        if not path:
            continue
        joined.extend(path[1:] if joined and joined[-1] == path[0] else path)
    return joined


def _replace_edge_path(
    edge: GridEdge,
    cells: list[tuple[int, int]],
    cost: np.ndarray,
    grid: WorldGridConfig,
) -> GridEdge:
    rows = np.asarray([cell[0] for cell in cells], dtype=np.int16)
    cols = np.asarray([cell[1] for cell in cells], dtype=np.int16)
    return replace(
        edge,
        length_km=path_length_km(rows, cols, grid.cell_size_km),
        route_cost=_path_cost_for_cells(cells, cost) * grid.cell_size_km,
        path_rows=tuple(int(value) for value in rows),
        path_cols=tuple(int(value) for value in cols),
    )


def _path_cost_for_cells(path: list[tuple[int, int]], cost: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    rows = np.asarray([cell[0] for cell in path], dtype=np.intp)
    cols = np.asarray([cell[1] for cell in path], dtype=np.intp)
    step = np.hypot(np.diff(rows.astype(np.float64)), np.diff(cols.astype(np.float64)))
    local = 0.5 * (cost[rows[:-1], cols[:-1]] + cost[rows[1:], cols[1:]])
    return float(np.sum(step * local))


def _near_merge_cost_improvement(
    first_original: GridEdge,
    second_original: GridEdge,
    first_snapped: GridEdge,
    second_snapped: GridEdge,
    common: list[tuple[int, int]],
    risk_cost: np.ndarray,
) -> float:
    original = _path_cost_for_cells(
        list(zip(first_original.path_rows, first_original.path_cols)), risk_cost
    ) + _path_cost_for_cells(list(zip(second_original.path_rows, second_original.path_cols)), risk_cost)
    merged = (
        _path_cost_for_cells(list(zip(first_snapped.path_rows, first_snapped.path_cols)), risk_cost)
        + _path_cost_for_cells(list(zip(second_snapped.path_rows, second_snapped.path_cols)), risk_cost)
        - _path_cost_for_cells(common, risk_cost)
    )
    return float((original - merged) / max(original, 1e-9))


def _append_implicit_junction(
    bus_id: int,
    cell: tuple[int, int],
    contributors: tuple[BranchElectricalParam, ...],
    buses: list[GridBus],
    bus_params: list[BusElectricalParam],
    bus_by_id: dict[int, GridBus],
    risk_cost: np.ndarray,
    grid: WorldGridConfig,
) -> None:
    row, col = cell
    bus = GridBus(
        bus_id=bus_id,
        kind="transit_bus",
        row=row,
        col=col,
        x=float(col / max(grid.width - 1, 1)),
        y=float(row / max(grid.height - 1, 1)),
        capacity_mw=0.0,
        suitability=float(1.0 / (1.0 + risk_cost[row, col])),
        externality_score=float(risk_cost[row, col]),
        source_kind="implicit_collinear_junction",
        source_id=-1,
    )
    buses.append(bus)
    bus_by_id[bus_id] = bus
    bus_params.append(
        BusElectricalParam(
            bus_id=bus_id,
            kind="transit_bus",
            nominal_kv=float(max(branch.nominal_kv for branch in contributors)),
            p_capacity_mw=0.0,
            q_capacity_mvar=0.0,
            base_load_mw=0.0,
            power_factor=1.0,
            voltage_setpoint_pu=1.0,
            control_mode="TRANSIT",
        )
    )


def _edge_cells_from_bus(edge: GridEdge, bus_id: int) -> list[tuple[int, int]]:
    cells = list(zip(edge.path_rows, edge.path_cols))
    return cells if int(edge.from_bus) == int(bus_id) else list(reversed(cells))


def _common_path_prefix(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    common: list[tuple[int, int]] = []
    for first_cell, second_cell in zip(first, second):
        if first_cell != second_cell:
            break
        common.append((int(first_cell[0]), int(first_cell[1])))
    return common


def _longest_internal_path_overlap(
    first_edge: GridEdge,
    second_edge: GridEdge,
) -> tuple[list[tuple[int, int]], bool] | None:
    first = list(zip(first_edge.path_rows, first_edge.path_cols))
    second = list(zip(second_edge.path_rows, second_edge.path_cols))
    second_positions: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(second):
        second_positions.setdefault(cell, []).append(index)

    best: tuple[list[tuple[int, int]], bool] | None = None
    for first_start in range(1, len(first) - 1):
        for second_start in second_positions.get(first[first_start], []):
            for direction in (1, -1):
                previous_second = second_start - direction
                if first_start > 0 and 0 <= previous_second < len(second):
                    if first[first_start - 1] == second[previous_second]:
                        continue
                length = 0
                while (
                    first_start + length < len(first)
                    and 0 <= second_start + direction * length < len(second)
                    and first[first_start + length] == second[second_start + direction * length]
                ):
                    length += 1
                if length < 2:
                    continue
                first_end = first_start + length - 1
                second_end = second_start + direction * (length - 1)
                if first_end >= len(first) - 1:
                    continue
                if min(second_start, second_end) <= 0 or max(second_start, second_end) >= len(second) - 1:
                    continue
                overlap = [(int(row), int(col)) for row, col in first[first_start : first_end + 1]]
                if best is None or len(overlap) > len(best[0]):
                    best = (overlap, direction == 1)
    return best


def _best_weighted_junction_cell(
    candidates: list[tuple[int, int]],
    endpoint_weights: dict[int, float],
    bus_by_id: dict[int, GridBus],
    risk_cost: np.ndarray,
    branch_endpoints: tuple[int, int],
) -> tuple[tuple[int, int] | None, float]:
    separating_candidates = [
        cell
        for cell in candidates
        if _paths_separate_at_cell(cell, branch_endpoints, bus_by_id, risk_cost)
    ]
    candidate_pool = separating_candidates or candidates[-1:]
    best_cell: tuple[int, int] | None = None
    best_key = (float("inf"), 0)
    for index, cell in enumerate(candidate_pool):
        score = 0.0
        for bus_id, weight in endpoint_weights.items():
            endpoint = bus_by_id[bus_id]
            score += weight * _astar_path_cost(cell, (int(endpoint.row), int(endpoint.col)), risk_cost)
        key = (float(score), -index)
        if key < best_key:
            best_key = key
            best_cell = cell
    return best_cell, float(best_key[0])


def _paths_separate_at_cell(
    cell: tuple[int, int],
    branch_endpoints: tuple[int, int],
    bus_by_id: dict[int, GridBus],
    risk_cost: np.ndarray,
) -> bool:
    paths = []
    for bus_id in branch_endpoints:
        endpoint = bus_by_id[bus_id]
        rows, cols = risk_path(cell[0], cell[1], endpoint.row, endpoint.col, risk_cost)
        paths.append(list(zip(rows.tolist(), cols.tolist())))
    return len(_common_path_prefix(paths[0], paths[1])) <= 1


def _astar_path_cost(start: tuple[int, int], end: tuple[int, int], cost: np.ndarray) -> float:
    rows, cols = risk_path(start[0], start[1], end[0], end[1], cost)
    if rows.size < 2:
        return 0.0
    step = np.hypot(np.diff(rows.astype(np.float32)), np.diff(cols.astype(np.float32)))
    local = 0.5 * (cost[rows[:-1], cols[:-1]] + cost[rows[1:], cols[1:]])
    return float(np.sum(step * local))


def _equivalent_path_segment(
    edge_id: int,
    from_bus_id: int,
    to_bus_id: int,
    contributors: tuple[BranchElectricalParam, ...],
    bus_by_id: dict[int, GridBus],
    risk_cost: np.ndarray,
    grid: WorldGridConfig,
    capacity_multiplier: float,
) -> tuple[GridEdge, BranchElectricalParam]:
    from_bus = bus_by_id[from_bus_id]
    to_bus = bus_by_id[to_bus_id]
    rows, cols = risk_path(from_bus.row, from_bus.col, to_bus.row, to_bus.col, risk_cost)
    length_km = path_length_km(rows, cols, grid.cell_size_km)
    path_cost = _astar_path_cost((from_bus.row, from_bus.col), (to_bus.row, to_bus.col), risk_cost)
    # Contributors may already represent a fractional or parallel circuit.
    # Recover each single-circuit template before applying the NEW multiplier;
    # otherwise repeated route refinement compounds impedance scaling.
    old_multipliers = [max(branch.rate_mva / _base_line_rate_mva(branch.nominal_kv), 1e-6) for branch in contributors]
    r_per_km = [_unit_value(branch.r_ohm, branch.length_km) * multiplier for branch, multiplier in zip(contributors, old_multipliers)]
    x_per_km = [_unit_value(branch.x_ohm, branch.length_km) * multiplier for branch, multiplier in zip(contributors, old_multipliers)]
    b_per_km = [_unit_value(branch.b_us, branch.length_km) / multiplier for branch, multiplier in zip(contributors, old_multipliers)]
    contributor_weights = np.asarray([max(branch.rate_mva, 1e-6) for branch in contributors], dtype=np.float64)
    contributor_weights /= contributor_weights.sum()
    base_r = float(np.dot(contributor_weights, np.asarray(r_per_km, dtype=np.float64)))
    base_x = float(np.dot(contributor_weights, np.asarray(x_per_km, dtype=np.float64)))
    base_b = float(np.dot(contributor_weights, np.asarray(b_per_km, dtype=np.float64)))
    equivalent_r = base_r / max(capacity_multiplier, 1e-6)
    equivalent_x = base_x / max(capacity_multiplier, 1e-6)
    equivalent_b = base_b * capacity_multiplier
    redundant = all(bool(branch.is_redundant) for branch in contributors)
    nominal_kv = float(max(branch.nominal_kv for branch in contributors))
    edge = GridEdge(
        edge_id=edge_id,
        from_bus=from_bus_id,
        to_bus=to_bus_id,
        length_km=float(length_km),
        route_cost=float(path_cost * grid.cell_size_km),
        is_redundant=redundant,
        path_rows=tuple(int(value) for value in rows),
        path_cols=tuple(int(value) for value in cols),
    )
    branch = BranchElectricalParam(
        edge_id=edge_id,
        from_bus=from_bus_id,
        to_bus=to_bus_id,
        nominal_kv=nominal_kv,
        length_km=float(length_km),
        r_ohm=float(max(equivalent_r * length_km, 1e-4)),
        x_ohm=float(max(equivalent_x * length_km, 1e-4)),
        b_us=float(max(equivalent_b * length_km, 1e-4)),
        rate_mva=float(_base_line_rate_mva(nominal_kv) * capacity_multiplier),
        is_redundant=redundant,
    )
    return edge, branch


def _unit_value(total: float, length_km: float) -> float:
    return float(total) / max(float(length_km), 1e-6)


def _flow_series_by_branch(power_flow: PowerFlowStore | None) -> dict[int, np.ndarray]:
    if power_flow is None:
        return {}
    return {
        int(branch_id): power_flow.line_flow_mw[:, index].astype(np.float64, copy=False)
        for index, branch_id in enumerate(power_flow.branch_ids)
    }


def _combined_flow_at_shared_bus(
    first_flow: np.ndarray | None,
    first_branch: BranchElectricalParam,
    second_flow: np.ndarray | None,
    second_branch: BranchElectricalParam,
    shared_bus: int,
) -> np.ndarray | None:
    if first_flow is None or second_flow is None:
        return None
    first_sign = 1.0 if int(first_branch.from_bus) == int(shared_bus) else -1.0
    second_sign = 1.0 if int(second_branch.from_bus) == int(shared_bus) else -1.0
    return first_sign * first_flow + second_sign * second_flow


def _combined_flow_on_internal_overlap(
    first_flow: np.ndarray | None,
    first_branch: BranchElectricalParam,
    first_edge: GridEdge,
    second_flow: np.ndarray | None,
    second_branch: BranchElectricalParam,
    second_edge: GridEdge,
    same_path_direction: bool,
) -> np.ndarray | None:
    if first_flow is None or second_flow is None:
        return None
    first_sign = 1.0 if int(first_branch.from_bus) == int(first_edge.from_bus) else -1.0
    second_sign = 1.0 if int(second_branch.from_bus) == int(second_edge.from_bus) else -1.0
    if not same_path_direction:
        second_sign *= -1.0
    return first_sign * first_flow + second_sign * second_flow


def _suitable_line_multiplier(
    flow: np.ndarray | None,
    branch: BranchElectricalParam,
    power_grid: PowerGridConfig | None = None,
) -> float:
    power_grid = power_grid or PowerGridConfig()
    if flow is None:
        return float(
            np.clip(
                float(branch.rate_mva) / _base_line_rate_mva(branch.nominal_kv),
                power_grid.min_line_multiplier,
                power_grid.max_upgrade_factor,
            )
        )
    return _suitable_line_multiplier_for_voltage(flow, branch.nominal_kv, power_grid)


def _suitable_line_multiplier_for_voltage(
    flow: np.ndarray | None,
    nominal_kv: float,
    power_grid: PowerGridConfig | None = None,
) -> float:
    power_grid = power_grid or PowerGridConfig()
    if flow is None or flow.size == 0:
        return float(power_grid.min_line_multiplier)
    required_rate = float(np.max(np.abs(flow))) / power_grid.merged_line_target_loading
    raw_multiplier = required_rate / _base_line_rate_mva(nominal_kv)
    step = float(power_grid.line_multiplier_step)
    stepped = np.ceil(raw_multiplier / step) * step
    return float(np.clip(stepped, power_grid.min_line_multiplier, power_grid.max_upgrade_factor))


def _base_line_rate_mva(nominal_kv: float) -> float:
    from world_generator.grid.electrical_builder import _standard_line_rating_mva
    return _standard_line_rating_mva(nominal_kv)


def _expected_line_rate(
    reference_rate_mva: float,
    nominal_kv: float,
    expected_upgrade_factor: float,
    power_grid: PowerGridConfig,
) -> tuple[float, float]:
    base_rate = _base_line_rate_mva(nominal_kv)
    upgrade_step = 2.0 * float(power_grid.line_multiplier_step)
    reference_multiplier = float(reference_rate_mva) / base_rate
    expected_multiplier = reference_multiplier * max(float(expected_upgrade_factor), 1.0)
    target_multiplier = float(
        np.clip(
            np.ceil(expected_multiplier / upgrade_step) * upgrade_step,
            power_grid.min_line_multiplier,
            power_grid.max_upgrade_factor,
        )
    )
    return float(base_rate * target_multiplier), target_multiplier


def _downgrade_low_utilization_lines(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    forecast: SourceLoadForecastStore,
    power_flow: PowerFlowStore,
    power_grid: PowerGridConfig,
) -> tuple[GridElectricalState, PowerFlowStore, list[dict[str, float | int | str]]]:
    current_electrical = electrical
    current_flow = power_flow
    actions: list[dict[str, float | int | str]] = []
    initial_peak = {
        int(branch_id): float(np.max(current_flow.line_loading_ratio[:, index]))
        for index, branch_id in enumerate(current_flow.branch_ids)
    }
    for branch_id in sorted(initial_peak, key=initial_peak.get):
        branch_by_id = {int(branch.edge_id): branch for branch in current_electrical.branch_params}
        branch = branch_by_id.get(branch_id)
        flow_index = {int(edge_id): index for index, edge_id in enumerate(current_flow.branch_ids)}.get(branch_id)
        if branch is None or flow_index is None:
            continue
        peak_before = float(np.max(current_flow.line_loading_ratio[:, flow_index]))
        if peak_before >= power_grid.low_utilization_peak_ratio:
            continue
        current_multiplier = float(branch.rate_mva / _base_line_rate_mva(branch.nominal_kv))
        target_multiplier = _suitable_line_multiplier(current_flow.line_flow_mw[:, flow_index], branch, power_grid)
        if target_multiplier >= current_multiplier - 0.5 * power_grid.line_multiplier_step:
            continue
        resized = _resize_branch_multiplier(branch, target_multiplier)
        trial_branches = tuple(
            resized if int(item.edge_id) == branch_id else item
            for item in current_electrical.branch_params
        )
        trial_electrical = GridElectricalState(
            bus_params=current_electrical.bus_params,
            branch_params=trial_branches,
        )
        trial_flow = solve_dc_power_flow(forecast, topology, trial_electrical)
        current_summary = current_flow.summary_dict()
        trial_summary = trial_flow.summary_dict()
        if (
            int(trial_summary["line_hours_over_100pct"]) > int(current_summary["line_hours_over_100pct"])
            or float(trial_summary["peak_line_loading_ratio"])
            > max(
                float(current_summary["peak_line_loading_ratio"]),
                float(power_grid.downgrade_max_network_loading),
            )
            + 1e-6
        ):
            continue
        trial_index = {int(edge_id): index for index, edge_id in enumerate(trial_flow.branch_ids)}[branch_id]
        actions.append(
            {
                "action": "downgrade_low_utilization_line",
                "branch_id": int(branch_id),
                "old_multiplier": float(current_multiplier),
                "new_multiplier": float(target_multiplier),
                "old_rate_mva": float(branch.rate_mva),
                "new_rate_mva": float(resized.rate_mva),
                "peak_loading_before": float(peak_before),
                "peak_loading_after": float(np.max(trial_flow.line_loading_ratio[:, trial_index])),
            }
        )
        current_electrical = trial_electrical
        current_flow = trial_flow
    return current_electrical, current_flow, actions


def _resize_branch_multiplier(
    branch: BranchElectricalParam,
    new_multiplier: float,
) -> BranchElectricalParam:
    old_multiplier = max(float(branch.rate_mva / _base_line_rate_mva(branch.nominal_kv)), 1e-6)
    impedance_scale = old_multiplier / max(float(new_multiplier), 1e-6)
    return replace(
        branch,
        r_ohm=float(branch.r_ohm * impedance_scale),
        x_ohm=float(branch.x_ohm * impedance_scale),
        b_us=float(branch.b_us / impedance_scale),
        rate_mva=float(_base_line_rate_mva(branch.nominal_kv) * new_multiplier),
    )


def _forecast_for_topology(
    forecast: SourceLoadForecastStore,
    topology: RefinedGridTopologyState,
) -> SourceLoadForecastStore:
    target_ids = np.asarray([int(bus.bus_id) for bus in topology.refined_buses], dtype=np.int32)
    if np.array_equal(target_ids, forecast.bus_ids):
        return forecast
    old_index = {int(bus_id): index for index, bus_id in enumerate(forecast.bus_ids)}

    def expand(values: np.ndarray) -> np.ndarray:
        expanded = np.zeros((values.shape[0], target_ids.size), dtype=values.dtype)
        for new_index, bus_id in enumerate(target_ids):
            source_index = old_index.get(int(bus_id))
            if source_index is not None:
                expanded[:, new_index] = values[:, source_index]
        return expanded

    return SourceLoadForecastStore(
        timestamps=forecast.timestamps.copy(),
        bus_ids=target_ids,
        bus_kinds=tuple(bus.kind for bus in topology.refined_buses),
        p_load_mw=expand(forecast.p_load_mw),
        p_gen_available_mw=expand(forecast.p_gen_available_mw),
        p_gen_scheduled_mw=expand(forecast.p_gen_scheduled_mw),
        q_load_mvar=expand(forecast.q_load_mvar),
        source_channels=forecast.source_channels,
    )


def _branch_adjacency(electrical: GridElectricalState) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for branch in electrical.branch_params:
        adjacency.setdefault(int(branch.from_bus), set()).add(int(branch.to_bus))
        adjacency.setdefault(int(branch.to_bus), set()).add(int(branch.from_bus))
    return adjacency


def _component_count(
    buses: list[GridBus],
    branches: list[BranchElectricalParam],
) -> int:
    adjacency = {int(bus.bus_id): set() for bus in buses}
    for branch in branches:
        a = int(branch.from_bus)
        b = int(branch.to_bus)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    remaining = set(adjacency)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            unseen = adjacency.get(node, set()) & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    return count


def _bus_role(bus: object | None) -> str:
    kind = str(getattr(bus, "kind", ""))
    if kind in {"wind_bus", "pv_bus", "thermal_bus"}:
        return "source"
    if kind == "load_bus":
        return "load"
    if kind == "transit_bus":
        return "transit"
    return kind


def _update_candidates_for_branch(
    branch: BranchElectricalParam,
    adjacency: dict[int, set[int]],
    existing_corridors: set[tuple[int, int]],
    refined_topology: RefinedGridTopologyState,
) -> list[tuple[int, int, str, str]]:
    a = int(branch.from_bus)
    b = int(branch.to_bus)
    neighbors_a = sorted(adjacency.get(a, set()) - {b})
    neighbors_b = sorted(adjacency.get(b, set()) - {a})
    bus_by_id = {int(bus.bus_id): bus for bus in refined_topology.refined_buses}
    role_a = _bus_role(bus_by_id.get(a))
    role_b = _bus_role(bus_by_id.get(b))
    raw: list[tuple[int, int, str, str]] = []

    # A reroute moves one endpoint along the adjacent high-flow chain while preserving its role.
    raw += [
        (a, node, "reroute-B-to-j", "reroute")
        for node in neighbors_b
        if _bus_role(bus_by_id.get(node)) == role_b
    ]
    raw += [
        (node, b, "reroute-A-to-i", "reroute")
        for node in neighbors_a
        if _bus_role(bus_by_id.get(node)) == role_a
    ]
    raw += [(a, node, "A-j", "bypass") for node in neighbors_b]
    raw += [(node, b, "i-B", "bypass") for node in neighbors_a]
    raw += [
        (node_a, node_b, "i-j", "bypass")
        for node_a in neighbors_a
        for node_b in neighbors_b
        if node_a != node_b
    ]
    candidates = []
    seen: set[tuple[str, tuple[int, int]]] = set()
    for from_bus, to_bus, kind, mode in raw:
        corridor = _corridor_key(from_bus, to_bus)
        key = (mode, corridor)
        if from_bus == to_bus or corridor in existing_corridors or key in seen:
            continue
        seen.add(key)
        candidates.append((from_bus, to_bus, kind, mode))
    return candidates[:64]


def _best_trial_bypass_action(
    source_branch: BranchElectricalParam,
    candidates: list[tuple[int, int, str, str]],
    electrical: GridElectricalState,
    current_power_flow: PowerFlowStore,
    forecast: SourceLoadForecastStore,
    refined_topology: RefinedGridTopologyState,
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    grid: WorldGridConfig,
    power_grid: PowerGridConfig,
    existing_corridors: set[tuple[int, int]],
    expected_upgrade_factor: float,
    priority_score: float,
    peak_loading_ratio: float,
    hours_over_100pct: int,
) -> dict[str, float | int | str] | None:
    current_objective = _flow_objective(current_power_flow)
    best_by_mode: dict[str, tuple[float, dict[str, float | int | str]]] = {}
    for from_bus, to_bus, kind, mode in candidates:
        for action in _candidate_action_variants(
            source_branch,
            from_bus,
            to_bus,
            kind,
            electrical,
            refined_topology,
            existing_corridors,
            power_grid,
            expected_upgrade_factor,
            priority_score,
            peak_loading_ratio,
            hours_over_100pct,
            mode=mode,
        ):
            action = action | {"blocked_corridors": [list(item) for item in sorted(existing_corridors)]}
            trial_topology, trial_electrical, applied = _apply_bypass_actions(
                refined_topology,
                electrical,
                [action],
                terrain,
                hydrology,
                land,
                grid,
                power_grid,
            )
            if not applied:
                continue
            trial_power_flow = solve_dc_power_flow(forecast, trial_topology, trial_electrical)
            objective = _flow_objective(trial_power_flow)
            mode_key = str(action.get("candidate_mode", "bypass"))
            previous = best_by_mode.get(mode_key)
            if objective < current_objective - 1e-6 and (previous is None or objective < previous[0] - 1e-6):
                best_by_mode[mode_key] = (objective, action | {
                    "objective_before": float(current_objective),
                    "objective_after": float(objective),
                    "objective_improvement": float(current_objective - objective),
                })
    if not best_by_mode:
        return None
    chosen_objective, chosen_action = min(best_by_mode.values(), key=lambda item: item[0])
    reroute = best_by_mode.get("reroute")
    bypass = best_by_mode.get("bypass")
    if reroute is not None and bypass is not None:
        reroute_gain = current_objective - reroute[0]
        bypass_gain = current_objective - bypass[0]
        if reroute_gain >= 0.75 * bypass_gain:
            chosen_objective, chosen_action = reroute
        chosen_action = chosen_action | {
            "best_reroute_improvement": float(reroute_gain),
            "best_bypass_improvement": float(bypass_gain),
        }
    return chosen_action | {"objective_after": float(chosen_objective)}


def _candidate_action_variants(
    source_branch: BranchElectricalParam,
    from_bus: int,
    to_bus: int,
    kind: str,
    electrical: GridElectricalState,
    refined_topology: RefinedGridTopologyState,
    existing_corridors: set[tuple[int, int]],
    power_grid: PowerGridConfig,
    expected_upgrade_factor: float,
    priority_score: float,
    peak_loading_ratio: float,
    hours_over_100pct: int,
    *,
    mode: str = "bypass",
) -> list[dict[str, float | int | str]]:
    direct = _make_bypass_action(
        source_branch,
        from_bus,
        to_bus,
        kind,
        power_grid,
        expected_upgrade_factor,
        priority_score,
        peak_loading_ratio,
        hours_over_100pct,
    )
    if mode == "reroute":
        direct = _make_reroute_action(direct, source_branch)
        if _has_too_acute_candidate_edges(
            [(from_bus, to_bus)],
            electrical,
            refined_topology,
            ignore_branch_ids={int(source_branch.edge_id)},
            min_angle_degrees=10.0,
        ):
            return []
        return [direct]
    crossed = _first_crossed_branch(from_bus, to_bus, electrical, refined_topology, exclude_branch_id=source_branch.edge_id)
    if crossed is None:
        if _has_too_acute_candidate_edges(
            [(from_bus, to_bus)],
            electrical,
            refined_topology,
            ignore_branch_ids=set(),
            min_angle_degrees=10.0,
        ):
            return []
        return [direct]
    swap_actions = _make_swap_actions(
        source_branch,
        crossed,
        from_bus,
        to_bus,
        kind,
        electrical,
        refined_topology,
        existing_corridors,
        power_grid,
        expected_upgrade_factor,
        priority_score,
        peak_loading_ratio,
        hours_over_100pct,
    )
    return [
        action
        for action in swap_actions
        if not _has_too_acute_candidate_edges(
            [
                (int(action["edge1_from_bus"]), int(action["edge1_to_bus"])),
                (int(action["edge2_from_bus"]), int(action["edge2_to_bus"])),
            ],
            electrical,
            refined_topology,
            ignore_branch_ids={int(crossed.edge_id)},
            min_angle_degrees=10.0,
        )
    ]


def _make_bypass_action(
    source_branch: BranchElectricalParam,
    from_bus: int,
    to_bus: int,
    kind: str,
    power_grid: PowerGridConfig,
    expected_upgrade_factor: float,
    priority_score: float,
    peak_loading_ratio: float,
    hours_over_100pct: int,
) -> dict[str, float | int | str]:
    length = max(float(source_branch.length_km), 1e-6)
    bypass_rate_mva, bypass_multiplier = _expected_line_rate(
        source_branch.rate_mva,
        source_branch.nominal_kv,
        expected_upgrade_factor,
        power_grid,
    )
    return {
        "action": "add_bypass_line",
        "candidate_mode": "bypass",
        "bypass_kind": kind,
        "source_branch_id": int(source_branch.edge_id),
        "from_bus": int(from_bus),
        "to_bus": int(to_bus),
        "bypass_rate_mva": bypass_rate_mva,
        "bypass_multiplier": bypass_multiplier,
        "expected_upgrade_factor": float(expected_upgrade_factor),
        "nominal_kv": float(source_branch.nominal_kv),
        "source_r_ohm_per_km": float(source_branch.r_ohm / length),
        "source_x_ohm_per_km": float(source_branch.x_ohm / length),
        "source_b_us_per_km": float(source_branch.b_us / length),
        "source_rate_mva": float(source_branch.rate_mva),
        "priority_score": float(priority_score),
        "peak_loading_ratio": float(peak_loading_ratio),
        "hours_over_100pct": int(hours_over_100pct),
    }


def _make_reroute_action(
    action: dict[str, float | int | str],
    source_branch: BranchElectricalParam,
) -> dict[str, float | int | str]:
    rerouted = dict(action)
    rerouted.update(
        {
            "action": "reroute_overloaded_endpoint",
            "candidate_mode": "reroute",
            "removed_branch_id": int(source_branch.edge_id),
            "old_from_bus": int(source_branch.from_bus),
            "old_to_bus": int(source_branch.to_bus),
        }
    )
    return rerouted


def _make_swap_actions(
    source_branch: BranchElectricalParam,
    crossed_branch: BranchElectricalParam,
    new_from_bus: int,
    new_to_bus: int,
    kind: str,
    electrical: GridElectricalState,
    refined_topology: RefinedGridTopologyState,
    existing_corridors: set[tuple[int, int]],
    power_grid: PowerGridConfig,
    expected_upgrade_factor: float,
    priority_score: float,
    peak_loading_ratio: float,
    hours_over_100pct: int,
) -> list[dict[str, float | int | str]]:
    a = int(crossed_branch.from_bus)
    c = int(crossed_branch.to_bus)
    b = int(new_from_bus)
    d = int(new_to_bus)
    options = [
        ((a, d), (b, c), "swap_ad_bc"),
        ((a, b), (c, d), "swap_ab_cd"),
    ]
    existing = set(existing_corridors) | {
        _corridor_key(branch.from_bus, branch.to_bus)
        for branch in electrical.branch_params
        if int(branch.edge_id) != int(crossed_branch.edge_id)
    }
    actions = []
    for (edge1, edge2, swap_kind) in options:
        if edge1[0] == edge1[1] or edge2[0] == edge2[1]:
            continue
        if _corridor_key(*edge1) in existing or _corridor_key(*edge2) in existing:
            continue
        length_1 = _bus_distance_km(edge1[0], edge1[1], refined_topology)
        length_2 = _bus_distance_km(edge2[0], edge2[1], refined_topology)
        if not np.isfinite(length_1 + length_2):
            continue
        actions.append(
            _make_swap_action(
                source_branch,
                crossed_branch,
                edge1,
                edge2,
                kind,
                swap_kind,
                power_grid,
                expected_upgrade_factor,
                priority_score,
                peak_loading_ratio,
                hours_over_100pct,
                length_1 + length_2,
            )
        )
    return sorted(actions, key=lambda item: float(item["swap_total_length_km"]))


def _make_swap_action(
    source_branch: BranchElectricalParam,
    crossed_branch: BranchElectricalParam,
    edge1: tuple[int, int],
    edge2: tuple[int, int],
    bypass_kind: str,
    swap_kind: str,
    power_grid: PowerGridConfig,
    expected_upgrade_factor: float,
    priority_score: float,
    peak_loading_ratio: float,
    hours_over_100pct: int,
    total_length_km: float,
) -> dict[str, float | int | str]:
    length = max(float(source_branch.length_km), 1e-6)
    nominal_kv = float(max(source_branch.nominal_kv, crossed_branch.nominal_kv))
    bypass_rate_mva, bypass_multiplier = _expected_line_rate(
        max(source_branch.rate_mva, crossed_branch.rate_mva),
        nominal_kv,
        expected_upgrade_factor,
        power_grid,
    )
    return {
        "action": "swap_crossing_lines",
        "bypass_kind": bypass_kind,
        "swap_kind": swap_kind,
        "source_branch_id": int(source_branch.edge_id),
        "crossed_branch_id": int(crossed_branch.edge_id),
        "removed_branch_id": int(crossed_branch.edge_id),
        "from_bus": int(edge1[0]),
        "to_bus": int(edge1[1]),
        "edge1_from_bus": int(edge1[0]),
        "edge1_to_bus": int(edge1[1]),
        "edge2_from_bus": int(edge2[0]),
        "edge2_to_bus": int(edge2[1]),
        "bypass_rate_mva": bypass_rate_mva,
        "bypass_multiplier": bypass_multiplier,
        "expected_upgrade_factor": float(expected_upgrade_factor),
        "nominal_kv": nominal_kv,
        "source_r_ohm_per_km": float(source_branch.r_ohm / length),
        "source_x_ohm_per_km": float(source_branch.x_ohm / length),
        "source_b_us_per_km": float(source_branch.b_us / length),
        "source_rate_mva": float(source_branch.rate_mva),
        "crossed_rate_mva": float(crossed_branch.rate_mva),
        "swap_total_length_km": float(total_length_km),
        "priority_score": float(priority_score),
        "peak_loading_ratio": float(peak_loading_ratio),
        "hours_over_100pct": int(hours_over_100pct),
    }


def _bus_distance_km(from_bus: int, to_bus: int, refined_topology: RefinedGridTopologyState) -> float:
    bus_by_id = {bus.bus_id: bus for bus in refined_topology.refined_buses}
    a = bus_by_id.get(int(from_bus))
    b = bus_by_id.get(int(to_bus))
    if a is None or b is None:
        return float("inf")
    # The absolute value is only used to rank swap alternatives; the electrical length is rebuilt later.
    return float(np.hypot(a.row - b.row, a.col - b.col))


def _flow_objective(power_flow: PowerFlowStore) -> float:
    summary = power_flow.summary_dict()
    return (
        14.0 * float(summary["line_hours_over_100pct"])
        + 2.0 * float(summary["line_hours_over_80pct"])
        + 120.0 * float(summary["peak_line_loading_ratio"])
    )


def _corridor_key(from_bus: int, to_bus: int) -> tuple[int, int]:
    return tuple(sorted((int(from_bus), int(to_bus))))


def _action_corridors(action: dict[str, float | int | str]) -> list[tuple[int, int]]:
    if "logical_edges" in action:
        return [(int(edge[0]), int(edge[1])) for edge in action["logical_edges"]]  # type: ignore[index]
    if action.get("action") == "swap_crossing_lines":
        return [
            (int(action["edge1_from_bus"]), int(action["edge1_to_bus"])),
            (int(action["edge2_from_bus"]), int(action["edge2_to_bus"])),
        ]
    return [(int(action["from_bus"]), int(action["to_bus"]))]


def _first_crossing_for_new_branches(
    branches: list[BranchElectricalParam],
    edges: list[GridEdge],
    buses: list[GridBus],
    branch_to_spec: dict[int, tuple[int, int, str]],
    removed_branch_ids: set[int],
) -> tuple[BranchElectricalParam, BranchElectricalParam, tuple[int, int, str]] | None:
    if not branch_to_spec:
        return None
    bus_by_id = {int(bus.bus_id): bus for bus in buses}
    branch_by_id = {int(branch.edge_id): branch for branch in branches}
    edge_by_id = {int(edge.edge_id): edge for edge in edges}
    new_ids = set(branch_to_spec)
    for new_id in sorted(new_ids):
        new_branch = branch_by_id.get(new_id)
        if new_branch is None:
            continue
        for branch in branches:
            branch_id = int(branch.edge_id)
            if branch_id in new_ids or branch_id in removed_branch_ids:
                continue
            if len({int(new_branch.from_bus), int(new_branch.to_bus), int(branch.from_bus), int(branch.to_bus)}) < 4:
                continue
            if edge_geometry_conflicts(new_branch, branch, edge_by_id, bus_by_id):
                return new_branch, branch, branch_to_spec[new_id]
    return None


def _first_crossing_between_new_branches(
    branches: list[BranchElectricalParam],
    edges: list[GridEdge],
    buses: list[GridBus],
    branch_to_spec: dict[int, tuple[int, int, str]],
) -> tuple[BranchElectricalParam, BranchElectricalParam] | None:
    if len(branch_to_spec) < 2:
        return None
    bus_by_id = {int(bus.bus_id): bus for bus in buses}
    branch_by_id = {int(branch.edge_id): branch for branch in branches}
    edge_by_id = {int(edge.edge_id): edge for edge in edges}
    new_ids = sorted(branch_to_spec)
    for left_index, left_id in enumerate(new_ids):
        left = branch_by_id.get(left_id)
        if left is None:
            continue
        for right_id in new_ids[left_index + 1 :]:
            if branch_to_spec[left_id] == branch_to_spec[right_id]:
                continue
            right = branch_by_id.get(right_id)
            if right is None:
                continue
            if len({int(left.from_bus), int(left.to_bus), int(right.from_bus), int(right.to_bus)}) < 4:
                continue
            if edge_geometry_conflicts(left, right, edge_by_id, bus_by_id):
                return left, right
    return None


def _existing_corridors_for_specs(
    branches: list[BranchElectricalParam],
    removed_branch_ids: set[int],
) -> set[tuple[int, int]]:
    return {
        _corridor_key(branch.from_bus, branch.to_bus)
        for branch in branches
        if int(branch.edge_id) not in removed_branch_ids
    }


def _blocked_corridors_from_action(action: dict[str, float | int | str]) -> set[tuple[int, int]]:
    raw = action.get("blocked_corridors")
    if not isinstance(raw, list):
        return set()
    blocked: set[tuple[int, int]] = set()
    for item in raw:
        if isinstance(item, list | tuple) and len(item) == 2:
            blocked.add(_corridor_key(int(item[0]), int(item[1])))
    return blocked


def _specs_are_usable(
    specs: list[tuple[int, int, str]],
    blocked_corridors: set[tuple[int, int]],
) -> bool:
    seen: set[tuple[int, int]] = set()
    for from_bus, to_bus, _key in specs:
        if int(from_bus) == int(to_bus):
            return False
        key = _corridor_key(from_bus, to_bus)
        if key in blocked_corridors or key in seen:
            return False
        seen.add(key)
    return True


def _first_crossed_branch(
    from_bus: int,
    to_bus: int,
    electrical: GridElectricalState,
    refined_topology: RefinedGridTopologyState,
    *,
    exclude_branch_id: int,
) -> BranchElectricalParam | None:
    bus_by_id = {bus.bus_id: bus for bus in refined_topology.refined_buses}
    start = bus_by_id.get(int(from_bus))
    end = bus_by_id.get(int(to_bus))
    if start is None or end is None:
        return None
    for branch in electrical.branch_params:
        if int(branch.edge_id) == int(exclude_branch_id):
            continue
        if len({int(from_bus), int(to_bus), int(branch.from_bus), int(branch.to_bus)}) < 4:
            continue
        branch_start = bus_by_id.get(int(branch.from_bus))
        branch_end = bus_by_id.get(int(branch.to_bus))
        if branch_start is None or branch_end is None:
            continue
        if _segments_cross(
            (float(start.col), float(start.row)),
            (float(end.col), float(end.row)),
            (float(branch_start.col), float(branch_start.row)),
            (float(branch_end.col), float(branch_end.row)),
        ):
            return branch
    return None


def _segments_cross(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    return _orientation(p1, p2, q1) * _orientation(p1, p2, q2) < 0.0 and _orientation(q1, q2, p1) * _orientation(q1, q2, p2) < 0.0


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _has_too_acute_candidate_edges(
    candidate_edges: list[tuple[int, int]],
    electrical: GridElectricalState,
    refined_topology: RefinedGridTopologyState,
    *,
    ignore_branch_ids: set[int],
    min_angle_degrees: float,
) -> bool:
    bus_by_id = {bus.bus_id: bus for bus in refined_topology.refined_buses}
    edge_pairs = {
        _corridor_key(branch.from_bus, branch.to_bus)
        for branch in electrical.branch_params
        if int(branch.edge_id) not in ignore_branch_ids
    }
    for from_bus, to_bus in candidate_edges:
        if _has_too_acute_edge_against_pairs(from_bus, to_bus, edge_pairs, bus_by_id, min_angle_degrees):
            return True
        edge_pairs.add(_corridor_key(from_bus, to_bus))
    return False


def _has_too_acute_edge_against_pairs(
    from_bus: int,
    to_bus: int,
    edge_pairs: set[tuple[int, int]],
    bus_by_id: dict[int, object],
    min_angle_degrees: float,
) -> bool:
    for center_bus, edge_other_bus in ((int(from_bus), int(to_bus)), (int(to_bus), int(from_bus))):
        center = bus_by_id.get(center_bus)
        side_a = bus_by_id.get(edge_other_bus)
        if center is None or side_a is None:
            continue
        for neighbor_bus in bus_by_id:
            if neighbor_bus in {center_bus, edge_other_bus}:
                continue
            if _corridor_key(center_bus, neighbor_bus) not in edge_pairs:
                continue
            side_b = bus_by_id[neighbor_bus]
            if _angle_degrees(center, side_a, side_b) < min_angle_degrees:
                return True
    return False


def _angle_degrees(center: object, side_a: object, side_b: object) -> float:
    vec_a = np.asarray([side_a.row - center.row, side_a.col - center.col], dtype=np.float32)
    vec_b = np.asarray([side_b.row - center.row, side_b.col - center.col], dtype=np.float32)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _iteration_summary(
    power_flow: PowerFlowStore,
    plan: GridUpgradePlanStore,
    actions: tuple[dict[str, float | int | str], ...],
) -> dict[str, float | int]:
    power_summary = power_flow.summary_dict()
    plan_summary = plan.summary_dict()
    return {
        "actions_applied": int(len(actions)),
        "reroute_actions": int(sum(str(action.get("candidate_mode", "")) == "reroute" for action in actions)),
        "peak_line_loading_ratio": float(power_summary["peak_line_loading_ratio"]),
        "line_hours_over_80pct": int(power_summary["line_hours_over_80pct"]),
        "line_hours_over_100pct": int(power_summary["line_hours_over_100pct"]),
        "total_unserved_load_mwh": float(power_summary["total_unserved_load_mwh"]),
        "recommended_upgrade_count": int(plan_summary["recommended_upgrade_count"]),
        "max_remaining_upgrade_factor": float(plan_summary["max_upgrade_factor"]),
        "total_added_rate_mva_this_iteration": float(
            sum(_net_added_rate_mva(action) for action in actions)
        ),
        "total_removed_rate_mva_this_iteration": float(
            sum(_removed_rate_mva(action) for action in actions)
        ),
    }


def _net_added_rate_mva(action: dict[str, float | int | str]) -> float:
    added = float(action.get("bypass_rate_mva", 0.0))
    if str(action.get("candidate_mode", "")) == "reroute":
        added -= float(action.get("source_rate_mva", 0.0))
    return max(added, 0.0)


def _removed_rate_mva(action: dict[str, float | int | str]) -> float:
    if str(action.get("action", "")) != "downgrade_low_utilization_line":
        return 0.0
    return max(float(action.get("old_rate_mva", 0.0)) - float(action.get("new_rate_mva", 0.0)), 0.0)
