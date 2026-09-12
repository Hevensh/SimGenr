from __future__ import annotations

from dataclasses import replace

import numpy as np

from world_generator.core.config import StorageConfig
from world_generator.core.contracts import entity_ids, interval_bounds_hours
from world_generator.core.errors import PhysicalInfeasibilityError, SolverError
from world_generator.core.datatypes import (
    GridElectricalState,
    PowerFlowStore,
    RefinedGridTopologyState,
    SourceLoadForecastStore,
    StorageDispatchStore,
    StoragePlanStore,
)
from world_generator.operation.power_flow import _branch_status, _branch_susceptance_mw_per_rad, _network_components, solve_dc_power_flow


def dispatch_storage_week(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    baseline_power_flow: PowerFlowStore,
    storage_plan: StoragePlanStore,
    config: StorageConfig,
    *, source_forecast: SourceLoadForecastStore | None = None,
    fixed_capacity: bool = False,
    branch_in_service: np.ndarray | None = None,
    thermal_land_limits_mw: dict[int, float] | None = None,
    initial_soc_mwh_by_site_id: dict[int, float] | None = None,
    previous_thermal_mw_by_bus_id: dict[int, float] | None = None,
    previous_storage_net_mw_by_site_id: dict[int, float] | None = None,
) -> tuple[StorageDispatchStore, SourceLoadForecastStore, PowerFlowStore, GridElectricalState]:
    """Perfect-foresight DC-OPF, with optional fixed assets and hourly outages.

    Capacity additions are bounded scenario decisions. Line expansion is thermal
    rerating at fixed impedance, not the addition of parallel circuits.
    """
    _validate_dispatch_inputs(baseline_power_flow, config)
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise SolverError("Stage 14 physical planning requires scipy.optimize.linprog", stage="stage_14_dispatch") from exc

    bus_ids = entity_ids(np.asarray([bus.bus_id for bus in topology.refined_buses]), "operating bus_ids").astype(np.int32)
    hours, bus_count = len(baseline_power_flow.timestamps), len(bus_ids)
    bus_index = {int(bus_id): index for index, bus_id in enumerate(bus_ids)}
    buses_by_id = {int(bus.bus_id): bus for bus in topology.refined_buses}
    params_by_id = {int(param.bus_id): param for param in electrical.bus_params}
    parameter_ids = entity_ids(np.asarray([p.bus_id for p in electrical.bus_params]), "operating electrical bus_ids")
    if not len(bus_ids) or set(parameter_ids) != set(bus_ids):
        raise ValueError("Operating topology and electrical bus IDs must match")
    bus_kinds = np.asarray([buses_by_id[int(bus_id)].kind for bus_id in bus_ids])
    renewable_bus_positions = np.flatnonzero(np.isin(bus_kinds, ("wind_bus", "pv_bus")))
    thermal_bus_positions = np.flatnonzero(bus_kinds == "thermal_bus")
    generator_bus_positions = np.concatenate((renewable_bus_positions, thermal_bus_positions))
    thermal_local_indices = np.arange(
        renewable_bus_positions.size,
        renewable_bus_positions.size + thermal_bus_positions.size,
        dtype=np.int32,
    )

    align_baseline = lambda values: _align_node_matrix(values, baseline_power_flow.bus_ids, bus_ids, bus_kinds)
    baseline_generation = align_baseline(baseline_power_flow.dispatched_generation_mw)
    gross_load = align_baseline(baseline_power_flow.served_load_mw + baseline_power_flow.unserved_load_mw)
    requested_generation = align_baseline(baseline_power_flow.dispatched_generation_mw + baseline_power_flow.curtailed_generation_mw)
    thermal_base_capacity = np.asarray(
        [float(params_by_id[int(bus_ids[position])].p_capacity_mw) for position in thermal_bus_positions],
        dtype=np.float64,
    )
    if source_forecast is not None:
        source_forecast.as_arrays()
        if not np.array_equal(source_forecast.timestamps, baseline_power_flow.timestamps):
            raise ValueError("Exogenous source/load timestamps must match dispatch hours")
        if len(source_forecast.bus_kinds) != len(source_forecast.bus_ids) or any(
                int(entity) not in buses_by_id or kind != buses_by_id[int(entity)].kind
                for entity, kind in zip(source_forecast.bus_ids, source_forecast.bus_kinds)):
            raise ValueError("Exogenous source bus kinds must match operating asset IDs")
        gross_load = _align_node_matrix(source_forecast.p_load_mw, source_forecast.bus_ids, bus_ids, bus_kinds)
        requested_generation = _align_node_matrix(source_forecast.p_gen_available_mw, source_forecast.bus_ids, bus_ids, bus_kinds)
        thermal_available = requested_generation[:, thermal_bus_positions]
    else:
        thermal_available = np.broadcast_to(thermal_base_capacity, (hours, len(thermal_base_capacity))).copy()
    if np.any(thermal_base_capacity < 0) or not np.isfinite(thermal_base_capacity).all() or np.any(thermal_available > thermal_base_capacity + 1e-5):
        raise ValueError("Thermal available power must lie between zero and the installed base capacity")
    thermal_availability_fraction = np.divide(thermal_available, thermal_base_capacity, out=np.zeros_like(thermal_available), where=thermal_base_capacity > 0)
    renewable_available = requested_generation[:, renewable_bus_positions]
    renewable_capacity = np.asarray([buses_by_id[int(bus_ids[p])].capacity_mw for p in renewable_bus_positions])
    if not np.isfinite(renewable_capacity).all() or np.any(renewable_capacity < 0) or np.any(renewable_available > renewable_capacity + 1e-5):
        raise ValueError("Renewable available power exceeds the installed nameplate capacity")
    load_bus_positions = np.flatnonzero(np.any(gross_load > 0.0, axis=0))
    land_limits = _id_values(thermal_land_limits_mw, bus_ids[thermal_bus_positions], thermal_base_capacity + config.max_thermal_expansion_mw_per_bus,
                             "thermal land limit", require_all=thermal_land_limits_mw is not None)
    if np.any(thermal_base_capacity > land_limits + 1e-6):
        raise ValueError("Installed thermal base capacity exceeds its thermal land limit")
    if not isinstance(fixed_capacity, (bool, np.bool_)):
        raise ValueError("fixed_capacity must be boolean")

    branches = electrical.branch_params
    branch_count = len(branches)
    branch_ids = entity_ids(np.asarray([branch.edge_id for branch in branches]), "operating branch_ids").astype(np.int32)
    line_rates = np.asarray([float(branch.rate_mva) for branch in branches], dtype=np.float64)
    if not np.isfinite(line_rates).all() or np.any(line_rates <= 0):
        raise ValueError("Line ratings must be finite and positive")
    line_lengths = np.asarray([max(float(branch.length_km), 0.1) for branch in branches], dtype=np.float64)
    susceptance = _branch_susceptance_mw_per_rad(branches)
    from_positions = np.asarray([bus_index[int(branch.from_bus)] for branch in branches], dtype=np.int32)
    to_positions = np.asarray([bus_index[int(branch.to_bus)] for branch in branches], dtype=np.int32)

    sites = storage_plan.sites
    site_count = len(sites)
    site_bus_positions = np.asarray([bus_index[int(site.bus_id)] for site in sites], dtype=np.int32)
    storage_base_power = np.asarray([site.power_mw for site in sites], dtype=np.float64)
    storage_base_energy = np.asarray([site.energy_mwh for site in sites], dtype=np.float64)
    site_ids = entity_ids(np.asarray([site.site_id for site in sites]), "operating site_ids")
    if len(np.unique(site_ids)) != site_count or any(not np.isfinite(a).all() or np.any(a < 0) for a in (storage_base_power, storage_base_energy)):
        raise ValueError("Storage sites require unique IDs and finite nonnegative power/energy")
    initial_soc = _id_values(initial_soc_mwh_by_site_id, site_ids, config.initial_soc_fraction * storage_base_energy, "initial SOC")
    previous_thermal = _id_values(previous_thermal_mw_by_bus_id, bus_ids[thermal_bus_positions], config.initial_thermal_mw, "previous thermal")
    previous_net = _id_values(previous_storage_net_mw_by_site_id, site_ids, config.initial_storage_net_mw, "previous storage net", signed=True)
    if config.cyclic_state_of_charge and any(value is not None for value in (initial_soc_mwh_by_site_id, previous_thermal_mw_by_bus_id, previous_storage_net_mw_by_site_id)):
        raise ValueError("Explicit pre-window states conflict with a cyclic boundary")
    if np.any(previous_thermal > thermal_base_capacity + 1e-6) or np.any(np.abs(previous_net) > storage_base_power + 1e-6):
        raise ValueError("Pre-window power must lie within installed base equipment ratings")
    if initial_soc_mwh_by_site_id is not None and (np.any(initial_soc < config.minimum_soc_fraction * storage_base_energy - 1e-7) or np.any(initial_soc > config.maximum_soc_fraction * storage_base_energy + 1e-7)):
        raise ValueError("Explicit initial SOC must lie within installed base energy limits")

    layout = _VariableLayout(
        hours=hours,
        branch_count=branch_count,
        generator_count=generator_bus_positions.size,
        site_count=site_count,
        thermal_count=thermal_bus_positions.size,
        load_count=load_bus_positions.size,
        bus_count=bus_count,
    )
    objective = np.zeros(layout.size, dtype=np.float64)
    lower = np.full(layout.size, -np.inf, dtype=np.float64)
    upper = np.full(layout.size, np.inf, dtype=np.float64)

    slack_position = bus_index.get(int(baseline_power_flow.slack_bus_id), 0)
    in_service = _branch_status(branch_in_service, hours, branch_count)
    pattern_models = {}
    island_id = np.zeros((hours, bus_count), dtype=np.int64)
    reserve_requirement = np.zeros((hours, bus_count))
    for pattern in np.unique(in_service, axis=0):
        active = tuple(branch for branch, on in zip(branches, pattern) if on)
        ptdf = np.zeros((branch_count, bus_count))
        ptdf[pattern] = _build_ptdf(bus_count, from_positions[pattern], to_positions[pattern], susceptance[pattern], slack_position)
        pattern_models[pattern.tobytes()] = (_network_components(bus_count, active, bus_index), ptdf)

    lower[layout.generation] = 0.0
    lower[layout.load_shed] = 0.0
    upper[layout.load_shed] = gross_load[:, load_bus_positions].ravel() if config.allow_load_shedding else 0.0
    objective[layout.load_shed] = float(config.load_shedding_cost)
    for hour in range(hours):
        for index in range(renewable_bus_positions.size):
            variable = layout.generation_index(hour, index)
            upper[variable] = max(float(renewable_available[hour, index]), 0.0)
            objective[variable] = -float(config.renewable_dispatch_credit)
        for index in thermal_local_indices:
            objective[layout.generation_index(hour, int(index))] = float(config.thermal_dispatch_cost)

    lower[layout.charge] = 0.0
    lower[layout.discharge] = 0.0
    lower[layout.emergency_discharge] = 0.0
    lower[layout.soc] = 0.0
    lower[layout.soc_below_band] = 0.0
    lower[layout.soc_above_band] = 0.0
    objective[layout.charge] = float(config.storage_cycle_cost)
    objective[layout.discharge] = float(config.storage_cycle_cost)
    objective[layout.emergency_discharge] = float(config.emergency_discharge_cost)
    objective[layout.soc_below_band] = float(config.soc_band_penalty)
    objective[layout.soc_above_band] = float(config.soc_band_penalty)
    for section in (layout.storage_reserve, layout.thermal_reserve, layout.reserve_shortfall):
        lower[section] = 0.0
    objective[layout.storage_reserve] = config.reserve_offer_cost
    objective[layout.thermal_reserve] = config.reserve_offer_cost
    objective[layout.reserve_shortfall] = config.reserve_shortfall_cost
    upper[layout.reserve_shortfall] = 0.0

    lower[layout.thermal_expansion] = 0.0
    upper[layout.thermal_expansion] = np.minimum(config.max_thermal_expansion_mw_per_bus, np.maximum(land_limits - thermal_base_capacity, 0.0))
    objective[layout.thermal_expansion] = float(config.thermal_capacity_cost)
    lower[layout.storage_power_expansion] = 0.0
    upper[layout.storage_power_expansion] = storage_base_power * float(config.max_storage_power_expansion_fraction)
    objective[layout.storage_power_expansion] = float(config.storage_power_cost)
    lower[layout.storage_energy_expansion] = 0.0
    upper[layout.storage_energy_expansion] = storage_base_energy * float(config.max_storage_energy_expansion_fraction)
    objective[layout.storage_energy_expansion] = float(config.storage_energy_cost)
    lower[layout.line_expansion] = 0.0
    upper[layout.line_expansion] = line_rates * float(config.max_line_expansion_fraction)
    objective[layout.line_expansion] = float(config.line_capacity_cost_per_mva_km) * line_lengths
    if fixed_capacity:
        for section in (layout.thermal_expansion, layout.storage_power_expansion, layout.storage_energy_expansion, layout.line_expansion):
            upper[section] = 0.0

    equalities = _SparseConstraintBuilder(layout.size)
    inequalities = _SparseConstraintBuilder(layout.size)
    line_operating_ratio = float(config.line_operating_limit_ratio)
    for hour in range(hours):
        components, ptdf = pattern_models[in_service[hour].tobytes()]
        for component in components:
            anchor = int(component[np.argmin(bus_ids[component])])
            island_id[hour, component] = int(bus_ids[anchor])
            island_load = float(gross_load[hour, component].sum())
            need = config.reserve_load_fraction * island_load + (config.reserve_contingency_mw if island_load > 0 else 0.0)
            reserve_requirement[hour, anchor] = need
            shortfall = layout.reserve_shortfall_index(hour, anchor)
            upper[shortfall] = need
            reserves = [(shortfall, -1.0)]
            reserves.extend((layout.storage_reserve_index(hour, i), -1.0) for i, p in enumerate(site_bus_positions) if p in component)
            reserves.extend((layout.thermal_reserve_index(hour, i), -1.0) for i, p in enumerate(thermal_bus_positions) if p in component)
            inequalities.add(reserves, -need)
            balance_terms = [
                (layout.generation_index(hour, generator_index), 1.0)
                for generator_index, position in enumerate(generator_bus_positions)
                if position in component
            ]
            for site_index, position in enumerate(site_bus_positions):
                if position not in component:
                    continue
                balance_terms.append((layout.discharge_index(hour, site_index), 1.0))
                balance_terms.append((layout.emergency_discharge_index(hour, site_index), 1.0))
                balance_terms.append((layout.charge_index(hour, site_index), -1.0))
            for load_index, position in enumerate(load_bus_positions):
                if position in component:
                    balance_terms.append((layout.load_shed_index(hour, load_index), 1.0))
            equalities.add(balance_terms, float(gross_load[hour, component].sum()))
        load_flow_offset = ptdf @ gross_load[hour]
        for edge_index in range(branch_count):
            if not in_service[hour, edge_index]:
                continue
            flow_terms: list[tuple[int, float]] = []
            for generator_index, bus_position in enumerate(generator_bus_positions):
                coefficient = float(ptdf[edge_index, int(bus_position)])
                if abs(coefficient) > 1e-12:
                    flow_terms.append((layout.generation_index(hour, generator_index), coefficient))
            for site_index, bus_position in enumerate(site_bus_positions):
                coefficient = float(ptdf[edge_index, int(bus_position)])
                if abs(coefficient) > 1e-12:
                    flow_terms.append((layout.discharge_index(hour, site_index), coefficient))
                    flow_terms.append((layout.emergency_discharge_index(hour, site_index), coefficient))
                    flow_terms.append((layout.charge_index(hour, site_index), -coefficient))
            for load_index, bus_position in enumerate(load_bus_positions):
                coefficient = float(ptdf[edge_index, int(bus_position)])
                if abs(coefficient) > 1e-12:
                    flow_terms.append((layout.load_shed_index(hour, load_index), coefficient))
            inequalities.add(
                flow_terms + [(layout.line_expansion.start + edge_index, -line_operating_ratio)],
                float(line_operating_ratio * line_rates[edge_index] + load_flow_offset[edge_index]),
            )
            inequalities.add(
                [(column, -value) for column, value in flow_terms]
                + [(layout.line_expansion.start + edge_index, -line_operating_ratio)],
                float(line_operating_ratio * line_rates[edge_index] - load_flow_offset[edge_index]),
            )

    ramp_fraction = float(config.thermal_ramp_fraction_per_hour)
    thermal_operating_ratio = float(config.thermal_operating_limit_ratio)
    for thermal_index, generation_index in enumerate(thermal_local_indices):
        expansion_index = layout.thermal_expansion.start + thermal_index
        base_capacity = float(thermal_base_capacity[thermal_index])
        for hour in range(hours):
            generation_variable = layout.generation_index(hour, int(generation_index))
            reserve_variable = layout.thermal_reserve_index(hour, thermal_index)
            available_ratio = thermal_operating_ratio * thermal_availability_fraction[hour, thermal_index]
            inequalities.add(
                [(generation_variable, 1.0), (reserve_variable, 1.0), (expansion_index, -available_ratio)],
                available_ratio * base_capacity,
            )
            response = ramp_fraction * config.reserve_response_hours
            inequalities.add([(reserve_variable, 1.0), (expansion_index, -response)], response * base_capacity)
            if hour == 0 and not config.cyclic_state_of_charge:
                prior = float(previous_thermal[thermal_index])
                inequalities.add([(generation_variable, 1.0), (expansion_index, -ramp_fraction)], prior + ramp_fraction * base_capacity)
                inequalities.add([(generation_variable, -1.0), (expansion_index, -ramp_fraction)], -prior + ramp_fraction * base_capacity)
                continue
            previous_hour = (hour - 1) % hours
            previous_variable = layout.generation_index(previous_hour, int(generation_index))
            inequalities.add(
                [(generation_variable, 1.0), (previous_variable, -1.0), (expansion_index, -ramp_fraction)],
                ramp_fraction * base_capacity,
            )
            inequalities.add(
                [(generation_variable, -1.0), (previous_variable, 1.0), (expansion_index, -ramp_fraction)],
                ramp_fraction * base_capacity,
            )

    charge_efficiency = float(config.charge_efficiency)
    discharge_efficiency = float(config.discharge_efficiency)
    minimum_soc_fraction = float(config.minimum_soc_fraction)
    maximum_soc_fraction = float(config.maximum_soc_fraction)
    preferred_soc_lower = float(config.preferred_soc_lower_fraction)
    preferred_soc_upper = float(config.preferred_soc_upper_fraction)
    normal_c_rate = float(config.normal_dispatch_c_rate)
    storage_ramp_fraction = float(config.storage_power_ramp_fraction_per_hour)
    for site_index in range(site_count):
        power_expansion_index = layout.storage_power_expansion.start + site_index
        energy_expansion_index = layout.storage_energy_expansion.start + site_index
        base_power = float(storage_base_power[site_index])
        base_energy = float(storage_base_energy[site_index])
        if config.cyclic_state_of_charge:
            equalities.add(
                [(layout.soc_index(0, site_index), 1.0), (layout.soc_index(hours, site_index), -1.0)], 0.0
            )
        else:
            fraction = config.initial_soc_fraction if initial_soc_mwh_by_site_id is None else 0.0
            equalities.add(
                [(layout.soc_index(0, site_index), 1.0), (energy_expansion_index, -fraction)],
                float(initial_soc[site_index]),
            )
        for step in range(hours + 1):
            soc_variable = layout.soc_index(step, site_index)
            inequalities.add(
                [(soc_variable, 1.0), (energy_expansion_index, -maximum_soc_fraction)],
                maximum_soc_fraction * base_energy,
            )
            inequalities.add(
                [(soc_variable, -1.0), (energy_expansion_index, minimum_soc_fraction)],
                -minimum_soc_fraction * base_energy,
            )
            inequalities.add(
                [
                    (soc_variable, -1.0),
                    (layout.soc_below_band_index(step, site_index), -1.0),
                    (energy_expansion_index, preferred_soc_lower),
                ],
                -preferred_soc_lower * base_energy,
            )
            inequalities.add(
                [
                    (soc_variable, 1.0),
                    (layout.soc_above_band_index(step, site_index), -1.0),
                    (energy_expansion_index, -preferred_soc_upper),
                ],
                preferred_soc_upper * base_energy,
            )
        for hour in range(hours):
            charge_variable = layout.charge_index(hour, site_index)
            discharge_variable = layout.discharge_index(hour, site_index)
            emergency_variable = layout.emergency_discharge_index(hour, site_index)
            reserve_variable = layout.storage_reserve_index(hour, site_index)
            equalities.add(
                [
                    (layout.soc_index(hour + 1, site_index), 1.0),
                    (layout.soc_index(hour, site_index), -1.0),
                    (charge_variable, -charge_efficiency),
                    (discharge_variable, 1.0 / discharge_efficiency),
                    (emergency_variable, 1.0 / discharge_efficiency),
                ],
                0.0,
            )
            inequalities.add([(charge_variable, 1.0), (power_expansion_index, -1.0)], base_power)
            inequalities.add(
                [(discharge_variable, 1.0), (emergency_variable, 1.0), (reserve_variable, 1.0), (power_expansion_index, -1.0)],
                base_power,
            )
            # Shared inverter capacity (convex hull); binary fallback below
            # enforces the mutually exclusive modes if this LP relaxes them.
            inequalities.add(
                [(charge_variable, 1.0), (discharge_variable, 1.0), (emergency_variable, 1.0), (reserve_variable, 1.0),
                 (power_expansion_index, -1.0)], base_power,
            )
            inequalities.add(
                [(charge_variable, 1.0), (energy_expansion_index, -normal_c_rate)],
                normal_c_rate * base_energy,
            )
            inequalities.add(
                [(discharge_variable, 1.0), (energy_expansion_index, -normal_c_rate)],
                normal_c_rate * base_energy,
            )
            # Reserve is a held option, not dispatched emergency energy. Its
            # activation must remain feasible above minimum SOC after dispatch.
            inequalities.add([(reserve_variable, config.reserve_duration_hours / discharge_efficiency),
                              (layout.soc_index(hour + 1, site_index), -1.0),
                              (energy_expansion_index, minimum_soc_fraction)], -minimum_soc_fraction * base_energy)
            response = storage_ramp_fraction * config.reserve_response_hours
            inequalities.add([(reserve_variable, 1.0), (power_expansion_index, -response)], response * base_power)
            if hour == 0 and not config.cyclic_state_of_charge:
                prior = float(previous_net[site_index])
                net_terms = [(discharge_variable, 1.0), (emergency_variable, 1.0), (charge_variable, -1.0)]
                inequalities.add(net_terms + [(power_expansion_index, -storage_ramp_fraction)], prior + storage_ramp_fraction * base_power)
                inequalities.add([(j, -v) for j, v in net_terms] + [(power_expansion_index, -storage_ramp_fraction)], -prior + storage_ramp_fraction * base_power)
                continue
            previous_hour = (hour - 1) % hours
            previous_charge = layout.charge_index(previous_hour, site_index)
            previous_discharge = layout.discharge_index(previous_hour, site_index)
            previous_emergency = layout.emergency_discharge_index(previous_hour, site_index)
            inequalities.add(
                [
                    (discharge_variable, 1.0),
                    (emergency_variable, 1.0),
                    (charge_variable, -1.0),
                    (previous_discharge, -1.0),
                    (previous_emergency, -1.0),
                    (previous_charge, 1.0),
                    (power_expansion_index, -storage_ramp_fraction),
                ],
                storage_ramp_fraction * base_power,
            )
            inequalities.add(
                [
                    (discharge_variable, -1.0),
                    (emergency_variable, -1.0),
                    (charge_variable, 1.0),
                    (previous_discharge, 1.0),
                    (previous_emergency, 1.0),
                    (previous_charge, -1.0),
                    (power_expansion_index, -storage_ramp_fraction),
                ],
                storage_ramp_fraction * base_power,
            )

    result = linprog(
        objective,
        A_ub=inequalities.matrix(coo_matrix),
        b_ub=np.asarray(inequalities.rhs, dtype=np.float64),
        A_eq=equalities.matrix(coo_matrix),
        b_eq=np.asarray(equalities.rhs, dtype=np.float64),
        bounds=np.column_stack((lower, upper)),
        method="highs",
        options={"presolve": True},
    )
    if not result.success and result.status == 4:
        # Degenerate planning LPs can hit a numerical HiGHS simplex status.
        # Retry the same constraints with its independent interior-point solver;
        # never turn an infeasible result into a success or relax physics.
        result = linprog(
            objective, A_ub=inequalities.matrix(coo_matrix), b_ub=np.asarray(inequalities.rhs),
            A_eq=equalities.matrix(coo_matrix), b_eq=np.asarray(equalities.rhs),
            bounds=np.column_stack((lower, upper)), method="highs-ipm", options={"presolve": False},
        )
    if not result.success:
        reason = "infeasible under the configured capacity bounds" if result.status == 2 else "solver failed"
        failure = PhysicalInfeasibilityError if result.status == 2 else SolverError
        raise failure(f"Stage 14 DC-OPF {reason}: {result.message}", stage="stage_14_dispatch")

    if site_count:
        lp_charge = result.x[layout.charge]
        lp_discharge = result.x[layout.discharge] + result.x[layout.emergency_discharge] + result.x[layout.storage_reserve]
        if np.any(np.minimum(lp_charge, lp_discharge) > 1e-6):
            result = _solve_exclusive_storage_modes(
                objective, lower, upper, equalities, inequalities, layout,
                storage_base_power + upper[layout.storage_power_expansion],
            )

    solution = np.asarray(result.x, dtype=np.float64)
    load_shed = np.zeros_like(gross_load)
    load_shed[:, load_bus_positions] = _clean(solution[layout.load_shed].reshape(hours, load_bus_positions.size))
    generation = _clean(solution[layout.generation].reshape(hours, generator_bus_positions.size))
    charge = _clean(solution[layout.charge].reshape(hours, site_count))
    normal_discharge = _clean(solution[layout.discharge].reshape(hours, site_count))
    emergency_discharge = _clean(solution[layout.emergency_discharge].reshape(hours, site_count))
    discharge = normal_discharge + emergency_discharge
    soc = _clean(solution[layout.soc].reshape(hours + 1, site_count))
    thermal_expansion = _clean(solution[layout.thermal_expansion])
    storage_power_expansion = _clean(solution[layout.storage_power_expansion])
    storage_energy_expansion = _clean(solution[layout.storage_energy_expansion])
    line_expansion = _clean(solution[layout.line_expansion])
    storage_reserve = _clean(solution[layout.storage_reserve].reshape(hours, site_count))
    thermal_reserve = _clean(solution[layout.thermal_reserve].reshape(hours, len(thermal_bus_positions)))
    reserve_shortfall = _clean(solution[layout.reserve_shortfall].reshape(hours, bus_count))
    storage_power_capacity = storage_base_power + storage_power_expansion
    storage_energy_capacity = storage_base_energy + storage_energy_expansion
    cycle_boundary_soc = soc[0].copy()

    planned_electrical = _expanded_electrical(
        electrical,
        bus_ids[thermal_bus_positions],
        thermal_expansion,
        branch_ids,
        line_expansion,
    )
    dispatched_load = gross_load - load_shed
    dispatched_available = np.zeros((hours, bus_count))
    dispatched_generation = np.zeros((hours, bus_count))
    dispatched_available[:, renewable_bus_positions] = renewable_available
    dispatched_generation[:, generator_bus_positions] = generation
    planned_thermal_capacity = thermal_base_capacity + thermal_expansion
    installed_thermal_available = thermal_availability_fraction * planned_thermal_capacity[None, :]
    dispatched_available[:, thermal_bus_positions] = installed_thermal_available
    physical_generation = dispatched_generation.copy()
    physical_available = dispatched_available.copy()
    for site_index, bus_position in enumerate(site_bus_positions):
        dispatched_load[:, bus_position] += charge[:, site_index]
        dispatched_available[:, bus_position] += float(storage_power_capacity[site_index])
        dispatched_generation[:, bus_position] += discharge[:, site_index]

    dispatched_forecast = SourceLoadForecastStore(
        timestamps=baseline_power_flow.timestamps.copy(),
        bus_ids=bus_ids.copy(),
        bus_kinds=tuple(str(kind) for kind in bus_kinds),
        p_load_mw=dispatched_load,
        p_gen_available_mw=dispatched_available,
        p_gen_scheduled_mw=dispatched_generation,
        q_load_mvar=np.zeros_like(dispatched_load),
        source_channels=(
            "p_load_mw",
            "p_gen_available_mw",
            "p_gen_scheduled_mw",
            "q_load_mvar",
            "storage_charge_mw",
            "storage_discharge_mw",
        ),
        data_semantics="perfect_foresight_dispatch",
    )
    dispatched_power_flow = solve_dc_power_flow(dispatched_forecast, topology, planned_electrical,
                                               branch_in_service=in_service, rebalance=False)
    # Solve on the actually served demand, then retain the *requested* demand
    # and explicit bus-level shortfall in the public stores. Do not rebalance
    # by proportionally spreading the OPF's local shortfall across other buses.
    dispatched_forecast = replace(dispatched_forecast, p_load_mw=dispatched_load + load_shed)
    dispatched_power_flow = replace(
        dispatched_power_flow,
        unserved_load_mw=load_shed.copy(),
    )
    # The OPF has already reduced renewable dispatch before the power-flow
    # check. Keep that intentional spill in the accounting, in addition to any
    # subsequent numerical/island rebalancing. Passing renewable availability
    # as scheduled generation to the solver would undo the optimized dispatch.
    optimized_spill = np.zeros_like(dispatched_power_flow.curtailed_generation_mw)
    optimized_spill[:, renewable_bus_positions] = np.maximum(
        renewable_available - generation[:, :renewable_bus_positions.size], 0.0
    )
    dispatched_power_flow = replace(
        dispatched_power_flow,
        curtailed_generation_mw=dispatched_power_flow.curtailed_generation_mw + optimized_spill,
    )
    baseline_thermal = baseline_generation[:, thermal_bus_positions].sum(axis=1)
    scheduled_thermal = physical_generation[:, thermal_bus_positions].sum(axis=1)
    dispatched_unserved = load_shed.sum(axis=1)
    renewable_spill = np.zeros_like(gross_load)
    renewable_spill[:, renewable_bus_positions] = renewable_available - physical_generation[:, renewable_bus_positions]
    thermal_unused = np.zeros_like(gross_load)
    thermal_unused[:, thermal_bus_positions] = installed_thermal_available - physical_generation[:, thermal_bus_positions]
    prior_schedule = (baseline_generation if source_forecast is None else
                      _align_node_matrix(source_forecast.p_gen_scheduled_mw, source_forecast.bus_ids, bus_ids, bus_kinds))
    thermal_backdown = np.zeros_like(gross_load)
    thermal_backdown[:, thermal_bus_positions] = np.maximum(prior_schedule[:, thermal_bus_positions] - physical_generation[:, thermal_bus_positions], 0)
    operation_metadata = {
        "schema_version": "operation_v1", "store_kind": "storage_dispatch", "duration_hours": 1.0,
        "fixed_capacity": bool(fixed_capacity),
        "source_basis": "exogenous_E_forecast" if source_forecast is not None else "legacy_baseline_reconstruction_with_nameplate_thermal_assumption",
        "thermal_land_guarantee": thermal_land_limits_mw is not None,
        "network_model": "lossless_DC_fixed_impedance; line_expansion_is_thermal_rerating_only",
        "reserve_scope": "active_island_aggregate_upward_headroom_energy_ramp; not_network_deliverability_frequency_or_full_N_minus_1",
        "reserve_anchor": "minimum_bus_id_in_each_active_island; requirement_and_shortfall_zero_on_other_nodes",
        "reserve_load_fraction": config.reserve_load_fraction, "reserve_contingency_mw": config.reserve_contingency_mw,
        "reserve_duration_hours": config.reserve_duration_hours, "reserve_response_hours": config.reserve_response_hours,
        "cyclic_boundary": bool(config.cyclic_state_of_charge),
        "initial_soc_mode": "cyclic_optimized" if config.cyclic_state_of_charge else ("explicit_mwh" if initial_soc_mwh_by_site_id is not None else "configured_fraction_of_installed_energy"),
        "thermal_ramp_fraction_per_hour": ramp_fraction, "storage_power_ramp_fraction_per_hour": storage_ramp_fraction,
        "thermal_operating_limit_ratio": thermal_operating_ratio,
        "charge_efficiency": charge_efficiency, "discharge_efficiency": discharge_efficiency,
        "legacy_power_basis": "legacy_load_includes_storage_charge_and_generation_includes_storage_discharge; op_fields_are_exogenous_assets_only",
        "thermal_backdown_basis": "positive_input_scheduled_generation_minus_actual; not_all_unused_available_power",
        "emergency_discharge_semantics": "actually_dispatched_energy_above_normal_C_rate; not_held_reserve",
    }
    operation_arrays = {
        "bus_ids": bus_ids, "bus_kinds": bus_kinds, "requested_load_mw": gross_load,
        "served_load_mw": gross_load - load_shed, "unserved_load_mw": load_shed,
        "generation_available_mw": physical_available, "generator_dispatch_mw": physical_generation,
        "renewable_curtailment_mw": renewable_spill, "thermal_backdown_mw": thermal_backdown,
        "thermal_unused_available_mw": thermal_unused, "island_id": island_id, "branch_in_service": in_service,
        "storage_reserve_mw": storage_reserve, "thermal_reserve_mw": thermal_reserve,
        "reserve_requirement_mw": reserve_requirement, "reserve_shortfall_mw": reserve_shortfall,
        "initial_soc_mwh": soc[0].copy(),
        "previous_storage_net_mw": discharge[-1] - charge[-1] if config.cyclic_state_of_charge else previous_net,
        "previous_thermal_mw": physical_generation[-1, thermal_bus_positions] if config.cyclic_state_of_charge else previous_thermal,
        "thermal_land_limit_mw": land_limits, "thermal_installed_capacity_mw": planned_thermal_capacity,
        "thermal_available_mw": installed_thermal_available, "thermal_dispatch_mw": physical_generation[:, thermal_bus_positions],
    }
    flow_appendix = {key: operation_arrays[key] for key in (
        "requested_load_mw", "generation_available_mw", "renewable_curtailment_mw", "thermal_backdown_mw",
        "thermal_unused_available_mw", "island_id", "branch_in_service")}
    dispatched_power_flow = replace(dispatched_power_flow, operation_arrays=flow_appendix,
                                    operation_metadata={**operation_metadata, "store_kind": "power_flow"})
    target_soc = np.broadcast_to(cycle_boundary_soc[None, :], (hours, site_count)).copy()
    store = StorageDispatchStore(
        timestamps=baseline_power_flow.timestamps.copy(),
        site_ids=np.asarray([site.site_id for site in sites], dtype=np.int32),
        site_bus_ids=np.asarray([site.bus_id for site in sites], dtype=np.int32),
        site_power_capacity_mw=storage_power_capacity,
        site_energy_capacity_mwh=storage_energy_capacity,
        storage_power_expansion_mw=storage_power_expansion,
        storage_energy_expansion_mwh=storage_energy_expansion,
        charge_mw=charge,
        discharge_mw=discharge,
        emergency_discharge_mw=emergency_discharge,
        soc_mwh=soc,
        target_soc_mwh=target_soc,
        cycle_boundary_soc_mwh=cycle_boundary_soc,
        minimum_soc_fraction=minimum_soc_fraction,
        maximum_soc_fraction=maximum_soc_fraction,
        preferred_soc_lower_fraction=preferred_soc_lower,
        preferred_soc_upper_fraction=preferred_soc_upper,
        total_load_mw=gross_load.sum(axis=1),
        renewable_available_mw=renewable_available.sum(axis=1),
        baseline_thermal_mw=baseline_thermal,
        scheduled_thermal_mw=scheduled_thermal,
        baseline_unserved_mw=baseline_power_flow.unserved_load_mw.sum(axis=1),
        dispatched_unserved_mw=dispatched_unserved,
        baseline_curtailed_mw=baseline_power_flow.curtailed_generation_mw.sum(axis=1),
        dispatched_curtailed_mw=dispatched_power_flow.curtailed_generation_mw.sum(axis=1),
        branch_ids=branch_ids,
        line_capacity_expansion_mva=line_expansion,
        thermal_bus_ids=bus_ids[thermal_bus_positions].copy(),
        thermal_capacity_expansion_mw=thermal_expansion,
        baseline_line_loading_ratio=baseline_power_flow.line_loading_ratio.copy(),
        dispatched_line_loading_ratio=dispatched_power_flow.line_loading_ratio.copy(),
        operation_arrays=operation_arrays, operation_metadata=operation_metadata,
    )
    return store, dispatched_forecast, dispatched_power_flow, planned_electrical


class _VariableLayout:
    def __init__(
        self,
        *,
        hours: int,
        branch_count: int,
        generator_count: int,
        site_count: int,
        thermal_count: int,
        load_count: int = 0,
        bus_count: int = 0,
    ) -> None:
        self.hours = hours
        self.branch_count = branch_count
        self.generator_count = generator_count
        self.site_count = site_count
        self.load_count = load_count
        self.bus_count = bus_count
        self.thermal_count = thermal_count
        offset = 0

        def allocate(size: int) -> slice:
            nonlocal offset
            result = slice(offset, offset + size)
            offset += size
            return result

        self.generation = allocate(hours * generator_count)
        self.load_shed = allocate(hours * load_count)
        self.charge = allocate(hours * site_count)
        self.discharge = allocate(hours * site_count)
        self.emergency_discharge = allocate(hours * site_count)
        self.soc = allocate((hours + 1) * site_count)
        self.soc_below_band = allocate((hours + 1) * site_count)
        self.soc_above_band = allocate((hours + 1) * site_count)
        self.thermal_expansion = allocate(thermal_count)
        self.storage_power_expansion = allocate(site_count)
        self.storage_energy_expansion = allocate(site_count)
        self.line_expansion = allocate(branch_count)
        self.storage_reserve = allocate(hours * site_count)
        self.thermal_reserve = allocate(hours * thermal_count)
        self.reserve_shortfall = allocate(hours * bus_count)
        self.size = offset

    def generation_index(self, hour: int, generator: int) -> int:
        return self.generation.start + hour * self.generator_count + generator

    def load_shed_index(self, hour: int, load: int) -> int:
        return self.load_shed.start + hour * self.load_count + load

    def charge_index(self, hour: int, site: int) -> int:
        return self.charge.start + hour * self.site_count + site

    def discharge_index(self, hour: int, site: int) -> int:
        return self.discharge.start + hour * self.site_count + site

    def emergency_discharge_index(self, hour: int, site: int) -> int:
        return self.emergency_discharge.start + hour * self.site_count + site

    def soc_index(self, step: int, site: int) -> int:
        return self.soc.start + step * self.site_count + site

    def soc_below_band_index(self, step: int, site: int) -> int:
        return self.soc_below_band.start + step * self.site_count + site

    def soc_above_band_index(self, step: int, site: int) -> int:
        return self.soc_above_band.start + step * self.site_count + site

    def storage_reserve_index(self, hour: int, site: int) -> int:
        return self.storage_reserve.start + hour * self.site_count + site

    def thermal_reserve_index(self, hour: int, thermal: int) -> int:
        return self.thermal_reserve.start + hour * self.thermal_count + thermal

    def reserve_shortfall_index(self, hour: int, bus: int) -> int:
        return self.reserve_shortfall.start + hour * self.bus_count + bus


class _SparseConstraintBuilder:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = variable_count
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.values: list[float] = []
        self.rhs: list[float] = []

    def add(self, terms: list[tuple[int, float]], rhs: float) -> None:
        row = len(self.rhs)
        for column, value in terms:
            if abs(value) > 1e-14:
                self.rows.append(row)
                self.cols.append(int(column))
                self.values.append(float(value))
        self.rhs.append(float(rhs))

    def matrix(self, coo_matrix: object) -> object:
        return coo_matrix(
            (self.values, (self.rows, self.cols)),
            shape=(len(self.rhs), self.variable_count),
            dtype=np.float64,
        ).tocsr()


def _id_values(mapping: dict[int, float] | None, ids: np.ndarray, default: object, name: str,
               *, signed: bool = False, require_all: bool = False) -> np.ndarray:
    values = np.broadcast_to(np.asarray(default, dtype=float), (len(ids),)).copy()
    if mapping is not None:
        if any(not isinstance(key, (int, np.integer)) for key in mapping) or set(mapping) - set(ids):
            raise ValueError(f"Unknown/noninteger IDs in {name}")
        if require_all and set(mapping) != set(ids):
            raise ValueError(f"{name} must specify every applicable ID")
        for i, entity in enumerate(ids):
            if int(entity) in mapping:
                values[i] = mapping[int(entity)]
    if not np.isfinite(values).all() or (not signed and np.any(values < 0)):
        raise ValueError(f"{name} must be finite{' and nonnegative' if not signed else ''}")
    return values


def _align_node_matrix(values: np.ndarray, source_ids: np.ndarray, target_ids: np.ndarray,
                       target_kinds: np.ndarray) -> np.ndarray:
    source_ids, values = np.asarray(source_ids), np.asarray(values, dtype=float)
    if source_ids.ndim != 1 or len(np.unique(source_ids)) != len(source_ids) or len(np.unique(target_ids)) != len(target_ids):
        raise ValueError("Power arrays require unique bus IDs")
    if values.ndim != 2 or values.shape[1] != len(source_ids) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Power arrays must be finite nonnegative [T,N]")
    if set(source_ids) - set(target_ids):
        raise ValueError("Source power has a bus absent from the operating topology")
    source_index = {int(entity): i for i, entity in enumerate(source_ids)}
    output = np.zeros((len(values), len(target_ids)))
    for index, (entity, kind) in enumerate(zip(target_ids, target_kinds)):
        if int(entity) in source_index:
            output[:, index] = values[:, source_index[int(entity)]]
        elif kind != "transit_bus":
            raise ValueError("Only new transit buses may be zero-filled when aligning source power")
    return output


def _expanded_electrical(
    electrical: GridElectricalState,
    thermal_bus_ids: np.ndarray,
    thermal_expansion: np.ndarray,
    branch_ids: np.ndarray,
    line_expansion: np.ndarray,
) -> GridElectricalState:
    thermal_by_id = {int(bus_id): float(value) for bus_id, value in zip(thermal_bus_ids, thermal_expansion)}
    line_by_id = {int(branch_id): float(value) for branch_id, value in zip(branch_ids, line_expansion)}
    buses = tuple(
        replace(
            bus,
            p_capacity_mw=float(bus.p_capacity_mw) + thermal_by_id.get(int(bus.bus_id), 0.0),
        )
        for bus in electrical.bus_params
    )
    branches = tuple(
        replace(
            branch,
            rate_mva=float(branch.rate_mva) + line_by_id.get(int(branch.edge_id), 0.0),
        )
        for branch in electrical.branch_params
    )
    return GridElectricalState(bus_params=buses, branch_params=branches)


def _build_ptdf(
    bus_count: int,
    from_positions: np.ndarray,
    to_positions: np.ndarray,
    susceptance: np.ndarray,
    slack_position: int,
) -> np.ndarray:
    incidence = np.zeros((from_positions.size, bus_count), dtype=np.float64)
    incidence[np.arange(from_positions.size), from_positions] = 1.0
    incidence[np.arange(to_positions.size), to_positions] = -1.0
    b_matrix = incidence.T @ (susceptance[:, None] * incidence)
    ptdf = np.zeros((from_positions.size, bus_count), dtype=np.float64)
    # One reference per island; the OPF separately enforces each island balance.
    from types import SimpleNamespace
    branches = tuple(SimpleNamespace(from_bus=int(i), to_bus=int(j)) for i, j in zip(from_positions, to_positions))
    components = _network_components(bus_count, branches, {i: i for i in range(bus_count)})
    for component in components:
        reference = slack_position if slack_position in component else int(component[0])
        non_slack = component[component != reference]
        if not non_slack.size:
            continue
        reduced = b_matrix[np.ix_(non_slack, non_slack)]
        try:
            ptdf[:, non_slack] = np.linalg.solve(
                reduced, (susceptance[:, None] * incidence[:, non_slack]).T
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular PTDF island; check line reactances and topology") from exc
    return ptdf


def _validate_dispatch_inputs(power_flow: PowerFlowStore, config: StorageConfig) -> None:
    interval_bounds_hours(power_flow.timestamps, 1.0)
    if not (0 < config.charge_efficiency <= 1 and 0 < config.discharge_efficiency <= 1):
        raise ValueError("Storage charge/discharge efficiency must lie in (0, 1]")
    if not (0 <= config.minimum_soc_fraction <= config.preferred_soc_lower_fraction <=
            config.preferred_soc_upper_fraction <= config.maximum_soc_fraction <= 1):
        raise ValueError("SOC limits and preferred band must be ordered in [0, 1]")
    if not config.minimum_soc_fraction <= config.initial_soc_fraction <= config.maximum_soc_fraction:
        raise ValueError("Initial SOC must be within the physical SOC limits")
    for name in ("thermal_operating_limit_ratio", "line_operating_limit_ratio"):
        if not 0 < getattr(config, name) <= 1:
            raise ValueError(f"{name} must lie in (0, 1]")
    for name in ("normal_dispatch_c_rate", "storage_power_ramp_fraction_per_hour", "thermal_ramp_fraction_per_hour"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    for name in ("max_thermal_expansion_mw_per_bus", "max_storage_power_expansion_fraction",
                 "max_storage_energy_expansion_fraction", "max_line_expansion_fraction", "thermal_capacity_cost",
                 "storage_power_cost", "storage_energy_cost", "line_capacity_cost_per_mva_km",
                 "thermal_dispatch_cost", "storage_cycle_cost", "emergency_discharge_cost", "soc_band_penalty", "load_shedding_cost"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.allow_load_shedding and config.load_shedding_cost <= 0:
        raise ValueError("Load shedding must have a strictly positive penalty")


def _solve_exclusive_storage_modes(
    objective: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    equalities: _SparseConstraintBuilder, inequalities: _SparseConstraintBuilder,
    layout: _VariableLayout, maximum_power: np.ndarray,
) -> object:
    """Recover a physical dispatch when the continuous relaxation cycles energy.

    Big-M bounds come from the *bounded* installed plus expandable inverter power.
    All investment and dispatch constraints are retained in the mixed-integer solve.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    mode_count = layout.hours * layout.site_count
    size = layout.size + mode_count
    equalities.variable_count = size
    inequalities.variable_count = size
    modes = _SparseConstraintBuilder(size)
    for hour in range(layout.hours):
        for site in range(layout.site_count):
            mode = layout.size + hour * layout.site_count + site
            bound = float(maximum_power[site])
            modes.add([(layout.charge_index(hour, site), 1.0), (mode, -bound)], 0.0)
            modes.add([(layout.discharge_index(hour, site), 1.0),
                       (layout.emergency_discharge_index(hour, site), 1.0),
                       (layout.storage_reserve_index(hour, site), 1.0), (mode, bound)], bound)
    result = milp(
        np.concatenate((objective, np.zeros(mode_count))),
        integrality=np.concatenate((np.zeros(layout.size), np.ones(mode_count))),
        bounds=Bounds(np.concatenate((lower, np.zeros(mode_count))), np.concatenate((upper, np.ones(mode_count)))),
        constraints=(
            LinearConstraint(equalities.matrix(coo_matrix), equalities.rhs, equalities.rhs),
            LinearConstraint(inequalities.matrix(coo_matrix), -np.inf, inequalities.rhs),
            LinearConstraint(modes.matrix(coo_matrix), -np.inf, modes.rhs),
        ),
        options={"mip_rel_gap": 1e-7},
    )
    if not result.success:
        failure = PhysicalInfeasibilityError if result.status == 2 else SolverError
        raise failure(f"Storage dispatch with exclusive charge/discharge modes failed: {result.message}", stage="stage_14_dispatch")
    return result


def _clean(values: np.ndarray, tolerance: float = 1e-7) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    # Retain arbitrarily small positive ENS/power; only remove solver-negative
    # roundoff on variables whose declared lower bound is zero.
    if np.any(result < -tolerance):
        raise SolverError("Solver returned a materially negative nonnegative variable", stage="stage_14_dispatch")
    result[(result < 0) & (result >= -tolerance)] = 0.0
    return result
