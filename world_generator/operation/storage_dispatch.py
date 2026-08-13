from __future__ import annotations

from dataclasses import replace

import numpy as np

from world_generator.core.config import StorageConfig
from world_generator.core.datatypes import (
    GridElectricalState,
    PowerFlowStore,
    RefinedGridTopologyState,
    SourceLoadForecastStore,
    StorageDispatchStore,
    StoragePlanStore,
)
from world_generator.operation.power_flow import _branch_susceptance_mw_per_rad, solve_dc_power_flow


def dispatch_storage_week(
    topology: RefinedGridTopologyState,
    electrical: GridElectricalState,
    baseline_power_flow: PowerFlowStore,
    storage_plan: StoragePlanStore,
    config: StorageConfig,
) -> tuple[StorageDispatchStore, SourceLoadForecastStore, PowerFlowStore, GridElectricalState]:
    """Jointly plan capacity and dispatch a zero-unserved multi-period DC-OPF."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("Stage 14 physical planning requires scipy.optimize.linprog") from exc

    bus_ids = baseline_power_flow.bus_ids.astype(np.int32, copy=False)
    hours, bus_count = baseline_power_flow.served_load_mw.shape
    bus_index = {int(bus_id): index for index, bus_id in enumerate(bus_ids)}
    buses_by_id = {int(bus.bus_id): bus for bus in topology.refined_buses}
    params_by_id = {int(param.bus_id): param for param in electrical.bus_params}
    bus_kinds = np.asarray([buses_by_id[int(bus_id)].kind for bus_id in bus_ids])
    renewable_bus_positions = np.flatnonzero(np.isin(bus_kinds, ("wind_bus", "pv_bus")))
    thermal_bus_positions = np.flatnonzero(bus_kinds == "thermal_bus")
    generator_bus_positions = np.concatenate((renewable_bus_positions, thermal_bus_positions))
    thermal_local_indices = np.arange(
        renewable_bus_positions.size,
        renewable_bus_positions.size + thermal_bus_positions.size,
        dtype=np.int32,
    )

    gross_load = (
        baseline_power_flow.served_load_mw + baseline_power_flow.unserved_load_mw
    ).astype(np.float64)
    requested_generation = (
        baseline_power_flow.dispatched_generation_mw + baseline_power_flow.curtailed_generation_mw
    ).astype(np.float64)
    renewable_available = requested_generation[:, renewable_bus_positions]
    thermal_base_capacity = np.asarray(
        [max(float(params_by_id[int(bus_ids[position])].p_capacity_mw), 0.0) for position in thermal_bus_positions],
        dtype=np.float64,
    )

    branches = electrical.branch_params
    branch_count = len(branches)
    branch_ids = np.asarray([branch.edge_id for branch in branches], dtype=np.int32)
    line_rates = np.asarray([max(float(branch.rate_mva), 1e-6) for branch in branches], dtype=np.float64)
    line_lengths = np.asarray([max(float(branch.length_km), 0.1) for branch in branches], dtype=np.float64)
    susceptance = _branch_susceptance_mw_per_rad(branches)
    from_positions = np.asarray([bus_index[int(branch.from_bus)] for branch in branches], dtype=np.int32)
    to_positions = np.asarray([bus_index[int(branch.to_bus)] for branch in branches], dtype=np.int32)

    sites = storage_plan.sites
    site_count = len(sites)
    site_bus_positions = np.asarray([bus_index[int(site.bus_id)] for site in sites], dtype=np.int32)
    storage_base_power = np.asarray([site.power_mw for site in sites], dtype=np.float64)
    storage_base_energy = np.asarray([site.energy_mwh for site in sites], dtype=np.float64)

    layout = _VariableLayout(
        hours=hours,
        branch_count=branch_count,
        generator_count=generator_bus_positions.size,
        site_count=site_count,
        thermal_count=thermal_bus_positions.size,
    )
    objective = np.zeros(layout.size, dtype=np.float64)
    lower = np.full(layout.size, -np.inf, dtype=np.float64)
    upper = np.full(layout.size, np.inf, dtype=np.float64)

    slack_position = bus_index[int(baseline_power_flow.slack_bus_id)]
    ptdf = _build_ptdf(bus_count, from_positions, to_positions, susceptance, slack_position)

    lower[layout.generation] = 0.0
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

    lower[layout.thermal_expansion] = 0.0
    upper[layout.thermal_expansion] = float(config.max_thermal_expansion_mw_per_bus)
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

    equalities = _SparseConstraintBuilder(layout.size)
    inequalities = _SparseConstraintBuilder(layout.size)
    line_operating_ratio = float(np.clip(config.line_operating_limit_ratio, 1e-3, 1.0))
    for hour in range(hours):
        balance_terms = [
            (layout.generation_index(hour, generator_index), 1.0)
            for generator_index in range(generator_bus_positions.size)
        ]
        for site_index in range(site_count):
            balance_terms.append((layout.discharge_index(hour, site_index), 1.0))
            balance_terms.append((layout.emergency_discharge_index(hour, site_index), 1.0))
            balance_terms.append((layout.charge_index(hour, site_index), -1.0))
        equalities.add(balance_terms, float(gross_load[hour].sum()))
        load_flow_offset = ptdf @ gross_load[hour]
        for edge_index in range(branch_count):
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
            inequalities.add(
                flow_terms + [(layout.line_expansion.start + edge_index, -line_operating_ratio)],
                float(line_operating_ratio * line_rates[edge_index] + load_flow_offset[edge_index]),
            )
            inequalities.add(
                [(column, -value) for column, value in flow_terms]
                + [(layout.line_expansion.start + edge_index, -line_operating_ratio)],
                float(line_operating_ratio * line_rates[edge_index] - load_flow_offset[edge_index]),
            )

    ramp_fraction = max(float(config.thermal_ramp_fraction_per_hour), 1e-3)
    thermal_operating_ratio = float(np.clip(config.thermal_operating_limit_ratio, 1e-3, 1.0))
    for thermal_index, generation_index in enumerate(thermal_local_indices):
        expansion_index = layout.thermal_expansion.start + thermal_index
        base_capacity = float(thermal_base_capacity[thermal_index])
        for hour in range(hours):
            generation_variable = layout.generation_index(hour, int(generation_index))
            inequalities.add(
                [(generation_variable, 1.0), (expansion_index, -thermal_operating_ratio)],
                thermal_operating_ratio * base_capacity,
            )
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

    charge_efficiency = float(np.clip(config.charge_efficiency, 1e-3, 1.0))
    discharge_efficiency = float(np.clip(config.discharge_efficiency, 1e-3, 1.0))
    minimum_soc_fraction = float(np.clip(config.minimum_soc_fraction, 0.0, 1.0))
    maximum_soc_fraction = float(np.clip(config.maximum_soc_fraction, minimum_soc_fraction, 1.0))
    preferred_soc_lower = float(
        np.clip(config.preferred_soc_lower_fraction, minimum_soc_fraction, maximum_soc_fraction)
    )
    preferred_soc_upper = float(
        np.clip(config.preferred_soc_upper_fraction, preferred_soc_lower, maximum_soc_fraction)
    )
    normal_c_rate = max(float(config.normal_dispatch_c_rate), 1e-3)
    storage_ramp_fraction = max(float(config.storage_power_ramp_fraction_per_hour), 1e-3)
    for site_index in range(site_count):
        power_expansion_index = layout.storage_power_expansion.start + site_index
        energy_expansion_index = layout.storage_energy_expansion.start + site_index
        base_power = float(storage_base_power[site_index])
        base_energy = float(storage_base_energy[site_index])
        equalities.add(
            [
                (layout.soc_index(0, site_index), 1.0),
                (layout.soc_index(hours, site_index), -1.0),
            ],
            0.0,
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
                [(discharge_variable, 1.0), (emergency_variable, 1.0), (power_expansion_index, -1.0)],
                base_power,
            )
            inequalities.add(
                [(charge_variable, 1.0), (energy_expansion_index, -normal_c_rate)],
                normal_c_rate * base_energy,
            )
            inequalities.add(
                [(discharge_variable, 1.0), (energy_expansion_index, -normal_c_rate)],
                normal_c_rate * base_energy,
            )
            previous_hour = (hour - 1) % hours
            previous_charge = layout.charge_index(previous_hour, site_index)
            previous_discharge = layout.discharge_index(previous_hour, site_index)
            inequalities.add(
                [
                    (discharge_variable, 1.0),
                    (charge_variable, -1.0),
                    (previous_discharge, -1.0),
                    (previous_charge, 1.0),
                    (power_expansion_index, -storage_ramp_fraction),
                ],
                storage_ramp_fraction * base_power,
            )
            inequalities.add(
                [
                    (discharge_variable, -1.0),
                    (charge_variable, 1.0),
                    (previous_discharge, 1.0),
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
    if not result.success:
        raise RuntimeError(f"Stage 14 zero-unserved DC-OPF is infeasible: {result.message}")

    solution = np.asarray(result.x, dtype=np.float64)
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
    dispatched_load = gross_load.astype(np.float32)
    dispatched_available = np.zeros((hours, bus_count), dtype=np.float32)
    dispatched_generation = np.zeros((hours, bus_count), dtype=np.float32)
    dispatched_available[:, renewable_bus_positions] = renewable_available.astype(np.float32)
    dispatched_generation[:, generator_bus_positions] = generation.astype(np.float32)
    planned_thermal_capacity = thermal_base_capacity + thermal_expansion
    dispatched_available[:, thermal_bus_positions] = planned_thermal_capacity.astype(np.float32)[None, :]
    for site_index, bus_position in enumerate(site_bus_positions):
        dispatched_load[:, bus_position] += charge[:, site_index].astype(np.float32)
        dispatched_available[:, bus_position] += float(storage_power_capacity[site_index])
        dispatched_generation[:, bus_position] += discharge[:, site_index].astype(np.float32)

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
    )
    dispatched_power_flow = solve_dc_power_flow(dispatched_forecast, topology, planned_electrical)
    baseline_thermal = baseline_power_flow.dispatched_generation_mw[:, thermal_bus_positions].sum(axis=1)
    scheduled_thermal = dispatched_power_flow.dispatched_generation_mw[:, thermal_bus_positions].sum(axis=1)
    dispatched_unserved = dispatched_power_flow.unserved_load_mw.sum(axis=1).astype(np.float32)
    dispatched_unserved[dispatched_unserved < 1e-3] = 0.0
    target_soc = np.broadcast_to(cycle_boundary_soc[None, :], (hours, site_count)).copy()
    store = StorageDispatchStore(
        timestamps=baseline_power_flow.timestamps.copy(),
        site_ids=np.asarray([site.site_id for site in sites], dtype=np.int32),
        site_bus_ids=np.asarray([site.bus_id for site in sites], dtype=np.int32),
        site_power_capacity_mw=storage_power_capacity.astype(np.float32),
        site_energy_capacity_mwh=storage_energy_capacity.astype(np.float32),
        storage_power_expansion_mw=storage_power_expansion.astype(np.float32),
        storage_energy_expansion_mwh=storage_energy_expansion.astype(np.float32),
        charge_mw=charge.astype(np.float32),
        discharge_mw=discharge.astype(np.float32),
        emergency_discharge_mw=emergency_discharge.astype(np.float32),
        soc_mwh=soc.astype(np.float32),
        target_soc_mwh=target_soc.astype(np.float32),
        cycle_boundary_soc_mwh=cycle_boundary_soc.astype(np.float32),
        minimum_soc_fraction=minimum_soc_fraction,
        maximum_soc_fraction=maximum_soc_fraction,
        preferred_soc_lower_fraction=preferred_soc_lower,
        preferred_soc_upper_fraction=preferred_soc_upper,
        total_load_mw=gross_load.sum(axis=1).astype(np.float32),
        renewable_available_mw=renewable_available.sum(axis=1).astype(np.float32),
        baseline_thermal_mw=baseline_thermal.astype(np.float32),
        scheduled_thermal_mw=scheduled_thermal.astype(np.float32),
        baseline_unserved_mw=baseline_power_flow.unserved_load_mw.sum(axis=1).astype(np.float32),
        dispatched_unserved_mw=dispatched_unserved,
        baseline_curtailed_mw=baseline_power_flow.curtailed_generation_mw.sum(axis=1).astype(np.float32),
        dispatched_curtailed_mw=dispatched_power_flow.curtailed_generation_mw.sum(axis=1).astype(np.float32),
        branch_ids=branch_ids,
        line_capacity_expansion_mva=line_expansion.astype(np.float32),
        thermal_bus_ids=bus_ids[thermal_bus_positions].copy(),
        thermal_capacity_expansion_mw=thermal_expansion.astype(np.float32),
        baseline_line_loading_ratio=baseline_power_flow.line_loading_ratio.copy(),
        dispatched_line_loading_ratio=dispatched_power_flow.line_loading_ratio.copy(),
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
    ) -> None:
        self.hours = hours
        self.branch_count = branch_count
        self.generator_count = generator_count
        self.site_count = site_count
        offset = 0

        def allocate(size: int) -> slice:
            nonlocal offset
            result = slice(offset, offset + size)
            offset += size
            return result

        self.generation = allocate(hours * generator_count)
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
        self.size = offset

    def generation_index(self, hour: int, generator: int) -> int:
        return self.generation.start + hour * self.generator_count + generator

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
    non_slack = np.asarray([index for index in range(bus_count) if index != slack_position], dtype=np.int32)
    reduced = b_matrix[np.ix_(non_slack, non_slack)]
    try:
        inverse = np.linalg.inv(reduced)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(reduced)
    ptdf = np.zeros((from_positions.size, bus_count), dtype=np.float64)
    ptdf[:, non_slack] = susceptance[:, None] * incidence[:, non_slack] @ inverse
    return ptdf


def _clean(values: np.ndarray, tolerance: float = 1e-7) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[np.abs(result) < tolerance] = 0.0
    return result
