from dataclasses import replace

import numpy as np
import pytest

from world_generator.core.config import StorageConfig
from world_generator.core.datatypes import (
    BranchElectricalParam, GridBus, GridEdge, GridElectricalState,
    RefinedGridTopologyState, SourceLoadForecastStore, StoragePlanStore, StorageSite,
)
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.operation.power_flow import _balance_dispatch, solve_dc_power_flow
from world_generator.operation.storage_dispatch import _build_ptdf, dispatch_storage_week


def network(load, generation, *, disconnected=False, generation_kind="wind_bus"):
    load = np.asarray(load, dtype=np.float32)
    generation = np.asarray(generation, dtype=np.float32)
    hours = load.size
    buses = (
        GridBus(10, generation_kind, 0, 0, 0., 0., 20., 1., 0., "test", 0),
        GridBus(20, "load_bus", 0, 1, 1., 0., 10., 1., 0., "test", 1),
    )
    edges = () if disconnected else (GridEdge(7, 10, 20, 1., 1., False, (0, 0), (0, 1)),)
    zeros = np.zeros((2, 2), dtype=np.float32)
    topology = RefinedGridTopologyState(zeros, zeros.astype(np.int16), zeros.astype(np.int16), buses, edges)
    electrical = build_grid_electrical(topology)
    loads = np.column_stack((np.zeros(hours), load)).astype(np.float32)
    supply = np.column_stack((generation, np.zeros(hours))).astype(np.float32)
    forecast = SourceLoadForecastStore(
        np.arange(hours, dtype=np.int32), np.array([10, 20]), tuple(bus.kind for bus in buses),
        loads, supply, supply.copy(), np.zeros_like(loads), (),
    )
    return topology, electrical, forecast


def storage_plan(hours, power=4., energy=8.):
    return StoragePlanStore(
        np.arange(hours), np.array([20]), np.array([0]), np.zeros((hours, 1)),
        (StorageSite(0, 20, 0, 1, (20,), power, energy, .5 * energy, 1.),),
    )


def test_dc_branch_units_and_nodal_balance():
    topology, electrical, forecast = network([3., 7.], [3., 7.])
    flow = solve_dc_power_flow(forecast, topology, electrical)
    branch = electrical.branch_params[0]
    expected_angle = -forecast.p_load_mw[:, 1] * branch.x_ohm / branch.nominal_kv**2
    np.testing.assert_allclose(flow.bus_angle_rad[:, 1], expected_angle, rtol=1e-6)
    np.testing.assert_allclose(flow.line_flow_mw[:, 0], [3., 7.], rtol=1e-6)
    np.testing.assert_allclose(flow.line_flow_mw[:, 0], flow.bus_p_injection_mw[:, 0], atol=1e-6)


def test_island_cannot_import_from_a_global_slack():
    topology, electrical, forecast = network([5., 5.], [8., 8.], disconnected=True)
    flow = solve_dc_power_flow(forecast, topology, electrical)
    np.testing.assert_allclose(flow.unserved_load_mw[:, 1], 5.)
    np.testing.assert_allclose(flow.curtailed_generation_mw[:, 0], 8.)
    np.testing.assert_allclose(flow.bus_p_injection_mw, 0.)


def test_thermal_is_backed_down_before_renewables():
    _, generation, _, _ = _balance_dispatch(
        np.array([0., 0., 6.]), np.array([5., 5., 0.]),
        np.array([True, False, False]), np.array([True, True, False]),
    )
    np.testing.assert_allclose(generation, [5., 1., 0.])


def test_invalid_bus_order_and_reactance_are_rejected():
    topology, electrical, forecast = network([3.], [3.])
    with pytest.raises(ValueError, match="bus order"):
        solve_dc_power_flow(replace(forecast, bus_ids=forecast.bus_ids[::-1]), topology, electrical)
    invalid = replace(electrical, branch_params=(replace(electrical.branch_params[0], x_ohm=0.),))
    with pytest.raises(ValueError, match="reactance"):
        solve_dc_power_flow(forecast, topology, invalid)


def test_ptdf_uses_independent_island_references():
    ptdf = _build_ptdf(4, np.array([0, 2]), np.array([1, 3]), np.array([5., 7.]), 0)
    np.testing.assert_allclose(ptdf @ np.array([3., -3., 4., -4.]), [3., 4.])


def test_storage_energy_balance_excludes_simultaneous_modes_even_with_spill_reward():
    topology, electrical, forecast = network([1.] * 4, [10.] * 4)
    baseline = solve_dc_power_flow(forecast, topology, electrical)
    config = replace(
        StorageConfig(), storage_cycle_cost=0., emergency_discharge_cost=0., soc_band_penalty=0.,
        renewable_dispatch_credit=100., max_storage_power_expansion_fraction=0.,
        max_storage_energy_expansion_fraction=0., max_line_expansion_fraction=0.,
        minimum_soc_fraction=0., maximum_soc_fraction=1.,
    )
    storage, _, flow, _ = dispatch_storage_week(topology, electrical, baseline, storage_plan(4), config)
    assert np.max(np.minimum(storage.charge_mw, storage.discharge_mw)) < 1e-5
    np.testing.assert_allclose(
        np.diff(storage.soc_mwh, axis=0),
        config.charge_efficiency * storage.charge_mw - storage.discharge_mw / config.discharge_efficiency,
        atol=2e-6,
    )
    np.testing.assert_allclose(storage.soc_mwh[0], storage.soc_mwh[-1], atol=2e-6)
    assert np.max(np.abs(np.diff(storage.discharge_mw - storage.charge_mw, axis=0))) <= 1. + 1e-5
    assert np.max(flow.unserved_load_mw) < 1e-5


def test_noncyclic_horizon_uses_configured_initial_energy():
    topology, electrical, forecast = network([3.], [0.])
    baseline = solve_dc_power_flow(forecast, topology, electrical)
    config = replace(StorageConfig(), cyclic_state_of_charge=False, minimum_soc_fraction=0.,
                     maximum_soc_fraction=1., max_storage_power_expansion_fraction=0.,
                     max_storage_energy_expansion_fraction=0.)
    # This case isolates the initial energy boundary. The explicit preceding
    # 3 MW discharge also makes the first ramp feasible; an unspecified prior
    # power must no longer grant a free jump in the first operating interval.
    storage, _, flow, _ = dispatch_storage_week(
        topology, electrical, baseline, storage_plan(1, 5., 10.), config,
        previous_storage_net_mw_by_site_id={0: 3.0},
    )
    np.testing.assert_allclose(storage.soc_mwh[0], [5.], atol=1e-6)
    np.testing.assert_allclose(storage.soc_mwh[-1], [5. - 3. / .95], atol=1e-5)
    assert np.sum(flow.unserved_load_mw) < 1e-5


def test_opf_does_not_hide_infeasible_islands_or_invalid_efficiency():
    topology, electrical, forecast = network([5.] * 2, [10.] * 2, disconnected=True)
    baseline = solve_dc_power_flow(forecast, topology, electrical)
    plan = replace(storage_plan(2), sites=())
    with pytest.raises(RuntimeError, match="infeasible"):
        dispatch_storage_week(topology, electrical, baseline, plan, replace(StorageConfig(), allow_load_shedding=False))
    with pytest.raises(ValueError, match="efficiency"):
        dispatch_storage_week(topology, electrical, baseline, plan, replace(StorageConfig(), charge_efficiency=1.1))


def test_bounded_investment_reports_nodal_shortfall_without_fabricating_supply():
    topology, electrical, forecast = network([5.] * 2, [10.] * 2, disconnected=True)
    baseline = solve_dc_power_flow(forecast, topology, electrical)
    plan = replace(storage_plan(2), sites=())
    storage, requested, flow, _ = dispatch_storage_week(topology, electrical, baseline, plan, StorageConfig())
    np.testing.assert_allclose(flow.unserved_load_mw[:, 1], 5.)
    np.testing.assert_allclose(requested.p_load_mw, forecast.p_load_mw)
    np.testing.assert_allclose(flow.served_load_mw + flow.unserved_load_mw, requested.p_load_mw)
    np.testing.assert_allclose(flow.bus_p_injection_mw, 0.)
    np.testing.assert_allclose(storage.dispatched_unserved_mw, 5.)
