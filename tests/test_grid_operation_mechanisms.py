"""F analytic operating cases: outages, exogenous supply, boundary state/reserve."""
from dataclasses import replace

import numpy as np
import pytest

from world_generator.core.config import StorageConfig
from world_generator.core.datatypes import (
    GridBus, GridEdge, RefinedGridTopologyState, SourceLoadForecastStore,
    StoragePlanStore, StorageSite, StorageDispatchStore, PowerFlowStore,
)
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.storage_dispatch import dispatch_storage_week


def _case(load=(5., 5., 5.), available=(10., 10., 10.), *, second_load=False, kind="thermal_bus", power=0., energy=0.):
    hours = len(load)
    buses = [GridBus(10, kind, 0, 0, 0., 0., 20., 1., 0., "fixture", 0),
             GridBus(20, "load_bus", 0, 1, 1., 0., 20., 1., 0., "fixture", 1)]
    edges = [GridEdge(7, 10, 20, 1., 1., False, (0, 0), (0, 1))]
    if second_load:
        buses.append(GridBus(30, "load_bus", 0, 2, 2., 0., 20., 1., 0., "fixture", 2))
        edges.append(GridEdge(8, 10, 30, 1., 1., False, (0, 0), (0, 2)))
    zeros = np.zeros((1, len(buses)))
    topology = RefinedGridTopologyState(zeros, zeros, zeros, tuple(buses), tuple(edges))
    electrical = build_grid_electrical(topology)
    requested = np.zeros((hours, len(buses))); requested[:, 1:] = np.asarray(load)[:, None]
    supply = np.zeros_like(requested); supply[:, 0] = available
    source = SourceLoadForecastStore(np.arange(hours), np.array([b.bus_id for b in buses]), tuple(b.kind for b in buses),
                                     requested, supply, supply.copy(), np.zeros_like(requested), ())
    sites = () if power == 0 and energy == 0 else (StorageSite(4, 20, 0, 1, (20,), power, energy, .5*energy, 1.),)
    plan = StoragePlanStore(np.arange(hours), np.array([20]), np.array([4]), np.zeros((hours, 1)), sites)
    return topology, electrical, source, plan


def _config(**overrides):
    values = dict(cyclic_state_of_charge=False, minimum_soc_fraction=0., maximum_soc_fraction=1.,
                  preferred_soc_lower_fraction=0., preferred_soc_upper_fraction=1., normal_dispatch_c_rate=2.,
                  storage_power_ramp_fraction_per_hour=1., thermal_ramp_fraction_per_hour=1.,
                  thermal_operating_limit_ratio=1., line_operating_limit_ratio=1.,
                  max_thermal_expansion_mw_per_bus=0., max_storage_power_expansion_fraction=0.,
                  max_storage_energy_expansion_fraction=0., max_line_expansion_fraction=0.)
    values.update(overrides)
    return replace(StorageConfig(), **values)


def _run(case, config=None, **kwargs):
    topology, electrical, source, plan = case
    baseline = solve_dc_power_flow(source, topology, electrical)
    return dispatch_storage_week(topology, electrical, baseline, plan, config or _config(),
                                 source_forecast=source, **kwargs)


def test_hourly_outage_recomputes_local_islands_with_persistent_branch_ids():
    case = _case(second_load=True, available=(15., 15., 15.))
    mask = np.array([[1, 1], [0, 1], [1, 1]], bool)
    topology, electrical, source, _ = case
    screened = solve_dc_power_flow(source, topology, electrical, branch_in_service=mask)
    assert screened.unserved_load_mw[1, 1] == 5
    assert screened.unserved_load_mw[1, 2] == 0
    dispatch, _, flow, _ = _run(case, fixed_capacity=True, branch_in_service=mask, thermal_land_limits_mw={10:20.})
    np.testing.assert_array_equal(flow.branch_ids, [7, 8])
    assert flow.line_flow_mw[1, 0] == 0
    np.testing.assert_allclose(dispatch.operation_arrays["unserved_load_mw"], [[0,0,0],[0,5,0],[0,0,0]])
    assert dispatch.operation_arrays["island_id"][1, 1] != dispatch.operation_arrays["island_id"][1, 2]
    np.testing.assert_allclose(flow.line_flow_mw[:, 1], 5)
    for hour in range(3):
        for island in np.unique(dispatch.operation_arrays["island_id"][hour]):
            members = dispatch.operation_arrays["island_id"][hour] == island
            assert abs(flow.bus_p_injection_mw[hour, members].sum()) < 1e-8


def test_true_thermal_availability_can_be_zero_despite_nameplate_and_expansion():
    case = _case(available=(10., 0., 10.))
    config = _config(max_thermal_expansion_mw_per_bus=100., thermal_capacity_cost=0.)
    dispatch, _, flow, _ = _run(case, config, thermal_land_limits_mw={10:25.})
    assert dispatch.operation_arrays["thermal_dispatch_mw"][1, 0] == 0
    assert dispatch.operation_arrays["thermal_available_mw"][1, 0] == 0
    assert flow.unserved_load_mw[1, 1] == 5
    assert dispatch.operation_arrays["thermal_installed_capacity_mw"][0] <= 25


def test_exogenous_source_reordering_and_new_transit_zero_fill():
    topology, electrical, source, plan = _case()
    transit = GridBus(40, "transit_bus", 0, 2, 2., 0., 0., 0., 0., "fixture", 4)
    topology = replace(topology, refined_buses=(*topology.refined_buses, transit))
    electrical = build_grid_electrical(topology)
    # Baseline deliberately has all axes in operating order, E source reversed.
    padded = replace(source, bus_ids=np.array([10,20,40]), bus_kinds=(*source.bus_kinds,"transit_bus"),
                     p_load_mw=np.pad(source.p_load_mw,((0,0),(0,1))),
                     p_gen_available_mw=np.pad(source.p_gen_available_mw,((0,0),(0,1))),
                     p_gen_scheduled_mw=np.pad(source.p_gen_scheduled_mw,((0,0),(0,1))),q_load_mvar=np.zeros((3,3)))
    reversed_source = replace(source, bus_ids=source.bus_ids[::-1], bus_kinds=source.bus_kinds[::-1],
                              p_load_mw=source.p_load_mw[:,::-1],p_gen_available_mw=source.p_gen_available_mw[:,::-1],
                              p_gen_scheduled_mw=source.p_gen_scheduled_mw[:,::-1])
    dispatch, _, _, _ = dispatch_storage_week(topology,electrical,solve_dc_power_flow(padded,topology,electrical),plan,_config(),
                                              source_forecast=reversed_source,fixed_capacity=True)
    np.testing.assert_array_equal(dispatch.operation_arrays["requested_load_mw"],padded.p_load_mw)
    assert np.all(dispatch.operation_arrays["generation_available_mw"][:,2] == 0)


def test_fixed_assets_forbid_all_four_capacity_expansions_and_reject_existing_land_excess():
    case = _case(load=(30.,), available=(20.,), power=1., energy=1.)
    config = _config(max_thermal_expansion_mw_per_bus=100.,max_line_expansion_fraction=3.,
                     max_storage_power_expansion_fraction=3.,max_storage_energy_expansion_fraction=3.)
    dispatch, _, _, _ = _run(case,config,fixed_capacity=True,thermal_land_limits_mw={10:25.})
    for field in ("thermal_capacity_expansion_mw","line_capacity_expansion_mva","storage_power_expansion_mw","storage_energy_expansion_mwh"):
        assert np.all(getattr(dispatch,field) == 0)
    with pytest.raises(ValueError,match="land limit"):
        _run(case,config,thermal_land_limits_mw={10:19.})
    with pytest.raises(ValueError,match="every"):
        _run(case,config,thermal_land_limits_mw={})


def test_soc_and_first_hour_ramp_use_declared_previous_storage_net_power():
    case = _case(load=(3.,),available=(0.,),kind="wind_bus",power=5.,energy=10.)
    config = _config(storage_power_ramp_fraction_per_hour=.25)
    first,_,_,_ = _run(case,config,fixed_capacity=True,initial_soc_mwh_by_site_id={4:5.})
    assert first.discharge_mw[0,0] == pytest.approx(1.25)
    assert first.dispatched_unserved_mw[0] == pytest.approx(1.75)
    continued,_,_,_ = _run(case,config,fixed_capacity=True,initial_soc_mwh_by_site_id={4:5.},previous_storage_net_mw_by_site_id={4:3.})
    assert continued.discharge_mw[0,0] == pytest.approx(3.)
    np.testing.assert_allclose(np.diff(continued.soc_mwh,axis=0),config.charge_efficiency*continued.charge_mw-continued.discharge_mw/config.discharge_efficiency)


def test_soc_continues_across_islanding_and_reconnection():
    case = _case(power=5.,energy=10.)
    config = _config(storage_cycle_cost=10.,emergency_discharge_cost=10.,charge_efficiency=1.,discharge_efficiency=1.)
    result,_,flow,_ = _run(case,config,fixed_capacity=True,branch_in_service=np.array([[1],[0],[1]],bool),initial_soc_mwh_by_site_id={4:8.})
    assert result.discharge_mw[1,0] == pytest.approx(5.)
    assert flow.line_flow_mw[1,0] == 0
    assert result.soc_mwh[2,0] == pytest.approx(result.soc_mwh[1,0]-5)
    np.testing.assert_allclose(np.diff(result.soc_mwh,axis=0),result.charge_mw-result.discharge_mw,atol=1e-8)


@pytest.mark.parametrize("energy,response,expected",[(2.,1.,.475),(20.,.1,.1),(20.,1.,1.)])
def test_storage_reserve_respects_energy_duration_efficiency_and_response_ramp(energy,response,expected):
    case = _case(load=(0.,),available=(0.,),kind="wind_bus",power=4.,energy=energy)
    # A small load makes the isolated island's contingency requirement explicit;
    # another generator provides its deterministic supply, leaving reserve local.
    topology,electrical,source,plan=case
    source=replace(source,p_load_mw=np.array([[0.,.01]]),p_gen_available_mw=np.array([[.01,0.]]),p_gen_scheduled_mw=np.array([[.01,0.]]))
    config=_config(reserve_contingency_mw=2.,reserve_duration_hours=2.,reserve_response_hours=response,
                   storage_power_ramp_fraction_per_hour=.25,storage_cycle_cost=10.)
    result,_,_,_=_run((topology,electrical,source,plan),config,fixed_capacity=True)
    reserve=result.operation_arrays["storage_reserve_mw"][0,0]
    assert reserve == pytest.approx(expected,abs=1e-6)
    assert result.charge_mw[0,0] == 0
    assert result.operation_arrays["reserve_shortfall_mw"].sum() == pytest.approx(2.-reserve)
    assert result.emergency_discharge_mw[0,0] == 0


def test_reserve_island_cannot_borrow_disconnected_thermal_headroom():
    case=_case(load=(5.,),available=(20.,))
    result,_,_,_=_run(case,_config(reserve_load_fraction=.2),fixed_capacity=True,branch_in_service=np.array([[0]],bool))
    assert result.operation_arrays["reserve_requirement_mw"][0,1] == 1
    assert result.operation_arrays["reserve_shortfall_mw"][0,1] == 1
    assert result.operation_arrays["thermal_reserve_mw"].sum() == 0


def test_zero_ramp_really_holds_pre_window_power_and_small_ens_is_retained():
    case=_case(load=(.0005,),available=(20.,))
    result,_,_,_=_run(case,_config(thermal_ramp_fraction_per_hour=0),fixed_capacity=True)
    assert result.operation_arrays["thermal_dispatch_mw"][0,0] == 0
    assert result.dispatched_unserved_mw[0] == pytest.approx(.0005)
    np.testing.assert_array_equal(result.dispatched_unserved_mw,result.operation_arrays["unserved_load_mw"].sum(axis=1))


def test_modern_outputs_round_trip_and_keep_physical_generation_separate_from_storage():
    result,_,flow,_=_run(_case(load=(3.,),available=(0.,),power=5.,energy=10.),fixed_capacity=True)
    saved=StorageDispatchStore.from_arrays(result.as_arrays())
    reread=PowerFlowStore.from_arrays(flow.as_arrays())
    np.testing.assert_array_equal(saved.operation_arrays["bus_ids"],[10,20])
    assert saved.operation_arrays["generator_dispatch_mw"].sum() == 0
    assert saved.discharge_mw.sum() == pytest.approx(3.)
    np.testing.assert_array_equal(reread.operation_arrays["requested_load_mw"],[[0,3]])


def test_surplus_energy_cannot_charge_and_offer_reserve_in_the_same_hour():
    case=_case(load=(.01,),available=(10.,),kind="wind_bus",power=4.,energy=2.)
    config=_config(reserve_contingency_mw=2.,reserve_duration_hours=2.,reserve_response_hours=1.,
                   storage_power_ramp_fraction_per_hour=.25)
    result,_,_,_=_run(case,config,fixed_capacity=True)
    reserve=result.operation_arrays["storage_reserve_mw"]
    assert reserve[0,0] == pytest.approx(.475,abs=1e-6)
    assert result.charge_mw[0,0] == 0
    assert np.max(np.minimum(result.charge_mw,result.discharge_mw+reserve)) < 1e-8


def test_thermal_reserve_is_limited_by_availability_headroom_and_response_ramp():
    case=_case(load=(5.,),available=(6.,))
    config=_config(thermal_ramp_fraction_per_hour=.1,reserve_load_fraction=.2,reserve_response_hours=.25)
    result,_,_,_=_run(case,config,fixed_capacity=True,previous_thermal_mw_by_bus_id={10:5.})
    assert result.operation_arrays["thermal_dispatch_mw"][0,0] == pytest.approx(5.)
    assert result.operation_arrays["thermal_reserve_mw"][0,0] == pytest.approx(.5)
    assert result.operation_arrays["reserve_shortfall_mw"].sum() == pytest.approx(.5)
    assert result.operation_arrays["thermal_unused_available_mw"].sum() == pytest.approx(1.)
    assert result.operation_arrays["renewable_curtailment_mw"].sum() == 0


def test_default_costs_serve_demand_before_holding_unavailable_reserve():
    case=_case(load=(10.,),available=(10.,))
    result,_,_,_=_run(case,_config(reserve_load_fraction=.1),fixed_capacity=True)
    assert StorageConfig().reserve_shortfall_cost < StorageConfig().load_shedding_cost
    assert result.dispatched_unserved_mw[0] == 0
    assert result.operation_arrays["thermal_dispatch_mw"][0,0] == pytest.approx(10.)
    assert result.operation_arrays["thermal_reserve_mw"].sum() == 0
    assert result.operation_arrays["reserve_shortfall_mw"].sum() == pytest.approx(1.)


@pytest.mark.parametrize("problem",["bad_status","duplicate_branch","negative_initial","overfull_initial","wrong_kind","renewable_over_nameplate"])
def test_invalid_operating_contracts_are_rejected(problem):
    topology,electrical,source,plan=_case(power=4.,energy=8.,kind="wind_bus")
    kwargs={"fixed_capacity":True}
    if problem == "bad_status": kwargs["branch_in_service"]=np.full((3,1),.5)
    if problem == "duplicate_branch": electrical=replace(electrical,branch_params=electrical.branch_params*2)
    if problem == "negative_initial": kwargs["initial_soc_mwh_by_site_id"]={4:-1.}
    if problem == "overfull_initial": kwargs["initial_soc_mwh_by_site_id"]={4:9.}
    if problem == "wrong_kind": source=replace(source,bus_kinds=("thermal_bus","load_bus"))
    if problem == "renewable_over_nameplate": source=replace(source,p_gen_available_mw=source.p_gen_available_mw*3)
    with pytest.raises(ValueError):
        _run((topology,electrical,source,plan),**kwargs)
