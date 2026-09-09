"""F controller counterfactuals: planning input independence, not label checks."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from world_generator.core.config import PlanningConfig, StorageConfig, WorldConfig, dump_config_snapshot, load_world_config
from world_generator.core.datatypes import GridBus, GridEdge, RefinedGridTopologyState, SourceLoadForecastStore
from world_generator.core.random_state import derive_module_seed
from world_generator.core.output_layout import WorldDataLayout
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.operation import asset_planning as ap
from world_generator.operation.grid_update_loop import GridUpdateIteration, GridUpdateLoopResult
from world_generator.operation.stage_cache import load_asset_boundary_checkpoint, save_stage12_checkpoint, load_stage13_checkpoint


def _case():
    buses = (GridBus(10,"thermal_bus",0,0,0.,0.,20.,1.,0.,"fixture",0),
             GridBus(20,"load_bus",0,1,1.,0.,20.,1.,0.,"fixture",1))
    edges = (GridEdge(7,10,20,1.,1.,False,(0,0),(0,1)),)
    zeros = np.zeros((2,2))
    topology = RefinedGridTopologyState(zeros,zeros,zeros,buses,edges)
    electrical = build_grid_electrical(topology)
    load = np.array([[0.,5.],[0.,8.],[0.,5.]],np.float32)
    available = np.array([[20.,0.]]*3,np.float32)
    source = SourceLoadForecastStore(np.arange(2400,2403),np.array([10,20]),("thermal_bus","load_bus"),
                                     load,available,available.copy(),np.zeros_like(load),())
    return topology,electrical,source


def _config(mode, **planning):
    storage = replace(StorageConfig(),cyclic_state_of_charge=False,minimum_soc_fraction=0.,maximum_soc_fraction=1.,
                      preferred_soc_lower_fraction=0.,preferred_soc_upper_fraction=1.,normal_dispatch_c_rate=2.,
                      thermal_operating_limit_ratio=1.,line_operating_limit_ratio=1.,thermal_ramp_fraction_per_hour=1.,
                      storage_power_ramp_fraction_per_hour=1.,max_thermal_expansion_mw_per_bus=0.,
                      max_storage_power_expansion_fraction=0.,max_storage_energy_expansion_fraction=0.,max_line_expansion_fraction=0.)
    return replace(WorldConfig(),storage=storage,planning=PlanningConfig(mode=mode,**planning))


def _run(path, config, source=None):
    topology,electrical,original = _case()
    return ap.run_asset_planning(config=config,terrain=None,hydrology=None,climate=None,land=None,land_use=None,
                                topology=topology,base_topology=None,electrical=electrical,
                                operation_source=original if source is None else source,thermal_land_limits_mw={10:20.},
                                output_dir=path / "data" / "planning")


def _independent_design(monkeypatch):
    """Small deterministic planner responding to design load; dispatch remains real."""
    def grid_update(source,topology,base,electrical,*args):
        electrical = replace(electrical,branch_params=(replace(electrical.branch_params[0],rate_mva=float(source.p_load_mw[-1,1])*5),))
        flow = ap.solve_dc_power_flow(source,topology,electrical)
        upgrade = ap.build_grid_upgrade_plan(flow,electrical)
        iteration = GridUpdateIteration(1,topology,electrical,flow,upgrade,(),flow.summary_dict())
        return GridUpdateLoopResult(flow.summary_dict(),(iteration,))
    monkeypatch.setattr(ap,"run_grid_update_loop",grid_update)
    monkeypatch.setattr(ap,"plan_storage_sites",lambda topology,need,storage: ap.fixed_storage_plan(topology,need.timestamps,_config("fixed_assets")))
    def weather(*args,**kwargs):
        rng = args[-1]
        return SimpleNamespace(as_arrays=lambda: {"design_random_value":np.asarray([rng.uniform()])})
    monkeypatch.setattr(ap,"generate_daily_weather",weather)
    monkeypatch.setattr(ap,"generate_hourly_weather_week",weather)
    def source(weather,topology,electrical,rng,**kwargs):
        original = _case()[2]
        load = original.p_load_mw.copy(); load[:,1] = 8+rng.uniform()
        return replace(original,timestamps=np.arange(3),p_load_mw=load)
    monkeypatch.setattr(ap,"generate_source_load_forecast",source)


def test_fixed_assets_never_calls_runtime_driven_asset_planners(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Fixed assets called a weather-driven asset planner")
    monkeypatch.setattr(ap,"run_grid_update_loop",forbidden)
    monkeypatch.setattr(ap,"plan_storage_sites",forbidden)
    source = _case()[2]
    changed = replace(source,p_load_mw=np.array([[0.,5.],[0.,8.],[0.,80.]],np.float32))
    first = _run(tmp_path/"one",_config("fixed_assets"),source)
    second = _run(tmp_path/"two",_config("fixed_assets"),changed)
    assert first.metadata["initial_assets_sha256"] == first.metadata["frozen_assets_sha256"]
    assert first.metadata["frozen_assets_sha256"] == second.metadata["frozen_assets_sha256"]
    assert not first.storage_plan.sites
    assert second.storage_dispatch.operation_arrays["unserved_load_mw"][-1,1] > 0
    np.testing.assert_array_equal(second.dispatched_forecast.timestamps,source.timestamps)


def test_preplanned_design_hash_and_assets_ignore_changed_operation_future(tmp_path,monkeypatch):
    _independent_design(monkeypatch)
    config = _config("preplanned",design_days=1,design_seed=81)
    source = _case()[2]
    changed = replace(source,p_load_mw=np.array([[0.,5.],[0.,8.],[0.,80.]],np.float32))
    first = _run(tmp_path/"one",config,source)
    second = _run(tmp_path/"two",config,changed)
    assert first.metadata["planning_input_hashes"] == second.metadata["planning_input_hashes"]
    assert first.metadata["frozen_assets_sha256"] == second.metadata["frozen_assets_sha256"]
    assert first.metadata["derived_planning_seed"] == derive_module_seed(config.seed,"planning:81")
    assert first.metadata["initial_assets_sha256"] != first.metadata["frozen_assets_sha256"]
    assert first.metadata["planning_time_bounds_hours"] == [0.,3.]
    assert first.metadata["operation_time_bounds_hours"] == [2400.,2403.]
    np.testing.assert_array_equal(second.dispatched_forecast.timestamps,source.timestamps)
    assert second.storage_dispatch.operation_arrays["unserved_load_mw"][-1,1] > 0
    load_asset_boundary_checkpoint(tmp_path/"one",expected_timestamps=source.timestamps)


def test_oracle_assets_actually_respond_to_changed_full_window(tmp_path,monkeypatch):
    _independent_design(monkeypatch)
    source = _case()[2]
    changed = replace(source,p_load_mw=np.array([[0.,5.],[0.,8.],[0.,12.]],np.float32))
    first = _run(tmp_path/"one",_config("full_window_planning"),source)
    second = _run(tmp_path/"two",_config("full_window_planning"),changed)
    assert first.metadata["planning_input_hashes"] != second.metadata["planning_input_hashes"]
    assert first.metadata["frozen_assets_sha256"] != second.metadata["frozen_assets_sha256"]
    assert second.metadata["planning_time_bounds_hours"] == second.metadata["operation_time_bounds_hours"]


def test_explicit_fixed_site_initial_soc_reaches_noncyclic_dispatch(tmp_path):
    config = _config("fixed_assets",fixed_storage_sites=({"site_id":91,"bus_id":20,"power_mw":2.,"energy_mwh":8.,"initial_soc_fraction":.25},))
    result = _run(tmp_path,config)
    assert result.storage_plan.sites[0].initial_soc_mwh == 2.
    assert result.storage_dispatch.soc_mwh[0,0] == 2.
    assert result.metadata["initial_state_policy"] == "fixed_prior_boundary"


def test_periodic_boundary_is_not_claimed_as_frozen_prior(tmp_path):
    config = _config("fixed_assets",fixed_storage_sites=({"bus_id":20,"power_mw":2.,"energy_mwh":8.},))
    config = replace(config,storage=replace(config.storage,cyclic_state_of_charge=True))
    result = _run(tmp_path,config)
    assert result.metadata["initial_state_policy"] == "optimized_periodic_runtime"
    assert ap.storage_initial_boundary(result.storage_plan,config) is None


def test_fault_mask_uses_persistent_id_and_only_changes_declared_intervals(tmp_path):
    config = _config("fixed_assets",line_faults=({"branch_id":7,"start_offset_hours":1,"duration_hours":1},))
    result = _run(tmp_path,config)
    np.testing.assert_array_equal(result.dispatched_power_flow.branch_ids,[7])
    np.testing.assert_array_equal(result.dispatched_power_flow.operation_arrays["branch_in_service"],[[1],[0],[1]])
    assert result.metadata["initial_assets_sha256"] == result.metadata["frozen_assets_sha256"]
    assert result.storage_dispatch.operation_arrays["unserved_load_mw"][1,1] == 8


def test_snapshot_hash_survives_asset_export_precision_and_rejects_tampering(tmp_path):
    result = _run(tmp_path,_config("fixed_assets"))
    arrays = load_asset_boundary_checkpoint(tmp_path)["frozen_assets"]
    reordered = dict(reversed(list(arrays.items())))
    assert ap.asset_snapshot_sha256(reordered) == result.metadata["frozen_assets_sha256"]
    arrays["bus_nameplate_capacity_mw"][0] += 1
    np.savez_compressed(tmp_path/"data/planning/frozen_assets.npz",**arrays)
    with pytest.raises(ValueError,match="hash mismatch"):
        load_asset_boundary_checkpoint(tmp_path)


def test_frozen_snapshot_hash_survives_existing_stage12_and_stage13_cache(tmp_path):
    config = _config("fixed_assets",fixed_storage_sites=({"bus_id":20,"power_mw":3.3333,"energy_mwh":7.7779,"initial_soc_fraction":.333},))
    result = _run(tmp_path,config)
    layout = WorldDataLayout(tmp_path/"data"); layout.create()
    save_stage12_checkpoint(result.checkpoint_iteration,layout.grid_update)
    np.savez_compressed(layout.topology/"static_maps.npz",elevation=np.zeros((2,2)))
    np.savez_compressed(layout.storage_planning/"storage_plan.npz",**result.storage_plan.as_arrays())
    (layout.storage_planning/"storage_plan.json").write_text(json.dumps({"sites":result.storage_plan.as_dicts()}),encoding="utf-8")
    _,topology,electrical,_,plan = load_stage13_checkpoint(tmp_path)
    assert ap.asset_snapshot_sha256(ap.asset_snapshot(topology,electrical,plan,{10:20.})) == result.metadata["frozen_assets_sha256"]


def test_config_snapshot_round_trip_and_explicit_mode_validation(tmp_path):
    config = _config("fixed_assets",fixed_storage_sites=[{"bus_id":20,"power_mw":2.,"energy_mwh":8.}])
    path = tmp_path/"config.yaml"
    dump_config_snapshot(config,path)
    assert config == load_world_config(path)
    assert WorldConfig().planning.mode == "full_window_planning"


@pytest.mark.parametrize("kwargs",[
    {"mode":"unknown"},{"design_days":0},{"design_seed":True},
    {"design_start_day_of_year":365},{"design_weather_overrides":{"days":1}},
    {"mode":"preplanned","fixed_storage_sites":[{"bus_id":20,"power_mw":2.,"energy_mwh":8.}]},
    {"line_faults":[{"branch_id":7,"start_offset_hours":0,"duration_hours":0}]},
    {"mode":"fixed_assets","fixed_storage_sites":[{"bus_id":20.1,"power_mw":2.,"energy_mwh":8.}]},
])
def test_bad_planning_config_fails_before_generation(kwargs):
    with pytest.raises(ValueError):
        PlanningConfig(**kwargs)


def test_land_bounds_and_fault_window_are_checked_before_operation(tmp_path):
    topology,electrical,source = _case()
    ledger = np.array([[10,0,0,1,20,20,1,19,20]],float)
    with pytest.raises(ValueError,match="reserved land"):
        ap.thermal_land_limits_from_ledger(ledger,topology)
    with pytest.raises(ValueError,match="ID/window"):
        ap.operation_branch_mask(electrical,source.timestamps,({"branch_id":8,"start_offset_hours":0,"duration_hours":1},))
    with pytest.raises(ValueError,match="ID/window"):
        ap.operation_branch_mask(electrical,source.timestamps,({"branch_id":7,"start_offset_hours":2,"duration_hours":2},))


def test_cached_land_saturated_capacity_recovers_only_proven_float32_rounding():
    topology,electrical,_ = _case()
    limit = 220.5274611711502
    cached = float(np.float32(limit))
    assert cached > limit
    topology = replace(topology,refined_buses=(replace(topology.refined_buses[0],capacity_mw=cached),topology.refined_buses[1]))
    electrical = replace(electrical,bus_params=(replace(electrical.bus_params[0],p_capacity_mw=cached),electrical.bus_params[1]))
    ledger = np.array([[10,0,0,1,limit,limit,1,limit,limit]],float)
    restored_topology,restored_electrical = ap.restore_cached_thermal_land_boundary(topology,electrical,ledger)
    assert restored_electrical.bus_params[0].p_capacity_mw == limit
    assert restored_topology.refined_buses[0].capacity_mw == limit
    assert float(np.float32(restored_electrical.bus_params[0].p_capacity_mw)) == cached
    # A neighbouring float32 number, or a non-export float64 overshoot, is
    # not evidence of rounding the land boundary and cannot be forgiven.
    for bad in (float(np.nextafter(np.float32(cached),np.float32(np.inf))),limit+1e-7,cached+.01):
        bad_electrical = replace(electrical,bus_params=(replace(electrical.bus_params[0],p_capacity_mw=bad),electrical.bus_params[1]))
        with pytest.raises(ValueError,match="beyond float32"):
            ap.restore_cached_thermal_land_boundary(topology,bad_electrical,ledger)


def test_frozen_thermal_availability_uses_float64_fraction_then_nameplate():
    topology,electrical,source = _case()
    installed = 220.5274611711502
    topology = replace(topology,refined_buses=(replace(topology.refined_buses[0],capacity_mw=installed),topology.refined_buses[1]))
    electrical = replace(electrical,bus_params=(replace(electrical.bus_params[0],p_capacity_mw=installed),electrical.bus_params[1]))
    availability = source.p_gen_available_mw.copy(); availability[:,0] = [0.,10.,20.]
    source = replace(source,nameplate_capacity_mw=np.array([20.,0.]),p_gen_available_mw=availability,p_gen_scheduled_mw=availability.copy())
    aligned = ap.align_source_to_assets(source,topology,electrical)
    assert aligned.p_gen_available_mw.dtype == np.float64
    np.testing.assert_array_equal(aligned.p_gen_available_mw[:,0],[0.,installed*.5,installed])
    assert aligned.p_gen_available_mw.max() <= installed
    bad = availability.copy(); bad[-1,0] = np.nextafter(np.float32(20),np.float32(np.inf))
    with pytest.raises(ValueError,match="declared nameplate"):
        ap.align_source_to_assets(replace(source,p_gen_available_mw=bad),topology,electrical)
