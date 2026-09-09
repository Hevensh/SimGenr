"""G: configured infeasibility is distinct from solver and input failures."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import optimize

from scripts import generate_static_world as entry
from world_generator.core.config import StorageConfig, SourceLoadConfig, load_world_config
from world_generator.core.datatypes import GridBus, GridEdge, RefinedGridTopologyState, SourceLoadForecastStore, StoragePlanStore
from world_generator.core.errors import PhysicalInfeasibilityError, SolverError, generation_failure_category
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.storage_dispatch import dispatch_storage_week, _solve_exclusive_storage_modes, _VariableLayout, _SparseConstraintBuilder


def _isolated_dispatch(*, allow_shedding):
    buses = (GridBus(10,"thermal_bus",0,0,0.,0.,20.,1.,0.,"fixture",0),
             GridBus(20,"load_bus",0,1,1.,0.,20.,1.,0.,"fixture",1))
    edges = (GridEdge(7,10,20,1.,1.,False,(0,0),(0,1)),)
    z = np.zeros((1,2))
    topology = RefinedGridTopologyState(z,z,z,buses,edges)
    electrical = build_grid_electrical(topology)
    source = SourceLoadForecastStore(np.array([0]),np.array([10,20]),("thermal_bus","load_bus"),
        np.array([[0.,5.]]),np.array([[20.,0.]]),np.array([[20.,0.]]),z,())
    plan = StoragePlanStore(np.array([0]),np.array([20]),np.array([-1]),np.zeros((1,0)),())
    baseline = solve_dc_power_flow(source,topology,electrical)
    config = replace(StorageConfig(),allow_load_shedding=allow_shedding,cyclic_state_of_charge=False,
                     thermal_ramp_fraction_per_hour=1.)
    return dispatch_storage_week(topology,electrical,baseline,plan,config,source_forecast=source,
                                 fixed_capacity=True,branch_in_service=np.array([[False]]),thermal_land_limits_mw={10:20.})


def test_strict_no_shedding_on_an_isolated_load_is_typed_physical_infeasibility():
    with pytest.raises(PhysicalInfeasibilityError) as caught:
        _isolated_dispatch(allow_shedding=False)
    assert isinstance(caught.value,RuntimeError)
    assert caught.value.stage == "stage_14_dispatch"
    assert generation_failure_category(caught.value) == "PHYSICAL_INFEASIBILITY"


@pytest.mark.parametrize("status",[1,3,4])
def test_lp_solver_termination_is_not_evidence_of_physical_infeasibility(monkeypatch,status):
    calls = []
    def failed(*args,**kwargs):
        calls.append(kwargs.get("method"))
        return SimpleNamespace(success=False,status=status,message="injected solver termination")
    monkeypatch.setattr(optimize,"linprog",failed)
    with pytest.raises(SolverError) as caught:
        _isolated_dispatch(allow_shedding=True)
    assert generation_failure_category(caught.value) == "GENERATION_ERROR"
    assert calls == (["highs","highs-ipm"] if status == 4 else ["highs"])


@pytest.mark.parametrize("status,exception",[(2,PhysicalInfeasibilityError),(1,SolverError),(3,SolverError),(4,SolverError)])
def test_milp_status_two_is_the_only_physical_infeasibility_status(monkeypatch,status,exception):
    monkeypatch.setattr(optimize,"milp",lambda *a,**k:SimpleNamespace(success=False,status=status,message="injected MILP status"))
    layout = _VariableLayout(hours=1,branch_count=0,generator_count=0,site_count=1,thermal_count=0)
    eq,ub = _SparseConstraintBuilder(layout.size),_SparseConstraintBuilder(layout.size)
    eq.add([(layout.soc_index(0,0),1.)],0.)
    ub.add([(layout.charge_index(0,0),1.)],1.)
    with pytest.raises(exception):
        _solve_exclusive_storage_modes(np.zeros(layout.size),np.zeros(layout.size),np.ones(layout.size),eq,ub,layout,np.ones(1))


def test_entry_point_records_actual_infeasible_failure_and_rethrows(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,"PROJECT_ROOT",tmp_path)
    def generate(context):
        context.output_dir = tmp_path/"world_seed42"
        context.config_path = str(tmp_path/"config.yaml")
        context.seed = 42
        context.stage = "Stages 12-14 asset planning"
        _isolated_dispatch(allow_shedding=False)
    monkeypatch.setattr(entry,"_generate",generate)
    with pytest.raises(PhysicalInfeasibilityError):
        entry.main()
    report = json.loads((tmp_path/"world_seed42/generation_failure.json").read_text())
    assert report["status"] == "FAIL"
    assert report["category"] == "PHYSICAL_INFEASIBILITY"
    assert report["stage"] == "stage_14_dispatch"
    assert report["error_type"] == "PhysicalInfeasibilityError"
    assert report["seed"] == 42
    assert report["config"] == str(tmp_path/"config.yaml")


@pytest.mark.parametrize("problem,category",[("schema","INVALID_INPUT"),("config","INVALID_INPUT"),("bug","GENERATION_ERROR"),("solver","GENERATION_ERROR")])
def test_recorded_input_errors_and_generation_errors_remain_distinct(tmp_path,monkeypatch,problem,category):
    monkeypatch.setattr(entry,"PROJECT_ROOT",tmp_path)
    def generate(context):
        context.output_dir = tmp_path/"world"
        context.stage = "configuration" if problem == "config" else "Stage 11 source/load"
        if problem == "schema":
            source = SourceLoadForecastStore(np.array([0]),np.array([10.5]),("load_bus",),*(np.zeros((1,1)) for _ in range(4)),())
            source.as_arrays()
        elif problem == "config":
            SourceLoadConfig(unknown_schema_property=1)
        elif problem == "solver":
            raise SolverError("numerical factorization failed",stage="stage_14_dispatch")
        else:
            raise TypeError("internal implementation error")
    monkeypatch.setattr(entry,"_generate",generate)
    with pytest.raises((ValueError,TypeError,SolverError)):
        entry.main()
    assert json.loads((tmp_path/"world/generation_failure.json").read_text())["category"] == category


def test_a_successful_dispatch_with_ens_clears_stale_failure_marker(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,"PROJECT_ROOT",tmp_path)
    world = tmp_path/"world"; world.mkdir()
    marker = world/"generation_failure.json"; marker.write_text('{"status":"FAIL"}')
    results = []
    def generate(context):
        context.output_dir = world
        results.append(_isolated_dispatch(allow_shedding=True))
    monkeypatch.setattr(entry,"_generate",generate)
    entry.main()
    assert results[0][0].dispatched_unserved_mw[0] == 5.
    assert not marker.exists()


def test_unknown_or_outside_workspace_output_never_gets_a_failure_record(tmp_path,monkeypatch):
    allowed = tmp_path/"scope"; allowed.mkdir()
    outside = tmp_path/"elsewhere"
    monkeypatch.setattr(entry,"PROJECT_ROOT",allowed)
    for path in (None,outside,allowed):
        entry._record_generation_failure(entry.GenerationContext(output_dir=path),RuntimeError("failed before a valid world directory"))
    assert not outside.exists()
    assert not (allowed/"generation_failure.json").exists()


def test_cli_invalid_config_exits_nonzero_without_guessing_a_world_directory(tmp_path):
    config = tmp_path/"invalid.yaml"
    config.write_text("source_load:\n  unknown_schema_property: 1\n",encoding="utf-8")
    with pytest.raises(TypeError) as caught:
        load_world_config(config)
    assert generation_failure_category(caught.value,stage="configuration") == "INVALID_INPUT"
    output = tmp_path/"must_not_be_guessed"
    process = subprocess.run([sys.executable,str(entry.PROJECT_ROOT/"scripts/generate_static_world.py"),
                              "--config",str(config),"--output",str(output),"--no-figures"],
                              cwd=entry.PROJECT_ROOT,capture_output=True,text=True)
    assert process.returncode != 0
    assert "unknown_schema_property" in process.stderr
    assert not output.exists()


def test_cli_invalid_stage_records_failure_after_output_resolution_and_exits_nonzero(tmp_path):
    config = tmp_path/"valid.yaml"
    config.write_text("seed: 42\noutput:\n  world_name: failure_case\n",encoding="utf-8")
    output = tmp_path/"outputs"
    process = subprocess.run([sys.executable,str(entry.PROJECT_ROOT/"scripts/generate_static_world.py"),
                              "--config",str(config),"--output",str(output),"--from-stage","99","--no-figures"],
                              cwd=entry.PROJECT_ROOT,capture_output=True,text=True)
    assert process.returncode == 2
    report = json.loads((output/"failure_case_seed42/generation_failure.json").read_text())
    assert report["category"] == "INVALID_INPUT"
    assert report["error_type"] == "SystemExit"
    assert report["resolved_config"]["seed"] == report["seed"] == 42
