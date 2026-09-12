"""F export tampering counterexamples; assertions name the broken identity."""
from dataclasses import replace

import numpy as np
import pytest

from test_grid_storage_physics import network, storage_plan
from scripts.operation_validation import check_operation_contracts
from scripts.validate_world_physics import Checks
from world_generator.core.config import WorldConfig, StorageConfig
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.storage_dispatch import dispatch_storage_week


def _fixture():
    topology, electrical, source = network([5., 5., 5.], [20., 20., 20.], generation_kind="thermal_bus")
    cfg = replace(StorageConfig(), reserve_load_fraction=.1, cyclic_state_of_charge=False,
                  minimum_soc_fraction=0., maximum_soc_fraction=1., initial_soc_fraction=.5,
                  thermal_ramp_fraction_per_hour=1., storage_power_ramp_fraction_per_hour=1.)
    config = replace(WorldConfig(), storage=cfg)
    service = np.array([[True], [False], [True]])
    baseline = solve_dc_power_flow(source, topology, electrical, branch_in_service=service)
    result = dispatch_storage_week(topology, electrical, baseline, storage_plan(3, 4., 20.), cfg,
        source_forecast=source, fixed_capacity=True, branch_in_service=service, thermal_land_limits_mw={10: 20.})
    storage, _, flow, final_electrical = result
    arrays = source.as_arrays()
    arrays["nameplate_capacity_mw"] = np.array([20., 10.])
    return [storage.as_arrays(), flow.as_arrays(), final_electrical.as_arrays(), arrays, config]


def _check(args):
    checks = Checks()
    summary = check_operation_contracts(checks, *args)
    return {row["name"]: row for row in checks.rows}, summary


def test_exported_outage_initial_state_and_reserve_have_independent_accounts():
    rows, summary = _check(_fixture())
    assert all(row["passed"] for row in rows.values()), [r for r in rows.values() if not r["passed"]]
    assert summary["inactive_branch_hours"] == 1
    assert summary["max_simultaneous_islands"] == 2
    assert summary["exogenous_unserved_mwh"] >= 1.0


@pytest.mark.parametrize("field,index,delta,expected", [
    ("op__unserved_load_mw", (1, 1), .5, "served_plus_unserved"),
    ("op__generation_available_mw", (1, 0), 2., "exogenous_availability_preserved"),
    ("op__reserve_requirement_mw", (1, 1), .5, "reserve_requirement_configuration"),
    ("op__reserve_shortfall_mw", (0, 0), 1., "reserve_shortfall_within_requirement"),
    ("op__storage_reserve_mw", (1, 0), 100., "storage_reserve_inverter"),
    ("op__thermal_reserve_mw", (0, 0), 100., "thermal_with_reserve_availability"),
    ("op__thermal_land_limit_mw", (0,), -1., "thermal_land_capacity"),
    ("op__previous_storage_net_mw", (0,), 100., "storage_first_step_ramp_boundary"),
    ("op__island_id", (1, 1), -10, "island_labels_connectivity"),
])
def test_corrupted_operation_export_identifies_the_failure(field, index, delta, expected):
    args = _fixture()
    args[0][field][index] += delta
    rows, _ = _check(args)
    assert not rows["operation_" + expected]["passed"]


def test_fault_flow_cannot_be_hidden_by_an_unchanged_asset_list():
    args = _fixture()
    args[1]["line_flow_mw"][1, 0] = 1
    rows, _ = _check(args)
    assert not rows["operation_inactive_branch_zero_flow"]["passed"]
    assert not rows["operation_dc_angle_flow_relation"]["passed"]
