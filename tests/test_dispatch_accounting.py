from dataclasses import replace

import numpy as np
import pytest

from world_generator.core.config import StorageConfig, WorldGridConfig
from world_generator.core.datatypes import GridBus, GridEdge, RefinedGridTopologyState, SourceLoadForecastStore, StoragePlanStore
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.operation.grid_update_loop import _equivalent_path_segment, _new_branch_from_template, _resize_branch_multiplier
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.storage_dispatch import dispatch_storage_week


def _network():
    buses = (
        GridBus(10, "wind_bus", 0, 0, 0.0, 0.0, 20.0, 1.0, 0.0, "test", 0),
        GridBus(20, "load_bus", 0, 2, 1.0, 0.0, 10.0, 1.0, 0.0, "test", 1),
    )
    edge = GridEdge(7, 10, 20, 2.0, 2.0, False, (0, 0, 0), (0, 1, 2))
    zeros = np.zeros((3, 3), dtype=np.float32)
    topology = RefinedGridTopologyState(zeros, zeros.astype(int), zeros.astype(int), buses, (edge,))
    return topology, build_grid_electrical(topology)


def test_opf_curtailed_accounting_preserves_available_generation_without_undoing_dispatch():
    topology, electrical = _network()
    load = np.array([[0.0, 3.0], [0.0, 4.0]], dtype=np.float32)
    available = np.array([[10.0, 0.0], [12.0, 0.0]], dtype=np.float32)
    timestamps = np.arange(2, dtype=np.int32)
    forecast = SourceLoadForecastStore(timestamps, np.array([10, 20]), ("wind_bus", "load_bus"), load, available, available.copy(), np.zeros_like(load), ())
    baseline = solve_dc_power_flow(forecast, topology, electrical)
    plan = StoragePlanStore(timestamps, np.array([20]), np.array([-1]), np.empty((2, 0)), ())
    dispatch, scheduled, power_flow, _ = dispatch_storage_week(topology, electrical, baseline, plan, StorageConfig())
    np.testing.assert_allclose(scheduled.p_gen_available_mw[:, 0], [10.0, 12.0])
    np.testing.assert_allclose(scheduled.p_gen_scheduled_mw[:, 0], [3.0, 4.0])
    np.testing.assert_allclose(power_flow.line_flow_mw[:, 0], [3.0, 4.0], atol=1e-5)
    np.testing.assert_allclose(power_flow.curtailed_generation_mw[:, 0], [7.0, 8.0], atol=1e-5)
    np.testing.assert_allclose(dispatch.dispatched_curtailed_mw, [7.0, 8.0], atol=1e-5)
    np.testing.assert_allclose(power_flow.dispatched_generation_mw + power_flow.curtailed_generation_mw, available, atol=1e-5)
    assert power_flow.summary_dict()["total_curtailed_generation_mwh"] == pytest.approx(15.0)


@pytest.mark.parametrize("old_multiplier,new_multiplier", [(2.0, 2.0), (0.25, 1.5)])
def test_routed_equivalent_recovers_single_circuit_before_scaling(old_multiplier, new_multiplier):
    topology, electrical = _network()
    original = electrical.branch_params[0]
    old = _resize_branch_multiplier(original, old_multiplier)
    grid = WorldGridConfig(height=3, width=3, cell_size_km=1.0)
    bus_by_id = {bus.bus_id: bus for bus in topology.refined_buses}
    _, routed = _equivalent_path_segment(8, 10, 20, (old,), bus_by_id, np.ones((3, 3)), grid, new_multiplier)
    expected = _resize_branch_multiplier(original, new_multiplier)
    np.testing.assert_allclose([routed.r_ohm, routed.x_ohm, routed.b_us, routed.rate_mva], [expected.r_ohm, expected.x_ohm, expected.b_us, expected.rate_mva], rtol=1e-6)
    # Rebuilding the route again at the same rating must not halve impedance.
    _, rerouted = _equivalent_path_segment(9, 10, 20, (routed,), bus_by_id, np.ones((3, 3)), grid, new_multiplier)
    np.testing.assert_allclose([rerouted.r_ohm, rerouted.x_ohm, rerouted.b_us], [routed.r_ohm, routed.x_ohm, routed.b_us], rtol=1e-6)


def test_merged_contributors_with_different_existing_multipliers_share_template():
    topology, electrical = _network()
    original = electrical.branch_params[0]
    contributors = (_resize_branch_multiplier(original, 0.5), replace(_resize_branch_multiplier(original, 2.0), edge_id=8))
    _, merged = _equivalent_path_segment(9, 10, 20, contributors, {bus.bus_id: bus for bus in topology.refined_buses}, np.ones((3, 3)), WorldGridConfig(height=3, width=3, cell_size_km=1.0), 2.5)
    expected = _resize_branch_multiplier(original, 2.5)
    np.testing.assert_allclose([merged.r_ohm, merged.x_ohm, merged.b_us], [expected.r_ohm, expected.x_ohm, expected.b_us], rtol=1e-6)


def test_bypass_parallel_equivalent_scales_impedance_and_shunt_from_source_rating():
    topology, electrical = _network()
    original = electrical.branch_params[0]
    source = _resize_branch_multiplier(original, 0.5)
    action = {
        "nominal_kv": source.nominal_kv,
        "source_r_ohm_per_km": source.r_ohm / source.length_km,
        "source_x_ohm_per_km": source.x_ohm / source.length_km,
        "source_b_us_per_km": source.b_us / source.length_km,
        "source_rate_mva": source.rate_mva,
        "bypass_rate_mva": 3.0 * source.rate_mva,
    }
    bypass = _new_branch_from_template(
        source, action, 8, 10, 20, {bus.bus_id: bus for bus in topology.refined_buses},
        WorldGridConfig(height=3, width=3, cell_size_km=1.0), length_km_override=4.0,
    )
    assert bypass is not None
    expected = _resize_branch_multiplier(original, 1.5)
    length_scale = 4.0 / original.length_km
    np.testing.assert_allclose(
        [bypass.r_ohm, bypass.x_ohm, bypass.b_us],
        np.array([expected.r_ohm, expected.x_ohm, expected.b_us]) * length_scale, rtol=1e-6,
    )
    assert bypass.rate_mva == pytest.approx(expected.rate_mva)


def test_connected_congested_feeder_sheds_only_its_local_load_and_preserves_kcl():
    topology, _ = _network()
    third_bus = GridBus(30, "load_bus", 2, 0, 0.0, 1.0, 10.0, 1.0, 0.0, "test", 2)
    third_edge = GridEdge(8, 10, 30, 2.0, 2.0, False, (0, 1, 2), (0, 0, 0))
    topology = replace(topology, refined_buses=topology.refined_buses + (third_bus,), refined_edges=topology.refined_edges + (third_edge,))
    electrical = build_grid_electrical(topology)
    config = replace(StorageConfig(), max_line_expansion_fraction=0.0)
    electrical = replace(electrical, branch_params=(replace(electrical.branch_params[0], rate_mva=1.0 / config.line_operating_limit_ratio), electrical.branch_params[1]))
    load = np.array([[0.0, 4.0, 4.0]], dtype=np.float32)
    available = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    forecast = SourceLoadForecastStore(np.array([0]), np.array([10, 20, 30]), ("wind_bus", "load_bus", "load_bus"), load, available, available.copy(), np.zeros_like(load), ())
    baseline = solve_dc_power_flow(forecast, topology, electrical)
    plan = StoragePlanStore(np.array([0]), np.array([20, 30]), np.array([-1, -1]), np.empty((1, 0)), ())
    _, requested, flow, _ = dispatch_storage_week(topology, electrical, baseline, plan, config)
    np.testing.assert_allclose(flow.unserved_load_mw, [[0.0, 3.0, 0.0]], atol=1e-5)
    np.testing.assert_allclose(flow.served_load_mw, [[0.0, 1.0, 4.0]], atol=1e-5)
    np.testing.assert_allclose(flow.line_flow_mw, [[1.0, 4.0]], atol=1e-5)
    np.testing.assert_allclose(requested.p_load_mw, load)
    np.testing.assert_allclose(flow.served_load_mw + flow.unserved_load_mw, requested.p_load_mw, atol=1e-5)
    np.testing.assert_allclose(flow.bus_p_injection_mw, [[5.0, -1.0, -4.0]], atol=1e-5)
    np.testing.assert_allclose(flow.dispatched_generation_mw + flow.curtailed_generation_mw, available, atol=1e-5)
