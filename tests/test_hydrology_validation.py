"""D analytic export counterexamples, independent of the water generator."""
from dataclasses import replace

import numpy as np
import pytest

from scripts.hydrology_validation import check_dynamic_hydrology
from scripts.validate_world_physics import Checks
from world_generator.core.config import HydrologyDynamicConfig, WorldConfig, WorldGridConfig
from world_generator.core.datatypes import HydrologyTimeSeriesStore
from world_generator.core.hydrology_contracts import HYDROLOGY_GROUP_UNITS


def _example():
    cfg = HydrologyDynamicConfig(enabled=True, initial_soil_fraction=0, initial_groundwater_fraction=0)
    config = replace(WorldConfig(), world=WorldGridConfig(3, 3, 2), hydrology_dynamic=cfg)
    shape, hours = (3, 3), 2
    sizes = {"state": (hours + 1, *shape), "flux": (hours, *shape), "static": shape, "budget": (hours,)}
    s, f, m, b = [{name: np.zeros(sizes[group]) for name in names} for group, names in HYDROLOGY_GROUP_UNITS.items()]
    m["impervious_fraction"][:] = .02
    m["pervious_fraction"][:] = .98
    m["soil_capacity_mm"][:] = cfg.soil_capacity_mm * .98
    m["groundwater_capacity_mm"][:] = cfg.groundwater_capacity_mm
    m["routing_receiver_flat_index"][:] = -1
    m["routing_receiver_flat_index"][1, 1] = -2
    m["closed_sink_mask"][1, 1] = 1
    # An isolated central bucket receives 1 mm on 4 km2: 4000 m3.
    # 0.98 mm infiltrates; the 80 m3 surface remainder stays in closed storage.
    f["precipitation_mm"][0, 1, 1] = 1
    f["infiltration_mm"][0, 1, 1] = .98
    f["surface_runoff_mm"][0, 1, 1] = .02
    s["soil_storage_mm"][1:, 1, 1] = .98
    s["channel_storage_m3"][1:, 1, 1] = 80
    b["initial_storage_m3"][:] = [0, 4000]
    b["precipitation_m3"][:] = [4000, 0]
    b["final_storage_m3"][:] = 4000
    stamps = np.arange(hours)
    bounds = np.column_stack((stamps, stamps + 1))
    store = HydrologyTimeSeriesStore(stamps, bounds, np.arange(hours + 1), s, f, m, b)
    static = {"river": np.zeros(shape, bool), "lake": np.zeros(shape, bool), "flow_direction": np.full(shape, -1),
              "hydrology_elevation": np.zeros(shape), "water_depth": np.zeros(shape)}
    static.update({"land_use_fraction_" + name: np.full(shape, float(name == "natural")) for name in cfg.impervious_fraction_by_use})
    hourly = {"timestamps": stamps.copy(), "time_bounds_hours": bounds.copy(), "channel_names": np.array(["precipitation", "irradiance"]),
              "dynamic": np.stack([f["precipitation_mm"].copy(), np.zeros((hours, *shape))], axis=1)}
    return store.as_arrays(), hourly, static, config


def _run(args):
    checks = Checks()
    summary = check_dynamic_hydrology(checks, *args)
    return {row["name"]: row for row in checks.rows}, summary


def test_analytic_closed_bucket_conserves_local_rain_volume():
    rows, summary = _run(_example())
    assert all(row["status"] != "FAIL" for row in rows.values()), [r for r in rows.values() if r["status"] == "FAIL"]
    assert rows["hydrology_cell_water_balance"]["status"] == "PASS"
    assert rows["hydrology_domain_interval_balance"]["status"] == "PASS"
    assert rows["hydrology_lake_bed_static_anchor"]["status"] == "NOT_RUN"
    assert rows["hydrology_lake_spill_level"]["status"] == "NOT_RUN"
    assert summary["precipitation_m3"] == 4000
    assert summary["final_storage_m3"] == 4000
    assert summary["boundary_outflow_m3"] == 0


@pytest.mark.parametrize("case,check", [
    ("hidden_cell_injection", "cell_water_balance"), ("fake_reported_residual", "reported_cell_residual"),
    ("fake_budget", "reported_budget_precipitation_m3"), ("wrong_discharge_unit", "discharge_rate_support"),
    ("closed_sink_outflow", "boundary_outflow_external_only"), ("wrong_routing_receiver", "routing_incidence"),
    ("soil_gain_without_rain", "dry_soil_cannot_gain"), ("forcing_changed", "precipitation_forcing"),
])
def test_corrupt_export_identifies_violated_water_relationship(case, check):
    args = _example()
    arrays = args[0]
    if case == "hidden_cell_injection": arrays["state__channel_storage_m3"][1:, 1, 1] += 1
    elif case == "fake_reported_residual": arrays["flux__cell_budget_residual_m3"][0, 1, 1] = 4
    elif case == "fake_budget": arrays["budget__precipitation_m3"][0] = 4001
    elif case == "wrong_discharge_unit": arrays["flux__discharge_m3_s"][0, 1, 1] = 3600
    elif case == "closed_sink_outflow": arrays["flux__boundary_outflow_m3"][0, 1, 1] = 1
    elif case == "wrong_routing_receiver":
        arrays["static__routing_receiver_flat_index"][0, 0] = 1
        arrays["flux__routing_outflow_m3"][0, 0, 0] = 1
        arrays["flux__routing_inflow_m3"][0, 0, 2] = 1
    elif case == "soil_gain_without_rain": arrays["state__soil_storage_mm"][2, 1, 1] += .01
    elif case == "forcing_changed": arrays["flux__precipitation_mm"][0, 1, 1] += .1
    rows, _ = _run(args)
    assert not rows["hydrology_" + check]["passed"]


def test_large_other_cell_inventory_cannot_hide_a_local_water_injection():
    args = _example()
    args[0]["state__channel_storage_m3"][:, 0, 0] = 1e14
    args[0]["state__channel_storage_m3"][1:, 1, 1] += 1
    rows, _ = _run(args)
    assert not rows["hydrology_cell_water_balance"]["passed"]


def test_missing_terminal_state_is_a_contract_error():
    args = _example()
    args[0]["state__soil_storage_mm"] = args[0]["state__soil_storage_mm"][:-1]
    with pytest.raises(ValueError, match="finite"):
        _run(args)


def test_relative_water_tolerance_is_elementwise():
    checks = Checks()
    checks.equal("cell", np.array([1, 1e-6]), 1e-5, relative_tolerance=1e-10, scale=np.array([1, 1e14]))
    assert not checks.rows[0]["passed"]


def _lake_example():
    arrays, hourly, static, config = _example()
    config = replace(config, hydrology_dynamic=replace(config.hydrology_dynamic, initial_lake_storage_fraction=1))
    selected = np.zeros((3, 3), bool)
    selected[0, :2] = True
    static["lake"] = selected
    static["land_use_fraction_water"][selected] = 1
    static["land_use_fraction_natural"][selected] = 0
    for key in ("impervious_fraction", "pervious_fraction", "soil_capacity_mm", "groundwater_capacity_mm"):
        arrays["static__" + key][selected] = 0
    arrays["static__lake_id"][selected] = 1
    arrays["static__lake_bed_elevation_m"][selected] = [0, 1]
    arrays["static__lake_spill_elevation_m"][selected] = 2
    static["hydrology_elevation"][selected] = [0, 1]
    static["water_depth"][selected] = [2, 1]
    arrays["static__lake_capacity_m3"][selected] = [8e6, 4e6]
    arrays["state__lake_storage_m3"][:, selected] = [8e6, 4e6]
    arrays["state__lake_water_level_m"][:, selected] = 2
    arrays["state__lake_wetted_area_m2"][:, selected] = 4e6
    for key in ("initial_storage_m3", "final_storage_m3"):
        arrays["budget__" + key] += 12e6
    return arrays, hourly, static, config


def test_lake_common_level_and_geometry_are_checked_separately_from_mass():
    arrays, hourly, static, config = _lake_example()
    rows, _ = _run((arrays, hourly, static, config))
    assert all(row["passed"] for row in rows.values())
    arrays["state__lake_water_level_m"][1, 0, 0] -= .1
    rows, _ = _run((arrays, hourly, static, config))
    assert rows["hydrology_cell_water_balance"]["passed"]
    assert not rows["hydrology_lake_volume_geometry"]["passed"]
    assert not rows["hydrology_lake_1.0_common_level"]["passed"]


@pytest.mark.parametrize("corruption,expected", [("overflow", "hydrology_lake_1.0_pool_balance"),
    ("datum", "hydrology_lake_bed_static_anchor"), ("split", "hydrology_lake_1.0_connected_identity")])
def test_lake_internal_diagnostic_and_static_anchor_cannot_fake_a_pass(corruption, expected):
    args = _lake_example()
    arrays = args[0]
    if corruption == "overflow":
        arrays["flux__lake_overflow_m3"][0, 0, 0] = 1e12
    elif corruption == "datum":
        for name in ("static__lake_bed_elevation_m", "static__lake_spill_elevation_m", "state__lake_water_level_m"):
            arrays[name][..., 0, :2] += 100
    else:
        arrays["static__lake_id"][0, 1] = 2
    rows, _ = _run(args)
    assert not rows[expected]["passed"]
