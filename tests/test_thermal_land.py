"""Analytic area/retreat counterexamples for the Stage9 thermal land contract."""
from dataclasses import replace

import numpy as np
import pytest

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import (
    EnergyCandidate, EnergyCandidateState, GridBus, HydrologyState, LandUseState,
    StaticLandState, TerrainFeatures,
)
from world_generator.grid import node_builder
from world_generator.grid.thermal_land import allocate_thermal_land, thermal_land_eligibility


def _bus(identifier, row, col, capacity=1000.0):
    return GridBus(identifier, "thermal_bus", row, col, 0.0, 0.0, capacity, 1.0, 0.0, "thermal", identifier)


def _config(**kwargs):
    return replace(PowerGridConfig(), thermal_project_radius_km=10.0,
                   thermal_capacity_density_mw_km2=200.0, **kwargs)


def _states(shape=(1, 4)):
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    false = np.zeros(shape, dtype=bool)
    ids = np.full(shape, -1, dtype=np.int16)
    terrain = TerrainFeatures(*(zeros.copy() for _ in range(6)))
    water = HydrologyState(ids, ones, false, false, false, zeros, zeros, ids, ones * 10, zeros)
    land = StaticLandState(ones.astype(np.int16), ones, false, ones, zeros, false,
                           landform=ones.astype(np.int16), land_cover_type=np.full(shape, 3, dtype=np.int16),
                           allocatable_land_fraction=ones.copy())
    land_use = LandUseState(zeros, zeros, zeros, zeros, zeros, zeros, ids)
    energy = EnergyCandidateState(ones, ones, ones, ids, ids, ids, ids, (), (), ())
    return terrain, water, land, land_use, energy


def test_remaining_area_is_km2_and_wind_pv_budget_is_not_reused():
    # One 4 km2 cell: Stage8 reserved 3 km2; 1 km2 remains for thermal.
    grid = WorldGridConfig(1, 1, 2.0)
    remaining = np.array([[1.0]])
    buses, ledger, cube, maps = allocate_thermal_land([_bus(17, 0, 0)], remaining, np.ones((1, 1), bool), grid, _config())
    assert buses[0].bus_id == 17
    assert buses[0].capacity_mw == pytest.approx(200.0)
    np.testing.assert_allclose(ledger[0], [17, 0, 0, 1, 200, 200, 1, 200, 1000])
    assert 3.0 + cube.sum() == pytest.approx(4.0)
    np.testing.assert_array_equal(remaining, [[1.0]])
    np.testing.assert_array_equal(maps["energy_unallocated_after_thermal_area_km2"], [[0.0]])


def test_two_overlapping_thermal_envelopes_share_each_cell_once():
    grid = WorldGridConfig(1, 3, 1.0)
    remaining = np.full((1, 3), .5)
    buses, ledger, cube, maps = allocate_thermal_land(
        [_bus(11, 0, 0), _bus(23, 0, 2)], remaining, np.ones((1, 3), bool), grid, _config(),
    )
    np.testing.assert_allclose(cube, [[[.5, .25, 0]], [[0, .25, .5]]])
    np.testing.assert_allclose(ledger[:, 3], [.75, .75])
    np.testing.assert_allclose([b.capacity_mw for b in buses], [150, 150])
    np.testing.assert_allclose(maps["thermal_allocated_area_km2"], remaining)
    np.testing.assert_allclose(cube.sum(axis=0), remaining)


def test_design_nameplate_keeps_reserved_land_headroom_without_forced_generation():
    buses, ledger, _, _ = allocate_thermal_land(
        [_bus(0, 0, 0, 20)], np.ones((1, 1)), np.ones((1, 1), bool), WorldGridConfig(1, 1, 1), _config(),
    )
    assert buses[0].capacity_mw == 20
    assert ledger[0, 6] == pytest.approx(.1)
    assert ledger[0, 7] == 200


@pytest.mark.parametrize("shape,spacing", [((1, 1), 2.0), ((2, 2), 1.0), ((4, 4), .5)])
def test_same_uniform_physical_area_has_same_capacity_at_each_resolution(shape, spacing):
    buses, ledger, _, _ = allocate_thermal_land(
        [_bus(4, 0, 0)], np.full(shape, .25 * spacing**2), np.ones(shape, bool),
        WorldGridConfig(*shape, spacing), _config(),
    )
    assert ledger[0, 3] == pytest.approx(1.0)
    assert buses[0].capacity_mw == pytest.approx(200.0)


def test_water_protected_wetland_and_nonallocatable_cells_never_receive_area():
    _, water, land, use, energy = _states((1, 5))
    water = replace(water, river=np.array([[True, False, False, False, False]]))
    # The complete C land axis excludes wetland (col2) and hard slopes (col3).
    land = replace(land, protected=np.array([[False, True, False, False, False]]),
                   allocatable_land_fraction=np.array([[1., 1., 0., 0., 1.]]))
    grid = WorldGridConfig(1, 5, 1)
    eligible = thermal_land_eligibility(land, water, use, energy, grid, _config())
    np.testing.assert_array_equal(eligible, [[False, False, False, False, True]])
    _, _, cube, maps = allocate_thermal_land([_bus(0, 0, 4)], np.full((1, 5), .5), eligible, grid, _config())
    np.testing.assert_array_equal(cube.sum(axis=0), [[0, 0, 0, 0, .5]])
    np.testing.assert_array_equal(maps["energy_unallocated_after_thermal_area_km2"], [[.5, .5, .5, .5, 0]])


def test_residential_and_renewable_setbacks_apply_to_entire_envelope():
    _, water, land, use, energy = _states((1, 7))
    use = replace(use, residential=np.array([[1., 0, 0, 0, 0, 0, 0]]))
    candidate = EnergyCandidate(0, "wind", 0, 6, 1., 0., 100., 1.)
    energy = replace(energy, wind_candidates=(candidate,))
    config = _config(thermal_residential_buffer_km=2., thermal_min_renewable_distance_km=2.)
    grid = WorldGridConfig(1, 7, 1)
    eligible = thermal_land_eligibility(land, water, use, energy, grid, config)
    np.testing.assert_array_equal(eligible, [[False, False, True, True, True, False, False]])
    _, _, cube, _ = allocate_thermal_land([_bus(2, 0, 3)], np.full((1, 7), .5), eligible, grid, config)
    np.testing.assert_array_equal(cube.sum(axis=0), [[0, 0, .5, .5, .5, 0, 0]])


def test_no_candidates_or_no_remaining_area_preserves_explicit_empty_ledger():
    grid = WorldGridConfig(1, 1, 1)
    for buses, remaining in [([], np.array([[.5]])), ([_bus(0, 0, 0)], np.zeros((1, 1)))]:
        result, ledger, cube, maps = allocate_thermal_land(buses, remaining, np.ones((1, 1), bool), grid, _config())
        assert not result
        assert ledger.shape == (0, 9)
        assert cube.shape == (0, 1, 1)
        np.testing.assert_array_equal(maps["thermal_allocated_area_km2"], [[0.]])
        np.testing.assert_array_equal(maps["energy_unallocated_after_thermal_area_km2"], remaining)


@pytest.mark.parametrize("remaining", [np.array([[np.nan]]), np.array([[-.1]]), np.array([[1.1]]), np.ones((2, 1))])
def test_remaining_area_rejects_nonfinite_negative_overfull_or_wrong_shape(remaining):
    with pytest.raises(ValueError, match="area"):
        allocate_thermal_land([], remaining, np.ones((1, 1), bool), WorldGridConfig(1, 1, 1), _config())


def test_node_builder_clips_after_adequacy_and_exports_contract(monkeypatch):
    terrain, water, land, use, energy = _states((1, 1))
    energy = replace(energy, land_accounting_maps={"energy_unallocated_area_km2": np.array([[.1]])},
                     load_candidates=(EnergyCandidate(55, "load", 0, 0, 0., 0., 1000., 1.),))
    # Initial suitability asks for 98 MW; adequacy raises to 260 MW, then land
    # permits only 20 MW. A later adequacy pass must not undo the land cap.
    monkeypatch.setattr(node_builder, "_thermal_suitability", lambda *args: (np.full((1, 1), .01), np.zeros((1, 1))))
    config = _config(thermal_scale_with_load=False, thermal_candidate_count=1, thermal_relax_iterations=0)
    result = node_builder.build_grid_nodes(terrain, water, land, use, energy, WorldGridConfig(1, 1, 1), config)
    assert next(bus for bus in result.buses if bus.kind == "thermal_bus").capacity_mw == pytest.approx(20.)
    assert result.thermal_land_accounting_mode == "exclusive_project_envelopes"
    arrays = result.buses_as_arrays()
    assert arrays["thermal_land_ledger"].shape == (1, 9)
    assert arrays["thermal_project_area_by_cell_km2"].shape == (1, 1, 1)
    assert arrays["thermal_land_columns"][7] == "land_capacity_upper_bound_mw"
    assert arrays["thermal_land_ledger"][0, 8] == 260.
    assert result.as_maps()["thermal_land_eligible"].dtype == np.bool_


def test_node_builder_legacy_is_explicit_and_partial_accounting_is_rejected():
    terrain, water, land, use, energy = _states((1, 1))
    args = (terrain, water, land, use)
    grid = WorldGridConfig(1, 1, 1)
    config = _config(thermal_candidate_count=0, thermal_scale_with_load=False)
    legacy = node_builder.build_grid_nodes(*args, energy, grid, config)
    assert legacy.thermal_land_accounting_mode == "legacy_no_land_guarantee"
    assert "thermal_land_ledger" not in legacy.buses_as_arrays()
    assert str(legacy.buses_as_arrays()["thermal_land_accounting_mode"]) == "legacy_no_land_guarantee"
    partial = replace(energy, land_accounting_maps={"energy_available_area_km2": np.ones((1, 1))})
    with pytest.raises(ValueError, match="missing energy_unallocated"):
        node_builder.build_grid_nodes(*args, partial, grid, config)
    empty = replace(energy, land_accounting_maps={"energy_unallocated_area_km2": np.ones((1, 1))})
    result = node_builder.build_grid_nodes(*args, empty, grid, config)
    assert result.buses_as_arrays()["thermal_land_ledger"].shape == (0, 9)
    np.testing.assert_array_equal(result.as_maps()["energy_unallocated_after_thermal_area_km2"], [[1.]])


def test_configurable_score_threshold_and_quadrature_are_used():
    _, water, land, use, energy = _states((1, 1))
    use = replace(use, residential=np.array([[.3]]))
    grid = WorldGridConfig(1, 1, 1)
    low = thermal_land_eligibility(land, water, use, energy, grid, _config(thermal_residential_score_threshold=.2))
    high = thermal_land_eligibility(land, water, use, energy, grid, _config(thermal_residential_score_threshold=.5))
    assert not low.any() and high.all()
    for count in (1, 3, 8):
        _, ledger, _, maps = allocate_thermal_land(
            [_bus(0, 0, 0)], np.array([[.1]]), high, grid,
            _config(thermal_project_area_subcells_per_axis=count),
        )
        assert ledger[0, 3] == pytest.approx(.1)
        assert maps["energy_unallocated_after_thermal_area_km2"].min() >= 0
