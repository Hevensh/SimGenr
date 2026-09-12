from dataclasses import replace

import numpy as np
import pytest

from world_generator.city.city_generator import _effective_developable_area_km2, generate_initial_cities
from world_generator.core.config import CityConfig, EnergyConfig, LandConfig, LandUseConfig, PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import CityState, ClimateBaseline, EnergyCandidate, EnergyCandidateState, GridNodeState, HydrologyState, TerrainFeatures
from world_generator.core.output_layout import WorldDataLayout
from world_generator.dataset.builder import _land_accounting_payload
from world_generator.energy.energy_candidate_generator import _allocate_source_capacity, generate_energy_candidates
from world_generator.land.land_generator import generate_static_land
from world_generator.land_use.land_use_generator import generate_land_use_zones


def _fixture(shape=(3, 3), cell_size=1.0):
    z = np.zeros(shape, dtype=np.float32)
    terrain = TerrainFeatures(*(z.copy() for _ in range(6)))
    hydrology = HydrologyState(z.astype(int), z.copy(), z.astype(bool), z.astype(bool), z.astype(bool), z.copy(), z.copy(), z.astype(int), z + 10, z.copy())
    climate = ClimateBaseline(z + 18, z + 10, z + .6, z + 6, z.copy(), z + 800, z + .4, z + 180)
    grid = WorldGridConfig(height=shape[0], width=shape[1], cell_size_km=cell_size)
    land = generate_static_land(terrain, hydrology, grid, LandConfig(protected_fraction=0), np.random.default_rng(1), climate)
    city = CityState((), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.astype(bool), z.astype(int) - 1)
    return terrain, hydrology, climate, grid, land, city


def _project(kind="wind", row=0, col=0):
    return EnergyCandidate(0, kind, row, col, 0.0, 0.0, 1.0, 1.0)


def test_cover_and_landform_do_not_change_when_only_protection_identity_changes():
    terrain, hydrology, climate, grid, land, _ = _fixture()
    protected = generate_static_land(terrain, hydrology, grid, LandConfig(protected_fraction=1), np.random.default_rng(1), climate)
    np.testing.assert_array_equal(land.landform, protected.landform)
    np.testing.assert_array_equal(land.land_cover_type, protected.land_cover_type)
    assert np.all(protected.protected_mask)
    assert np.any(land.land_cover != protected.land_cover)  # legacy display intentionally differs
    for value in protected.as_maps().values():
        assert value.shape == terrain.elevation.shape


def test_fractions_close_and_water_protection_cannot_be_overridden_by_high_scores():
    terrain, hydrology, _, grid, land, city = _fixture((1, 3))
    water = np.array([[True, False, False]])
    protected = np.array([[True, True, False]])  # overlapping masks do not double count
    hydrology = replace(hydrology, river=water)
    land = replace(land, protected=protected, land_cover_type=np.array([[1, 4, 3]]), buildability=np.ones((1, 3)))
    city = replace(city, population_density=np.array([[0.0, 0.0, 1000.0]]), economic_activity=np.ones((1, 3)), urban_density=np.ones((1, 3)))
    uses = generate_land_use_zones(terrain, hydrology, land, city, grid, LandUseConfig())
    np.testing.assert_allclose(sum(uses.use_fractions.values()), 1, atol=2e-7)
    np.testing.assert_array_equal(uses.use_fractions["water"], [[1, 0, 0]])
    for name in ("residential", "commercial", "industrial", "agriculture", "energy_reserve"):
        np.testing.assert_array_equal(uses.use_fractions[name][0, :2], [0, 0])
    for value in uses.as_maps().values():
        assert value.shape == (1, 3)


def test_partial_eligible_area_caps_whole_cell_built_demand_without_rescaling_it():
    terrain, hydrology, _, grid, land, city = _fixture((1, 1))
    land = replace(land, allocatable_land_fraction=np.array([[.25]], dtype=np.float32))
    city = replace(city, population_density=np.array([[600.0]], dtype=np.float32))
    uses = generate_land_use_zones(terrain, hydrology, land, city, grid, LandUseConfig())
    built = sum(uses.use_fractions[name] for name in ("residential", "commercial", "industrial"))
    np.testing.assert_allclose(built, .1, atol=1e-7)  # 600 persons / 6000 persons per built km2
    np.testing.assert_allclose(sum(uses.use_fractions.values()), 1, atol=2e-7)
    assert sum(uses.use_fractions[name][0, 0] for name in ("residential", "commercial", "industrial", "agriculture", "energy_reserve")) <= .25 + 1e-7


def test_zero_scores_have_explicit_residual_area_and_finite_closed_fractions():
    terrain, hydrology, _, grid, land, city = _fixture((1, 1))
    land = replace(land, vegetation=np.zeros((1, 1)), buildability=np.zeros((1, 1)))
    config = LandUseConfig(agriculture_fraction_of_remaining=0, park_fraction_of_remaining=0, energy_reserve_fraction_of_remaining=0)
    uses = generate_land_use_zones(terrain, hydrology, land, city, grid, config)
    np.testing.assert_allclose(uses.use_fractions["natural"], 1)
    assert all(np.isfinite(value).all() for value in uses.use_fractions.values())


def test_city_scaling_area_uses_hard_geometry_not_buildability_score():
    _, hydrology, _, grid, land, _ = _fixture((2, 2), cell_size=2)
    low = replace(land, buildability=np.full((2, 2), .01))
    high = replace(land, buildability=np.ones((2, 2)))
    assert _effective_developable_area_km2(low, hydrology, grid) == pytest.approx(16)
    assert _effective_developable_area_km2(high, hydrology, grid) == pytest.approx(16)


def test_city_cannot_silently_drop_positive_population_and_hard_support_is_respected():
    terrain, hydrology, climate, grid, land, _ = _fixture()
    config = CityConfig(city_count=1, total_population=1000)
    blocked = replace(land, protected=np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="positive target population"):
        generate_initial_cities(terrain, hydrology, blocked, climate, grid, config, np.random.default_rng(1))
    with pytest.raises(ValueError, match="at least one city"):
        generate_initial_cities(terrain, hydrology, land, climate, grid, replace(config, city_count=0), np.random.default_rng(1))
    area = np.zeros((3, 3), dtype=np.float32)
    area[1, 1] = 1
    city = generate_initial_cities(terrain, hydrology, replace(land, allocatable_land_fraction=area), climate, grid, config, np.random.default_rng(1))
    np.testing.assert_allclose(city.population_density.sum() * grid.cell_size_km**2, 1000, rtol=2e-6)
    assert np.all(city.population_density[area == 0] == 0)
    assert city.population_budget["target_population_persons"] == 1000


@pytest.mark.parametrize("cover_code", [1, 2])
def test_explicit_water_or_wetland_cover_cannot_place_population_even_with_available_area(cover_code):
    terrain, hydrology, climate, grid, land, _ = _fixture()
    land = replace(land, land_cover_type=np.full((3, 3), cover_code), allocatable_land_fraction=np.ones((3, 3)))
    with pytest.raises(ValueError, match="positive target population"):
        generate_initial_cities(terrain, hydrology, land, climate, grid, CityConfig(city_count=1, total_population=100), np.random.default_rng(1))


def test_project_with_excluded_centre_cannot_acquire_neighbouring_land():
    with pytest.raises(ValueError, match="centre"):
        _allocate_source_capacity([_project()], [], np.full((1, 2), .5), WorldGridConfig(height=1, width=2), EnergyConfig(wind_cluster_radius_km=10), wind_eligible=np.array([[False, True]]))


@pytest.mark.parametrize("shape,spacing", [((1, 1), 2.0), ((2, 2), 1.0)])
def test_one_square_km_reserve_is_not_double_allocated_between_wind_and_pv(shape, spacing):
    config = EnergyConfig(wind_cluster_radius_km=10, pv_cluster_radius_km=10)
    wind, pv, ledger, cube, maps = _allocate_source_capacity([_project()], [_project("pv")], np.full(shape, .25), WorldGridConfig(height=shape[0], width=shape[1], cell_size_km=spacing), config, return_accounting=True)
    assert sum(item.capacity_mw / (config.wind_capacity_density_mw_km2 if item.kind == "wind" else config.pv_capacity_density_mw_km2) for item in wind + pv) <= 1 + 1e-10
    assert ledger[:, 5].sum() == pytest.approx(1)
    np.testing.assert_allclose(cube.sum(axis=(1, 2)), ledger[:, 5])
    np.testing.assert_allclose(maps["energy_wind_project_area_km2"] + maps["energy_pv_project_area_km2"] + maps["energy_unallocated_area_km2"], maps["energy_available_area_km2"])


def test_entire_project_envelope_obeys_technology_hard_eligibility_and_design_cap():
    eligible = np.array([[False, False, True, True]])
    config = EnergyConfig(wind_cluster_radius_km=3, wind_capacity_max_mw=.1, wind_cluster_capacity_max_multiplier=1)
    _, _, ledger, cube, _ = _allocate_source_capacity([_project(row=0, col=2)], [], np.ones((1, 4)), WorldGridConfig(height=1, width=4, cell_size_km=1), config, wind_eligible=eligible, return_accounting=True)
    assert np.all(cube[:, :, :2] == 0)
    assert ledger[0, 6] == pytest.approx(.1)
    assert ledger[0, 8] < ledger[0, 5]  # reserved envelope is distinct from P/density


def test_no_projects_preserves_every_square_km_in_unused_budget():
    _, _, ledger, cube, maps = _allocate_source_capacity([], [], np.full((2, 2), .25), WorldGridConfig(height=2, width=2, cell_size_km=2), EnergyConfig(), return_accounting=True)
    assert ledger.shape == (0, 9) and cube.shape == (0, 2, 2)
    np.testing.assert_allclose(maps["energy_unallocated_area_km2"], maps["energy_available_area_km2"])


@pytest.mark.parametrize("case", ["shape", "fractional_coordinate", "outside", "duplicate_id", "wrong_technology"])
def test_project_allocation_rejects_invalid_grid_and_candidate_identity(case):
    grid, available, projects = WorldGridConfig(height=2, width=2), np.ones((2, 2)), [_project()]
    if case == "shape":
        available = np.ones((1, 2))
    elif case == "fractional_coordinate":
        projects = [replace(projects[0], row=.5)]
    elif case == "outside":
        projects = [replace(projects[0], col=2)]
    elif case == "duplicate_id":
        projects.append(replace(projects[0], col=1))
    else:
        projects = [_project("pv")]
    with pytest.raises(ValueError):
        _allocate_source_capacity(projects, [], available, grid, EnergyConfig())


@pytest.mark.parametrize("state_kind", ["renewable", "thermal"])
@pytest.mark.parametrize("case", ["missing_cube", "missing_ledger", "wrong_columns", "wrong_grid", "nonfinite", "negative_area"])
def test_project_state_rejects_incomplete_or_invalid_serialization(state_kind, case):
    z = np.zeros((2, 2))
    state = (EnergyCandidateState(*(z for _ in range(7)), (), (), ()) if state_kind == "renewable"
             else GridNodeState(*(z for _ in range(6)), ()))
    ledger, cube = np.zeros((1, 9)), np.zeros((1, 2, 2))
    if case == "missing_cube": cube = None
    elif case == "missing_ledger": ledger = None
    elif case == "wrong_columns": ledger = np.zeros((1, 8))
    elif case == "wrong_grid": cube = np.zeros((1, 3, 2))
    elif case == "nonfinite": ledger[0, 3] = np.nan
    else: cube[0, 0, 0] = -1
    kwargs = ({"project_land_ledger": ledger, "project_area_by_cell_km2": cube} if state_kind == "renewable"
              else {"thermal_project_land_ledger": ledger, "thermal_project_area_by_cell_km2": cube})
    with pytest.raises(ValueError):
        replace(state, **kwargs)


@pytest.mark.parametrize("config_class,kwargs", [(LandConfig, {"allocatable_max_slope": -1}), (LandConfig, {"woodland_vegetation_threshold": 2}), (LandUseConfig, {"energy_reserve_fraction_of_remaining": 1.01}), (LandUseConfig, {"built_population_density_persons_km2": 0}), (EnergyConfig, {"project_area_subcells_per_axis": 0}), (EnergyConfig, {"pv_max_project_slope": np.nan}), (PowerGridConfig, {"thermal_project_area_subcells_per_axis": 1.5}), (PowerGridConfig, {"thermal_residential_score_threshold": 1.1})])
def test_new_area_priors_reject_invalid_configuration(config_class, kwargs):
    with pytest.raises(ValueError):
        config_class(**kwargs)


def test_dataset_appendix_preserves_independent_axes_and_project_ledgers(tmp_path):
    terrain, hydrology, climate, grid, land, city = _fixture()
    uses = generate_land_use_zones(terrain, hydrology, land, city, grid, LandUseConfig())
    energy = generate_energy_candidates(terrain, hydrology, land, city, uses, climate.as_maps(), grid, EnergyConfig(wind_candidate_count=1, pv_candidate_count=1, load_node_count=0))
    static = terrain.as_maps() | land.as_maps() | uses.as_maps() | energy.as_maps()
    static.update(thermal_land_eligible=np.ones((3, 3), dtype=bool), thermal_allocated_area_km2=np.zeros((3, 3)), energy_unallocated_after_thermal_area_km2=static["energy_unallocated_area_km2"].copy())
    layout = WorldDataLayout(tmp_path)
    layout.energy.mkdir(parents=True)
    layout.buses.mkdir(parents=True)
    np.savez(layout.energy / "source_load_candidates.npz", **energy.candidates_as_arrays())
    np.savez(layout.buses / "grid_nodes.npz", thermal_land_ledger=np.empty((0, 9)), thermal_land_columns=np.asarray([str(index) for index in range(9)]), thermal_project_area_by_cell_km2=np.empty((0, 3, 3)), thermal_land_accounting_mode=np.asarray("exclusive_project_envelopes"))
    packed = _land_accounting_payload(layout, static, {"land_accounting_version": "land_use_v1"})
    np.testing.assert_array_equal(packed["land_cover_type"], land.land_cover_type)
    np.testing.assert_array_equal(packed["protected_mask"], land.protected_mask)
    assert packed["wind_candidates"].shape[1] == 7
    np.testing.assert_allclose(packed["energy_project_area_by_cell_km2"].sum(axis=(1, 2)), packed["energy_project_land_ledger"][:, 5])
    invalid = dict(static)
    del invalid["land_use_fraction_water"]
    with pytest.raises(ValueError, match="Incomplete modern land"):
        _land_accounting_payload(layout, invalid, {"land_accounting_version": "land_use_v1"})
