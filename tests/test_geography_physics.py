"""Conservation, datum, resolution and exclusion checks for synthetic geography."""
from dataclasses import replace

import numpy as np
import pytest

from world_generator.city.city_generator import _select_city_centers, _spread_city_fields
from world_generator.core.config import CityConfig, EnergyConfig, HydrologyConfig, LandConfig, WorldGridConfig
from world_generator.core.datatypes import CityNode, CityState, ClimateBaseline, EnergyCandidate, LandUseState, TerrainBase
from world_generator.energy.energy_candidate_generator import (
    _allocate_load_capacity, _allocate_source_capacity, _best_relaxed_cell,
    _best_city_load_cell, generate_energy_candidates,
)
from world_generator.hydrology.hydrology_generator import (
    D8_OFFSETS, _carve_hydrology_elevation, _compute_flow_accumulation,
    _height_above_drainage, _expand_rivers_by_catchment, generate_hydrology,
)
from world_generator.land.land_generator import generate_static_land
from world_generator.terrain.derivatives import derive_terrain_features


def _flat_world(size=11, cell_size=1.0):
    grid = WorldGridConfig(height=size, width=size, cell_size_km=cell_size)
    terrain = derive_terrain_features(TerrainBase(np.full((size, size), 5000.0), np.ones((size, size), bool)), grid)
    hydrology = generate_hydrology(terrain, grid, replace(HydrologyConfig(), river_min_catchment_km2=1e9))
    return grid, terrain, hydrology


def _climate(shape, temperature=20.0, rain=850.0):
    return ClimateBaseline(*(np.full(shape, value, dtype=np.float32)
                             for value in (temperature, 12.0, 0.6, 6.0, 0.0, rain, 0.4, 180.0)))


def _city(shape, population_density):
    ones = np.ones(shape, np.float32)
    zeros = np.zeros(shape, np.float32)
    return CityState((), ones, ones, zeros, np.asarray(population_density, np.float32),
                     ones, zeros, np.zeros(shape, bool), np.full(shape, -1, np.int16))


def _land_use(shape):
    ones = np.ones(shape, np.float32)
    return LandUseState(ones, ones, ones, ones, ones, ones, np.ones(shape, np.int16))


def _candidate(kind, row, col, index=0):
    return EnergyCandidate(index, kind, row, col, 0.5, 0.5, 100.0, 1.0)


def test_flat_high_elevation_routes_to_boundary_and_conserves_catchment():
    _, terrain, hydrology = _flat_world()
    direction = hydrology.flow_direction
    assert (direction[1:-1, 1:-1] >= 0).all()
    assert hydrology.flow_accumulation[direction < 0].sum() == terrain.elevation.size
    assert not hydrology.lake.any(), "Numerical flat gradients must not create lakes"
    assert (hydrology.flood_risk == 0.0).all(), "No drainage implies no mapped riverine flood hazard"
    for start in np.ndindex(direction.shape):
        visited = set()
        row, col = start
        while direction[row, col] >= 0:
            assert (row, col) not in visited
            visited.add((row, col))
            dr, dc = D8_OFFSETS[direction[row, col]]
            row, col = row + int(dr), col + int(dc)
        assert row in (0, direction.shape[0] - 1) or col in (0, direction.shape[1] - 1)


def test_flow_accumulation_uses_graph_order_and_rejects_cycles():
    # Three equal-height cells flowing west: elevation sorting alone gets this wrong.
    direction = np.asarray([[-1, 6, 6]], dtype=np.int8)
    np.testing.assert_array_equal(_compute_flow_accumulation(np.ones((1, 3)), direction), [[3, 2, 1]])
    with pytest.raises(ValueError, match="cycle"):
        _compute_flow_accumulation(np.ones((1, 2)), np.asarray([[2, 6]], dtype=np.int8))


def test_lake_depth_is_spill_depth_and_preserves_shallow_shores_and_bed():
    grid = WorldGridConfig(height=5, width=5, cell_size_km=1.0)
    elevation = np.full((5, 5), 110.0, np.float32)
    elevation[1:4, 1:4] = 109.0
    elevation[2, 2] = 100.0
    terrain = derive_terrain_features(TerrainBase(elevation, np.ones((5, 5), bool)), grid)
    hydro = generate_hydrology(terrain, grid, replace(HydrologyConfig(), river_min_catchment_km2=1e9))
    assert hydro.lake.sum() == 9
    assert hydro.water_depth[2, 2] == 10.0
    assert hydro.water_depth[1, 1] == 1.0
    np.testing.assert_array_equal(hydro.hydrology_elevation[hydro.lake], elevation[hydro.lake])
    np.testing.assert_allclose(hydro.hydrology_elevation[hydro.lake] + hydro.water_depth[hydro.lake], 110.0)
    np.testing.assert_array_equal(_carve_hydrology_elevation(np.asarray([[1500.0]]), np.asarray([[2.0]])), [[1498.0]])


def test_hand_uses_downstream_channel_and_is_vertical_datum_invariant():
    elevation = np.asarray([[120.0, 115.0, 110.0, 100.0]])
    direction = np.asarray([[2, 2, 2, -1]], dtype=np.int8)
    drainage = np.asarray([[False, False, False, True]])
    expected = [[20.0, 15.0, 10.0, 0.0]]
    np.testing.assert_array_equal(_height_above_drainage(elevation, direction, drainage), expected)
    np.testing.assert_array_equal(_height_above_drainage(elevation + 3000.0, direction, drainage), expected)
    surface = np.asarray([[120.0, 115.0, 110.0, 105.0]])
    np.testing.assert_array_equal(_height_above_drainage(elevation, direction, drainage, surface), [[15.0, 10.0, 5.0, 0.0]])


def test_river_width_is_metric_and_does_not_expand_thirty_metre_stream_to_kilometres():
    center = np.zeros((9, 9), bool)
    center[4, 4] = True
    area = np.full((9, 9), 128.0)
    coarse = _expand_rivers_by_catchment(center, area, 1.0, 4, 128.0, 0.35, 30.0)
    fine = _expand_rivers_by_catchment(center, area, 0.01, 4, 128.0, 0.35, 30.0)
    assert coarse.sum() == 1
    assert fine.sum() == 5


def test_slope_and_roughness_have_local_metric_units():
    for cell in (0.5, 2.0):
        grid = WorldGridConfig(height=7, width=7, cell_size_km=cell)
        elevation = np.broadcast_to(np.arange(7) * cell * 1000 * 0.1, (7, 7)).copy()
        features = derive_terrain_features(TerrainBase(elevation, np.ones((7, 7), bool)), grid)
        np.testing.assert_allclose(features.slope, 0.1, atol=1e-6)
        np.testing.assert_allclose(features.roughness[1:-1, 1:-1], 0.1, atol=1e-6)
    one = derive_terrain_features(TerrainBase(np.ones((1, 1)), np.ones((1, 1), bool)), WorldGridConfig(1, 1, 1.0))
    assert one.slope[0, 0] == 0.0


def test_vegetation_responds_to_climate_and_protected_fraction_can_be_zero():
    grid, terrain, hydro = _flat_world()
    config = replace(LandConfig(), protected_fraction=0.0, vegetation_noise_weight=0.0)
    dry = generate_static_land(terrain, hydro, grid, config, np.random.default_rng(1), _climate(terrain.elevation.shape, rain=20.0))
    wet = generate_static_land(terrain, hydro, grid, config, np.random.default_rng(1), _climate(terrain.elevation.shape, rain=1000.0))
    cold = generate_static_land(terrain, hydro, grid, config, np.random.default_rng(1), _climate(terrain.elevation.shape, temperature=-15.0, rain=1000.0))
    assert not dry.protected.any()
    assert np.all(wet.vegetation > dry.vegetation)
    assert np.all(wet.vegetation > cold.vegetation)
    partial = generate_static_land(terrain, hydro, grid, replace(config, protected_fraction=0.25), np.random.default_rng(1))
    assert partial.protected.sum() == round(terrain.elevation.size * 0.25)


@pytest.mark.parametrize("cell_size", [0.5, 2.0])
def test_population_density_conserves_each_city_before_superposition(cell_size):
    shape = (21, 21)
    grid = WorldGridConfig(*shape, cell_size)
    cities = [CityNode(0, 7, 7, 0.3, 0.3, 10000.0, 3.0, 1.0),
              CityNode(1, 15, 15, 0.7, 0.7, 30000.0, 5.0, 1.0)]
    blocked = np.zeros(shape, bool)
    blocked[:, :3] = True
    args = (np.ones(shape), np.zeros(shape), np.zeros(shape), blocked, np.zeros(shape, bool), grid)
    combined = _spread_city_fields(cities, *args)[0]
    individual = [_spread_city_fields([city], *args)[0] for city in cities]
    np.testing.assert_allclose(combined, sum(individual), rtol=1e-6, atol=1e-5)
    for city, values in zip(cities, individual):
        assert np.isclose(values.sum(dtype=np.float64) * cell_size**2, city.population, rtol=1e-7)
    assert np.isclose(combined.sum(dtype=np.float64) * cell_size**2, 40000.0, rtol=1e-7)
    assert np.all(combined[blocked] == 0.0)


def test_city_centers_do_not_relax_minimum_separation_or_create_zero_count_city():
    grid = WorldGridConfig(7, 7, 1.0)
    config = replace(CityConfig(), min_city_distance_km=4.0)
    suitability = np.ones((7, 7))
    assert _select_city_centers(suitability, grid, config, np.random.default_rng(1), 0) == []
    centers = _select_city_centers(suitability, grid, config, np.random.default_rng(1), 20)
    for index, (row, col) in enumerate(centers):
        assert all(np.hypot(row - rr, col - cc) >= 4.0 for rr, cc in centers[:index])


def test_source_capacity_integrates_area_with_no_overlapping_project_double_count():
    config = replace(EnergyConfig(), wind_capacity_max_mw=1e6, wind_cluster_radius_km=2.0,
                     pv_capacity_max_mw=1e6, pv_cluster_radius_km=2.0)
    capacities = []
    for size, cell in ((11, 1.0), (44, 0.25)):
        grid = WorldGridConfig(size, size, cell)
        candidates = [_candidate("wind", size // 2, size // 2)]
        wind, _ = _allocate_source_capacity(candidates, [], np.ones((size, size)), grid, config)
        capacities.append(wind[0].capacity_mw)
        assert np.isclose(wind[0].capacity_mw, np.pi * 4.0 * config.wind_capacity_density_mw_km2, rtol=0.06)
    assert np.isclose(*capacities, rtol=0.06)
    wind, pv = _allocate_source_capacity([_candidate("wind", 5, 5)], [_candidate("pv", 5, 5)],
                                         np.ones((11, 11)), WorldGridConfig(11, 11, 1.0), config)
    assert not pv, "A co-located project must not double-count already allocated land"
    assert wind[0].capacity_mw == capacities[0]


def test_peak_load_is_population_scaled_and_independent_of_number_of_candidates():
    shape = (9, 9)
    grid = WorldGridConfig(*shape, 2.0)
    city = _city(shape, np.full(shape, 100.0))
    config = replace(EnergyConfig(), per_capita_peak_load_kw=1.5)
    candidates = [_candidate("load", 2, 2), _candidate("load", 7, 7, 1)]
    values = _allocate_load_capacity(candidates, city, _land_use(shape), grid, config)
    one = _allocate_load_capacity(candidates[:1], city, _land_use(shape), grid, config)
    expected = 100.0 * 81 * 4 * 1.5 / 1000
    assert np.isclose(sum(x.capacity_mw for x in values), expected)
    assert np.isclose(one[0].capacity_mw, expected)
    doubled = _allocate_load_capacity(candidates, replace(city, population_density=city.population_density * 2), _land_use(shape), grid, config)
    np.testing.assert_allclose([x.capacity_mw for x in doubled], np.asarray([x.capacity_mw for x in values]) * 2)


def test_relaxation_preserves_cross_technology_setback_and_no_feasible_load_returns_zero():
    suitability = np.ones((11, 11))
    utility = np.ones((11, 11))
    utility[5, 7] = 100.0
    row, col, _ = _best_relaxed_cell(suitability, utility, 5, 10, (),
                                    (_candidate("wind", 5, 5),), 5, 2.0, 4.0, 0.0, 2.0)
    assert np.hypot(row - 5, col - 5) >= 4.0
    result = _best_city_load_cell(suitability, np.ones((11, 11)), np.ones((11, 11), bool),
                                  0, 5, 5, 4.0, [], 2.0)
    assert result[2] == 0.0


def test_uniform_positive_renewable_resources_are_not_normalized_away():
    grid, terrain, hydro = _flat_world(size=21)
    climate = _climate(terrain.elevation.shape)
    land = generate_static_land(terrain, hydro, grid, replace(LandConfig(), protected_fraction=0.0),
                                np.random.default_rng(1), climate)
    city = _city(terrain.elevation.shape, np.zeros(terrain.elevation.shape))
    energy = generate_energy_candidates(terrain, hydro, land, city, _land_use(terrain.elevation.shape),
                                        climate.as_maps(), grid, EnergyConfig())
    assert energy.wind_candidates and energy.pv_candidates
    assert np.all(energy.wind_suitability > 0.0)
    assert np.all(energy.pv_suitability > 0.0)
