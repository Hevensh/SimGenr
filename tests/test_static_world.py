from __future__ import annotations

from dataclasses import replace

import numpy as np

from world_generator.core.config import PowerGridConfig, WorldConfig, WorldGridConfig
from world_generator.core.random_state import build_rng_registry
from world_generator.climate.climate_generator import generate_climate_baseline
from world_generator.city.city_generator import _edge_buffer, _resolve_city_targets, generate_initial_cities
from world_generator.energy.energy_candidate_generator import generate_energy_candidates
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.grid.node_builder import build_grid_nodes
from world_generator.grid.refinement_builder import refine_grid_topology
from world_generator.grid.topology_builder import build_grid_topology
from world_generator.hydrology.hydrology_generator import _priority_flood_fill, generate_hydrology
from world_generator.land.land_generator import generate_static_land
from world_generator.land_use.land_use_generator import LAND_USE_ZONE, generate_land_use_zones
from world_generator.terrain.derivatives import derive_terrain_features
from world_generator.terrain.terrain_generator import generate_terrain_base
from world_generator.weather.weather_generator import generate_daily_weather
from world_generator.weather.weather_generator import generate_hourly_weather_week, generate_hourly_weather_week_from_baseline
from world_generator.operation.source_load_forecast import generate_source_load_forecast
from world_generator.operation.storage_planning import (
    _load_weights_from_receiver,
    _max_contiguous_energy,
    analyze_storage_need,
    plan_storage_sites,
)
from world_generator.operation.storage_dispatch import dispatch_storage_week
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.grid_upgrade import build_grid_upgrade_plan
from world_generator.operation.grid_update_loop import (
    _apply_bypass_actions,
    _longest_near_parallel_match,
    _merge_collinear_branches,
    _expected_line_rate,
    _resize_branch_multiplier,
    _suitable_line_multiplier_for_voltage,
    run_grid_update_loop,
)
from world_generator.core.datatypes import (
    BranchElectricalParam,
    BusElectricalParam,
    GridBus,
    GridEdge,
    GridElectricalState,
    HydrologyState,
    RefinedGridTopologyState,
    StaticLandState,
    TerrainFeatures,
)


def test_static_terrain_is_reproducible() -> None:
    config = WorldConfig()
    rngs_a = build_rng_registry(config.seed)
    rngs_b = build_rng_registry(config.seed)

    terrain_a = generate_terrain_base(config.world, config.terrain, rngs_a.generator("terrain"))
    terrain_b = generate_terrain_base(config.world, config.terrain, rngs_b.generator("terrain"))

    np.testing.assert_array_equal(terrain_a.elevation, terrain_b.elevation)


def test_multiscale_terrain_tiles_match_larger_extent() -> None:
    config = WorldConfig(seed=42)
    terrain_config = replace(config.terrain, algorithm="multiscale_v2")
    large_grid = replace(
        config.world,
        height=128,
        width=128,
        cell_size_km=1.0,
        origin_x_km=0.0,
        origin_y_km=0.0,
    )
    large = generate_terrain_base(
        large_grid,
        terrain_config,
        build_rng_registry(config.seed).generator("terrain"),
    ).elevation

    rows: list[np.ndarray] = []
    for origin_y in (0.0, 64.0):
        columns: list[np.ndarray] = []
        for origin_x in (0.0, 64.0):
            tile_grid = replace(
                large_grid,
                height=64,
                width=64,
                origin_x_km=origin_x,
                origin_y_km=origin_y,
            )
            tile = generate_terrain_base(
                tile_grid,
                terrain_config,
                build_rng_registry(config.seed).generator("terrain"),
            ).elevation
            columns.append(tile)
        rows.append(np.concatenate(columns, axis=1))
    stitched = np.concatenate(rows, axis=0)
    np.testing.assert_array_equal(stitched, large)


def test_scale_aware_city_targets_grow_sublinearly_with_effective_area() -> None:
    def states(shape: tuple[int, int]) -> tuple[StaticLandState, HydrologyState]:
        zeros = np.zeros(shape, dtype=np.float32)
        false = np.zeros(shape, dtype=bool)
        land = StaticLandState(
            land_cover=np.ones(shape, dtype=np.int16),
            vegetation=zeros.copy(),
            protected=false.copy(),
            buildability=np.full(shape, 0.5, dtype=np.float32),
            terrain_cost=zeros.copy(),
            water_buffer=false.copy(),
        )
        hydrology = HydrologyState(
            flow_direction=np.zeros(shape, dtype=np.int8),
            flow_accumulation=zeros.copy(),
            river_centerline=false.copy(),
            river=false.copy(),
            lake=false.copy(),
            water_depth=zeros.copy(),
            hydrology_elevation=zeros.copy(),
            watershed_id=np.zeros(shape, dtype=np.int16),
            distance_to_water=np.ones(shape, dtype=np.float32),
            flood_risk=zeros.copy(),
        )
        return land, hydrology

    config = replace(
        WorldConfig().city,
        scaling_mode="scale_aware",
        # A score of 0.5 ranks land; it no longer halves the geometric area.
        reference_effective_area_km2=4096.0,
    )
    land_64, hydrology_64 = states((64, 64))
    land_128, hydrology_128 = states((128, 128))
    count_64, population_64 = _resolve_city_targets(
        land_64,
        hydrology_64,
        WorldGridConfig(height=64, width=64, cell_size_km=1.0),
        config,
        np.random.default_rng(4),
    )
    count_128, population_128 = _resolve_city_targets(
        land_128,
        hydrology_128,
        WorldGridConfig(height=128, width=128, cell_size_km=1.0),
        config,
        np.random.default_rng(4),
    )

    assert count_64 == 8
    assert 23 <= count_128 <= 25
    assert population_64 == config.total_population
    assert 4_100_000.0 < population_128 < 4_200_000.0


def test_priority_flood_conditions_closed_depression() -> None:
    elevation = np.asarray(
        [
            [9.0, 8.0, 7.0, 8.0, 9.0],
            [8.0, 6.0, 5.0, 6.0, 8.0],
            [7.0, 5.0, 1.0, 5.0, 7.0],
            [8.0, 6.0, 5.0, 6.0, 8.0],
            [9.0, 8.0, 7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    filled = _priority_flood_fill(elevation, epsilon_m=0.01)
    assert filled[2, 2] > elevation[2, 2]
    assert filled[2, 2] >= np.min(filled[[0, -1], :])


def test_static_terrain_datum_and_relief_vary_by_seed() -> None:
    config = WorldConfig()
    # Only the legacy generator rescales each window to the full sampled relief.
    # multiscale_v2 keeps absolute coordinates; tile consistency is tested above.
    config = replace(config, terrain=replace(config.terrain, algorithm="legacy"))
    minima: list[float] = []
    reliefs: list[float] = []

    for seed in range(32):
        terrain = generate_terrain_base(config.world, config.terrain, np.random.default_rng(seed))
        minimum = float(np.min(terrain.elevation))
        relief = float(np.max(terrain.elevation)) - minimum
        minima.append(minimum)
        reliefs.append(relief)

        assert config.terrain.lowland_elevation_min_m - 1e-3 <= minimum
        assert minimum <= config.terrain.lowland_elevation_max_m + 1e-3
        assert relief > 0.0

    assert np.ptp(minima) > 1.0
    assert np.ptp(reliefs) > 1.0
    assert abs(float(np.mean(reliefs)) - config.terrain.relief_mean_m) < config.terrain.relief_std_m
    assert 0.4 * config.terrain.relief_std_m < float(np.std(reliefs)) < 1.6 * config.terrain.relief_std_m


def test_storage_need_event_energy_and_network_decay() -> None:
    assert _max_contiguous_energy(np.asarray([0.0, 2.0, 3.0, 0.0, 4.0, 1.0])) == 5.0
    weights = _load_weights_from_receiver(
        0,
        np.asarray([0, 2], dtype=np.int32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        {0: {1}, 1: {0, 2}, 2: {1}},
        1.0,
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_city_edge_buffer_keeps_boundary_candidates_viable() -> None:
    config = WorldConfig()
    edge_factor = _edge_buffer(
        (9, 9),
        config.world,
        config.city.edge_buffer_km,
        config.city.edge_buffer_min_factor,
    )
    floor = config.city.edge_buffer_min_factor
    assert np.isclose(edge_factor[0, 4], floor)
    expected_one_cell = floor + (1.0 - floor) * min(config.world.cell_size_km / config.city.edge_buffer_km, 1.0)
    assert np.isclose(edge_factor[1, 4], expected_one_cell)
    assert edge_factor[2, 4] == 1.0


def test_terrain_feature_shapes_and_finiteness() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    features = derive_terrain_features(base, config.world)

    for values in features.as_maps().values():
        assert values.shape == (config.world.height, config.world.width)
        assert np.isfinite(values).all()


def test_climate_baseline_shapes_and_expected_relationships() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )

    for values in climate.as_maps().values():
        assert values.shape == (config.world.height, config.world.width)
        assert np.isfinite(values).all()

    elevation_corr = np.corrcoef(
        terrain.elevation.ravel(),
        climate.mean_temperature.ravel(),
    )[0, 1]
    assert elevation_corr < -0.25

    near_threshold = np.quantile(hydrology.distance_to_water, 0.25)
    far_threshold = np.quantile(hydrology.distance_to_water, 0.75)
    near_water = hydrology.distance_to_water <= near_threshold
    far_water = hydrology.distance_to_water >= far_threshold
    assert climate.annual_temperature_amplitude[near_water].mean() < climate.annual_temperature_amplitude[far_water].mean()


def test_daily_weather_shapes_and_basic_relationships() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    weather = generate_daily_weather(
        terrain,
        hydrology,
        climate,
        config.world,
        config.weather,
        rngs.generator("operation"),
    )
    channel_index = {name: index for index, name in enumerate(weather.channel_names)}

    assert weather.dynamic.shape == (
        config.weather.days,
        len(weather.channel_names),
        config.world.height,
        config.world.width,
    )
    assert weather.weather_class.shape == (config.weather.days, config.world.height, config.world.width)
    assert np.isfinite(weather.dynamic).all()

    temperature_mean = weather.dynamic[:, channel_index["temperature"]].mean(axis=(1, 2))
    assert temperature_mean[172] > temperature_mean[355]

    day = min(172, config.weather.days - 1)
    cloud = weather.dynamic[day, channel_index["cloud"]].ravel()
    irradiance = weather.dynamic[day, channel_index["irradiance"]].ravel()
    assert np.corrcoef(cloud, irradiance)[0, 1] < -0.5


def test_hourly_weather_week_shapes_and_diurnal_signal() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    operation_rng = rngs.generator("operation")
    daily = generate_daily_weather(terrain, hydrology, climate, config.world, config.weather, operation_rng)
    hourly = generate_hourly_weather_week(daily, config.weather, operation_rng)
    channel_index = {name: index for index, name in enumerate(hourly.channel_names)}

    assert hourly.dynamic.shape == (
        config.weather.hourly_week_days * 24,
        len(hourly.channel_names),
        config.world.height,
        config.world.width,
    )
    assert hourly.time_unit == "hour"
    assert np.isfinite(hourly.dynamic).all()

    irradiance = hourly.dynamic[:, channel_index["irradiance"]].mean(axis=(1, 2))
    midday = np.asarray([hour % 24 == 12 for hour in range(hourly.dynamic.shape[0])])
    midnight = np.asarray([hour % 24 == 0 for hour in range(hourly.dynamic.shape[0])])
    assert irradiance[midday].mean() > irradiance[midnight].mean() + 20.0


def test_hourly_weather_cloud_systems_are_patchy() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    hourly = generate_hourly_weather_week_from_baseline(
        terrain,
        hydrology,
        climate,
        config.world,
        config.weather,
        rngs.generator("operation"),
    )
    channel_index = {name: index for index, name in enumerate(hourly.channel_names)}
    cloud = hourly.dynamic[:, channel_index["cloud"]]
    precipitation = hourly.dynamic[:, channel_index["precipitation"]]
    irradiance = hourly.dynamic[:, channel_index["irradiance"]]

    assert hourly.dynamic.shape == (
        config.weather.hourly_week_days * 24,
        len(hourly.channel_names),
        config.world.height,
        config.world.width,
    )
    assert np.nanpercentile(cloud, 95) - np.nanpercentile(cloud, 5) > 0.20
    # Cloud frequency is a scenario output; a fixed 35% cap is not a physical
    # constraint. Check spatial variation, without tuning a climate to a seed.
    assert float(cloud.std(axis=(1, 2)).mean()) > 0.05
    assert float((precipitation > 0.05).mean()) < float((cloud > 0.35).mean())
    midday_index = 12
    high_cloud = cloud[midday_index] >= np.nanpercentile(cloud[midday_index], 80)
    low_cloud = cloud[midday_index] <= np.nanpercentile(cloud[midday_index], 20)
    from world_generator.weather.physics import clear_sky_transmissivity, extraterrestrial_hourly_irradiance, latitude_grid
    toa = extraterrestrial_hourly_irradiance(latitude_grid(config.world, terrain.elevation.shape), hourly.timestamps[midday_index] // 24)
    clear = toa[midday_index] * clear_sky_transmissivity(terrain.elevation)
    transmission = np.divide(irradiance[midday_index], clear, out=np.zeros_like(clear), where=clear > 0)
    assert float(transmission[high_cloud].mean()) < float(transmission[low_cloud].mean())


def test_initial_cities_respect_core_constraints() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    water = hydrology.river | hydrology.lake

    assert len(city.cities) == config.city.city_count
    assert np.isfinite(city.city_suitability).all()
    assert np.isfinite(city.population_density).all()
    assert city.urban_mask.any()
    for node in city.cities:
        assert not water[node.row, node.col]
        assert not land.protected[node.row, node.col]
        assert hydrology.flood_risk[node.row, node.col] <= config.city.max_core_flood_risk


def test_land_use_zones_and_load_density_are_consistent() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    maps = land_use.as_maps()
    water = hydrology.river | hydrology.lake

    for values in maps.values():
        assert values.shape == (config.world.height, config.world.width)
        assert np.isfinite(values).all()

    assert land_use.load_density_base.max() > 0.0
    assert (land_use.load_density_base[water] == 0.0).all()
    assert (land_use.load_density_base[land.protected] == 0.0).all()
    assert (land_use.land_use_zone[water] == LAND_USE_ZONE["water"]).all()
    assert (land_use.land_use_zone[land.protected & ~water] == LAND_USE_ZONE["protected"]).all()
    assert np.isin(land_use.land_use_zone, list(LAND_USE_ZONE.values())).all()

    gentle = terrain.slope <= np.quantile(terrain.slope, 0.35)
    steep = terrain.slope >= np.quantile(terrain.slope, 0.85)
    eligible_gentle = gentle & ~water & ~land.protected
    eligible_steep = steep & ~water & ~land.protected
    assert land_use.agriculture[eligible_steep].mean() < land_use.agriculture[eligible_gentle].mean()

    urban_high = city.urban_density >= np.quantile(city.urban_density, 0.85)
    urban_low = city.urban_density <= np.quantile(city.urban_density, 0.45)
    eligible_high = urban_high & ~water & ~land.protected
    eligible_low = urban_low & ~water & ~land.protected
    assert land_use.residential[eligible_high].mean() > land_use.residential[eligible_low].mean()


def test_energy_candidates_respect_exclusion_masks_and_density() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    energy = generate_energy_candidates(
        terrain,
        hydrology,
        land,
        city,
        land_use,
        climate.as_maps(),
        config.world,
        config.energy,
    )
    water = hydrology.river | hydrology.lake

    assert len(energy.wind_candidates) == config.energy.wind_candidate_count
    assert len(energy.pv_candidates) == config.energy.pv_candidate_count
    assert len(energy.load_candidates) == config.energy.load_node_count
    for candidate in (*energy.wind_candidates, *energy.pv_candidates, *energy.load_candidates):
        assert 0 <= candidate.row < config.world.height
        assert 0 <= candidate.col < config.world.width
        assert not water[candidate.row, candidate.col]
        assert not land.protected[candidate.row, candidate.col]
        assert candidate.capacity_mw > 0.0
        assert candidate.suitability > 0.0

    load_values = np.asarray([energy.load_node_density[item.row, item.col] for item in energy.load_candidates])
    assert (load_values > 0.0).all()
    assert load_values.mean() > energy.load_node_density.mean()

    for node in city.cities:
        radius_cells = max(node.radius_km / max(config.world.cell_size_km, 1e-6), 1.0)
        assert any(np.hypot(item.row - node.row, item.col - node.col) <= radius_cells for item in energy.load_candidates)

    sources = [*energy.wind_candidates, *energy.pv_candidates]
    min_source_cells = config.energy.min_source_distance_km / max(config.world.cell_size_km, 1e-6)
    min_cross_source_cells = config.energy.min_cross_source_distance_km / max(config.world.cell_size_km, 1e-6)
    for index, candidate in enumerate(sources):
        for other in sources[index + 1 :]:
            required_distance = min_source_cells if candidate.kind == other.kind else min_cross_source_cells
            assert np.hypot(candidate.row - other.row, candidate.col - other.col) >= required_distance * 0.65


def test_grid_bus_candidates_include_sources_loads_and_thermal_buffers() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    energy = generate_energy_candidates(
        terrain,
        hydrology,
        land,
        city,
        land_use,
        climate.as_maps(),
        config.world,
        config.energy,
    )
    grid_nodes = build_grid_nodes(terrain, hydrology, land, land_use, energy, config.world, config.power_grid)
    water = hydrology.river | hydrology.lake
    bus_kinds = [item.kind for item in grid_nodes.buses]

    assert bus_kinds.count("load_bus") == len(energy.load_candidates)
    assert bus_kinds.count("wind_bus") == len(energy.wind_candidates)
    assert bus_kinds.count("pv_bus") == len(energy.pv_candidates)
    assert bus_kinds.count("thermal_bus") >= config.power_grid.thermal_candidate_count
    for bus in grid_nodes.buses:
        assert not water[bus.row, bus.col]
        assert not land.protected[bus.row, bus.col]
        assert bus.capacity_mw > 0.0
        assert bus.suitability > 0.0
    thermal_externalities = [item.externality_score for item in grid_nodes.buses if item.kind == "thermal_bus"]
    assert max(thermal_externalities) < 0.80
    generation_buses = [item for item in grid_nodes.buses if item.kind in {"wind_bus", "pv_bus"}]
    thermal_buses = [item for item in grid_nodes.buses if item.kind == "thermal_bus"]
    min_generation_cells = config.power_grid.thermal_min_renewable_distance_km / max(config.world.cell_size_km, 1e-6)
    for thermal in thermal_buses:
        for generator in generation_buses:
            assert np.hypot(thermal.row - generator.row, thermal.col - generator.col) >= min_generation_cells * 0.65


def test_grid_topology_is_connected_and_has_routes() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    energy = generate_energy_candidates(
        terrain,
        hydrology,
        land,
        city,
        land_use,
        climate.as_maps(),
        config.world,
        config.energy,
    )
    grid_nodes = build_grid_nodes(terrain, hydrology, land, land_use, energy, config.world, config.power_grid)
    topology = build_grid_topology(terrain, hydrology, land, grid_nodes, config.world, config.power_grid)
    bus_ids = {item.bus_id for item in grid_nodes.buses}
    parent = {item: item for item in bus_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in topology.edges:
        assert edge.from_bus in bus_ids
        assert edge.to_bus in bus_ids
        assert edge.length_km > 0.0
        assert edge.route_cost >= edge.length_km
        assert len(edge.path_rows) == len(edge.path_cols)
        parent[find(edge.to_bus)] = find(edge.from_bus)

    assert len({find(item) for item in bus_ids}) == 1
    assert len(topology.edges) >= len(grid_nodes.buses) - 1
    assert topology.line_route_map.max() > 0.0


def test_refined_grid_topology_keeps_full_astar_paths_without_transit_buses() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    energy = generate_energy_candidates(
        terrain,
        hydrology,
        land,
        city,
        land_use,
        climate.as_maps(),
        config.world,
        config.energy,
    )
    grid_nodes = build_grid_nodes(terrain, hydrology, land, land_use, energy, config.world, config.power_grid)
    topology = build_grid_topology(terrain, hydrology, land, grid_nodes, config.world, config.power_grid)
    refined = refine_grid_topology(terrain, hydrology, land, grid_nodes, topology, config.world, config.power_grid)
    refined_bus_ids = {item.bus_id for item in refined.refined_buses}
    parent = {item: item for item in refined_bus_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    assert len(refined.refined_buses) == len(grid_nodes.buses)
    assert len(refined.refined_edges) == len(topology.edges)
    assert not (refined.transit_bus_map >= 0).any()
    bus_by_id = {bus.bus_id: bus for bus in refined.refined_buses}
    for edge in refined.refined_edges:
        assert len(edge.path_rows) == len(edge.path_cols)
        assert len(edge.path_rows) >= 2
        assert (edge.path_rows[0], edge.path_cols[0]) == (bus_by_id[edge.from_bus].row, bus_by_id[edge.from_bus].col)
        assert (edge.path_rows[-1], edge.path_cols[-1]) == (bus_by_id[edge.to_bus].row, bus_by_id[edge.to_bus].col)
        parent[find(edge.to_bus)] = find(edge.from_bus)
    assert len({find(item) for item in refined_bus_ids}) == 1


def test_grid_electrical_parameters_cover_refined_topology() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    energy = generate_energy_candidates(
        terrain,
        hydrology,
        land,
        city,
        land_use,
        climate.as_maps(),
        config.world,
        config.energy,
    )
    grid_nodes = build_grid_nodes(terrain, hydrology, land, land_use, energy, config.world, config.power_grid)
    topology = build_grid_topology(terrain, hydrology, land, grid_nodes, config.world, config.power_grid)
    refined = refine_grid_topology(terrain, hydrology, land, grid_nodes, topology, config.world, config.power_grid)
    electrical = build_grid_electrical(refined)

    assert len(electrical.bus_params) == len(refined.refined_buses)
    assert len(electrical.branch_params) == len(refined.refined_edges)
    for bus in electrical.bus_params:
        assert bus.nominal_kv in {110.0, 220.0}
        assert 0.9 <= bus.power_factor <= 1.0
        assert 0.98 <= bus.voltage_setpoint_pu <= 1.03
    for branch in electrical.branch_params:
        assert branch.nominal_kv in {110.0, 220.0}
        assert branch.r_ohm > 0.0
        assert branch.x_ohm > branch.r_ohm
        assert branch.b_us > 0.0
        assert branch.rate_mva > 0.0


def test_source_load_forecast_matches_refined_buses() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    operation_rng = rngs.generator("operation")
    daily = generate_daily_weather(terrain, hydrology, climate, config.world, config.weather, operation_rng)
    hourly = generate_hourly_weather_week(daily, config.weather, operation_rng)
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    energy = generate_energy_candidates(
        terrain,
        hydrology,
        land,
        city,
        land_use,
        climate.as_maps(),
        config.world,
        config.energy,
    )
    grid_nodes = build_grid_nodes(terrain, hydrology, land, land_use, energy, config.world, config.power_grid)
    topology = build_grid_topology(terrain, hydrology, land, grid_nodes, config.world, config.power_grid)
    refined = refine_grid_topology(terrain, hydrology, land, grid_nodes, topology, config.world, config.power_grid)
    electrical = build_grid_electrical(refined)
    forecast = generate_source_load_forecast(hourly, refined, electrical, operation_rng)

    assert forecast.p_load_mw.shape == (config.weather.hourly_week_days * 24, len(refined.refined_buses))
    assert forecast.p_gen_available_mw.shape == forecast.p_load_mw.shape
    assert forecast.p_gen_scheduled_mw.shape == forecast.p_load_mw.shape
    assert np.isfinite(forecast.p_load_mw).all()
    assert np.isfinite(forecast.p_gen_available_mw).all()
    assert forecast.p_load_mw.sum() > 0.0
    assert forecast.p_gen_available_mw.sum() > 0.0
    assert forecast.p_gen_scheduled_mw.sum() > 0.0
    pv_indices = [index for index, bus in enumerate(refined.refined_buses) if bus.kind == "pv_bus"]
    if pv_indices:
        pv_capacity = np.asarray([refined.refined_buses[index].capacity_mw for index in pv_indices], dtype=np.float32)
        assert np.all(forecast.p_gen_available_mw[:, pv_indices] <= 1.06 * pv_capacity[None, :] + 1e-5)


def test_dc_power_flow_matches_forecast_and_refined_edges() -> None:
    config = WorldConfig()
    rngs = build_rng_registry(config.seed)
    base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain = derive_terrain_features(base, config.world)
    hydrology = generate_hydrology(terrain, config.world, config.hydrology)
    land = generate_static_land(terrain, hydrology, config.world, config.land, rngs.generator("city"))
    climate = generate_climate_baseline(
        terrain,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    operation_rng = rngs.generator("operation")
    hourly = generate_hourly_weather_week_from_baseline(
        terrain,
        hydrology,
        climate,
        config.world,
        config.weather,
        operation_rng,
    )
    city = generate_initial_cities(
        terrain,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(terrain, hydrology, land, city, config.world, config.land_use)
    energy = generate_energy_candidates(
        terrain,
        hydrology,
        land,
        city,
        land_use,
        climate.as_maps(),
        config.world,
        config.energy,
    )
    grid_nodes = build_grid_nodes(terrain, hydrology, land, land_use, energy, config.world, config.power_grid)
    topology = build_grid_topology(terrain, hydrology, land, grid_nodes, config.world, config.power_grid)
    refined = refine_grid_topology(terrain, hydrology, land, grid_nodes, topology, config.world, config.power_grid)
    electrical = build_grid_electrical(refined)
    forecast = generate_source_load_forecast(hourly, refined, electrical, operation_rng)
    power_flow = solve_dc_power_flow(forecast, refined, electrical)

    assert power_flow.bus_p_injection_mw.shape == forecast.p_load_mw.shape
    assert power_flow.bus_angle_rad.shape == forecast.p_load_mw.shape
    assert power_flow.line_flow_mw.shape == (forecast.p_load_mw.shape[0], len(refined.refined_edges))
    assert power_flow.line_loading_ratio.shape == power_flow.line_flow_mw.shape
    assert np.isfinite(power_flow.bus_angle_rad).all()
    assert np.isfinite(power_flow.line_flow_mw).all()
    assert np.isfinite(power_flow.line_loading_ratio).all()
    np.testing.assert_allclose(power_flow.bus_p_injection_mw.sum(axis=1), 0.0, atol=1e-3)
    assert power_flow.line_loading_ratio.max() >= 0.0
    assert power_flow.summary_dict()["branch_count"] == len(refined.refined_edges)

    upgrade_plan = build_grid_upgrade_plan(power_flow, electrical)
    assert upgrade_plan.branch_ids.shape == (len(refined.refined_edges),)
    assert upgrade_plan.recommended_rate_mva.shape == upgrade_plan.current_rate_mva.shape
    assert np.isfinite(upgrade_plan.priority_score).all()
    assert np.all(upgrade_plan.recommended_rate_mva >= upgrade_plan.current_rate_mva)
    assert np.all(upgrade_plan.upgrade_factor >= 1.0)
    assert upgrade_plan.summary_dict()["branch_count"] == len(refined.refined_edges)

    update_loop = run_grid_update_loop(
        forecast,
        refined,
        topology,
        electrical,
        terrain,
        hydrology,
        land,
        config.world,
        config.power_grid,
        actions_per_iteration=2,
    )
    assert len(update_loop.iterations) == 1
    assert update_loop.iterations[0].iteration_index == 1
    base_pairs = {tuple(sorted((edge.from_bus, edge.to_bus))) for edge in topology.edges}
    for iteration in update_loop.iterations:
        assert iteration.power_flow.line_flow_mw.shape[0] == power_flow.line_flow_mw.shape[0]
        assert iteration.power_flow.line_flow_mw.shape[1] >= power_flow.line_flow_mw.shape[1]
        assert iteration.upgrade_plan.branch_ids.shape[0] == iteration.power_flow.line_flow_mw.shape[1]
        assert len(iteration.refined_topology.refined_buses) == len(iteration.electrical.bus_params)
        assert len(iteration.refined_topology.refined_edges) == len(iteration.electrical.branch_params)
        assert "line_hours_over_100pct" in iteration.summary
        for action in iteration.actions:
            for edge_from, edge_to in _action_edges_for_test(action):
                assert tuple(sorted((edge_from, edge_to))) not in base_pairs
        assert all(
            bus.kind != "transit_bus" or bus.source_kind == "implicit_collinear_junction"
            for bus in iteration.refined_topology.refined_buses
        )
    assert update_loop.summary_dict()["iteration_count"] == len(update_loop.iterations)
    final_iteration = update_loop.iterations[-1]
    storage_need = analyze_storage_need(
        final_iteration.refined_topology,
        final_iteration.electrical,
        final_iteration.power_flow,
        config.world,
        config.storage,
    )
    assert storage_need.support_requirement_mw.shape[0] == hourly.dynamic.shape[0]
    assert storage_need.support_requirement_mw.shape[1] == storage_need.bus_ids.size
    assert storage_need.need_score_map.shape == (config.world.height, config.world.width)
    assert np.isfinite(storage_need.support_requirement_mw).all()
    assert np.isfinite(storage_need.need_score).all()
    storage_plan = plan_storage_sites(final_iteration.refined_topology, storage_need, config.storage)
    load_bus_ids = {int(bus.bus_id) for bus in final_iteration.refined_topology.refined_buses if bus.kind == "load_bus"}
    assert len(storage_plan.sites) <= config.storage.site_max_count
    assert all(site.bus_id in load_bus_ids for site in storage_plan.sites)
    assert all(site.power_mw >= 0.0 and site.energy_mwh >= site.power_mw * config.storage.minimum_duration_hours for site in storage_plan.sites)
    assert np.all((storage_plan.assigned_site_ids < len(storage_plan.sites)) & (storage_plan.assigned_site_ids >= -1))
    storage_dispatch, dispatched_forecast, dispatched_power_flow, planned_electrical = dispatch_storage_week(
        final_iteration.refined_topology,
        final_iteration.electrical,
        final_iteration.power_flow,
        storage_plan,
        config.storage,
    )
    assert storage_dispatch.charge_mw.shape == storage_dispatch.discharge_mw.shape
    assert storage_dispatch.soc_mwh.shape[0] == storage_dispatch.timestamps.size + 1
    assert storage_dispatch.cycle_boundary_soc_mwh.shape == (len(storage_plan.sites),)
    assert dispatched_forecast.p_load_mw.shape == final_iteration.power_flow.served_load_mw.shape
    assert dispatched_power_flow.line_loading_ratio.shape == final_iteration.power_flow.line_loading_ratio.shape
    assert len(planned_electrical.branch_params) == len(final_iteration.electrical.branch_params)
    assert storage_dispatch.dispatched_unserved_mw.sum() <= 1e-3
    assert storage_dispatch.dispatched_line_loading_ratio.max() <= config.storage.line_operating_limit_ratio + 1e-4
    assert not np.any((storage_dispatch.charge_mw > 1e-5) & (storage_dispatch.discharge_mw > 1e-5))
    np.testing.assert_allclose(storage_dispatch.soc_mwh[-1], storage_dispatch.soc_mwh[0], atol=1e-3)
    normal_discharge = storage_dispatch.discharge_mw - storage_dispatch.emergency_discharge_mw
    normal_limit = config.storage.normal_dispatch_c_rate * storage_dispatch.site_energy_capacity_mwh
    assert np.all(storage_dispatch.charge_mw <= normal_limit[None, :] + 1e-3)
    assert np.all(normal_discharge <= normal_limit[None, :] + 1e-3)
    normal_net_power = storage_dispatch.discharge_mw - storage_dispatch.charge_mw
    cyclic_ramp = np.abs(normal_net_power - np.roll(normal_net_power, 1, axis=0))
    ramp_limit = config.storage.storage_power_ramp_fraction_per_hour * storage_dispatch.site_power_capacity_mw
    assert np.all(cyclic_ramp <= ramp_limit[None, :] + 1e-3)
    for index, site in enumerate(storage_plan.sites):
        planned_energy = storage_dispatch.site_energy_capacity_mwh[index]
        assert planned_energy + 1e-3 >= site.energy_mwh
        soc_tolerance = max(1e-4, 1e-6 * float(planned_energy))
        minimum_margin = storage_dispatch.soc_mwh[:, index].min() - planned_energy * config.storage.minimum_soc_fraction
        assert minimum_margin >= -soc_tolerance, (index, minimum_margin, planned_energy)
        assert np.all(storage_dispatch.soc_mwh[:, index] <= planned_energy * config.storage.maximum_soc_fraction + soc_tolerance)


def _action_edges_for_test(action: dict[str, object]) -> list[tuple[int, int]]:
    if action.get("action") == "downgrade_low_utilization_line":
        return []
    if "logical_edges" in action:
        return [(int(edge[0]), int(edge[1])) for edge in action["logical_edges"]]
    if action.get("action") == "swap_crossing_lines":
        return [
            (int(action["edge1_from_bus"]), int(action["edge1_to_bus"])),
            (int(action["edge2_from_bus"]), int(action["edge2_to_bus"])),
        ]
    return [(int(action["from_bus"]), int(action["to_bus"]))]


def test_stage12_segmented_crossing_swap_replans_connection() -> None:
    config = WorldConfig()
    shape = (8, 8)
    terrain = TerrainFeatures(
        elevation=np.zeros(shape, dtype=np.float32),
        slope=np.zeros(shape, dtype=np.float32),
        aspect_sin=np.zeros(shape, dtype=np.float32),
        aspect_cos=np.ones(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        curvature=np.zeros(shape, dtype=np.float32),
    )
    hydrology = HydrologyState(
        flow_direction=np.zeros(shape, dtype=np.int16),
        flow_accumulation=np.zeros(shape, dtype=np.float32),
        river_centerline=np.zeros(shape, dtype=bool),
        river=np.zeros(shape, dtype=bool),
        lake=np.zeros(shape, dtype=bool),
        water_depth=np.zeros(shape, dtype=np.float32),
        hydrology_elevation=np.zeros(shape, dtype=np.float32),
        watershed_id=np.zeros(shape, dtype=np.int16),
        distance_to_water=np.ones(shape, dtype=np.float32),
        flood_risk=np.zeros(shape, dtype=np.float32),
    )
    land = StaticLandState(
        land_cover=np.zeros(shape, dtype=np.int16),
        vegetation=np.zeros(shape, dtype=np.float32),
        protected=np.zeros(shape, dtype=bool),
        buildability=np.ones(shape, dtype=np.float32),
        terrain_cost=np.zeros(shape, dtype=np.float32),
        water_buffer=np.zeros(shape, dtype=bool),
    )
    buses = (
        GridBus(0, "load_bus", 0, 0, 0.0, 0.0, 20.0, 1.0, 0.0, "test", 0),
        GridBus(1, "load_bus", 7, 7, 1.0, 1.0, 20.0, 1.0, 0.0, "test", 1),
        GridBus(2, "load_bus", 0, 7, 1.0, 0.0, 20.0, 1.0, 0.0, "test", 2),
        GridBus(3, "load_bus", 7, 0, 0.0, 1.0, 20.0, 1.0, 0.0, "test", 3),
    )
    edge = GridEdge(0, 2, 3, 9.9, 9.9, False, tuple(range(8)), tuple(range(7, -1, -1)))
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.zeros(shape, dtype=np.float32),
        refined_grid_edge_map=np.full(shape, -1, dtype=np.int16),
        transit_bus_map=np.full(shape, -1, dtype=np.int16),
        refined_buses=buses,
        refined_edges=(edge,),
    )
    electrical = GridElectricalState(
        bus_params=tuple(
            BusElectricalParam(bus.bus_id, bus.kind, 110.0, 0.0, 0.0, 0.0, 1.0, 1.0, "TEST")
            for bus in buses
        ),
        branch_params=(
            BranchElectricalParam(0, 2, 3, 110.0, 9.9, 1.0, 4.0, 10.0, 120.0, False),
        ),
    )
    action = {
        "action": "add_bypass_line",
        "bypass_kind": "test",
        "source_branch_id": 0,
        "from_bus": 0,
        "to_bus": 1,
        "bypass_rate_mva": 140.0,
        "nominal_kv": 110.0,
        "source_r_ohm_per_km": 0.1,
        "source_x_ohm_per_km": 0.4,
        "source_b_us_per_km": 2.0,
        "source_rate_mva": 120.0,
        "priority_score": 1.0,
        "peak_loading_ratio": 1.5,
        "hours_over_100pct": 4,
    }

    updated_topology, updated_electrical, applied = _apply_bypass_actions(
        topology,
        electrical,
        [action],
        terrain,
        hydrology,
        land,
        config.world,
        config.power_grid,
    )

    assert applied
    assert applied[0]["action"] == "segmented_swap_crossing_lines"
    assert applied[0]["removed_branch_id"] == 0
    assert len(updated_topology.refined_edges) == len(updated_electrical.branch_params)
    assert len(updated_topology.refined_buses) == len(updated_electrical.bus_params)


def test_stage12_new_segment_crossing_swap_replans_both_sides() -> None:
    config = WorldConfig()
    shape = (8, 8)
    terrain = TerrainFeatures(
        elevation=np.zeros(shape, dtype=np.float32),
        slope=np.zeros(shape, dtype=np.float32),
        aspect_sin=np.zeros(shape, dtype=np.float32),
        aspect_cos=np.ones(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        curvature=np.zeros(shape, dtype=np.float32),
    )
    hydrology = HydrologyState(
        flow_direction=np.zeros(shape, dtype=np.int16),
        flow_accumulation=np.zeros(shape, dtype=np.float32),
        river_centerline=np.zeros(shape, dtype=bool),
        river=np.zeros(shape, dtype=bool),
        lake=np.zeros(shape, dtype=bool),
        water_depth=np.zeros(shape, dtype=np.float32),
        hydrology_elevation=np.zeros(shape, dtype=np.float32),
        watershed_id=np.zeros(shape, dtype=np.int16),
        distance_to_water=np.ones(shape, dtype=np.float32),
        flood_risk=np.zeros(shape, dtype=np.float32),
    )
    land = StaticLandState(
        land_cover=np.zeros(shape, dtype=np.int16),
        vegetation=np.zeros(shape, dtype=np.float32),
        protected=np.zeros(shape, dtype=bool),
        buildability=np.ones(shape, dtype=np.float32),
        terrain_cost=np.zeros(shape, dtype=np.float32),
        water_buffer=np.zeros(shape, dtype=bool),
    )
    buses = (
        GridBus(0, "load_bus", 0, 0, 0.0, 0.0, 20.0, 1.0, 0.0, "test", 0),
        GridBus(1, "load_bus", 7, 7, 1.0, 1.0, 20.0, 1.0, 0.0, "test", 1),
        GridBus(2, "load_bus", 0, 7, 1.0, 0.0, 20.0, 1.0, 0.0, "test", 2),
        GridBus(3, "load_bus", 7, 0, 0.0, 1.0, 20.0, 1.0, 0.0, "test", 3),
    )
    edge = GridEdge(0, 0, 2, 7.0, 7.0, False, (0, 0), (0, 7))
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.zeros(shape, dtype=np.float32),
        refined_grid_edge_map=np.full(shape, -1, dtype=np.int16),
        transit_bus_map=np.full(shape, -1, dtype=np.int16),
        refined_buses=buses,
        refined_edges=(edge,),
    )
    electrical = GridElectricalState(
        bus_params=tuple(
            BusElectricalParam(bus.bus_id, bus.kind, 110.0, 0.0, 0.0, 0.0, 1.0, 1.0, "TEST")
            for bus in buses
        ),
        branch_params=(
            BranchElectricalParam(0, 0, 2, 110.0, 7.0, 1.0, 4.0, 10.0, 120.0, False),
        ),
    )
    action = {
        "action": "swap_crossing_lines",
        "bypass_kind": "test",
        "source_branch_id": 0,
        "removed_branch_id": 0,
        "edge1_from_bus": 0,
        "edge1_to_bus": 1,
        "edge2_from_bus": 2,
        "edge2_to_bus": 3,
        "bypass_rate_mva": 140.0,
        "nominal_kv": 110.0,
        "source_r_ohm_per_km": 0.1,
        "source_x_ohm_per_km": 0.4,
        "source_b_us_per_km": 2.0,
        "source_rate_mva": 120.0,
        "priority_score": 1.0,
        "peak_loading_ratio": 1.5,
        "hours_over_100pct": 4,
    }

    updated_topology, updated_electrical, applied = _apply_bypass_actions(
        topology,
        electrical,
        [action],
        terrain,
        hydrology,
        land,
        config.world,
        config.power_grid,
    )

    assert applied
    assert applied[0]["action"] == "segmented_swap_new_lines"
    assert "swap_new_segment_edges" in applied[0]
    assert len(updated_topology.refined_edges) == len(updated_electrical.branch_params)
    assert len(updated_topology.refined_buses) == len(updated_electrical.bus_params)


def test_stage12_reroute_replaces_old_branch_without_splitting_network() -> None:
    config = WorldConfig()
    shape = (5, 5)
    terrain = TerrainFeatures(
        elevation=np.zeros(shape, dtype=np.float32),
        slope=np.zeros(shape, dtype=np.float32),
        aspect_sin=np.zeros(shape, dtype=np.float32),
        aspect_cos=np.ones(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        curvature=np.zeros(shape, dtype=np.float32),
    )
    hydrology = HydrologyState(
        flow_direction=np.zeros(shape, dtype=np.int16),
        flow_accumulation=np.zeros(shape, dtype=np.float32),
        river_centerline=np.zeros(shape, dtype=bool),
        river=np.zeros(shape, dtype=bool),
        lake=np.zeros(shape, dtype=bool),
        water_depth=np.zeros(shape, dtype=np.float32),
        hydrology_elevation=np.zeros(shape, dtype=np.float32),
        watershed_id=np.zeros(shape, dtype=np.int16),
        distance_to_water=np.ones(shape, dtype=np.float32),
        flood_risk=np.zeros(shape, dtype=np.float32),
    )
    land = StaticLandState(
        land_cover=np.zeros(shape, dtype=np.int16),
        vegetation=np.zeros(shape, dtype=np.float32),
        protected=np.zeros(shape, dtype=bool),
        buildability=np.ones(shape, dtype=np.float32),
        terrain_cost=np.zeros(shape, dtype=np.float32),
        water_buffer=np.zeros(shape, dtype=bool),
    )
    buses = (
        GridBus(0, "transit_bus", 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0, "test", 0),
        GridBus(1, "load_bus", 2, 2, 0.5, 0.5, 20.0, 1.0, 0.0, "test", 1),
        GridBus(2, "load_bus", 4, 4, 1.0, 1.0, 20.0, 1.0, 0.0, "test", 2),
    )
    edges = (
        GridEdge(0, 0, 1, 2.8, 2.8, False, (0, 1, 2), (0, 1, 2)),
        GridEdge(1, 1, 2, 2.8, 2.8, False, (2, 3, 4), (2, 3, 4)),
    )
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.zeros(shape, dtype=np.float32),
        refined_grid_edge_map=np.full(shape, -1, dtype=np.int16),
        transit_bus_map=np.full(shape, -1, dtype=np.int16),
        refined_buses=buses,
        refined_edges=edges,
    )
    electrical = GridElectricalState(
        bus_params=tuple(
            BusElectricalParam(bus.bus_id, bus.kind, 110.0, 0.0, 0.0, 0.0, 1.0, 1.0, "TEST")
            for bus in buses
        ),
        branch_params=(
            BranchElectricalParam(0, 0, 1, 110.0, 2.8, 0.28, 1.12, 5.6, 120.0, False),
            BranchElectricalParam(1, 1, 2, 110.0, 2.8, 0.28, 1.12, 5.6, 120.0, False),
        ),
    )
    action = {
        "action": "reroute_overloaded_endpoint",
        "candidate_mode": "reroute",
        "bypass_kind": "reroute-B-to-j",
        "source_branch_id": 0,
        "removed_branch_id": 0,
        "old_from_bus": 0,
        "old_to_bus": 1,
        "from_bus": 0,
        "to_bus": 2,
        "bypass_rate_mva": 140.0,
        "nominal_kv": 110.0,
        "source_r_ohm_per_km": 0.1,
        "source_x_ohm_per_km": 0.4,
        "source_b_us_per_km": 2.0,
        "source_rate_mva": 120.0,
        "priority_score": 1.0,
        "peak_loading_ratio": 1.5,
        "hours_over_100pct": 4,
    }

    updated_topology, updated_electrical, applied = _apply_bypass_actions(
        topology, electrical, [action], terrain, hydrology, land, config.world, config.power_grid
    )

    assert applied
    pairs = {frozenset((branch.from_bus, branch.to_bus)) for branch in updated_electrical.branch_params}
    assert frozenset((0, 1)) not in pairs
    assert frozenset((1, 2)) in pairs
    assert applied[0]["logical_edges"] == [[0, 2]]
    adjacency: dict[int, set[int]] = {}
    for branch in updated_electrical.branch_params:
        adjacency.setdefault(int(branch.from_bus), set()).add(int(branch.to_bus))
        adjacency.setdefault(int(branch.to_bus), set()).add(int(branch.from_bus))
    reachable = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        unseen = adjacency.get(node, set()) - reachable
        reachable.update(unseen)
        frontier.extend(unseen)
    assert {0, 1, 2} <= reachable
    assert len(updated_topology.refined_edges) == len(updated_electrical.branch_params)


def test_stage12_collinear_paths_use_hidden_parallel_equivalent_junction() -> None:
    config = WorldConfig()
    shape = (5, 5)
    terrain = TerrainFeatures(
        elevation=np.zeros(shape, dtype=np.float32),
        slope=np.zeros(shape, dtype=np.float32),
        aspect_sin=np.zeros(shape, dtype=np.float32),
        aspect_cos=np.ones(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        curvature=np.zeros(shape, dtype=np.float32),
    )
    hydrology = HydrologyState(
        flow_direction=np.zeros(shape, dtype=np.int16),
        flow_accumulation=np.zeros(shape, dtype=np.float32),
        river_centerline=np.zeros(shape, dtype=bool),
        river=np.zeros(shape, dtype=bool),
        lake=np.zeros(shape, dtype=bool),
        water_depth=np.zeros(shape, dtype=np.float32),
        hydrology_elevation=np.zeros(shape, dtype=np.float32),
        watershed_id=np.zeros(shape, dtype=np.int16),
        distance_to_water=np.ones(shape, dtype=np.float32),
        flood_risk=np.zeros(shape, dtype=np.float32),
    )
    land = StaticLandState(
        land_cover=np.zeros(shape, dtype=np.int16),
        vegetation=np.zeros(shape, dtype=np.float32),
        protected=np.zeros(shape, dtype=bool),
        buildability=np.ones(shape, dtype=np.float32),
        terrain_cost=np.zeros(shape, dtype=np.float32),
        water_buffer=np.zeros(shape, dtype=bool),
    )
    buses = (
        GridBus(0, "load_bus", 0, 0, 0.0, 0.0, 30.0, 1.0, 0.0, "test", 0),
        GridBus(1, "wind_bus", 4, 2, 0.5, 1.0, 30.0, 1.0, 0.0, "test", 1),
        GridBus(2, "pv_bus", 4, 4, 1.0, 1.0, 30.0, 1.0, 0.0, "test", 2),
    )
    edges = (
        GridEdge(0, 0, 1, 5.0, 5.0, False, (0, 1, 2, 3, 4), (0, 1, 2, 2, 2)),
        GridEdge(1, 0, 2, 5.6, 5.6, True, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4)),
    )
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.zeros(shape, dtype=np.float32),
        refined_grid_edge_map=np.full(shape, -1, dtype=np.int16),
        transit_bus_map=np.full(shape, -1, dtype=np.int16),
        refined_buses=buses,
        refined_edges=edges,
    )
    electrical = GridElectricalState(
        bus_params=tuple(
            BusElectricalParam(bus.bus_id, bus.kind, 110.0, 0.0, 0.0, 0.0, 1.0, 1.0, "TEST")
            for bus in buses
        ),
        branch_params=(
            BranchElectricalParam(0, 0, 1, 110.0, 5.0, 0.5, 2.0, 10.0, 120.0, False),
            BranchElectricalParam(1, 0, 2, 110.0, 5.6, 0.56, 2.24, 11.2, 180.0, True),
        ),
    )

    merged_topology, merged_electrical, actions = _merge_collinear_branches(
        topology, electrical, terrain, hydrology, land, config.world, config.power_grid
    )

    assert len(actions) == 1
    junctions = [bus for bus in merged_topology.refined_buses if bus.kind == "transit_bus"]
    assert len(junctions) == 1
    assert junctions[0].source_kind == "implicit_collinear_junction"
    assert len(merged_topology.refined_edges) == 3
    assert len(merged_electrical.branch_params) == 3
    trunk = next(
        branch
        for branch in merged_electrical.branch_params
        if {branch.from_bus, branch.to_bus} == {0, junctions[0].bus_id}
    )
    assert np.isclose(trunk.rate_mva, 300.0)
    assert trunk.r_ohm / trunk.length_km < 0.1
    assert merged_topology.transit_bus_map[junctions[0].row, junctions[0].col] == junctions[0].bus_id


def test_stage12_internal_path_overlap_uses_two_hidden_junctions() -> None:
    config = WorldConfig()
    shape = (7, 7)
    terrain = TerrainFeatures(
        elevation=np.zeros(shape, dtype=np.float32),
        slope=np.zeros(shape, dtype=np.float32),
        aspect_sin=np.zeros(shape, dtype=np.float32),
        aspect_cos=np.ones(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        curvature=np.zeros(shape, dtype=np.float32),
    )
    hydrology = HydrologyState(
        flow_direction=np.zeros(shape, dtype=np.int16),
        flow_accumulation=np.zeros(shape, dtype=np.float32),
        river_centerline=np.zeros(shape, dtype=bool),
        river=np.zeros(shape, dtype=bool),
        lake=np.zeros(shape, dtype=bool),
        water_depth=np.zeros(shape, dtype=np.float32),
        hydrology_elevation=np.zeros(shape, dtype=np.float32),
        watershed_id=np.zeros(shape, dtype=np.int16),
        distance_to_water=np.ones(shape, dtype=np.float32),
        flood_risk=np.zeros(shape, dtype=np.float32),
    )
    land = StaticLandState(
        land_cover=np.zeros(shape, dtype=np.int16),
        vegetation=np.zeros(shape, dtype=np.float32),
        protected=np.zeros(shape, dtype=bool),
        buildability=np.ones(shape, dtype=np.float32),
        terrain_cost=np.zeros(shape, dtype=np.float32),
        water_buffer=np.zeros(shape, dtype=bool),
    )
    buses = (
        GridBus(0, "load_bus", 0, 0, 0.0, 0.0, 30.0, 1.0, 0.0, "test", 0),
        GridBus(1, "wind_bus", 6, 4, 4 / 6, 1.0, 30.0, 1.0, 0.0, "test", 1),
        GridBus(2, "pv_bus", 6, 2, 2 / 6, 1.0, 30.0, 1.0, 0.0, "test", 2),
        GridBus(3, "thermal_bus", 0, 6, 1.0, 0.0, 30.0, 1.0, 0.0, "test", 3),
    )
    edges = (
        GridEdge(0, 0, 1, 8.0, 8.0, False, (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 3, 3, 4)),
        GridEdge(1, 2, 3, 8.0, 8.0, True, (6, 5, 4, 3, 2, 1, 0), (2, 3, 3, 3, 4, 5, 6)),
    )
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.zeros(shape, dtype=np.float32),
        refined_grid_edge_map=np.full(shape, -1, dtype=np.int16),
        transit_bus_map=np.full(shape, -1, dtype=np.int16),
        refined_buses=buses,
        refined_edges=edges,
    )
    electrical = GridElectricalState(
        bus_params=tuple(
            BusElectricalParam(bus.bus_id, bus.kind, 110.0, 0.0, 0.0, 0.0, 1.0, 1.0, "TEST")
            for bus in buses
        ),
        branch_params=(
            BranchElectricalParam(0, 0, 1, 110.0, 8.0, 0.8, 3.2, 16.0, 120.0, False),
            BranchElectricalParam(1, 2, 3, 110.0, 8.0, 0.8, 3.2, 16.0, 180.0, True),
        ),
    )

    merged_topology, merged_electrical, actions = _merge_collinear_branches(
        topology, electrical, terrain, hydrology, land, config.world, config.power_grid
    )

    assert len(actions) == 1
    assert actions[0]["action"] == "merge_internal_collinear_lines"
    junctions = [bus for bus in merged_topology.refined_buses if bus.kind == "transit_bus"]
    assert {(bus.row, bus.col) for bus in junctions} == {(3, 3), (5, 3)}
    assert len(merged_topology.refined_edges) == 5
    assert len(merged_electrical.branch_params) == 5
    junction_ids = {bus.bus_id for bus in junctions}
    common = [
        branch
        for branch in merged_electrical.branch_params
        if {branch.from_bus, branch.to_bus} == junction_ids
    ]
    assert len(common) == 1
    assert common[0].rate_mva == 300.0


def test_stage12_rechecks_internal_overlap_after_prefix_merge() -> None:
    config = WorldConfig()
    shape = (7, 7)
    terrain = TerrainFeatures(
        elevation=np.zeros(shape, dtype=np.float32),
        slope=np.zeros(shape, dtype=np.float32),
        aspect_sin=np.zeros(shape, dtype=np.float32),
        aspect_cos=np.ones(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        curvature=np.zeros(shape, dtype=np.float32),
    )
    hydrology = HydrologyState(
        flow_direction=np.zeros(shape, dtype=np.int16),
        flow_accumulation=np.zeros(shape, dtype=np.float32),
        river_centerline=np.zeros(shape, dtype=bool),
        river=np.zeros(shape, dtype=bool),
        lake=np.zeros(shape, dtype=bool),
        water_depth=np.zeros(shape, dtype=np.float32),
        hydrology_elevation=np.zeros(shape, dtype=np.float32),
        watershed_id=np.zeros(shape, dtype=np.int16),
        distance_to_water=np.ones(shape, dtype=np.float32),
        flood_risk=np.zeros(shape, dtype=np.float32),
    )
    land = StaticLandState(
        land_cover=np.zeros(shape, dtype=np.int16),
        vegetation=np.zeros(shape, dtype=np.float32),
        protected=np.zeros(shape, dtype=bool),
        buildability=np.ones(shape, dtype=np.float32),
        terrain_cost=np.zeros(shape, dtype=np.float32),
        water_buffer=np.zeros(shape, dtype=bool),
    )
    buses = (
        GridBus(0, "load_bus", 0, 0, 0.0, 0.0, 30.0, 1.0, 0.0, "test", 0),
        GridBus(1, "wind_bus", 6, 2, 2 / 6, 1.0, 30.0, 1.0, 0.0, "test", 1),
        GridBus(2, "pv_bus", 6, 6, 1.0, 1.0, 30.0, 1.0, 0.0, "test", 2),
        GridBus(3, "thermal_bus", 2, 4, 4 / 6, 2 / 6, 30.0, 1.0, 0.0, "test", 3),
        GridBus(4, "load_bus", 6, 4, 4 / 6, 1.0, 30.0, 1.0, 0.0, "test", 4),
    )
    edges = (
        GridEdge(0, 0, 1, 7.0, 7.0, False, (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 2, 2, 2, 2)),
        GridEdge(1, 0, 2, 8.5, 8.5, False, (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 6)),
        GridEdge(2, 3, 4, 5.0, 5.0, True, (2, 3, 4, 5, 6), (4, 3, 4, 5, 4)),
    )
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.zeros(shape, dtype=np.float32),
        refined_grid_edge_map=np.full(shape, -1, dtype=np.int16),
        transit_bus_map=np.full(shape, -1, dtype=np.int16),
        refined_buses=buses,
        refined_edges=edges,
    )
    electrical = GridElectricalState(
        bus_params=tuple(
            BusElectricalParam(bus.bus_id, bus.kind, 110.0, 0.0, 0.0, 0.0, 1.0, 1.0, "TEST")
            for bus in buses
        ),
        branch_params=tuple(
            BranchElectricalParam(edge.edge_id, edge.from_bus, edge.to_bus, 110.0, edge.length_km, 0.8, 3.2, 16.0, 120.0, edge.is_redundant)
            for edge in edges
        ),
    )

    merged_topology, _, actions = _merge_collinear_branches(
        topology, electrical, terrain, hydrology, land, config.world, config.power_grid
    )

    assert [action["action"] for action in actions] == [
        "merge_collinear_lines",
        "merge_internal_collinear_lines",
    ]
    assert [action["merge_pass"] for action in actions] == [1, 2]
    junctions = [bus for bus in merged_topology.refined_buses if bus.kind == "transit_bus"]
    assert len(junctions) == 3


def test_stage12_merges_near_parallel_internal_corridor() -> None:
    topology, electrical, terrain, hydrology, land, grid = _near_parallel_merge_fixture((2, 3))
    merged_topology, merged_electrical, actions = _merge_collinear_branches(
        topology,
        electrical,
        terrain,
        hydrology,
        land,
        grid,
        PowerGridConfig(),
    )

    assert [action["action"] for action in actions] == ["merge_internal_near_parallel_lines"]
    assert actions[0]["near_parallel_mean_distance_km"] == 1.0
    junctions = [bus for bus in merged_topology.refined_buses if bus.kind == "transit_bus"]
    assert len(junctions) == 2
    assert len(merged_topology.refined_edges) == 5
    assert len(merged_electrical.branch_params) == 5


def test_near_parallel_match_keeps_last_straight_cell_before_turn() -> None:
    first = [(7, col) for col in range(28, 34)] + [(8, 34), (9, 35)]
    second = [(5, col) for col in range(28, 38)]
    match = _longest_near_parallel_match(first, second, 2.0, 20.0, 1.0, 4.0, allow_reverse=True)

    assert match is not None
    assert match["first_end_cell"] == (7, 33)
    assert match["length_km"] == 4.0


def test_near_parallel_match_counts_a_shared_source_endpoint() -> None:
    first = [(7, 2), (8, 2), (9, 3), (10, 4), (11, 5), (12, 5), (13, 5), (14, 5)]
    second = [(7, 2), (8, 3), (9, 4), (10, 5), (11, 6), (11, 7), (11, 8)]
    match = _longest_near_parallel_match(
        first,
        second,
        2.0,
        20.0,
        1.0,
        4.0,
        allow_reverse=False,
        include_start=True,
    )

    assert match is not None
    assert (7, 2) in {match["first_start_cell"], match["second_start_cell"]}
    assert match["length_km"] >= 4.0


def test_stage12_rechecks_near_parallel_corridor_with_third_line() -> None:
    topology, electrical, terrain, hydrology, land, grid = _near_parallel_merge_fixture((2, 3, 4))
    merged_topology, merged_electrical, actions = _merge_collinear_branches(
        topology,
        electrical,
        terrain,
        hydrology,
        land,
        grid,
        PowerGridConfig(),
    )

    near_actions = [action for action in actions if "near_parallel" in str(action["action"])]
    assert len(near_actions) >= 2
    assert max(action["merge_pass"] for action in near_actions) >= 2
    assert max(branch.rate_mva for branch in merged_electrical.branch_params) >= 360.0
    assert len(merged_topology.refined_edges) == len(merged_electrical.branch_params)


def _near_parallel_merge_fixture(
    columns: tuple[int, ...],
) -> tuple[
    RefinedGridTopologyState,
    GridElectricalState,
    TerrainFeatures,
    HydrologyState,
    StaticLandState,
    WorldGridConfig,
]:
    shape = (10, 10)
    terrain = TerrainFeatures(
        elevation=np.zeros(shape, dtype=np.float32),
        slope=np.zeros(shape, dtype=np.float32),
        aspect_sin=np.zeros(shape, dtype=np.float32),
        aspect_cos=np.ones(shape, dtype=np.float32),
        roughness=np.zeros(shape, dtype=np.float32),
        curvature=np.zeros(shape, dtype=np.float32),
    )
    hydrology = HydrologyState(
        flow_direction=np.zeros(shape, dtype=np.int16),
        flow_accumulation=np.zeros(shape, dtype=np.float32),
        river_centerline=np.zeros(shape, dtype=bool),
        river=np.zeros(shape, dtype=bool),
        lake=np.zeros(shape, dtype=bool),
        water_depth=np.zeros(shape, dtype=np.float32),
        hydrology_elevation=np.zeros(shape, dtype=np.float32),
        watershed_id=np.zeros(shape, dtype=np.int16),
        distance_to_water=np.ones(shape, dtype=np.float32),
        flood_risk=np.zeros(shape, dtype=np.float32),
    )
    land = StaticLandState(
        land_cover=np.zeros(shape, dtype=np.int16),
        vegetation=np.zeros(shape, dtype=np.float32),
        protected=np.zeros(shape, dtype=bool),
        buildability=np.ones(shape, dtype=np.float32),
        terrain_cost=np.zeros(shape, dtype=np.float32),
        water_buffer=np.zeros(shape, dtype=bool),
    )
    buses: list[GridBus] = []
    edges: list[GridEdge] = []
    for edge_id, col in enumerate(columns):
        from_bus = 2 * edge_id
        to_bus = from_bus + 1
        buses.extend(
            (
                GridBus(from_bus, "load_bus", 0, col, col / 9, 0.0, 30.0, 1.0, 0.0, "test", from_bus),
                GridBus(to_bus, "wind_bus", 9, col, col / 9, 1.0, 30.0, 1.0, 0.0, "test", to_bus),
            )
        )
        edges.append(
            GridEdge(
                edge_id,
                from_bus,
                to_bus,
                9.0,
                9.0,
                False,
                tuple(range(10)),
                tuple([col] * 10),
            )
        )
    topology = RefinedGridTopologyState(
        refined_line_route_map=np.zeros(shape, dtype=np.float32),
        refined_grid_edge_map=np.full(shape, -1, dtype=np.int16),
        transit_bus_map=np.full(shape, -1, dtype=np.int16),
        refined_buses=tuple(buses),
        refined_edges=tuple(edges),
    )
    electrical = GridElectricalState(
        bus_params=tuple(
            BusElectricalParam(bus.bus_id, bus.kind, 110.0, 0.0, 0.0, 0.0, 1.0, 1.0, "TEST")
            for bus in buses
        ),
        branch_params=tuple(
            BranchElectricalParam(edge.edge_id, edge.from_bus, edge.to_bus, 110.0, 9.0, 0.9, 3.6, 18.0, 120.0, False)
            for edge in edges
        ),
    )
    return topology, electrical, terrain, hydrology, land, WorldGridConfig(10, 10, 1.0)


def test_stage12_line_multiplier_uses_actual_peak_flow() -> None:
    assert _suitable_line_multiplier_for_voltage(np.asarray([20.0, 24.0]), 110.0) == 0.25
    assert _suitable_line_multiplier_for_voltage(np.asarray([20.0, 30.0]), 110.0) == 0.375
    assert _suitable_line_multiplier_for_voltage(np.asarray([120.0, 150.0]), 110.0) == 1.625
    assert _suitable_line_multiplier_for_voltage(np.asarray([390.0, 410.0]), 220.0) == 1.5
    constrained = PowerGridConfig(line_multiplier_step=0.25, min_line_multiplier=0.5, max_upgrade_factor=1.5)
    assert _suitable_line_multiplier_for_voltage(np.asarray([20.0, 24.0]), 110.0, constrained) == 0.5
    assert _suitable_line_multiplier_for_voltage(np.asarray([390.0, 410.0]), 220.0, constrained) == 1.5


def test_stage12_bypass_capacity_uses_expected_upgrade_factor() -> None:
    config = PowerGridConfig(line_multiplier_step=0.125, min_line_multiplier=0.125, max_upgrade_factor=2.8)
    # 50 Hz 243-AL1/39-ST1A template, S=sqrt(3)*110 kV*0.645 kA.
    base_rating = np.sqrt(3.0) * 110.0 * 0.645
    rate, multiplier = _expected_line_rate(base_rating, 110.0, 1.38, config)
    assert np.isclose(rate, 1.5 * base_rating)
    assert multiplier == 1.5
    capped_rate, capped_multiplier = _expected_line_rate(2.8 * base_rating, 110.0, 1.25, config)
    assert np.isclose(capped_rate, 2.8 * base_rating)
    assert capped_multiplier == 2.8


def test_stage12_line_downgrade_preserves_parallel_equivalent_scaling() -> None:
    base_rating = np.sqrt(3.0) * 110.0 * 0.645
    branch = BranchElectricalParam(7, 1, 2, 110.0, 5.0, 0.5, 2.0, 10.0, 2.0 * base_rating, True)
    resized = _resize_branch_multiplier(branch, 1.0)
    assert np.isclose(resized.rate_mva, base_rating)
    assert resized.r_ohm == 1.0
    assert resized.x_ohm == 4.0
    assert resized.b_us == 5.0
