from __future__ import annotations

import numpy as np

from world_generator.core.config import WorldConfig
from world_generator.core.random_state import build_rng_registry
from world_generator.climate.climate_generator import generate_climate_baseline
from world_generator.city.city_generator import generate_initial_cities
from world_generator.energy.energy_candidate_generator import generate_energy_candidates
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.grid.node_builder import build_grid_nodes
from world_generator.grid.refinement_builder import refine_grid_topology
from world_generator.grid.topology_builder import build_grid_topology
from world_generator.hydrology.hydrology_generator import generate_hydrology
from world_generator.land.land_generator import generate_static_land
from world_generator.land_use.land_use_generator import LAND_USE_ZONE, generate_land_use_zones
from world_generator.terrain.derivatives import derive_terrain_features
from world_generator.terrain.terrain_generator import generate_terrain_base
from world_generator.weather.weather_generator import generate_daily_weather
from world_generator.weather.weather_generator import generate_hourly_weather_week
from world_generator.operation.source_load_forecast import generate_source_load_forecast


def test_static_terrain_is_reproducible() -> None:
    config = WorldConfig()
    rngs_a = build_rng_registry(config.seed)
    rngs_b = build_rng_registry(config.seed)

    terrain_a = generate_terrain_base(config.world, config.terrain, rngs_a.generator("terrain"))
    terrain_b = generate_terrain_base(config.world, config.terrain, rngs_b.generator("terrain"))

    np.testing.assert_array_equal(terrain_a.elevation, terrain_b.elevation)


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
    assert bus_kinds.count("thermal_bus") == config.power_grid.thermal_candidate_count
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


def test_refined_grid_topology_adds_transit_buses_and_limits_segment_length() -> None:
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
    refined = refine_grid_topology(grid_nodes, topology, config.world, config.power_grid)
    refined_bus_ids = {item.bus_id for item in refined.refined_buses}
    parent = {item: item for item in refined_bus_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    assert len(refined.refined_buses) > len(grid_nodes.buses)
    assert len(refined.refined_edges) >= len(topology.edges)
    assert (refined.transit_bus_map >= 0).any()
    for edge in refined.refined_edges:
        assert edge.length_km <= config.power_grid.max_line_segment_km + config.world.cell_size_km
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
    refined = refine_grid_topology(grid_nodes, topology, config.world, config.power_grid)
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
    refined = refine_grid_topology(grid_nodes, topology, config.world, config.power_grid)
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
