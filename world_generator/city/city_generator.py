from __future__ import annotations

import numpy as np

from world_generator.core.config import CityConfig, WorldGridConfig
from world_generator.core.datatypes import CityNode, CityState, ClimateBaseline, HydrologyState, StaticLandState, TerrainFeatures


def generate_initial_cities(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    climate: ClimateBaseline,
    grid: WorldGridConfig,
    config: CityConfig,
    rng: np.random.Generator,
) -> CityState:
    water = hydrology.river | hydrology.lake
    protected = land.protected.astype(bool)
    flood = np.clip(hydrology.flood_risk, 0.0, 1.0)
    buildability = np.clip(land.buildability, 0.0, 1.0)
    terrain_cost = np.clip(land.terrain_cost, 0.0, 1.0)
    slope_n = _normalize01(terrain.slope)
    roughness_n = _normalize01(terrain.roughness)

    waterfront_amenity = _waterfront_amenity(hydrology.distance_to_water, water, protected, flood, config)
    water_access = _gaussian_preference(
        hydrology.distance_to_water,
        config.water_access_optimal_km,
        config.water_access_sigma_km,
    )
    climate_comfort = _climate_comfort(climate)
    flatness = np.clip(1.0 - 0.72 * slope_n - 0.28 * roughness_n, 0.0, 1.0)

    city_suitability = (
        0.34 * buildability
        + 0.18 * flatness
        + 0.15 * climate_comfort
        + 0.13 * water_access
        + 0.12 * waterfront_amenity
        + 0.08 * (1.0 - terrain_cost)
        - 0.26 * flood
        - 0.42 * protected.astype(np.float32)
    )
    city_suitability = _normalize01(np.clip(city_suitability, 0.0, None))
    city_suitability[water] = 0.0
    city_suitability[protected] *= 0.05

    urban_core_suitability = city_suitability.copy()
    urban_core_suitability[hydrology.distance_to_water < config.core_water_min_distance_km] *= 0.30
    urban_core_suitability[flood > config.max_core_flood_risk] *= 0.15
    urban_core_suitability *= _edge_buffer(
        urban_core_suitability.shape,
        grid,
        config.edge_buffer_km,
        config.edge_buffer_min_factor,
    )
    urban_core_suitability[water | protected] = 0.0
    urban_core_suitability = _normalize01(urban_core_suitability)

    centers = _select_city_centers(urban_core_suitability, grid, config, rng)
    cities = _build_city_nodes(centers, urban_core_suitability, grid, config, rng)
    population_density, economic_activity, urban_density, city_id_map = _spread_city_fields(
        cities,
        buildability,
        waterfront_amenity,
        flood,
        water,
        protected,
        grid,
    )
    urban_mask = urban_density > 0.18

    return CityState(
        cities=tuple(cities),
        city_suitability=city_suitability.astype(np.float32),
        urban_core_suitability=urban_core_suitability.astype(np.float32),
        waterfront_amenity=waterfront_amenity.astype(np.float32),
        population_density=population_density.astype(np.float32),
        economic_activity=economic_activity.astype(np.float32),
        urban_density=urban_density.astype(np.float32),
        urban_mask=urban_mask,
        city_id_map=city_id_map.astype(np.int16),
    )


def _waterfront_amenity(
    distance_to_water: np.ndarray,
    water: np.ndarray,
    protected: np.ndarray,
    flood: np.ndarray,
    config: CityConfig,
) -> np.ndarray:
    amenity = _gaussian_preference(distance_to_water, config.waterfront_optimal_km, config.waterfront_sigma_km)
    amenity[water] = 0.0
    amenity[protected] *= 0.45
    amenity *= 1.0 - 0.55 * flood
    return _normalize01(np.clip(amenity, 0.0, None))


def _climate_comfort(climate: ClimateBaseline) -> np.ndarray:
    temp = _gaussian_preference(climate.mean_temperature, 18.0, 8.0)
    humidity = _gaussian_preference(climate.mean_humidity, 0.62, 0.24)
    rain = _gaussian_preference(climate.mean_precipitation, 850.0, 520.0)
    return _normalize01(0.52 * temp + 0.30 * humidity + 0.18 * rain)


def _select_city_centers(
    suitability: np.ndarray,
    grid: WorldGridConfig,
    config: CityConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    min_distance_cells = max(config.min_city_distance_km / max(grid.cell_size_km, 1e-6), 1.0)
    row_col = np.argwhere(suitability > np.quantile(suitability, 0.72))
    if row_col.size == 0:
        row_col = np.argwhere(suitability > 0.0)
    scores = suitability[row_col[:, 0], row_col[:, 1]]
    jitter = rng.uniform(0.92, 1.08, size=scores.shape)
    order = np.argsort(-(scores * jitter))
    centers: list[tuple[int, int]] = []
    for index in order:
        row, col = int(row_col[index, 0]), int(row_col[index, 1])
        if all(np.hypot(row - r0, col - c0) >= min_distance_cells for r0, c0 in centers):
            centers.append((row, col))
            if len(centers) >= config.city_count:
                break
    if len(centers) < config.city_count:
        fallback = np.argsort(-suitability.ravel())
        for flat_index in fallback:
            row, col = np.unravel_index(int(flat_index), suitability.shape)
            if suitability[row, col] <= 0.0:
                break
            if all(np.hypot(row - r0, col - c0) >= min_distance_cells * 0.72 for r0, c0 in centers):
                centers.append((int(row), int(col)))
                if len(centers) >= config.city_count:
                    break
    return centers


def _edge_buffer(
    shape: tuple[int, int],
    grid: WorldGridConfig,
    buffer_km: float,
    min_factor: float,
) -> np.ndarray:
    rows, cols = np.indices(shape)
    edge_cells = np.minimum.reduce([rows, cols, shape[0] - 1 - rows, shape[1] - 1 - cols]).astype(np.float32)
    buffer_cells = max(buffer_km / max(grid.cell_size_km, 1e-6), 1e-6)
    progress = np.clip(edge_cells / buffer_cells, 0.0, 1.0)
    floor = float(np.clip(min_factor, 0.0, 1.0))
    return (floor + (1.0 - floor) * progress).astype(np.float32)


def _build_city_nodes(
    centers: list[tuple[int, int]],
    suitability: np.ndarray,
    grid: WorldGridConfig,
    config: CityConfig,
    rng: np.random.Generator,
) -> list[CityNode]:
    if not centers:
        return []
    center_scores = np.asarray([suitability[row, col] for row, col in centers], dtype=np.float32)
    ranks = np.argsort(np.argsort(-center_scores))
    size_weights = (len(centers) - ranks).astype(np.float32) ** config.city_size_alpha
    size_weights *= rng.uniform(0.85, 1.18, size=size_weights.shape).astype(np.float32)
    populations = config.total_population * size_weights / max(float(size_weights.sum()), 1e-6)
    score_n = _normalize01(center_scores)
    cities = []
    for city_id, ((row, col), population, score) in enumerate(zip(centers, populations, score_n)):
        radius = config.urban_radius_min_km + (config.urban_radius_max_km - config.urban_radius_min_km) * np.sqrt(population / populations.max())
        cities.append(
            CityNode(
                city_id=city_id,
                row=int(row),
                col=int(col),
                x=float(col / max(grid.width - 1, 1)),
                y=float(row / max(grid.height - 1, 1)),
                population=float(population),
                radius_km=float(radius),
                suitability=float(suitability[row, col]),
            )
        )
    return cities


def _spread_city_fields(
    cities: list[CityNode],
    buildability: np.ndarray,
    waterfront_amenity: np.ndarray,
    flood: np.ndarray,
    water: np.ndarray,
    protected: np.ndarray,
    grid: WorldGridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices(buildability.shape)
    population = np.zeros(buildability.shape, dtype=np.float32)
    economy = np.zeros(buildability.shape, dtype=np.float32)
    density = np.zeros(buildability.shape, dtype=np.float32)
    city_id_map = np.full(buildability.shape, -1, dtype=np.int16)
    best_influence = np.zeros(buildability.shape, dtype=np.float32)
    developable = buildability * (1.0 - 0.75 * flood)
    developable[water | protected] = 0.0

    for city in cities:
        radius_cells = max(city.radius_km / max(grid.cell_size_km, 1e-6), 1.0)
        distance = np.hypot(rows - city.row, cols - city.col)
        core = np.exp(-(distance**2) / (2.0 * (0.38 * radius_cells) ** 2))
        halo = 0.42 * np.exp(-(distance**2) / (2.0 * radius_cells**2))
        influence = (core + halo) * developable
        population += city.population * influence
        economy += (0.55 + 0.45 * waterfront_amenity) * np.sqrt(max(city.population, 1.0)) * influence
        density = np.maximum(density, influence)
        update = influence > best_influence
        city_id_map[update] = city.city_id
        best_influence[update] = influence[update]

    if population.sum() > 0.0:
        population *= sum(city.population for city in cities) / float(population.sum())
    population = _normalize01(population)
    economy = _normalize01(economy)
    density = _normalize01(density)
    city_id_map[density <= 0.03] = -1
    return population, economy, density, city_id_map


def _gaussian_preference(values: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-((values - center) ** 2) / (2.0 * max(sigma, 1e-6) ** 2)).astype(np.float32)


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)
