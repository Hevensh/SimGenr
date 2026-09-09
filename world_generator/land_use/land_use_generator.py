from __future__ import annotations

import numpy as np

from world_generator.core.config import LandUseConfig, WorldGridConfig
from world_generator.core.datatypes import CityState, HydrologyState, LandUseState, StaticLandState, TerrainFeatures


LAND_USE_ZONE = {
    "background": 0,
    "residential": 1,
    "commercial": 2,
    "industrial": 3,
    "agriculture": 4,
    "park_green": 5,
    "protected": 6,
    "water": 7,
}


def generate_land_use_zones(
    terrain: TerrainFeatures,
    hydrology: HydrologyState,
    land: StaticLandState,
    city: CityState,
    grid: WorldGridConfig,
    config: LandUseConfig,
) -> LandUseState:
    del grid
    water = hydrology.river | hydrology.lake
    protected = land.protected.astype(bool)
    developable = np.clip(land.buildability, 0.0, 1.0)
    developable[water | protected] = 0.0
    flood_safe = 1.0 - np.clip(hydrology.flood_risk, 0.0, 1.0)
    flat = 1.0 - np.clip(terrain.slope / max(config.agriculture_slope_hard_limit, 1e-6), 0.0, 1.0)
    agriculture_slope_suitability = _slope_suitability(
        terrain.slope,
        config.agriculture_slope_soft_limit,
        config.agriculture_slope_hard_limit,
    )
    population_index = _normalize01(city.population_density)
    urban_edge = np.clip(city.urban_density - population_index, 0.0, 1.0)
    non_urban = np.clip(1.0 - city.urban_density, 0.0, 1.0)

    residential_base = _normalize01(city.population_density * (0.65 + 0.35 * developable) * flood_safe)
    residential_core_anchor = _normalize01(city.urban_density * developable * flood_safe)
    residential_load_anchor = _normalize01(
        (0.62 * city.economic_activity + 0.38 * city.urban_density)
        * developable
        * flood_safe
    )
    residual_weight = max(0.0, 1.0 - config.residential_core_weight - config.residential_load_anchor_weight)
    residential = _normalize01(
        residual_weight * residential_base
        + config.residential_core_weight * residential_core_anchor
        + config.residential_load_anchor_weight * residential_load_anchor
    )
    commercial = _normalize01(
        city.economic_activity
        * (0.45 + 0.55 * city.urban_density)
        * (0.78 + 0.22 * city.waterfront_amenity)
        * flood_safe
    )
    industrial = _normalize01(
        (config.industrial_edge_preference * urban_edge + (1.0 - config.industrial_edge_preference) * city.urban_density)
        * developable
        * flat
        * (1.0 - 0.65 * city.waterfront_amenity)
        * flood_safe
    )
    agriculture_base = (
        non_urban
        * flat
        * np.clip(land.vegetation + 0.22, 0.0, 1.0)
        * developable
        * (city.urban_density < config.agriculture_max_urban_density)
    )
    agriculture = _normalize01(agriculture_base) * agriculture_slope_suitability
    park_green = _normalize01(
        config.park_waterfront_weight * city.waterfront_amenity
        + 0.24 * land.vegetation
        + 0.16 * protected.astype(np.float32)
        + 0.10 * land.water_buffer.astype(np.float32)
    )
    residential[water | protected] = 0.0
    commercial[water | protected] = 0.0
    industrial[water | protected] = 0.0
    agriculture[water | protected] = 0.0
    park_green[water] = 0.0

    load_density = _normalize01(
        config.load_residential_weight * residential
        + config.load_commercial_weight * commercial
        + config.load_industrial_weight * industrial
    )
    load_density[water | protected] = 0.0

    land_use_zone = _classify_land_use(
        residential=residential,
        commercial=commercial,
        industrial=industrial,
        agriculture=agriculture,
        park_green=park_green,
        water=water,
        protected=protected,
    )
    return LandUseState(
        residential=residential.astype(np.float32),
        commercial=commercial.astype(np.float32),
        industrial=industrial.astype(np.float32),
        agriculture=agriculture.astype(np.float32),
        park_green=park_green.astype(np.float32),
        load_density_base=load_density.astype(np.float32),
        land_use_zone=land_use_zone.astype(np.int16),
    )


def _classify_land_use(
    residential: np.ndarray,
    commercial: np.ndarray,
    industrial: np.ndarray,
    agriculture: np.ndarray,
    park_green: np.ndarray,
    water: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    stack = np.stack([residential, commercial, industrial, agriculture, park_green], axis=0)
    labels = np.asarray(
        [
            LAND_USE_ZONE["residential"],
            LAND_USE_ZONE["commercial"],
            LAND_USE_ZONE["industrial"],
            LAND_USE_ZONE["agriculture"],
            LAND_USE_ZONE["park_green"],
        ],
        dtype=np.int16,
    )
    best = labels[np.argmax(stack, axis=0)]
    confidence = np.max(stack, axis=0)
    zone = np.where(confidence > 0.12, best, LAND_USE_ZONE["background"]).astype(np.int16)
    zone[protected] = LAND_USE_ZONE["protected"]
    zone[water] = LAND_USE_ZONE["water"]
    return zone


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.full_like(values, np.clip(vmax, 0.0, 1.0), dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _slope_suitability(slope: np.ndarray, soft_limit: float, hard_limit: float) -> np.ndarray:
    if hard_limit <= soft_limit:
        return (slope <= soft_limit).astype(np.float32)
    t = np.clip((slope - soft_limit) / (hard_limit - soft_limit), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return (1.0 - smooth).astype(np.float32)
