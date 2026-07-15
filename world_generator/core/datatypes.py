from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FloatMap = np.ndarray
BoolMap = np.ndarray


@dataclass(frozen=True)
class TerrainBase:
    elevation: FloatMap
    land_mask: BoolMap


@dataclass(frozen=True)
class TerrainFeatures:
    elevation: FloatMap
    slope: FloatMap
    aspect_sin: FloatMap
    aspect_cos: FloatMap
    roughness: FloatMap
    curvature: FloatMap

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "elevation": self.elevation,
            "slope": self.slope,
            "aspect_sin": self.aspect_sin,
            "aspect_cos": self.aspect_cos,
            "roughness": self.roughness,
            "curvature": self.curvature,
        }


@dataclass(frozen=True)
class HydrologyState:
    flow_direction: np.ndarray
    flow_accumulation: FloatMap
    river_centerline: BoolMap
    river: BoolMap
    lake: BoolMap
    water_depth: FloatMap
    hydrology_elevation: FloatMap
    watershed_id: np.ndarray
    distance_to_water: FloatMap
    flood_risk: FloatMap

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "flow_direction": self.flow_direction,
            "flow_accumulation": self.flow_accumulation,
            "river_centerline": self.river_centerline,
            "river": self.river,
            "lake": self.lake,
            "water_depth": self.water_depth,
            "hydrology_elevation": self.hydrology_elevation,
            "watershed_id": self.watershed_id,
            "distance_to_water": self.distance_to_water,
            "flood_risk": self.flood_risk,
        }


@dataclass(frozen=True)
class StaticLandState:
    land_cover: np.ndarray
    vegetation: FloatMap
    protected: BoolMap
    buildability: FloatMap
    terrain_cost: FloatMap
    water_buffer: BoolMap

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "land_cover": self.land_cover,
            "vegetation": self.vegetation,
            "protected": self.protected,
            "buildability": self.buildability,
            "terrain_cost": self.terrain_cost,
            "water_buffer": self.water_buffer,
        }


@dataclass(frozen=True)
class ClimateBaseline:
    mean_temperature: FloatMap
    annual_temperature_amplitude: FloatMap
    mean_humidity: FloatMap
    prevailing_wind_u: FloatMap
    prevailing_wind_v: FloatMap
    mean_precipitation: FloatMap
    mean_cloud: FloatMap
    mean_irradiance: FloatMap

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "mean_temperature": self.mean_temperature,
            "annual_temperature_amplitude": self.annual_temperature_amplitude,
            "mean_humidity": self.mean_humidity,
            "prevailing_wind_u": self.prevailing_wind_u,
            "prevailing_wind_v": self.prevailing_wind_v,
            "mean_precipitation": self.mean_precipitation,
            "mean_cloud": self.mean_cloud,
            "mean_irradiance": self.mean_irradiance,
        }


@dataclass(frozen=True)
class WeatherStore:
    dynamic: np.ndarray
    weather_class: np.ndarray
    timestamps: np.ndarray
    channel_names: tuple[str, ...]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "dynamic": self.dynamic,
            "weather_class": self.weather_class,
            "timestamps": self.timestamps,
            "channel_names": np.asarray(self.channel_names),
        }


@dataclass(frozen=True)
class CityNode:
    city_id: int
    row: int
    col: int
    x: float
    y: float
    population: float
    radius_km: float
    suitability: float


@dataclass(frozen=True)
class CityState:
    cities: tuple[CityNode, ...]
    city_suitability: FloatMap
    urban_core_suitability: FloatMap
    waterfront_amenity: FloatMap
    population_density: FloatMap
    economic_activity: FloatMap
    urban_density: FloatMap
    urban_mask: BoolMap
    city_id_map: np.ndarray

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "city_suitability": self.city_suitability,
            "urban_core_suitability": self.urban_core_suitability,
            "waterfront_amenity": self.waterfront_amenity,
            "population_density": self.population_density,
            "economic_activity": self.economic_activity,
            "urban_density": self.urban_density,
            "urban_mask": self.urban_mask,
            "city_id_map": self.city_id_map,
        }

    def cities_as_dicts(self) -> list[dict[str, float | int]]:
        return [
            {
                "city_id": city.city_id,
                "row": city.row,
                "col": city.col,
                "x": city.x,
                "y": city.y,
                "population": city.population,
                "radius_km": city.radius_km,
                "suitability": city.suitability,
            }
            for city in self.cities
        ]


@dataclass(frozen=True)
class LandUseState:
    residential: FloatMap
    commercial: FloatMap
    industrial: FloatMap
    agriculture: FloatMap
    park_green: FloatMap
    load_density_base: FloatMap
    land_use_zone: np.ndarray

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "residential": self.residential,
            "commercial": self.commercial,
            "industrial": self.industrial,
            "agriculture": self.agriculture,
            "park_green": self.park_green,
            "load_density_base": self.load_density_base,
            "land_use_zone": self.land_use_zone,
        }


@dataclass(frozen=True)
class EnergyCandidate:
    candidate_id: int
    kind: str
    row: int
    col: int
    x: float
    y: float
    capacity_mw: float
    suitability: float


@dataclass(frozen=True)
class EnergyCandidateState:
    wind_suitability: FloatMap
    pv_suitability: FloatMap
    load_node_density: FloatMap
    wind_candidate_map: np.ndarray
    pv_candidate_map: np.ndarray
    source_candidate_map: np.ndarray
    load_candidate_map: np.ndarray
    wind_candidates: tuple[EnergyCandidate, ...]
    pv_candidates: tuple[EnergyCandidate, ...]
    load_candidates: tuple[EnergyCandidate, ...]

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "wind_suitability": self.wind_suitability,
            "pv_suitability": self.pv_suitability,
            "load_node_density": self.load_node_density,
            "wind_candidate_map": self.wind_candidate_map,
            "pv_candidate_map": self.pv_candidate_map,
            "source_candidate_map": self.source_candidate_map,
            "load_candidate_map": self.load_candidate_map,
        }

    def candidates_as_dicts(self) -> dict[str, list[dict[str, float | int | str]]]:
        return {
            "wind": [_candidate_as_dict(item) for item in self.wind_candidates],
            "pv": [_candidate_as_dict(item) for item in self.pv_candidates],
            "load": [_candidate_as_dict(item) for item in self.load_candidates],
        }

    def candidates_as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "wind_candidates": _candidates_to_array(self.wind_candidates),
            "pv_candidates": _candidates_to_array(self.pv_candidates),
            "load_candidates": _candidates_to_array(self.load_candidates),
        }


def _candidate_as_dict(candidate: EnergyCandidate) -> dict[str, float | int | str]:
    return {
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "row": candidate.row,
        "col": candidate.col,
        "x": candidate.x,
        "y": candidate.y,
        "capacity_mw": candidate.capacity_mw,
        "suitability": candidate.suitability,
    }


def _candidates_to_array(candidates: tuple[EnergyCandidate, ...]) -> np.ndarray:
    rows = [
        [
            item.candidate_id,
            item.row,
            item.col,
            item.x,
            item.y,
            item.capacity_mw,
            item.suitability,
        ]
        for item in candidates
    ]
    return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class GridBus:
    bus_id: int
    kind: str
    row: int
    col: int
    x: float
    y: float
    capacity_mw: float
    suitability: float
    externality_score: float
    source_kind: str
    source_id: int


@dataclass(frozen=True)
class GridNodeState:
    thermal_suitability: FloatMap
    thermal_externality: FloatMap
    bus_site_map: np.ndarray
    load_bus_map: np.ndarray
    source_bus_map: np.ndarray
    thermal_bus_map: np.ndarray
    buses: tuple[GridBus, ...]

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "thermal_suitability": self.thermal_suitability,
            "thermal_externality": self.thermal_externality,
            "bus_site_map": self.bus_site_map,
            "load_bus_map": self.load_bus_map,
            "source_bus_map": self.source_bus_map,
            "thermal_bus_map": self.thermal_bus_map,
        }

    def buses_as_dicts(self) -> list[dict[str, float | int | str]]:
        return [_bus_as_dict(item) for item in self.buses]

    def buses_as_arrays(self) -> dict[str, np.ndarray]:
        return {"grid_buses": _buses_to_array(self.buses)}


def _bus_as_dict(bus: GridBus) -> dict[str, float | int | str]:
    return {
        "bus_id": bus.bus_id,
        "kind": bus.kind,
        "row": bus.row,
        "col": bus.col,
        "x": bus.x,
        "y": bus.y,
        "capacity_mw": bus.capacity_mw,
        "suitability": bus.suitability,
        "externality_score": bus.externality_score,
        "source_kind": bus.source_kind,
        "source_id": bus.source_id,
    }


def _buses_to_array(buses: tuple[GridBus, ...]) -> np.ndarray:
    rows = [
        [
            item.bus_id,
            item.row,
            item.col,
            item.x,
            item.y,
            item.capacity_mw,
            item.suitability,
            item.externality_score,
            item.source_id,
        ]
        for item in buses
    ]
    return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class GridEdge:
    edge_id: int
    from_bus: int
    to_bus: int
    length_km: float
    route_cost: float
    is_redundant: bool
    path_rows: tuple[int, ...]
    path_cols: tuple[int, ...]


@dataclass(frozen=True)
class GridTopologyState:
    routing_cost: FloatMap
    line_route_map: FloatMap
    grid_edge_map: np.ndarray
    edges: tuple[GridEdge, ...]

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "routing_cost": self.routing_cost,
            "line_route_map": self.line_route_map,
            "grid_edge_map": self.grid_edge_map,
        }

    def edges_as_dicts(self) -> list[dict[str, float | int | bool | list[int]]]:
        return [_edge_as_dict(item) for item in self.edges]

    def edges_as_arrays(self) -> dict[str, np.ndarray]:
        return {"grid_edges": _edges_to_array(self.edges)}


@dataclass(frozen=True)
class RefinedGridTopologyState:
    refined_line_route_map: FloatMap
    refined_grid_edge_map: np.ndarray
    transit_bus_map: np.ndarray
    refined_buses: tuple[GridBus, ...]
    refined_edges: tuple[GridEdge, ...]

    def as_maps(self) -> dict[str, np.ndarray]:
        return {
            "refined_line_route_map": self.refined_line_route_map,
            "refined_grid_edge_map": self.refined_grid_edge_map,
            "transit_bus_map": self.transit_bus_map,
        }

    def buses_as_dicts(self) -> list[dict[str, float | int | str]]:
        return [_bus_as_dict(item) for item in self.refined_buses]

    def edges_as_dicts(self) -> list[dict[str, float | int | bool | list[int]]]:
        return [_edge_as_dict(item) for item in self.refined_edges]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "refined_grid_buses": _buses_to_array(self.refined_buses),
            "refined_grid_edges": _edges_to_array(self.refined_edges),
        }


def _edge_as_dict(edge: GridEdge) -> dict[str, float | int | bool | list[int]]:
    return {
        "edge_id": edge.edge_id,
        "from_bus": edge.from_bus,
        "to_bus": edge.to_bus,
        "length_km": edge.length_km,
        "route_cost": edge.route_cost,
        "is_redundant": edge.is_redundant,
        "path_rows": list(edge.path_rows),
        "path_cols": list(edge.path_cols),
    }


def _edges_to_array(edges: tuple[GridEdge, ...]) -> np.ndarray:
    rows = [
        [
            item.edge_id,
            item.from_bus,
            item.to_bus,
            item.length_km,
            item.route_cost,
            float(item.is_redundant),
        ]
        for item in edges
    ]
    return np.asarray(rows, dtype=np.float32)
