from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_generator.core.datatypes import WeatherStore
from world_generator.visualization.city_figures import save_city_figures
from world_generator.visualization.climate_figures import save_climate_figures
from world_generator.visualization.energy_figures import save_energy_figures
from world_generator.visualization.grid_node_figures import save_grid_node_figures
from world_generator.visualization.grid_topology_figures import save_grid_topology_figures
from world_generator.visualization.hydrology_figures import save_hydrology_figures
from world_generator.visualization.land_use_figures import save_land_use_figures
from world_generator.visualization.refined_topology_figures import save_refined_topology_figures
from world_generator.visualization.static_land_figures import save_static_land_figures
from world_generator.visualization.terrain_figures import save_terrain_figures
from world_generator.visualization.weather_figures import save_weather_figures


def save_static_map_figures(
    static_maps: dict[str, np.ndarray],
    output_dir: str | Path,
    weather: WeatherStore | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[str]] = {}
    manifest["stage_01_terrain"] = save_terrain_figures(static_maps, output_dir / "stage_01_terrain")

    if "flow_accumulation" in static_maps:
        manifest["stage_02_hydrology"] = save_hydrology_figures(static_maps, output_dir / "stage_02_hydrology")

    if "land_cover" in static_maps:
        manifest["stage_03_static_land"] = save_static_land_figures(static_maps, output_dir / "stage_03_static_land")

    if "mean_temperature" in static_maps:
        manifest["stage_04_climate_background"] = save_climate_figures(static_maps, output_dir / "stage_04_climate_background")

    if weather is not None:
        manifest["stage_05_daily_weather"] = save_weather_figures(weather, output_dir / "stage_05_daily_weather")

    if "city_suitability" in static_maps:
        manifest["stage_06_initial_cities"] = save_city_figures(static_maps, output_dir / "stage_06_initial_cities")

    if "land_use_zone" in static_maps:
        manifest["stage_07_land_use_load_zones"] = save_land_use_figures(static_maps, output_dir / "stage_07_land_use_load_zones")

    if "wind_suitability" in static_maps:
        manifest["stage_08_energy_load_candidates"] = save_energy_figures(static_maps, output_dir / "stage_08_energy_load_candidates")

    if "bus_site_map" in static_maps:
        manifest["stage_09_grid_bus_candidates"] = save_grid_node_figures(static_maps, output_dir / "stage_09_grid_bus_candidates")

    if "line_route_map" in static_maps:
        manifest["stage_10_grid_topology"] = save_grid_topology_figures(static_maps, output_dir / "stage_10_grid_topology")

    if "refined_line_route_map" in static_maps:
        manifest["stage_11_transit_buses"] = save_refined_topology_figures(static_maps, output_dir / "stage_11_transit_buses")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
