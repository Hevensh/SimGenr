from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_generator.core.datatypes import GridUpgradePlanStore, PowerFlowStore, SourceLoadForecastStore, WeatherStore
from world_generator.operation.grid_update_loop import GridUpdateLoopResult
from world_generator.visualization.city_figures import save_city_figures
from world_generator.visualization.climate_figures import save_climate_figures
from world_generator.visualization.electrical_figures import save_electrical_figures
from world_generator.visualization.energy_figures import save_energy_figures
from world_generator.visualization.grid_node_figures import save_grid_node_figures
from world_generator.visualization.grid_topology_figures import save_grid_topology_figures
from world_generator.visualization.grid_update_figures import save_grid_update_loop
from world_generator.visualization.hydrology_figures import save_hydrology_figures
from world_generator.visualization.land_use_figures import save_land_use_figures
from world_generator.visualization.operation_figures import save_operation_figures
from world_generator.visualization.power_flow_figures import save_power_flow_figures
from world_generator.visualization.static_land_figures import save_static_land_figures
from world_generator.visualization.terrain_figures import save_terrain_figures
from world_generator.visualization.upgrade_figures import save_upgrade_figures
from world_generator.visualization.weather_figures import save_weather_figures


def save_static_map_figures(
    static_maps: dict[str, np.ndarray],
    output_dir: str | Path,
    weather: WeatherStore | None = None,
    hourly_weather: WeatherStore | None = None,
    source_load_forecast: SourceLoadForecastStore | None = None,
    power_flow: PowerFlowStore | None = None,
    upgrade_plan: GridUpgradePlanStore | None = None,
    update_loop: GridUpdateLoopResult | None = None,
    render_weather_gif: bool = True,
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
        manifest["stage_05_weather"] = save_weather_figures(
            weather,
            output_dir / "stage_05_weather",
            hourly_weather,
            static_maps,
            render_hourly_gif=render_weather_gif,
        )

    if "city_suitability" in static_maps:
        manifest["stage_06_initial_cities"] = save_city_figures(static_maps, output_dir / "stage_06_initial_cities")

    if "land_use_zone" in static_maps:
        manifest["stage_07_land_use_load_zones"] = save_land_use_figures(static_maps, output_dir / "stage_07_land_use_load_zones")

    if "wind_suitability" in static_maps:
        manifest["stage_08_energy_load_candidates"] = save_energy_figures(static_maps, output_dir / "stage_08_energy_load_candidates")

    if "bus_site_map" in static_maps:
        manifest["stage_09_grid_bus_candidates"] = save_grid_node_figures(static_maps, output_dir / "stage_09_grid_bus_candidates")

    if "line_route_map" in static_maps:
        topology_dir = output_dir / "stage_10_grid_topology"
        manifest["stage_10_grid_topology"] = save_grid_topology_figures(static_maps, topology_dir)

    operation_files: list[str] = []
    operation_dir = output_dir / "stage_11_grid_operation"
    if "electrical_branches" in static_maps:
        operation_files += save_electrical_figures(static_maps, operation_dir)

    if hourly_weather is not None and source_load_forecast is not None:
        operation_files += save_operation_figures(source_load_forecast, operation_dir)

    if power_flow is not None:
        operation_files += save_power_flow_figures(static_maps, power_flow, operation_dir)

    if operation_files:
        manifest["stage_11_grid_operation"] = operation_files

    if update_loop is not None:
        manifest["stage_12_grid_update"] = save_grid_update_loop(
            static_maps,
            update_loop,
            output_dir / "stage_12_grid_update",
            hourly_weather,
            render_gif=render_weather_gif,
        )
    elif upgrade_plan is not None:
        manifest["stage_12_grid_update"] = save_upgrade_figures(static_maps, upgrade_plan, output_dir / "stage_12_grid_update")

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
