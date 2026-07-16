from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.core.config import dump_config_snapshot, load_world_config
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
from world_generator.land_use.land_use_generator import generate_land_use_zones
from world_generator.operation.source_load_forecast import generate_source_load_forecast
from world_generator.terrain.derivatives import derive_terrain_features
from world_generator.terrain.terrain_generator import generate_terrain_base
from world_generator.visualization.map_plot import save_static_map_figures
from world_generator.weather.weather_generator import generate_daily_weather, generate_hourly_weather_week


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible static terrain world.")
    parser.add_argument("--config", default="configs/small_debug.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the seed from the config file.")
    args = parser.parse_args()

    config = load_world_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    rngs = build_rng_registry(config.seed)
    output_root = Path(args.output or config.output.root)
    world_id = f"{config.output.world_name}_seed{config.seed}"
    output_dir = output_root / world_id
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    terrain_base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain_features = derive_terrain_features(terrain_base, config.world)
    hydrology = generate_hydrology(terrain_features, config.world, config.hydrology)
    land = generate_static_land(
        terrain_features,
        hydrology,
        config.world,
        config.land,
        rngs.generator("city"),
    )
    climate = generate_climate_baseline(
        terrain_features,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("weather"),
    )
    operation_rng = rngs.generator("operation")
    weather = generate_daily_weather(
        terrain_features,
        hydrology,
        climate,
        config.world,
        config.weather,
        operation_rng,
    )
    city = generate_initial_cities(
        terrain_features,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    land_use = generate_land_use_zones(
        terrain_features,
        hydrology,
        land,
        city,
        config.world,
        config.land_use,
    )
    climate_maps = climate.as_maps()
    energy = generate_energy_candidates(
        terrain_features,
        hydrology,
        land,
        city,
        land_use,
        climate_maps,
        config.world,
        config.energy,
    )
    grid_nodes = build_grid_nodes(
        terrain_features,
        hydrology,
        land,
        land_use,
        energy,
        config.world,
        config.power_grid,
    )
    grid_topology = build_grid_topology(
        terrain_features,
        hydrology,
        land,
        grid_nodes,
        config.world,
        config.power_grid,
    )
    refined_topology = refine_grid_topology(
        grid_nodes,
        grid_topology,
        config.world,
        config.power_grid,
    )
    grid_electrical = build_grid_electrical(refined_topology)
    hourly_weather = generate_hourly_weather_week(weather, config.weather, operation_rng)
    source_load_forecast = generate_source_load_forecast(
        hourly_weather,
        refined_topology,
        grid_electrical,
        operation_rng,
    )
    static_maps = (
        terrain_features.as_maps()
        | hydrology.as_maps()
        | land.as_maps()
        | climate_maps
        | city.as_maps()
        | land_use.as_maps()
        | energy.as_maps()
        | grid_nodes.as_maps()
        | grid_topology.as_maps()
        | refined_topology.as_maps()
        | grid_nodes.buses_as_arrays()
        | grid_topology.edges_as_arrays()
        | refined_topology.as_arrays()
        | grid_electrical.as_arrays()
    )

    np.savez_compressed(output_dir / "static_maps.npz", **static_maps)
    np.savez_compressed(output_dir / "daily_weather.npz", **weather.as_arrays())
    np.savez_compressed(output_dir / "hourly_weather_week.npz", **hourly_weather.as_arrays())
    np.savez_compressed(output_dir / "source_load_forecast.npz", **source_load_forecast.as_arrays())
    np.savez_compressed(output_dir / "source_load_candidates.npz", **energy.candidates_as_arrays())
    np.savez_compressed(output_dir / "grid_nodes.npz", **grid_nodes.buses_as_arrays())
    np.savez_compressed(output_dir / "grid_topology.npz", **grid_topology.edges_as_arrays())
    np.savez_compressed(output_dir / "refined_grid_topology.npz", **refined_topology.as_arrays())
    np.savez_compressed(output_dir / "grid_electrical.npz", **grid_electrical.as_arrays())
    dump_config_snapshot(config, output_dir / "config_snapshot.yaml")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "world_id": world_id,
                "seed": config.seed,
                "module_seeds": rngs.module_seeds,
                "static_maps": {name: list(value.shape) for name, value in static_maps.items()},
                "daily_weather": {
                    "dynamic": list(weather.dynamic.shape),
                    "weather_class": list(weather.weather_class.shape),
                    "channels": list(weather.channel_names),
                },
                "cities": city.cities_as_dicts(),
                "energy_candidates": energy.candidates_as_dicts(),
                "grid_buses": grid_nodes.buses_as_dicts(),
                "grid_edges": grid_topology.edges_as_dicts(),
                "refined_grid_buses": refined_topology.buses_as_dicts(),
                "refined_grid_edges": refined_topology.edges_as_dicts(),
                "grid_electrical": grid_electrical.as_dicts(),
                "hourly_weather_week": {
                    "dynamic": list(hourly_weather.dynamic.shape),
                    "weather_class": list(hourly_weather.weather_class.shape),
                    "channels": list(hourly_weather.channel_names),
                    "time_unit": hourly_weather.time_unit,
                    "start_day_of_year": hourly_weather.start_day_of_year,
                },
                "source_load_forecast": source_load_forecast.summary_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "energy_sites.json").write_text(
        json.dumps(energy.candidates_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "bus_sites.json").write_text(
        json.dumps(grid_nodes.buses_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "grid_edges.json").write_text(
        json.dumps(grid_topology.edges_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "refined_bus_sites.json").write_text(
        json.dumps(refined_topology.buses_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "refined_grid_edges.json").write_text(
        json.dumps(refined_topology.edges_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "grid_electrical.json").write_text(
        json.dumps(grid_electrical.as_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "source_load_forecast.json").write_text(
        json.dumps(source_load_forecast.summary_dict(), indent=2),
        encoding="utf-8",
    )
    save_static_map_figures(static_maps, figure_dir, weather, hourly_weather, source_load_forecast)

    print(f"Generated static world: {output_dir}")
    print(f"Saved maps: {output_dir / 'static_maps.npz'}")
    print(f"Saved daily weather: {output_dir / 'daily_weather.npz'}")
    print(f"Saved hourly weather week: {output_dir / 'hourly_weather_week.npz'}")
    print(f"Saved source/load forecast: {output_dir / 'source_load_forecast.npz'}")
    print(f"Saved source/load candidates: {output_dir / 'source_load_candidates.npz'}")
    print(f"Saved energy candidates: {output_dir / 'energy_sites.json'}")
    print(f"Saved grid buses: {output_dir / 'bus_sites.json'}")
    print(f"Saved grid edges: {output_dir / 'grid_edges.json'}")
    print(f"Saved refined grid topology: {output_dir / 'refined_grid_topology.npz'}")
    print(f"Saved grid electrical parameters: {output_dir / 'grid_electrical.npz'}")
    print(f"Saved figures: {figure_dir}")


if __name__ == "__main__":
    main()
