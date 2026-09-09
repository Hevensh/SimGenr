from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.core.config import dump_config_snapshot, load_world_config
from world_generator.core.contracts import GENERATOR_VERSION, field_contract_document
from world_generator.core.output_layout import WorldDataLayout
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
from world_generator.operation.grid_upgrade import build_grid_upgrade_plan
from world_generator.operation.grid_update_loop import run_grid_update_loop
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.source_load_forecast import generate_source_load_forecast
from world_generator.operation.stage_cache import (
    load_hourly_weather_checkpoint,
    load_stage12_checkpoint,
    load_stage13_checkpoint,
    save_stage12_checkpoint,
)
from world_generator.operation.storage_dispatch import dispatch_storage_week
from world_generator.operation.storage_planning import analyze_storage_need, plan_storage_sites
from world_generator.terrain.derivatives import derive_terrain_features
from world_generator.terrain.terrain_generator import generate_terrain_base
from world_generator.visualization.map_plot import save_static_map_figures
from world_generator.visualization.storage_figures import save_storage_dispatch_figures, save_storage_need_figures
from world_generator.weather.weather_generator import generate_daily_weather, generate_hourly_weather_week


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible static terrain world.")
    parser.add_argument("--config", default="configs/small_debug.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the seed from the config file.")
    parser.add_argument("--start-day", type=int, default=None, help="Zero-based first day of the 365-day climatology (0..364).")
    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        help="Start from a cached stage. Supports 1 (full), 13 (storage planning), and 14 (storage dispatch).",
    )
    parser.add_argument(
        "--skip-weather-animation",
        "--skip-weather-gif",
        dest="skip_weather_animation",
        action="store_true",
        help="Skip hourly weather and line-loading animations.",
    )
    parser.add_argument(
        "--skip-storage-animation",
        "--skip-storage-gif",
        dest="skip_storage_animation",
        action="store_true",
        help="Skip the Stage 14 storage dispatch animation.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Generate machine-readable checkpoints only; skip all PNG/WebP output.",
    )
    args = parser.parse_args()

    config = load_world_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.start_day is not None:
        if not 0 <= args.start_day < 365:
            parser.error("--start-day must be in 0..364")
        config = replace(config, weather=replace(config.weather, start_day_of_year=args.start_day))
    rngs = build_rng_registry(config.seed)
    output_root = Path(args.output or config.output.root)
    world_id = f"{config.output.world_name}_seed{config.seed}"
    output_dir = output_root / world_id
    data_dir = output_dir / "data"
    data_layout = WorldDataLayout(data_dir)
    figure_dir = output_dir / "figures"
    if args.from_stage in (13, 14):
        _validate_cached_physics(output_dir, config, args.from_stage)
    data_layout.create()
    if args.from_stage == 13:
        _run_stage13_from_cache(
            config,
            output_dir,
            data_dir,
            figure_dir,
            render_storage_animation=not args.skip_storage_animation,
            render_figures=not args.no_figures,
        )
        return
    if args.from_stage == 14:
        _run_stage14_from_cache(
            config,
            output_dir,
            data_dir,
            figure_dir,
            render_storage_animation=not args.skip_storage_animation,
            render_figures=not args.no_figures,
        )
        return
    if args.from_stage != 1:
        parser.error("--from-stage currently supports only 1, 13, or 14")

    progress = tqdm(total=15, unit="stage", dynamic_ncols=True)
    progress.set_description("Stage 01 terrain")
    terrain_base = generate_terrain_base(config.world, config.terrain, rngs.generator("terrain"))
    terrain_features = derive_terrain_features(terrain_base, config.world)
    progress.update()
    progress.set_description("Stage 02 hydrology")
    hydrology = generate_hydrology(terrain_features, config.world, config.hydrology)
    progress.update()
    progress.set_description("Stage 04 climate")
    climate = generate_climate_baseline(
        terrain_features,
        hydrology,
        config.world,
        config.climate,
        rngs.generator("climate"),
    )
    progress.update()
    progress.set_description("Stage 03 static land (climate conditioned)")
    land = generate_static_land(
        terrain_features,
        hydrology,
        config.world,
        config.land,
        rngs.generator("land"),
        climate=climate,
    )
    progress.update()
    progress.set_description("Stage 05 weather")
    weather = generate_daily_weather(
        terrain_features,
        hydrology,
        climate,
        config.world,
        config.weather,
        rngs.generator("weather"),
    )
    hourly_weather = generate_hourly_weather_week(
        weather,
        config.weather,
        rngs.generator("weather_hourly"),
        grid=config.world,
    )
    progress.update()
    progress.set_description("Stage 06 cities")
    city = generate_initial_cities(
        terrain_features,
        hydrology,
        land,
        climate,
        config.world,
        config.city,
        rngs.generator("evolution"),
    )
    progress.update()
    progress.set_description("Stage 07 land use")
    land_use = generate_land_use_zones(
        terrain_features,
        hydrology,
        land,
        city,
        config.world,
        config.land_use,
    )
    progress.update()
    progress.set_description("Stage 08 energy sites")
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
    progress.update()
    progress.set_description("Stage 09 grid buses")
    grid_nodes = build_grid_nodes(
        terrain_features,
        hydrology,
        land,
        land_use,
        energy,
        config.world,
        config.power_grid,
    )
    progress.update()
    progress.set_description("Stage 10 topology")
    grid_topology = build_grid_topology(
        terrain_features,
        hydrology,
        land,
        grid_nodes,
        config.world,
        config.power_grid,
    )
    refined_topology = refine_grid_topology(
        terrain_features,
        hydrology,
        land,
        grid_nodes,
        grid_topology,
        config.world,
        config.power_grid,
    )
    grid_electrical = build_grid_electrical(refined_topology, config.power_grid.nominal_voltage_kv)
    progress.update()
    progress.set_description("Stage 11 operation")
    source_load_forecast = generate_source_load_forecast(
        hourly_weather,
        refined_topology,
        grid_electrical,
        rngs.generator("source_load"),
        config=config.source_load,
        land_use=land_use,
        grid=config.world,
    )
    power_flow = solve_dc_power_flow(
        source_load_forecast,
        refined_topology,
        grid_electrical,
    )
    upgrade_plan = build_grid_upgrade_plan(
        power_flow,
        grid_electrical,
        power_grid=config.power_grid,
    )
    progress.update()
    progress.set_description("Stage 12 grid update")
    update_loop = run_grid_update_loop(
        source_load_forecast,
        refined_topology,
        grid_topology,
        grid_electrical,
        terrain_features,
        hydrology,
        land,
        config.world,
        config.power_grid,
    )
    progress.update()
    progress.set_description("Stage 13 storage need")
    if update_loop.iterations:
        final_iteration = update_loop.iterations[-1]
        storage_need = analyze_storage_need(
            final_iteration.refined_topology,
            final_iteration.electrical,
            final_iteration.power_flow,
            config.world,
            config.storage,
        )
    else:
        storage_need = analyze_storage_need(
            refined_topology,
            grid_electrical,
            power_flow,
            config.world,
            config.storage,
        )
    storage_plan = plan_storage_sites(
        final_iteration.refined_topology if update_loop.iterations else refined_topology,
        storage_need,
        config.storage,
    )
    progress.update()
    progress.set_description("Stage 14 storage dispatch")
    final_topology = final_iteration.refined_topology if update_loop.iterations else refined_topology
    final_electrical = final_iteration.electrical if update_loop.iterations else grid_electrical
    final_power_flow = final_iteration.power_flow if update_loop.iterations else power_flow
    storage_dispatch, storage_dispatch_forecast, storage_dispatch_power_flow, storage_dispatch_electrical = dispatch_storage_week(
        final_topology,
        final_electrical,
        final_power_flow,
        storage_plan,
        config.storage,
    )
    progress.update()
    progress.set_description("Writing outputs")
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

    np.savez_compressed(data_layout.topology / "static_maps.npz", **static_maps)
    np.savez_compressed(data_layout.weather / "daily_weather.npz", **weather.as_arrays())
    np.savez_compressed(data_layout.weather / "hourly_weather_week.npz", **hourly_weather.as_arrays())
    np.savez_compressed(data_layout.operation / "source_load_forecast.npz", **source_load_forecast.as_arrays())
    np.savez_compressed(data_layout.operation / "power_flow_hourly.npz", **power_flow.as_arrays())
    np.savez_compressed(data_layout.operation / "grid_upgrade_plan.npz", **upgrade_plan.as_arrays())
    np.savez_compressed(data_layout.energy / "source_load_candidates.npz", **energy.candidates_as_arrays())
    np.savez_compressed(data_layout.buses / "grid_nodes.npz", **grid_nodes.buses_as_arrays())
    np.savez_compressed(data_layout.topology / "grid_topology.npz", **grid_topology.edges_as_arrays())
    np.savez_compressed(data_layout.topology / "refined_grid_topology.npz", **refined_topology.as_arrays())
    np.savez_compressed(data_layout.topology / "grid_electrical.npz", **grid_electrical.as_arrays())
    np.savez_compressed(data_layout.storage_planning / "storage_need.npz", **storage_need.as_arrays())
    np.savez_compressed(data_layout.storage_planning / "storage_plan.npz", **storage_plan.as_arrays())
    np.savez_compressed(data_layout.storage_dispatch / "storage_dispatch.npz", **storage_dispatch.as_arrays())
    np.savez_compressed(
        data_layout.storage_dispatch / "storage_dispatch_forecast.npz", **storage_dispatch_forecast.as_arrays()
    )
    np.savez_compressed(
        data_layout.storage_dispatch / "storage_dispatch_power_flow.npz", **storage_dispatch_power_flow.as_arrays()
    )
    np.savez_compressed(
        data_layout.storage_dispatch / "storage_dispatch_electrical.npz", **storage_dispatch_electrical.as_arrays()
    )
    if update_loop.iterations:
        save_stage12_checkpoint(update_loop.iterations[-1], data_layout.grid_update)
    dump_config_snapshot(config, data_layout.config_snapshot)
    if config.contracts.export_field_contracts:
        (data_dir / "field_contracts.json").write_text(
            json.dumps(field_contract_document(config.contracts.time_step_hours), indent=2), encoding="utf-8"
        )
    data_layout.metadata.write_text(
        json.dumps(
            {
                "world_id": world_id,
                "generator_version": GENERATOR_VERSION,
                "field_contracts": "field_contracts.json" if config.contracts.export_field_contracts else None,
                "time_step_hours": config.contracts.time_step_hours,
                "execution_stage_order": [1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                "scenario_semantics": "synthetic_realization_with_perfect_foresight_planning",
                "grid_model": "single_voltage_lossless_DC_transmission_equivalent",
                "stage14_line_expansion": "thermal_rerating_fixed_impedance",
                "parameter_status": "uncalibrated_scenario_priors_except_documented_physical_constants_and_line_templates",
                "time_convention": "local_solar_time_365_day_climatology",
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
                "power_flow": power_flow.summary_dict(),
                "grid_upgrade_plan": upgrade_plan.summary_dict(),
                "grid_update_loop": update_loop.summary_dict(),
                "storage_need": storage_need.summary_dict(),
                "storage_plan": storage_plan.summary_dict(),
                "storage_dispatch": storage_dispatch.summary_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_layout.energy / "energy_sites.json").write_text(
        json.dumps(energy.candidates_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (data_layout.buses / "bus_sites.json").write_text(
        json.dumps(grid_nodes.buses_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (data_layout.topology / "grid_edges.json").write_text(
        json.dumps(grid_topology.edges_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (data_layout.topology / "refined_bus_sites.json").write_text(
        json.dumps(refined_topology.buses_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (data_layout.topology / "refined_grid_edges.json").write_text(
        json.dumps(refined_topology.edges_as_dicts(), indent=2),
        encoding="utf-8",
    )
    (data_layout.topology / "grid_electrical.json").write_text(
        json.dumps(grid_electrical.as_dicts(), indent=2),
        encoding="utf-8",
    )
    (data_layout.operation / "source_load_forecast.json").write_text(
        json.dumps(source_load_forecast.summary_dict(), indent=2),
        encoding="utf-8",
    )
    (data_layout.operation / "power_flow_hourly.json").write_text(
        json.dumps(power_flow.summary_dict(), indent=2),
        encoding="utf-8",
    )
    (data_layout.operation / "grid_upgrade_plan.json").write_text(
        json.dumps(
            {
                "summary": upgrade_plan.summary_dict(),
                "branches": upgrade_plan.as_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_layout.grid_update / "grid_update_loop.json").write_text(
        json.dumps(update_loop.summary_dict(), indent=2),
        encoding="utf-8",
    )
    (data_layout.storage_planning / "storage_need.json").write_text(
        json.dumps(
            {
                "summary": storage_need.summary_dict(),
                "load_regions": storage_need.as_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_layout.storage_planning / "storage_plan.json").write_text(
        json.dumps(
            {
                "summary": storage_plan.summary_dict(),
                "sites": storage_plan.as_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_layout.storage_dispatch / "storage_dispatch.json").write_text(
        json.dumps(
            {
                "summary": storage_dispatch.summary_dict(),
                "capacity_expansion": storage_dispatch.expansion_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_figures:
        figure_maps = dict(static_maps)
        figure_maps["grid_edge_paths"] = {
            int(edge.edge_id): (edge.path_rows, edge.path_cols) for edge in grid_topology.edges
        }
        figure_maps["refined_grid_edge_paths"] = {
            int(edge.edge_id): (edge.path_rows, edge.path_cols) for edge in refined_topology.refined_edges
        }
        save_static_map_figures(
            figure_maps,
            figure_dir,
            weather,
            hourly_weather,
            source_load_forecast,
            power_flow,
            upgrade_plan,
            update_loop,
            storage_need,
            storage_plan,
            storage_dispatch,
            storage_dispatch_electrical,
            render_weather_animation=not args.skip_weather_animation,
            render_storage_animation=not args.skip_storage_animation,
        )
    progress.update()
    progress.close()

    print(f"Generated static world: {output_dir}")
    print(f"Saved staged data: {data_dir}")
    if not args.no_figures:
        print(f"Saved figures: {figure_dir}")


def _run_stage13_from_cache(
    config: object,
    output_dir: Path,
    data_dir: Path,
    figure_dir: Path,
    *,
    render_storage_animation: bool,
    render_figures: bool = True,
) -> None:
    data_layout = WorldDataLayout(data_dir)
    data_layout.create()
    progress = tqdm(total=2, unit="stage", dynamic_ncols=True, desc="Stage 13 storage need")
    static_maps, topology, electrical, power_flow = load_stage12_checkpoint(output_dir)
    storage_need = analyze_storage_need(topology, electrical, power_flow, config.world, config.storage)
    storage_plan = plan_storage_sites(topology, storage_need, config.storage)
    np.savez_compressed(data_layout.storage_planning / "storage_need.npz", **storage_need.as_arrays())
    np.savez_compressed(data_layout.storage_planning / "storage_plan.npz", **storage_plan.as_arrays())
    (data_layout.storage_planning / "storage_need.json").write_text(
        json.dumps(
            {
                "summary": storage_need.summary_dict(),
                "load_regions": storage_need.as_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_layout.storage_planning / "storage_plan.json").write_text(
        json.dumps(
            {
                "summary": storage_plan.summary_dict(),
                "sites": storage_plan.as_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stage_dir = figure_dir / "stage_13_storage_need"
    if render_figures:
        files = save_storage_need_figures(static_maps, storage_need, storage_plan, topology, electrical, stage_dir)
        manifest_path = figure_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest["stage_13_storage_need"] = files
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress.update()
    progress.set_description("Stage 14 storage dispatch")
    _save_stage14_outputs(
        config,
        static_maps,
        topology,
        electrical,
        power_flow,
        storage_plan,
        data_dir,
        figure_dir,
        render_animation=render_storage_animation,
        render_figures=render_figures,
    )
    progress.update()
    progress.close()
    print(f"Regenerated Stages 13-14 from cached Stage 12: {data_layout.storage_planning}")


def _run_stage14_from_cache(
    config: object,
    output_dir: Path,
    data_dir: Path,
    figure_dir: Path,
    *,
    render_storage_animation: bool,
    render_figures: bool = True,
) -> None:
    progress = tqdm(total=1, unit="stage", dynamic_ncols=True, desc="Stage 14 storage dispatch")
    static_maps, topology, electrical, power_flow, storage_plan = load_stage13_checkpoint(output_dir)
    _save_stage14_outputs(
        config,
        static_maps,
        topology,
        electrical,
        power_flow,
        storage_plan,
        data_dir,
        figure_dir,
        render_animation=render_storage_animation,
        render_figures=render_figures,
    )
    progress.update()
    progress.close()
    print(f"Regenerated Stage 14 from cached Stage 13: {WorldDataLayout(data_dir).storage_dispatch}")


def _save_stage14_outputs(
    config: object,
    static_maps: dict[str, np.ndarray],
    topology: object,
    electrical: object,
    power_flow: object,
    storage_plan: object,
    data_dir: Path,
    figure_dir: Path,
    *,
    render_animation: bool,
    render_figures: bool = True,
) -> None:
    data_layout = WorldDataLayout(data_dir)
    data_layout.create()
    storage_dispatch, dispatched_forecast, dispatched_power_flow, planned_electrical = dispatch_storage_week(
        topology,
        electrical,
        power_flow,
        storage_plan,
        config.storage,
    )
    np.savez_compressed(data_layout.storage_dispatch / "storage_dispatch.npz", **storage_dispatch.as_arrays())
    np.savez_compressed(
        data_layout.storage_dispatch / "storage_dispatch_forecast.npz", **dispatched_forecast.as_arrays()
    )
    np.savez_compressed(
        data_layout.storage_dispatch / "storage_dispatch_power_flow.npz", **dispatched_power_flow.as_arrays()
    )
    np.savez_compressed(
        data_layout.storage_dispatch / "storage_dispatch_electrical.npz", **planned_electrical.as_arrays()
    )
    (data_layout.storage_dispatch / "storage_dispatch.json").write_text(
        json.dumps(
            {
                "summary": storage_dispatch.summary_dict(),
                "capacity_expansion": storage_dispatch.expansion_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata = json.loads(data_layout.metadata.read_text(encoding="utf-8"))
    metadata["storage_dispatch"] = storage_dispatch.summary_dict()
    metadata["storage_plan"] = storage_plan.summary_dict()
    need_path = data_layout.storage_planning / "storage_need.json"
    if need_path.exists():
        metadata["storage_need"] = json.loads(need_path.read_text(encoding="utf-8"))["summary"]
    data_layout.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dump_config_snapshot(config, data_layout.config_snapshot)
    if not render_figures:
        return
    stage_dir = figure_dir / "stage_14_storage_dispatch"
    hourly_weather = load_hourly_weather_checkpoint(data_dir.parent)
    files = save_storage_dispatch_figures(
        static_maps,
        storage_dispatch,
        storage_plan,
        topology,
        planned_electrical,
        stage_dir,
        hourly_weather,
        render_animation=render_animation,
    )
    manifest_path = figure_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["stage_14_storage_dispatch"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _validate_cached_physics(output_dir: Path, config: object, from_stage: int) -> None:
    """Prevent silent mixtures of old equations, new parameters and stale worlds."""
    layout = WorldDataLayout(output_dir / "data")
    if not layout.metadata.exists() or not layout.config_snapshot.exists():
        raise ValueError(f"No complete {GENERATOR_VERSION} checkpoint; run --from-stage 1 first")
    metadata = json.loads(layout.metadata.read_text(encoding="utf-8"))
    if metadata.get("generator_version") != GENERATOR_VERSION:
        raise ValueError(f"Checkpoint uses different physical semantics; regenerate {GENERATOR_VERSION} with --from-stage 1")
    cached = load_world_config(layout.config_snapshot).to_dict()
    current = config.to_dict()
    upstream = set(current) - {"output", "storage"}
    changed = sorted(name for name in upstream if cached[name] != current[name])
    if changed:
        raise ValueError(f"Upstream configuration changed ({', '.join(changed)}); run --from-stage 1")
    if from_stage == 14 and cached["storage"] != current["storage"]:
        raise ValueError("Storage parameters changed; rerun storage sizing with --from-stage 13")


if __name__ == "__main__":
    main()
