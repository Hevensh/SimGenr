# Procedural Multimodal Source-Load-Grid World Generator

This project builds a reproducible synthetic world for multimodal source-load-grid experiments. The current pipeline starts from terrain, then derives hydrology, static land constraints, climate background, daily weather, initial cities, and land-use/load zones.

The generated data are written under `outputs/`. For the default debug config, the current output folder is:

```text
outputs/small_debug_seed42/
```

Each output run contains:

- `static_maps.npz`: static raster layers, such as elevation, rivers, cities, land use, and base load density.
- `daily_weather.npz`: daily dynamic weather tensor and weather-class maps.
- `source_load_candidates.npz`: numeric wind, photovoltaic, and load-node candidate arrays.
- `grid_nodes.npz`: numeric bus candidate array.
- `grid_topology.npz`: numeric grid edge array.
- `refined_grid_topology.npz`: refined bus and branch arrays after transit bus insertion.
- `metadata.json`: world id, seed, module seeds, map shapes, weather channels, and city metadata.
- `energy_sites.json`: wind, photovoltaic, and load-node candidate metadata.
- `bus_sites.json`: load, wind, PV, and thermal bus candidate metadata.
- `grid_edges.json`: initial line edge metadata with sampled route cells.
- `refined_bus_sites.json`: bus metadata after transit bus insertion.
- `refined_grid_edges.json`: branch metadata after long-line segmentation.
- `config_snapshot.yaml`: the exact config used for this run.
- `figures/`: staged visualization outputs for inspection.

## Run

```powershell
& 'C:\Users\Lenovo\.conda\envs\myEnv\python.exe' scripts\generate_static_world.py --config configs\small_debug.yaml
```

## Figure Outputs

The figure outputs are split by generation stage so the intermediate products can be inspected step by step.

### Stage 01: Terrain

Folder: `outputs/<world_id>/figures/stage_01_terrain/`

- `terrain_overview.png`: compact overview of elevation, slope, roughness, and curvature.
- `elevation.png`: base terrain elevation in meters.
- `slope.png`: local terrain slope, useful for construction cost, runoff, and wind steering.
- `roughness.png`: local elevation variation, highlighting broken or mountainous terrain.
- `curvature.png`: terrain curvature, showing ridges, valleys, and local surface shape.

### Stage 02: Hydrology

Folder: `outputs/<world_id>/figures/stage_02_hydrology/`

- `hydrology_overview.png`: combined hydrology summary with terrain, water, flow accumulation, water distance, and flood risk.
- `hydrology_elevation.png`: terrain elevation with river and lake overlays.
- `flow_accumulation.png`: log-scaled upstream flow accumulation used to trace river networks.
- `distance_to_water.png`: distance from each grid cell to the nearest river or lake.
- `flood_risk.png`: relative flood-risk surface around low terrain and water bodies.

### Stage 03: Static Land

Folder: `outputs/<world_id>/figures/stage_03_static_land/`

- `static_land_overview.png`: overview of land cover, vegetation, protected areas, buildability, and terrain cost.
- `land_cover.png`: categorical land cover map, including plains, hills, mountains, water, wetlands, and protected land.
- `vegetation.png`: vegetation intensity.
- `protected.png`: protected-area mask.
- `buildability.png`: relative suitability for construction after terrain, water, and protection constraints.
- `terrain_cost.png`: terrain-driven construction-cost surface.

### Stage 04: Climate Background

Folder: `outputs/<world_id>/figures/stage_04_climate_background/`

- `climate_overview.png`: compact overview of baseline temperature, humidity, wind, precipitation, cloud, and irradiance.
- `mean_temperature.png`: annual mean temperature background.
- `annual_temperature_amplitude.png`: seasonal temperature swing; usually smaller near water.
- `mean_humidity.png`: annual mean humidity.
- `prevailing_wind_speed.png`: prevailing wind speed with arrows showing local wind direction.
- `mean_precipitation.png`: annual mean precipitation.
- `mean_cloud.png`: annual mean cloudiness.
- `mean_irradiance.png`: annual mean solar irradiance.

### Stage 05: Daily Weather

Folder: `outputs/<world_id>/figures/stage_05_daily_weather/`

- `daily_weather_overview.png`: selected daily weather fields for quick inspection.
- `seasonal_temperature.png`: representative seasonal temperature maps.
- `seasonal_precipitation.png`: representative seasonal precipitation maps.
- `seasonal_irradiance.png`: representative seasonal irradiance maps.
- `annual_weather_summary.png`: annual summary curves and distributions.

The dynamic weather tensor in `daily_weather.npz` currently uses these channels:

```text
wind_u, wind_v, wind_speed, temperature, humidity, pressure, cloud, precipitation, irradiance
```

### Stage 06: Initial Cities

Folder: `outputs/<world_id>/figures/stage_06_initial_cities/`

- `city_overview.png`: overview of city suitability, urban core suitability, waterfront amenity, population density, economic activity, and urban density.
- `city_suitability.png`: overall city-placement suitability.
- `urban_core_suitability.png`: stricter city-core suitability, avoiding unsafe water-adjacent or protected cells.
- `waterfront_amenity.png`: amenity value from being moderately close to water.
- `population_density.png`: synthetic population density around generated city centers.
- `economic_activity.png`: relative economic activity intensity.
- `urban_density.png`: built-up urban density footprint.

### Stage 07: Land Use And Load Zones

Folder: `outputs/<world_id>/figures/stage_07_land_use_load_zones/`

- `land_use_overview.png`: overview of land-use categories, base load density, and main land-use tendencies.
- `land_use_zone.png`: categorical land-use map with residential, commercial, industrial, agriculture, park/green, protected, water, and background cells.
- `residential.png`: residential tendency surface.
- `commercial.png`: commercial tendency surface.
- `industrial.png`: industrial tendency surface.
- `agriculture.png`: agriculture tendency surface.
- `park_green.png`: park and green-space tendency surface.
- `load_density_base.png`: base electric load-density prior from residential, commercial, and industrial tendencies.
- `built_environment_overlay.png`: visual overlay on terrain and hydrology, using small colored sub-cell patterns to show city/building texture and land-use character.

The `built_environment_overlay.png` texture uses each `1 km x 1 km` cell as a small `3 x 3` visual pattern. Brighter or denser sub-cells generally mean stronger land-use tendency, higher base load density, or stronger economic activity.

- Residential: warm yellow/orange sub-cells, usually medium density around population clusters.
- Commercial: bright pink/magenta sub-cells with pale highlights, usually denser near high economic activity.
- Industrial: blue/purple sub-cells with cold white accents, often arranged around urban edges.
- Agriculture: sparse warm yellow sub-cells, representing low-density rural activity.
- Park/green: green sub-cells with occasional cyan waterfront accents, representing recreational or green-space areas.

### Stage 08: Energy And Load Candidates

Folder: `outputs/<world_id>/figures/stage_08_energy_load_candidates/`

- `energy_candidate_overview.png`: overview of wind suitability, PV suitability, load-node density, and all candidate sites.
- `wind_suitability.png`: wind farm siting score from wind speed, terrain, city distance, flood risk, and exclusion masks.
- `pv_suitability.png`: photovoltaic siting score from irradiance, cloudiness, slope, land use, city distance, and exclusion masks.
- `load_node_density.png`: load-node abstraction density from base load, residential, commercial, industrial, and flood constraints.
- `candidate_sites_overlay.png`: wind, PV, and load-node candidate points over terrain and hydrology.

The candidate metadata are saved in `energy_sites.json`. Wind candidates use triangle markers, PV candidates use square markers, and load candidates use circular markers in the overlay figures.

### Stage 09: Grid Bus Candidates

Folder: `outputs/<world_id>/figures/stage_09_grid_bus_candidates/`

- `grid_node_overview.png`: overview of thermal suitability, thermal externality, load density, and bus locations.
- `bus_site_overview.png`: load, wind, PV, and thermal bus candidates over terrain and hydrology.
- `thermal_suitability.png`: traditional generation bus suitability from load access, residential buffer, industrial preference, buildability, slope, and flood safety.
- `thermal_externality.png`: relative residential externality score for thermal bus candidates. It is recorded for later planning feedback but does not rewrite current city layout.
- `load_bus_overlay.png`: load bus candidates over terrain and hydrology.
- `source_bus_overlay.png`: wind, PV, and thermal source bus candidates over terrain and hydrology.

Thermal bus candidates are deliberately selected near load access but outside high-residential cores. Their `externality_score` is intended for later one- or two-round planning evolution.

### Stage 10: Grid Topology

Folder: `outputs/<world_id>/figures/stage_10_grid_topology/`

- `grid_topology_overview.png`: overview of routing cost, line route density, and initial topology over terrain/load density.
- `routing_cost.png`: rasterized line routing cost from water, protected land, slope, and terrain construction cost.
- `line_route_map.png`: rasterized line-route density from the selected initial edge set.
- `line_routes_overlay.png`: initial line routes and bus sites over terrain and hydrology.

The first topology version uses k-nearest bus candidate edges, a minimum spanning tree for connectivity, and a small number of low-cost redundant edges. It does not run power flow yet.

### Stage 11: Transit Buses

Folder: `outputs/<world_id>/figures/stage_11_transit_buses/`

- `refined_topology_overview.png`: overview of segmented routes and inserted transit buses.
- `transit_bus_candidates.png`: transit buses over the initial route-density map.
- `segmented_line_overlay.png`: segmented line routes over terrain and hydrology.
- `refined_line_route_map.png`: route density after long lines are split into shorter branches.

Transit buses are inserted along long line routes so later power-flow branches are not unrealistically long. They are abstract switching or corridor nodes, not new load or generator nodes.

## Current Modeling Notes

- The debug world uses a `64 x 64` grid with `1 km x 1 km` cells.
- Hydrology is rendered as an overlay on the original terrain elevation, so water remains visually distinct from land elevation.
- City centers avoid water, protected cells, and high flood-risk cells, while waterfront amenity can still support nearby urban or park development.
- Land-use/load zones are an intermediate planning layer. They are not yet the final electric grid, source placement, or load time-series stage.
