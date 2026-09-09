# Procedural Multimodal Source-Load-Grid World Generator

This project builds a reproducible synthetic world for multimodal source-load-grid experiments. The current physics_v4 pipeline derives terrain and drainage first, then climate, climate-conditioned land, hourly weather with explicit diagnostics, cities, land use, sources, the grid, and constrained operation.

## 当前实现入口（physics_v4，A–G）

完整的十专题文献与机理调查见 [调查入口](docs/mechanism_research/README.md) 与 [综合报告](docs/mechanism_research/synthesis_report.md)。当前已实现范围、单位、验证结果与剩余边界见 [physics_v4 实现总结](docs/IMPLEMENTATION_SUMMARY.md) 和 [A–G 实现映射及验收记录](docs/MECHANISM_IMPLEMENTATION.md)。文献建议不等于已启用的功能；以实现总结和字段契约为准。

| 工作包 | 当前实现 |
|---|---|
| A–B | 单位/实体/时间契约；小时原始天气、统一湿空气诊断与真实日汇总 |
| C–D | 独立土地轴、人口/项目用地账；可选小时土壤、河道与湖泊水量账 |
| E | 共同天气驱动风光与部门负荷，显式热初态、容量因子和能量积分 |
| F | 固定/事前/oracle 资产模式，岛内调度、SOC、临时故障与备用 |
| G | 可定位的四态校验、成对机理干预、跨世界比较和结构化生成失败 |

配置中的经验权重仍是待校准的场景先验。`source_load_forecast` 是兼容名称，Stage11 数据代表外生合成实况与初步计划；Stage14 的供电、弃电、缺供和储能是运行结果。历史 physics_v3 的 [修改总览](docs/PHYSICS_REVIEW_ZH.md) 与 [验证记录](docs/VALIDATION_RESULTS.md) 保留供追溯。

在仓库根目录 Git Bash 中使用已有的 D 盘 Anaconda（以下 `python` 均由此 PATH 选择）：

```bash
export PATH="/d/anaconda:/usr/bin:/bin:$PATH"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mkdir -p outputs/.mplconfig outputs/.tmp
export MPLCONFIGDIR="$(cygpath -m "$PWD/outputs/.mplconfig")"
export TEMP="$(cygpath -m "$PWD/outputs/.tmp")"
export TMP="$TEMP"
python -B scripts/generate_static_world.py --config configs/small_debug.yaml --seed 42 --no-figures
python -B scripts/validate_world_physics.py outputs/small_debug_seed42
```

增加 `--start-day 180 --output outputs/summer` 可检查夏季。移除 `--no-figures` 可保留原有分阶段可视化。生成依赖见 [requirements-generator.txt](requirements-generator.txt)，所有运行输出默认位于仓库 `outputs/`。

数据集 schema 为 **0.10.0**，兼容旧主字段并附加天气诊断、水文状态、外生源荷、初始/冻结资产及运行明细。详见 [字段和预测语义](datasets/DATA_DESCRIPTION.md)。现代恢复要求相应契约、规划输入和资产哈希；缺少必要标记的旧世界须从 Stage1 重新生成。

当前 [requirements-generator.txt](requirements-generator.txt) 覆盖生成、可视化、独立物理验证和 NumPy 数据打包的运行依赖。模型训练依赖另行准备；本次环境缺少可选 `torch_geometric`，未安装，也不将模型测试计入生成器验收。

### 资产模式与信息边界

在配置文件设置 `planning.mode`：

| 模式 | 资产来源 | 运行期间 |
|---|---|---|
| `fixed_assets` | 静态生成资产，储能仅由 `planning.fixed_storage_sites` 明确指定，默认无储能 | 不扩容，不按运行周重新选址 |
| `preplanned` | 独立设计天气/源荷、配置、随机流及时间轴，设计后冻结 | 保留完整运行周，用固定容量调度 |
| `full_window_planning`（默认） | 使用完整运行实况规划，明确标为 oracle | 允许配置边界内的联合扩容 |

三种模式的调度器当前都使用完整运行窗口，不能据此声称在线预测控制。规划快照、内容哈希和输入边界保存在 `data/planning/`。详见 [资产规划说明](docs/WORK_PACKAGE_F_PLANNING.md)。

### 单位、小时/日支撑与验证

主生成入口使用 1h 区间。天气九通道顺序保留；温度为 °C，RH 为 %，压强为 hPa，风为 m/s，GHI 为区间平均 W/m²，降雨为区间累计 mm。小时导出的真实日统计在 `hourly_daily_summary.npz`；`daily_weather.npz` 是日驱动锚点，不把日均 RH 或标量风速强制等同于由日均原始态重算的诊断。当前无曙暮光/人工照明的太阳模式夜间 GHI 为零。

高程为 m，坡度为 rise/run，曲率为 m⁻¹，汇水面积为 km²；功率为 MW，区间能量为 MW×h 得到的 MWh。SOC/水库存有 T+1 个边界，区间通量有 T 个值。完整来源、形状和时间支撑见各世界 `data/field_contracts.json` 及各附录字段 schema。

```bash
# 已生成世界的独立检查，输出写在各世界目录中
python scripts/validate_world_physics.py outputs/small_debug_seed42
# 同资产、同时间轴、共同随机流的成对机理干预
python scripts/validate_mechanism_interventions.py --seeds 42 123 --output-dir outputs/validation/interventions
# 仅当两个世界都已生成时，比较各自检查和描述性统计
python scripts/validate_world_matrix.py outputs/small_debug_seed42 outputs/small_debug_seed123 --output outputs/validation/world_matrix.json
```

验证区分 `PASS`、`FAIL`、`NOT_RUN`、`UNSUPPORTED`；PASS 不代表零缺供、现实校准或 AC 安全认证。种子/分辨率比较也不等同于成对因果干预。生成异常在能确定合法世界目录后记录 `generation_failure.json`，保持非零退出；合法 ENS 样本仍可通过物理检查，见 [失败分类说明](docs/WORK_PACKAGE_G_FAILURES.md)。

Project repository: <https://github.com/Hevensh/SimGenr>

## Generate One Seed

For an inspectable single world, run the following commands from the repository root in Git Bash. Replace `42`
with any integer seed:

```bash
python scripts/generate_static_world.py \
  --config configs/small_debug.yaml \
  --seed 42
```

The generated world is written to:

```text
outputs/small_debug_seed42/
```

This default mode retains the full staged visualization. The output directory contains stage-grouped numeric data
under `data/`, intermediate PNG figures and animated WebP files under `figures/`, the exact configuration snapshot,
and world metadata. Use this mode when checking whether terrain, hydrology, weather, cities, grid routing, power
flow, and storage behavior are reasonable. The same seed and configuration reproduce the same world.

For batch dataset production, add `--no-figures`. This mode skips all intermediate PNG/WebP visualization and keeps
the machine-readable stage data used by dataset packaging:

```bash
python scripts/generate_static_world.py \
  --config configs/small_debug.yaml \
  --seed 42 \
  --no-figures
```

The multi-seed dataset generator uses this no-figure workflow because rendered figures are inspection artifacts and
are not embedded in the final `.npz` samples.

To resume from an existing stage checkpoint, use `--from-stage`. For example, this reruns Stage 13 and Stage 14
without rebuilding terrain, weather, cities, or the initial grid:

```bash
python scripts/generate_static_world.py \
  --config configs/small_debug.yaml \
  --seed 42 \
  --from-stage 13
```

The generated data are written under `outputs/`. For the default debug config, the current output folder is:

```text
outputs/small_debug_seed42/
```

Each output run contains:

- `data/stage_05_weather/`: daily and hourly weather arrays.
- `data/dynamic_hydrology/`: optional hourly water states, fluxes and budgets.
- `data/stage_08_energy_sites/`: wind, photovoltaic, and load candidate arrays and metadata.
- `data/stage_09_grid_buses/`: load, wind, PV, and thermal bus candidates.
- `data/stage_10_grid_topology/`: integrated static maps, A* topology, and initial electrical parameters.
- `data/stage_11_operation/`: source/load forecast, baseline power flow, and upgrade proposal.
- `data/stage_12_grid_update/`: final updated topology, electrical state, and power flow.
- `data/stage_13_storage_planning/`: storage need analysis and selected sites.
- `data/stage_14_storage_dispatch/`: final dispatch, expansion, SOC, and network operation.
- `data/planning/`: independent design inputs when applicable, initial/frozen assets and information-boundary hashes.
- `data/metadata.json`: world-wide dimensions, module seeds, and summaries.
- `data/config_snapshot.yaml`: the exact config used for this run.
- `figures/`: staged visualization outputs for inspection.

Legacy flat outputs can be reorganized without recomputation using `python scripts/migrate_output_layout.py`.

## Additional Run Options

```bash
python scripts/generate_static_world.py --config configs/small_debug.yaml
```

To rerun Stage 13 and all later stages from the cached final Stage 12 state:

```bash
python scripts/generate_static_world.py --config configs/small_debug.yaml --seed 42 --from-stage 13
```

In fixed/preplanned modes, Stage13 resume reuses the frozen site plan; it does not select sites from cached operation weather. Changes to their storage configuration currently require Stage1 regeneration. Oracle mode retains Stage13 replanning. Stage14 resume requires unchanged storage settings and verifies the asset boundary.

The batch script forwards the same argument to every configured seed:

```bash
PYTHON_BIN=/d/anaconda/python.exe CONFIGS=configs/small_debug.yaml bash scripts/generate_all.sh --from-stage 13
```

The batch script resolves relative config/output paths from the repository root and can be invoked from another directory. Its default config path supports spaces in the checkout path. `CONFIGS="configs/first.yaml configs/second.yaml"` preserves the whitespace-separated multi-config interface; `SEEDS="42 123"` selects seeds.

To retain the existing storage plan and rerun only Stage 14 dispatch:

```bash
python scripts/generate_static_world.py --config configs/small_debug.yaml --seed 42 --from-stage 14
```

Use `--skip-storage-animation` when only the numeric dispatch and static figures are needed. The previous
`--skip-storage-gif` spelling remains available as a compatibility alias.

Hourly weather, line-loading, and storage-dispatch animations are written as animated WebP files at 4 fps.
Use `--skip-weather-animation` to omit the weather and line-loading animations on later reruns.

## Dataset packaging

Package generated worlds into aligned static, dynamic, graph, and operation samples with:

```bash
python scripts/build_dataset.py
```

Each world seed is stored as one compressed file such as `samples/seed42.npz`. Metadata and the exact
configuration snapshot are embedded in the same file; rendered PNG/WebP figures are excluded.

Generate the default seed 1-50 dataset without figures, with resumable tqdm progress and a prime-seed
validation/test split:

```bash
python scripts/generate_dataset.py
```

See `datasets/DATASET_USAGE.md` for generation and loading examples. Field definitions, units, graph semantics,
and the observable-data inclusion policy are documented separately in `datasets/DATA_DESCRIPTION.md`.

For model training, `scripts/train_physics_gst.py` preloads all selected train and validation worlds directly
onto the target device by default. It caches world-level typed graph indices, bidirectional edges, static
normalization results, and masks. Temporal windows remain lightweight indices and are prepared on demand rather
than retained on CUDA. Use `runtime.preload_to_device: false` when the selected worlds do not fit in GPU memory.
Checkpoints default to `checkpoints/<model>/seed_<seed>/<horizon>.pt`.
All model, loss, window, optimizer, runtime, seed, and output settings live under `configs/models/`.

Example: use one day to forecast the following day:

```bash
python scripts/train_physics_gst.py \
  --config configs/models/physics_gst_24to24.yaml
```

Stage 14 solves a multi-period DC optimal power flow with mode-dependent investment bounds, nodal active-power balance, line limits, thermal capacity/ramping, storage power/energy limits and exclusive charge/discharge modes. Cyclic SOC is the default; a fixed prior SOC is available. Insufficient supply is explicitly reported as nodal load shedding when allowed. Set `storage.allow_load_shedding: false` to require zero shortfall; an infeasible solve then fails explicitly. Dispatch has full-window foresight in every asset mode. A validation PASS confirms evaluated physical accounting and constraints, not supply adequacy or empirical realism.

Both full and resumed runs show a stage-level `tqdm` progress bar.

## Figure Outputs

The figure outputs are split by generation stage so the intermediate products can be inspected step by step.

### Stage 01: Terrain

Folder: `outputs/<world_id>/figures/stage_01_terrain/`

`configs/small_debug.yaml` uses the coordinate-stable `multiscale_v2` terrain. Noise is sampled in kilometer world coordinates with fixed continental, erosion, ridge, detail, and domain-warp wavelengths. Increasing the grid extent therefore reveals new terrain instead of rescaling the same normalized map. The legacy generator remains available with `terrain.algorithm: legacy`.

Chunk consistency and the 64/128 km hydrology comparison can be regenerated with:

```bash
python scripts/compare_multiscale_terrain.py \
  --config configs/small_debug.yaml \
  --seed 42 \
  --output outputs/terrain_v2_comparison_seed42
```

- `terrain_overview.png`: compact overview of elevation, slope, roughness, and curvature.
- `elevation.png`: base terrain elevation in meters.
- `slope.png`: local terrain slope, useful for construction cost, runoff, and wind steering.
- `roughness.png`: local elevation variation, highlighting broken or mountainous terrain.
- `curvature.png`: terrain curvature, showing ridges, valleys, and local surface shape.

### Stage 02: Hydrology

Folder: `outputs/<world_id>/figures/stage_02_hydrology/`

The `conditioned_v2` hydrology path fills closed DEM depressions for drainage, identifies bounded depression lakes, extracts streams from physical upstream catchment area, and scales river width from catchment size. The legacy quantile-based path remains available with `hydrology.algorithm: legacy`.

- `hydrology_overview.png`: combined hydrology summary with terrain, water, flow accumulation, water distance, and flood risk.
- `hydrology_elevation.png`: terrain elevation with river and lake overlays.
- `flow_accumulation.png`: log-scaled upstream flow accumulation used to trace river networks.
- `distance_to_water.png`: distance from each grid cell to the nearest river or lake.
- `flood_risk.png`: relative flood-risk surface around low terrain and water bodies.

### Stage 03: Static Land

Folder: `outputs/<world_id>/figures/stage_03_static_land/`

`land_cover` remains a legacy mixed display label. The current data also expose independent `landform`, `land_cover_type`, and `protected_mask` axes; protection is not treated as a vegetation class.

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

The dynamic weather tensor in `data/stage_05_weather/daily_weather.npz` currently uses these channels:

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
