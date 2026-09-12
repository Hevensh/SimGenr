# 工作包 C：土地三轴、人口与能源面积守恒

本包实现综合报告 A5 以及专题06/07中的土地身份、面积分配和站点容量关系。已通过代码、配置、解析测试、两个完整世界、数据集打包和阶段恢复验收。这里的面积守恒不代表真实区域土地利用校准；所有选址、人口密度和容量密度默认值仍是未校准场景先验。

## 1. 修改文件与目的

| 文件 | 修改目的 |
|---|---|
| `core/config.py`、`configs/small_debug.yaml` | 新先验、范围检查、完整配置快照 |
| `core/datatypes.py`、`core/contracts.py` | 独立土地轴、九类份额、项目台账；单位、来源和下游契约；完整形状校验 |
| `land/land_generator.py` | 地貌与潜在背景覆盖分开，保护不覆盖这两个轴；提供硬可分配面积 |
| `city/city_generator.py` | 用硬面积计算人口/城市规模，人口无支持时报错，导出请求与分配预算 |
| `land_use/land_use_generator.py` | 从真实面积预算依次分配用途，保留原评分和显示标签 |
| `energy/energy_candidate_generator.py` | 风光共用能源预留面积，导出逐格、逐项目容量和面积账 |
| `grid/node_builder.py`、新增 `grid/thermal_land.py` | 火电领取风光后的剩余面积，按土地约束原始名牌容量 |
| `dataset/builder.py`、`dataset/loader.py` | 新 `land` 附录、静态共享和旧张量通道兼容 |
| `scripts/generate_static_world.py`、`scripts/validate_world_physics.py` | 导出、缓存版本门槛和独立土地/人口/容量账验证 |
| 新增 `tests/test_land_accounting.py`、`test_land_validation.py`、`test_thermal_land.py` | 解析面积、禁止重复分配、无支持人口、损坏导出和接口反例 |
| `tests/test_static_world.py`、`tests/test_dataset_physics.py` | 修正旧评分当面积的断言，验证0.7附录兼容 |

路径除明确前缀外均位于 `world_generator/`。热电实现细节和附加测试见 [WORK_PACKAGE_C_THERMAL.md](WORK_PACKAGE_C_THERMAL.md)。

## 2. 机理分类和顺序

```mermaid
flowchart LR
  TH[地形与静态水文] --> BG[气候与潜在背景覆盖]
  BG --> ID[独立地貌 覆盖 保护身份]
  ID --> A[硬可分配面积]
  A --> POP[城市目标与人口空间分配]
  POP --> USE[九类独占用途份额]
  A --> USE
  USE --> WP[风光项目面积台账]
  WP --> TP[火电领取剩余面积]
  TP --> GRID[原始电网资产]
```

**P：面积与人口记账。** 对格距 `dx`（km），格面积 `a=dx²`（km²）。用途份额 `f[k,i]` 无量纲，逐格满足 `sum_k f[k,i]=1`。用途总面积为 `sum_i a*f[k,i]`。人口密度单位 persons/km²，逐城空间权重先归一化再叠加，使 `sum_i density[i]*a=sum_city population`。

**S：场景先验。** 地貌/潜在覆盖阈值、保护份额、禁建坡度、城市规模规律、建设密度目标、农业/公园/能源份额、项目容量密度和退距均为场景选择。它们不是全球普适常数。面积恒等式为 P，选择分给哪一种用途的规则为 S。

**工程简化。** 九类用途依次分配，项目包络采用配置的子格求积；每个子格最多给一个候选。格内剩余份额按均匀分布降阶，不输出地籍多边形或声称真实宗地无交叠。水体和湿地掩膜使用整格占用/排除，是当前粗网格的保守场景支撑，不把窄河所在整格面积解释为真实河流水面测量。

`land_cover_type` 在城市前由气候潜在植被生成，是**潜在背景覆盖**；不随随后规划的建筑与农业份额重新分类。建设后的占用由用途份额单独表达，避免将背景覆盖标签伪装成已开发地块的精确实况。

## 3. 新配置

下表列出默认值；所有字段进入配置快照，有限性、范围和有关阈值顺序在配置入口检查。

| 配置 | 默认 | 单位和语义 |
|---|---:|---|
| `land.hill_elevation_m` / `mountain_elevation_m` | 1800 / 2800 | m，地貌标签先验 |
| `land.hill_slope_threshold` / `mountain_slope_threshold` | 0.12 / 0.22 | rise/run，地貌标签先验 |
| `land.wetland_flood_threshold` | 0.45 | 无量纲静态评分阈值，不是洪水概率 |
| `land.bare_vegetation_threshold` / `woodland_vegetation_threshold` | 0.12 / 0.45 | 潜在植被指数阈值 |
| `land.allocatable_max_slope` | 0.35 | rise/run，硬排除先验 |
| `land_use.built_population_density_persons_km2` | 6000 | persons/建设用地km²，面积需求目标 |
| `land_use.maximum_built_fraction` | 0.80 | 可用地最大建设份额；不限制请求人口 |
| `land_use.agriculture_fraction_of_remaining` | 0.45 | 农业领取当时剩余面积的份额 |
| `land_use.park_fraction_of_remaining` | 0.10 | 公园领取当时剩余面积的份额 |
| `land_use.energy_reserve_fraction_of_remaining` | 0.50 | 能源领取当时剩余面积的份额 |
| `energy.project_area_subcells_per_axis` | 4 | 每轴求积数，整数1–32，非物理距离 |
| `energy.wind_max_project_slope` / `pv_max_project_slope` | 0.35 / 0.24 | rise/run，全包络技术禁地阈值 |
| `power_grid.thermal_capacity_density_mw_km2` | 200 | MW/km²，火电包络支持密度先验 |
| `power_grid.thermal_project_radius_km` | 2 | km，火电包络半径 |
| `power_grid.thermal_project_area_subcells_per_axis` | 4 | 每轴求积数，整数1–32 |
| `power_grid.thermal_residential_score_threshold` | 0.22 | 旧住宅评分的场景核心阈值，不是覆盖份额 |

沿用既有风光容量密度、项目半径、城市退距、农业坡度和城市规模配置；本包没有把新的地区参数写进公式常数。

## 4. 字段、单位与消费者

| 字段 | 支撑与单位 | 来源 → 下游 |
|---|---|---|
| `landform` | `[H,W]`静态标签：plain/hill/mountain | 土地 → 分析/展示 |
| `land_cover_type` | `[H,W]`潜在背景覆盖：water/wetland/herbaceous/woodland/bare | 土地 → 用途/后续水文 |
| `protected_mask` | `[H,W]`独立布尔身份，旧`protected`的别名 | 土地 → 硬面积与站点约束 |
| `allocatable_land_fraction` | `[H,W]`整格可分配面积份额 | 土地 → 城市目标/用途/能源 |
| `land_use_fraction_*` | 九张`[H,W]`静态规划面积份额，无量纲 | 用途 → 站点、D水文、数据集 |
| `energy_*_area_km2` | `[H,W]`，km²；预留、风光分配和Stage8余量 | 风光 → 火电/独立验证 |
| `energy_project_land_ledger` | `[N,9]`项目身份、面积km²、容量MW、密度MW/km² | 风光 → 跨表检查/数据集 |
| `energy_project_area_by_cell_km2` | `[N,H,W]`静态规划km²，与台账同行 | 风光 → 项目/格点双重积分 |
| `thermal_land_ledger`、`thermal_project_area_by_cell_km2` | `[Nt,9]`、`[Nt,H,W]`，同上并带支持容量上界MW | 原始火电 → F规划边界/验证 |
| `thermal_allocated_area_km2`、`energy_unallocated_after_thermal_area_km2` | `[H,W]`，km²，Stage9账面状态 | 火电 → 土地总账 |
| `wind_land_eligible`、`pv_land_eligible`、`thermal_land_eligible` | `[H,W]`布尔技术边界，不是面积 | 站点 → 整个包络硬排除 |
| `city_population_budget` | 静态元数据；请求/目标/分配人数、可用km²、请求/目标/实放城市数 | 城市 → 独立人口预算 |

九类份额为 `water, wetland, residential, commercial, industrial, agriculture, park_green, natural, energy_reserve`。保护身份不加入这九类求和。能源项目账是 `energy_reserve` 内部的子账，不能再加到用途总面积里；风电项目包络也不是不透水面。

## 5. 可执行关系与守恒账

建设面积需求的整格份额为 `population_density / built_population_density_persons_km2`，实际建设份额取该需求与 `maximum_built_fraction * allocatable_land_fraction` 的较小值。住宅/商业/工业评分只分配已确定的建设预算；评分总和为零时明确回退住宅。农业与公园再领取合格剩余面积，随后预留能源，最后残余归自然用地。部分可用面积不再对已经按整格定义的人口需求重复缩放。

水体、保护、湿地和上游硬禁地不能靠高评分补偿。共用硬面积函数会对外部输入重新应用已知水体/湿地覆盖和保护掩膜。人口在禁地上为零；固定模式正目标无中心直接报错。`scale_aware`零可用面积产生明确的零目标，配置基准人数仍保留在预算中。

风光项目共享逐格面积，火电只消费Stage8剩余部分。对项目面积 `A`（km²）和配置密度 `d`（MW/km²），`P_nameplate <= A*d`。设计容量上限较小时，`P/d` 小于预留包络，二者分别保存。火电先进行原有充分性容量请求，再受土地上限约束，不通过全局增益补回被裁容量。

## 6. 实际命令与测试范围

使用 Git Bash 和 `/d/anaconda/python.exe`。临时目录、Matplotlib缓存与pytest基目录位于仓库`outputs/`；设置`PYTHONDONTWRITEBYTECODE=1`，后续命令另设置UTF-8和`MPLCONFIGDIR`，没有安装或修改全局环境。

```bash
/d/anaconda/python.exe -B -m pytest tests --ignore=tests/test_forecasting_model.py -q -p no:cacheprovider --basetemp=outputs/.implementation-tools/pytest_c_all
for seed in 42 123; do
  /d/anaconda/python.exe -B scripts/generate_static_world.py --config configs/small_debug.yaml --seed "$seed" --output outputs/work_package_c --no-figures
  /d/anaconda/python.exe -B scripts/validate_world_physics.py "outputs/work_package_c/small_debug_seed${seed}"
done
/d/anaconda/python.exe -B scripts/generate_static_world.py --config configs/small_debug.yaml --seed 42 --output outputs/work_package_c_resume --from-stage 14 --no-figures
/d/anaconda/python.exe -B scripts/validate_world_physics.py outputs/work_package_c_resume/small_debug_seed42
```

最终生成器回归 **242项通过，42.51s**。可选预测模型测试因基础环境缺少`torch_geometric`未运行，未伪造通过。新增覆盖包括份额闭合、保护独立、部分可用面积、零评分、无支持人口、格距变化、风光/热电重叠、全包络退距、非法坐标/标识/维度/参数、旁表损坏、零项目和兼容读取。

原静态测试把`buildability=.5`当作面积减半，现将其物理基准面积由2048改为4096km²，并保留次线性规模检验；这是评分与面积语义纠错，未放宽物理恒等式。

## 7. 完整世界数值

两例均64×64、1km格距、365日天气驱动、168小时运行。

| 指标 | seed42 | seed123 |
|---|---:|---:|
| 独立世界检查 | 177 PASS | 178 PASS |
| 区域/可安置面积，km² | 4096 / 3356 | 4096 / 3491 |
| 目标人口，persons | 2635759.441549 | 2730994.829277 |
| 栅格积分人口，persons | 2635759.437222 | 2730994.829599 |
| 能源预留面积，km² | 814.427513 | 834.555217 |
| 风电/光伏项目面积，km² | 21.648137 / 13.791786 | 22.890985 / 13.808985 |
| 火电项目面积，km² | 22.036648 | 17.864418 |
| Stage9后剩余能源面积，km² | 756.950942 | 779.990828 |
| Stage9火电容量/土地支持上界，MW | 2907.817791 / 4407.329652 | 2590.354811 / 3572.883505 |
| 运行缺供，MWh | 0.001233 | 0.008585 |

细节见本地输出 `outputs/work_package_c/land_population_summary.json` 及各世界 `physics_validation.json`。面积账逐格闭合；人口浮点误差保留并在允许数值精度内检验。未把非零缺供抹成零或将通过守恒解释为供电充分性。

## 8. 保存、读取和恢复

数据集升级0.7；旧静态连续/类别张量通道与候选7列顺序保留。新增32个土地附录字段单独打包并作为静态信息共享给时间窗口，避免项目数恰等于小时数时被误切片。两世界、48个24h历史+6h未来窗口及校验和检查通过。旧无土地附录数据可读取，但不补造面积保证；新字段不完整、旁表/面积立方体缺配对或形状非法时拒绝。

新增dataclass字段均追加在旧位置参数之后并带兼容默认。`land_cover`和`land_use_zone`保留混合显示语义，独立轴与用途份额才用于新土地账。新缓存要求`land_accounting_version=land_use_v1`，旧生成缓存必须重新生成，不能在B前的土地状态上恢复C后阶段。

Stage14恢复后177项检查通过，8个天气、静态、外生源荷与土地台账上游文件的SHA-256与原世界逐一相同。新增过程没有消耗新随机流；硬土地预算改变会改变城市目标和后续站点，故不承诺旧版本固定seed逐值相同。

## 9. 尚未覆盖的现实过程

目前为规划面积会计，未解析混合地块的真实边界、地籍、竖向建筑容量、风机基础实际封地、农光共用、保护政策分级、动态土地变化、冷却水和燃料物流。密度目标只是需求先验，建设面积上限不应被读作已经通过住房承载力验收。没有把背景覆盖或场景份额称为真实地区数据。

**本包容量土地保证覆盖Stage8/9原始资产。Stage14扩容尚未消费火电土地容量上界，将在F接入；当前不能提前声明最终扩建资产通过土地检查。**

## 10. 后续依赖

D可消费土地份额与小时降雨，但需单独定义不透水比例和土壤容量，不能把能源包络全当封地，也不能用测试周动态水量反向重选城市。F必须按业务bus_id连接火电土地支持上界，固定资产运行不得反向扩建。G继续将守恒、未实现过程与模型适用范围分别报告。

机理出处与关系边界见 [专题06](mechanism_research/task_06_vegetation_landuse_city.md)、[专题07](mechanism_research/task_07_energy_resources.md) 和 [综合报告](mechanism_research/synthesis_report.md)。本工作包没有新增真实数据校准或替代这些文献的经验参数。
