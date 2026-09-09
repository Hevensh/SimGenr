# C：Stage9 火电土地面积台账

本项把火电候选站接入 Stage8 风光分配之后的剩余能源用地。适宜度仍决定站点排序；面积来自已经从其他用途扣出的 `energy_reserve`，不再把适宜度当作占地比例。火电之间以及火电与 Stage8 风光之间共享同一本逐格面积预算。

## 机制、先验与适用边界

**物理关系：面积与容量上界。** 格距 $\Delta x$ 的单位是 km，方格面积 $A_{cell}=\Delta x^2$ 的单位是 km²。Stage8 输入 $R_i$ 为第 $i$ 格尚未分配的能源面积，单位 km²。火电项目 $j$ 在该格预留 $a_{ji}$ km²，满足：

$$0\leq\sum_j a_{ji}\leq R_i,\qquad R'_i=R_i-\sum_j a_{ji}.$$

项目面积 $A_j=\sum_i a_{ji}$，单位 km²；配置容量密度 $d$，单位 MW/km²；规划请求名牌 $P^{req}_j$，单位 MW。实际 Stage9 名牌和土地支持上界分别为：

$$P_j=\min(P^{req}_j,A_jd),\qquad P^{land,max}_j=A_jd.$$

`capacity_equivalent_area_km2` 为 $P_j/d$。设计名牌较小时，该折算面积可以小于预留包络面积，剩余支持能力保留在 `land_capacity_upper_bound_mw`；本阶段不会为凑足充分性目标而重新放大被土地限制的机组。

**场景先验：** `power_grid.thermal_capacity_density_mw_km2=200`、`power_grid.thermal_project_radius_km=2` 均为未校准的合成配置，要求有限且大于零。它们不是煤电、燃气或任何地区的通用典型值，不能从本默认推断真实电厂用地需求。前者将项目面积换为名牌容量上界，后者限制候选中心周围可预留范围。后续可针对设备、冷却方式、厂内燃料设施和区域规划分别设置先验；本次不下载或拟合真实电厂数据。

**工程简化：** 沿用网格的地理距离与候选中心。`power_grid.thermal_project_area_subcells_per_axis=4` 为每轴整数 $n\in[1,32]$ 求积数，默认每格 4×4 子格；每个子格最多向一个项目分配该格剩余预算的 $1/n^2$。最近候选优先，等距按输入候选顺序确定。项目间面积互斥是逐格面积预算意义上的互斥。Stage8 旁表没有保留子格地籍几何，因此把剩余份额视为格内均匀分布；不得将该算法描述为已经证明真实地籍包络无交叠。

## 硬边界与选址顺序

1. 从 `energy.land_accounting_maps['energy_unallocated_area_km2']` 读取 km² 余量；拒绝非有限、负值、超过整格面积及形状不符的输入。
2. 使用 C 的 `allocatable_land_fraction`，保留上游水体、保护区、湿地与坡度禁地。水体和保护区掩膜在共用函数中再次应用。
3. 将已有住宅评分超过 `power_grid.thermal_residential_score_threshold=0.22` 的格定义为住宅核心；该阈值是有限 $[0,1]$ 的无量纲场景先验，仍为评分阈值，不改称实际住宅覆盖份额。沿用 `thermal_residential_buffer_km`；对现有风光候选中心沿用 `thermal_min_renewable_distance_km`。所有项目格都应用该退距，而非只筛选中心。距离是格心间的 km 距离，尚未精确建模住宅和场站的实体边缘。没有住宅核心或风光站点时，对应退距不产生限制。
4. 站点选择与已有火电间距约束完成后，先运行原有充分性容量规划，再按独占土地上界裁减名牌。没有可用包络或请求容量为零的候选不形成已安装火电节点，已有 bus_id 不重新编号。

`thermal_land_eligible` 是整格是否满足上述土地/退距条件的布尔图，不是新增面积份额；可分配的具体数量始终由 Stage8 余量限定。非火电可用的剩余能源面积仍保留在余量图中，不被偷偷改分给其他用途。

## 输出与兼容

`grid_nodes.npz` 保留旧 `grid_buses` 列顺序，追加：

| 字段 | 形状与语义 |
|---|---|
| `thermal_land_accounting_mode` | 标量字符串；新面积路径为 `exclusive_project_envelopes` |
| `thermal_land_ledger` | `[N,9]`，每行对应一个实际 Stage9 火电 bus_id |
| `thermal_land_columns` | 九个列名，按下述顺序 |
| `thermal_project_area_by_cell_km2` | `[N,H,W]`，与台账同行，每格为该项目预留 km² |

列顺序为 `bus_id, row, col, reserved_area_km2, capacity_mw, capacity_density_mw_km2, capacity_equivalent_area_km2, land_capacity_upper_bound_mw, requested_capacity_mw`。标识和索引无量纲，面积为 km²，容量为 MW，容量密度为 MW/km²。跨表通过 bus_id 关联，不假定 bus_id 等于数组行号。

静态图新增 `thermal_land_eligible[H,W]`、`thermal_allocated_area_km2[H,W]`、`energy_unallocated_after_thermal_area_km2[H,W]`。保留 Stage8 `energy_unallocated_area_km2` 原样，以区分两个阶段的账面状态。零火电也导出台账 `[0,9]`、立方体 `[0,H,W]`、零分配图和原始余量。

旧手构造 `EnergyCandidateState` 完全没有面积台账时，保持旧选址流程，并显式输出 `legacy_no_land_guarantee`，不伪造土地证明；只有部分台账而缺少余量时直接报错。新增状态字段放在旧位置参数后并带默认值。

## 验证和后续接口

`tests/test_thermal_land.py` 使用独立解析案例覆盖：4 km² 格中仅余 1 km² 时最多安装 200 MW；两火电重叠包络共享中间格；住宅/风光退距作用于整个包络；水体/保护/上游禁地零占用；整域均匀面积的分辨率不变性；设计名牌保留土地余量；空资产、非法面积、旧输入与部分损坏台账；Stage9 在充分性规划后裁减并完整导出。16 项热电测试与 14 项原地理回归合计 30 项通过，运行命令为 `python -B -m pytest -q -p no:cacheprovider tests/test_thermal_land.py tests/test_geography_physics.py`。

本项保证 **Stage9 原始火电资产** 的土地约束。Stage14 的热电扩容尚未消费该上界；F 应按 bus_id 将 `land_capacity_upper_bound_mw` 接入总容量约束，再检查扩容后的土地可行性。在这项接线完成前，不能声称 Stage14 已通过土地验收。冷却水取用、燃料运输、排放许可、热负荷和真实场站土建面积也未由本台账实现。

机制研究背景见 [能源资源与选址研究](mechanism_research/task_07_energy_resources.md) 及 [电网运行与储能研究](mechanism_research/task_09_grid_operation_and_storage.md)。面积恒等式是单位与守恒关系；本页新增数值配置全部是显式场景先验。
