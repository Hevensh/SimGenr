# 机理驱动实现与基础验证

基线：`3d6d532`，已合并 physics_v3 和十专题调查。实施分支：`mechanism-implementation`。本文件随 A→G 的实际验收更新，未运行项不记为通过。P=物理关系，E=经验规律，S=场景先验；工程简化单独注明。

## 实现映射（修改前审计）

| 机理编号/工作包 | 代码位置 | 当前状态 | 修改目标 | 验证方式 |
|---|---|---|---|---|
| A / 综合报告A1、A2 | `core/config.py`、`core/datatypes.py`、生成入口、阶段缓存 | NPZ已有命名单位，缺统一时间/空间支撑和字段契约 | 集中换算、形状校验、字段来源/消费者、版本化元数据 | MW→MWh、mm·km²→m³、时间聚合、非法配置反例 |
| B / A3、A4 | `weather/physics.py`、`weather/weather_generator.py`、`climate/climate_generator.py` | 日小时预算已有，风均值等式排除转向，诊断量归属不统一 | 原始态与统一诊断、标量/矢量均值、物理尺度配置 | 反向风、湿空气诊断、夜间辐照、42/123及尺度检查 |
| C / A5 | `land/land_generator.py`、`land_use/land_use_generator.py`、`energy/energy_candidate_generator.py` | 覆盖/地貌/保护混合，评分被用于面积预算 | 三轴分离、逐格用途闭合、能源占地台账 | 水体/保护硬边界、土地与人口守恒、分辨率 |
| D / A6 | `hydrology/hydrology_generator.py`、`core/datatypes.py` | Priority-Flood/D8骨架已有，无小时水量 | 可选土壤/地下水/河湖状态及水量账 | 零雨、闭合、非负容量、湖泊单调、关闭不变 |
| E / A7及任务07/08 | `operation/source_load_forecast.py` | 风机曲线、PV和负荷热记忆已有 | 复用共同诊断天气，强化时间与可用/计划/送出边界 | 设备边界、云光及温度成对干预、容量与能量 |
| F / A7及任务09 | `operation/storage_dispatch.py`、`power_flow.py`、生成入口 | 分岛DC/SOC/互斥已有，全窗口规划为默认 | 三类资产信息模式、固定资产故障与初态、备用约束 | KCL/线限/SOC、断线孤岛、冻结输入不受测试期影响 |
| G / A8及任务10 | `scripts/validate_world_physics.py`、`tests/` | 布尔PASS与最大残差已有，缺四态与位置 | 四态、独立账本、定位、成对及跨种子验证 | 原有回归+新增反例+42/123+缓存恢复 |

执行入口当前顺序为 `[1,2,4,3,5,6,7,8,9,10,11,12,13,14]`。配置由 dataclass 加载并完整快照，随机流由世界种子与模态名哈希派生。NPZ/JSON保持兼容，新增字段用附加键/旁表保存；语义变更将更新缓存版本并要求重新生成。没有读取真实区域数据作校准。

## 工作包记录

以下记录只填写实际完成的代码、命令与结果。新增参数是可检查的合成情景配置，不能据此推出真实区域精度或工程安全。

### A：字段、单位与时间支撑（通过）

- 文件：`core/contracts.py`（新增）、`core/config.py`、`core/datatypes.py`、`scripts/generate_static_world.py`、`configs/small_debug.yaml`、`tests/test_field_contracts.py`。
- P：集中处理正有限步长、方格面积、雨深体积换算、功率积分、完整日聚合；拒绝非有限值、错误维度、非连续时间、负容量。S/工程接口：`contracts.time_step_hours=1.0`、`export_field_contracts=true`；完整配置进入快照。主生成器明确只接受1h，通用积分函数支持其他显式时长。
- 字段：天气/源荷新增`time_bounds_hours[T,2]`，单位h，表示区间起止，供聚合、源荷、水文、验证消费；`data/field_contracts.json`记录来源、消费者、量类型和信息层。SOC仍为MWh、T+1边界；雨深为mm区间累计；评分不当作面积。旧站点`x/y`仍为[0,1]显示坐标。
- 基线命令：`/d/anaconda/python.exe -B -m pytest tests/test_static_world.py tests/test_weather_physics.py -q`，50通过；A验收加`tests/test_field_contracts.py`，58通过（30.28s）。实际运行另指定仓库内`cache_dir`和`--basetemp`；所有命令由Git Bash执行。
- seed42最小探针：64×64，格面积1km²，区域4096km²；均匀1mm降水对应4,096,000m³；原地形高程484.874–2439.573m。此为单位/静态探针，不冒充完整运行结果。
- 兼容：NPZ保留旧键并附加时间边界；缓存版本升级为`physics_v4`，旧physics_v3可读取研究数据，但不能混用新生成阶段恢复，须重新生成。没有新增随机抽样或改变原模块随机流。
- 未覆盖：动态水量、天气重构、用途面积、故障与备用尚未由A实现。B应复用本契约并保留九通道读取兼容。
- 数据集兼容复核：`dataset/builder.py`原先将非physics_v3一律视为旧数据；已加入明确的physics_v3/v4支持集合，继续要求外生源荷并保持人口单位persons/km²。`tests/test_dataset_physics.py`与A契约共19项通过（0.38s）；未知版本仍不自动宣称具有物理单位。
