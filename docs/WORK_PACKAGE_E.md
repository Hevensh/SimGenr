# E：源荷诊断、容量与能量接口

状态：E 完成，代码、回归、双种子、打包与 Stage14 恢复均通过。D 基线为 `7fe0d67`；本包不修改 D 水文过程。

## 1. 文件

主要文件与职责如下。

- `world_generator/operation/source_load_forecast.py`：设备出力、热记忆和天气诊断。
- `world_generator/core/config.py`：`SourceLoadConfig` 默认、验证、快照。
- `world_generator/core/datatypes.py`：追加 Store 字段及完整序列化校验。
- `world_generator/core/source_load_contracts.py`：明确字段名单、单位、时间支撑。
- `world_generator/core/contracts.py`：P/E/S、来源和消费者的机器契约。
- `world_generator/operation/stage_cache.py`：独立源荷缓存读取与天气对齐。
- `world_generator/dataset/builder.py`、`loader.py`：原节点身份、附录与窗口。
- `scripts/generate_static_world.py`：声明输出来源、恢复前检查机理版本。
- `tests/test_source_load_interfaces.py`：接口、损坏数据、窗口与缓存反例。
- `scripts/source_load_validation.py`：独立物理验收，由父任务维护。

数值关系、引用和配置解释见 [E 数值子文档](WORK_PACKAGE_E_PHYSICS.md)。

## 2. 原因与机理

对应 A7：源荷除功率曲线外还需明确设备额定量、容量因子定义、天气采样、负荷热初态与记忆，以及 MW 到 MWh 的时间支撑。单位积分和容量上下界为 P；设备曲线、损耗、热记忆和负荷波动参数依据数值子文档分别标明 E/S。

名牌容量是设计属性；可用功率是天气和设备关系作用后的外生实况。
计划功率是规划层量，不能据此宣称实际发电或实际供电。
负荷基准是设定的规模，受温度和时间模式驱动的请求可超过该基准。
容量因子仅对名牌大于零的风、光、热发电节点定义。
负荷与无效节点的 CF 值为零，并以独立有效掩膜表达“不适用”。

## 3. 配置

新增和有效配置完整保存于 `SourceLoadConfig` 及世界快照；默认、范围和来源由数值核说明。本接口不另设设备参数。

风机的高度、切入/额定/切出速度、参考密度与风切变先验可配置。
光伏的倾角、方位、温度系数、Faiman 散热参数和损耗可配置。
负荷的热时间尺度、初态模式、温度响应、日历及随机残差先验可配置。
配置在构造和显式验证入口检查，非法值在天气/源荷数组生成前报错。
所有有效值进入 `config_snapshot.yaml`；不存在只写文档的隐藏覆盖项。
数值先验不冒称由所引论文直接给出或已经地域校准。

## 4. 字段

旧四个功率字段保持。追加节点名牌、基准负荷、热初态、天气采样 row/col，六项 T×N 诊断；小时请求/可用/计划能量及全时段能量，CF 和有效掩膜。所有新字段按显式名单标明 N 或 T×N，避免 N=T 时错误切片。

| 字段族 | 形状与单位 | 时间与消费者 |
|---|---|---|
| `nameplate_capacity_mw`, `reference_load_mw` | N，MW | 静态设计属性；源荷核、验收、数据集 |
| `weather_sample_row/col` | N，整数索引 | 静态采样位置；与原始母线及同小时天气核对 |
| `capacity_factor_valid` | N，bool | 静态发电适用掩膜 |
| `initial_effective_temperature_c` | N，°C | 区间前边界热状态；窗口需重取 |
| `diag__load_effective_temperature_c` | T×N，°C | 每小时更新后的末边界状态 |
| `diag__hub_wind_speed_mps`, `diag__wind_air_density_kg_m3` | T×N，m/s、kg/m³ | 风节点小时代表诊断 |
| `diag__pv_poa_w_m2`, `diag__pv_module_temperature_c` | T×N，W/m²、°C | 光伏节点小时代表诊断 |
| `diag__load_log_residual` | T×N，无量纲 | 负荷对数随机残差 |
| `*_energy_mwh` | T×N，MWh | 功率乘区间小时数；区间累计 |
| `period_*_energy_mwh` | N，MWh | 已实现整段积分；审计或标签 |
| `available/planned_capacity_factor` | T×N，无量纲 | 对名牌容量归一化 |

不适用节点的六项诊断为零，适用种类另存于 metadata。
热初态与 T 个末状态共同构成 T+1 个边界状态；未把热状态标成区间均值。
源荷 NPZ 共 33 项：旧 10 项、新 20 项数值字段及 3 项文本声明。
文本声明包含版本、严格字段 schema 和来源 metadata。
不根据第一个维度恰好等于 T 来推断任何字段语义。

## 5. 命令

实际定向命令：

```bash
/d/anaconda/python.exe -B -m pytest tests/test_source_load_interfaces.py tests/test_dataset_physics.py tests/test_field_contracts.py -q -p no:cacheprovider --basetemp=outputs/.implementation-tools/pytest_e_interfaces
```

该命令保存在 `outputs/.implementation-tools/test_e_interfaces.sh`。
完整回归、生成、验收与打包实际命令如下，分别保存在同目录 `test_work_package_e_all.sh`、`generate_work_package_e.sh`、`package_work_package_e.sh`。

```bash
/d/anaconda/python.exe -B -m pytest tests --ignore=tests/test_forecasting_model.py -q -p no:cacheprovider --basetemp=outputs/.implementation-tools/pytest_e_all
for seed in 42 123; do
  /d/anaconda/python.exe -B scripts/generate_static_world.py --config configs/small_debug.yaml --seed "$seed" --output outputs/work_package_e --no-figures
  /d/anaconda/python.exe -B scripts/validate_world_physics.py "outputs/work_package_e/small_debug_seed${seed}"
done
/d/anaconda/python.exe -B scripts/build_dataset.py --input-root outputs/work_package_e --output outputs/work_package_e/dataset --worlds small_debug_seed42 small_debug_seed123
/d/anaconda/python.exe -B outputs/.implementation-tools/summarize_e.py
```
全部命令由仓库内 Git Bash 脚本运行，不安装依赖。
`PYTHONDONTWRITEBYTECODE=1`，`PYTHONUTF8=1`，`PYTHONIOENCODING=utf-8`。
`MPLCONFIGDIR`、`TMPDIR` 均在仓库 outputs；TMP/TEMP 用 `cygpath -m` 转成 Windows 路径。

## 6. 结果与未覆盖

接口、数据集和字段契约定向测试为 **48 passed，1.20 s**。
覆盖缺字段、非法坐标、错误形状、非适用诊断非零、非有限值。
覆盖篡改小时/整段 MWh、CF 与掩膜、错时间边界和只保留声明的残缺附录。
覆盖 N=T 时静态不被切片、原始节点顺序独立于最终电网节点顺序。
覆盖子窗口前热状态、窗口内能量积分和后期实况不影响早期历史输入。
覆盖旧 Stage14 Store 兼容及当前入口拒绝跨源荷机理恢复。
完整生成器回归 **383 passed，48.30 s**；未运行可选预测模型测试 `test_forecasting_model.py`，本包没有安装 PyTorch 依赖。
数值核与配置的两个测试文件合计 56 项，已包含在完整回归中，不另加到 383。
seed42 独立物理验收 **PASS 292 checks**，seed123 **PASS 269 checks**。
父任务完成 seed42 Stage14 恢复，仍 **PASS 292**；9 份上游文件 SHA256 完全一致。
本包两个种子的 8 份 D 上游 NPZ、每份世界共 197 个数组逐值相同，证明源荷调整没有反馈污染天气、土地或水文。
两世界 checksum 加载与全部 **48 个 24 h 历史 / 6 h 未来窗口** 通过，源荷附录 33 项完整，水文 T+1 状态保留。
Stage11 外生需求、可用发电和计划发电不冒充最终送达、弃电或缺供，后者由 F 运行层处理。

## 7. 种子摘要

结果位于 `outputs/work_package_e`，完整按技术容量与能量账见 `source_load_summary.json`。

| 168 h 世界统计 | seed42 | seed123 |
|---|---:|---:|
| 请求负荷 MWh | 412740.824898 | 448438.537531 |
| 可用发电 MWh | 510619.064686 | 460857.120816 |
| 计划发电 MWh | 410746.597470 | 424249.558735 |
| 容量加权风电可用 CF | 0.385178 | 0.355351 |
| 容量加权光伏可用 CF | 0.220765 | 0.265747 |

以上为 Stage11 外生/计划账；差额不直接称为节点缺供或实际弃电。
日志为 `tests.log`、`generate_seed*.log`、`validate_seed*.log`、`package.log`、`summary.log`，恢复证据为 `resume_comparison.json`。

## 8. 数据与图

机器可读 NPZ/JSON 和附录窗口为主要交付，使用 `--no-figures`。

数据集 schema 升至 `0.9.0`，新增独立 `source_load` 附录组。
组内保存 Stage11 原始 bus_id 和顺序，不强行重排成 Stage14 电网。
旧 `operation` 通道保持原有最终调度语义。
整世界读取保留全部积分审计；时间窗口重新计算各自的区间能量和。
`source_load_static` 仅提供固定资产、适用性和来源，不包含整段能量摘要。
窗口热初态从开始前一小时末状态读取，首窗口使用世界配置初态。
完整世界的未来能量不会作为历史窗口的静态模型输入。

## 9. 兼容

Store 新字段追加在旧参数后；旧 Stage14 构造无 E 附录时不捏造。新完整 E 输出声明 `source_load_v1`，缺失或损坏附录明确拒绝。数据集保留旧通道并新增独立 `source_load` 组。

通用旧 NPZ 和数据集加载仍可运行，缺 E 附录会明确保留 legacy 语义。
当前生成 CLI 的 `--from-stage 13/14` 要求 `source_load_v1` 机理声明。
D 及更早缓存必须从 Stage1 重生成，避免旧上游曲线与新配置快照混用。
完整新缓存仍检查上游配置、小时戳、采样网格和声明路径一致。
该限制由独立机理 marker 实现，不借全局生成器版本掩盖数据来源。

## 10. 后续依赖

F 可以显式对比外生可用量、规划量和实际运行量。
E 不把本周规划量当作真实观测或未来可知信息。
设备关系和需求响应仍是合成情景近似，需要真实地区时序作校准。
E 附录不会宣称电网约束已经满足，运行层需另外核验网络、储能和缺供。
本包通过后才推进 F；当前不修改 F 的调度模型。
