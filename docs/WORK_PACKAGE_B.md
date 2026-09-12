# 工作包 B：天气原始量、诊断量与物理尺度

本工作包以 A 的单位和时间契约为基础，已经实现天气接口、数值核、气候空间尺度、保存/读取和验证接线。保持九个天气通道的顺序；没有改动源荷出力函数、电网运行函数或储能策略。气候子项的独立说明和尺度实验见 [WORK_PACKAGE_B_CLIMATE.md](WORK_PACKAGE_B_CLIMATE.md)。

## 1. 主要行为

`temperature`、水汽比湿 `specific_humidity_kg_kg`、参考海平面气压 `sea_level_pressure_hpa` 和风矢量驱动逐小时天气。由同一组原始量计算相对湿度、地面气压、空气密度和风速。默认允许日内风向改变，不能把日均风速写成日均风矢量的模。

原 `daily_weather.npz` 现在明确标记为 **daily_driver_anchors**，是合成过程的日驱动值；其中 RH/p/rho 是日驱动状态的诊断，不能声称是某条隐藏小时实况的平均值。新增 `hourly_daily_summary.npz`，其每个日值都由实际输出的 24 个小时汇总：降雨深度求和，其他天气通道及诊断取均值。非线性诊断必须先逐小时计算，再平均。

| 模式 | 硬性条件 | 其余日统计 | 用途 |
|---|---|---|---|
| `primitive_hourly`，默认 | T、u、v、云量的日均，日降雨总深度 | 标量风速、RH、p、GHI 由小时过程产生 | 默认多模态合成实况 |
| `daily_conditioned` | 上述条件，另加日均标量风速与 GHI | RH/p 仍由逐小时原始量诊断 | 具有明确日输入条件的相容性实验 |

每个小时存储的 `weather_metadata_json` 包含 `generation_mode`、`daily_constraints`、日驱动与实际汇总的逐通道最大绝对偏差，以及辐射和湿度模型边界。默认模式的 GHI/RH/p 统计偏差可见，并未伪装成日约束闭合。

## 2. 关系分类与公式

**P：几何关系和理想湿空气气体关系。** 若 q 为 kg 水汽/kg 湿空气，\(\epsilon=R_d/R_v\)，则

\[
e=p\frac{q}{\epsilon+(1-\epsilon)q},\quad
T_v=T_K[1+(R_v/R_d-1)q],\quad
\rho=\frac{100p_{hPa}}{R_dT_v},\quad RH=\frac e{e_s(T)}.
\]

`diagnose_moist_air(T_c, q_kg_kg, elevation_m, p0_hpa)` 返回 RH（0–1 比例）、p（hPa）、rho（kg/m³）。T 的输入单位为 °C，内部气体状态方程使用 K；气压转为 Pa 时乘 100。逆函数 `specific_humidity_from_relative_humidity(T_c, RH, p_hpa)` 使用相同水汽定义。独立验证用干空气和水汽的分压分别计算密度，避免仅用同一实现自检。

**E 与工程简化：** 饱和水汽压沿用 FAO-56/Tetens 液水近似；地面气压采用静力柱近似

\[
p=p_0\exp[-gz/(R_d\overline T_v)],\quad
\overline T_K\approx T_{surface,K}+0.00325z.
\]

层平均温度中 0.00325 的单位为 K/m，是 6.5 K/km 递减率的一半，不是从本地图拟合得到的真实温度剖面。当前湿空气新核仅接受 **[-90,65] °C 的数值支持域**，超界报错；该范围不是精度保证。尤其不会让 80 °C 偷用旧饱和压函数裁剪后的 65 °C 值。未实现冰面饱和压、混合相、垂直探空与完整边界层。

**P：风统计的三角不等式。**

\[
\overline{|\mathbf U|}\ge|\overline{\mathbf U}|.
\]

小时二维扰动去掉均值，保留输入 u/v 日均；默认由小时向量模产生标量均值。条件模式求解扰动幅度匹配给定标量均值。等号情形必须保留共同方向；不等号允许转向。独立反例为前 12 小时 `(u,v)=(5,0)` m/s，后 12 小时 `(-5,0)`：日矢量为零、日均速为 5 m/s，已能生成并通过检查。小于矢量模的标量均值拒绝。

**S：条件云量与简化辐射。** 模型使用

\[
G_h=G_{clear,h}(1-k_c C_h),\quad 0\le C_h\le1.
\]

默认逐小时直接计算 GHI。条件模式在固定日均云量下，按太阳权重排序构造加权云量的上下界；若给定日均 GHI 不在该区间，拒绝。可行时，将随机云形状与相应端点作凸组合，同时满足云量、GHI 两个日矩约束。不是分别归一化云和辐射、随后假装两者仍满足同一个关系。云量整日为零、晴空 GHI 日均 200 W/m²、给定日均 GHI 20 W/m² 的反例会失败；极夜正 GHI 同样失败。

这条辐射关系的系数是未校准情景先验。晴空近似和太阳积分支持源于 [任务05](mechanism_research/task_05_cloud_rainfall_radiation.md) 所列 FAO-56、辐射模型及云边增强文献；本项目的 `GHI ≤ 水平TOA` 仅是**无侧向辐射、无曙暮光的平面平行太阳模式上限**，不是现实局地辐射的普适定律。

**S：湿度强迫与饱和调整。** 当前 q 是外部规定的情景强迫。若逐小时冷却导致过饱和，以显式有界求解减少 q，并保存 `specific_humidity_adjustment_kg_kg`。这个负差值既不是毫米降雨，也不是守恒的云水库存；不把它转换成独立雨量。当前 B 没有宣称大气柱水量或陆气水热闭合，后续动态水文不能将此差值直接加入河流。完整预算需要另设大气柱质量/库存及同一步双边交换记账。

## 3. 空间、时间与边界配置

| 配置 | 默认 | 单位/含义 | 类型 |
|---|---:|---|---|
| `weather.innovation_correlation_length_km` | 10 | 潜在高斯协方差约降至 e^-1 的距离，km | S |
| `weather.synoptic_memory_hours` | 288 | 日驱动潜在扰动的相关记忆时间，h | S |
| `weather.hourly_memory_hours` | 8 | 小时潜在扰动的相关记忆时间，h | S |
| `weather.synoptic_advection_speed_km_per_hour` | 0.12 | 日驱动扰动的情景移动速度，km/h；不是 10 m 风速 | S |
| `weather.hourly_wind_direction_std_degrees` | 15 | 小扰动近似的转向幅度参数，度；非最终风向精确标准差 | S |
| `weather.spatial_boundary` | open | 日驱动平流遇边界时补充独立情景入流 | S |
| `weather.spatial_scale_mode` | physical | physical / legacy | 工程模式 |
| `weather.temporal_scale_mode` | physical | physical / legacy | 工程模式 |

物理模式使用 \(\sigma_{cell}=L/(2\Delta x)\)、\(\rho(\Delta t)=\exp(-\Delta t/\tau)\)、位移 \((\Delta row,\Delta col)=(-v,u)\Delta t/\Delta x\)。网格间距以 km/cell、速度以 km/h、时间以 h 输入。独立 helper 支持半小时；主生成入口继续遵守 A 的固定 1 h 契约。

当前平流对潜在日驱动场进行随机整数位移，以保持边际方差；声明的速度是随机舍入位移的期望，而非每次精确亚格点平流。`open` 平流不会从另一侧绕回，也不会将最边一行无限复制。**创新随机场的滤波在 open 情况下仍使用 reflect 数值延拓**，元数据分别记录 `advection_boundary` 与 `innovation_filter_boundary`；不能把后者称为物理开放边界。`periodic` 明确使用环绕合成域。

`innovation_smoothing_steps`、`synoptic_shift_cells_per_day` 在 `spatial_scale_mode: legacy` 时生效，`advection_rho` 在 `temporal_scale_mode: legacy` 时生效。旧字段可解析、旧位置参数顺序保留；旧配置未写新模式时使用新的 physical 默认，**不保证旧世界逐值复现**。完整快照记录所有模式与参数。配置非有限值、非法枚举、非正物理尺度及不可行衰减率会拒绝。`wet_day_probability` 支持 0/1：0 是显式全干情景，覆盖年均雨量先验；1 为全湿情景。

## 4. 文件和公共接口

- `weather/physics.py`：湿空气正向与逆向诊断，显式数值域校验。
- `weather/random_fields.py`：共用物理尺度高斯场、时间记忆、位移换算。
- `weather/weather_generator.py`：两种小时模式、风统计、云辐射联合可行性、日汇总；修正有界分配器只统计正权重可到达容量。
- `core/config.py` / `configs/small_debug.yaml`：默认、验证和完整快照；`core/contracts.py` 扩展新字段、三种天气文件的统计用途。
- `core/datatypes.py::WeatherStore`：末尾追加 `diagnostics`、`metadata`、`static_elevation_m` 可选字段，九通道和旧构造参数顺序保留。
- `operation/stage_cache.py`：旧无诊断缓存可读；新诊断及元数据完整恢复。
- `dataset/builder.py` / `dataset/loader.py`：附加诊断打包与按 T 切片，避免 H==T 时把静态高程误切成时间序列。
- `scripts/generate_static_world.py`：保存小时实况的 `hourly_daily_summary.npz`；`scripts/validate_world_physics.py`：用真实汇总验收，再按声明的硬约束比较日驱动。

NPZ 新增键为 `diagnostic__specific_humidity_kg_kg`、`diagnostic__sea_level_pressure_hpa`、`diagnostic__air_density_kg_m3`、小时额外的 `diagnostic__specific_humidity_adjustment_kg_kg`，以及 `static_elevation_m`、`weather_metadata_json`；沿用 A 的 `time_bounds_hours`。诊断为 `[T,H,W]`，高程为 `[H,W]`，时间边界为 `[T,2]`。旧缓存没有高程时，小时生成显式采用海平面假设并记入 `elevation_source`，不凭数据数值猜测海拔。

## 5. 验证记录

全部命令使用 Git Bash，Python 为 `/d/anaconda/python.exe`；输出仅位于仓库 `outputs/work_package_b`，临时测试文件也位于仓库。基础环境缺少可选模型依赖 `torch_geometric`，因此预测模型测试不能收集，未安装或改变全局环境。生成器回归命令为：

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
/d/anaconda/python.exe -m pytest -q -p no:cacheprovider tests --ignore=tests/test_forecasting_model.py
```

最终生成器回归 **173 项通过**，包括气候尺度、原始量/诊断、正反向风、联合不可行日输入、正权重支撑、半小时单位换算、元数据/缓存/数据集窗口和旧天气兼容回归。重复 seed 42/123 的小算例验证全部天气与诊断逐值可复现。

已实际运行完整 `small_debug.yaml`（64×64，1 km 网格，365 天驱动，168 小时实况），seed 42 和 seed 123 各通过 83 项独立世界检查。两个世界均完成数据打包、校验和验证和 24/6 小时时间窗口读取。连续周仍按天气子种子随机选取，元数据明确 `hourly_window_selection=seeded_random_contiguous_window_from_daily_anchors`；两例零基起始日分别为 **91、103**，因此它们是不同世界和日期的回归样本，**不是同季节成对干预实验**。后续 G 的成对干预需要另行固定同一世界和时间轴。

| 指标 | seed 42 | seed 123 |
|---|---:|---:|
| 小时温度最小/最大，°C | 0.027 / 30.754 | -7.673 / 26.595 |
| 小时 RH 最小/最大 | 0.217 / 1.000 | 0.140 / 1.000 |
| 地面气压最小/最大，hPa | 749.117 / 980.682 | 638.898 / 922.518 |
| 空气密度最小/最大，kg/m³ | 0.924 / 1.165 | 0.805 / 1.111 |
| 日均速减日矢量模，最小/最大 m/s | 0.012 / 0.821 | 0.011 / 0.632 |
| 硬锁日降雨最大绝对残差，mm | 1.73e-6 | 1.31e-6 |
| 硬锁日温度最大绝对残差，°C | 3.97e-7 | 3.97e-7 |

默认模式中，GHI 驱动与实际日均最大差为 52.58/44.92 W/m²、RH 为 0.0786/0.0756、气压为 0.0956/0.1046 hPa；这些是明确未被硬锁的输出统计变化，不是守恒误差。详情保存于 `outputs/work_package_b/weather_metrics.json` 和各小时文件元数据。

旧 `test_hourly_weather_cloud_systems_are_patchy` 中 `(cloud>.75).mean()<.35` 没有普适物理依据，新物理尺度气候场改变了该情景比例，故替换为非退化空间结构检查，并继续验证云量边界和云遮蔽传递。对云—GHI 的比较使用去除当地晴空辐射差异后的透过率，避免海拔/太阳几何混淆；没有调整模型参数来迎合一个旧种子的贴图比例。

## 6. 边界与后续接口

本工作包不产生动力学云对象，也没有云水液相/冰相库存、三维辐射、热力对流闭合或真实区域参数校准。云、雨、气温存在共享潜在扰动及条件关系，不应解释成大气水量守恒模型。小时云噪声具有连续记忆，尚未实现小时云场完整平流；新日驱动平流不等于卫星云团追踪。不同分辨率在同一坐标处不保证为同一个随机实现，本次验证的是公里尺度响应，不是区域拼接。

文献支持的关系、工程近似和情景参数完整来源分别见 [任务04：气候与地形气象](mechanism_research/task_04_climate_and_orography.md)、[任务05：云雨辐射](mechanism_research/task_05_cloud_rainfall_radiation.md)。下一步动态水文应消费小时降雨区间深度及明确的潜在蒸散强迫，并维持同一积分区间两边使用同一交换通量；当前比湿调整不能替代这笔交换。
