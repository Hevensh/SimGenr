# 工作包 G：可执行成对机制干预

本协议通过现有 B、D、E、F 数值函数，检查控制变量变化是否产生预期机制响应。固定小世界提供可解析反例；它不估计真实地区参数、预测技巧、故障概率或现实电网可靠性。本子项只增加协议脚本、测试与本文档，不更改世界生成公式。

## 运行与产物

在仓库根目录使用 Git Bash 执行：

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=outputs/.implementation-tools/mpl
export TMP=outputs/.implementation-tools/tmp TEMP=outputs/.implementation-tools/tmp
/d/anaconda/python.exe -B scripts/validate_mechanism_interventions.py \
  --seeds 42 123 --output-dir outputs/work_package_g/interventions
/d/anaconda/python.exe -B -m pytest -q -p no:cacheprovider tests/test_mechanism_interventions.py
```

输出目录内生成 `mechanism_interventions.json` 与 `mechanism_interventions.md`。默认输出目录为 `outputs/validation/mechanism_interventions`。脚本在执行数值函数之前解析并检查目标路径，拒绝写出仓库边界。运行期间不下载数据、不拟合观测、不安装依赖。

JSON 的 `schema_version` 为 `mechanism_interventions_v1`。每个已执行场景记录种子、阶段、处理字段及单位、预期方向、前后数值、检查残差、实际调用函数、控制变量哈希和模型限制。`relation_class` 中 P 表示物理关系，E 表示经验规律，S 表示场景先验；标签不表示该案例已经现实校准。

状态严格区分：

| 状态 | 含义 | 对执行结论的影响 |
|---|---|---|
| PASS | 此场景实际执行，列出的关系及控制变量检查均通过 | 计入已支持检查 |
| FAIL | 实际关系检查失败，或数值函数/公共夹具执行异常 | `supported_execution_status=FAIL`，CLI 返回 1 |
| UNSUPPORTED | 当前数值模型不提供所需物理能力 | 单独计数，不伪装成通过 |
| NOT_RUN | 公共夹具创建失败，依赖它的场景未执行 | 保留原因；创建失败本身另记 FAIL |

正常完成时 CLI 返回 0，表示列出的已支持干预均执行通过。即使返回 0，`coverage_status` 仍为 `PARTIAL_WITH_EXPLICIT_UNSUPPORTED`，不能解释为全部物理机制已验证。异常场景保留失败报告，后续独立场景继续执行。

## 成对控制与随机数

每一对使用相同静态资产与小时时间轴。主要夹具为 1 × 5 网格、格边长 1 km、48 个小时区间；备用夹具为 2 节点、1 小时，时间支持均为 `[t,t+1)`。两种子使用相同确定性小世界，目的是改变现有模块的随机实现后重复同一干预，而不是抽样两个地区。

随机函数复用 `core.random_state.derive_module_seed(world_seed, module_name)`。天气使用已有 `weather_hourly` 模块名，源荷使用已有 `source_load` 模块名。每对分别从同一派生种子创建两个生成器，不使用新种子算法，不依赖先前案例消耗的随机数。报告记录世界种子、模块名、派生种子、派生方法及初末随机状态哈希；末状态一致是随机调用路径一致的检查。源荷案例还直接比较输出的负荷随机残差，确保处理前后保持同一实现，而不只比较种子声明。

动态水量与固定资产运行核不接收随机数，因此报告明确标记为确定性函数。故障案例先生成并固定同一个源荷实现，之后只改变线路状态。成对资产哈希在运行前后重算，时间轴也分别计算哈希并直接比较，避免仅记录一个“固定资产”的文字声明。

## 实际执行的九类干预

| 案例 | 实际函数 | 处理与分类 | 检查内容及范围 |
|---|---|---|---|
| `reverse_wind_scalar_vector` | `generate_hourly_weather_week`、`aggregate_daily_weather` | 日均矢量由 `(4,0)` 改为 `(0,0)` m/s，日均标量仍为 4 m/s；P/S | 日条件小时构造前后半天反向，标量均值仍为 4，矢量均值抵消为 0；不表示现实转向频率 |
| `zero_rain_water_budget_and_disable` | `generate_dynamic_hydrology` | 初态全零，首小时每格 1 mm 雨改为全程零雨；另关闭动态水文；P/S | 从实际状态和通量独立重算逐格及全域预算；零雨无库存自增；关闭返回静态模式 `None` |
| `clear_cloud_to_radiation_to_pv` | B 小时天气、E 源荷完整函数 | 云份额 0.6 改为 0，并同步改变可行的日 GHI 条件；P/E/S | 实际小时 GHI 与光伏可用电量增加；保持源荷随机实现；PV 不重复扣除云量 |
| `solar_driver_ablation` | `generate_source_load_forecast` | 小时 GHI 置零；P/S | PV 为零，负荷不变；这是转换驱动消融，不把其余天气不变的白天零 GHI 称为完整大气情景 |
| `temperature_step_memory_load` | `diagnose_moist_air`、E 源荷函数 | 第 24 小时起原始温度加 20 K；固定比湿及海平面气压，重新诊断 RH、气压、空气密度；P/E/S | 干预前负荷不变，首小时有效温度增量小于外温阶跃，随后制冷需求增加；固定 GHI 下热组件 PV 减少 |
| `shared_wind_to_wind_generation` | `generate_source_load_forecast` | `u/v` 同乘 1.4，并重算 `hypot(u,v)`；P/E/S | 轮毂风和本案例额定以下风机可用电量增加；负荷残差与需求不变。结论不外推到切出以上 |
| `line_fault_local_unserved_energy` | `solve_dc_power_flow`、`dispatch_storage_week` | 固定资产、固定源荷，第 24 小时后断开指定持久线路 ID；P/S | 故障线路零潮流，缺供仅在断开的负荷孤岛，另一负荷继续被供给，所有扩容数组为零 |
| `initial_soc_reserve_boundary` | `dispatch_storage_week` | 初始储能由 0 改为 2 MWh，设备不变；P/S | 持有备用受能量和效率限制，SOC 守恒，不与充电共存，显式报告备用短缺 |
| `reserve_response_boundary` | `dispatch_storage_week` | 备用响应时间由 0.1 h 改为 1 h，初始储能均为 2 MWh；P/S | 备用受净功率爬坡与能量持续时间的共同约束，不能仅按铭牌功率声称可用 |

云实验遵守日条件小时模型的输入语义：清空云同时提高日 GHI 条件，避免要求一个更晴朗的小时过程仍精确聚合到原阴天的日辐射预算。它检验明确的联合干预链，不声称只改云标签就能绕过日辐射边界。天气条件将温度日振幅、温度噪声与风向扰动设为零，使该解析对照中的这些因素保持固定；源荷随机残差仍实际生成。

风干预是条件转换实验：改变同一个小时风向量驱动风机，PV 模块散热也接收该物理风字段。它没有重跑大气平流，不能解读为完整动力天气响应。温度干预同理重新诊断耦合热湿字段，但不估计热浪发生概率。

## 单位与解析基准

降雨通量使用每个实际网格的面积：`雨量 mm × 格面积 km² × 1000 = m³`。水量测试直接读取 S/G 深度状态与 channel/lake 体积状态，以及雨、ET、边界入出流、逐格路由和湖泊混合交换，重建相应总账；不将静态汇水面积再次乘入雨量，也不直接把核导出的残差为零当作独立验证。该小夹具没有湖泊，因此湖位—面积—体积与多格湖混合的验收继续由 D 独立检查器和 D 解析测试承担。

48 小时 GHI 能量用小时均值 W/m² 乘以 1 h 后求和，单位 Wh/m²；源荷和缺供电量用 MW 乘以 1 h 后求和，单位 MWh。水量为 m³，风为 m/s，温差为 K。报告明确时间支持，不用 MW 数值冒充跨区间电量。

备用解析夹具中设备功率 4 MW、能量 2 MWh、放电效率 0.95、备用持续 2 h、净功率爬坡比例 0.25/h。这些都是 S 类测试先验。可持有备用能量上界为 `0.95 × 2 MWh / 2 h = 0.95 MW`；响应上界为 `0.25/h × 4 MW × 响应时间 h`。因此响应 0.1 h 时为 0.1 MW，1 h 时受能量限制为 0.95 MW。备用需求为 2 MW，短缺分别为 1.9 和 1.05 MW。测试同时重建实际 SOC 方程，避免仅检查一条期望标量。

种子 42 的一次实际运行结果如下；测试主要断言关系、解析边界与容差，不依赖该种子专属的随机负荷小数：

| 指标 | 基线 | 干预后 | 单位 |
|---|---:|---:|---|
| 反向风日均矢量模长 | 4 | 0 | m/s |
| 反向风日均标量速度 | 4 | 4 | m/s |
| 全域总雨量 | 5000 | 0 | m³ |
| 48 h 平均格 GHI 能量 | 10266.595 | 17342.219 | Wh/m² |
| 云/GHI 联合干预下 PV 可用电量 | 209.034 | 343.654 | MWh |
| 加温后第二天请求电量 | 3043.555 | 3882.655 | MWh |
| 首小时有效温度响应增量 | — | 2.350 | K |
| 轮毂风速 | 5.441 | 7.617 | m/s |
| 风机可用电量 | 77.904 | 241.133 | MWh |
| 指定断线后的全域缺供电量 | 0 | 1512.427 | MWh |
| 未断开负荷的缺供电量 | 0 | 0 | MWh |

## 检查器自身验证与实际验收

定向测试不是只重复成功运行：它故意将实际 E 函数返回的 PV 可用功率破坏为零，确认云/GHI 对照出现 FAIL；另将案例函数替换为运行异常，确认错误不会被记成 UNSUPPORTED。公共夹具异常会产生 FAIL 和依赖案例的 NOT_RUN；CLI 失败仍保存报告并返回非零码。还有相同种子整份报告逐字节内容哈希重现、两种子负荷随机实现确有区别、派生种子来源、单位、输出路径边界和 JSON/Markdown 写入检查。

本子项验收实际运行了种子 42 与 123：18 PASS、0 FAIL、3 UNSUPPORTED、0 NOT_RUN；`tests/test_mechanism_interventions.py` 共 15 项通过。报告中的三个未支持能力为备用激活后的网络可送达性、AC 电压/频率安全、曙暮光 PV。无曙暮光模型下的夜间零值仅是模型边界，不能称为太阳中心位于地平线下的一切自然散射辐射都为零。

本协议的通过不能替代主 G 的完整世界、分辨率、模式、恢复缓存和独立守恒检查。尤其本协议不进行全故障枚举，不估计概率可靠性指标，不验证 AC 损耗、电压或频率动态，也不将孤岛聚合备用当作网络可送达备用。未支持条目明确保留，供后续模型扩展后升级为实际执行案例。
