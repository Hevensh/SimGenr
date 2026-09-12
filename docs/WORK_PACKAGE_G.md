# G：机理验证框架与最终验收

对应综合报告 A8、任务10。G 增加诊断和实验协议，不修改地形、天气、水量、设备或调度公式，也没有放宽已有物理容差。P 为被检查的守恒、几何和边界关系；E/S 用于经验传递与受控情景。通过给定机制检查不等于真实地区校准。

## 修改文件、参数和字段

| 文件 | 原因与实现 | 最小验证 |
|---|---|---|
| `scripts/validation_checks.py` | 四态、显式轴、最大原始残差/违规的分别定位 | 非有限数、相对容差、T=N、不适用空数组、0条扩展位置 |
| `scripts/validate_world_physics.py`、三个领域验证 helper | 注明阶段、字段、支撑；动态水文关闭与未支持过程不记PASS | 六个完整世界；导出篡改反例 |
| `core/config.py`、`configs/small_debug.yaml` | `validation.max_failure_locations=5`，整数0..1000，仅控制失败详情条数 | 配置快照和非法配置 |
| `scripts/validate_mechanism_interventions.py` | 同资产/时间/RNG下的真实生成函数配对 | 两seed各9种处理，共18PASS；故意破坏PV时FAIL |
| `scripts/validate_world_matrix.py` | 种子、物理范围、分辨率、开关、模式的描述性汇总 | 六个世界、缺少完整产物的失败记录 |
| `core/errors.py`、`operation/storage_dispatch.py` | 仅LP/MILP status2归物理不可行；其他求解失败分开 | 真实严格孤岛与求解器状态反例 |
| `scripts/generate_static_world.py` | 失败记录后仍非零退出；成功清除残留标记 | CLI、已有缺供有效结果、禁止写到未确认目录 |
| `dataset/builder.py` | 有失败标记的世界不可打包旧残留 | 不读取旧产物、不建立伪成功数据集 |
| `tests/test_validation_framework.py`、`test_mechanism_interventions.py`、`test_generation_failure_reporting.py`、`test_world_matrix.py` | 可运行的最小反例 | 全生成器回归 |
| 两个既有验证测试 | 无湖/无项目的空数组改为断言NOT_RUN，必要水量/面积账仍必须PASS | 未删除物理失败断言 |
| `scripts/generate_all.sh`、`README.md` | 当前含空格工作区的批量调用保持单个默认config参数；说明v4入口 | Bash语法与6次替身argv验证 |

详细接口见 [四态与定位](WORK_PACKAGE_G_VALIDATION.md)、[成对实验](WORK_PACKAGE_G_INTERVENTIONS.md)、[失败分类](WORK_PACKAGE_G_FAILURES.md)。`max_failure_locations=0` 仍保留主最大位置，只省略额外失败点；不改变PASS/FAIL判断。没有新增生产随机流；实验使用现有 `derive_module_seed`。

新增报告字段包含 `status`、兼容 `passed`（True/False/None）、`max_residual`、`max_violation`、两种最大值各自的位置/时间、字段名、阶段、P/E/S及工程简化。数值位置使用明确数组轴；时间不能从T=N或长度相似猜测。聚合统计/静态量/未实现过程明确标记时间不适用。功率残差为MW，SOC为MWh，水量为m³；它们不能用一个全域相对比例掩盖局部错误。

`generation_failure.json` 记录 `FAIL`、类别、阶段、异常、seed、配置路径与展开配置。给定模型约束被求解器判不可行记 `PHYSICAL_INFEASIBILITY`；非法输入和一般生成/求解错误分开。验证器和打包器发现标记后拒绝读取旧NPZ；成功生成才会清除它。这个标记并不宣称真实物理世界不可行。

## 成对实验

两个seed（42、123）各使用固定资产与小时轴执行以下9类真实函数调用；保存静态/时间/RNG状态哈希、处理变量、前后指标、单位及局限性：

1. 反向风：日均向量抵消，标量均速保持4m/s。
2. 清空云与可行GHI锚点共同重生成：先天气、后PV，辐照和PV能量增加。
3. 零辐照转换消融：PV为零，不冒充完整大气实况。
4. 第二日温度阶跃：统一重新诊断湿空气，前一日负荷不受未来处理影响，热记忆延迟响应。
5. 亚额定风速增加：轮毂风与风电增加，温度负荷保持；不外推到切出区。
6. 零雨及动态过程关闭：无初始库存无凭空造水，局地/全域水账闭合，关闭返回静态模式。
7. 固定线路退出：仅断开的负荷失供，其他母线不能借全局比例重新分配。
8. 初始SOC改变：备用受可用能量/效率/持续时间限制。
9. 备用响应时间改变：响应爬坡限制可提供功率。

实际结果18PASS、0FAIL、3UNSUPPORTED。协议不支持AC/频率、备用激活网络可送达及真实地区参数校准；没有把这些项目计入PASS。

## 真实世界矩阵与数值摘要

| 世界 | 网格/时长 | PASS | FAIL | UNSUPPORTED | NOT_RUN | 缺供MWh |
|---|---|---:|---:|---:|---:|---:|
| seed42 默认 | 64²×1km，168h | 334 | 0 | 4 | 0 | 0 |
| seed123 默认 | 64²×1km，168h | 309 | 0 | 4 | 2 | 0 |
| seed42 2km | 32²×2km，168h | 328 | 0 | 4 | 0 | 0 |
| seed42 静态水文 | 32²×2km，24h，fixed | 236 | 0 | 4 | 22 | 917.263475 |
| seed42 固定资产/故障/备用 | 64²×1km，24h | 335 | 0 | 4 | 0 | 2737.678448 |
| seed42 事前规划/故障/备用 | 64²×1km，24h | 334 | 0 | 4 | 0 | 863.187421 |

各地图面积均4096km²。1km/2km seed42人口积分分别2,635,759.437 / 2,528,071.022人，请求负荷412,740.824898 / 404,273.579756MWh。此处只验证物理尺度和各自守恒，不要求跨网格人口、城市或像素相同，也不据两张网格宣称分辨率收敛。

默认seed42的KCL最大残差1.0573e-11MW（小时索引85、母线数组位置53、当地太阳时第2269小时）；SOC递推最大残差7.8160e-14MWh。seed123分别9.8908e-12MW、4.5475e-13MWh。seed123两条湖泊锚点检查无适用湖对象，正确为NOT_RUN；静态水文场景也没有伪造动态水账PASS。

G相对于F，每个默认世界25份NPZ、445个数组逐一精确相同。所有seed42/123检查都有非空阶段和涉及字段。默认世界明确不支持的4项为AC无功/电压、频率、备用激活可送达、真实区域校准。

## 命令、产物与未覆盖范围

以下为实际Git Bash命令结构，Python使用 `/d/anaconda/python.exe`；测试临时目录和Matplotlib缓存均指定为仓库outputs。完整日志在 `outputs/work_package_g/`。

```bash
python -B -m pytest tests --ignore=tests/test_forecasting_model.py -q -p no:cacheprovider
python -B scripts/generate_static_world.py --config configs/small_debug.yaml --seed 42 --output outputs/work_package_g --no-figures
python -B scripts/generate_static_world.py --config configs/small_debug.yaml --seed 123 --output outputs/work_package_g --no-figures
python -B scripts/validate_world_physics.py outputs/work_package_g/small_debug_seed42 outputs/work_package_g/small_debug_seed123
python -B scripts/validate_mechanism_interventions.py --seeds 42 123 --output-dir outputs/work_package_g/interventions
python -B scripts/validate_world_matrix.py outputs/work_package_g/small_debug_seed42 outputs/work_package_g/small_debug_seed123 --output outputs/work_package_g/world_matrix.json
python -B scripts/build_dataset.py --input-root outputs/work_package_g --output outputs/work_package_g/dataset --worlds small_debug_seed42 small_debug_seed123
```

尺度情景由small_debug全部配置复制，仅改world为32×32、cell_size_km=2和world_name；静态水文探针在该配置上关闭hydrology_dynamic、planning=fixed_assets、hourly_week_days=1。因此开关探针不用于和默认168h的调度性能作因果比较。F固定/事前情景的配置和恢复证据见F文档。

失败标记防护追加后的最终完整回归 **521项通过（47.43s）**；数据集0.10.0两世界48个24+6h窗口、校验和、SOC、前序功率及外生区间/窗口MWh全部通过。精简机器可读证据见 [validation_results.json](validation_results.json)。可选预测模型测试因缺 `torch_geometric` 未执行，未伪造通过、未在工作区外安装包。仅诊断修改无需生成辅助图；水量、能量和故障证据来自数组和解析反例。

兼容 `Checks` 的旧方法和 `passed` 字段；消费者应改用四态status，不能把None解释成通过。NPZ和模型/可视化接口保留，失败标记是防止消费陈旧数据的新增边界。总体PASS仅代表已执行、被支持的约束。完整限制和下一步优先级见 [最终总结](IMPLEMENTATION_SUMMARY.md)。
