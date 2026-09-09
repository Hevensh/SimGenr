# SimGenr 物理与研究修订说明（physics_v3）

基线仓库：[Hevensh/SimGenr](https://github.com/Hevensh/SimGenr)，基线提交 `c280ea785cb4274a8f67b1d52af409fb177ea6e5`。本次按完整的 14 个生成阶段核对研究与实现，修正量纲、守恒和因果依赖，并将未校准的情景假设与物理关系分开说明。新版本用于生成可检查的合成场景，尚未依据某实际省网/城市的数据完成统计校准。

## 生成顺序

实际执行顺序调整为：

**地形 → 水文拓扑 → 气候 → 静态土地/植被 → 日天气 → 同一次日天气的小时降尺度 → 城市人口 → 土地用途 → 源荷站点 → 母线 → 线路与电气参数 → 源荷实况及基线潮流 → 线路更新 → 储能候选 → 有界规划调度。**

气候只依赖地形和水体，可以先于植被；人口和负荷不应反过来决定年气候。河网此处是静态地形排水骨架，可以先于天气；若未来实现流量、土壤水或水电，必须在气象之后增加动态降雨—径流和水量平衡，不能把汇水面积当瞬时流量。

保留原输出阶段编号以兼容目录读取，元数据记录真实顺序 `[1,2,4,3,5,6,7,8,9,10,11,12,13,14]`。随机数按气候、土地、日天气、小时天气和源荷分别派生；改变年天气模拟长度不再仅因为多消耗随机数就改变后续的负荷扰动。

## 各阶段处置

| 原阶段 | 本次处置 | 依据及完整公式 |
|---|---|---|
| 1 地形 | 默认坐标稳定的多尺度噪声；截止小于两格的波长；坡度/局部起伏使用实际格距 | [地理研究 §1](research_geography.md) |
| 2 水文 | Priority-Flood 数值坡度与湖深分离；D8 无环汇流；面积阈值、米制河宽；河湖床保留当地海拔；HAND 易感性 | [地理研究 §2](research_geography.md) |
| 3 静态土地 | 移至气候之后，植被受温度/水分约束，保留建设与保护规则 | [地理研究](research_geography.md) |
| 4 气候 | 实际纬度梯度、海拔递减、气象风向 FROM 方位与迎风坡、太阳几何约束 | [气候天气研究](research_weather.md) |
| 5 天气 | 平稳相关扰动、湿日 Markov–gamma 雨量、气压/湿度物理关系；小时雨量加和与其他变量均值匹配日值，夜间辐照为零 | [气候天气研究](research_weather.md) |
| 6 城市 | 人口单位 persons/km²、总人口守恒，不再把人数再次压成 0–1 | [地理研究](research_geography.md) |
| 7 用地 | 相对活动强度与真实人口密度分离；区划仍为透明启发式 | [地理研究](research_geography.md) |
| 8 源荷站点 | 装机由可用 km² 与功率密度约束，避免重复分配同一土地；距离硬约束；负荷总量由人口决定 | [地理研究](research_geography.md) |
| 9 母线 | 保留人口/电源驱动的地理位置；火电初装机作为候选组合，最终检查是否充足 | [电网储能研究](research_grid_storage.md) |
| 10 线路 | 统一传输网电压等值；50 Hz 标准导线 R/X/C/I；额定容量由 √3UI 推导 | [电网储能研究](research_grid_storage.md) |
| 11 源荷与潮流 | 风速高度/空气密度/机组曲线；GHI 分解与 POA、Faiman 温升、DC/AC；日历/温度记忆/时空负荷；逐岛 DC 平衡 | [源荷研究](research_source_load.md)、[电网储能研究](research_grid_storage.md) |
| 12 电网更新 | 修复重复缩放阻抗及旁路容量/阻抗不一致；保留走廊启发式并重算潮流 | [电网储能研究](research_grid_storage.md) |
| 13 储能选址 | 事件 AC 能量换算为 SOC 可用区间与效率约束下的铭牌容量；候选覆盖仍为启发式 | [电网储能研究](research_grid_storage.md) |
| 14 调度与扩建 | 逐岛 PTDF、母线级缺供、SOC/爬坡/互斥、正确弃电账；区分循环和给定初态 | [电网储能研究](research_grid_storage.md) |

每份分项报告都给出作者/机构、原始研究或官方技术资料链接、对应公式、此次实现与文献模型的区别。不把配置中的权重、容量密度、气候均值、负荷温敏系数或相对成本宣称为普适的“真实参数”。

## 预测任务的数据语义

`generate_source_load_forecast` 为兼容保留名称，输出实际是**以合成实况天气为条件的源荷实现值**，不是有发布日期、提前量和气象预报误差的预测。Stage14 则是全窗口已知的规划调度。

数据集升级到 0.6.0，保留原字段，同时增加不含储能操作的 `exogenous_p_load_mw`、原始 `exogenous_p_gen_available_mw`、`exogenous_p_renewable_available_mw` 和母线映射掩码。源荷预测应明确选用这些外生字段；`node_dynamic` 原有最终调度负荷包含充电，图及扩建也使用过整周实况。当前模型训练接口保留原目标，不能直接把其分数解释为无未来信息的真实运营预测性能。详见 [数据说明](../datasets/DATA_DESCRIPTION.md)。

要做严格预测实验，还需要真实/独立的 NWP 或气象预报误差模型、发报时点、规划期与测试期隔离，以及按世界/年份拆分的数据。区域校准建议至少检验负荷持续曲线、日周季节周期、温度响应、爬坡分布、自相关、空间互相关、风光容量因子及联合极端事件；公式和不变量通过不能代替这些实证检验。

## 复现

在仓库根目录的 PowerShell 中，使用已有 D 盘 Anaconda：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $PWD 'outputs/.mplconfig'
$env:TEMP = Join-Path $PWD 'outputs/.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& D:/anaconda/python.exe -B scripts/generate_static_world.py --config configs/small_debug.yaml --seed 42 --no-figures
& D:/anaconda/python.exe -B scripts/validate_world_physics.py outputs/small_debug_seed42
```

更换季节可用 `--start-day 180`（零基日号，0–364）；同种子多个季节应使用不同 `--output` 根目录以保留各次结果。小时采用当地太阳时、365 日气候年；没有 UTC、时区/DST 或严格闰年气候时钟。

完整运行后各世界目录内的 `physics_validation.md/.json` 给出逐项最大误差与缺供/弃电统计。可用 `--from-stage 13/14 --no-figures` 恢复新版缓存。旧版缓存不含新版物理语义，必须完整再生成；修改上游配置也需要重新生成。修改储能配置后应从阶段 13 重新定容。

生成只需 [requirements-generator.txt](../requirements-generator.txt)。原有模型回归测试另需 PyTorch/PyG；本次测试仅把缺少的 PyG 安装到仓库 `outputs/.test-deps`，未修改外部 Conda。验证使用 `python -B`，数值线程层可在本进程设置 `MKL_THREADING_LAYER=SEQUENTIAL`，避免本机现有 PyTorch/MKL 运行库并用时的冲突。详见 [验证结果](VALIDATION_RESULTS.md)。
