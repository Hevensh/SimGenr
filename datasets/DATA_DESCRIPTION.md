# SimGenr 数据内容说明

## 1. 收录原则

数据集面向可迁移到真实观测的多模态预测任务。模型输入只保留：

1. 可直接观测或统计获得的量；
2. 能从观测量确定性计算的物理量，例如坡度、流向和线路负载率；
3. 构成仿真场景真值的设施、电气参数和运行状态。

生成器内部的启发式评分不进入样本，包括宜居度、城市核心适宜度、滨水便利度、建设适宜度、能源适宜度、外部性、候选密度、线路建设成本和土地用途倾向。这样可避免模型读取现实中不可获得的隐藏规划分数。

一个 NPZ 包含四组数据：

| 前缀 | 模态 |
|---|---|
| `static__` | 静态地理、土地和社会栅格 |
| `dynamic__` | 168 小时天气栅格 |
| `graph__` | node、line 及电气拓扑 |
| `operation__` | node、line、storage 的小时运行状态 |

符号：`H=W=64`，`T=168`，`N` 为 node 数，`E` 为 line 数，`S` 为储能站数。每格代表 1 km×1 km。

## 2. Static 静态栅格

### 2.1 连续栅格

`static__continuous`：`float32 [22,H,W]`。通道顺序保存在 `static__continuous_channels`。

| 通道 | 含义/单位 |
|---|---|
| `elevation` | 高程，m |
| `slope` | 坡度，rise/run |
| `aspect_sin`, `aspect_cos` | 坡向正弦/余弦 |
| `roughness` | 3×3 局部高程极差归一化值，0–1 |
| `curvature` | 高程离散拉普拉斯曲率，约 1/m |
| `flow_accumulation` | 上游汇流栅格累计量 |
| `water_depth` | 河湖相对水深 |
| `hydrology_elevation` | 水文计算采用的地表高程，m |
| `distance_to_water` | 至最近河流或湖泊距离，km |
| `vegetation` | 相对植被覆盖度，0–1 |
| `mean_temperature` | 气候平均温度，°C |
| `annual_temperature_amplitude` | 年温度振幅，°C |
| `mean_humidity` | 气候平均相对湿度，0–1 |
| `prevailing_wind_u`, `prevailing_wind_v` | 盛行风分量，m/s |
| `mean_precipitation` | 年平均降水量，mm/year |
| `mean_cloud` | 气候平均云量，0–1 |
| `mean_irradiance` | 气候平均辐照度，W/m² |
| `population_density` | 相对人口密度，0–1 |
| `economic_activity` | 相对经济活动强度，0–1 |
| `urban_density` | 相对建成区密度，0–1 |

### 2.2 离散栅格

`static__categorical`：`int32 [7,H,W]`。通道顺序保存在 `static__categorical_channels`。

| 通道 | 编码 |
|---|---|
| `flow_direction` | D8：0北、1东北、2东、3东南、4南、5西南、6西、7西北，-1 汇点 |
| `water_type` | 0陆地、1河流、2湖泊；湖泊覆盖重叠的河流格 |
| `watershed_id` | 分水岭类别 ID |
| `land_cover` | 1平原、2丘陵、3山地、4水体、5湿地、6保护地 |
| `urban_mask` | 建成区，0/1 |
| `city_id_map` | 城市 ID，-1 表示不属于城市 |
| `land_use_zone` | 0背景、1居住、2商业、3工业、4农业、5公园；水体/保护地由其他通道表达 |

类别 ID 不具有连续大小关系。`watershed_id`、`city_id_map` 和 `land_use_zone` 应采用 embedding、one-hot 或掩码编码。

## 3. Dynamic 天气栅格

`dynamic__weather`：`float32 [T,9,H,W]`，布局为 TCHW。通道顺序保存在 `dynamic__weather_channels`。

| 通道 | 含义/单位 |
|---|---|
| `wind_u`, `wind_v` | 东西/南北风速分量，m/s |
| `wind_speed` | 风速大小，m/s |
| `temperature` | 近地面温度，°C |
| `humidity` | 相对湿度，0–1 |
| `pressure` | 地面气压，hPa |
| `cloud` | 云覆盖率，0–1 |
| `precipitation` | 小时降水量，mm/hour |
| `irradiance` | 地表辐照度，W/m² |

| 其他字段 | 形状 | 含义 |
|---|---:|---|
| `dynamic__timestamps` | `[T]` | 从当年开始累计的小时编号 |
| `dynamic__start_day_of_year` | 标量 | 本周起始日序号 |
| `dynamic__time_unit` | 标量字符串 | `hour` |
| `dynamic__weather_class` | `[T,H,W]` | 0晴/少云、1多云、2降雨、3强降雨、4强风 |

`weather_class` 是连续天气场的阈值派生标签，适合作为辅助分类目标，不应替代连续物理输入。

## 4. Graph：Node 和 Line

图结构只有 node 和 line。A* 路径只是 line 的可选空间轨迹，不是独立实体，也不是基础结构参数。

### 4.1 Node 类型

| `node_type` | 类型 | 含义 |
|---:|---|---|
| 0 | `load_bus` | 负荷汇集节点 |
| 1 | `wind_bus` | 风电并网节点 |
| 2 | `pv_bus` | 光伏并网节点 |
| 3 | `thermal_bus` | 火电并网节点 |
| 4 | `transit_bus` | 不直接发电/用电的中转节点 |

储能不是第六种 node，而是通过 `site_bus_ids` 挂接到已有 node。

### 4.2 Node 静态字段

| 字段 | 形状 | 含义 |
|---|---:|---|
| `graph__node_id` | `[N]` | 业务 bus ID |
| `graph__node_type` | `[N]` | node 类型编号 |
| `graph__node_source_id` | `[N]` | 对应原始设施/候选关系 ID；仅用于关联 |
| `graph__node_features` | `[N,5]` | `row, col, x_km, y_km, capacity_mw` |
| `graph__node_electrical` | `[N,6]` | 下表电气属性 |

`graph__node_electrical` 列顺序：

| 列 | 名称 | 单位 |
|---:|---|---|
| 0 | `nominal_kv` | kV |
| 1 | `p_capacity_mw` | MW |
| 2 | `q_capacity_mvar` | Mvar |
| 3 | `base_load_mw` | MW |
| 4 | `power_factor` | 无量纲 |
| 5 | `voltage_setpoint_pu` | p.u. |

### 4.3 Line 静态字段

Line 不做人为类别划分。

| 字段 | 形状 | 含义 |
|---|---:|---|
| `graph__edge_id` | `[E]` | 业务 line ID |
| `graph__edge_index` | `[2,E]` | 从 0 开始的本地 node 下标，用于 GNN |
| `graph__edge_bus_ids` | `[2,E]` | from/to 业务 bus ID |
| `graph__edge_features` | `[E,7]` | 线路连续属性 |

`graph__edge_features` 列顺序：

| 列 | 名称 | 单位 |
|---:|---|---|
| 0 | `route_length_km` | km |
| 1 | `nominal_kv` | kV |
| 2 | `electrical_length_km` | km |
| 3 | `r_ohm` | Ω |
| 4 | `x_ohm` | Ω |
| 5 | `b_us` | μS |
| 6 | `rate_mva` | MVA |

### 4.4 A* 辅助几何

| 字段 | 形状 | 含义 |
|---|---:|---|
| `graph__path_ptr` | `[E+1]` | 每条 line 在扁平路径数组中的边界 |
| `graph__path_row`, `graph__path_col` | `[sum_path_cells]` | 路径经过的栅格坐标 |

基础图模型可完全忽略这三个字段。拓扑只由 `node_id`、`edge_index` 和 line 属性定义。

## 5. Operation 运行数据

### 5.1 Node 多通道时序

`operation__node_dynamic` 为 `float32 [T,10,N]`。通道顺序保存在 `operation__node_dynamic_channels`，node 顺序保存在 `operation__bus_ids [N]`。

| 通道 | 含义/单位 |
|---|---|
| `p_load_mw` | 请求负荷，MW |
| `p_gen_available_mw` | 可用发电功率，MW |
| `p_gen_scheduled_mw` | 计划发电功率，MW |
| `q_load_mvar` | 无功负荷，Mvar |
| `bus_angle_rad` | DC 节点相角，rad |
| `bus_p_injection_mw` | 净有功注入，MW；发电为正 |
| `served_load_mw` | 实际供给负荷，MW |
| `dispatched_generation_mw` | 实际调度发电，MW |
| `unserved_load_mw` | 未满足负荷，MW |
| `curtailed_generation_mw` | 削减发电，MW |

应按业务 ID 与 `graph__node_id` 对齐，不依赖偶然列顺序。

### 5.2 Line 多通道时序

`operation__line_dynamic` 为 `float32 [T,3,E]`。通道顺序保存在 `operation__line_dynamic_channels`，line 顺序保存在 `operation__branch_ids [E]`。

| 通道 | 含义/单位 |
|---|---|
| `line_flow_mw` | 有符号潮流，正方向为 from→to，MW |
| `line_loading_ratio` | 最终规划调度后的线路负载率 |
| `baseline_line_loading_ratio` | 最终扩建调度前的线路负载率 |

`operation__line_capacity_expansion_mva [E]` 单独保存规划增加容量，因为它不是小时序列。

### 5.3 Storage

| 字段 | 形状 | 含义/单位 |
|---|---:|---|
| `site_ids` | `[S]` | 储能站 ID |
| `site_bus_ids` | `[S]` | 挂接的 node ID |
| `site_power_capacity_mw` | `[S]` | 功率容量，MW |
| `site_energy_capacity_mwh` | `[S]` | 能量容量，MWh |
| `charge_mw` | `[T,S]` | 充电功率，MW |
| `discharge_mw` | `[T,S]` | 放电功率，MW |
| `emergency_discharge_mw` | `[T,S]` | 快速备用放电，MW |
| `soc_mwh` | `[T+1,S]` | 小时边界 SOC，MWh |

## 6. 内嵌元数据

| 字段 | 含义 |
|---|---|
| `metadata_json` | 通道名、统计量、维度、单位和校验结果 |
| `config_yaml` | 生成该 seed 时的完整配置快照 |

数据 schema 版本为 `0.5.0`。
