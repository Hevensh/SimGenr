# F 运行与资产边界接口

本文记录公共 Store、缓存、字段契约和数据集接口。
数值调度、规划模式和全世界验收分别由 F 数值子文档及主记录汇总。
新增字段追加到旧构造参数之后，旧 NPZ 与 API 保持可读。

## 1. 文件和入口

- `core/operation_contracts.py` 定义具名字段、单位、维度、时间支撑与 P/E/S。
- `core/datatypes.py` 扩展 `PowerFlowStore` 与 `StorageDispatchStore`。
- `core/contracts.py` 将两类完整字段契约加入公开 JSON。
- `operation/stage_cache.py` 检查资产来源、内容哈希和持久运行状态。
- `dataset/builder.py`、`loader.py` 保留旧通道并增加三个独立组。
- `tests/test_operation_interfaces.py` 验证非法数据、时间窗口和缓存篡改。

## 2. Store 与序列化

两个 Store 末尾均追加 `operation_arrays={}`、`operation_metadata={}`。
两个字典均空时保留旧文件，不生成全零 F 附录冒充新结果。
现代输出的字段名为 `op__{name}`，内部数值核字典不带前缀。
声明为 `operation_schema_version=operation_v1`。
`operation_field_schema_json` 保存明确的附录字段名单。
`operation_metadata_json` 含 `schema_version`、`store_kind` 及数值核语义。
`operation_time_bounds_hours[T,2]` 明确每小时区间。
`operation_store_field_schema` 同时定义旧字段和新增字段的完整时间支撑。
`from_arrays` 拒绝残缺声明、未知字段、非法维度、重复实体 ID 和非有限值。

## 3. 运行量和适用范围

PowerFlow 附录含外生/输入口径请求和可用发电、可再生弃电、热电回调与未用可用量。
逐时孤岛标签为 `[T,N]`，线路投运状态为 `[T,E]`。
原 PowerFlow 字段保留历史求解口径；dispatch 后新 op 字段明确物理供需口径。
StorageDispatch 的 op 请求、送达、未送达均为外生 gross 负荷，不含储能充电。
op 物理发电可用量和出力不把储能逆变器当成一次发电设备。
风光弃电、热电相对计划回调、热电未用可用量分别存储，避免相互冒称。
热机 `[T,K]` 可用量、实际出力和备用与母线 `[T,N]` 分账同时保存。
`bus_ids[N]`、`bus_kinds[N]`、热机 ID 和站点 ID 指明适用实体。
`thermal_installed_capacity_mw[K]`、`thermal_land_limit_mw[K]` 为冻结属性/场景上界。
备用需求和短缺在每个孤岛的最小母线 ID 处记账，其余位置为零。
备用是受容量、能量和爬坡约束的岛内聚合能力，不宣称网络可送达或频率稳定。

## 4. 储能时间边界

充电、总放电及其中的应急放电分别为 `[T,S]` 区间平均 MW。
`soc_mwh[T+1,S]` 包含初边界和末边界，单位 MWh。
`op__initial_soc_mwh[S]` 必须与 SOC 第零边界一致。
`previous_storage_net_mw[S]` 和 `previous_thermal_mw[K]` 为前一区间功率。
子窗口初 SOC 取该窗口起点的边界；前功率取前一小时的实际值。
储能前净功率为总放电减充电；应急放电是总放电的子集，不重复相加。
当 T 恰好等于 N、E、S 或 K 时仍按明确字段 schema 切片，不猜第一维。

## 5. 原始/冻结资产与信息时间

`data/planning/initial_assets.npz` 与 `frozen_assets.npz` 保存固定 ID、容量和电气参数。
`asset_boundary.json` 声明 `asset_planning_v1`、原始/冻结哈希和规划输入哈希。
模式严格区分 `fixed_assets`、`preplanned`、`full_window_planning`。
哈希调用 controller 的唯一 `asset_snapshot_sha256`，不重复实现不同算法。
缓存还核验 preplanned 三份设计输入，或 oracle 使用的原始 Stage11 源荷输入。
独立设计气候轴不强行按数值与运行轴比较；另核信息可用时刻不晚于运行开始。
同一时轴的预规划输入则必须在运行开始前结束。
固定资产和调度的完美预知是两个维度，固定资产不代表在线因果调度。
缺少 F 边界的旧缓存不能作为当前入口 Stage13/14 的现代恢复输入。

## 6. 数据集和兼容

数据集 schema 为 `0.10.0`，旧 static/dynamic/graph/operation 通道保持。
旧 graph capacity 列保留 Stage12 设计值；F 冻结快照是最终资产依据，metadata 明确两者来源。
`operation_detail` 和 `power_flow_detail` 保存两个完整 F Store。
`asset_planning` 保存原始/冻结快照和边界 metadata，独立于运行结果。
窗口中运行量按明确 T/T+1 支撑切片，固定实体及容量不切片。
未给传统文件补造故障、孤岛、物理发电或资产冻结声明。
provenance 按实际模式标注规划信息，另存 dispatch foresight。
全窗规划形成的资产仍标为 oracle 上下文，不包装成发行时刻可知特征。

## 7. 验证命令和边界

定向命令保存在 `outputs/.implementation-tools/test_f_interfaces.sh`。
运行 `/d/anaconda/python.exe -B -m pytest`，包含本包接口、数据集、字段契约及 E 接口回归。
使用仓库内 Git Bash、TMP/TEMP、Matplotlib 缓存与 UTF-8，不安装依赖。
接口测试覆盖 T=N=E=S=K、缺末 SOC、坏初 SOC、错投运维度、重复 ID、坏哈希及设计输入篡改。
还覆盖旧数据不捏造新声明、窗口前功率和固定资产不冒称因果调度。
接口、数据集、字段契约及 E 接口首轮 **71 passed，1.50 s**；加入现有电网/储能回归与应急非零窗口反例后 **87 passed，1.38 s**，两组有重叠不相加。
完整数值测试、三模式世界与恢复结果以 F 主记录为准，本接口文档不重复计数。
