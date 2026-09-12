# F：固定资产、分岛运行与储能备用数值核

本文记录 `operation/storage_dispatch.py` 与 `operation/power_flow.py` 的实际实现。复用原有 SciPy HiGHS LP、必要时的 MILP 充放电互斥回退，以及各电气孤岛 PTDF。规划模式的控制器、设计窗口隔离与资产快照由本工作包的其他文件负责；本核只执行显式传入的资产和时间窗，不自行选择设计样本。

## 1. 运行接口与信息边界

`dispatch_storage_week(topology, electrical, baseline_power_flow, storage_plan, config, *, source_forecast=None, fixed_capacity=False, branch_in_service=None, thermal_land_limits_mw=None, initial_soc_mwh_by_site_id=None, previous_thermal_mw_by_bus_id=None, previous_storage_net_mw_by_site_id=None)` 保留旧位置参数与四元返回值。

- `source_forecast` 为 E 阶段的外生请求负荷、可用发电量和参考计划。按 bus ID 重排；仅新增 `transit_bus` 可补零。母线类型、ID、时轴和可用量上限必须一致。固定运行中的请求需求不从已经缺供的基线倒推，也不因设备容量不足而裁剪。
- 未传 `source_forecast` 的旧调用仍可运行，但在元数据中标记 `legacy_baseline_reconstruction_with_nameplate_thermal_assumption`：负荷用基线已供加缺供恢复，风光用基线发电加弃电恢复，热电采用铭牌可用性假设。这一分支不声称掌握真实的时变热电可用量。
- `fixed_capacity=True` 将热电、线路额定值、储能功率和储能能量四类扩容变量的上界全部设为精确零。其余调度约束仍保留。默认 `False` 是兼容旧的窗口内容量优化；它属于完整窗口信息下的规划核，不是无前视运行回测。
- `branch_in_service[T,E]` 与完整的 `electrical.branch_params` 顺序对应，持久 `branch_ids[E]` 始终保留。不能通过删掉故障线路列改变 ID 轴。
- 明确的初始 SOC 字典以 site ID 对应 MWh；前窗热电出力以 bus ID 对应 MW，前窗储能净功率以 site ID 对应 MW，正值放电、负值充电。未知 ID、负库存、超出既有设备范围的前态均拒绝。

所有时间步为一小时区间。`soc_mwh[T+1,S]` 是区间边界库存；`charge_mw[T,S]`、`discharge_mw[T,S]`、备用和各类发电/负荷是区间功率或在该区间持有的备用。核不消耗随机数。

## 2. 热电可用量、扩容与土地

令 $P_k^0$ 为输入电气资产中的热电既有容量 MW，$A_{t,k}^0$ 为 E 对该既有容量给出的小时可用 MW，$x_k$ 为容量扩展 MW。时变可用率为

$$a_{t,k}=A_{t,k}^0/P_k^0,\qquad A_{t,k}=a_{t,k}(P_k^0+x_k).$$

零既有容量对应的可用率为零。输入必须满足 $0\le A^0\le P^0$。将可用率应用于整台扩大后的等效机组是 **S：同一机组包络具有相同可用率** 的简化，不是独立新机组故障模型；其作用是保证某小时 `A^0=0` 时扩容也不能凭空恢复出力。预先规划后的运行若传入新的既有资产，应传入以该冻结容量为基准的可用量，核不会重复按旧容量缩放。

土地参数 `thermal_land_limits_mw` 必须列出全部运行热电 ID。其值 $L_k$ 为 C 已分配独占土地所支持的容量上界 MW：

$$P_k^0\le L_k,\qquad 0\le x_k\le\min(x_k^{max},L_k-P_k^0).$$

原有容量越界直接拒绝，而不是在运行时悄悄缩小既有机组。未传土地参数的旧分支标记 `thermal_land_guarantee=False`，其导出有限上界只是旧优化容量限额，不能当作土地验证。

调度量 $g_{t,k}$ 和热电上调备用 $R^T_{t,k}$ 使用

$$g_{t,k}+R^T_{t,k}\le\alpha_T A_{t,k},$$

其中 $\alpha_T$ 为 `thermal_operating_limit_ratio`，无量纲工程场景折减。原始可用量 $A$、这一运行折减和实际 $g$ 分开导出。核没有最小稳定出力、最短启停时间、燃料/冷却水库存或机组热状态模型。

## 3. 逐模式分岛、潮流与局部缺供

【物理关系及工程简化】对每种不同的线路开关向量，重新求连通分量与 PTDF。每个孤岛设置自己的相角参考。若母线 $i$ 的发电加储能净注入减实际供给需求为 $p_i$ MW，则各岛满足 $\sum_i p_i=0$；线路功率采用

$$f_{ij}=\frac{U_{LL}^2}{X_{ij}}(\theta_i-\theta_j),\qquad |f_{ij}|\le\alpha_L(S_{ij}^0+y_{ij}).$$

$U_{LL}$ 为 kV，$X$ 为欧姆，$\theta$ 为 rad，因此 $U_{LL}^2/X$ 对应 MW/rad 的数值系数；$S^0,y$ 为线路额定与增额 MVA。在此 DC 简化中未求无功，使用有功 MW 对 MVA 名义额定值筛查。$\alpha_L$ 为无量纲运行限额比例。关闭线路不进入岛的矩阵，流量严格零。

逐母线缺供变量 $u_{t,i}$ 满足 $0\le u\le D$，其中 $D$ 是外生请求 MW；岛的可供资源不足时，只在该岛分配缺供。核不会把某一个孤岛的缺口按全网负荷比例散布。开关变化不重置储能库存。

`solve_dc_power_flow(..., branch_in_service=..., rebalance=True)` 可独立用于各岛基线平衡和潮流筛查。核调用 `rebalance=False` 核对已经优化好的局部供需，不再二次改变 LP 的逐节点缺供；它拒绝超过数值容差的岛不平衡。

DC 模型忽略 $I^2R$ 有功损耗、无功与电压幅值限制，不验证频率动态。线路扩展变量仅提高额定限额、保持阻抗不变，不能解读为自动增加平行线路。一次指定故障时序通过不代表全套 N−1 校核通过。

## 4. SOC、充放电互斥和首小时爬坡

【物理关系】储能 $s$ 在第 $t$ 小时满足

$$E_{t+1,s}=E_{t,s}+\eta_c c_{t,s}\Delta t-d_{t,s}\Delta t/\eta_d,$$

$E$ 为 MWh，$c,d$ 为电网侧 MW，$\Delta t=1$ h，$\eta_c,\eta_d\in(0,1]$。$d$ 包含正常放电与旧字段 `emergency_discharge_mw`；后者是实际已消耗库存的出力，不是备用。最低与最高 SOC 约束作用于全部 $T+1$ 状态。

非循环边界默认初态为 `initial_soc_fraction` 乘优化后安装的能量容量，这是显式的初始库存场景假设；显式传入 MWh 字典时固定该库存，不随扩容赠送能量。循环模式优化满足 $E_0=E_T$ 的边界，不能同时指定前窗字典。循环库存不代表空储能，元数据明确这一假设。

净功率 $n=d-c$ 的相邻小时变化满足 $|n_t-n_{t-1}|\le r_S P$；热电满足 $|g_t-g_{t-1}|\le r_T P_T$。$r_S,r_T$ 的单位为 h⁻¹，这里步长固定一小时。非循环第一个小时用显式前窗 MW 或默认零；循环第一个小时与最后一个小时相连。零爬坡率真实表示功率不能变化，代码不再改成 `1e-3`。发生外生可用量骤降时仍须同时满足声明的爬坡约束；若前态与可用率无法兼容，核明确报不可行，不暗中解除约束。

LP 首先使用共用逆变器容量的凸约束。如解中同一站充电与放电/备用共正，复用 MILP 模式变量求解整个问题，保留全部投资、网络与时序约束。不存在先求一个非法解、再截断充放电而破坏水平方程的后处理。

## 5. 备用能力和不足量

备用需求为 **S：场景指定**。对有外生负荷的活动孤岛 $I$：

$$R^{req}_{t,I}=f_R\sum_{i\in I}D_{t,i}+R_0,\qquad
\sum_{s\in I}R^S_{t,s}+\sum_{k\in I}R^T_{t,k}+R^{short}_{t,I}\ge R^{req}_{t,I}.$$

$f_R$ 无量纲，$D,R_0,R^{req},R^{short}$ 均为 MW。没有负荷的岛不加固定 $R_0$。需求以请求负荷计算，不能通过先切负荷缩小备用要求。每岛需求与不足量在最小 bus ID 所在列导出，其余节点列为零，避免汇总重复计数。

热电备用除容量余量外满足 $R^T\le r_T P_T\tau_R$。储能备用同时满足

$$d+R^S\le P_S,\qquad R^S\le r_SP_S\tau_R,\qquad
R^S\tau_D/\eta_d\le E_{t+1}-e_{min}E^{cap}.$$

$P_S,P_T,R$ 为 MW；$\tau_R$ 为响应时长 h，$\tau_D$ 为要求持续时长 h；$E^{cap},E_{t+1}$ 为 MWh；$e_{min}$ 为最低 SOC 比例。保留区间末库存中的可用能量，是预定放电后还能持续提供备用的保守约束。备用与充电模式互斥，故不能用尚未完成的同小时充电证明该备用可用。正常放电另受 `normal_dispatch_c_rate * Ecap` 约束；备用和实际紧急放电共用逆变器总功率，不强加这一正常调度 C-rate。

这只验证**岛内汇总备用**，未对备用激活场景再做 PTDF 线路可送达约束，也未验证故障瞬间频率、惯量、调速或备用机组启机时间。因此不能称为全网络可交付备用或 N−1 可靠性保证。

## 6. 新配置与成本优先级

以下均在 `storage:` 下，除量纲恒等式外都是可调整的 S 参数；没有下载地区数据来拟合默认值。

| 配置 | 默认 | 单位与合法范围 | 用途 |
|---|---:|---|---|
| `reserve_load_fraction` | 0 | 无量纲，有限且 ≥0 | 请求负荷的备用比例 |
| `reserve_contingency_mw` | 0 | MW，有限且 ≥0 | 每个有负荷孤岛的附加需求 |
| `reserve_duration_hours` | 1 | h，有限且 >0 | 储能备用需持续的时长 |
| `reserve_response_hours` | 0.25 | h，有限且 >0 | 爬坡可达到的响应窗口 |
| `reserve_shortfall_cost` | 1000 | 每 MW·小时不足的目标函数费用，有限且 >0 | 显式备用不足惩罚 |
| `reserve_offer_cost` | 0.01 | 每 MW·小时备用的目标函数费用，有限且 ≥0 | 避免无需求的任意备用分配 |
| `initial_thermal_mw` | 0 | MW，有限且 ≥0 | 无显式字典时的前窗热电出力 |
| `initial_storage_net_mw` | 0 | MW，有限有符号 | 无显式字典时的前窗储能净功率 |

默认缺供成本为 10000，高于备用不足成本 1000；在能量仅够供当前负荷时，默认选择供给负荷并报告备用不足。用户可自定费用改变优先级，模型不会把优化目标权重称作自然定律。扩容费用、弃电激励、正常/紧急循环成本仍沿用现有配置；极端权重可能鼓励大量循环，应根据场景选择并阅读实际损耗账。

## 7. 导出和旧字段兼容

两个 Store 均使用 `operation_arrays` 和 `operation_metadata`；序列化为严格 `operation_v1` 的 `op__*` 字段。PowerFlow 附录 7 项，StorageDispatch 附录 23 项，维度由 `core/operation_contracts.py` 显式声明，不能用某一维恰好等于 T 猜测时间轴。

StorageDispatch 的 `op__requested_load_mw`、`served_load_mw`、`unserved_load_mw` 只含外生需求；`generator_dispatch_mw`、`generation_available_mw` 只含原生发电资产，不把储能放电并入同母线热电。热电自身的安装容量、可用量和实际出力另按 thermal ID 轴导出。储能充放电仍在原站点数组。旧 PowerFlow/调度 forecast 的负荷可包含储能充电、发电可包含储能放电，元数据保留这一兼容口径，避免读取者把它们再次作为原始需求。

- 风光弃电 = 风光可用 − 风光实际发电。
- 热电回退 = max(输入参考计划 − 实际热电发电, 0)。这是相对于先前计划的下降，不等于全部可用能力未用。
- 热电未使用可用量 = 时变热电可用 − 实际热电发电，其中可包含备用和运行折减余量。
- 旧 `curtailed_generation_mw` 等历史字段继续保留；解释物理类型时使用新的分项。
- `dispatched_unserved_mw` 严格由实际逐母线缺供求和，删除了原来 `<1e-3` 的摘要清零。求解器变量保留任意正小量；仅对下界为零的变量清除 `[-1e-7,0)` 负舍入值，较大负值拒绝。

## 8. 验证与限制

`tests/test_grid_operation_mechanisms.py` 的 21 项解析/参数化案例，加原 `tests/test_grid_storage_physics.py` 的 9 项共 30 项通过。覆盖三母线局部断线、持久线路 ID 与零故障流、时变热电零可用量、源荷 ID 重排与新增中转节点、四类定容开关、土地不足、首小时 SOC/爬坡、故障前后库存连续、储能备用的能量/效率/持续与响应功率约束、真实触发 MILP 的充电/备用互斥、备用不能跨孤岛、默认费用优先供负荷、零爬坡与小额缺供保留、两类现代 Store 回读及非法契约。

独立导出验证、多个完整世界种子和恢复检查由 F 总验收另外执行；本小核测试不替代这些集成检查。现有全窗口优化仍有完整窗口信息，不能将它当作实时控制策略的预测表现。模型也不包括动态线路故障概率、故障恢复随机过程、AC 电压安全、频率稳定、可交付备用、储能老化/自放电或热电启停细节。

## 9. 已有研究依据

本次未扩大文献综述，复用 [任务09：电网运行与储能](mechanism_research/task_09_grid_operation_and_storage.md) 中的功率平衡、DC 近似、储能效率/SOC、备用可送达边界及信息集区分；风光可用量与请求负荷来自已验收 E 模块。物理守恒与场景先验在上述各节分别说明，容量与损耗的地区/技术参数仍需使用者按目标系统设定。
