# 电网、潮流、扩建和储能：研究依据与实现

本文覆盖阶段 9–14。模型定位是合成的、单电压等级的有功传输网等值；每个母线可代表下级配电区域。线路和有功平衡有明确物理关系，但没有实现交流电压、无功潮流、变压器、故障或暂态模型。

## 研究来源与保留的简化

1. **Birchfield, Xu, Gegner, Shetye & Overbye (2017)**, *Grid Structural Characteristics as Validation Criteria for Synthetic Networks*, IEEE TPWRS 32(4), 3258–3265，[DOI/出版页面](https://doi.org/10.1109/TPWRS.2016.2616385)。先按人口和发电设施安排站点，再生成线路，并以网络结构和电气统计验证。这支持本项目保留“用地/人口→负荷和电源站点→母线→线路”的依赖顺序；它不证明任意 MST 加冗余边就是现实电网。
2. **Birchfield et al. (2017)**, *A Metric-Based Validation Process to Assess the Realism of Synthetic Power Grids*, Energies 10, 1233，[论文](https://www.mdpi.com/1996-1073/10/8/1233)。需要同时考虑拓扑和电气指标。本项目增加守恒和运行约束验证，仍未拟合现实网络的度分布、线路长度分布、Delaunay 重叠率或 N−1 安全性，不能据此宣称统计真实。
3. **Zimmerman & Murillo-Sánchez**, MATPOWER User’s Manual，[DC 模型](https://matpower.app/manual/matpower/DCModeling.html)、[DC 潮流](https://matpower.app/manual/matpower/DCPowerFlow.html)。采用无损、近单位电压、小角差近似，得到线性有功潮流。
4. **pandapower 官方标准线路库**，[50 Hz 参数表](https://pandapower.readthedocs.io/en/v2.10.0/std_types/basic.html)。采用明确模板替代按节点容量增加线路额定值的经验式；模板并非中国某实际线路的实测参数。
5. **Brown, Hörsch & Schlachtberger (2018)**, *PyPSA: Python for Power System Analysis*, JORS 6(1):4，[作者论文](https://arxiv.org/abs/1707.09913)，[储能约束文档](https://docs.pypsa.org/latest/user-guide/optimization/storage/)。多时段线性运行/投资框架、储能效率和状态递推，以及循环和给定初态的区别。
6. **Garifi, Baker, Christensen & Touri (2018)**, *Control of Energy Storage in Home Energy Management Systems: Non-Simultaneous Charging and Discharging Guarantees*，[作者论文](https://arxiv.org/abs/1805.00100)。连续充放电松弛在某些目标和剩余光伏条件下可同时充放。本项目不假设任意价格下松弛都精确，而是在发现同时充放时加二进制模式重新求解。

## 阶段 9：母线与火电初始容量

负荷、电源母线继续由上游站点创建。人口决定负荷容量的修正见 [地理报告](research_geography.md)。火电的选址适宜性、居民缓冲、距离权重仍是可配置场景规则。

初始火电配置使用 `target=max(f_load*reserve_multiplier*P_load − c_wind*P_wind − c_pv*P_pv,0)`，在单站上下限内补齐。默认机组候选数为配置下限和 `ceil(target/P_unit_max)` 两者的较大者，避免城市人口增长后仍固定只有四座电厂。若受选址限制未能达到目标，后续仍显式报告缺供。`thermal_scale_with_load: false` 可固定候选数制作容量不足的压力场景。1.15 的裕度倍数、峰荷比例和容量信用只是初始组合假设；`thermal_wind_capacity_credit` 和 `thermal_solar_capacity_credit` **不是**由可靠性计算得到的 ELCC。最终是否供得上，由真实时序、网络及有界扩建求解，并报告缺供。尚未引入机组强迫停运率、季节容量降额、燃料约束或概率可靠性指标。

## 阶段 10：拓扑与线路参数

保留地形代价、候选近邻边、MST、冗余和 A* 走廊。此类连接规则是生成启发式，不能视为已优化的工程设计。地形、保护区、水体的软惩罚也不等于真实许可条件。

原实现混合 110/220 kV 母线却没有变压器，同时以 `max(base,1.35*endpoint_capacity)` 和冗余标记改变线路额定容量。现默认全网采用 220 kV 等值，可配置 110 kV；每条初始线路来自同一电压模板，母线和线路电压一致：

| 电压/模板 | R (Ω/km) | X (Ω/km) | C (nF/km) | I (kA) | S (MVA) |
|---|---:|---:|---:|---:|---:|
| 110 kV，243-AL1/39-ST1A | 0.1188 | 0.39 | 9 | 0.645 | 122.889 |
| 220 kV，490-AL1/64-ST1A | 0.059 | 0.285 | 10 | 0.96 | 365.809 |

`R=rL`，`X=xL`，`B_μS = 2π·50·C_nF·L/1000`，`S_MVA=√3·U_kV·I_kA`。节点负荷大小不直接改变导线载流量。R、B 和无功参数仍导出供将来 AC 扩展，目前 DC 求解不使用这些量。

## 阶段 11：逐岛有功平衡

`F_ij = (U_kV²/X_Ω) (θ_i−θ_j)`，单位 MW，θ 为 rad。对各连通分量分别建立 `Bθ=P_generation−P_served`，每岛一个角度参考。以前全网总量平衡和奇异矩阵最小二乘可掩盖孤岛缺供；现孤岛只使用本地供给，缺供和发电削减显式记录。负荷/母线顺序、非有限数及非正电抗被拒绝。

初筛调度遇到过剩供给先降低可调电源出力，再削减可再生供给。此阶段是容量和线路压力筛查，没有强加线路限值；最终多时段优化才实施网络约束。基线 `curtailed_generation_mw` 表示被减少的原计划发电，可包含火电回退，不等同于纯弃风弃光。角度按岛批量求解，取消逐小时重复分解。

## 阶段 12：线路更新

保留按过载提出旁路、走廊合并和低利用率缩容的启发式；该阶段不是全局最优输电规划。若线路按连续并联回路倍数 `m` 等值，必须满足 `R∝1/m`、`X∝1/m`、`B∝m`、`S∝m`。

修复了走廊合并中对已经缩放的 R/X 再次除以倍数的问题：先恢复单位回路参数，再根据新倍数缩放。新建旁路也按原/新容量比同步变换 R/X/B。倍数允许小数，属于连续等值模型，不代表真实建造了分数条线路。改变容量后重新计算潮流，不将容量与阻抗混为一谈。

## 阶段 13：储能需求与初始选址

保留沿网络距离衰减的拥塞/缺供分摊和覆盖式选址，作为候选集生成。连续支持事件所需 AC 能量为 `E_event=Σ P_support·1h`。先前直接把它当电池铭牌容量，现改为：

`E_nameplate_required = E_event / [η_discharge·(SOC_max−SOC_min)]`。

再加容量裕度及设定的铭牌 `E/P` 时长上下限。若最大时长限制截断需求，并不保证候选储能能覆盖整个事件；阶段 14 会检验实际充电机会、SOC、线路限制及供给缺口。储能不会产生净能量，也不会凭选址评分消除长期能源不足。

## 阶段 14：有界投资、时序调度与显式缺供

模型是**已知整个窗口实况的规划优化**，不冒充滚动运行预测。所有输入为连续 1 小时区间；拒绝非小时步长以避免 MW/MWh 混用。

对每个岛、每个小时：

`Σ(P_gen + P_dis − P_charge + P_shed) = Σ P_gross_load`。

`0≤P_shed≤P_gross_load`，线路潮流由相同节点注入经 PTDF 计算。初始与扩建容量均受配置上限约束。默认允许高惩罚的削负荷（情景成本 10000），在不足时输出母线级缺供而不是生成失败；`allow_load_shedding: false` 恢复严格零缺供模式，无解会报错。缺供在公共数据中满足 `requested_load=served_load+unserved_load`，不按全网比例摊到其他母线。

储能递推为 `E[t+1]=E[t]+η_c·P_c[t]·1h−P_d[t]·1h/η_d`。`P_d` 已包括 emergency，计算 SOC 和爬坡时只计一次；净出力爬坡包含全部放电。默认循环边界 `E[T]=E[0]`，起点由优化选择；非循环模式用 `initial_soc_fraction` 固定初态，并不把窗口末尾当作第一小时的前一时刻。

共享逆变器满足 `P_c+P_d≤P_max`。先求连续 LP；如仍有同时充放电，使用受最大扩建功率约束的二进制 `z`：`P_c≤M z`、`P_d≤M(1−z)`，重新优化同一投资和物理约束。正常和紧急放电费用、SOC 偏好带费用均是情景偏好，不能视为观测电价或电池寿命模型。

Stage14 的线路增容明确采用**固定阻抗的热额定提升**，与 Stage12 的并联等值不同。它不构成精确导线更换或并联建设；若要研究真实工程建设，应引入离散导线/回路选择及更新阻抗的迭代或混合整数模型。

优化中主动弃掉的可再生供给现在加入 `curtailed_generation_mw`，不再只记录最终重新平衡时的微小削减。火电容量、运行上限、爬坡，以及储能功率/能量边界和线路上限均按规划结果核验。对于求解器数值异常，换用同一约束的 HiGHS 内点算法，不放宽物理限制；真实不可行与数值失败分开报告。

## 使用与验证边界

`scripts/validate_world_physics.py` 检查 KCL、供需核算、SOC 递推、互斥、容量、爬坡和线路约束，并显示缺供、弃电。**物理核算 PASS 不代表零缺供，也不代表现实参数已校准。** AC 电压、无功/损耗、机组启停/最小稳定出力、备用、N−1、故障概率、储能自放电/老化和多年投资经济学仍未实现。默认投资成本与运行成本是同一窗口内的相对权重，并没有从市场货币单位换算为严格年化成本，因此只适合场景权衡。
