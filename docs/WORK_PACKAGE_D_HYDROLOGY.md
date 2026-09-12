# D：可选小时动态水量核

`hydrology_dynamic.enabled=false` 时，生成器保持静态水文模式；本核立即返回 `None`，不读取其他输入、不调用随机数、不修改 D8、水深、汇水面积或其他上游图。启用后，使用同一份静态地形、水体和 D8 骨架，生成独立的 `bucket_routing_v1` 小时水量结果。源于静态面积的 `flow_accumulation` 仍是汇水指标，不能当作实时流量。

本项复用已有专题03及[机制研究综合报告](mechanism_research/synthesis_report.md)的水量守恒、土壤桶、线性库路由与湖盆库容思想，未新增真实数据下载或拟合。下述参数均是可审查的合成情景，不代表某个地区已校准的降雨径流关系。

## 状态、通量及单位

公开函数为 `generate_dynamic_hydrology(terrain, hydrology, land_use, hourly_weather, grid, config, *, boundary_inflow_m3=None)`。没有 RNG 参数。启用时返回 `HydrologyTimeSeriesStore`：`timestamps[T]`、`time_bounds_hours[T,2]` 是一小时区间，`state_time_hours[T+1]` 是其边界时刻。序列化的 `state__`、`flux__`、`static__`、`budget__` 前缀区分时间支持，避免时间长度恰好等于网格高度时误切静态图。

状态有六项，全部 `[T+1,H,W]`：

| 字段 | 单位与含义 |
|---|---|
| `soil_storage_mm` | mm，整格等效土壤水深 |
| `groundwater_storage_mm` | mm，整格等效概念地下水库存 |
| `channel_storage_m3` | m³，概念路由库存，包含尚未传到下游的水和湖溢流 |
| `lake_storage_m3` | m³，该格所占湖池库存 |
| `lake_water_level_m` | m，与地形相同高程基准；同湖群各格共享 |
| `lake_wetted_area_m2` | m²，本格当前淹没面积；粗网格下为阶梯函数 |

十九项 `[T,H,W]` 通量为：`precipitation_mm`、`infiltration_mm`、`soil_evapotranspiration_mm`、`percolation_mm`、`groundwater_overflow_mm`、`baseflow_mm`、`surface_runoff_mm`、`potential_et_mm`；其余体积项为 `open_water_evaporation_m3`、`routing_inflow_m3`、`routing_outflow_m3`、`lake_mixing_inflow_m3`、`lake_mixing_outflow_m3`、`lake_overflow_m3`、`boundary_inflow_m3`、`boundary_outflow_m3`、`actual_et_m3`、`cell_budget_residual_m3`。只有 `discharge_m3_s` 是区间平均 m³/s，其他水深或体积通量均为区间累计。

`static__` 导出不透水/透水份额、整格土壤/地下水容量，以及 `lake_id`、`closed_sink_mask`、`lake_bed_elevation_m`、`lake_spill_elevation_m`、`lake_capacity_m3`、`routing_receiver_flat_index`。湖 ID 为正整数，非湖为 0。接收格索引非负时表示内部 D8 接收格，−1 表示边界出口，−2 表示封闭库存；非湖的湖床/水位/溢流水位/湖容量输出为 0，不使用 NaN。

## 土地到土壤桶

**物理关系：** 格距 $\Delta x$ 的单位为 km，格面积 $a=\Delta x^2$ 的单位为 km²。整格等效水深 $p$ mm 的体积为 $1000ap$ m³。任何降雨转换都只乘本格面积，不再乘静态汇水面积。

**场景先验：** 使用 C 已闭合的九类土地份额 $f_k$ 和明确的不透水率 $c_k$。非开阔水面格的不透水份额为 $f_{imp}=\sum_k c_kf_k$，透水份额为 $f_{perv}=1-f_{imp}$。开阔河湖格的两者都为 0，另由水面库处理；`water` 份额必须与静态河湖掩膜一致。能源用地预留不是全部封闭地表，默认 `energy_reserve` 只使用 0.05 的情景不透水系数。

土壤参数容量 $S_{max}$ mm 按透水份额转换为整格容量 $C_S=f_{perv}S_{max}$ mm。地下水整格容量 $C_G$ 按非开阔水面份额缩放；允许地下水概念库存在于不透水地表之下，不把二者等同。

每小时顺序为：入渗、土壤实际 ET、重力渗漏、地下水超容量返流和基流。令区间雨深为 $p$ mm，入渗能力为 $k_I$ mm/h，时步 $\Delta t=1$ h：

$$I=\min(pf_{perv},\ k_I\Delta t f_{perv},\ \max(C_S-S,0)).$$

先更新 $S\leftarrow S+I$。土壤实际 ET 受库存限制；随后以田间容量份额 $f_{FC}$ 和时间尺度 $\tau_P$ h 求渗漏：

$$ET_S=\min(S,ET_p f_{perv}),\quad S\leftarrow S-ET_S,$$
$$D=\max(S-f_{FC}C_S,0)[1-\exp(-\Delta t/\tau_P)],\quad S\leftarrow S-D.$$

地下水收到 $D$ 后，超过 $C_G$ 的部分计入 `groundwater_overflow_mm`，转入路由库；剩余库存按 $B=G[1-\exp(-\Delta t/\tau_B)]$ 释放基流。非湖面 `surface_runoff_mm` 定义为非湖雨深减入渗，包含河面直接降雨，但不包含地下水超容量返流或基流。湖面降雨直接进入湖库，不先计入地表径流，以免重复入账。

这些公式是明确的低阶桶模型；它们没有计算入渗锋、非饱和土壤水势或空间地下水梯度。容量限制是模型中的有效库存限制，不是某地土壤剖面的观测校准。

## ET、水面与能量需求

**工程简化：** 小时平均 GHI $R_s$ 的单位为 W/m²，吸收份额 $\alpha_s$ 与分配给潜热的份额 $f_\lambda$ 无量纲，汽化潜热 $L_v$ 的单位为 J/kg：

$$ET_p=R_s\alpha_s f_\lambda(3600\Delta t)/L_v.$$

这里 $ET_p$ 为 mm 区间累计，因为 1 kg/m² 水对应 1 mm。它仅是已声明的短波能量需求简化，**不是完整地表能量平衡，也不是 FAO 参考蒸散量**；没有长波、显热平流和冠层空气动力阻力。实际土壤 ET 受土壤水库存限制；实际河面蒸发受该格路由水量限制；湖面蒸发只对当前有水的湖格请求，并受实际湖水量限制。干湖不能凭空蒸发。概念路由库存位于非水面格时，本模型不据此推断洪泛面积或额外蒸发。

`actual_et_m3 = soil_evapotranspiration_mm × 1000a + open_water_evaporation_m3`。该水量是动态水文核的外部损失；本阶段没有把它反馈成天气比湿或大气水汽库存，因此不能声称已实现陆气双向守恒耦合。

## 同步路由、内陆汇与湖泊

**物理关系：** 内部每次转移在供水格记录 `routing_outflow_m3`，在接收格记录等量 `routing_inflow_m3`。边界出流只写 `boundary_outflow_m3`，不重复计入内部出流。`discharge_m3_s=(routing_outflow_m3+boundary_outflow_m3)/3600`。

**场景先验与工程简化：** 每条 D8 边的长度 $\ell$ 为 m，路由速度 $v$ 为 m/s，概念停留时间 $\tau=\ell/(3600v)$ 为 h。轴向/斜向边分别使用格距与其 $\sqrt{2}$ 倍；边界出口使用半格距离。每子步 $\delta t$ 的出水为：

$$Q_{vol}=C[1-\exp(-\delta t/\tau)].$$

所有格先同时计算并扣除出水，再同时增加下游来水。因此刚收到的水在当前子步不能沿下一条边继续跳跃，默认一小时一个子步；已有库存与有限子步提供明确的时间延迟。该线性路由库不是动量方程，速度不是通过河宽、坡降和糙率校准的洪水传播速度。

内部非湖 `direction=-1` 默认标为 `closed_sink_mask=true`，水留在 `channel_storage_m3`；`interior_sink_policy=reject` 可要求直接拒绝这类静态图。内陆汇不会被误作域外出口。显式边界入水只能给边缘格的非负 `[T,H,W]` m³ 数组。负入流、内陆外部注水、越界 D8 指向和 D8 环均被拒绝。

湖群由静态湖掩膜的八邻接分量定义。使用其中最低静态溢流水位 `bed + static_water_depth` 作为共同溢流高程；大于该高程的格可以是当前干燥湖岸。不同小盆地被粗网格连接时，最低共同溢流面是简化选择，不是已重建真实水库连通几何。出口在已有 D8 离湖路径中选取，要求原路径不重入本湖；聚合成共同湖池后形成的路由环也拒绝处理，避免假装实现了回水。

对湖群床高 $z_i$ m，公共水位 $h$ m，单格面积 $A=10^6a$ m²：

$$V(h)=A\sum_i\max(h-z_i,0),\qquad A_{wet}(h)=A\sum_i\mathbf1(h>z_i).$$

由分段线性库容曲线反解水位，水位—库存与水位—淹没面积单调。默认初始湖库存份额为 0，静态湖深不代表动态初始湖已满。工作深度使用相对最低湖床的坐标计算，防止绝对高程抵消误差制造微小水源；导出的绝对高程仍有浮点表示精度。

超过溢流容量的水转到湖出口格的 `channel_storage_m3`，之后才参加同步路由。`lake_overflow_m3` 是这一内部转移的诊断子项，不再作为总账外部损失。湖群为了保持共同水位而进行的格间水量重分配，包括将溢流水移到出口格，分别计入 `lake_mixing_inflow_m3/outflow_m3`。河道概念库没有无依据的 bankfull 容量上限；它包含待路由库存，不能据其体积宣称已验证河道洪水水位或安全容量。

## 独立逐格和全域预算

令整格合计储水量 $W=1000a(S+G)+C+L$，单位 m³。每格每小时满足：

$$W_t+1000ap+F_{boundary,in}+F_{routing,in}+F_{mix,in}
=W_{t+1}+ET+F_{boundary,out}+F_{routing,out}+F_{mix,out}.$$

土壤入渗、渗漏、基流、地下水溢出、湖溢出都是内部库间转移，不能再加入该总账。内部 D8 和湖混合双边在全域相消，得到初储水＋降雨＋边界入水＝末储水＋实际 ET＋边界出水。

`budgets[T]` 输出七项：`initial_storage_m3`、`precipitation_m3`、`boundary_inflow_m3`、`actual_et_m3`、`boundary_outflow_m3`、`final_storage_m3`、`residual_m3`。核内部先检查逐格再检查全域；root 的独立验证器从状态和各通量重建等式，不只相信报告残差为零。

默认允许误差为 $10^{-5}$ m³ 加 $10^{-10}\max(|lhs|,|rhs|)$；二者分别对应配置 `budget_absolute_tolerance_m3`、`budget_relative_tolerance`，必须非负且不能同时为零。这里的误差界用于浮点算术，不是水量校准误差或未建模过程的豁免。

## 默认参数与验证

全部字段位于 YAML `hydrology_dynamic:`，快照保存实际值。`enabled=false`；土壤容量 150 mm、初始份额 0.35、入渗能力 20 mm/h、田间容量份额 0.65、渗漏时间 48 h；地下水容量 500 mm、初始份额 0.10、基流时间 240 h；初始河道整格等效水深 0 mm、初始湖库存份额 0；路由速度 0.5 m/s、每小时子步数 1；短波吸收份额 0.77、潜热份额 0.65、汽化潜热约值 $2.45\times10^6$ J/kg。容量和时间尺度为有限正数，非负量和份额的校验分别限制到非负域和 $[0,1]$，路由子步为 1–60 的整数。

九类不透水系数依次为水体 0、湿地 0、住宅 0.65、商业 0.85、工业 0.8、农业 0.05、公园 0.02、自然地 0.02、能源预留 0.05。它们是组合情景先验，不是某个区域的部门调查结果；短波模型中的潜热约值是工程物性近似，其余过程强度没有区域拟合。

`tests/test_dynamic_hydrology.py` 的 18 项解析测试通过：关闭不读输入/不耗 RNG；局地面积与线性库衰减；不乘汇水倍数；三格有限传播；封闭汇与拒绝模式；土壤与地下水库存/容量；ET 受水量限制；按土地份额计算不透水性；湖默认空、共同水位和溢流延迟；封闭湖的初始库存与溢流留存；河格蒸发及空湖蒸发限制；单调库容曲线；5000 m 高程极浅雨的守恒；多子步与边界水量；负边界流、内陆外部注水、D8 环、缺失用途份额和矛盾时间边界的拒绝。独立用状态与双边通量重建了逐格和全域账。

本阶段不实现雪和冻融、三维地下水、回水、洪泛区面积扩张、取用水许可、湖底渗漏、潮汐或陆气反馈。它提供有明确适用边界、可独立复核的动态合成水量，而不是地区水文预报或洪水风险评估。
