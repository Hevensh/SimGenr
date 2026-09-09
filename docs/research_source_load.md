# 源荷生成：研究依据、公式与适用边界

## 这一步生成什么

`generate_source_load_forecast` 保留旧接口名和四个数组通道，但生成的是**给定合成实况气象的源荷实现**，`data_semantics=synthetic_realization`。它没有预报起报时刻、预报提前量或气象预报输入，因此不能称为已经通过外样本验证的预测。其后火电出力是一个满足容量上限的静态分配启发式，不是机组组合或网络约束经济调度；实际供需缺口、弃电与线路问题由后续潮流/储能步骤处理。

建议顺序为：确定地理与长期气候 → 用地/人口/装机与电网 → 联合气象实现 → 同一气象场驱动风、光和负荷 → 平衡/潮流/储能 → 训练数据切片与独立预测评估。风光负荷独立随机抽样会破坏同一次天气事件中的源荷关系。Bloomfield 等（2020）的联合研究说明，最大总负荷和最大净负荷对应的天气未必相同；仅验证单变量均值不充分。[原始研究](https://doi.org/10.1155/2020/5481010)

气象接口约定：温度为 2 m 摄氏温度；风速为 10 m m/s；气压为站点 hPa；辐照为小时区间平均水平面总辐照 GHI，单位 W/m²。整数时间戳是从代表年年初开始的小时偏移，全部使用当地太阳时，不是 UTC 或含夏令时的民用时钟。`calendar_start_date` 为此合成年锚定星期，默认 2025-01-01；天气使用 365 日气候周期。多年份/闰年精确对应和地区法定节假日尚未建模。

## 改掉的具体问题

| 原实现 | 新实现与依据 |
|---|---|
| 10 m 风速直接代入机组曲线，额定前用任意平方 | 高度换算、空气密度修正、切入/立方段/额定段/切出；明确参考机组与简化曲线的区别 |
| GHI 直接当组件受光，不区分地面空气温度与组件温度 | Erbs 分解 + 太阳方向投影 + 各向同性散射；Faiman 组件温升/风冷；DC 与 AC 额定容量分开 |
| 冷天组件收益被禁止，固定性能比混合所有损失 | 温度系数允许相对 25°C 的正负变化；阵列损失、逆变效率和 AC 限幅分开 |
| 每次序列从小时 0 重新计时，无星期效应，所有负荷同一日形 | 真实时间偏移、工作日/周末、居民/商业/工业不同形状，用地强度影响混合比例 |
| 温度仅瞬时响应且被硬截断；宜居评分再次改需求 | 因果热记忆与冷热度时；去掉重复的选址评分乘数，保留极端天气需求增长 |
| 独立节点噪声，AR 创新方差不守恒 | 距离核与局地残差，平稳 AR(1)，严格使用 `sqrt(1-rho²)` 创新缩放 |

## 风电

Jonkman、Butterfield、Musial、Scott（2009），*Definition of a 5-MW Reference Wind Turbine for Offshore System Development*，NREL/TP-500-38060，给出 90 m 轮毂、3/11.4/25 m/s 切入/额定/切出速度。本实现只采用这组参考阈值，不声称复现其气动控制和实际功率曲线。[DOE/NREL 一手报告，DOI 10.2172/947422](https://doi.org/10.2172/947422)

计算链：

\[
v_h=v_{10}(h/10)^\alpha,\quad
p_h=p_s\exp[-gh/(R_dT)],\quad
\rho_h=p_h/(R_dT),\quad
v_e=v_h(\rho_h/1.225)^{1/3}.
\]

\[
P=P_{rated}(1-\ell_w)\,
\operatorname{clip}\left(\frac{v_e^3-v_{in}^3}{v_{rated}^3-v_{in}^3},0,1\right),
\]

并在物理轮毂风速 `v_h < v_in` 或 `v_h >= v_out` 时置零。切出不能作用于密度等效风速，否则低密度地区会漏掉大风停机。没有气压时仅对旧调用使用标准密度；完整天气流程传入站点气压。采用干空气理想气体近似，未建模湿空气修正、风切变随稳定度变化、尾流、故障停机、切出重启滞回和亚小时阵风。`alpha=0.14`、系统损失 0.08 是可配置情景先验；三次函数是功率随风能通量变化的低阶近似，实际风场需换成制造商曲线并用观测校准。

## 光伏

1. **GHI → POA。** Erbs、Klein、Duffie（1982），*Estimation of the diffuse radiation fraction for hourly, daily and monthly-average global radiation*，Solar Energy 28(4):293–302，建立小时晴朗指数与散射比例的经验关系。[DOI 10.1016/0038-092X(82)90302-4](https://doi.org/10.1016/0038-092X(82)90302-4)

   `Kt = GHI / I0h`，其中 `I0h` 是上游太阳几何给出的该小时积分平均地外水平辐照，不是中午辐照或日平均辐照。散射比例采用 Erbs 分段式：`1−0.09Kt`（Kt≤0.22）、`0.9511−0.1604Kt+4.388Kt²−16.638Kt³+12.336Kt⁴`（0.22<Kt≤0.8）、0.165（Kt>0.8）。

   `DHI = Fd GHI`，直射按每小时 12 个太阳位置计算平均入射投影；倾角 β、方位角（北0°顺时针）和地表反射率 a 可配。各向同性近似为 `POA = DNI〈max(cosθi,0)〉 + DHI(1+cosβ)/2 + a GHI(1−cosβ)/2`。太阳在地平线下时没有直射，日出/日落部分小时保留能量。DNI 不得超过地外法向辐照，截出的部分归到散射，使水平安装 β=0 时恢复 GHI。夜间 TOA 为零时 POA 必须为零。

   原研究是若干实测地点的经验关系，不等于所有云型的辐射传输解。各向同性天空忽略环日/地平增亮，未处理山体遮挡、排间遮挡、积雪及跟踪支架。默认 30°、180°适用于北半球代表情景；南半球需自行改成朝北，不做隐式猜测。

2. **组件温度。** Faiman（2008），*Assessing the outdoor operating temperature of photovoltaic modules*，Progress in Photovoltaics 16(4):307–315：[DOI 10.1002/pip.813](https://doi.org/10.1002/pip.813)。使用 `Tm=Ta+POA/(U0+U1 vm)`。默认 `U0=25`、`U1=6.84` 的来源是 Negev 开架组件实验；它们与安装方式和测风高度有关，不能直接当屋顶组件校准值。[pvlib 对原实验参数和单位的说明](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html)

   目前通过同一幂律把 10 m 风换算到代表性 2 m 组件高度，将模块温度近似为电池温度，未求解电池与背板温差/热惯性。高辐照升温降低输出；较大风速散热提高输出。

3. **功率与容量。** Dobos（2014），*PVWatts Version 5 Manual*，NREL/TP-6A20-62641：[DOI 10.2172/1158421](https://doi.org/10.2172/1158421)。采用其 DC 温度响应结构 `Pdc=Pdc,STC (POA/1000)[1+γ(Tm−25)](1−loss)`；然后 `Pac=clip(ηinv Pdc,0,Pac,rated)`。`bus.capacity_mw` 是交流并网额定容量，直流容量是 `pv_dc_ac_ratio * bus.capacity_mw`。

   默认 DC/AC=1.2、γ=−0.004/°C、阵列损失 10%、逆变效率 96% 是显式场景设定。逆变器当前采用常效率，未复现 PVWatts 完整部分负荷效率、光学损失或全部子模型，因此应称 PVWatts 形式的简化实现。

## 负荷：日历、温度和随机过程

Su、Kern、Characklis（2017），*The impact of wind power growth and hydrological uncertainty on financial losses from oversupply events in hydropower-dominated systems*，Applied Energy 194:172–183，§2.3.3 使用温度分段响应、月份/星期效应、ARMA 残差和日内曲线重构合成负荷。[作者机构原文](https://kern.wordpress.ncsu.edu/files/2018/08/su-2017.pdf)，[DOI 10.1016/j.apenergy.2017.02.067](https://doi.org/10.1016/j.apenergy.2017.02.067)

这里采用可配置低阶版本，而不是把美国 BPA 的拟合值迁移到虚构中国电网：

\[
T_t^{eff}=aT_{t-1}^{eff}+(1-a)T_t,\quad a=e^{-1/\tau},
\]
\[
H_t=\max(T_H-T_t^{eff},0),\quad C_t=\max(T_t^{eff}-T_C,0),
\]
\[
L_{i,t}=L_{i,base}\left[\sum_s w_{i,s}S_s(h,d)
+A_{i,t}(\beta_HH_t+\beta_CC_t)\right]
\exp\{r_{i,t}-\sigma^2/2\}.
\]

各部门 `S_s` 在独立的代表性完整周上归一到均值1，不用待生成序列的最大值或均值归一，避免随请求时长改变需求尺度；`base_load_mw` 表示舒适温度时的参考周均负荷。居民有早晚活动峰，商业主要随工作时段，工业保留较高过程基荷。这些形状和周末折减是透明先验。用地图是相对强度而非计量电量，先与配置部门权重相乘后归一，不能宣称为真实行业用电比例。`A` 中居民/商业保留完整空调响应，工业权重0.15同样是示例先验。

Wang、Liu、Hong（2016），*Electric load forecasting with recency effect: A big data approach*，International Journal of Forecasting 32(3):585–597，研究滞后小时温度和滑动平均温度对需求的影响。[出版者页面](https://www.sciencedirect.com/science/article/pii/S0169207015001557)，[DOI 10.1016/j.ijforecast.2015.09.006](https://doi.org/10.1016/j.ijforecast.2015.09.006)。本实现的指数记忆是对该类温度滞后效应的低阶近似，并非复现论文变量选择结果。首小时假设有效温度等于输入温度；长期生产应用应传入历史预热段。

残差由以下自行选定的正定协方差族产生；这是用于保留空间/时间结构的工程模型，并非声称文献给出了该地区的核参数：

\[
K_{ij}=q\exp(-d_{ij}/\ell)+(1-q)\delta_{ij},\quad
z_t=\rho z_{t-1}+\sqrt{1-\rho^2}\epsilon_t,
\quad \epsilon_t\sim N(0,K),\ r_t=\sigma z_t.
\]

初始 `z0` 也服从 `N(0,K)`，因此边际方差从首小时起就是设定值。远距离节点仍因共享天气和日历相关，而近距离节点另有共同活动残差。指数乘数和 `−σ²/2` 修正同时保持非负与条件均值。默认温度阈值16/22°C、斜率0.016/0.025每°C、记忆8小时、残差3.5%、rho0.85、q0.55、空间尺度20 km 均须用目标地区实测数据校准；没有数据时不能据此报告预测准确度。

空间核中的 `d_ij` 必须以 **km** 为单位。源荷入口使用 `world.origin_x_km + (col+0.5)*cell_size_km` 和 `world.origin_y_km + (row+0.5)*cell_size_km` 构造格心坐标；平移原点不影响两节点距离。既有 `GridBus.x/y` 为0–1归一化地图坐标，为保持旧模型兼容而保留，不能直接代入单位为km的相关核。入口回归测试验证：仅改变归一化x/y不改变源荷；改变真实格距会相应改变残差空间相关。

## 验证与还不能声称的结论

Li、Yeo、Bornsheuer、Overbye（2019），*The Creation and Validation of Load Time Series for Synthetic Electric Power Systems*，提出母线级居民/商业/工业原型聚合，并用负荷率、负荷持续/分布曲线和自相关检验合成数据，而非要求逐小时复刻某条真实序列。[作者预印本与完整验证章节](https://arxiv.org/abs/1911.06934)，DOI 10.48550/arXiv.1911.06934。

`tests/test_source_load_physics.py` 验证：风电切入/额定/切出与空气密度方向；水平面辐照能量一致性；倾角方位作用；光伏夜间零出力、温升/风冷和 AC 容量上限；日历偏移与商业周末；参考周均值；冷热单调响应与温度因果性；长样本残差方差、AR1和距离相关；固定种子可复现；无功功率因数关系；非法配置拒绝。统计测试验证随机过程合同，不等于与真实电网对标。

获得目标地区数据后，应在保留时间/空间外样本上检查日/月/年能量、峰谷、负荷率、持续曲线、小时及周滞后ACF、爬坡分布、不同距离相关、条件温度响应、净负荷与联合极端事件；再拟合部门比例、温度系数、相关核及设备曲线。预测训练不得把实况未来气象与预测未来气象混淆，数据集需明确特征的可用时刻，并按世界/年份分组切分，避免相邻窗口泄漏。
