# 工作包 E：天气到源荷的数值核

本记录描述 E 数值核、配置与成对机理测试；数据容器和打包接口另见 E 主记录。
复用原有风电曲线、Erbs 分解、倾斜面辐照、Faiman 温度和部门负荷模型，没有拟合区域数据。
`generate_source_load_forecast` 保留旧函数名，输出含义仍是天气条件下的合成实况，不是发行时刻预测。
新增字段与旧构造兼容；未改动 F/G 调度、故障或成对实验编排。

## 1. 风电：同一天气诊断与连续密度响应

**P：密度来源一致性。** 优先直接采样 B 的 `air_density_kg_m3[T,H,W]`，不再从 T/p 重算干空气密度覆盖湿空气结果。
风、温度、地面气压和密度按同一小时及同一站点行列采样；保存实际行列供独立验证。
**E：风高度换算。** `u_h=u_ref(z_h/z_ref)^alpha`；u 单位 m/s，z 单位 m AGL，alpha 无量纲。
**工程近似：** 默认采用等温虚温层，将地面密度换算到轮毂：

\[
\rho_h=\rho_s\exp[-g\rho_s z_h/(100p_s)].
\]

这里 rho 为 kg/m³，p_s 为 hPa，g 为 m/s²；100p_s 转为 Pa，指数无量纲。
B 气压位于地形地面，因此高度是完整 z_h AGL，不能减去仅属于风观测的 z_ref。
可选择 `surface_proxy` 直接使用地面密度作为轮毂代理；两种模式均明确记录。
**E/S：工程功率曲线。**

\[
f=\operatorname{clip}\left[\frac{\rho_h}{\rho_{ref}}
\frac{u_h^3-u_{ci}^3}{u_r^3-u_{ci}^3},0,1\right],\quad P=P_{nameplate}(1-\ell)f.
\]

实际 u_h≤u_ci 或 u_h≥u_co 时 P=0；切入和切出不由密度等效风决定。
u_r 是参考密度下的名义额定风速；实际达到平台的风速随密度变化，额定点附近保持连续。
rho_ref、损失和各速度阈值为配置；本曲线不是 OEM 曲线，也不代表尾流或机组控制器。
旧输入缺少诊断时，可明确回退为地面 p/T 干空气近似；若 p 也缺失，则使用轮毂参考密度。
回退来源逐母线写入 `wind_density_source_by_bus_id`；`wind_density_fallback: error` 可禁止回退。

## 2. 光伏：GHI 中的云效应只使用一次

**P/E：** 太阳几何和已有 GHI → Erbs 直散分解 → 各向同性天空倾斜面 POA。
地面反射、倾角、方位角沿用已有配置；不在 POA 或出力阶段另乘一次 cloud 因子。
**E：Faiman 温度近似。** `T_module=T_air+G_POA/(U0+U1*u_module)`。
G_POA 为 W/m²，U0 为 W/(m² K)，U1 为 W s/(m³ K)，温升单位 K。
模块温度作为电池温度代理；不宣称计算了完整组件瞬态热储存。
**S：模块高度风。** 采用可配置的模块高度和独立剪切指数，从参考高度风换算。
**E/S：** DC 功率按 POA/1000、参考温度 25°C、温度系数、DC/AC 比和系统损失计算。
经逆变器效率后限制到 AC 名牌；夜间零出力依据上游太阳模式的零 GHI。
此处额定容量为 AC MW；模块安装条件和系数仍是未校准场景配置。

## 3. 负荷：初态、因果记忆与共同随机数

保留居民、商业、工业周历曲线，以及以供暖/制冷平衡温度计算的温度响应。
**E/S：一阶热记忆代理。** `T_eff[n+1]=a*T_eff[n]+(1-a)*T_air[n]`，`a=exp(-dt/tau)`。
tau、dt 单位 h；主流程 dt=1 h，换算 helper 支持半小时。
保存的 `initial_effective_temperature_c[N]` 是窗口前边界，诊断序列是每小时更新后的区间终点状态。
`first_hour` 明确以 T_air[0] 作为初始状态；`configured` 使用配置初温，首小时也必须递推。
这不是建筑能量守恒模型，初态不使用未来温度均值；没有增加不必要的跨块残差状态协议。
**S：残差。** 沿用 AR(1) 和 `exp(-distance_km/L_km)` 空间协方差，保存 `load_log_residual`。
按负荷母线 ID 排序生成残差，同一时间轴、同一资产和 RNG 子种子允许成对干预共用随机扰动。
`calendar_start_date` 是代表性周历先验；太阳日序仍来自天气的 365 日气候时间轴，不另移太阳原点。
设计负荷容量是峰值参考；实际请求负荷不按设计峰值裁剪，极端需求可超过它。
供电是否送达、是否缺供由 F 决定，E 不根据全局可用功率差伪造节点缺供。

## 4. 新配置与兼容

| SourceLoadConfig 属性 | 默认值 | 单位/作用 |
|---|---|---|
| wind_reference_density_kg_m3 | 1.225 | kg/m³，名义曲线参考密度 |
| wind_density_height_mode | isothermal_surface_to_hub | 可改为 surface_proxy |
| wind_density_fallback | dry_air_then_reference | 可改为 error，禁止旧格式回退 |
| pv_module_height_m | 2 | m AGL，模块风高度 |
| pv_wind_shear_exponent | 0.14 | 无量纲，模块高度换算先验 |
| load_initial_temperature_mode | first_hour | 或 configured |
| load_initial_temperature_c | 20 | °C，仅 configured 初态使用 |
| thermal_dispatch_half_distance_km | 24 | km，火电计划分配的局地偏好半衰距离 |

新配置均追加在旧字段之后；完整快照和 `small_debug.yaml` 明确保存；构造/配置读取立即调用同一 `validate()`，不等生成中途报错。
物理常数 g、气体常数与单位换算不作为任意调参项；新增经验/场景参数均由配置给出。
火电局地偏好改用 `exp(-ln(2)*distance_km/half_distance_km)`，不是电气距离或线路约束。
旧内部 helper 可显式传像元半衰距离并换算为 km；主生成入口仅使用物理 km 配置。

## 5. 输入和输出契约

入口拒绝重复/非法 ID、缺失电气母线、容量不一致、负容量、越界或非整数采样坐标。
显式网格尺寸必须与天气一致；grid=None 仅保留已声明的默认网格假设，不伪装真实区域信息。
天气时间轴、九通道及附加诊断必须有限且维度一致；负风速/GHI、非法 RH、非正气压/密度拒绝。
温度不再被静默裁成 150 K；配置初温和外部温度必须超过绝对零度。
输出六个诊断：hub_wind_speed_mps、wind_air_density_kg_m3、pv_poa_w_m2、pv_module_temperature_c、load_effective_temperature_c、load_log_residual。
诊断输出 float32、内部计算 float64；非适用母线填零并由明确的 bus-kind 适用表解释。
名牌容量、基准需求和实际采样行列作为固定属性传给 Store；初温是窗口前边界状态，N 形状不代表它是静态属性。
Store 按 A 的显式 MW×h 积分导出请求、可用和计划能量，以及容量因子有效掩码；零容量不除零。
`available-scheduled` 中的闲置火电不称弃电；实际送达、火电回调、弃风弃光和缺供等待 F 运行结果。

## 6. 已执行验证

通过仓库内 Git Bash 脚本调用 `/d/anaconda/python.exe`，未安装依赖或提交 Git。
`tests/test_source_load_physics.py` 与 `tests/test_source_load_mechanisms.py` 合计 **56 项通过**。
新增测试固定时间轴、资产和随机种子，对 seed 42/123 验证湿度→密度→风电以及负荷残差不变。
解析检查覆盖半/一/双参考密度、切入/额定附近连续性、切出停机及轮毂 AGL 基准。
固定 GHI 升温使未削顶 PV 降低；未来热浪不改变历史负荷；只改云量而不改 GHI 不二次压低 PV。
配置初态首小时递推、半小时/整小时记忆一致、设计峰值不裁需求及 km 半衰距离均有独立反例。
10 MW×24 h=240 MWh，半小时积分=120 MWh；零名牌容量因子使用无效掩码。
独立只读复核额外验证了母线逆序后按 ID 对齐的负荷复现；显式网格形状反例已修复并回归。
全世界、缓存恢复及数据集总验收由 E 主记录汇总，此处不将核心测试替代总验收。
