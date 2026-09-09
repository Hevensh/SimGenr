# 气候、日天气和小时天气：研究依据与实现

本次修改将这部分定位为**具有物理约束的随机情景生成器**。没有输入某地多年观测并估计参数，因此结果不能宣称是该地区真实气候、可验证的气象预报或完整大气数值模式。下述物理关系与随机建模结构有文献依据；默认经验系数是演示先验，明确区分于论文结论。

## 文献与对应决策

| 一手论文或官方技术资料 | 支持的内容 | 本仓库采用方式 |
| --- | --- | --- |
| Allen et al. (1998), [FAO Irrigation and Drainage Paper 56, Chapter 3](https://www.fao.org/4/X0490E/x0490e07.htm) | 饱和水汽压、太阳赤纬、日地距离、太阳时角及晴空日辐射近似 | `weather/physics.py` 使用式 11、23–25、37 的结构，对正太阳高度的每个小时区间解析积分。 |
| Richardson (1981), [Stochastic simulation of daily precipitation, temperature, and solar radiation](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/WR017i001p00182), DOI 10.1029/WR017i001p00182 | 先生成降水，再根据干湿状态条件生成其他变量 | 采用这一因果生成次序；原论文使用 Markov–exponential，本实现扩展为 Markov–gamma，不能将 gamma 或本实现系数归于原论文。 |
| Parlange & Katz (2000), [An Extended Version of the Richardson Model for Simulating Daily Weather Variables](https://journals.ametsoc.org/view/journals/apme/39/5/1520-0450-39.5.610.xml) | 日天气可扩展到风速、露点，需考虑偏态分布与变量关系 | 正风速用对数正态扰动；先生成水汽压再由温度求 RH。没有复现文中的拟合参数或完整多变量协方差矩阵。 |
| Smith & Barstad (2004), [A Linear Theory of Orographic Precipitation](https://journals.ametsoc.org/abstract/journals/atsc/61/12/1520-0469_2004_061_1377_altoop_2.0.co_2.xml) | 地形抬升源与 `U·∇h` 有关，降水还取决于云水输送、转化、落出及下坡蒸发 | 修正迎风坡符号和地形梯度单位。当前仅使用有界坡向代理；不声称实现该论文的傅里叶传递函数，也不声称闭合大气水收支。 |
| [AMS Glossary: Hypsometric equation](https://glossary.ametsoc.org/wiki/hypsometric-equation/)，及 [An Example of Uncertainty in Sea Level Pressure Reduction (1998)](https://journals.ametsoc.org/view/journals/wefo/13/3/1520-0434_1998_013_0833_aeouis_2_0_co_2.xml) | 静力平衡压强随高度指数下降，并依赖层平均虚温 | 用测高公式替代海拔线性扣减；未观测气柱由 6.5 K/km 递减率近似。 |
| [NOAA MADIS wind-component convention](https://madis.ncep.noaa.gov/faq_datadisplay.shtml) | 未投影风场 `u/v` 分别是东西、南北分量 | 统一 u 向东、v 向北，地图行号向南递增。 |
| Reda & Andreas (2004), [NREL Solar Position Algorithm](https://midcdmz.nrel.gov/spa/) | 高精度太阳位置算法及其输入需求 | 用作精度边界参考。当前采用 FAO 简化几何，没有复现 SPA，也不借用其精度声明。 |

## 发现的具体问题

1. 原气候温度将整幅地图归一化行号直接乘 12°C，使 128 km 和 1000 km 地图都具有同一南北温差，并存在纬向方向含义不清的问题。现在加入中心纬度，使用 `latitude = latitude_center + ((H−1)/2−row)·cell_size_km/111.195`；`latitude_temperature_gradient_c` 表示赤道至极区的经验温差，按 `abs(latitude)/90` 应用。
2. 原迎风坡在 climate 和 weather 中使用不同符号，且混淆向北的 v 与向南增加的行号。现在使用 `u·dh/dx − v·dh/drow`，梯度除以米制单元尺寸；平地抬升为零。
3. 原日雨量公式有始终大于零的基线，几乎天天下雨；小时又添加独立暴雨，使日报与小时累计不一致。主流程现在只生成一条日序列，然后选择连续若干天降尺度。
4. 原日辐照在气候与天气中重复衰减云量，小时在两个入口用不同倍数；固定 6–18 点日照无法表现纬度或季节。现在日、小时共享同一太阳几何与量纲。
5. 原气压仅按 `0.012 hPa/m` 扣减，3000 m 才下降 36 hPa，远小于大气实际高度效应。新测高公式在标准气柱下约为 701 hPa。
6. 原动态噪声每次用空间最小值和最大值归一化，使局地天气取决于地图另一端极值，也破坏 AR 方差。现在用高斯卷积算子的理论边际方差归一化，随后做 AR 更新。

## 主要公式与数据契约

### 日天气条件随机模型

设长期湿日概率为 `p`、湿日持续性参数为 `r`，则

```text
P(wet_t | dry_(t−1)) = p(1−r)
P(wet_t | wet_(t−1)) = p+r(1−p)
E[wet_t] = p
R_t = wet_t · Gamma(k, scale=1/k) · P_annual · s(DOY)/(365p)
s(DOY) = 1 + 0.35 · hemisphere · cos(2π(DOY−112)/365)
```

初始干湿状态按平稳概率抽样。`s` 的整年平均是 1，因此**年累计在统计期望上等于气候基准**，保留不同年份与位置的随机年际差异，不强行将每个实现年归一化为同样雨量。空间相关高斯变量经正态 CDF 和 gamma 逆 CDF 产生相关 gamma 边际。湿日概率默认 0.30、持续性 0.55、shape 1.5；均为需要本地数据估计的演示先验。

天气扰动为 `X_t = rho·advect(X_(t−1)) + sqrt(1−rho²)·epsilon_t`。卷积采用反射边界；平移采用近邻边界，不将右侧云系瞬移到左侧。小数位移通过随机取整保留平均速度。这个平移速度是天气相关结构的经验传播速度，并不是把 10 m 风速当作整层云系风速。降水发生场具有空间相关和 Markov 时间持续性，未强制整个降雨系统按同一速度平流。

RH 来自 `e/e_s(T)`。气压为 `p(z)=p0·exp[−gz/(Rd·Tv_bar)]`，`Tv_bar≈(T_surface+273.15+0.00325z)(1+0.61q)`。温湿雨云使用共用扰动及干湿条件联系；混合权重、湿日降温 1.5°C、云量湿日增量 0.38 等都只是本实现的经验先验。

### 辐照与保守小时降尺度

`extraterrestrial_hourly_irradiance` 对 `[h,h+1)` 内太阳高度为正的时段积分，日出日落小时保留其部分白昼贡献；极夜结果严格为零。输出是 **W/m² 区间均值**。日 GHI 用当日 TOA 均值乘 FAO 晴空透过率和一次经验云量衰减。气候年 GHI 用 365 天平均计算，气候层云衰减先验固定 0.68；若用户改变天气层云敏感度，两者应重新联合校准。

小时序列先生成相关扰动，再逐格满足：

```text
sum_h rain_hourly[h] = rain_daily
mean_h irradiance_hourly[h] = irradiance_daily
mean_h T, RH, pressure, cloud, u, v, speed = respective daily value
0 ≤ RH, cloud ≤ 1
0 ≤ irradiance_hourly ≤ TOA_hourly
speed = hypot(u, v)
```

雨量采用条件于日总量的有限湿小时脉冲，没有附加降雨源。GHI 按太阳几何和小时云量分配，同时限幅在 TOA 之内；不可能的外部日 GHI 输入会明确报错。晴空逐小时透过率并未进行严格辐射传输计算，小时上限只是保守的 TOA 物理上限。

温度日变化在当地太阳时约 15 点达到峰值，云多时振幅较小；偏差去除日均值。小时 RH 通过求解一个日内水汽压使其满足日均 RH，饱和时上限为 1。这是假设日内水汽压近似恒定，未模拟凝结潜热收支。风在日内保持方向，仅调幅，以便同时守恒风矢量及风速日均值；变化的日边界仍可能有跳变，不能据此训练秒级或分钟级阵风模型。

### 单位、位置、日期

- 温度：2 m °C；风：10 m m/s；气压：站点 hPa；湿度：RH 0–1；云量：0–1。
- 降水：日或小时的累计水当量 mm；没有雨雪相态和积雪消融模型。
- GHI：水平面短波辐照 W/m²，包含夜间零值；不是辐照能量，也不是倾斜组件平面的 POA。
- 风向配置：从北顺时针的来向角度；270° 表示西风、向东吹。
- 天气整数时间戳：非闰年 Jan 1 为第 0 天，小时戳是 `day·24+h`；太阳几何按 365 天周期。当地太阳时不是 UTC，也未含时区/经度/夏令时/均时差。源荷日历以另行配置的代表年份解释星期。
- 地图使用局部球面近似；经度跨度未进入太阳时换算。不能把该几何直接当作跨时区大陆模型。跨极点的纬度范围会报错。

## 兼容性和后续校准

`generate_hourly_weather_week_from_baseline` 保留旧调用接口，内部改为生成配置起始日后的连续日天气再降尺度，不再随机选择月份或运行独立云团生成器。要与保存的日数据匹配，使用主流程的 `generate_hourly_weather_week(daily, ..., grid=grid)`。

以下旧参数为载入旧 YAML 保留，已不参与新的生成式：`ClimateConfig.irradiance_base_w_m2`；`WeatherConfig.hourly_storm_event_rate`、全部 `hourly_cloud_system_count/cloudlet_count/cloud_radius_*/cloud_motion_km_per_hour/raining_cloud_fraction`、`precipitation_event_scale`、`solar_seasonal_lag_days`、`low_frequency_temperature_c`、`low_frequency_irradiance_weight`。太阳季节相位现在由天文几何确定，不能再自由平移。`hourly_precipitation_burstiness` 现在控制日雨量在湿小时之间的分配强度。

下一步应使用目标区至少数年逐小时实测或再分析，估计月别湿日概率、转移概率、gamma 参数、温湿风云残差协方差、空间相关距离及小时降雨簇分布，再留出整年检验。可使用 [NOAA ISD 官方观测集](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) 中的温度、露点、风、站点压强、云及分时段降水；辐照还需专门的辐射观测/再分析。此次没有下载或拟合观测资料。

## 验证

`tests/test_weather_physics.py` 包含独立数值积分核对太阳小时解析积分、季节/半球/极昼极夜、标准大气压强基准、跨年日小时 9 通道守恒、夜间零辐照、TOA 上限、干日存在性、Markov 转移概率、三年统计年雨量预算、边界高斯方差及迎风坡方向测试。

已执行：`D:\anaconda\python.exe -B -m pytest tests/test_weather_physics.py tests/test_static_world.py -k 'climate or weather or solar or pressure or hourly or rainfall or noise or latitude' -q -p no:cacheprovider`；首次 11 项通过。随后补充 9 种非法外部日输入的参数化回归，拒绝非有限数据、负降水/风速/辐照、无效 RH/云量/压强/绝对温度及无法同时守恒的风矢量与风速；与调度核算和电网储能物理测试合跑 29 项通过。该验证证明实现约束和统计构造正确，不等价于现实气候拟合成功。
