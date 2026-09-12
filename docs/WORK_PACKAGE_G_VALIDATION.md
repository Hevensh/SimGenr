# G：四态物理验证与失败定位

本包升级验证报告的表达和诊断，不修改生成公式或放宽物理约束。
公开入口仍是 `scripts/validate_world_physics.py`；旧 `Checks` 导入保持兼容。

## 1. 文件

- `scripts/validation_checks.py`：四态行记录、数值残差、显式上下文和输入错误结构。
- `scripts/validate_world_physics.py`：公开 CLI、域上下文、覆盖范围与 JSON/Markdown。
- `scripts/hydrology_validation.py`：水量账位置与时间标注。
- `scripts/source_load_validation.py`：能量、设备关系、热状态时间支撑与 P/E/S。
- `scripts/operation_validation.py`：逐岛平衡、备用、DC 关系与资产信息边界标注。
- `core/config.py`、`configs/small_debug.yaml`：仅诊断展示参数。
- `tests/test_validation_framework.py`：四态与定位的解析反例。

## 2. 四态和旧 API

| status | passed | 含义 |
|---|---|---|
| PASS | true | 已执行的支持范围内约束满足原有容差 |
| FAIL | false | 已执行约束失败或数值不合法 |
| UNSUPPORTED | null | 没有实现或无法由现有过程支持的检验 |
| NOT_RUN | null | 没有适用对象、过程关闭或缺少该版本过程 |

`equal(name,residual,tolerance,unit,note)`、`upper(...)`、`condition(...)` 保留旧调用形式。
相对容差仍逐元素使用，不能用远处的大库存掩盖局部水量错误。
空残差数组返回 NOT_RUN，不利用空集上的 `all()` 制造 PASS。
`condition` 只接受布尔值/布尔数组，NaN、2、整数数组均明确拒绝。
现代总结果仅汇总已执行的 PASS/FAIL，不把覆盖缺口当成通过或物理失败。
只有 UNSUPPORTED/NOT_RUN 而没有已执行约束时，总状态为 NOT_RUN。

## 3. 原始误差和违规幅度

`max_residual` 是原始残差绝对值的最大值，未扣容差。
`raw_max_signed` 另存原始残差的有符号最大值。
`max_violation` 是超过各元素允许误差之后的最大正超量。
等式的比较量为 `abs(residual)`，上界的比较量为 `values-upper`。
旧 `max_error` 字段保留原含义，便于旧报告消费者迁移。
`raw_max_location` 与 `max_violation_location` 分别对应两个最大值。
两者在相对容差下可能位于不同元素；不会把一个位置配给另一项数字。
`location_selection` 说明主要 location 是最大违规点还是通过情况下的最大原始残差点。
非有限残差为 FAIL，记录数量和索引；JSON 不写非标准 NaN/Infinity。

## 4. 显式时间和位置

每行至少包含 name、status、passed、max_residual、location、time、fields、stage、relation_class、engineering_simplification。
位置保存残差数组的索引和明确的轴名，不从第一维长度推断时间。
`context(stage,fields,timestamps,axes,time_support,...)` 支持域默认值和单检查覆盖。
时间轴必须明确命名 `time` 或 `state_time`，其 timestamps 长度必须匹配。
静态 N 恰好等于 T 时仍不会被解释成逐小时量。
区间水量、KCL、能量账使用区间起始时间；SOC 使用 T+1 边界时间。
负荷有效温度诊断使用每小时末边界；初温使用运行前边界。
每岛平衡的扁平观察序列额外记录逐观察时间，允许同一小时有多个岛。
全周期积分或未给出明确时间轴的聚合，其 time.status 为 not_applicable。
失败列表按违规排序截断，但即使列表预算为零仍保存主最大位置。

## 5. 覆盖边界与错误类别

报告显式列出 AC 电压/无功、频率稳定、备用激活后的网络可送达性和真实地区校准为 UNSUPPORTED。
动态水文关闭时水量过程是 NOT_RUN，静态汇流骨架不冒称动态流量。
P 表示守恒、单位和约束账；E 表示已声明的工程/经验关系；S 表示场景先验和服务规则。
水桶/线性路由、Faiman 温度、风电曲线和聚合备用均说明工程近似边界。
物理约束通过可以包含已记录缺供，不自动等于供电充足。
读文件、生成数据、形状或时间对齐错误保持 `status=FAIL`，另以 `error_category=INPUT_OR_GENERATION_ERROR` 分类。
该类别和真正已求值的物理约束 FAIL 分开，不能据输入错误宣称物理不可行。
若 `generation_failure.json` 仍存在，验证器报告本次生成失败并保留其类别，打包入口也拒绝读取旧产物；成功重生成才会清除该标记。

## 6. 配置与快照

`ValidationConfig.max_failure_locations` 默认 5，允许整数 0..1000，禁止 bool。
该参数只控制额外失败位置的展示数量，不修改容差或生成行为。
完整参数进入世界快照；更改验证展示不应要求重生成物理世界。

## 7. 实际验证

定向脚本为 `outputs/.implementation-tools/test_g_validation.sh`。
运行框架、天气、土地、水文、源荷、运行验证及字段契约七个测试文件。
加入失败标记与旧产物拒绝反例后，实际结果 **105 passed，1.36 s**，物理公式和原约束容差保持不变。
原测试仅两处空对象断言更新：零能源项目和无湖闭桶。
它们继续要求没有 FAIL、关键面积/水量账 PASS，并明确断言空项目/湖检查 NOT_RUN。
新增反例覆盖 raw/violation 分居两格、T=N、零失败列表预算、非法布尔真值和错误时间轴。
所有执行使用仓库内 Git Bash 脚本、D 盘 Python、UTF-8 和仓库内缓存/临时路径。
完整世界矩阵、干预检验和最终全回归由 G 主记录汇总，此处不重复计数。
