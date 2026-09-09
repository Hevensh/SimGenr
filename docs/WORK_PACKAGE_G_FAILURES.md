# G：生成失败分类与可审计失败记录

本次改动区分模型约束不可行、输入错误以及求解/生成失败，避免把软件或数值错误解释成
物理上无法供电。分类针对本次配置和简化模型，不是对现实电网可行性的证明。
运行公式、优化目标、约束和原有数值重试策略均未改变。

## 1. 类型和边界

新增 `world_generator/core/errors.py`：

| 异常或条件 | 记录类别 | 含义 |
|---|---|---|
| `PhysicalInfeasibilityError(RuntimeError)` | `PHYSICAL_INFEASIBILITY` | LP/MILP 明确返回 `status == 2`，配置约束下不可行 |
| `SolverError(RuntimeError)` | `GENERATION_ERROR` | 数值错误、求解器终止、依赖缺失或非法数值解，不能推断不可行 |
| 既有配置/数组契约的 `ValueError` | `INVALID_INPUT` | 输入未满足声明的单位、形状、范围或配置规则 |
| 配置阶段未知字段导致的 `TypeError` | `INVALID_INPUT` | 配置模式不受支持 |
| 其他阶段 `TypeError`、其他异常 | `GENERATION_ERROR` | 保留为生成/实现错误，避免误归输入 |
| CLI 参数错误 `SystemExit(2)` | `INVALID_INPUT` | 命令参数不合法，仍以非零状态退出 |

异常可携带具体 `stage`。Stage14 的 LP 和 MILP 错误会明确标记 `stage_14_dispatch`，
覆盖入口当时较粗的“Stages 12-14”进度描述。无成功结论的求解终止，包括时限、
无界或数值失败，均不能写成物理不可行。

原 LP 对 status4 的独立内点算法重试保留，使用相同约束；只依据最终结果分类。
储能充放电互斥 MILP 使用相同规则。输入上界、守恒和缺供规则没有被放宽。

## 2. 输出记录

入口 `main()` 使用轻量上下文调用原生成流程 `_generate(context)`。发生异常后，在已解析
且位于项目工作区内的世界目录写入 `generation_failure.json`，随后重新抛出原异常，
保留 traceback 和非零退出。结构为：

```json
{
  "schema_version": "generation_failure_v1",
  "status": "FAIL",
  "category": "PHYSICAL_INFEASIBILITY",
  "stage": "stage_14_dispatch",
  "error_type": "PhysicalInfeasibilityError",
  "message": "Stage 14 DC-OPF infeasible under the configured capacity bounds: ...",
  "config": "配置文件绝对路径",
  "seed": 42,
  "resolved_config": {"seed": 42},
  "world_directory": "已经解析的世界目录绝对路径"
}
```

`resolved_config` 在合法配置解析和 CLI 覆盖完成后记录实际完整配置；示例为节略。
阶段来源是当前生成进度或异常携带的更具体阶段。普通生成、Stage13 恢复和 Stage14
恢复均经过同一外层边界。

若配置在世界名、种子和输出目录可确定之前就解析失败，只保留非零退出及 stderr，
不猜测世界目录，不写一个归属不明的标记。未解析输出、工作区外路径和项目根目录
本身也不接收失败记录。世界名必须为单个目录名，不能包含路径分隔符。
记录自身的文件写入错误只写 stderr，不替换原始失败原因。

同一世界成功完成后删除残留失败标记，避免先前失败污染本次结果。
修改仅影响校验显示的 `validation` 配置不会使物理缓存过期；其与 `output` 一样
从恢复时的上游物理配置比较中排除。

## 3. 缺供不是默认失败

允许缺供时，孤岛负荷可通过正的 ENS/MWh 结果满足优化模型的守恒和约束，
这属于有效运行样本，不能因存在 ENS 自动写 `FAIL`。
禁止缺供、供电通道全部断开且没有本地可用供能时，同一负荷案例的约束无解，
求解器 status2 才对应 `PhysicalInfeasibilityError`。
物理可行性与研究情景“运行效果好坏”由此保持独立。

## 4. 测试

新增 `tests/test_generation_failure_reporting.py`，覆盖：

- 真实两母线孤岛、严格禁止缺供的 LP，返回 typed 物理不可行。
- LP status1/3/4 保持求解错误，status4 的原 IPM 重试仍执行。
- MILP 注入 status1/2/3/4，只有 status2 属于物理不可行。
- 真实不可行异常在入口写结构化记录后仍抛出，阶段、配置与种子正确。
- 非整数实体 ID、未知配置字段、模拟实现错误和求解器错误的类别区分。
- 相同孤岛在允许缺供时成功，产生 ENS 并删除既有失败记录。
- 未知输出、工作区外目录和项目根目录不被失败记录写入。
- CLI 坏配置实际非零退出；输出已解析后的非法阶段实际产生失败记录并退出 2。

这些测试验证错误分类与持久化行为，不模拟新的物理过程，也不改变合法样本分布。
