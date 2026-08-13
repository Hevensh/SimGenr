# 模型结构与配置参数说明

本文说明 `models/` 下 Physics-GST、Physics-GST-Causal、LSTM、GRU 和 ARIMA 的输入、结构、输出、损失函数，以及
`configs/models/*.yaml` 中每个参数实际控制的模型部分。当前任务统一为：使用过去 24 小时数据，预测未来
24 小时的 node 负荷、node 计划发电和 line 潮流/负载率。

相关文件：

| 文件 | 职责 |
|---|---|
| `models/physics_gst.py` | Physics-GST 主模型和物理约束损失 |
| `models/baselines.py` | LSTM、GRU 与 ARIMA 基线 |
| `models/preprocessing.py` | 静态图预处理、时窗预处理和分位归一化 |
| `models/training_config.py` | 神经网络 YAML 配置读取 |
| `scripts/train_physics_gst.py` | Physics-GST/LSTM 训练、验证选模和测试 |
| `scripts/evaluate_arima_baseline.py` | ARIMA 训练集拟合及 val/test 评估 |

## 1. 预测对象和输入

记：栅格大小为 `H x W`，历史长度为 `Th`，预测长度为 `Tf`，node 数为 `N`，line 数为 `E`。

### 1.1 模型输入

| 模态 | 数据形状 | 当前主要内容 | 使用模型 |
|---|---:|---|---|
| 静态连续栅格 | `[22,H,W]` | 高程、水文、气候均值、人口和城市密度等 | Physics-GST |
| 静态离散栅格 | `[7,H,W]` | 流向、水体、土地覆盖、城市和土地用途等 | Physics-GST |
| 历史天气 | `[Th,9,H,W]` | 风、温湿度、气压、云、降雨、辐照度 | Physics-GST |
| 已知未来天气 | `[Tf,9,H,W]` | 预测时段的天气条件 | Physics-GST、LSTM |
| node 静态属性 | `[N,5]`、`[N,6]` | 坐标、容量及电气参数 | Physics-GST、LSTM |
| line 静态属性 | `[E,7]` | 长度、电压、阻抗和容量 | Physics-GST、LSTM |
| node 历史运行量 | `[Th,10,N]` | 负荷、发电、注入和供电状态等 | Physics-GST、LSTM、ARIMA |
| line 历史运行量 | `[Th,3,E]` | 潮流及线路负载率 | Physics-GST、LSTM、ARIMA |
| 图拓扑 | `[2,E]` | line 两端的本地 node 下标 | Physics-GST |

图结构只包含 **node 和 line**。A* 路径格点 `path_ptr/path_row/path_col` 仅用于线路空间轨迹和绘图，
不作为图结构，也不输入当前模型。

### 1.2 模型输出

| 输出 | 形状 | 约束方式 |
|---|---:|---|
| `p_load_mw` | `[Tf,N]` | `softplus` 后乘 load mask 和 node 功率尺度 |
| `p_generation_mw` | `[Tf,N]` | `softplus` 后乘 genr/storage mask 和 node 功率尺度 |
| `line_flow_mw` | `[Tf,E]` | `tanh` 后乘 `rate_mva`，保留潮流方向 |
| `line_loading_ratio` | `[Tf,E]` | 由 `abs(line_flow_mw) / rate_mva` 确定性计算 |

因此，负荷和发电不会被模型直接预测为负数，线路潮流不会超过当前 line 的额定容量，负载率也不会与
潮流和容量相互矛盾。

## 2. Physics-GST 结构

Physics-GST 是当前主模型，由栅格编码、历史时序编码、图消息传递和逐小时递归解码四部分组成。

```text
静态连续栅格 ─┐
静态离散 embedding ─┴─ static CNN ── node 坐标采样 ─┐
历史天气 ───────────── weather CNN + GRU ──────────┤
node 历史运行量 ─────────────── node GRU ──────────┼─ node 初始隐状态
node 静态/电气属性、类型、储能标记 ────────────────┘

line 历史运行量 ─ line GRU ─┐
line 静态属性 ───────────────┴─ edge 初始隐状态

未来第 t 小时天气 ─ weather CNN ─ node 采样
                         ↓
node 隐状态 ─ edge-aware GATv2 ─ node head ─ load/genr
       │             ↑                 │
       └──────── line 隐状态 ─ edge head ─ signed flow
                         ↓
                 GRUCell 更新到 t+1
```

### 2.1 栅格编码

静态连续通道先按每个世界、每个通道标准化。7 个离散通道分别经过 embedding，再与连续通道拼接。
静态栅格和天气栅格分别使用：

```text
Conv2d(kernel=5, padding=2) -> GELU
-> Conv2d(kernel=3, padding=1) -> GELU
```

两个卷积都不改变 `H x W`。输出通道数由 `grid_dim` 决定。编码后的栅格特征按照每个 node 的
`row/col` 位置采样，因此地形和天气最终以 node 特征的形式进入电网模型。

### 2.2 历史编码

- node 的 10 通道历史运行量由共享 GRU 编码，每个 node 得到一个 `hidden_dim` 隐状态；
- line 的 3 通道历史运行量由共享 GRU 编码，每条 line 得到一个 `hidden_dim` 隐状态；
- node 位置处的历史天气 CNN 特征由另一个 GRU 编码；
- node 类型使用 `5 -> 8` 的 embedding；
- node 初始状态还拼接 node 静态/电气属性、静态栅格特征和储能挂接标记；
- line 初始状态还拼接 7 个归一化 line 静态属性。

### 2.3 图消息传递

每个预测小时先把该小时的已知未来天气加入 node 状态，再进行 `graph_layers` 层 GATv2：

```text
message = GATv2(node_state, bidirectional_edge_index, edge_state)
node_state = LayerNorm(node_state + GELU(message))
```

原始 line 在消息传递时扩展为双向边，但输出潮流仍按原始 `edge_index` 的 from-to 方向预测。
GATv2 使用 line 隐状态作为 edge feature，因此线路历史和电气属性会参与邻居注意力权重计算。

### 2.4 GRU 与 causal-attention 历史 backbone

`physics_gst` 使用三套 GRU 分别编码 node 历史、line 历史和 node 位置处的历史天气。
`physics_gst_causal` 保留其余结构，只把这三套历史 GRU 换为因果 Transformer encoder：

```text
input projection + sinusoidal position encoding
-> causal multi-head self-attention
-> feed-forward network
-> 取最后一个历史时刻作为实体历史状态
```

上三角 causal mask 禁止任意历史时刻读取未来位置。`temporal_attention_layers` 控制堆叠层数，
`temporal_attention_heads` 控制每层注意力头数。该变体的逐小时输出端仍保留 GRUCell，作为共享的轻量
自回归解码器；因此比较重点是 GRU 历史 backbone 与 causal-attention 历史 backbone。

### 2.5 逐小时解码

node head 输出 load/genr 两个相对功率，line head 同时读取 line 两端 node 状态和 line 状态，输出有符号
潮流。当前小时结果再分别送入 node/line `GRUCell`，形成下一小时的状态。因此 24 个未来时刻不是一次性
独立输出，而是按时间递归生成。

## 3. LSTM 与 GRU 基线

LSTM 和 GRU 保留相同的输入输出语义和硬输出约束，但不使用静态/天气 CNN、图拓扑、GATv2 或物理损失。

- LSTM 版本为每个 node/line 共享历史 LSTM 和解码 LSTMCell；
- GRU 版本在完全相同的位置改用 GRU 和 GRUCell；
- node 初始状态包含 node 静态/电气属性、node 类型和储能标记；
- node 每步输入为上一时刻 load/genr 比例和该 node 的未来天气；
- line 每步输入为上一时刻有符号潮流比例；
- node 与 node、line 与 line 之间不进行消息传递。

这两个基线用于衡量单实体循环结构的差异，以及仅依靠时间序列、静态属性和未来天气能达到怎样的水平。它们与 Physics-GST
使用相同的监督目标和分位归一化，便于直接比较。

## 4. ARIMA 基线

ARIMA 对 load、genr 和 line flow 分别建立共享的 `ARIMA(p,d,q)` 动力学基线：

1. 只从 train worlds 收集有效 load node、genr/storage node 和全部 line 的完整序列；
2. 每条训练序列先按自身均值和标准差标准化；
3. 每类实体最多均匀选取 `max_train_series_per_entity` 条序列独立拟合；
4. 在拟合参数中选择最接近参数中位中心的一组合法参数，作为该实体类型的共享参数；
5. val/test 不重新拟合参数，只用各自 24 小时历史初始化固定参数滤波器并预测未来 24 小时；
6. line loading 仍由预测 flow 和 `rate_mva` 计算。

ARIMA 不使用天气、静态栅格或图拓扑。其作用是提供纯单变量时间序列参照。

## 5. Physics-GST 配置参数

旧版完整 Physics-GST 结构仍由 `models/physics_gst.py` 保留；当前 GRU-GNN 对照配置改为
`configs/models/gru_gnn_supervised_24to24.yaml` 和 `configs/models/gru_gnn_physics_24to24.yaml`。

### 5.1 `model`

| 参数 | 当前值 | 作用位置 | 含义和影响 |
|---|---:|---|---|
| `type` | `physics_gst` | 模型构造 | 选择 Physics-GST |
| `parameters.hidden_dim` | 256 | GRU、GATv2、heads、GRUCell | node/line 隐状态宽度；增大可提升容量，也会明显增加显存和计算量 |
| `parameters.grid_dim` | 128 | 静态 CNN、天气 CNN | 每个栅格编码器的输出通道数；主要控制空间环境表征容量 |
| `parameters.categorical_embedding_dim` | 4 | 7 个离散栅格 embedding | 每个离散通道的 embedding 宽度 |
| `parameters.attention_heads` | 4 | 每层 GATv2 | 多头注意力数量；必须整除 `hidden_dim` |
| `parameters.graph_layers` | 2 | GATv2 堆叠 | 每个预测小时的图传播层数；2 层允许信息传播到约两跳邻域 |
| `parameters.dropout` | 0.10 | GATv2 attention | 图注意力中的 dropout 比例 |

旧版 `physics_gst_causal` 在以上参数之外增加：

| 参数 | 当前值 | 作用位置 | 含义和影响 |
|---|---:|---|---|
| `model.type` | `physics_gst_causal` | 模型构造 | 选择 causal-attention 历史 backbone |
| `temporal_backbone` | `causal_attention` | 三路历史编码器 | 将 node/line/weather 历史 GRU 替换为 causal attention |
| `temporal_attention_heads` | 4 | 历史自注意力 | 注意力头数，必须整除 `hidden_dim` |
| `temporal_attention_layers` | 2 | 历史自注意力 | 每一路历史 Transformer encoder 层数 |

代码中另有 `categorical_cardinalities=(10,4,4097,8,3,128,8)`，依次给 7 个离散栅格通道设置 embedding
词表上限。它目前没有写入 YAML；只有类别编码范围发生变化时才需要调整。

### 5.2 `loss`

| 参数 | 当前值 | 乘到的损失 | 作用 |
|---|---:|---|---|
| `loading_weight` | 0.25 | `line_loading` | 监督损失中线路负载率的相对权重 |
| `balance_weight` | 0.20 | `power_balance` | node 有功功率平衡约束权重 |
| `ramp_weight` | 0.02 | `genr_ramp` | 相邻预测小时发电变化惩罚权重 |
| `nonnegative_weight` | 0.0 | load/genr 非负诊断 | 默认不重复惩罚，因为输出已由 `softplus` 保证非负 |
| `line_capacity_weight` | 0.0 | line 超容量诊断 | 默认不重复惩罚，因为 `tanh * rate_mva` 已限制潮流 |
| `loading_consistency_weight` | 0.0 | loading 一致性诊断 | 默认不重复惩罚，因为 loading 是确定性计算值 |

监督损失使用 Smooth L1：

```text
L_supervised = L_load + L_genr + L_line_flow
             + loading_weight * L_line_loading
```

node 功率平衡使用原始物理量计算。设 line 正方向为 from-to，`divergence` 在 from node 加正潮流、在
to node 加负潮流，则：

```text
residual = generation - load - divergence
L_balance = mean((residual / node_power_scale)^2)
```

发电爬坡损失为：

```text
L_ramp = mean(diff(generation / node_power_scale, time)^2)
```

最终损失：

```text
L_total = L_supervised
        + balance_weight * L_balance
        + ramp_weight * L_ramp
        + nonnegative_weight * (L_load_nonnegative + L_genr_nonnegative)
        + line_capacity_weight * L_line_capacity
        + loading_consistency_weight * L_loading_consistency
```

权重为 0 的物理项仍会计算和记录，用于确认硬约束是否正常。

## 6. LSTM 与 GRU 配置参数

当前配置文件：`configs/models/lstm_24to24.yaml`。

| 参数 | 当前值 | 作用位置 | 含义和影响 |
|---|---:|---|---|
| `model.type` | `lstm` | 模型构造 | 选择 LSTM 基线 |
| `model.parameters.hidden_dim` | 512 | 历史 LSTM、LSTMCell 和 heads | 单实体时序隐状态宽度 |
| `model.parameters.dropout` | 0.10 | node/line 初始投影 | 初始状态投影后的 dropout；单层 LSTM 本身未设置 dropout |
| `loss.loading_weight` | 0.25 | 监督损失 | 与 Physics-GST 相同的 loading 相对权重 |

LSTM 的网络结构本身不包含图传播，但损失与结构已解耦：`loss.physics_enabled: false` 时
`total` 就是 `supervised`；设为 `true` 时，可对同一 LSTM 输出加入功率平衡、爬坡等物理软约束。

`configs/models/gru_24to24.yaml` 复制了相同的训练、数据和损失设置，仅有以下模型差异：

| 参数 | 当前值 | 作用位置 |
|---|---:|---|
| `model.type` | `gru` | 选择 GRU baseline |
| `model.parameters.hidden_dim` | 512 | 历史 GRU、GRUCell 和 heads |
| `model.parameters.dropout` | 0.10 | node/line 初始投影 |

GRU 同样由 `loss.physics_enabled` 决定使用纯监督损失还是物理约束损失。

## 7. 神经网络公共配置

下列参数同时适用于 Physics-GST 和 LSTM。

`experiment.name` 只负责实验目录和结果分组；`model.type` 只负责选择网络结构；
`loss.physics_enabled` 则独立决定采用纯监督损失还是叠加物理约束。八组正式对照实验可统一运行：

```bash
bash scripts/train_eight_variants.sh
```

四个架构分别为 LSTM、GRU、GRU-GNN 和 Attention-GNN，每个架构都有一份 supervised 配置和一份
physics 配置。同一对配置的模型参数完全一致，避免把结构变化误判为物理约束带来的变化。

当前 `gru_gnn` 是普通 GRU 的受控扩展：load/genr 节点分支保持与 GRU 基线相同，GATv2 只向
line-flow 输出头提供残差修正。修正头以全零输出初始化，因此训练起点与普通 GRU 的三类预测严格一致；
图状态通过可学习门控接入，且不会写回节点递归状态，从而避免在 24 个预测小时中反复传播后干扰节点预测。

### 7.1 `normalization`

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `method` | `quantile` | 启用 train-only 分位归一化 |
| `left_quantile` | 5 | 归一化下基准 `Q5` |
| `right_quantile` | 95 | 归一化上基准 `Q95` |
| `left_clip` | 1 | 归一化结果最小截断到 `-1` |
| `right_clip` | 1 | 归一化结果最大截断到 `2`，即 `1 + right_clip` |
| `max_samples` | 500000 | 每个目标统计分位数时最多抽取的 train 数值数 |

对 load、genr、line flow 和 line loading 分别计算：

```text
z = clip((x - Q5) / (Q95 - Q5), -left_clip, 1 + right_clip)
```

分位数只从训练集估计，并保存进 checkpoint。监督损失及 RMSE/MAE/R2 在这一归一化尺度上计算；功率
平衡和其他物理约束仍在 MW/MVA 物理量上计算，再通过 node/line 尺度无量纲化。

天气采用每个时窗历史 24 小时的通道均值和标准差进行标准化，同一组统计量用于该时窗的未来天气。
静态连续栅格则按每个世界、每个通道标准化。

### 7.2 `data`

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `dataset` | `datasets/seed_1_50` | 数据集根目录及 train/val/test manifest 来源 |
| `history_hours` | 24 | 每个样本输入的历史小时数 `Th` |
| `forecast_hours` | 24 | 每个样本输出的未来小时数 `Tf` |
| `stride` | 24 | 同一世界相邻时窗起点间隔；24 表示每天取一个窗口 |

改变历史或预测长度不会改变网络层定义，但会改变 GRU 处理长度、递归解码步数和样本数量。

### 7.3 `training` 与 `optimizer`

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `training.epochs` | 20 | 最大训练轮数 |
| `training.gradient_clip_norm` | 1.0 | 反向传播后对全模型梯度范数裁剪 |
| `training.max_train_windows` | 0 | 每轮最多训练窗口数；0 表示全部 |
| `training.max_eval_windows` | 0 | 每轮最多验证窗口数；0 表示全部 |
| `optimizer.learning_rate` | 0.0003 | AdamW 学习率 |
| `optimizer.weight_decay` | 0.0001 | AdamW 权重衰减 |

当前训练按“一个可变大小 world graph 一个 step”执行，没有把多个世界拼成 PyG batch。

### 7.4 `evaluation`

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `selection_loss` | `total` | 每轮验证后用于选择最佳 checkpoint 的损失字段 |
| `evaluate_test` | `true` | 训练结束后是否加载最佳验证模型并评估 test |

训练只根据最小 `val total` 选择模型。训练结束后恢复该轮权重，再对 test 评估一次；test 不参与调参和
checkpoint 选择。若把 `selection_loss` 改为其他名称，该名称必须实际存在于模型返回的 loss 字典中。

每次评估记录 `load`、`genr`、`line` 和 `average` 四行 RMSE/MAE/R2。`average` 是前三类实体对应指标的
算术宏平均；由于三类误差使用同一训练集分位尺度，该平均值可作为整体比较项。R2 平均同样是三个实体
R2 的算术平均，不是把三类张量拼接后重新计算。

### 7.5 `runtime` 与 `output`

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `runtime.device` | `cuda` | 模型和张量所在设备 |
| `runtime.preload_to_device` | `true` | 是否把选中 worlds 和世界级静态/图预处理结果预载到设备 |
| `runtime.initialization_seed` | 2026 | Python、NumPy、PyTorch 和 CUDA 初始化随机种子 |
| `output.path` | `null` | 自定义 checkpoint 路径；null 时按模型/种子/时窗自动生成 |

预载只缓存完整 world 和与模型参数无关的静态图结果，不缓存所有时间窗口。窗口索引保留在 CPU，天气
标准化、历史切片、node 天气采样和 target 在取窗口时生成，以控制显存占用。

默认输出示例：

```text
checkpoints/physics_gst/seed_2026/20260721_1530/24to24.pt
checkpoints/physics_gst/seed_2026/20260721_1530/24to24.json
checkpoints/lstm/seed_2026/20260721_1614/24to24.pt
```

每次神经网络训练会在 `seed_<初始化种子>` 下创建本地时间目录 `YYYYMMDD_HHMM`。结果 JSON 记录
`run_id`、`started_at`、原配置路径、解析后的 `model_config` 和完整 `training_config`。同一模型、同一初始化
种子的多次运行不会覆盖。若显式设置 `output.path`，则仍以该路径为准。自动时间目录若已存在则直接报错，
避免同一分钟内重复运行时静默覆盖已有结果。

## 8. ARIMA 配置参数

当前配置文件：`configs/models/arima_24to24.yaml`。

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `model.type` | `arima` | 标识 ARIMA 基线 |
| `model.order` | `[2,0,1]` | ARIMA 的 `(p,d,q)`：2 阶 AR、无差分、1 阶 MA |
| `model.max_train_series_per_entity` | 128 | load/genr/line flow 每类最多用于拟合的训练序列数；0 表示全部 |
| `data.history_hours` | 24 | val/test 固定参数滤波器的初始化历史长度 |
| `data.forecast_hours` | 24 | 每个窗口的预测长度 |
| `data.stride` | 24 | 评估窗口间隔 |
| `evaluation.partitions` | `[val,test]` | 需要评估的数据分区 |
| `evaluation.max_windows_per_partition` | 0 | 每个分区最多评估窗口数；0 表示全部 |
| `output.path` | `null` | 自定义结果路径；null 时按 ARIMA 阶数和时窗命名 |

ARIMA 配置中的 `normalization` 只用于统一 val/test 指标尺度；ARIMA 拟合本身使用每条序列自身的均值和
标准差。默认结果路径为：

```text
checkpoints/arima/order_2_0_1/24to24_val_test.json
```

## 9. 参数调整建议

| 目标 | 优先调整 | 注意事项 |
|---|---|---|
| 提升图关系表达能力 | `hidden_dim`、`graph_layers` | `hidden_dim` 必须被 `attention_heads` 整除；层数过多可能过平滑 |
| 提升地形/天气表达能力 | `grid_dim` | 会增加两套 CNN 及 node 天气 GRU 的计算量 |
| 降低显存 | `hidden_dim`、`grid_dim`、`preload_to_device` | 时窗目前未全量缓存，通常先减网络宽度 |
| 更强调守恒 | `balance_weight` | 过大可能牺牲单项预测误差，需要同时查看 load/genr/line 指标 |
| 让发电曲线更平滑 | `ramp_weight` | 过大会压制真实的快速调峰和新能源变化 |
| 调整 line loading 重要性 | `loading_weight` | flow 已是独立监督项，loading 是其按容量换算后的补充目标 |
| 快速调试 | `epochs`、`max_train_windows`、`max_eval_windows` | 调试值不应直接用于最终模型比较 |
| 控制 ARIMA 运行时间 | `max_train_series_per_entity`、`max_windows_per_partition` | 减少后会降低代表性，应在正式评估时恢复全部窗口 |

## 10. 运行方式

所有模型参数均从 YAML 读取，命令行只指定配置文件。

```bash
source C:/ProgramData/miniconda3/etc/profile.d/conda.sh
conda activate myEnv

python scripts/train_physics_gst.py \
  --config configs/models/gru_gnn_physics_24to24.yaml

python scripts/train_physics_gst.py \
  --config configs/models/attn_gnn_physics_24to24.yaml

python scripts/train_physics_gst.py \
  --config configs/models/lstm_24to24.yaml

python scripts/train_physics_gst.py \
  --config configs/models/gru_24to24.yaml

python scripts/evaluate_arima_baseline.py \
  --config configs/models/arima_24to24.yaml
```

进行正式对比时，应保持三类模型的 `dataset/history_hours/forecast_hours/stride` 和归一化参数一致，并使用
相同 manifest 中的 train/val/test 划分。

## 11. 分模块推理速度测试

`scripts/benchmark_inference_speed.py` 根据 `configs/models/inference_speed.yaml` 依次测试四个神经网络。
配置可指定设备、seed、窗口、热身次数、计时次数、模型配置和可选 checkpoint。`checkpoint: null` 表示用
随机初始化权重测试纯结构速度；加载已训练权重不会改变计算图和参数量。

```bash
python scripts/benchmark_inference_speed.py \
  --config configs/models/inference_speed.yaml
```

输出目录默认为 `checkpoints/results/inference_speed/`：

| 文件 | 内容 |
|---|---|
| `inference_speed.json` | 设备、模型参数量、端到端延迟和全部分模块记录 |
| `inference_speed.csv` | 每行一个模型模块，便于汇总绘图 |
| `inference_module_speed.csv` | 脱离完整调度后的模块单次耗时、调用次数和加权耗时 |

端到端延迟在关闭 profiler 的独立循环中测量。分模块耗时由 profiler 语义区间统计，包括：

- `input.static_grid_encoding`：静态栅格 embedding、CNN 和 node 采样；
- `input.weather_grid_encoding`：历史/未来天气 CNN 和 node 采样；
- `input.*_history_encoding`：node、line、weather 的 GRU 或 causal attention；
- `input.future_weather_sampling`：循环 baseline 的未来天气 node 采样；
- `fusion.initial_state`：时序、静态、类型和储能特征融合；
- `fusion.graph_attention`：每个预测小时的 GATv2 传播；
- `decode.*`：node/line 输出 head 及逐小时递归更新。

每条记录同时包含每次推理的 CPU 时间、CUDA 时间和调用次数。CUDA 时间用于判断 GPU 算子开销，独立的
端到端时间用于模型实际速度比较；不要把存在嵌套关系的 profiler 时间再次简单求和。

脚本还会对真实 forward 中出现的顶层模块注册 hook，捕获其实际输入和实际调用次数。随后把 CNN、
GRU/LSTM/causal encoder、每层 GATv2、LayerNorm、projection、head 和 recurrent cell 脱离完整调度单独
重复执行，并计算：

```text
weighted_device_ms = unit_device_ms * calls_per_inference
module_sum_device_ms = sum(weighted_device_ms)
```

该总和适合比较模块本身的纯计算成本，不包含 Python 循环、张量索引/拼接、mask、采样和模块之间的数据
调度。`benchmark.module_warmup_iterations` 和 `benchmark.module_measured_iterations` 分别控制模块热身与
重复计时次数。
