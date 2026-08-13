# SimGenr 数据集使用说明

字段、形状、单位和编码含义见 `datasets/DATA_DESCRIPTION.md`。本文只说明生成、加载和训练切窗方式。

## 1. 数据集结构

一个样本对应一个程序化世界及其连续 168 小时运行数据：

```text
datasets/seed_1_50/
├── manifest.json
├── schema.json
├── generation_report.json
└── samples/
    ├── seed1.npz
    ├── seed2.npz
    └── ...
```

每个 `seed*.npz` 是一个完整样本，内部使用 `static__`、`dynamic__`、`graph__` 和 `operation__` 前缀分组。PNG/WebP 不进入数据集。

## 2. 生成与打包

批量生成 seed 1–50：

```bash
python scripts/generate_dataset.py
```

命令默认不渲染图片，并带有 seed 总进度、单个世界阶段进度和打包进度。重复执行时会跳过完整世界；`--force` 强制重跑，`--fail-fast` 在首个失败 seed 停止。

只查看 train/val/test 划分：

```bash
python scripts/generate_dataset.py --plan-only
```

仅重新打包已有世界：

```bash
python scripts/build_dataset.py \
  --input-root outputs \
  --output datasets/small_debug_preview \
  --worlds small_debug_seed42 small_debug_seed123
```

## 3. NumPy 读取

```python
from world_generator.dataset import SimGenrDataset, channel_index

dataset = SimGenrDataset(
    "datasets/seed_1_50",
    verify_checksums=True,
)
sample = dataset[0]

print(sample["sample_id"])
print(sample["partition"])
print(sample["static"]["continuous"].shape)   # [22, 64, 64]
print(sample["static"]["categorical"].shape)  # [7, 64, 64]
print(sample["dynamic"]["weather"].shape)     # [168, 9, 64, 64]
print(sample["graph"]["node_features"].shape)
print(sample["graph"]["edge_features"].shape)
print(sample["operation"]["node_dynamic"].shape)
print(sample["operation"]["line_dynamic"].shape)

elevation_index = channel_index(
    sample["static"]["continuous_channels"],
    "elevation",
)
elevation = sample["static"]["continuous"][elevation_index]

cloud_index = channel_index(
    sample["dynamic"]["weather_channels"],
    "cloud",
)
cloud = sample["dynamic"]["weather"][:, cloud_index]
```

Loader 不做归一化。均值、方差、极值和类别词表只能用 train seed 计算。

## 4. PyTorch DataLoader

```python
from world_generator.dataset import SimGenrDataset, make_dataloader

dataset = SimGenrDataset(
    "datasets/seed_1_50",
    as_torch=True,
)
loader = make_dataloader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))

print(batch["static"]["continuous"].shape)  # [B, 22, 64, 64]
print(batch["dynamic"]["weather"].shape)   # [B, 168, 9, 64, 64]
print(len(batch["graph"]))                  # B
```

不同世界的 node、line 和 storage 数量不同，因此 `batch["graph"]` 保持为长度 `B` 的列表。对应的节点/线路时序也可能保持列表，不应把补零节点误当作真实设施。

## 5. 时间窗口

将每个世界切成“过去 24 小时预测未来 6 小时”，每 6 小时取一个窗口：

```python
from world_generator.dataset import SimGenrDataset, TemporalWindowDataset, make_dataloader

worlds = SimGenrDataset(
    "datasets/seed_1_50",
    as_torch=True,
    cache_size=2,
)
windows = TemporalWindowDataset(
    worlds,
    history_hours=24,
    forecast_hours=6,
    stride=6,
)
loader = make_dataloader(windows, batch_size=4, shuffle=True)
batch = next(iter(loader))

weather_history = batch["history"]["weather"]
weather_future = batch["future"]["weather"]
node_channels = batch["operation_static"]["node_dynamic_channels"][0]
line_channels = batch["operation_static"]["line_dynamic_channels"][0]
load_index = channel_index(node_channels, "p_load_mw")
loading_index = channel_index(line_channels, "line_loading_ratio")

load_history = batch["history"]["operation"]["node_dynamic"][:, :, load_index]
load_target = batch["future"]["operation"]["node_dynamic"][:, :, load_index]
line_loading_target = batch["future"]["operation"]["line_dynamic"][:, :, loading_index]
```

`soc_mwh` 含小时边界状态，因此 24 小时历史对应 25 个 SOC，未来 6 小时对应 7 个 SOC。

## 6. 数据划分

必须按 seed 划分，不能把同一世界的不同时段分到不同集合。正式数据集使用：

- 非质数 seed：train；
- 质数 seed：按顺序交替进入 val/test；

当前 50 个样本全部通过验证，实际数量为 train 35、val 8、test 7。
