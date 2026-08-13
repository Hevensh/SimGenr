# Physics-informed graph spatiotemporal forecaster

For the complete Chinese architecture and configuration reference, see
[`MODEL_STRUCTURE_AND_CONFIG.md`](MODEL_STRUCTURE_AND_CONFIG.md).

`PhysicsInformedGraphTemporalModel` uses a configurable history and known future weather horizon to forecast:

- node load in MW;
- node scheduled generation in MW;
- signed branch flow in MW;
- branch loading derived as `abs(flow) / rate_mva`.

The graph contains only buses and lines. A* route cells are not model structure.

## Model path

1. Continuous and categorical static grids are encoded by CNNs and sampled at bus locations.
2. Historical weather, node operation, and line operation are encoded by GRUs.
3. Future weather is sampled at each bus and injected at every forecast step.
4. Edge-aware GATv2 layers exchange information only along observed lines.
5. Node and line recurrent decoders produce the configured forecast horizon.

The loss combines supervised node/line errors with nodal active-power balance and generation-ramp penalties. Signed line flow is decoded first, so loading ratio is physically consistent with the planned line rating.

## Training

Activate the environment in Bash before running model commands:

```bash
source C:/ProgramData/miniconda3/etc/profile.d/conda.sh
conda activate myEnv
```

Run the complete train/validation split:

```bash
python scripts/train_physics_gst.py \
  --config configs/models/gru_gnn_physics_24to24.yaml
```

Model architecture, loss weights, temporal windows, optimizer, runtime, initialization seed, and output path are
defined in the YAML file rather than as separate command-line flags. For a smoke test, copy a model config and set
`training.epochs`, `training.max_train_windows`, and `training.max_eval_windows` there.

`model.type` controls the architecture, while `loss.physics_enabled` independently controls whether the physical
loss terms are included. Run the four architectures both with and without physical constraints using:

```bash
bash scripts/train_eight_variants.sh
```

The script trains matched LSTM, GRU, GRU-GNN, and Attention-GNN pairs in sequence. Each pair keeps identical model
parameters and differs only in its physical-loss configuration. `experiment.name` separates their checkpoint roots.

The current `gru_gnn` is a controlled extension of the GRU baseline: load and generation retain the unchanged local
GRU decoder, while gated GATv2 context provides only a residual correction to the line-flow head. The correction head
is initialized to zero, so training starts from the exact GRU prediction. Graph messages are never written back into
the node recurrent state, preventing repeated topology mixing from degrading node forecasts across the 24-hour horizon.

Rerun only the matched GRU-GNN pair after graph-branch changes with:

```bash
bash scripts/train_gru_gnn_pair.sh
```

The follow-up line-focused parameter experiment is isolated from the current best checkpoints and runs with:

```bash
bash scripts/train_gru_gnn_tuned_pair.sh
```

It uses a less closed initial graph gate, doubles the line-loading loss weight, trains for 30 epochs, and weakens the
balance/ramp penalties in the physics variant. Results are written under `gru_gnn_tuned_supervised` and
`gru_gnn_tuned_physics` experiment roots.

By default, training loads every selected world and its world-level graph tensors onto the target device. Temporal
windows are indexed but not cached on CUDA; weather normalization, history slices, node-sampled weather, and targets
are prepared on demand and released after each step. This keeps the 24-to-24 train/validation storage near the
world-level footprint instead of retaining every derived window.

Supervised load, genr, line-flow, and line-loading losses use train-only 5th/95th percentile normalization with the
configured clipping bounds. Validation RMSE, MAE, and R2 use the same normalized scale so load, genr, and line rows
are comparable. Power balance and other physical constraints remain in physical units with explicit scale factors.

Each epoch prints three console tables rather than a JSON dictionary. Validation metrics use `load/genr/line` rows;
prediction losses and physical constraint losses use `train/val` rows with one loss component per column. The
checkpoint JSON retains the nested machine-readable representation.

Checkpoints are grouped by experiment and initialization seed, for example:

```text
checkpoints/physics_gst/seed_2026/20260721_1530/24to24.pt
checkpoints/physics_gst/seed_2026/20260721_1530/24to24.json
checkpoints/lstm/seed_2026/20260721_1614/24to24.pt
checkpoints/arima/order_2_0_1/24to24_val_test.json
```

Each neural training run creates a minute-resolution local-time directory below its initialization seed. The checkpoint JSON records
the run ID, start time, original config path, resolved model config, and complete resolved training config. An explicit `output.path`
remains authoritative.
An automatically generated run directory must not already exist, preventing a second run in the same minute from silently
overwriting the first one.

All four neural architectures select the checkpoint with the minimum validation loss configured by
`evaluation.selection_loss` (normally `total`). After training, that state is restored and evaluated once on the
test seeds. Test data never participate in model selection.

The current training loop processes one variable-size world graph at a time. A later optimization can batch
disconnected graphs with PyG without changing model semantics.

## Baselines

Train the shared per-node/per-line LSTM without graph propagation:

```bash
python scripts/train_physics_gst.py \
  --config configs/models/lstm_24to24.yaml
```

Train the causal-attention Physics-GST and matching GRU baseline:

```bash
python scripts/train_physics_gst.py \
  --config configs/models/attn_gnn_physics_24to24.yaml

python scripts/train_physics_gst.py \
  --config configs/models/gru_24to24.yaml
```

The current Attention-GNN differs from GRU-GNN only in the node/line history encoders: causal self-attention replaces
the GRUs, while the local recurrent forecast decoder and zero-initialized gated line-graph correction stay unchanged.
Rerun only its matched pair with `bash scripts/train_attn_gnn_pair.sh`.

Evaluate the shared `ARIMA(2,0,1)` baseline:

```bash
python scripts/evaluate_arima_baseline.py \
  --config configs/models/arima_24to24.yaml
```

ARIMA dynamics are fitted only from train-world load, genr, and line series. The fitted parameter vectors are then
held fixed while val/test histories initialize the filter state and produce forecasts. Both val and test metrics are
stored in one result file. `model.max_train_series_per_entity` limits train fitting cost, while
`evaluation.max_windows_per_partition: 0` evaluates every available window.

Evaluation records RMSE, MAE, and R2 for load, scheduled generation, and line loading.
It also records their macro-average as an `average` row.

## Inference benchmark

Profile input encoders, fusion/graph attention, recurrent decoding, and output heads separately:

```bash
python scripts/benchmark_inference_speed.py \
  --config configs/models/inference_speed.yaml
```

The benchmark also writes `inference_module_speed.csv`, which times each captured module independently and multiplies
its unit CUDA latency by the number of calls observed in one real forward pass.

## Result summary

Collect legacy and current checkpoint results without rerunning any model:

```bash
python scripts/organize_model_results.py
```

The script writes `checkpoints/results/summary.json`, `summary.csv`, and `losses.csv`. `summary.csv` contains three
rows per validation/test result (`load`, `genr`, and `line`) with `rmse`, `mae`, and `r2` columns. `losses.csv` keeps
supervised and physical loss components in separate columns. Metrics absent from legacy results remain empty.
