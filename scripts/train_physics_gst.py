from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.dataset import SimGenrDataset, TemporalWindowDataset
from models import (
    AttentionGraphForecastConfig,
    AttentionGraphLineForecast,
    ForecastTrainingConfig,
    ForecastModelConfig,
    GRUForecastBaseline,
    GRUForecastConfig,
    GRUGraphForecastConfig,
    GRUGraphLineForecast,
    LSTMForecastBaseline,
    LSTMForecastConfig,
    PhysicsInformedGraphTemporalModel,
    forecast_targets,
    fit_forecast_quantiles,
    physics_informed_forecast_loss,
    prepare_forecast_world,
    prepare_forecast_window,
    quantile_normalize,
    supervised_forecast_loss,
    load_forecast_training_config,
    serializable_quantile_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configured multimodal forecasting model.")
    parser.add_argument("--config", default="configs/models/gru_gnn_physics_24to24.yaml")
    cli = parser.parse_args()
    run = load_forecast_training_config(cli.config)
    started_at = datetime.now().astimezone()
    output, run_id = _run_artifact_paths(run, started_at)
    output.parent.mkdir(parents=True, exist_ok=bool(run.output))
    config_source = Path(cli.config)

    random.seed(run.initialization_seed)
    np.random.seed(run.initialization_seed)
    torch.manual_seed(run.initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run.initialization_seed)
    device = torch.device(run.device)
    partitions = _partition_seeds(Path(run.dataset))
    train = _windows(run, partitions["train"], device)
    val = _windows(run, partitions["val"], device)
    normalization_stats = {
        name: values.to(device)
        for name, values in fit_forecast_quantiles(train.worlds, run.normalization_parameters).items()
    }
    first = _window_on_device(train[0], device)
    if run.model_type in {"physics_gst", "physics_gst_causal"}:
        model_parameters = dict(run.model_parameters)
        if run.model_type == "physics_gst_causal":
            model_parameters.setdefault("temporal_backbone", "causal_attention")
        config = ForecastModelConfig(**model_parameters)
        model = PhysicsInformedGraphTemporalModel.from_window(first, config=config).to(device)
    elif run.model_type == "lstm":
        config = LSTMForecastConfig(**run.model_parameters)
        model = LSTMForecastBaseline.from_window(first, config=config).to(device)
    elif run.model_type == "gru_gnn":
        config = GRUGraphForecastConfig(**run.model_parameters)
        model = GRUGraphLineForecast.from_window(first, config=config).to(device)
    elif run.model_type == "attn_gnn":
        config = AttentionGraphForecastConfig(**run.model_parameters)
        model = AttentionGraphLineForecast.from_window(first, config=config).to(device)
    else:
        config = GRUForecastConfig(**run.model_parameters)
        model = GRUForecastBaseline.from_window(first, config=config).to(device)
    del first
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_total = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, run.epochs + 1):
        model.train()
        indices = list(range(len(train)))
        random.shuffle(indices)
        if run.max_train_windows > 0:
            indices = indices[: run.max_train_windows]
        running_losses: dict[str, list[float]] = {}
        for index in tqdm(indices, desc=f"Epoch {epoch} train", unit="window", dynamic_ncols=True):
            window = _window_on_device(train[index], device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(window)
            losses = _forecast_loss(run, prediction, forecast_targets(window), normalization_stats)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), run.gradient_clip_norm)
            optimizer.step()
            _append_losses(running_losses, losses)
        validation = _evaluate(model, val, device, run.max_eval_windows, run, normalization_stats)
        row = {
            "epoch": epoch,
            "train": {"losses": _mean_losses(running_losses)},
            "val": validation,
        }
        history.append(row)
        _print_epoch_tables(row, epoch, run.epochs)
        val_total = float(row["val"]["losses"][run.selection_loss])
        if val_total < best_val_total:
            best_epoch = epoch
            best_val_total = val_total
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without a valid best checkpoint")
    model.load_state_dict(best_state)
    del train, val
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_result = None
    if run.evaluate_test:
        test = _windows(run, partitions["test"], device)
        test_result = _evaluate(model, test, device, 0, run, normalization_stats)
        _print_test_tables(test_result, best_epoch)

    experiment_name = run.experiment_name or run.model_type
    record = {
        "model": experiment_name,
        "architecture": run.model_type,
        "physics_enabled": run.physics_enabled,
        "initialization_seed": int(run.initialization_seed),
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "model_config": config.__dict__,
        "training_config": run.to_dict(),
        "config_path": str(config_source),
        "normalization_stats": serializable_quantile_stats(normalization_stats),
        "selection": {
            "criterion": f"minimum val {run.selection_loss} loss",
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val_total,
        },
        "test": test_result,
        "history": history,
    }
    torch.save({"model_state": model.state_dict(), **record}, output)
    output.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {output}")


def _run_artifact_paths(
    run: ForecastTrainingConfig,
    started_at: datetime,
) -> tuple[Path, str]:
    run_id = started_at.strftime("%Y%m%d_%H%M")
    if run.output:
        output = Path(run.output)
    else:
        horizon = f"{run.history_hours}to{run.forecast_hours}"
        output = (
            Path("checkpoints")
            / (run.experiment_name or run.model_type)
            / f"seed_{run.initialization_seed}"
            / run_id
            / f"{horizon}.pt"
        )
    return output, run_id


def _windows(run: ForecastTrainingConfig, seeds: list[int], device: torch.device) -> TemporalWindowDataset:
    preload = bool(run.preload_to_device)
    worlds = SimGenrDataset(
        run.dataset,
        seeds=seeds,
        as_torch=True,
        cache_size=len(seeds) if preload else 2,
        device=device if preload else None,
        transform=prepare_forecast_world if preload else None,
    )
    if preload:
        worlds.preload()
        tensor_bytes = worlds.cached_tensor_bytes()
        print(
            f"Preloaded {len(seeds)} worlds to {device}: "
            f"{tensor_bytes / 1024**3:.2f} GiB tensor storage"
        )
    windows = TemporalWindowDataset(
        worlds,
        history_hours=run.history_hours,
        forecast_hours=run.forecast_hours,
        stride=run.stride,
        transform=prepare_forecast_window if preload else None,
        cache_size=0,
    )
    print(f"Indexed {len(windows)} temporal windows without caching window tensors")
    return windows


def _partition_seeds(root: Path) -> dict[str, list[int]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = {"train": [], "val": [], "test": []}
    for sample in manifest["samples"]:
        result[str(sample["partition"])].append(int(sample["seed"]))
    return result


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    dataset: TemporalWindowDataset,
    device: torch.device,
    limit: int,
    run: ForecastTrainingConfig,
    normalization_stats: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    count = len(dataset) if limit <= 0 else min(len(dataset), limit)
    loss_values: dict[str, list[float]] = {}
    metric_values = {
        "load": {"rmse": [], "mae": [], "r2": []},
        "genr": {"rmse": [], "mae": [], "r2": []},
        "line": {"rmse": [], "mae": [], "r2": []},
    }
    for index in range(count):
        window = _window_on_device(dataset[index], device)
        prediction = model(window)
        target = forecast_targets(window)
        _append_losses(loss_values, _forecast_loss(run, prediction, target, normalization_stats))
        _append_metrics(
            metric_values["load"],
            prediction["p_load_mw"],
            target["p_load_mw"],
            normalization_stats.get("load"),
            run.normalization_parameters,
        )
        _append_metrics(
            metric_values["genr"],
            prediction["p_generation_mw"],
            target["p_generation_mw"],
            normalization_stats.get("genr"),
            run.normalization_parameters,
        )
        _append_metrics(
            metric_values["line"],
            prediction["line_loading_ratio"],
            target["line_loading_ratio"],
            normalization_stats.get("line_loading"),
            run.normalization_parameters,
        )
    metrics = {
        entity: {metric: float(np.mean(values)) for metric, values in values.items()}
        for entity, values in metric_values.items()
    }
    metrics["average"] = _average_metrics(metrics)
    return {"losses": _mean_losses(loss_values), "metrics": metrics}


def _forecast_loss(
    run: ForecastTrainingConfig,
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    normalization_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalization = {
        "normalization_stats": normalization_stats,
        "normalization_parameters": run.normalization_parameters,
    }
    if run.physics_enabled:
        return physics_informed_forecast_loss(prediction, target, **run.loss_parameters, **normalization)
    return supervised_forecast_loss(prediction, target, **run.loss_parameters, **normalization)


def _append_losses(storage: dict[str, list[float]], losses: dict[str, torch.Tensor]) -> None:
    for name, value in losses.items():
        storage.setdefault(name, []).append(float(value.detach().cpu()))


def _mean_losses(storage: dict[str, list[float]]) -> dict[str, float]:
    return {name: float(np.mean(values)) for name, values in storage.items()}


def _append_metrics(
    storage: dict[str, list[float]],
    prediction: torch.Tensor,
    target: torch.Tensor,
    statistics: torch.Tensor | None,
    parameters: dict[str, Any],
) -> None:
    if statistics is not None:
        prediction = quantile_normalize(prediction, statistics, parameters)
        target = quantile_normalize(target, statistics, parameters)
    difference = prediction - target
    storage["rmse"].append(float(torch.mean(difference**2).sqrt().cpu()))
    storage["mae"].append(float(torch.mean(torch.abs(difference)).cpu()))
    storage["r2"].append(_r2_score(prediction, target))


def _print_epoch_tables(row: dict[str, Any], epoch: int, epochs: int) -> None:
    print(f"\nEpoch {epoch}/{epochs}")
    metrics = dict(row["val"]["metrics"])
    metrics.setdefault("average", _average_metrics(metrics))
    _print_table(
        "Validation metrics (train-quantile scale)",
        ("entity", "rmse", "mae", "r2"),
        [
            (entity, metrics[entity])
            for entity in ("load", "genr", "line", "average")
        ],
    )
    train_losses = row["train"]["losses"]
    val_losses = row["val"]["losses"]
    _print_table(
        "Prediction losses",
        ("split", "total", "supervised", "load", "genr", "line_flow", "line_loading"),
        [("train", train_losses), ("val", val_losses)],
    )
    physical_columns = {
        "balance": "power_balance",
        "genr_ramp": "genr_ramp",
        "load_nonneg": "load_nonnegative",
        "genr_nonneg": "genr_nonnegative",
        "line_capacity": "line_capacity",
        "loading_consistency": "line_loading_consistency",
    }
    _print_table(
        "Physical constraint losses",
        ("split", *physical_columns),
        [
            (split, {label: losses.get(source) for label, source in physical_columns.items()})
            for split, losses in (("train", train_losses), ("val", val_losses))
        ],
    )


def _print_test_tables(result: dict[str, Any], best_epoch: int) -> None:
    print(f"\nBest epoch: {best_epoch}")
    metrics = dict(result["metrics"])
    metrics.setdefault("average", _average_metrics(metrics))
    _print_table(
        "Test metrics (train-quantile scale)",
        ("entity", "rmse", "mae", "r2"),
        [(entity, metrics[entity]) for entity in ("load", "genr", "line", "average")],
    )
    _print_table(
        "Test losses",
        ("split", "total", "supervised", "load", "genr", "line_flow", "line_loading"),
        [("test", result["losses"])],
    )


def _print_table(
    title: str,
    columns: tuple[str, ...],
    rows: list[tuple[str, dict[str, Any]]],
) -> None:
    rendered: list[list[str]] = []
    for label, values in rows:
        rendered.append(
            [label]
            + [_format_table_value(values.get(column)) for column in columns[1:]]
        )
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rendered))
        for index in range(len(columns))
    ]
    separator = "-+-".join("-" * width for width in widths)
    print(f"\n{title}")
    print(" | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)))
    print(separator)
    for values in rendered:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(values)))


def _format_table_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = torch.sum((prediction - target) ** 2)
    centered = target - torch.mean(target)
    total = torch.sum(centered**2)
    if float(total) <= 1e-12:
        return 1.0 if float(residual) <= 1e-12 else 0.0
    return float((1.0 - residual / total).cpu())


def _average_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    entities = ("load", "genr", "line")
    return {
        metric: float(np.mean([metrics[entity][metric] for entity in entities]))
        for metric in ("rmse", "mae", "r2")
    }


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _window_on_device(window: dict[str, Any], device: torch.device) -> dict[str, Any]:
    prepared = window.get("prepared")
    if prepared is not None and prepared["edge_index"].device == device:
        return window
    return _to_device(window, device)


if __name__ == "__main__":
    main()
