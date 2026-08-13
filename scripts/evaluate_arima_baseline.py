from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.dataset import SimGenrDataset, TemporalWindowDataset
from models import (
    arima_forecast_window,
    fit_arima_baseline,
    fit_forecast_quantiles,
    forecast_targets,
    prepare_forecast_world,
    serializable_quantile_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate independent per-entity ARIMA forecasts.")
    parser.add_argument("--config", default="configs/models/arima_24to24.yaml")
    cli = parser.parse_args()
    raw = yaml.safe_load(Path(cli.config).read_text(encoding="utf-8"))
    model_config = raw["model"]
    data_config = raw["data"]
    evaluation_config = raw.get("evaluation", {})
    normalization_config = raw.get("normalization", {})
    output_config = raw.get("output", {})
    dataset = str(data_config["dataset"])
    history_hours = int(data_config["history_hours"])
    forecast_hours = int(data_config["forecast_hours"])
    stride = int(data_config["stride"])
    order = tuple(int(value) for value in model_config.get("order", (2, 0, 1)))
    max_train_series = int(model_config.get("max_train_series_per_entity", 128))
    partitions = tuple(str(value) for value in evaluation_config.get("partitions", ("val", "test")))
    max_windows = int(evaluation_config.get("max_windows_per_partition", 0))

    train_seeds = _partition_seeds(Path(dataset), "train")
    training_worlds = SimGenrDataset(
        dataset,
        seeds=train_seeds,
        as_torch=True,
        cache_size=len(train_seeds),
        transform=prepare_forecast_world,
    ).preload()
    normalization_stats = fit_forecast_quantiles(training_worlds, normalization_config)
    fitted_state = fit_arima_baseline(
        training_worlds,
        order=order,
        max_series_per_entity=max_train_series,
    )
    evaluations = {
        partition: _evaluate_partition(
            dataset,
            partition,
            history_hours,
            forecast_hours,
            stride,
            max_windows,
            fitted_state,
            normalization_stats,
            normalization_config,
        )
        for partition in partitions
    }
    result = {
        "model": "train_fitted_shared_arima",
        "initialization_seed": None,
        "deterministic": True,
        "history_hours": history_hours,
        "forecast_hours": forecast_hours,
        "stride": stride,
        "normalization_stats": serializable_quantile_stats(normalization_stats),
        "fitted_state": fitted_state,
        "evaluations": evaluations,
    }
    order_name = "_".join(str(value) for value in order)
    partition_name = "_".join(partitions)
    output = Path(
        output_config.get("path")
        or f"checkpoints/arima/order_{order_name}/{history_hours}to{forecast_hours}_{partition_name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_evaluations(evaluations)
    print(f"Saved result: {output}")


def _evaluate_partition(
    dataset: str,
    partition: str,
    history_hours: int,
    forecast_hours: int,
    stride: int,
    max_windows: int,
    fitted_state: dict[str, object],
    normalization_stats: dict[str, object],
    normalization_config: dict[str, object],
) -> dict[str, object]:
    seeds = _partition_seeds(Path(dataset), partition)
    worlds = SimGenrDataset(dataset, seeds=seeds, as_torch=True, cache_size=1)
    windows = TemporalWindowDataset(
        worlds,
        history_hours=history_hours,
        forecast_hours=forecast_hours,
        stride=stride,
    )
    count = len(windows) if max_windows <= 0 else min(len(windows), max_windows)
    totals = {
        "load_squared_error": 0.0,
        "load_count": 0,
        "generation_squared_error": 0.0,
        "generation_count": 0,
        "loading_squared_error": 0.0,
        "loading_count": 0,
        "filtered": 0,
        "constant": 0,
        "fallback": 0,
    }
    started = time.perf_counter()
    for index in tqdm(range(count), desc=f"ARIMA {partition}", unit="window", dynamic_ncols=True):
        window = windows[index]
        prediction, diagnostics = arima_forecast_window(window, fitted_state)
        target = {name: _as_numpy(values) for name, values in forecast_targets(window).items()}
        _accumulate(
            totals,
            "load",
            prediction["p_load_mw"],
            target["p_load_mw"],
            normalization_stats.get("load"),
            normalization_config,
        )
        _accumulate(
            totals,
            "generation",
            prediction["p_generation_mw"],
            target["p_generation_mw"],
            normalization_stats.get("genr"),
            normalization_config,
        )
        _accumulate(
            totals,
            "loading",
            prediction["line_loading_ratio"],
            target["line_loading_ratio"],
            normalization_stats.get("line_loading"),
            normalization_config,
        )
        for name, value in diagnostics.items():
            totals[name] += int(value)
    elapsed = time.perf_counter() - started
    metrics = {
        "load": {
            "rmse": _rmse(totals, "load"),
            "mae": _mae(totals, "load"),
            "r2": _r2(totals, "load"),
        },
        "genr": {
            "rmse": _rmse(totals, "generation"),
            "mae": _mae(totals, "generation"),
            "r2": _r2(totals, "generation"),
        },
        "line": {
            "rmse": _rmse(totals, "loading"),
            "mae": _mae(totals, "loading"),
            "r2": _r2(totals, "loading"),
        },
    }
    metrics["average"] = {
        metric: float(np.mean([metrics[entity][metric] for entity in ("load", "genr", "line")]))
        for metric in ("rmse", "mae", "r2")
    }
    return {
        "partition": partition,
        "windows": count,
        "elapsed_seconds": elapsed,
        "seconds_per_window": elapsed / max(count, 1),
        "metrics": metrics,
        "series_diagnostics": {name: int(totals[name]) for name in ("filtered", "constant", "fallback")},
    }


def _print_evaluations(evaluations: dict[str, dict[str, object]]) -> None:
    rows = []
    for partition, evaluation in evaluations.items():
        metrics = evaluation["metrics"]
        for entity in ("load", "genr", "line", "average"):
            values = metrics[entity]
            rows.append((partition, entity, values["rmse"], values["mae"], values["r2"]))
    headers = ("split", "entity", "rmse", "mae", "r2")
    rendered = [[str(row[0]), str(row[1]), *(f"{float(value):.6f}" for value in row[2:])] for row in rows]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rendered)) for index in range(len(headers))]
    print("\nARIMA evaluation (train-quantile scale)")
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rendered:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _partition_seeds(root: Path, partition: str) -> list[int]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return [int(sample["seed"]) for sample in manifest["samples"] if sample["partition"] == partition]


def _accumulate(
    totals: dict[str, float | int],
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    statistics: object | None = None,
    normalization: dict[str, object] | None = None,
) -> None:
    if statistics is not None:
        values = np.asarray(statistics.detach().cpu(), dtype=np.float64)  # type: ignore[union-attr]
        lower, upper = float(values[0]), float(values[1])
        scale = max(upper - lower, 1e-6)
        parameters = normalization or {}
        prediction = np.clip(
            (np.asarray(prediction) - lower) / scale,
            -float(parameters.get("left_clip", 1)),
            1.0 + float(parameters.get("right_clip", 1)),
        )
        target = np.clip(
            (np.asarray(target) - lower) / scale,
            -float(parameters.get("left_clip", 1)),
            1.0 + float(parameters.get("right_clip", 1)),
        )
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    totals[f"{name}_squared_error"] += float(np.sum(difference**2))
    totals[f"{name}_absolute_error"] = float(totals.get(f"{name}_absolute_error", 0.0)) + float(
        np.sum(np.abs(difference))
    )
    target64 = np.asarray(target, dtype=np.float64)
    totals[f"{name}_target_sum"] = float(totals.get(f"{name}_target_sum", 0.0)) + float(np.sum(target64))
    totals[f"{name}_target_squared_sum"] = float(totals.get(f"{name}_target_squared_sum", 0.0)) + float(
        np.sum(target64**2)
    )
    totals[f"{name}_count"] += int(difference.size)


def _rmse(totals: dict[str, float | int], name: str) -> float:
    return float(np.sqrt(float(totals[f"{name}_squared_error"]) / max(int(totals[f"{name}_count"]), 1)))


def _mae(totals: dict[str, float | int], name: str) -> float:
    return float(totals[f"{name}_absolute_error"]) / max(int(totals[f"{name}_count"]), 1)


def _r2(totals: dict[str, float | int], name: str) -> float:
    count = max(int(totals[f"{name}_count"]), 1)
    target_sum = float(totals[f"{name}_target_sum"])
    total_variance = float(totals[f"{name}_target_squared_sum"]) - target_sum**2 / count
    if total_variance <= 1e-12:
        return 1.0 if float(totals[f"{name}_squared_error"]) <= 1e-12 else 0.0
    return 1.0 - float(totals[f"{name}_squared_error"]) / total_variance


def _as_numpy(values: object) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(values)


if __name__ == "__main__":
    main()
