from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_ENTITIES = ("load", "genr", "line")
ENTITIES = (*BASE_ENTITIES, "average")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect checkpoint metrics into compact comparison tables.")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--output-dir", default="checkpoints/results")
    args = parser.parse_args()
    root = Path(args.checkpoints)
    output = Path(args.output_dir)
    runs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*.json")):
        if path.name == "metadata.json" or output in path.parents:
            continue
        parsed = _parse_result(path, root, json.loads(path.read_text(encoding="utf-8")))
        if parsed is None:
            continue
        run, metrics, losses = parsed
        runs.append(run)
        metric_rows.extend(metrics)
        loss_rows.extend(losses)

    metric_rows.sort(key=_sort_key)
    loss_rows.sort(key=_sort_key)
    output.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": root.as_posix(),
        "run_count": len(runs),
        "runs": runs,
        "metrics": metric_rows,
        "losses": loss_rows,
    }
    (output / "summary.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(
        output / "summary.csv",
        metric_rows,
        _base_fields() + ["entity", "rmse", "mae", "r2"],
    )
    loss_names = sorted({name for row in loss_rows for name in row if name not in _base_fields()})
    _write_csv(output / "losses.csv", loss_rows, _base_fields() + loss_names)
    print(f"Collected {len(runs)} runs, {len(metric_rows)} metric rows, and {len(loss_rows)} loss rows")
    print(f"Results: {output}")


def _parse_result(
    path: Path,
    root: Path,
    payload: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None:
    relative = path.relative_to(root)
    if isinstance(payload, dict) and "arima" in str(payload.get("model", "")):
        return _parse_arima(path, relative, payload)
    if isinstance(payload, list):
        history, metadata = payload, {}
    elif isinstance(payload, dict) and isinstance(payload.get("history"), list):
        history, metadata = payload["history"], payload
    else:
        return None
    if not history:
        return None

    model = str(metadata.get("model") or relative.parts[0])
    seed = metadata.get("initialization_seed", _seed_from_parts(relative.parts))
    history_hours, forecast_hours = _horizon(path.stem)
    training = metadata.get("training_config") or metadata.get("training_args") or {}
    history_hours = training.get("history_hours", history_hours)
    forecast_hours = training.get("forecast_hours", forecast_hours)
    checkpoint = path.with_suffix(".pt")
    common = _common(
        model,
        seed,
        path.stem,
        history_hours,
        forecast_hours,
        relative,
        checkpoint.relative_to(root) if checkpoint.exists() else None,
    )
    common["metric_scale"] = "quantile" if metadata.get("normalization_stats") else "physical_legacy"
    training_config = metadata.get("training_config", {})
    common["architecture"] = metadata.get("architecture", training_config.get("model_type", model))
    common["physics_enabled"] = metadata.get(
        "physics_enabled",
        training_config.get("physics_enabled", model.startswith("physics_gst")),
    )
    selected_epoch = metadata.get("selection", {}).get("best_epoch")
    best = next(
        (row for row in history if row.get("epoch") == selected_epoch),
        min(history, key=lambda row: _loss_value(row, "val", "total")),
    )
    final = history[-1]
    metric_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for selection, epoch_row in (("best", best), ("last", final)):
        epoch_common = {**common, "selection": selection, "epoch": epoch_row.get("epoch")}
        views = _epoch_views(epoch_row)
        metrics = _metrics_with_average(views["metrics"])
        metric_rows.extend(
            {**epoch_common, "entity": entity, **metrics.get(entity, {})}
            for entity in ENTITIES
        )
        loss_rows.append(
            {
                **epoch_common,
                **{f"train_{name}": value for name, value in views["train_losses"].items()},
                **{f"val_{name}": value for name, value in views["val_losses"].items()},
            }
        )
    test_result = metadata.get("test")
    if isinstance(test_result, dict):
        test_common = {
            **common,
            "selection": "test",
            "partition": "test",
            "epoch": metadata.get("selection", {}).get("best_epoch", best.get("epoch")),
        }
        metrics = _metrics_with_average(test_result.get("metrics", {}))
        metric_rows.extend(
            {**test_common, "entity": entity, **metrics.get(entity, {})}
            for entity in ENTITIES
        )
        loss_rows.append(
            {
                **test_common,
                **{f"test_{name}": value for name, value in test_result.get("losses", {}).items()},
            }
        )
    run = {
        **common,
        "epochs": len(history),
        "config_path": metadata.get("config_path"),
        "normalization_stats": metadata.get("normalization_stats"),
        "best_epoch": best.get("epoch"),
        "last_epoch": final.get("epoch"),
        "test": test_result,
        "history": history,
    }
    return run, metric_rows, loss_rows


def _parse_arima(
    path: Path,
    relative: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    history_hours, forecast_hours = _horizon(path.stem)
    base_common = _common(
        "arima",
        None,
        path.stem,
        payload.get("history_hours", history_hours),
        payload.get("forecast_hours", forecast_hours),
        relative,
        None,
    )
    base_common["metric_scale"] = "quantile" if payload.get("normalization_stats") else "physical_legacy"
    base_common["architecture"] = "arima"
    base_common["physics_enabled"] = False
    order = payload.get("order") or payload.get("fitted_state", {}).get("order")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, dict):
        evaluations = {
            str(payload.get("partition", "val")): {
                "windows": payload.get("windows"),
                "metrics": payload.get("metrics", {}),
                "series_diagnostics": payload.get("series_diagnostics"),
            }
        }
    metric_rows = []
    for partition, evaluation in evaluations.items():
        common = {
            **base_common,
            "partition": partition,
            "windows": evaluation.get("windows"),
            "order": _order_text(order),
        }
        source_names = {"load": "load", "genr": "generation", "line": "loading"}
        partition_metrics: dict[str, dict[str, Any]] = {}
        for entity, source in source_names.items():
            nested = evaluation.get("metrics", {}).get(entity, {})
            partition_metrics[entity] = {
                "rmse": nested.get("rmse", payload.get(f"{source}_rmse_mw", payload.get(f"{source}_rmse"))),
                "mae": nested.get("mae", payload.get(f"{source}_mae_mw", payload.get(f"{source}_mae"))),
                "r2": nested.get("r2", payload.get(f"{source}_r2")),
            }
        partition_metrics = _metrics_with_average(partition_metrics)
        metric_rows.extend(
            {
                **common,
                "selection": "evaluation",
                "epoch": None,
                "entity": entity,
                **partition_metrics[entity],
            }
            for entity in ENTITIES
        )
    run = {
        **base_common,
        "order": _order_text(order),
        "metrics": metric_rows,
        "fitted_state": payload.get("fitted_state"),
        "evaluations": evaluations,
    }
    return run, metric_rows, []


def _epoch_views(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("val"), dict):
        return {
            "metrics": row["val"].get("metrics", {}),
            "train_losses": row.get("train", {}).get("losses", {}),
            "val_losses": row["val"].get("losses", {}),
        }
    return {
        "metrics": {
            "load": {
                "rmse": row.get("val_load_rmse_mw"),
                "mae": row.get("val_load_mae_mw"),
                "r2": row.get("val_load_r2"),
            },
            "genr": {
                "rmse": row.get("val_generation_rmse_mw"),
                "mae": row.get("val_generation_mae_mw"),
                "r2": row.get("val_generation_r2"),
            },
            "line": {
                "rmse": row.get("val_loading_rmse"),
                "mae": row.get("val_loading_mae"),
                "r2": row.get("val_loading_r2"),
            },
        },
        "train_losses": {"total": row.get("train_loss")},
        "val_losses": {"total": row.get("val_loss")},
    }


def _loss_value(row: dict[str, Any], split: str, name: str) -> float:
    if isinstance(row.get(split), dict):
        value = row[split].get("losses", {}).get(name)
    else:
        value = row.get(f"{split}_loss" if name == "total" else f"{split}_loss_{name}")
    return float("inf") if value is None else float(value)


def _metrics_with_average(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {entity: dict(metrics.get(entity, {})) for entity in BASE_ENTITIES}
    existing = metrics.get("average")
    if isinstance(existing, dict):
        result["average"] = dict(existing)
        return result
    result["average"] = {}
    for metric in ("rmse", "mae", "r2"):
        values = [result[entity].get(metric) for entity in BASE_ENTITIES]
        finite = [float(value) for value in values if value is not None]
        result["average"][metric] = sum(finite) / len(finite) if finite else None
    return result


def _common(
    model: str,
    seed: Any,
    run_name: str,
    history_hours: Any,
    forecast_hours: Any,
    result_file: Path,
    checkpoint: Path | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "initialization_seed": seed,
        "run_name": run_name,
        "history_hours": history_hours,
        "forecast_hours": forecast_hours,
        "partition": "val",
        "windows": None,
        "order": None,
        "result_file": result_file.as_posix(),
        "checkpoint": checkpoint.as_posix() if checkpoint is not None else None,
    }


def _horizon(name: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)to(\d+)", name)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def _seed_from_parts(parts: tuple[str, ...]) -> int | None:
    for part in parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _order_text(order: Any) -> str | None:
    return ",".join(str(value) for value in order) if isinstance(order, (list, tuple)) else None


def _base_fields() -> list[str]:
    return [
        "model",
        "architecture",
        "physics_enabled",
        "initialization_seed",
        "run_name",
        "selection",
        "history_hours",
        "forecast_hours",
        "partition",
        "windows",
        "order",
        "epoch",
        "result_file",
        "checkpoint",
        "metric_scale",
    ]


def _sort_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
    seed = row.get("initialization_seed")
    return (
        str(row.get("model", "")),
        -1 if seed is None else int(seed),
        str(row.get("run_name", "")),
        str(row.get("selection", "")),
        str(ENTITIES.index(row["entity"])) if row.get("entity") in ENTITIES else "",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
