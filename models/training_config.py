from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ForecastTrainingConfig:
    model_type: str
    experiment_name: str = ""
    physics_enabled: bool = False
    model_parameters: dict[str, Any] = field(default_factory=dict)
    loss_parameters: dict[str, float] = field(default_factory=dict)
    normalization_parameters: dict[str, Any] = field(default_factory=dict)
    dataset: str = "datasets/seed_1_50"
    history_hours: int = 24
    forecast_hours: int = 24
    stride: int = 24
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    max_train_windows: int = 0
    max_eval_windows: int = 0
    selection_loss: str = "total"
    evaluate_test: bool = True
    device: str = "cuda"
    preload_to_device: bool = True
    initialization_seed: int = 2026
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_forecast_training_config(path: str | Path) -> ForecastTrainingConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = _mapping(raw, "experiment")
    model = _mapping(raw, "model")
    loss = _mapping(raw, "loss")
    data = _mapping(raw, "data")
    training = _mapping(raw, "training")
    optimizer = _mapping(raw, "optimizer")
    runtime = _mapping(raw, "runtime")
    output = _mapping(raw, "output")
    evaluation = _mapping(raw, "evaluation")
    model_type = str(model.get("type", ""))
    if model_type not in {
        "physics_gst",
        "physics_gst_causal",
        "lstm",
        "gru",
        "gru_gnn",
        "attn_gnn",
    }:
        raise ValueError(f"Unsupported model.type: {model_type!r}")
    physics_enabled = bool(loss.get("physics_enabled", model_type.startswith("physics_gst")))
    return ForecastTrainingConfig(
        model_type=model_type,
        experiment_name=str(experiment.get("name") or model_type),
        physics_enabled=physics_enabled,
        model_parameters=dict(model.get("parameters", {})),
        loss_parameters={
            str(key): float(value)
            for key, value in loss.items()
            if key != "physics_enabled"
        },
        normalization_parameters=dict(_mapping(raw, "normalization")),
        dataset=str(data.get("dataset", "datasets/seed_1_50")),
        history_hours=int(data.get("history_hours", 24)),
        forecast_hours=int(data.get("forecast_hours", 24)),
        stride=int(data.get("stride", 24)),
        epochs=int(training.get("epochs", 5)),
        learning_rate=float(optimizer.get("learning_rate", 3e-4)),
        weight_decay=float(optimizer.get("weight_decay", 1e-4)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        max_train_windows=int(training.get("max_train_windows", 0)),
        max_eval_windows=int(training.get("max_eval_windows", 0)),
        selection_loss=str(evaluation.get("selection_loss", "total")),
        evaluate_test=bool(evaluation.get("evaluate_test", True)),
        device=str(runtime.get("device", "cuda")),
        preload_to_device=bool(runtime.get("preload_to_device", True)),
        initialization_seed=int(runtime.get("initialization_seed", 2026)),
        output=None if output.get("path") in {None, ""} else str(output["path"]),
    )


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section {key!r} must be a mapping")
    return value
