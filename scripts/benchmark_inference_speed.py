from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    AttentionGraphForecastConfig,
    AttentionGraphLineForecast,
    ForecastModelConfig,
    GRUForecastBaseline,
    GRUForecastConfig,
    GRUGraphForecastConfig,
    GRUGraphLineForecast,
    LSTMForecastBaseline,
    LSTMForecastConfig,
    PhysicsInformedGraphTemporalModel,
    load_forecast_training_config,
    prepare_forecast_window,
    prepare_forecast_world,
)
from world_generator.dataset import SimGenrDataset, TemporalWindowDataset


PROFILE_PREFIXES = ("input.", "fusion.", "decode.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end and component inference latency.")
    parser.add_argument("--config", default="configs/models/inference_speed.yaml")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    benchmark = raw.get("benchmark", {})
    device = torch.device(str(benchmark.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but torch.cuda.is_available() is false")
    warmup = max(int(benchmark.get("warmup_iterations", 3)), 0)
    iterations = max(int(benchmark.get("measured_iterations", 10)), 1)
    module_warmup = max(int(benchmark.get("module_warmup_iterations", 20)), 0)
    module_iterations = max(int(benchmark.get("module_measured_iterations", 100)), 1)
    seed = int(benchmark.get("seed", 1))
    window_index = int(benchmark.get("window_index", 0))
    results = [
        _benchmark_model(
            item,
            device=device,
            seed=seed,
            window_index=window_index,
            warmup=warmup,
            iterations=iterations,
            module_warmup=module_warmup,
            module_iterations=module_iterations,
        )
        for item in raw.get("models", [])
    ]
    output = Path(benchmark.get("output_dir", "checkpoints/results/inference_speed"))
    output.mkdir(parents=True, exist_ok=True)
    document = {
        "benchmark_config": str(Path(args.config)),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "module_warmup_iterations": module_warmup,
        "module_measured_iterations": module_iterations,
        "results": results,
    }
    (output / "inference_speed.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = [
        {
            "model": result["model"],
            "config": result["config"],
            "weights": result["weights"],
            "section": section["section"],
            "cpu_ms_per_inference": section["cpu_ms_per_inference"],
            "device_ms_per_inference": section["device_ms_per_inference"],
            "calls_per_inference": section["calls_per_inference"],
            "end_to_end_ms_per_inference": result["end_to_end_ms_per_inference"],
        }
        for result in results
        for section in result["sections"]
    ]
    with (output / "inference_speed.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    module_rows = [
        {
            "model": result["model"],
            "module": module["module"],
            "calls_per_inference": module["calls_per_inference"],
            "unit_wall_ms": module["unit_wall_ms"],
            "unit_device_ms": module["unit_device_ms"],
            "weighted_wall_ms": module["weighted_wall_ms"],
            "weighted_device_ms": module["weighted_device_ms"],
            "module_sum_device_ms": result["module_sum_device_ms"],
            "end_to_end_ms": result["end_to_end_ms_per_inference"],
        }
        for result in results
        for module in result["module_benchmark"]
    ]
    with (output / "inference_module_speed.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(module_rows[0]) if module_rows else [])
        if module_rows:
            writer.writeheader()
            writer.writerows(module_rows)
    _print_results(results, device)
    print(f"Saved benchmark: {output}")


def _benchmark_model(
    item: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    window_index: int,
    warmup: int,
    iterations: int,
    module_warmup: int,
    module_iterations: int,
) -> dict[str, Any]:
    config_path = Path(item["config"])
    run = load_forecast_training_config(config_path)
    worlds = SimGenrDataset(
        run.dataset,
        seeds=[seed],
        as_torch=True,
        cache_size=1,
        device=device,
        transform=prepare_forecast_world,
    ).preload()
    windows = TemporalWindowDataset(
        worlds,
        history_hours=run.history_hours,
        forecast_hours=run.forecast_hours,
        stride=run.stride,
        transform=prepare_forecast_window,
        cache_size=0,
    )
    window = windows[window_index]
    model = _build_model(run.model_type, run.model_parameters, window).to(device).eval()
    checkpoint = item.get("checkpoint")
    weights = "random_initialization"
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        weights = str(checkpoint)

    with torch.inference_mode():
        for _ in range(warmup):
            model(window)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(window)
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / iterations

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=False, profile_memory=False) as measured:
            for _ in range(iterations):
                with record_function("model.total"):
                    model(window)
            _synchronize(device)

    section_events: dict[str, dict[str, float]] = {}
    for event in measured.key_averages():
        if not event.key.startswith(PROFILE_PREFIXES):
            continue
        row = section_events.setdefault(event.key, {"cpu_us": 0.0, "device_us": 0.0, "calls": 0.0})
        cpu_us = float(event.cpu_time_total)
        row["cpu_us"] = max(row["cpu_us"], cpu_us)
        row["device_us"] = max(row["device_us"], _device_time_us(event))
        if cpu_us > 0.0:
            row["calls"] = max(row["calls"], float(event.count))
    sections = [
        {
            "section": name,
            "cpu_ms_per_inference": values["cpu_us"] / 1000.0 / iterations,
            "device_ms_per_inference": values["device_us"] / 1000.0 / iterations,
            "calls_per_inference": values["calls"] / iterations,
        }
        for name, values in section_events.items()
    ]
    sections.sort(key=lambda row: row["section"])
    module_benchmark = _benchmark_model_modules(
        model,
        window,
        device=device,
        warmup=module_warmup,
        iterations=module_iterations,
    )
    return {
        "model": run.model_type,
        "config": str(config_path),
        "weights": weights,
        "seed": seed,
        "window_index": window_index,
        "node_count": int(window["graph"]["node_id"].shape[0]),
        "line_count": int(window["graph"]["edge_index"].shape[1]),
        "history_hours": run.history_hours,
        "forecast_hours": run.forecast_hours,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "end_to_end_ms_per_inference": elapsed_ms,
        "sections": sections,
        "module_sum_wall_ms": sum(row["weighted_wall_ms"] for row in module_benchmark),
        "module_sum_device_ms": sum(row["weighted_device_ms"] for row in module_benchmark),
        "module_benchmark": module_benchmark,
    }


def _build_model(model_type: str, parameters: dict[str, Any], window: dict[str, Any]) -> torch.nn.Module:
    if model_type in {"physics_gst", "physics_gst_causal"}:
        options = dict(parameters)
        if model_type == "physics_gst_causal":
            options.setdefault("temporal_backbone", "causal_attention")
        return PhysicsInformedGraphTemporalModel.from_window(window, ForecastModelConfig(**options))
    if model_type == "lstm":
        return LSTMForecastBaseline.from_window(window, LSTMForecastConfig(**parameters))
    if model_type == "gru_gnn":
        return GRUGraphLineForecast.from_window(window, GRUGraphForecastConfig(**parameters))
    if model_type == "attn_gnn":
        return AttentionGraphLineForecast.from_window(
            window,
            AttentionGraphForecastConfig(**parameters),
        )
    if model_type == "gru":
        return GRUForecastBaseline.from_window(window, GRUForecastConfig(**parameters))
    raise ValueError(f"Unsupported benchmark model.type: {model_type!r}")


def _device_time_us(event: Any) -> float:
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _benchmark_model_modules(
    model: torch.nn.Module,
    window: dict[str, Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    targets = _module_targets(model)
    captured: dict[str, tuple[Any, ...]] = {}
    calls = {name: 0 for name in targets}
    handles = []

    for name, module in targets.items():
        def capture(_module: torch.nn.Module, inputs: tuple[Any, ...], *, label: str = name) -> None:
            calls[label] += 1
            captured.setdefault(label, _detach_argument(inputs))

        handles.append(module.register_forward_pre_hook(capture))
    with torch.inference_mode():
        model(window)
    for handle in handles:
        handle.remove()
    _synchronize(device)

    rows = []
    with torch.inference_mode():
        for name, module in targets.items():
            if name not in captured or calls[name] <= 0:
                continue
            wall_ms, device_ms = _benchmark_module_call(
                module,
                captured[name],
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
            count = float(calls[name])
            rows.append(
                {
                    "module": name,
                    "calls_per_inference": count,
                    "unit_wall_ms": wall_ms,
                    "unit_device_ms": device_ms,
                    "weighted_wall_ms": wall_ms * count,
                    "weighted_device_ms": device_ms * count,
                }
            )
    rows.sort(key=lambda row: row["weighted_device_ms"], reverse=True)
    return rows


def _module_targets(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    targets: dict[str, torch.nn.Module] = {}

    def add(name: str, attribute: str) -> None:
        module = getattr(model, attribute, None)
        if isinstance(module, torch.nn.Module):
            targets[name] = module

    add("input.static_grid_encoder", "static_encoder")
    add("input.weather_grid_encoder", "weather_encoder")
    embeddings = getattr(model, "categorical_embeddings", ())
    for index, module in enumerate(embeddings):
        targets[f"input.categorical_embedding_{index}"] = module
    add("input.node_history_encoder", "node_history_encoder")
    add("input.weather_history_encoder", "weather_history_encoder")
    add("input.line_history_encoder", "line_history_encoder")
    add("input.node_type_embedding", "node_type_embedding")
    add("fusion.node_initial", "node_input")
    add("fusion.edge_initial", "edge_input")
    add("fusion.node_initial", "node_initial")
    add("fusion.edge_initial", "edge_initial")
    add("decode.future_weather_projection", "future_weather")
    graph_layers = getattr(model, "line_graph_layers", getattr(model, "graph_layers", ()))
    graph_norms = getattr(model, "line_graph_norms", getattr(model, "graph_norms", ()))
    for index, module in enumerate(graph_layers):
        targets[f"graph.gatv2_layer_{index}"] = module
    for index, module in enumerate(graph_norms):
        targets[f"graph.layer_norm_{index}"] = module
    add("decode.node_recurrence", "node_recurrence")
    add("decode.edge_recurrence", "edge_recurrence")
    add("decode.node_recurrence", "node_decoder")
    add("decode.edge_recurrence", "edge_decoder")
    add("decode.node_head", "node_head")
    add("decode.edge_head", "edge_head")
    add("decode.graph_edge_correction", "graph_edge_correction")
    return targets


def _benchmark_module_call(
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        module(*inputs)
    _synchronize(device)
    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    started = time.perf_counter()
    for _ in range(iterations):
        module(*inputs)
    if end_event is not None:
        end_event.record()
    _synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000.0 / iterations
    device_ms = float(start_event.elapsed_time(end_event)) / iterations if start_event is not None else wall_ms
    return wall_ms, device_ms


def _detach_argument(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_argument(item) for item in value)
    if isinstance(value, list):
        return [_detach_argument(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_argument(item) for key, item in value.items()}
    return value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _print_results(results: list[dict[str, Any]], device: torch.device) -> None:
    print(f"\nInference speed ({device})")
    for result in results:
        print(
            f"\n{result['model']}: {result['end_to_end_ms_per_inference']:.3f} ms/inference, "
            f"parameters={result['parameter_count']:,}"
        )
        print("section                        | cpu ms | device ms | calls")
        print("-------------------------------+--------+-----------+------")
        for section in result["sections"]:
            print(
                f"{section['section']:<30} | "
                f"{section['cpu_ms_per_inference']:>6.3f} | "
                f"{section['device_ms_per_inference']:>9.3f} | "
                f"{section['calls_per_inference']:>5.1f}"
            )
        print(
            f"module-unit weighted sum: device={result['module_sum_device_ms']:.3f} ms, "
            f"wall={result['module_sum_wall_ms']:.3f} ms"
        )
        print("module                         | unit gpu | calls | weighted gpu")
        print("-------------------------------+----------+-------+-------------")
        for module in result["module_benchmark"]:
            print(
                f"{module['module']:<30} | "
                f"{module['unit_device_ms']:>8.4f} | "
                f"{module['calls_per_inference']:>5.1f} | "
                f"{module['weighted_device_ms']:>11.3f}"
            )


if __name__ == "__main__":
    main()
