from __future__ import annotations

from typing import Any

import torch


def fit_forecast_quantiles(
    worlds: Any,
    parameters: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Fit load/genr/line normalization statistics from training worlds only."""
    if str(parameters.get("method", "none")).lower() != "quantile":
        return {}
    max_samples = max(int(parameters.get("max_samples", 500_000)), 1)
    samples: dict[str, list[torch.Tensor]] = {
        "load": [],
        "genr": [],
        "line_flow": [],
        "line_loading": [],
    }
    per_world = max(max_samples // max(len(worlds), 1), 1)
    for index in range(len(worlds)):
        world = worlds[index]
        prepared = world.get("prepared_static")
        if prepared is None:
            prepared = prepare_forecast_world(world)["prepared_static"]
        operation = world["operation"]
        node_channels = tuple(str(value) for value in operation["node_dynamic_channels"])
        line_channels = tuple(str(value) for value in operation["line_dynamic_channels"])
        node = operation["node_dynamic"].float()
        line = operation["line_dynamic"].float()
        samples["load"].append(
            _finite_sample(node[:, node_channels.index("p_load_mw"), prepared["load_mask"].bool()], per_world)
        )
        samples["genr"].append(
            _finite_sample(
                node[:, node_channels.index("p_gen_scheduled_mw"), prepared["generation_mask"].bool()],
                per_world,
            )
        )
        samples["line_flow"].append(
            _finite_sample(line[:, line_channels.index("line_flow_mw")], per_world)
        )
        samples["line_loading"].append(
            _finite_sample(line[:, line_channels.index("line_loading_ratio")], per_world)
        )

    quantiles = torch.tensor(
        [float(parameters.get("left_quantile", 5)) / 100.0, float(parameters.get("right_quantile", 95)) / 100.0],
        device=samples["load"][0].device,
        dtype=torch.float32,
    )
    result: dict[str, torch.Tensor] = {}
    for name, chunks in samples.items():
        values = _finite_sample(torch.cat(chunks), max_samples)
        if values.numel() == 0:
            raise ValueError(f"No finite training values available for quantile normalization: {name}")
        result[name] = torch.quantile(values.float(), quantiles)
    return result


def quantile_normalize(
    values: torch.Tensor,
    statistics: torch.Tensor,
    parameters: dict[str, Any],
) -> torch.Tensor:
    lower, upper = statistics[0], statistics[1]
    normalized = (values - lower) / (upper - lower).clamp_min(1e-6)
    return normalized.clamp(
        min=-float(parameters.get("left_clip", 1)),
        max=1.0 + float(parameters.get("right_clip", 1)),
    )


def serializable_quantile_stats(statistics: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {
        name: {"left": float(values[0].detach().cpu()), "right": float(values[1].detach().cpu())}
        for name, values in statistics.items()
    }


def _finite_sample(values: torch.Tensor, max_samples: int) -> torch.Tensor:
    flattened = values.reshape(-1)
    flattened = flattened[torch.isfinite(flattened)]
    if flattened.numel() <= max_samples:
        return flattened
    indices = torch.linspace(
        0,
        flattened.numel() - 1,
        steps=max_samples,
        device=flattened.device,
    ).long()
    return flattened[indices]


def prepare_forecast_world(world: dict[str, Any]) -> dict[str, Any]:
    """Prepare graph and static raster tensors once per generated world."""
    graph = world["graph"]
    node_features = graph["node_features"].float()
    node_electrical = graph["node_electrical"].float()
    node_type = graph["node_type"].long()
    edge_index = graph["edge_index"].long()
    edge_features = graph["edge_features"].float()
    node_ids = graph["node_id"].long()
    rows = node_features[:, 0].long().clamp(0, world["static"]["continuous"].shape[-2] - 1)
    cols = node_features[:, 1].long().clamp(0, world["static"]["continuous"].shape[-1] - 1)
    node_scale = node_power_scale(node_features, node_electrical)
    rate_mva = edge_features[:, 6].clamp_min(1.0)
    storage_mask = storage_bus_mask(node_ids, world["operation"].get("site_bus_ids"))
    world["prepared_static"] = {
        "static_continuous": channel_standardize(world["static"]["continuous"].float()),
        "static_categorical": world["static"]["categorical"].long(),
        "node_features": node_features,
        "node_electrical": node_electrical,
        "node_type": node_type,
        "edge_index": edge_index,
        "bidirectional_edge_index": torch.cat((edge_index, edge_index.flip(0)), dim=1),
        "edge_features": edge_features,
        "rows": rows,
        "cols": cols,
        "node_scale_mw": node_scale,
        "line_rate_mva": rate_mva,
        "normalized_node_static": normalize_node_static(node_features, node_electrical),
        "normalized_edge_static": normalize_edge_static(edge_features),
        "storage_mask": storage_mask,
        "load_mask": (node_type == 0).float(),
        "generation_mask": ((node_type >= 1) & (node_type <= 3)).float().maximum(storage_mask),
    }
    return world


def prepare_forecast_window(window: dict[str, Any]) -> dict[str, Any]:
    """Attach parameter-independent CUDA-ready tensors to a temporal window."""
    static_prepared = window.get("prepared_static")
    if static_prepared is None:
        temporary_world = {
            "static": window["static"],
            "graph": window["graph"],
            "operation": window["operation_static"],
        }
        static_prepared = prepare_forecast_world(temporary_world)["prepared_static"]

    history = window["history"]
    future = window["future"]
    history_weather, future_weather = normalize_weather(history["weather"], future["weather"])
    node_history = normalize_node_history(
        history["operation"]["node_dynamic"], static_prepared["node_scale_mw"]
    )
    line_history = normalize_line_history(
        history["operation"]["line_dynamic"], static_prepared["line_rate_mva"]
    )
    rows = static_prepared["rows"]
    cols = static_prepared["cols"]
    window["prepared"] = {
        **static_prepared,
        "history_weather": history_weather,
        "future_weather": future_weather,
        "future_weather_at_nodes": future_weather[:, :, rows, cols].permute(0, 2, 1),
        "normalized_node_history": node_history,
        "normalized_line_history": line_history,
        "targets": forecast_target_tensors(window),
    }
    return window


def forecast_target_tensors(window: dict[str, Any]) -> dict[str, torch.Tensor]:
    node_channels = tuple(str(value) for value in window["operation_static"]["node_dynamic_channels"])
    line_channels = tuple(str(value) for value in window["operation_static"]["line_dynamic_channels"])
    node = window["future"]["operation"]["node_dynamic"].float()
    line = window["future"]["operation"]["line_dynamic"].float()
    return {
        "p_load_mw": node[:, node_channels.index("p_load_mw")],
        "p_generation_mw": node[:, node_channels.index("p_gen_scheduled_mw")],
        "line_flow_mw": line[:, line_channels.index("line_flow_mw")],
        "line_loading_ratio": line[:, line_channels.index("line_loading_ratio")],
    }


def normalize_weather(history: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = history.float().mean(dim=(0, 2, 3), keepdim=True)
    std = history.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
    return (history.float() - mean) / std, (future.float() - mean) / std


def channel_standardize(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=(1, 2), keepdim=True)
    std = values.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
    return (values - mean) / std


def node_power_scale(node_features: torch.Tensor, node_electrical: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            node_features[:, 4].abs(),
            node_electrical[:, 1].abs(),
            node_electrical[:, 3].abs(),
            torch.full_like(node_features[:, 4], 10.0),
        ),
        dim=0,
    ).amax(dim=0)


def normalize_node_history(values: torch.Tensor, node_scale: torch.Tensor) -> torch.Tensor:
    result = values.float().clone()
    result[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]] /= node_scale[None, None, :]
    return result.clamp(-5.0, 5.0)


def normalize_line_history(values: torch.Tensor, rate_mva: torch.Tensor) -> torch.Tensor:
    result = values.float().clone()
    result[:, 0] /= rate_mva[None, :]
    return result.clamp(-5.0, 5.0)


def normalize_node_static(node_features: torch.Tensor, node_electrical: torch.Tensor) -> torch.Tensor:
    feature_scale = node_features.new_tensor((63.0, 63.0, 1.0, 1.0, 300.0))
    electrical_scale = node_electrical.new_tensor((220.0, 300.0, 300.0, 200.0, 1.0, 1.1))
    return torch.cat((node_features / feature_scale, node_electrical / electrical_scale), dim=-1).clamp(-5.0, 5.0)


def normalize_edge_static(edge_features: torch.Tensor) -> torch.Tensor:
    scale = edge_features.new_tensor((64.0, 220.0, 64.0, 20.0, 50.0, 100.0, 500.0))
    return (edge_features / scale).clamp(-5.0, 5.0)


def storage_bus_mask(node_ids: torch.Tensor, site_bus_ids: torch.Tensor | None) -> torch.Tensor:
    if site_bus_ids is None or site_bus_ids.numel() == 0:
        return torch.zeros(node_ids.shape[0], device=node_ids.device, dtype=torch.float32)
    return (node_ids[:, None] == site_bus_ids.long()[None, :]).any(dim=1).float()
