from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.profiler import record_function
from torch_geometric.nn import GATv2Conv

from .preprocessing import (
    channel_standardize as _channel_standardize,
    forecast_target_tensors,
    node_power_scale as _node_power_scale,
    normalize_edge_static as _normalize_edge_static,
    normalize_line_history as _normalize_line_history,
    normalize_node_history as _normalize_node_history,
    normalize_node_static as _normalize_node_static,
    normalize_weather,
    quantile_normalize,
    storage_bus_mask as _storage_bus_mask,
)
from .temporal import CausalTemporalEncoder


@dataclass(frozen=True)
class ForecastModelConfig:
    hidden_dim: int = 96
    grid_dim: int = 32
    categorical_embedding_dim: int = 4
    attention_heads: int = 4
    graph_layers: int = 2
    dropout: float = 0.10
    temporal_backbone: str = "gru"
    temporal_attention_heads: int = 4
    temporal_attention_layers: int = 2
    categorical_cardinalities: tuple[int, ...] = (10, 4, 4097, 8, 3, 128, 8)


class PhysicsInformedGraphTemporalModel(nn.Module):
    """Forecast node power and branch flow on the observed power-grid topology."""

    def __init__(
        self,
        static_continuous_channels: int,
        weather_channels: int,
        node_dynamic_channels: int,
        line_dynamic_channels: int,
        config: ForecastModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ForecastModelConfig()
        hidden = int(self.config.hidden_dim)
        grid_dim = int(self.config.grid_dim)
        heads = int(self.config.attention_heads)
        if hidden % heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.temporal_backbone = str(self.config.temporal_backbone).lower()
        if self.temporal_backbone not in {"gru", "causal_attention"}:
            raise ValueError("temporal_backbone must be 'gru' or 'causal_attention'")
        temporal_heads = int(self.config.temporal_attention_heads)
        if self.temporal_backbone == "causal_attention" and hidden % temporal_heads != 0:
            raise ValueError("hidden_dim must be divisible by temporal_attention_heads")

        cardinalities = self.config.categorical_cardinalities
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(int(cardinality), int(self.config.categorical_embedding_dim))
            for cardinality in cardinalities
        )
        raster_channels = static_continuous_channels + len(cardinalities) * int(
            self.config.categorical_embedding_dim
        )
        self.static_encoder = _GridEncoder(raster_channels, grid_dim)
        self.weather_encoder = _GridEncoder(weather_channels, grid_dim)
        if self.temporal_backbone == "gru":
            self.node_history_encoder = nn.GRU(node_dynamic_channels, hidden, batch_first=True)
            self.weather_history_encoder = nn.GRU(grid_dim, hidden, batch_first=True)
            self.line_history_encoder = nn.GRU(line_dynamic_channels, hidden, batch_first=True)
        else:
            temporal_options = {
                "hidden_dim": hidden,
                "heads": temporal_heads,
                "layers": int(self.config.temporal_attention_layers),
                "dropout": float(self.config.dropout),
            }
            self.node_history_encoder = CausalTemporalEncoder(node_dynamic_channels, **temporal_options)
            self.weather_history_encoder = CausalTemporalEncoder(grid_dim, **temporal_options)
            self.line_history_encoder = CausalTemporalEncoder(line_dynamic_channels, **temporal_options)
        self.node_type_embedding = nn.Embedding(5, 8)

        self.node_input = nn.Sequential(
            nn.Linear(hidden * 2 + grid_dim + 5 + 6 + 8 + 1, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.edge_input = nn.Sequential(
            nn.Linear(hidden + 7, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.future_weather = nn.Linear(grid_dim, hidden)
        self.graph_layers = nn.ModuleList(
            GATv2Conv(
                hidden,
                hidden // heads,
                heads=heads,
                edge_dim=hidden,
                add_self_loops=False,
                dropout=float(self.config.dropout),
            )
            for _ in range(int(self.config.graph_layers))
        )
        self.graph_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in self.graph_layers)
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.node_recurrence = nn.GRUCell(2, hidden)
        self.edge_recurrence = nn.GRUCell(1, hidden)

    @classmethod
    def from_window(
        cls,
        window: dict[str, Any],
        config: ForecastModelConfig | None = None,
    ) -> "PhysicsInformedGraphTemporalModel":
        return cls(
            static_continuous_channels=int(window["static"]["continuous"].shape[0]),
            weather_channels=int(window["history"]["weather"].shape[1]),
            node_dynamic_channels=int(window["history"]["operation"]["node_dynamic"].shape[1]),
            line_dynamic_channels=int(window["history"]["operation"]["line_dynamic"].shape[1]),
            config=config,
        )

    def forward(self, window: dict[str, Any]) -> dict[str, torch.Tensor]:
        graph = window["graph"]
        history = window["history"]
        future = window["future"]
        operation_static = window["operation_static"]
        prepared = window.get("prepared")

        node_features = prepared["node_features"] if prepared else graph["node_features"].float()
        node_electrical = prepared["node_electrical"] if prepared else graph["node_electrical"].float()
        node_type = prepared["node_type"] if prepared else graph["node_type"].long()
        edge_index = prepared["edge_index"] if prepared else graph["edge_index"].long()
        edge_features = prepared["edge_features"] if prepared else graph["edge_features"].float()
        rows = prepared["rows"] if prepared else node_features[:, 0].long().clamp(
            0, window["static"]["continuous"].shape[-2] - 1
        )
        cols = prepared["cols"] if prepared else node_features[:, 1].long().clamp(
            0, window["static"]["continuous"].shape[-1] - 1
        )

        with record_function("input.static_grid_encoding"):
            static_grid = self.static_encoder(self._static_raster(window["static"], prepared))
            static_at_nodes = _sample_nodes(static_grid, rows, cols)[0]
        if prepared:
            history_weather = prepared["history_weather"]
            future_weather = prepared["future_weather"]
        else:
            history_weather, future_weather = normalize_weather(history["weather"], future["weather"])
        with record_function("input.weather_grid_encoding"):
            history_weather_grid = self.weather_encoder(history_weather)
            future_weather_grid = self.weather_encoder(future_weather)
            history_weather_nodes = _sample_nodes(history_weather_grid, rows, cols).transpose(0, 1)
            future_weather_nodes = _sample_nodes(future_weather_grid, rows, cols)

        node_scale = prepared["node_scale_mw"] if prepared else _node_power_scale(node_features, node_electrical)
        rate_mva = prepared["line_rate_mva"] if prepared else edge_features[:, 6].clamp_min(1.0)
        normalized_node_history = prepared["normalized_node_history"] if prepared else _normalize_node_history(
            history["operation"]["node_dynamic"], node_scale
        )
        normalized_line_history = prepared["normalized_line_history"] if prepared else _normalize_line_history(
            history["operation"]["line_dynamic"], rate_mva
        )
        with record_function("input.node_history_encoding"):
            node_history_state = self._encode_history(
                self.node_history_encoder,
                normalized_node_history.permute(2, 0, 1),
            )
        with record_function("input.weather_history_encoding"):
            weather_history_state = self._encode_history(self.weather_history_encoder, history_weather_nodes)
        with record_function("input.line_history_encoding"):
            line_history_state = self._encode_history(
                self.line_history_encoder,
                normalized_line_history.permute(2, 0, 1),
            )

        storage_mask = prepared["storage_mask"] if prepared else _storage_bus_mask(
            graph["node_id"], operation_static.get("site_bus_ids")
        )
        normalized_node_static = prepared["normalized_node_static"] if prepared else _normalize_node_static(
            node_features, node_electrical
        )
        with record_function("fusion.initial_state"):
            node_state = self.node_input(
                torch.cat(
                    (
                        node_history_state,
                        weather_history_state,
                        static_at_nodes,
                        normalized_node_static,
                        self.node_type_embedding(node_type),
                        storage_mask[:, None],
                    ),
                    dim=-1,
                )
            )
            edge_state = self.edge_input(
                torch.cat(
                    (
                        line_history_state,
                        prepared["normalized_edge_static"] if prepared else _normalize_edge_static(edge_features),
                    ),
                    dim=-1,
                )
            )

        load_mask = prepared["load_mask"] if prepared else (node_type == 0).float()
        generation_mask = prepared["generation_mask"] if prepared else (
            ((node_type >= 1) & (node_type <= 3)).float().maximum(storage_mask)
        )
        bidirectional_index = prepared["bidirectional_edge_index"] if prepared else torch.cat(
            (edge_index, edge_index.flip(0)), dim=1
        )
        load_steps: list[torch.Tensor] = []
        generation_steps: list[torch.Tensor] = []
        flow_steps: list[torch.Tensor] = []

        for step in range(int(future_weather_nodes.shape[0])):
            with record_function("fusion.graph_attention"):
                current = node_state + self.future_weather(future_weather_nodes[step])
                bidirectional_edge = torch.cat((edge_state, edge_state), dim=0)
                for layer, norm in zip(self.graph_layers, self.graph_norms):
                    message = layer(current, bidirectional_index, bidirectional_edge)
                    current = norm(current + F.gelu(message))

            with record_function("decode.node_output"):
                node_ratio = F.softplus(self.node_head(current))
                load_ratio = node_ratio[:, 0] * load_mask
                generation_ratio = node_ratio[:, 1] * generation_mask
                load = load_ratio * node_scale
                generation = generation_ratio * node_scale

            with record_function("decode.line_output"):
                edge_context = torch.cat(
                    (current[edge_index[0]], current[edge_index[1]], edge_state),
                    dim=-1,
                )
                flow_ratio = torch.tanh(self.edge_head(edge_context).squeeze(-1))
                flow = flow_ratio * rate_mva
            load_steps.append(load)
            generation_steps.append(generation)
            flow_steps.append(flow)
            with record_function("decode.recurrent_update"):
                node_state = self.node_recurrence(torch.stack((load_ratio, generation_ratio), dim=-1), current)
                edge_state = self.edge_recurrence(flow_ratio[:, None], edge_state)

        line_flow = torch.stack(flow_steps, dim=0)
        return {
            "p_load_mw": torch.stack(load_steps, dim=0),
            "p_generation_mw": torch.stack(generation_steps, dim=0),
            "line_flow_mw": line_flow,
            "line_loading_ratio": line_flow.abs() / rate_mva[None, :],
            "node_scale_mw": node_scale,
            "line_rate_mva": rate_mva,
            "edge_index": edge_index,
            "load_mask": load_mask,
            "generation_mask": generation_mask,
        }

    def _static_raster(
        self,
        static: dict[str, torch.Tensor],
        prepared: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        continuous = prepared["static_continuous"] if prepared else _channel_standardize(static["continuous"].float())
        categorical = prepared["static_categorical"] if prepared else static["categorical"].long()
        embeddings = []
        for channel, embedding in enumerate(self.categorical_embeddings):
            values = (categorical[channel] + 1).clamp(0, embedding.num_embeddings - 1)
            embeddings.append(embedding(values).permute(2, 0, 1))
        return torch.cat((continuous, *embeddings), dim=0)

    def _encode_history(self, encoder: nn.Module, values: torch.Tensor) -> torch.Tensor:
        if self.temporal_backbone == "gru":
            _, state = encoder(values)
            return state[0]
        return encoder(values)


class _GridEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.unsqueeze(0)
        return self.network(values)


def forecast_targets(window: dict[str, Any]) -> dict[str, torch.Tensor]:
    prepared = window.get("prepared")
    return prepared["targets"] if prepared else forecast_target_tensors(window)


def physics_informed_forecast_loss(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    balance_weight: float = 0.20,
    loading_weight: float = 0.25,
    ramp_weight: float = 0.02,
    nonnegative_weight: float = 0.0,
    line_capacity_weight: float = 0.0,
    loading_consistency_weight: float = 0.0,
    normalization_stats: dict[str, torch.Tensor] | None = None,
    normalization_parameters: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    node_scale = prediction["node_scale_mw"][None, :]
    line_rate = prediction["line_rate_mva"][None, :]
    if normalization_stats:
        parameters = normalization_parameters or {}
        load_loss = F.smooth_l1_loss(
            quantile_normalize(prediction["p_load_mw"], normalization_stats["load"], parameters),
            quantile_normalize(target["p_load_mw"], normalization_stats["load"], parameters),
        )
        generation_loss = F.smooth_l1_loss(
            quantile_normalize(prediction["p_generation_mw"], normalization_stats["genr"], parameters),
            quantile_normalize(target["p_generation_mw"], normalization_stats["genr"], parameters),
        )
        flow_loss = F.smooth_l1_loss(
            quantile_normalize(prediction["line_flow_mw"], normalization_stats["line_flow"], parameters),
            quantile_normalize(target["line_flow_mw"], normalization_stats["line_flow"], parameters),
        )
        loading_loss = F.smooth_l1_loss(
            quantile_normalize(
                prediction["line_loading_ratio"], normalization_stats["line_loading"], parameters
            ),
            quantile_normalize(target["line_loading_ratio"], normalization_stats["line_loading"], parameters),
        )
    else:
        load_loss = F.smooth_l1_loss(prediction["p_load_mw"] / node_scale, target["p_load_mw"] / node_scale)
        generation_loss = F.smooth_l1_loss(
            prediction["p_generation_mw"] / node_scale,
            target["p_generation_mw"] / node_scale,
        )
        flow_loss = F.smooth_l1_loss(prediction["line_flow_mw"] / line_rate, target["line_flow_mw"] / line_rate)
        loading_loss = F.smooth_l1_loss(prediction["line_loading_ratio"], target["line_loading_ratio"])

    divergence = torch.zeros_like(prediction["p_load_mw"])
    edge_index = prediction["edge_index"]
    divergence.index_add_(1, edge_index[0], prediction["line_flow_mw"])
    divergence.index_add_(1, edge_index[1], -prediction["line_flow_mw"])
    nodal_residual = prediction["p_generation_mw"] - prediction["p_load_mw"] - divergence
    balance_loss = (nodal_residual / node_scale).square().mean()

    if prediction["p_generation_mw"].shape[0] > 1:
        generation_ramp = torch.diff(prediction["p_generation_mw"] / node_scale, dim=0)
        ramp_loss = generation_ramp.square().mean()
    else:
        ramp_loss = prediction["p_generation_mw"].sum() * 0.0
    load_nonnegative_loss = F.relu(-prediction["p_load_mw"] / node_scale).square().mean()
    genr_nonnegative_loss = F.relu(-prediction["p_generation_mw"] / node_scale).square().mean()
    line_capacity_loss = F.relu(prediction["line_flow_mw"].abs() / line_rate - 1.0).square().mean()
    loading_consistency_loss = (
        prediction["line_loading_ratio"] - prediction["line_flow_mw"].abs() / line_rate
    ).square().mean()
    supervised = load_loss + generation_loss + flow_loss + float(loading_weight) * loading_loss
    total = (
        supervised
        + float(balance_weight) * balance_loss
        + float(ramp_weight) * ramp_loss
        + float(nonnegative_weight) * (load_nonnegative_loss + genr_nonnegative_loss)
        + float(line_capacity_weight) * line_capacity_loss
        + float(loading_consistency_weight) * loading_consistency_loss
    )
    return {
        "total": total,
        "supervised": supervised,
        "load": load_loss,
        "genr": generation_loss,
        "line_flow": flow_loss,
        "line_loading": loading_loss,
        "power_balance": balance_loss,
        "genr_ramp": ramp_loss,
        "load_nonnegative": load_nonnegative_loss,
        "genr_nonnegative": genr_nonnegative_loss,
        "line_capacity": line_capacity_loss,
        "line_loading_consistency": loading_consistency_loss,
    }


def _sample_nodes(grid: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    if grid.ndim != 4:
        raise ValueError(f"Expected BCHW grid, received shape {tuple(grid.shape)}")
    return grid[:, :, rows, cols].permute(0, 2, 1)
