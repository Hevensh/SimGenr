from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.profiler import record_function
from torch_geometric.nn import GATv2Conv

from .preprocessing import (
    node_power_scale as _node_power_scale,
    normalize_edge_static as _normalize_edge_static,
    normalize_line_history as _normalize_line_history,
    normalize_node_history as _normalize_node_history,
    normalize_node_static as _normalize_node_static,
    quantile_normalize,
    storage_bus_mask as _storage_bus_mask,
)
from .temporal import CausalTemporalEncoder


@dataclass(frozen=True)
class LSTMForecastConfig:
    hidden_dim: int = 96
    dropout: float = 0.10


@dataclass(frozen=True)
class GRUForecastConfig:
    hidden_dim: int = 96
    dropout: float = 0.10


@dataclass(frozen=True)
class GRUGraphForecastConfig(GRUForecastConfig):
    attention_heads: int = 4
    graph_layers: int = 1
    graph_gate_init: float = -2.0


@dataclass(frozen=True)
class AttentionGraphForecastConfig(GRUGraphForecastConfig):
    temporal_attention_heads: int = 8
    temporal_attention_layers: int = 2


class _RecurrentForecastBaseline(nn.Module):
    """Shared recurrent baseline without graph message passing or physical loss."""

    recurrent_type = ""

    def __init__(
        self,
        weather_channels: int,
        node_dynamic_channels: int,
        line_dynamic_channels: int,
        config: LSTMForecastConfig | GRUForecastConfig | None = None,
    ) -> None:
        super().__init__()
        if self.recurrent_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported recurrent baseline: {self.recurrent_type!r}")
        default_config = LSTMForecastConfig() if self.recurrent_type == "lstm" else GRUForecastConfig()
        self.config = config or default_config
        hidden = int(self.config.hidden_dim)
        dropout = float(self.config.dropout)
        recurrent = nn.LSTM if self.recurrent_type == "lstm" else nn.GRU
        recurrent_cell = nn.LSTMCell if self.recurrent_type == "lstm" else nn.GRUCell
        self.node_history_encoder = recurrent(node_dynamic_channels, hidden, batch_first=True)
        self.line_history_encoder = recurrent(line_dynamic_channels, hidden, batch_first=True)
        self.node_type_embedding = nn.Embedding(5, 8)
        self.node_initial = nn.Sequential(
            nn.Linear(hidden + 5 + 6 + 8 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.edge_initial = nn.Sequential(
            nn.Linear(hidden + 7, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.node_decoder = recurrent_cell(2 + weather_channels, hidden)
        self.edge_decoder = recurrent_cell(1, hidden)
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.edge_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @classmethod
    def from_window(
        cls,
        window: dict[str, Any],
        config: LSTMForecastConfig | GRUForecastConfig | None = None,
    ) -> "_RecurrentForecastBaseline":
        return cls(
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
        edge_features = prepared["edge_features"] if prepared else graph["edge_features"].float()
        edge_index = prepared["edge_index"] if prepared else graph["edge_index"].long()
        rows = prepared["rows"] if prepared else node_features[:, 0].long().clamp(0, history["weather"].shape[-2] - 1)
        cols = prepared["cols"] if prepared else node_features[:, 1].long().clamp(0, history["weather"].shape[-1] - 1)
        node_scale = prepared["node_scale_mw"] if prepared else _node_power_scale(node_features, node_electrical)
        rate_mva = prepared["line_rate_mva"] if prepared else edge_features[:, 6].clamp_min(1.0)

        node_history = prepared["normalized_node_history"] if prepared else _normalize_node_history(
            history["operation"]["node_dynamic"], node_scale
        )
        line_history = prepared["normalized_line_history"] if prepared else _normalize_line_history(
            history["operation"]["line_dynamic"], rate_mva
        )
        with record_function("input.node_history_encoding"):
            node_hidden, node_cell = self._encode_history(
                self.node_history_encoder,
                node_history.permute(2, 0, 1),
            )
        with record_function("input.line_history_encoding"):
            edge_hidden, edge_cell = self._encode_history(
                self.line_history_encoder,
                line_history.permute(2, 0, 1),
            )
        storage_mask = prepared["storage_mask"] if prepared else _storage_bus_mask(
            graph["node_id"], operation_static.get("site_bus_ids")
        )
        with record_function("fusion.initial_state"):
            node_hidden = self.node_initial(
                torch.cat(
                    (
                        node_hidden,
                        prepared["normalized_node_static"] if prepared else _normalize_node_static(
                            node_features, node_electrical
                        ),
                        self.node_type_embedding(node_type),
                        storage_mask[:, None],
                    ),
                    dim=-1,
                )
            )
            edge_hidden = self.edge_initial(
                torch.cat(
                    (
                        edge_hidden,
                        prepared["normalized_edge_static"] if prepared else _normalize_edge_static(edge_features),
                    ),
                    dim=-1,
                )
            )

        with record_function("input.future_weather_sampling"):
            future_weather = prepared["future_weather_at_nodes"] if prepared else _future_weather_at_nodes(
                history["weather"], future["weather"], rows, cols
            )
        load_mask = prepared["load_mask"] if prepared else (node_type == 0).float()
        generation_mask = prepared["generation_mask"] if prepared else (
            ((node_type >= 1) & (node_type <= 3)).float().maximum(storage_mask)
        )
        last_node = node_history[-1]
        previous_load = last_node[0]
        previous_generation = last_node[2]
        previous_flow = line_history[-1, 0]
        load_steps: list[torch.Tensor] = []
        generation_steps: list[torch.Tensor] = []
        flow_steps: list[torch.Tensor] = []

        for step in range(int(future_weather.shape[0])):
            with record_function("decode.node_recurrence"):
                node_input = torch.cat(
                    (previous_load[:, None], previous_generation[:, None], future_weather[step]),
                    dim=-1,
                )
                node_hidden, node_cell = self._decode_step(
                    self.node_decoder,
                    node_input,
                    node_hidden,
                    node_cell,
                )
            with record_function("decode.node_output"):
                node_ratio = F.softplus(self.node_head(node_hidden))
                previous_load = node_ratio[:, 0] * load_mask
                previous_generation = node_ratio[:, 1] * generation_mask
                load_steps.append(previous_load * node_scale)
                generation_steps.append(previous_generation * node_scale)

            with record_function("decode.line_recurrence"):
                edge_hidden, edge_cell = self._decode_step(
                    self.edge_decoder,
                    previous_flow[:, None],
                    edge_hidden,
                    edge_cell,
                )
            with record_function("decode.line_output"):
                previous_flow = self._predict_flow_ratio(edge_hidden, node_hidden, edge_index)
                flow_steps.append(previous_flow * rate_mva)

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

    def _history_state(
        self,
        encoded: tuple[torch.Tensor, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = encoded[1]
        if self.recurrent_type == "lstm":
            hidden, cell = state
            return hidden[0], cell[0]
        return state[0], None

    def _encode_history(
        self,
        encoder: nn.Module,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._history_state(encoder(values))

    def _decode_step(
        self,
        decoder: nn.Module,
        values: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.recurrent_type == "lstm":
            assert cell is not None
            next_hidden, next_cell = decoder(values, (hidden, cell))
            return next_hidden, next_cell
        return decoder(values, hidden), None

    def _predict_flow_ratio(
        self,
        edge_hidden: torch.Tensor,
        node_hidden: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        del node_hidden, edge_index
        return torch.tanh(self.edge_head(edge_hidden).squeeze(-1))


class LSTMForecastBaseline(_RecurrentForecastBaseline):
    """Shared per-entity LSTM baseline."""

    recurrent_type = "lstm"


class GRUForecastBaseline(_RecurrentForecastBaseline):
    """Shared per-entity GRU baseline with the same features and heads as LSTM."""

    recurrent_type = "gru"


class GRUGraphLineForecast(_RecurrentForecastBaseline):
    """GRU baseline with gated graph context restricted to line-flow decoding."""

    recurrent_type = "gru"

    def __init__(
        self,
        weather_channels: int,
        node_dynamic_channels: int,
        line_dynamic_channels: int,
        config: GRUGraphForecastConfig | None = None,
    ) -> None:
        graph_config = config or GRUGraphForecastConfig()
        super().__init__(
            weather_channels,
            node_dynamic_channels,
            line_dynamic_channels,
            config=graph_config,
        )
        self.config = graph_config
        hidden = int(graph_config.hidden_dim)
        heads = int(graph_config.attention_heads)
        if hidden % heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.line_graph_layers = nn.ModuleList(
            GATv2Conv(
                hidden,
                hidden // heads,
                heads=heads,
                edge_dim=hidden,
                add_self_loops=False,
                dropout=float(graph_config.dropout),
            )
            for _ in range(int(graph_config.graph_layers))
        )
        self.line_graph_norms = nn.ModuleList(
            nn.LayerNorm(hidden) for _ in self.line_graph_layers
        )
        self.graph_gate = nn.Parameter(torch.tensor(float(graph_config.graph_gate_init)))
        self.graph_edge_correction = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.graph_edge_correction[-1].weight)
        nn.init.zeros_(self.graph_edge_correction[-1].bias)

    def _predict_flow_ratio(
        self,
        edge_hidden: torch.Tensor,
        node_hidden: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        with record_function("fusion.line_graph_attention"):
            bidirectional_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
            bidirectional_edge = torch.cat((edge_hidden, edge_hidden), dim=0)
            graph_nodes = node_hidden
            for layer, norm in zip(self.line_graph_layers, self.line_graph_norms):
                message = layer(graph_nodes, bidirectional_index, bidirectional_edge)
                graph_nodes = norm(graph_nodes + F.gelu(message))
            gate = torch.sigmoid(self.graph_gate)
        edge_context = torch.cat(
            (graph_nodes[edge_index[0]], graph_nodes[edge_index[1]], edge_hidden),
            dim=-1,
        )
        base_logit = self.edge_head(edge_hidden).squeeze(-1)
        graph_correction = self.graph_edge_correction(edge_context).squeeze(-1)
        return torch.tanh(base_logit + gate * graph_correction)


class AttentionGraphLineForecast(GRUGraphLineForecast):
    """Causal-attention history encoders with the gated line-only graph branch."""

    def __init__(
        self,
        weather_channels: int,
        node_dynamic_channels: int,
        line_dynamic_channels: int,
        config: AttentionGraphForecastConfig | None = None,
    ) -> None:
        attention_config = config or AttentionGraphForecastConfig()
        super().__init__(
            weather_channels,
            node_dynamic_channels,
            line_dynamic_channels,
            config=attention_config,
        )
        self.config = attention_config
        temporal_options = {
            "hidden_dim": int(attention_config.hidden_dim),
            "heads": int(attention_config.temporal_attention_heads),
            "layers": int(attention_config.temporal_attention_layers),
            "dropout": float(attention_config.dropout),
        }
        self.node_history_encoder = CausalTemporalEncoder(
            node_dynamic_channels,
            **temporal_options,
        )
        self.line_history_encoder = CausalTemporalEncoder(
            line_dynamic_channels,
            **temporal_options,
        )

    def _encode_history(
        self,
        encoder: nn.Module,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return encoder(values), None


def supervised_forecast_loss(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    loading_weight: float = 0.25,
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
    total = load_loss + generation_loss + flow_loss + float(loading_weight) * loading_loss
    return {
        "total": total,
        "supervised": total,
        "load": load_loss,
        "genr": generation_loss,
        "line_flow": flow_loss,
        "line_loading": loading_loss,
    }


def fit_arima_baseline(
    worlds: Any,
    order: tuple[int, int, int] = (2, 0, 1),
    max_series_per_entity: int = 128,
) -> dict[str, Any]:
    from statsmodels.tsa.arima.model import ARIMA

    series = {"load": [], "genr": [], "line_flow": []}
    for world_index in range(len(worlds)):
        world = worlds[world_index]
        operation = world["operation"]
        graph = world["graph"]
        node_channels = tuple(str(value) for value in operation["node_dynamic_channels"])
        line_channels = tuple(str(value) for value in operation["line_dynamic_channels"])
        node = _as_numpy(operation["node_dynamic"])
        line = _as_numpy(operation["line_dynamic"])
        node_type = _as_numpy(graph["node_type"]).astype(np.int64)
        node_ids = _as_numpy(graph["node_id"]).astype(np.int64)
        site_bus_ids = _as_numpy(operation.get("site_bus_ids", np.zeros(0, dtype=np.int64)))
        load_mask = node_type == 0
        genr_mask = np.isin(node_type, (1, 2, 3)) | np.isin(node_ids, site_bus_ids)
        series["load"].extend(node[:, node_channels.index("p_load_mw"), load_mask].T)
        series["genr"].extend(node[:, node_channels.index("p_gen_scheduled_mw"), genr_mask].T)
        series["line_flow"].extend(line[:, line_channels.index("line_flow_mw")].T)

    parameters: dict[str, list[float]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    parameter_names: list[str] | None = None
    for entity, values in series.items():
        selected = _evenly_select_series(values, max_series_per_entity)
        fitted_parameters: list[np.ndarray] = []
        skipped = 0
        for values_one in selected:
            normalized = _standardized_series(values_one)
            if normalized is None:
                skipped += 1
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(normalized, order=order, trend=_arima_trend(order)).fit(low_memory=True)
                fitted_parameters.append(np.asarray(fitted.params, dtype=np.float64))
                if parameter_names is None:
                    parameter_names = list(fitted.param_names)
            except (ValueError, np.linalg.LinAlgError):
                skipped += 1
        if not fitted_parameters:
            raise RuntimeError(f"No train ARIMA series could be fitted for {entity}")
        stacked = np.stack(fitted_parameters)
        median_center = np.median(stacked, axis=0)
        representative = stacked[int(np.argmin(np.sum((stacked - median_center) ** 2, axis=1)))]
        parameters[entity] = representative.tolist()
        diagnostics[entity] = {
            "available": len(values),
            "selected": len(selected),
            "fitted": len(fitted_parameters),
            "skipped": skipped,
        }
    return {
        "order": list(order),
        "trend": _arima_trend(order),
        "parameter_names": parameter_names,
        "parameters": parameters,
        "train_diagnostics": diagnostics,
    }


def arima_forecast_window(
    window: dict[str, Any],
    fitted_state: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    from statsmodels.tsa.arima.model import ARIMA

    order = tuple(int(value) for value in fitted_state["order"])
    node_channels = tuple(str(value) for value in window["operation_static"]["node_dynamic_channels"])
    line_channels = tuple(str(value) for value in window["operation_static"]["line_dynamic_channels"])
    node_history = _as_numpy(window["history"]["operation"]["node_dynamic"])
    line_history = _as_numpy(window["history"]["operation"]["line_dynamic"])
    node_type = _as_numpy(window["graph"]["node_type"]).astype(np.int64)
    node_ids = _as_numpy(window["graph"]["node_id"]).astype(np.int64)
    site_bus_ids = _as_numpy(window["operation_static"].get("site_bus_ids", np.zeros(0, dtype=np.int64)))
    rate_mva = np.maximum(_as_numpy(window["graph"]["edge_features"])[:, 6], 1.0)
    horizon = int(window["future"]["timestamps"].shape[0])
    load_history = node_history[:, node_channels.index("p_load_mw")]
    generation_history = node_history[:, node_channels.index("p_gen_scheduled_mw")]
    flow_history = line_history[:, line_channels.index("line_flow_mw")]
    load_mask = node_type == 0
    generation_mask = np.isin(node_type, (1, 2, 3)) | np.isin(node_ids, site_bus_ids)
    diagnostics = {"filtered": 0, "constant": 0, "fallback": 0}

    def forecast_matrix(values: np.ndarray, active: np.ndarray, entity: str) -> np.ndarray:
        result = np.zeros((horizon, values.shape[1]), dtype=np.float32)
        for index in np.flatnonzero(active):
            result[:, index], status = _filter_arima_series(
                values[:, index],
                horizon,
                order,
                np.asarray(fitted_state["parameters"][entity], dtype=np.float64),
                ARIMA,
            )
            diagnostics[status] += 1
        return result

    load = forecast_matrix(load_history, load_mask, "load")
    generation = forecast_matrix(generation_history, generation_mask, "genr")
    flow = forecast_matrix(flow_history, np.ones(flow_history.shape[1], dtype=bool), "line_flow")
    return {
        "p_load_mw": load,
        "p_generation_mw": generation,
        "line_flow_mw": flow,
        "line_loading_ratio": np.abs(flow) / rate_mva[None, :],
    }, diagnostics


def _filter_arima_series(
    values: np.ndarray,
    horizon: int,
    order: tuple[int, int, int],
    parameters: np.ndarray,
    arima_class: Any,
) -> tuple[np.ndarray, str]:
    series = np.asarray(values, dtype=np.float64)
    if not np.isfinite(series).all() or float(np.std(series)) < 1e-6:
        return np.full(horizon, float(series[-1]) if series.size else 0.0, dtype=np.float32), "constant"
    mean = float(np.mean(series))
    scale = max(float(np.std(series)), 1e-6)
    normalized = (series - mean) / scale
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = arima_class(normalized, order=order, trend=_arima_trend(order)).filter(
                parameters,
                low_memory=True,
            )
            forecast = np.asarray(filtered.forecast(horizon), dtype=np.float64) * scale + mean
        if not np.isfinite(forecast).all():
            raise ValueError("non-finite forecast")
        return forecast.astype(np.float32), "filtered"
    except (ValueError, np.linalg.LinAlgError):
        return np.full(horizon, float(series[-1]), dtype=np.float32), "fallback"


def _standardized_series(values: np.ndarray) -> np.ndarray | None:
    series = np.asarray(values, dtype=np.float64)
    if not np.isfinite(series).all() or float(np.std(series)) < 1e-6:
        return None
    return (series - float(np.mean(series))) / max(float(np.std(series)), 1e-6)


def _evenly_select_series(values: list[np.ndarray], maximum: int) -> list[np.ndarray]:
    if maximum <= 0 or len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, num=maximum, dtype=np.int64)
    return [values[int(index)] for index in indices]


def _arima_trend(order: tuple[int, int, int]) -> str:
    return "c" if order[1] == 0 else "t"


def _future_weather_at_nodes(
    history: torch.Tensor,
    future: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    mean = history.float().mean(dim=(0, 2, 3), keepdim=True)
    std = history.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
    normalized = (future.float() - mean) / std
    return normalized[:, :, rows, cols].permute(0, 2, 1)


def _as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)
