from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scripts.train_physics_gst import _print_epoch_tables, _run_artifact_paths

from world_generator.dataset import SimGenrDataset, TemporalWindowDataset
from models import (
    AttentionGraphForecastConfig,
    AttentionGraphLineForecast,
    ForecastModelConfig,
    ForecastTrainingConfig,
    GRUForecastBaseline,
    GRUForecastConfig,
    GRUGraphForecastConfig,
    GRUGraphLineForecast,
    LSTMForecastBaseline,
    LSTMForecastConfig,
    PhysicsInformedGraphTemporalModel,
    arima_forecast_window,
    prepare_forecast_window,
    prepare_forecast_world,
    forecast_targets,
    physics_informed_forecast_loss,
    quantile_normalize,
    load_forecast_training_config,
    supervised_forecast_loss,
)


EXPERIMENT_CONFIGS = {
    "lstm_supervised": "configs/models/lstm_24to24.yaml",
    "lstm_physics": "configs/models/lstm_physics_24to24.yaml",
    "gru_supervised": "configs/models/gru_24to24.yaml",
    "gru_physics": "configs/models/gru_physics_24to24.yaml",
    "gru_gnn_supervised": "configs/models/gru_gnn_supervised_24to24.yaml",
    "gru_gnn_physics": "configs/models/gru_gnn_physics_24to24.yaml",
    "attn_gnn_supervised": "configs/models/attn_gnn_supervised_24to24.yaml",
    "attn_gnn_physics": "configs/models/attn_gnn_physics_24to24.yaml",
}


def test_eight_experiment_configs_are_architecture_matched_pairs() -> None:
    runs = {
        name: load_forecast_training_config(path)
        for name, path in EXPERIMENT_CONFIGS.items()
    }
    assert {run.experiment_name for run in runs.values()} == set(EXPERIMENT_CONFIGS)
    pairs = (
        ("lstm_supervised", "lstm_physics", "lstm"),
        ("gru_supervised", "gru_physics", "gru"),
        ("gru_gnn_supervised", "gru_gnn_physics", "gru_gnn"),
        ("attn_gnn_supervised", "attn_gnn_physics", "attn_gnn"),
    )
    for supervised_name, physics_name, model_type in pairs:
        supervised = runs[supervised_name]
        physics = runs[physics_name]
        assert supervised.model_type == physics.model_type == model_type
        assert supervised.model_parameters == physics.model_parameters
        assert not supervised.physics_enabled
        assert physics.physics_enabled
        assert physics.loss_parameters["balance_weight"] == 0.01


def test_tuned_gru_gnn_configs_are_a_matched_pair() -> None:
    supervised = load_forecast_training_config(
        "configs/models/gru_gnn_tuned_supervised_24to24.yaml"
    )
    physics = load_forecast_training_config(
        "configs/models/gru_gnn_tuned_physics_24to24.yaml"
    )
    assert supervised.model_type == physics.model_type == "gru_gnn"
    assert supervised.model_parameters == physics.model_parameters
    assert supervised.model_parameters["graph_gate_init"] == -1.25
    assert supervised.loss_parameters["loading_weight"] == 0.50
    assert physics.loss_parameters["loading_weight"] == 0.50
    assert physics.loss_parameters["balance_weight"] == 0.003
    assert physics.loss_parameters["ramp_weight"] == 0.01
    assert supervised.epochs == physics.epochs == 30


def test_training_artifacts_use_timestamped_seed_directory() -> None:
    run = ForecastTrainingConfig(
        model_type="physics_gst",
        history_hours=24,
        forecast_hours=24,
        initialization_seed=2026,
    )
    output, run_id = _run_artifact_paths(
        run,
        datetime(2026, 7, 21, 15, 30, 12, 427000, tzinfo=timezone.utc),
    )
    assert run_id == "20260721_1530"
    assert output == Path(
        "checkpoints/physics_gst/seed_2026/20260721_1530/24to24.pt"
    )


def test_explicit_training_output_keeps_requested_checkpoint_path() -> None:
    run = ForecastTrainingConfig(model_type="gru", output="custom/run.pt")
    output, _ = _run_artifact_paths(run, datetime.now(timezone.utc))
    assert output == Path("custom/run.pt")


def test_physics_gst_forward_and_backward_on_real_sample() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    worlds = SimGenrDataset(root, seeds=[1], as_torch=True, cache_size=1)
    windows = TemporalWindowDataset(worlds, history_hours=24, forecast_hours=6, stride=6)
    window = windows[0]
    model = PhysicsInformedGraphTemporalModel.from_window(
        window,
        ForecastModelConfig(hidden_dim=32, grid_dim=16, attention_heads=4, graph_layers=1, dropout=0.0),
    )
    prediction = model(window)
    target = forecast_targets(window)
    node_count = int(window["graph"]["node_id"].shape[0])
    edge_count = int(window["graph"]["edge_index"].shape[1])
    assert prediction["p_load_mw"].shape == (6, node_count)
    assert prediction["p_generation_mw"].shape == (6, node_count)
    assert prediction["line_flow_mw"].shape == (6, edge_count)
    torch.testing.assert_close(
        prediction["line_loading_ratio"],
        prediction["line_flow_mw"].abs() / prediction["line_rate_mva"][None, :],
    )
    losses = physics_informed_forecast_loss(prediction, target)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_lstm_baseline_forward_and_backward_on_real_sample() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    worlds = SimGenrDataset(root, seeds=[1], as_torch=True, cache_size=1)
    window = TemporalWindowDataset(worlds, history_hours=24, forecast_hours=6, stride=6)[0]
    model = LSTMForecastBaseline.from_window(window, LSTMForecastConfig(hidden_dim=32, dropout=0.0))
    prediction = model(window)
    target = forecast_targets(window)
    assert prediction["p_load_mw"].shape == target["p_load_mw"].shape
    assert prediction["p_generation_mw"].shape == target["p_generation_mw"].shape
    assert prediction["line_flow_mw"].shape == target["line_flow_mw"].shape
    losses = supervised_forecast_loss(prediction, target)
    assert all(torch.isfinite(value) for value in losses.values())
    physics_losses = physics_informed_forecast_loss(prediction, target, balance_weight=0.01)
    assert all(torch.isfinite(value) for value in physics_losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_gru_baseline_forward_and_backward_on_real_sample() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    worlds = SimGenrDataset(root, seeds=[1], as_torch=True, cache_size=1)
    window = TemporalWindowDataset(worlds, history_hours=24, forecast_hours=3, stride=6)[0]
    model = GRUForecastBaseline.from_window(window, GRUForecastConfig(hidden_dim=32, dropout=0.0))
    prediction = model(window)
    target = forecast_targets(window)
    assert prediction["p_load_mw"].shape == target["p_load_mw"].shape
    assert prediction["p_generation_mw"].shape == target["p_generation_mw"].shape
    assert prediction["line_flow_mw"].shape == target["line_flow_mw"].shape
    losses = supervised_forecast_loss(prediction, target)
    physics_losses = physics_informed_forecast_loss(prediction, target, balance_weight=0.01)
    assert all(torch.isfinite(value) for value in physics_losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_gru_line_graph_preserves_local_node_predictions() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    worlds = SimGenrDataset(root, seeds=[1], as_torch=True, cache_size=1)
    window = TemporalWindowDataset(worlds, history_hours=24, forecast_hours=3, stride=6)[0]
    torch.manual_seed(17)
    baseline = GRUForecastBaseline.from_window(
        window,
        GRUForecastConfig(hidden_dim=32, dropout=0.0),
    ).eval()
    torch.manual_seed(17)
    graph_model = GRUGraphLineForecast.from_window(
        window,
        GRUGraphForecastConfig(
            hidden_dim=32,
            dropout=0.0,
            attention_heads=4,
            graph_layers=1,
            graph_gate_init=-2.0,
        ),
    ).eval()
    baseline_prediction = baseline(window)
    graph_prediction = graph_model(window)
    torch.testing.assert_close(
        graph_prediction["p_load_mw"],
        baseline_prediction["p_load_mw"],
    )
    torch.testing.assert_close(
        graph_prediction["p_generation_mw"],
        baseline_prediction["p_generation_mw"],
    )
    torch.testing.assert_close(
        graph_prediction["line_flow_mw"],
        baseline_prediction["line_flow_mw"],
    )
    target = forecast_targets(window)
    losses = physics_informed_forecast_loss(graph_prediction, target, balance_weight=0.01)
    losses["total"].backward()
    assert graph_model.graph_gate.grad is not None
    assert torch.isfinite(graph_model.graph_gate.grad)


def test_attention_line_graph_forward_and_backward_on_real_sample() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    worlds = SimGenrDataset(root, seeds=[1], as_torch=True, cache_size=1)
    window = TemporalWindowDataset(worlds, history_hours=24, forecast_hours=3, stride=6)[0]
    model = AttentionGraphLineForecast.from_window(
        window,
        AttentionGraphForecastConfig(
            hidden_dim=32,
            dropout=0.0,
            attention_heads=4,
            graph_layers=1,
            graph_gate_init=-2.0,
            temporal_attention_heads=4,
            temporal_attention_layers=1,
        ),
    )
    assert torch.count_nonzero(model.graph_edge_correction[-1].weight) == 0
    prediction = model(window)
    target = forecast_targets(window)
    assert prediction["p_load_mw"].shape == target["p_load_mw"].shape
    assert prediction["p_generation_mw"].shape == target["p_generation_mw"].shape
    assert prediction["line_flow_mw"].shape == target["line_flow_mw"].shape
    losses = physics_informed_forecast_loss(prediction, target, balance_weight=0.01)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.graph_edge_correction[-1].weight.grad is not None
    assert torch.isfinite(model.graph_edge_correction[-1].weight.grad).all()


def test_causal_attention_physics_gst_forward_and_backward_on_real_sample() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    worlds = SimGenrDataset(root, seeds=[1], as_torch=True, cache_size=1)
    window = TemporalWindowDataset(worlds, history_hours=24, forecast_hours=3, stride=6)[0]
    model = PhysicsInformedGraphTemporalModel.from_window(
        window,
        ForecastModelConfig(
            hidden_dim=32,
            grid_dim=16,
            attention_heads=4,
            graph_layers=1,
            dropout=0.0,
            temporal_backbone="causal_attention",
            temporal_attention_heads=4,
            temporal_attention_layers=1,
        ),
    )
    prediction = model(window)
    target = forecast_targets(window)
    assert prediction["p_load_mw"].shape == target["p_load_mw"].shape
    assert prediction["line_flow_mw"].shape == target["line_flow_mw"].shape
    losses = physics_informed_forecast_loss(prediction, target)
    losses["total"].backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_world_preload_reuses_device_tensors() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    worlds = SimGenrDataset(
        root,
        seeds=[1],
        as_torch=True,
        cache_size=1,
        device="cpu",
    ).preload()
    first = worlds[0]
    second = worlds[0]
    assert first is second
    assert first["dynamic"]["weather"].device.type == "cpu"
    assert first["graph"]["node_features"].device.type == "cpu"
    assert first["operation"]["node_dynamic"].device.type == "cpu"
    assert worlds.cached_tensor_bytes() > 0


def test_prepared_window_reuses_static_tensors_and_preserves_lstm_output() -> None:
    root = Path("datasets/seed_1_50")
    if not root.exists():
        return
    raw_worlds = SimGenrDataset(root, seeds=[1], as_torch=True, cache_size=1).preload()
    prepared_worlds = SimGenrDataset(
        root,
        seeds=[1],
        as_torch=True,
        cache_size=1,
        transform=prepare_forecast_world,
    ).preload()
    raw = TemporalWindowDataset(raw_worlds, history_hours=24, forecast_hours=6, stride=6)[0]
    prepared_windows = TemporalWindowDataset(
        prepared_worlds,
        history_hours=24,
        forecast_hours=6,
        stride=6,
        transform=prepare_forecast_window,
        cache_size=24,
    ).preload()
    prepared = prepared_windows[0]
    assert prepared["prepared"]["static_continuous"].data_ptr() == prepared_windows[1]["prepared"][
        "static_continuous"
    ].data_ptr()
    model = LSTMForecastBaseline.from_window(raw, LSTMForecastConfig(hidden_dim=32, dropout=0.0)).eval()
    with torch.no_grad():
        raw_prediction = model(raw)
        prepared_prediction = model(prepared)
    for name in ("p_load_mw", "p_generation_mw", "line_flow_mw", "line_loading_ratio"):
        torch.testing.assert_close(raw_prediction[name], prepared_prediction[name])


def test_quantile_normalization_and_physical_loss_records() -> None:
    statistics = {
        name: torch.tensor([0.0, 10.0])
        for name in ("load", "genr", "line_flow", "line_loading")
    }
    parameters = {"left_clip": 1, "right_clip": 1}
    torch.testing.assert_close(
        quantile_normalize(torch.tensor([-20.0, 0.0, 5.0, 10.0, 30.0]), statistics["load"], parameters),
        torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0]),
    )
    prediction = {
        "p_load_mw": torch.tensor([[2.0, 0.0]]),
        "p_generation_mw": torch.tensor([[0.0, 2.0]]),
        "line_flow_mw": torch.tensor([[2.0]]),
        "line_loading_ratio": torch.tensor([[0.2]]),
        "node_scale_mw": torch.tensor([10.0, 10.0]),
        "line_rate_mva": torch.tensor([10.0]),
        "edge_index": torch.tensor([[0], [1]]),
    }
    target = {
        name: prediction[name].clone()
        for name in ("p_load_mw", "p_generation_mw", "line_flow_mw", "line_loading_ratio")
    }
    losses = physics_informed_forecast_loss(
        prediction,
        target,
        normalization_stats=statistics,
        normalization_parameters=parameters,
    )
    assert set(losses) == {
        "total",
        "supervised",
        "load",
        "genr",
        "line_flow",
        "line_loading",
        "power_balance",
        "genr_ramp",
        "load_nonnegative",
        "genr_nonnegative",
        "line_capacity",
        "line_loading_consistency",
    }


def test_epoch_console_log_uses_compact_tables(capsys: object) -> None:
    losses = {
        "total": 0.12,
        "supervised": 0.10,
        "load": 0.02,
        "genr": 0.03,
        "line_flow": 0.04,
        "line_loading": 0.01,
        "power_balance": 0.05,
        "genr_ramp": 0.001,
        "load_nonnegative": 0.0,
        "genr_nonnegative": 0.0,
        "line_capacity": 0.0,
        "line_loading_consistency": 0.0,
    }
    row = {
        "epoch": 1,
        "train": {"losses": losses},
        "val": {
            "losses": losses,
            "metrics": {
                entity: {"rmse": 0.1, "mae": 0.08, "r2": 0.9}
                for entity in ("load", "genr", "line")
            },
        },
    }
    _print_epoch_tables(row, 1, 5)
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Validation metrics" in output
    assert "Prediction losses" in output
    assert "Physical constraint losses" in output
    assert "load" in output and "genr" in output and "line" in output
    assert "average" in output
    assert "train" in output and "val" in output
    assert "{" not in output


def test_arima_evaluation_filters_with_train_fitted_parameters() -> None:
    time = torch.arange(24, dtype=torch.float32)
    node_dynamic = torch.zeros((24, 10, 2), dtype=torch.float32)
    node_dynamic[:, 0, 0] = 5.0 + torch.sin(time / 4.0)
    node_dynamic[:, 2, 1] = 8.0 + torch.cos(time / 5.0)
    line_dynamic = torch.zeros((24, 2, 1), dtype=torch.float32)
    line_dynamic[:, 0, 0] = 2.0 + torch.sin(time / 3.0)
    window = {
        "graph": {
            "node_type": torch.tensor([0, 1]),
            "node_id": torch.tensor([0, 1]),
            "edge_features": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0]]),
        },
        "history": {
            "operation": {"node_dynamic": node_dynamic, "line_dynamic": line_dynamic},
        },
        "future": {"timestamps": torch.arange(24)},
        "operation_static": {
            "node_dynamic_channels": (
                "p_load_mw",
                "unused_1",
                "p_gen_scheduled_mw",
                "unused_3",
                "unused_4",
                "unused_5",
                "unused_6",
                "unused_7",
                "unused_8",
                "unused_9",
            ),
            "line_dynamic_channels": ("line_flow_mw", "line_loading_ratio"),
            "site_bus_ids": torch.zeros(0, dtype=torch.long),
        },
    }
    fitted_state = {
        "order": [1, 0, 0],
        "parameters": {
            "load": [0.0, 0.5, 1.0],
            "genr": [0.0, 0.5, 1.0],
            "line_flow": [0.0, 0.5, 1.0],
        },
    }
    prediction, diagnostics = arima_forecast_window(window, fitted_state)
    assert prediction["p_load_mw"].shape == (24, 2)
    assert prediction["p_generation_mw"].shape == (24, 2)
    assert prediction["line_flow_mw"].shape == (24, 1)
    assert diagnostics == {"filtered": 3, "constant": 0, "fallback": 0}
    assert np.isfinite(prediction["line_loading_ratio"]).all()
