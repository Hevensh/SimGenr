from __future__ import annotations

import numpy as np

from world_generator.core.config import PowerGridConfig
from world_generator.core.datatypes import GridElectricalState, GridUpgradePlanStore, PowerFlowStore


def build_grid_upgrade_plan(
    power_flow: PowerFlowStore,
    electrical: GridElectricalState,
    *,
    power_grid: PowerGridConfig | None = None,
    target_peak_loading: float = 0.92,
    target_p95_loading: float = 0.82,
    trigger_loading: float = 0.80,
) -> GridUpgradePlanStore:
    power_grid = power_grid or PowerGridConfig()
    branch_ids = np.asarray([branch.edge_id for branch in electrical.branch_params], dtype=np.int32)
    current_rate = np.asarray([branch.rate_mva for branch in electrical.branch_params], dtype=np.float32)
    loading = power_flow.line_loading_ratio.astype(np.float32, copy=False)
    if loading.size == 0:
        zeros = np.zeros(branch_ids.shape, dtype=np.float32)
        return GridUpgradePlanStore(branch_ids, current_rate, current_rate.copy(), np.ones_like(current_rate), zeros, zeros, zeros.astype(np.int32), zeros.astype(np.int32), zeros, zeros)

    peak = np.nanmax(loading, axis=0).astype(np.float32)
    p95 = np.nanpercentile(loading, 95, axis=0).astype(np.float32)
    hours_over_80 = (loading > trigger_loading).sum(axis=0).astype(np.int32)
    hours_over_100 = (loading > 1.0).sum(axis=0).astype(np.int32)
    rates_by_flow_order = _rates_by_branch_id(branch_ids, current_rate, power_flow.branch_ids)
    overload_mwh_proxy = (np.maximum(loading - 1.0, 0.0) * rates_by_flow_order[None, :]).sum(axis=0).astype(np.float32)

    factor_from_peak = peak / max(float(target_peak_loading), 1e-6)
    factor_from_p95 = p95 / max(float(target_p95_loading), 1e-6)
    raw_factor_by_flow_order = np.maximum.reduce(
        [
            np.ones_like(peak, dtype=np.float32),
            factor_from_peak.astype(np.float32),
            factor_from_p95.astype(np.float32),
        ]
    )
    raw_factor_by_flow_order = np.where(hours_over_80 > 0, raw_factor_by_flow_order, 1.0)
    upgrade_factor_by_flow_order = np.clip(
        _round_upgrade_factor(raw_factor_by_flow_order),
        1.0,
        float(power_grid.max_upgrade_factor),
    ).astype(np.float32)
    recommended_rate_by_flow_order = rates_by_flow_order * upgrade_factor_by_flow_order
    priority_score_by_flow_order = (
        2.4 * np.maximum(peak - 1.0, 0.0)
        + 1.0 * np.maximum(p95 - trigger_loading, 0.0)
        + 0.025 * hours_over_100
        + 0.006 * hours_over_80
        + 0.002 * overload_mwh_proxy / np.maximum(rates_by_flow_order, 1.0)
    ).astype(np.float32)
    priority_score_by_flow_order = np.where(upgrade_factor_by_flow_order > 1.01, priority_score_by_flow_order, 0.0)

    return GridUpgradePlanStore(
        branch_ids=branch_ids,
        current_rate_mva=current_rate,
        recommended_rate_mva=_values_by_branch_id(power_flow.branch_ids, recommended_rate_by_flow_order, branch_ids),
        upgrade_factor=_values_by_branch_id(power_flow.branch_ids, upgrade_factor_by_flow_order, branch_ids),
        peak_loading_ratio=_values_by_branch_id(power_flow.branch_ids, peak, branch_ids),
        p95_loading_ratio=_values_by_branch_id(power_flow.branch_ids, p95, branch_ids),
        hours_over_80pct=_values_by_branch_id(power_flow.branch_ids, hours_over_80, branch_ids).astype(np.int32),
        hours_over_100pct=_values_by_branch_id(power_flow.branch_ids, hours_over_100, branch_ids).astype(np.int32),
        overload_mwh_proxy=_values_by_branch_id(power_flow.branch_ids, overload_mwh_proxy, branch_ids),
        priority_score=_values_by_branch_id(power_flow.branch_ids, priority_score_by_flow_order, branch_ids),
    )


def _round_upgrade_factor(values: np.ndarray) -> np.ndarray:
    steps = np.asarray([1.0, 1.25, 1.5, 1.75, 2.0, 2.4, 2.8], dtype=np.float32)
    indices = np.searchsorted(steps, values, side="left")
    indices = np.clip(indices, 0, len(steps) - 1)
    return steps[indices]


def _rates_by_branch_id(source_ids: np.ndarray, source_values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    return _values_by_branch_id(source_ids, source_values, target_ids).astype(np.float32)


def _values_by_branch_id(source_ids: np.ndarray, source_values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    values_by_id = {int(branch_id): value for branch_id, value in zip(source_ids, source_values)}
    return np.asarray([values_by_id[int(branch_id)] for branch_id in target_ids])
