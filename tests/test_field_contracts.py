from dataclasses import replace

import numpy as np
import pytest

from world_generator.core.config import ContractConfig, WorldConfig, WorldGridConfig, dump_config_snapshot, load_world_config
from world_generator.core.contracts import (
    aggregate_complete_days, cell_area_km2, depth_mm_to_volume_m3,
    field_contract_document, integrate_power_mwh, interval_bounds_hours,
    validate_node_arrays, validate_weather_arrays,
)


def test_physical_integrals_and_non_hourly_helpers():
    assert integrate_power_mwh(np.array([7.0]), 1.0) == 7.0
    assert integrate_power_mwh(np.array([4.0, 8.0]), np.array([0.25, 0.5])) == 5.0
    assert cell_area_km2(1.0) == 1.0
    np.testing.assert_allclose(depth_mm_to_volume_m3(np.array([1.0, 2.0]), 1.0), [1000.0, 2000.0])
    # The same physical 4 km2 area has the same water volume at either resolution.
    assert depth_mm_to_volume_m3(np.ones((2, 2)), cell_area_km2(1)).sum() == depth_mm_to_volume_m3(np.ones((4, 4)), cell_area_km2(0.5)).sum()


def test_daily_means_rates_and_accumulations_are_distinct():
    times = np.arange(48) * 0.5
    _, power = aggregate_complete_days(np.full((48, 2), 3.0), times, quantity_kind="interval_mean", step_hours=0.5)
    _, depth = aggregate_complete_days(np.full((48, 2), 1.0), times, quantity_kind="accumulation", step_hours=0.5)
    _, rate_integral = aggregate_complete_days(np.full((48, 2), 2.0), times, quantity_kind="rate", step_hours=0.5)
    np.testing.assert_allclose(power, 3.0)
    np.testing.assert_allclose(depth, 48.0)
    np.testing.assert_allclose(rate_integral, depth)
    np.testing.assert_allclose(power * 24, integrate_power_mwh(np.full((48, 2), 3.0), 0.5)[None, :])


@pytest.mark.parametrize("step", [0, -1, float("nan"), float("inf")])
def test_illegal_time_steps_fail(step):
    with pytest.raises(ValueError):
        interval_bounds_hours(np.arange(24), step)
    with pytest.raises(ValueError):
        ContractConfig(time_step_hours=step)


def test_shapes_alignment_and_negative_capacities_fail():
    with pytest.raises(ValueError, match="time dimensions"):
        aggregate_complete_days(np.ones(12), np.arange(24), quantity_kind="accumulation")
    with pytest.raises(ValueError, match="complete"):
        aggregate_complete_days(np.ones(23), np.arange(23), quantity_kind="accumulation")
    with pytest.raises(ValueError, match="states"):
        aggregate_complete_days(np.ones(24), np.arange(24), quantity_kind="state")
    with pytest.raises(ValueError, match="weather_class"):
        validate_weather_arrays(np.zeros((24, 1, 2, 3)), np.zeros((24, 2, 2)), np.arange(24), ("cloud",), "hour")
    with pytest.raises(ValueError, match="expected"):
        validate_node_arrays(np.arange(24), np.arange(3), {"p_load_mw": np.zeros((24, 2))})
    with pytest.raises(ValueError, match="negative"):
        validate_node_arrays(np.arange(24), np.arange(3), {"p_load_mw": -np.ones((24, 3))})
    with pytest.raises(ValueError, match="capacity"):
        config = WorldConfig()
        replace(config, energy=replace(config.energy, wind_capacity_min_mw=-1))
    with pytest.raises(ValueError):
        WorldGridConfig(cell_size_km=0)
    with pytest.raises(ValueError, match="exactly 1 h"):
        ContractConfig(time_step_hours=0.5)


def test_config_snapshot_and_field_semantics(tmp_path):
    config = WorldConfig()
    path = tmp_path / "config.yaml"
    dump_config_snapshot(config, path)
    assert load_world_config(path) == config
    fields = field_contract_document()["fields"]
    assert fields["weather.precipitation"]["quantity_kind"] == "accumulation"
    assert fields["soc_mwh"]["time_support"] == "T+1_interval_boundaries"
    assert fields["buildability"]["quantity_kind"] == "score"
    assert fields["flow_accumulation"]["unit"] == "km2"
