"""Independent acceptance/rejection cases for declared daily weather statistics."""
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.validate_world_physics import Checks, check_weather_contracts
from world_generator.core.config import WorldConfig


def _turning_weather():
    channels = ("temperature", "humidity", "pressure", "wind_u", "wind_v", "wind_speed", "cloud", "precipitation", "irradiance")
    values = np.zeros((24, len(channels), 1, 1), dtype=np.float64)
    values[:, 0] = 20
    values[:, 1] = 0.5
    values[:, 2] = 1013
    values[:12, 3] = 5
    values[12:, 3] = -5
    values[:, 5] = 5
    es = 6.108 * np.exp(17.27 * 20 / (20 + 237.3))
    e = 0.5 * es
    epsilon = 287.05 / 461.5
    q = epsilon * e / (1013 - (1 - epsilon) * e)
    rho = 101300 / (287.05 * 293.15 * (1 + (461.5 / 287.05 - 1) * q))
    hourly = {
        "timestamps": np.arange(24), "dynamic": values, "channel_names": np.array(channels),
        "weather_metadata_json": np.asarray(json.dumps({"generation_mode": "primitive_hourly", "daily_constraints": ["temperature", "cloud", "precipitation", "wind_u", "wind_v"]})),
        "diagnostic__specific_humidity_kg_kg": np.full((24, 1, 1), q),
        "diagnostic__sea_level_pressure_hpa": np.full((24, 1, 1), 1013),
        "diagnostic__air_density_kg_m3": np.full((24, 1, 1), rho),
        "static_elevation_m": np.zeros((1, 1)),
    }
    summary = {"timestamps": np.array([0]), "channel_names": np.array(channels), "dynamic": values.mean(axis=0, keepdims=True)}
    anchors = {name: value.copy() for name, value in summary.items()}
    anchors["dynamic"][0, 5] = 0  # the old vector-only anchor is not a scalar-speed constraint
    anchors["dynamic"][0, 1] = 0.8  # RH from the prior is not the realized daily RH
    config = WorldConfig()
    return hourly, summary, anchors, replace(config, world=replace(config.world, height=1, width=1))


def test_reversing_wind_and_nonlinear_daily_diagnostics_are_valid():
    hourly, summary, anchors, config = _turning_weather()
    checks = Checks()
    check_weather_contracts(checks, anchors, hourly, config, summary)
    assert all(row["passed"] for row in checks.rows), [row for row in checks.rows if not row["passed"]]
    assert "declared_anchor_humidity" not in [row["name"] for row in checks.rows]


def test_missing_daily_summary_is_not_silently_replaced_by_anchors():
    hourly, _, anchors, config = _turning_weather()
    with pytest.raises(ValueError, match="hourly_daily_summary"):
        check_weather_contracts(Checks(), anchors, hourly, config)


@pytest.mark.parametrize("corruption,failed_name", [("temperature_anchor", "declared_anchor_temperature"), ("summary_speed", "daily_hourly_wind_speed"), ("density", "moist_air_density_identity"), ("p0", "hydrostatic_pressure_from_primitives")])
def test_weather_validation_localizes_inconsistent_inputs(corruption, failed_name):
    hourly, summary, anchors, config = _turning_weather()
    if corruption == "temperature_anchor":
        anchors["dynamic"][0, 0] += 1
    elif corruption == "summary_speed":
        summary["dynamic"][0, 5] += 1
    elif corruption == "p0":
        hourly["diagnostic__sea_level_pressure_hpa"][0, 0, 0] += 20
    else:
        hourly["diagnostic__air_density_kg_m3"][0, 0, 0] += 0.2
    checks = Checks()
    check_weather_contracts(checks, anchors, hourly, config, summary)
    assert any(row["name"] == failed_name and not row["passed"] for row in checks.rows)
