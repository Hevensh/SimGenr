"""D public configuration, checkpoint integrity and explicit T/T+1 support."""
from dataclasses import replace
import json

import numpy as np
import pytest

from world_generator.core.config import HydrologyConfig, HydrologyDynamicConfig, WorldConfig, WorldGridConfig, _parse_simple_yaml, dump_config_snapshot, load_world_config
from world_generator.core.datatypes import HydrologyTimeSeriesStore, TerrainFeatures
from world_generator.core.contracts import field_contract_document
from world_generator.core.hydrology_contracts import HYDROLOGY_GROUP_UNITS, hydrology_field_schema
from world_generator.core.output_layout import WorldDataLayout
from world_generator.dataset.loader import SimGenrDataset, TemporalWindowDataset, _split_hydrology
from world_generator.operation.stage_cache import load_dynamic_hydrology_checkpoint, validate_weather_checkpoint_time
from world_generator.hydrology.hydrology_generator import generate_hydrology


def _store(hours=4, shape=(4, 2)):
    timestamps = np.arange(24, 24 + hours)
    sizes = {"state": (hours + 1, *shape), "flux": (hours, *shape), "static": shape, "budget": (hours,)}
    groups = [{name: np.zeros(sizes[group]) for name in names} for group, names in HYDROLOGY_GROUP_UNITS.items()]
    groups[0]["soil_storage_mm"][:] = np.arange(hours + 1)[:, None, None]
    groups[2]["routing_receiver_flat_index"][:] = -1
    return HydrologyTimeSeriesStore(timestamps, np.column_stack((timestamps, timestamps + 1)), np.arange(24, 25 + hours), *groups)


def _metadata(enabled=True):
    return {"dynamic_hydrology": {"mode": "bucket_routing_v1" if enabled else "static_only", "schema_version": "hydrology_v1", "artifact": "dynamic_hydrology/hourly_hydrology.npz" if enabled else None}}


def test_default_off_and_complete_configuration_snapshot(tmp_path):
    assert not WorldConfig().hydrology_dynamic.enabled
    config = WorldConfig(hydrology_dynamic=HydrologyDynamicConfig(enabled=True, initial_soil_fraction=.4))
    path = tmp_path / "config.yaml"
    dump_config_snapshot(config, path)
    loaded = load_world_config(path)
    assert loaded.to_dict() == config.to_dict()
    assert len(loaded.hydrology_dynamic.impervious_fraction_by_use) == 9
    parsed = _parse_simple_yaml("hydrology_dynamic:\n  enabled: true\n  impervious_fraction_by_use:\n    residential: 0.65\n    natural: 0.02\nland:\n  protected_fraction: 0.1\n")
    assert parsed["hydrology_dynamic"]["impervious_fraction_by_use"]["natural"] == .02
    assert parsed["land"]["protected_fraction"] == .1


def test_corrupt_weather_interval_declaration_cannot_be_rebuilt_silently():
    weather = {"time_unit": np.asarray("hour"), "timestamps": np.arange(4), "time_bounds_hours": np.column_stack((np.arange(4), np.arange(4) + 1.0))}
    validate_weather_checkpoint_time(weather)
    weather["time_bounds_hours"][1, 1] += .5
    with pytest.raises(ValueError, match="interval bounds"):
        validate_weather_checkpoint_time(weather)


def test_static_catchment_area_is_explicit_and_does_not_relabel_cell_count_as_discharge():
    z = np.zeros((3, 3))
    terrain = TerrainFeatures(np.arange(9).reshape(3, 3).astype(float), *(z.copy() for _ in range(5)))
    grid = WorldGridConfig(height=3, width=3, cell_size_km=2)
    hydro = generate_hydrology(terrain, grid, HydrologyConfig())
    np.testing.assert_allclose(hydro.catchment_area_km2, hydro.flow_accumulation * 4)
    assert hydro.as_maps()["catchment_area_km2"].shape == (3, 3)
    contracts = field_contract_document()
    assert contracts["fields"]["flow_accumulation"]["unit"] == "upstream_cell_count"
    assert contracts["fields"]["catchment_area_km2"]["unit"] == "km2"
    assert contracts["dynamic_hydrology_field_schema"] == hydrology_field_schema()
    infiltration = contracts["fields"]["dynamic_hydrology.flux__infiltration_mm"]
    assert infiltration["generation_relation_type"] == infiltration["relation_class"] == "S"
    assert infiltration["accounting_relation_type"] == "P"
    assert contracts["fields"]["dynamic_hydrology.budget__residual_m3"]["relation_class"] == "P"


@pytest.mark.parametrize("kwargs", [
    {"enabled": "false"}, {"soil_capacity_mm": 0}, {"groundwater_capacity_mm": np.inf},
    {"initial_soil_fraction": -.1}, {"initial_lake_storage_fraction": 1.1},
    {"routing_substeps_per_hour": 1.5}, {"routing_substeps_per_hour": True}, {"routing_substeps_per_hour": 61},
    {"infiltration_capacity_mm_h": -1}, {"pet_latent_energy_fraction": np.nan},
    {"interior_sink_policy": "discard"}, {"impervious_fraction_by_use": {"natural": 0}},
    {"budget_absolute_tolerance_m3": 0, "budget_relative_tolerance": 0},
])
def test_invalid_dynamic_hydrology_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HydrologyDynamicConfig(**kwargs)


@pytest.mark.parametrize("corruption", ["missing_state", "static_as_temporal", "missing_terminal", "time_bounds", "nan", "negative_storage", "unknown_field", "schema_unit"])
def test_store_rejects_lossy_or_ambiguous_water_artifacts(corruption):
    arrays = _store().as_arrays()
    if corruption == "missing_state": del arrays["state__soil_storage_mm"]
    elif corruption == "static_as_temporal": arrays["static__lake_id"] = np.zeros((4, 4, 2))
    elif corruption == "missing_terminal": arrays["state__soil_storage_mm"] = np.zeros((4, 4, 2))
    elif corruption == "time_bounds": arrays["time_bounds_hours"][0, 1] += 1
    elif corruption == "nan": arrays["flux__precipitation_mm"][0, 0, 0] = np.nan
    elif corruption == "negative_storage": arrays["state__groundwater_storage_mm"][0, 0, 0] = -1
    elif corruption == "unknown_field": arrays["state__unknown_storage"] = np.zeros((5, 4, 2))
    else:
        spec = hydrology_field_schema()
        spec["flux__discharge_m3_s"]["unit"] = "m3"
        arrays["field_schema_json"] = np.asarray(json.dumps(spec))
    with pytest.raises(ValueError):
        HydrologyTimeSeriesStore.from_arrays(arrays)


def test_hourly_water_checkpoint_roundtrip_and_disabled_mode(tmp_path):
    layout = WorldDataLayout(tmp_path / "data")
    layout.create()
    layout.metadata.write_text(json.dumps(_metadata(False)), encoding="utf-8")
    assert load_dynamic_hydrology_checkpoint(tmp_path) is None
    layout.metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_dynamic_hydrology_checkpoint(tmp_path)
    layout.dynamic_hydrology.mkdir()
    store = _store()
    np.savez(layout.dynamic_hydrology / "hourly_hydrology.npz", **store.as_arrays())
    loaded = load_dynamic_hydrology_checkpoint(tmp_path, expected_timestamps=store.timestamps, expected_grid_shape=(4, 2))
    np.testing.assert_array_equal(loaded.states["soil_storage_mm"], store.states["soil_storage_mm"])
    with pytest.raises(ValueError, match="timestamps"):
        load_dynamic_hydrology_checkpoint(tmp_path, expected_timestamps=store.timestamps + 1)
    with pytest.raises(ValueError, match="grid"):
        load_dynamic_hydrology_checkpoint(tmp_path, expected_grid_shape=(2, 4))
    layout.metadata.write_text(json.dumps(_metadata(False)), encoding="utf-8")
    with pytest.raises(ValueError, match="Static-only"):
        load_dynamic_hydrology_checkpoint(tmp_path)


def test_window_keeps_static_geometry_when_height_equals_hours_and_includes_terminal_state():
    arrays = _store().as_arrays()
    history, future, static = _split_hydrology(arrays, 0, 2, 4, 4)
    assert history["state__soil_storage_mm"].shape == future["state__soil_storage_mm"].shape == (3, 4, 2)
    assert future["flux__precipitation_mm"].shape == (2, 4, 2)
    np.testing.assert_array_equal(history["state__soil_storage_mm"][-1], future["state__soil_storage_mm"][0])
    assert static["static__lake_id"] is arrays["static__lake_id"]
    assert static["static__lake_id"].shape == (4, 2)
    assert future["state_time_hours"][-1] == 28


def test_dataset_load_and_temporal_water_windows_preserve_complete_schema(tmp_path):
    (tmp_path / "samples").mkdir()
    store = _store()
    payload = {f"hydrology__{name}": value for name, value in store.as_arrays().items()}
    payload.update(dynamic__timestamps=store.timestamps, dynamic__weather=np.zeros((4, 9, 4, 2)), dynamic__weather_class=np.zeros((4, 4, 2)),
                   metadata_json=np.asarray(json.dumps({"hydrology": {"mode": "bucket_routing_v1"}})), config_yaml=np.asarray("seed: 1"))
    np.savez(tmp_path / "samples/seed1.npz", **payload)
    (tmp_path / "manifest.json").write_text(json.dumps({"samples": [{"sample_id": "world_seed1", "seed": 1, "hours": 4, "file": {"name": "seed1.npz"}}]}), encoding="utf-8")
    worlds = SimGenrDataset(tmp_path)
    windows = TemporalWindowDataset(worlds, history_hours=2, forecast_hours=2)
    window = windows[0]
    assert window["future"]["hydrology"]["state__lake_storage_m3"].shape == (3, 4, 2)
    assert window["hydrology_static"]["static__lake_id"].shape == (4, 2)
    assert window["hydrology_static"]["static__lake_id"] is worlds[0]["hydrology"]["static__lake_id"]
    del payload["hydrology__state__soil_storage_mm"]
    np.savez(tmp_path / "samples/seed1.npz", **payload)
    with pytest.raises(ValueError):
        SimGenrDataset(tmp_path)[0]
