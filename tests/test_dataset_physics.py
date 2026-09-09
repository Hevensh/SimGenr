"""Dataset source/dispatch separation, provenance and temporal-axis tests."""
from pathlib import Path
import json
import shutil
import uuid

import numpy as np
import pytest

from world_generator.dataset.builder import (
    SCHEMA_VERSION, STATIC_UNITS, _exogenous_payload, _has_physical_units, _world_provenance, dataset_schema,
)
from world_generator.dataset.loader import SimGenrDataset, TemporalWindowDataset, _split_operation, collate_multimodal


def _source():
    return {
        "timestamps": np.arange(4), "bus_ids": np.asarray([10, 20, 30, 40]),
        "bus_kinds": np.asarray(["load_bus", "pv_bus", "thermal_bus", "transit_bus"]),
        "p_load_mw": np.tile([5.0, 0.0, 0.0, 0.0], (4, 1)),
        "p_gen_available_mw": np.tile([0.0, 7.0, 9.0, 0.0], (4, 1)),
    }


def test_exogenous_mapping_preserves_original_injections_and_renewable_mask():
    source = _source()
    mapped = _exogenous_payload(source, np.asarray([30, 99, 20, 10]), np.arange(4))
    np.testing.assert_array_equal(mapped["exogenous_p_load_mw"], np.tile([0, 0, 0, 5], (4, 1)))
    np.testing.assert_array_equal(mapped["exogenous_p_gen_available_mw"], np.tile([9, 0, 7, 0], (4, 1)))
    np.testing.assert_array_equal(mapped["exogenous_p_renewable_available_mw"], np.tile([0, 0, 7, 0], (4, 1)))
    np.testing.assert_array_equal(mapped["exogenous_bus_present"], [True, False, True, True])
    # A zero-injection transit bus may disappear, but energy cannot disappear.
    np.testing.assert_array_equal(mapped["exogenous_p_load_mw"].sum(axis=1), source["p_load_mw"].sum(axis=1))


@pytest.mark.parametrize("case", ["missing_active_bus", "duplicate_bus", "time_mismatch", "negative", "shape"])
def test_exogenous_mapping_rejects_lossy_or_invalid_sources(case):
    source = _source()
    ids = source["bus_ids"].copy()
    timestamps = source["timestamps"].copy()
    if case == "missing_active_bus":
        ids = ids[1:]
    elif case == "duplicate_bus":
        ids[1] = ids[0]
    elif case == "time_mismatch":
        timestamps += 1
    elif case == "negative":
        source["p_load_mw"][0, 0] = -1.0
    elif case == "shape":
        source["p_load_mw"] = np.zeros((4, 3))
    with pytest.raises(ValueError):
        _exogenous_payload(source, ids, timestamps)


def test_provenance_does_not_rebrand_legacy_output_as_physics_v3():
    legacy = _world_provenance({})
    assert legacy["generator_version"] == "legacy_unspecified"
    source = {"generator_version": "physics_v3", "scenario_semantics": "synthetic_realization_with_perfect_foresight_planning", "time_convention": "local_solar_time_365_day_climatology"}
    current = _world_provenance(source)
    assert all(current[key] == value for key, value in source.items())
    assert current["dispatch_semantics"] == "perfect_foresight_dispatch"
    assert SCHEMA_VERSION == "0.8.0"
    assert STATIC_UNITS["population_density"] == "persons/km2"
    assert STATIC_UNITS["water_depth"] == STATIC_UNITS["hydrology_elevation"] == "m"
    assert "exogenous_p_load_mw" in dataset_schema()["sample_file"]["operation"]


def test_physics_v4_retains_population_units_and_requires_exogenous_data():
    assert _has_physical_units({"generator_version": "physics_v4"})
    assert _has_physical_units({"generator_version": "physics_v3"})
    assert not _has_physical_units({})
    assert not _has_physical_units({"generator_version": "unknown_future"})


def test_fractional_ids_cannot_alias_when_mapping_source_energy():
    source = _source()
    source["bus_ids"] = np.array([10.1, 10.2, 30, 40])
    with pytest.raises(ValueError, match="integer"):
        _exogenous_payload(source, source["bus_ids"].copy(), source["timestamps"])


def test_temporal_split_uses_field_semantics_when_nodes_or_sites_equal_hours():
    operation = {
        "timestamps": np.arange(4), "bus_ids": np.arange(4),
        "exogenous_p_load_mw": np.arange(16).reshape(4, 4),
        "exogenous_bus_present": np.asarray([True, True, False, True]),
        "site_power_capacity_mw": np.ones(4),
        "thermal_bus_ids": np.arange(4),
        "charge_mw": np.ones((4, 4)), "soc_mwh": np.arange(20).reshape(5, 4),
    }
    history, future, static = _split_operation(operation, 0, 2, 4, 4)
    assert history["exogenous_p_load_mw"].shape == future["exogenous_p_load_mw"].shape == (2, 4)
    assert history["soc_mwh"].shape == future["soc_mwh"].shape == (3, 4)
    for name in ["bus_ids", "site_power_capacity_mw", "thermal_bus_ids", "exogenous_bus_present"]:
        assert name in static and name not in history
        np.testing.assert_array_equal(static[name], operation[name])


def test_temporal_split_rejects_misaligned_exogenous_hours():
    with pytest.raises(ValueError, match="weather time axis"):
        _split_operation({"exogenous_p_load_mw": np.zeros((3, 2))}, 0, 2, 4, 4)


def test_loader_roundtrip_new_fields_and_legacy_compatibility():
    # All test files stay under the authorized repository, independent of TEMP.
    test_root = Path(__file__).resolve().parent
    root = test_root / ("dataset_test_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        (root / "samples").mkdir()
        source = _source()
        exogenous = _exogenous_payload(source, source["bus_ids"], np.arange(4))
        entries = []
        for seed in (1, 2):
            operation = {"timestamps": np.arange(4), "bus_ids": source["bus_ids"], "node_dynamic": np.zeros((4, 10, 4), dtype=np.float32)}
            if seed == 1:
                operation.update(exogenous)
            payload = {f"operation__{key}": value for key, value in operation.items()}
            if seed == 1:
                payload["land__land_use_fraction_energy_reserve"] = np.full((4, 4), .25)
                payload["land__energy_project_area_by_cell_km2"] = np.zeros((4, 4, 4))
            payload.update({
                "dynamic__timestamps": np.arange(4), "dynamic__weather": np.zeros((4, 9, 1, 1)),
                "dynamic__weather_class": np.zeros((4, 1, 1)), "graph__node_id": source["bus_ids"],
                "metadata_json": np.asarray(json.dumps(_world_provenance({"generator_version": "physics_v3"}))),
                "config_yaml": np.asarray("seed: 1"),
            })
            filename = f"seed{seed}.npz"
            np.savez_compressed(root / "samples" / filename, **payload)
            entries.append({"sample_id": f"world_seed{seed}", "seed": seed, "hours": 4, "file": {"name": filename}})
        (root / "manifest.json").write_text(json.dumps({"schema_version": "0.6.0", "samples": entries}), encoding="utf-8")
        worlds = SimGenrDataset(root)
        np.testing.assert_array_equal(worlds[0]["operation"]["exogenous_p_load_mw"], source["p_load_mw"])
        assert "exogenous_p_load_mw" not in worlds[1]["operation"]
        windows = TemporalWindowDataset(worlds, history_hours=2, forecast_hours=2, stride=2)
        assert windows[0]["future"]["operation"]["exogenous_p_load_mw"].shape == (2, 4)
        assert windows[0]["operation_static"]["exogenous_bus_present"].shape == (4,)
        assert worlds[0]["metadata"]["generator_version"] == "physics_v3"
        assert worlds[1]["land"] == {}
        assert windows[0]["land"] is worlds[0]["land"]
        assert windows[0]["land"]["energy_project_area_by_cell_km2"].shape == (4, 4, 4)
        batch = collate_multimodal([worlds[0], worlds[1]])
        assert isinstance(batch["operation"], list)  # Missing old fields are never fabricated.
    finally:
        assert root.resolve().parent == test_root
        shutil.rmtree(root)
