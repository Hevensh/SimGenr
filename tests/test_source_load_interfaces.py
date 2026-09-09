"""E source/load appendix integrity, original node identity and window support."""
from dataclasses import replace
import json

import numpy as np
import pytest

from world_generator.core.datatypes import SourceLoadForecastStore
from world_generator.core.source_load_contracts import SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY, source_load_field_schema
from world_generator.dataset.loader import SimGenrDataset, TemporalWindowDataset, _split_source_load
from world_generator.operation.stage_cache import load_source_load_checkpoint
from world_generator.core.output_layout import WorldDataLayout


def _store():
    count = hours = 4
    z = np.zeros((hours, count))
    kinds = ("load_bus", "wind_bus", "pv_bus", "thermal_bus")
    diagnostics = {name: z.copy() for name in SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY}
    diagnostics["hub_wind_speed_mps"][:, 1] = 5
    diagnostics["wind_air_density_kg_m3"][:, 1] = 1.2
    diagnostics["pv_poa_w_m2"][:, 2] = 500
    diagnostics["pv_module_temperature_c"][:, 2] = 25
    diagnostics["load_effective_temperature_c"][:, 0] = np.arange(16, 20)
    metadata = {"schema_version": "source_load_v1", "mode": "exogenous_realization", "weather_grid_shape": [4, 2],
                "diagnostic_applicability": {name: list(values) for name, values in SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY.items()}}
    return SourceLoadForecastStore(np.arange(4), np.array([10,20,30,40]), kinds,
        np.tile([8.,0,0,0],(4,1)), np.tile([0.,10,15,20],(4,1)), np.tile([0.,5,7.5,10],(4,1)), z,
        ("load", "wind", "pv", "thermal"), nameplate_capacity_mw=np.array([10.,20,30,40]), reference_load_mw=np.array([10.,0,0,0]),
        initial_effective_temperature_c=np.array([15.,0,0,0]), weather_sample_row=np.arange(4), weather_sample_col=np.zeros(4,dtype=int), diagnostics=diagnostics, metadata=metadata)


def test_energy_ledgers_and_capacity_factor_have_explicit_applicability():
    arrays = _store().as_arrays()
    np.testing.assert_allclose(arrays["requested_load_energy_mwh"], arrays["p_load_mw"])
    np.testing.assert_allclose(arrays["period_requested_load_energy_mwh"], [32,0,0,0])
    np.testing.assert_array_equal(arrays["capacity_factor_valid"], [False,True,True,True])
    np.testing.assert_allclose(arrays["available_capacity_factor"], np.tile([0,.5,.5,.5],(4,1)))
    assert source_load_field_schema()["diag__load_effective_temperature_c"]["time_kind"] == "interval_end_state"
    restored = SourceLoadForecastStore.from_arrays(arrays)
    np.testing.assert_array_equal(restored.as_arrays()["period_planned_generation_energy_mwh"], arrays["period_planned_generation_energy_mwh"])


@pytest.mark.parametrize("case", ["missing_attribute", "wrong_static_shape", "fractional_sample", "outside_grid", "missing_diagnostic", "nonapplicable_diagnostic", "nonfinite", "applicability"])
def test_modern_store_rejects_ambiguous_or_invalid_diagnostics(case):
    store = _store()
    if case == "missing_attribute": store = replace(store, nameplate_capacity_mw=None)
    elif case == "wrong_static_shape": store = replace(store, reference_load_mw=np.zeros((4,4)))
    elif case == "fractional_sample": store = replace(store, weather_sample_row=np.array([.5,1,2,3]))
    elif case == "outside_grid": store = replace(store, weather_sample_col=np.array([2,0,0,0]))
    elif case == "missing_diagnostic": del store.diagnostics["load_log_residual"]
    elif case == "nonapplicable_diagnostic": store.diagnostics["pv_poa_w_m2"][0,0] = 1
    elif case == "nonfinite": store.diagnostics["load_log_residual"][0,0] = np.nan
    else: del store.metadata["diagnostic_applicability"]
    with pytest.raises(ValueError): store.as_arrays()


@pytest.mark.parametrize("case", ["energy", "period_energy", "cf", "valid_mask", "missing", "bounds", "schema", "marker_only"])
def test_reading_rejects_corrupted_derived_energy_or_partial_appendix(case):
    arrays = _store().as_arrays()
    if case == "energy": arrays["requested_load_energy_mwh"][0,0] += 1
    elif case == "period_energy": arrays["period_requested_load_energy_mwh"][0] += 1
    elif case == "cf": arrays["available_capacity_factor"][0,1] += .1
    elif case == "valid_mask": arrays["capacity_factor_valid"][0] = True
    elif case == "missing": del arrays["diag__load_effective_temperature_c"]
    elif case == "bounds": arrays["time_bounds_hours"][0,1] = 2
    elif case == "schema": arrays["source_load_field_schema_json"] = np.asarray("{}")
    else:
        for name in source_load_field_schema(): arrays.pop(name)
    with pytest.raises(ValueError): SourceLoadForecastStore.from_arrays(arrays)


def test_legacy_stage14_remains_readable_without_invented_capacity_or_diagnostics():
    store = _store()
    legacy = SourceLoadForecastStore(store.timestamps,store.bus_ids,store.bus_kinds,store.p_load_mw,store.p_gen_available_mw,store.p_gen_scheduled_mw,store.q_load_mvar,store.source_channels)
    arrays = legacy.as_arrays()
    assert "nameplate_capacity_mw" not in arrays and "source_load_schema_version" not in arrays
    restored = SourceLoadForecastStore.from_arrays(arrays)
    assert restored.diagnostics == {} and restored.nameplate_capacity_mw is None


def test_window_recomputes_period_energy_and_thermal_initial_state_when_nodes_equal_hours():
    arrays = _store().as_arrays()
    history,future,shared = _split_source_load(arrays,1,2,4,4)
    assert shared["nameplate_capacity_mw"].shape == (4,)
    assert shared["nameplate_capacity_mw"] is arrays["nameplate_capacity_mw"]
    np.testing.assert_allclose(history["period_requested_load_energy_mwh"],[8,0,0,0])
    np.testing.assert_allclose(future["period_requested_load_energy_mwh"],[16,0,0,0])
    np.testing.assert_array_equal(history["initial_effective_temperature_c"],arrays["diag__load_effective_temperature_c"][0])
    np.testing.assert_array_equal(future["initial_effective_temperature_c"],arrays["diag__load_effective_temperature_c"][1])
    assert future["diag__load_effective_temperature_c"].shape == (2,4)


def test_later_realization_cannot_change_earlier_window_assets_or_energy_history():
    original = _store().as_arrays()
    altered = {name: value.copy() for name, value in original.items()}
    altered['p_load_mw'][2:,0] += 100
    altered['requested_load_energy_mwh'][2:,0] += 100
    altered['period_requested_load_energy_mwh'][0] += 200
    altered['diag__load_effective_temperature_c'][2:,0] += 20
    h1,_,s1 = _split_source_load(original,0,2,4,4)
    h2,_,s2 = _split_source_load(altered,0,2,4,4)
    for name in h1: np.testing.assert_array_equal(h1[name],h2[name])
    for name in s1: np.testing.assert_array_equal(s1[name],s2[name])
    assert not any(name.startswith('period_') for name in s1)
    assert 'initial_effective_temperature_c' not in s1


def test_declared_checkpoint_requires_appendix_and_matches_weather(tmp_path):
    layout=WorldDataLayout(tmp_path/'data'); layout.create()
    layout.metadata.write_text(json.dumps({"source_load_appendix":{"schema_version":"source_load_v1","mode":"exogenous_realization","artifact":"stage_11_operation/source_load_forecast.npz"}}),encoding='utf-8')
    np.savez(layout.operation/'source_load_forecast.npz',**_store().as_arrays())
    assert load_source_load_checkpoint(tmp_path,expected_timestamps=np.arange(4),expected_grid_shape=(4,2)).nameplate_capacity_mw.shape==(4,)
    with pytest.raises(ValueError,match='timestamps'): load_source_load_checkpoint(tmp_path,expected_timestamps=np.arange(4)+1)
    with pytest.raises(ValueError,match='grid'): load_source_load_checkpoint(tmp_path,expected_grid_shape=(2,4))


@pytest.mark.parametrize("from_stage", [13, 14])
def test_current_cli_rejects_old_physics_cache_without_source_marker(tmp_path, from_stage):
    from scripts.generate_static_world import _validate_cached_physics, GENERATOR_VERSION
    layout = WorldDataLayout(tmp_path/'data'); layout.create()
    layout.metadata.write_text(json.dumps({'generator_version':GENERATOR_VERSION,'land_accounting_version':'land_use_v1'}), encoding='utf-8')
    layout.config_snapshot.write_text('seed: 42', encoding='utf-8')
    with pytest.raises(ValueError, match='source_load_v1.*from-stage 1'):
        _validate_cached_physics(tmp_path, object(), from_stage)


def test_dataset_roundtrip_keeps_original_bus_ids_and_window_diagnostics(tmp_path):
    (tmp_path/'samples').mkdir()
    payload={f'source_load__{name}':value for name,value in _store().as_arrays().items()}
    payload.update(dynamic__timestamps=np.arange(4),dynamic__weather=np.zeros((4,9,4,2)),dynamic__weather_class=np.zeros((4,4,2)),
                   graph__node_id=np.array([30,10,40,20]),metadata_json=np.asarray(json.dumps({'source_load':{'mode':'exogenous_realization'}})),config_yaml=np.asarray('seed: 1'))
    np.savez(tmp_path/'samples/seed1.npz',**payload)
    (tmp_path/'manifest.json').write_text(json.dumps({'samples':[{'sample_id':'world_seed1','seed':1,'hours':4,'file':{'name':'seed1.npz'}}]}),encoding='utf-8')
    worlds=SimGenrDataset(tmp_path); windows=TemporalWindowDataset(worlds,history_hours=2,forecast_hours=2)
    window=windows[0]
    np.testing.assert_array_equal(window['source_load_static']['bus_ids'],[10,20,30,40])
    np.testing.assert_array_equal(window['graph']['node_id'],[30,10,40,20])
    assert window['future']['source_load']['available_capacity_factor'].shape==(2,4)
    assert window['source_load_static']['capacity_factor_valid'].shape==(4,)
