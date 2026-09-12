"""F persistent IDs, physical account appendices and explicit window support."""
from dataclasses import fields, replace
import json
import numpy as np
import pytest

from world_generator.core.datatypes import PowerFlowStore, StorageDispatchStore
from world_generator.core.operation_contracts import operation_field_schema, operation_store_field_schema
from world_generator.dataset.loader import SimGenrDataset, TemporalWindowDataset, _split_operation_detail
from world_generator.dataset.builder import _world_provenance
from world_generator.operation.asset_planning import asset_snapshot_sha256
from world_generator.operation.stage_cache import load_asset_boundary_checkpoint


def _storage():
    # T=N=E=S=K deliberately collides: semantic names must determine slicing.
    size = 4
    schema = operation_store_field_schema("storage_dispatch")
    base = {}
    for item in fields(StorageDispatchStore):
        if item.name in {"operation_arrays","operation_metadata"}: continue
        dimensions = schema[item.name]["shape"]
        shape = () if dimensions == "scalar" else tuple(5 if dim == "T+1" else size for dim in dimensions.split(","))
        base[item.name] = np.zeros(shape)
    for name in ("timestamps","site_ids","site_bus_ids","branch_ids","thermal_bus_ids"):
        base[name] = np.arange(size,dtype=np.int32)
    base["site_power_capacity_mw"][:] = 10
    base["site_energy_capacity_mwh"][:] = 20
    base["soc_mwh"][:] = np.arange(5)[:,None]+5
    base["discharge_mw"][:] = np.arange(4)[:,None]+2
    base["emergency_discharge_mw"][:] = 1
    base["charge_mw"][:] = .5
    append = {}
    for name,spec in operation_field_schema("storage_dispatch").items():
        append[name] = np.zeros(tuple(size for dim in spec["shape"].split(",")))
    append["bus_ids"] = np.arange(size,dtype=np.int32)
    append["bus_kinds"] = np.asarray(["thermal_bus"]*size)
    append["branch_in_service"] = np.ones((size,size),dtype=bool)
    append["initial_soc_mwh"][:] = 5
    append["thermal_dispatch_mw"][:] = np.arange(4)[:,None]+2
    append["thermal_installed_capacity_mw"][:] = 10
    append["thermal_land_limit_mw"][:] = 20
    return StorageDispatchStore(**base,operation_arrays=append,operation_metadata={"schema_version":"operation_v1","store_kind":"storage_dispatch","duration_hours":1.0})


def _power():
    z = np.zeros((4,4))
    arrays = {name:(np.ones_like(z,dtype=bool) if name == "branch_in_service" else z.copy()) for name in operation_field_schema("power_flow")}
    return PowerFlowStore(np.arange(4),np.arange(4),np.arange(4),z,z,z,z,z,z,z,z,0,
        operation_arrays=arrays,operation_metadata={"schema_version":"operation_v1","store_kind":"power_flow","duration_hours":1.0})


def _assets():
    ids = np.arange(4,dtype=np.int32)
    arrays = {"bus_ids":ids,"bus_nameplate_capacity_mw":np.full(4,10.),"branch_ids":ids,
        "branch_capacity_mva":np.full(4,20.),"electrical_buses":np.column_stack((ids,np.ones(4))),
        "electrical_branches":np.column_stack((ids,np.ones(4))),"storage_site_ids":ids,"storage_bus_ids":ids,
        "storage_power_mw":np.full(4,10.),"storage_energy_mwh":np.full(4,20.),"storage_initial_soc_mwh":np.full(4,5.),
        "thermal_bus_ids":ids,"thermal_land_capacity_upper_bound_mw":np.full(4,20.)}
    metadata = {"schema_version":"asset_planning_v1","mode":"fixed_assets","initial_assets_sha256":asset_snapshot_sha256(arrays),
        "frozen_assets_sha256":asset_snapshot_sha256(arrays),"planning_input_hashes":{},"planning_time_bounds_hours":None,
        "operation_time_bounds_hours":[0,4],"planning_input_source":"configured_assets","dispatch_foresight":"full_operation_window_perfect_foresight_dispatch"}
    return arrays,metadata


def test_runtime_roundtrip_preserves_ids_outages_and_soc_boundaries():
    store = _storage()
    store.operation_arrays["branch_in_service"][1,2] = False
    restored = StorageDispatchStore.from_arrays(store.as_arrays())
    assert not restored.operation_arrays["branch_in_service"][1,2]
    np.testing.assert_array_equal(restored.soc_mwh,store.soc_mwh)
    np.testing.assert_array_equal(restored.operation_arrays["bus_ids"],np.arange(4))
    PowerFlowStore.from_arrays(_power().as_arrays())


@pytest.mark.parametrize("case",["missing","unknown","soc_terminal","soc_initial","bounds","runtime_shape","island_fraction","duplicate_id","nan","negative_reserve","label"])
def test_modern_operation_rejects_incomplete_or_impossible_interfaces(case):
    data = _storage().as_arrays()
    if case == "missing": del data["op__generator_dispatch_mw"]
    elif case == "unknown": data["op__mystery"] = np.zeros(4)
    elif case == "soc_terminal": data["soc_mwh"] = data["soc_mwh"][:-1]
    elif case == "soc_initial": data["op__initial_soc_mwh"][0] += 1
    elif case == "bounds": data["operation_time_bounds_hours"][0,1] = 2
    elif case == "runtime_shape": data["op__branch_in_service"] = np.ones((4,3))
    elif case == "island_fraction": data["op__island_id"][0,0] = .5
    elif case == "duplicate_id": data["op__bus_ids"][1] = 0
    elif case == "nan": data["op__generator_dispatch_mw"][0,0] = np.nan
    elif case == "negative_reserve": data["op__reserve_shortfall_mw"][0,0] = -1
    else: data["op__bus_kinds"] = np.asarray(["mystery"]*4)
    with pytest.raises(ValueError): StorageDispatchStore.from_arrays(data)


def test_legacy_operation_read_does_not_invent_frozen_assets_or_runtime_state():
    legacy = replace(_storage(),operation_arrays={},operation_metadata={})
    restored = StorageDispatchStore.from_arrays(legacy.as_arrays())
    assert restored.operation_arrays == {} and restored.operation_metadata == {}
    assert "operation_schema_version" not in restored.as_arrays()


def test_window_explicit_time_support_and_prior_power_with_equal_axes():
    data = _storage().as_arrays()
    history,future,shared = _split_operation_detail(data,1,2,4,4,"storage_dispatch")
    assert shared["site_power_capacity_mw"].shape == (4,)
    assert shared["op__thermal_land_limit_mw"].shape == (4,)
    assert history["soc_mwh"].shape == (2,4) and future["soc_mwh"].shape == (3,4)
    np.testing.assert_array_equal(history["op__initial_soc_mwh"],np.full(4,6.))
    np.testing.assert_array_equal(history["op__previous_storage_net_mw"],np.full(4,1.5))
    np.testing.assert_array_equal(future["op__previous_storage_net_mw"],np.full(4,2.5))
    np.testing.assert_array_equal(future["op__previous_thermal_mw"],np.full(4,3.))
    assert "op__initial_soc_mwh" not in shared


def test_window_previous_net_does_not_count_emergency_discharge_twice():
    data = _storage().as_arrays()
    data["discharge_mw"][:] = 3.0  # Total discharge already includes 1 MW emergency.
    data["emergency_discharge_mw"][:] = 1.0
    data["charge_mw"][:] = .5
    history,future,_ = _split_operation_detail(data,1,2,4,4,"storage_dispatch")
    for block in (history,future):
        np.testing.assert_array_equal(block["op__previous_storage_net_mw"],np.full(4,2.5))


@pytest.mark.parametrize("case",["good","hash","missing","clock","id"])
def test_asset_checkpoint_checks_content_and_information_boundary(tmp_path,case):
    arrays,metadata = _assets()
    directory = tmp_path/'data/planning'; directory.mkdir(parents=True)
    np.savez(directory/'initial_assets.npz',**arrays)
    np.savez(directory/'frozen_assets.npz',**arrays)
    if case == "hash": metadata["frozen_assets_sha256"] = '0'*64
    elif case == "missing": del metadata["planning_input_source"]
    elif case == "clock": metadata["operation_time_bounds_hours"] = [1,5]
    (directory/'asset_boundary.json').write_text(json.dumps(metadata),encoding='utf-8')
    kwargs = dict(expected_timestamps=np.arange(4),expected_bus_ids=np.arange(4),expected_branch_ids=np.arange(4))
    if case == "id": kwargs["expected_bus_ids"] = np.arange(4)+10
    if case == "good": assert load_asset_boundary_checkpoint(tmp_path,**kwargs)["metadata"]["mode"] == "fixed_assets"
    else:
        with pytest.raises(ValueError): load_asset_boundary_checkpoint(tmp_path,**kwargs)


def test_dataset_complete_f_groups_roundtrip_and_windows(tmp_path):
    arrays,boundary = _assets()
    payload = {f'operation_detail__{name}':value for name,value in _storage().as_arrays().items()}
    payload.update({f'power_flow_detail__{name}':value for name,value in _power().as_arrays().items()})
    for group in ('initial_assets','frozen_assets'):
        payload.update({f'asset_planning__{group}__{name}':value for name,value in arrays.items()})
    payload['asset_planning__boundary_metadata_json'] = np.asarray(json.dumps(boundary))
    payload.update(dynamic__timestamps=np.arange(4),dynamic__weather=np.zeros((4,9,4,4)),dynamic__weather_class=np.zeros((4,4,4)),
        metadata_json=np.asarray(json.dumps({'operation_detail':{'mode':'operation_v1'}})),config_yaml=np.asarray('seed: 1'))
    (tmp_path/'samples').mkdir()
    np.savez(tmp_path/'samples/seed1.npz',**payload)
    (tmp_path/'manifest.json').write_text(json.dumps({'samples':[{'sample_id':'seed1','seed':1,'hours':4,'file':{'name':'seed1.npz'}}]}),encoding='utf-8')
    worlds = SimGenrDataset(tmp_path)
    window = TemporalWindowDataset(worlds,history_hours=2,forecast_hours=2)[0]
    assert window['future']['operation_detail']['soc_mwh'].shape == (3,4)
    np.testing.assert_array_equal(window['future']['operation_detail']['op__initial_soc_mwh'],np.full(4,7.))
    assert 'frozen_assets__bus_ids' in window['asset_planning']


@pytest.mark.parametrize("mode",["preplanned","full_window_planning"])
def test_changed_planning_input_is_rejected_even_when_assets_are_intact(tmp_path,mode):
    arrays,metadata = _assets()
    directory = tmp_path/'data/planning'; directory.mkdir(parents=True)
    for label in ('initial_assets','frozen_assets'): np.savez(directory/f'{label}.npz',**arrays)
    metadata.update(mode=mode,planning_time_bounds_hours=[0,4],planning_time_axis='independent_design_climatology_not_operation_clock',planning_information_available_at_operation_hour=0)
    inputs = {'test_forcing':np.arange(4,dtype=float)}
    if mode == 'preplanned':
        paths = {name:directory/'design_input'/f'{name}.npz' for name in ('daily_weather','hourly_weather','source_load_forecast')}
    else:
        paths = {'operation_source_used_as_design':tmp_path/'data/stage_11_operation/source_load_forecast.npz'}
    for name,path in paths.items():
        path.parent.mkdir(parents=True,exist_ok=True); np.savez(path,**inputs)
        metadata['planning_input_hashes'][name] = asset_snapshot_sha256(inputs)
    (directory/'asset_boundary.json').write_text(json.dumps(metadata),encoding='utf-8')
    assert load_asset_boundary_checkpoint(tmp_path,expected_timestamps=np.arange(4))['metadata']['mode'] == mode
    np.savez(next(iter(paths.values())),test_forcing=np.arange(4,dtype=float)+1)
    with pytest.raises(ValueError,match='Planning input hash mismatch'):
        load_asset_boundary_checkpoint(tmp_path,expected_timestamps=np.arange(4))


def test_fixed_asset_provenance_does_not_imply_causal_dispatch():
    provenance = _world_provenance({'asset_planning':{'mode':'fixed_assets','dispatch_foresight':'full_operation_window_perfect_foresight_dispatch'}})
    assert provenance['asset_planning_uses_operation_future'] is False
    assert provenance['dispatch_semantics'] == 'full_operation_window_perfect_foresight_dispatch'
    assert _world_provenance({'asset_planning':{'mode':'full_window_planning'}})['asset_planning_uses_operation_future'] is True
