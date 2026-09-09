"""G diagnostics preserve constraints while distinguishing missing physical coverage."""
import json
from dataclasses import replace
import numpy as np
import pytest

from scripts.validation_checks import Checks, input_error_report
from scripts.validate_world_physics import Checks as CompatibleChecks, _markdown
from world_generator.core.config import ValidationConfig, WorldConfig, dump_config_snapshot, load_world_config


def test_four_states_and_supported_scope_preserve_legacy_passed_values():
    checks = Checks()
    checks.equal('mass',np.array([0.]),1e-5)
    checks.status('frequency','UNSUPPORTED','No frequency model')
    checks.equal('empty_storage',np.empty((4,0)),1e-5)
    assert [row['status'] for row in checks.rows] == ['PASS','UNSUPPORTED','NOT_RUN']
    assert [row['passed'] for row in checks.rows] == [True,None,None]
    assert checks.summary()['passed'] is True
    checks.upper('capacity',np.array([2.]),1.,0)
    assert checks.summary()['status'] == 'FAIL' and checks.rows[-1]['passed'] is False
    assert CompatibleChecks is Checks


@pytest.mark.parametrize('method',['equal','upper','condition'])
def test_empty_objects_are_not_exercised_passes(method):
    checks = Checks()
    if method == 'equal': checks.equal('empty',np.empty((2,0)),0)
    elif method == 'upper': checks.upper('empty',np.empty((2,0)),1,0)
    else: checks.condition('empty',np.array([],dtype=bool))
    row = checks.rows[0]
    assert row['status'] == 'NOT_RUN' and row['passed'] is None
    assert row['max_residual'] is None and row['location'] is None
    assert checks.summary()['passed'] is None


def test_raw_max_and_relative_tolerance_violation_keep_their_own_locations():
    checks = Checks(max_failure_locations=0)
    with checks.context(stage='D_water',fields=['soil_storage_mm','infiltration_mm'],axes=('time','cell'),timestamps=[10,11],time_support='hour_interval'):
        checks.equal('budget',np.array([[100.,0.],[0.,1.]]),.1,relative_tolerance=.01,scale=np.array([[20000.,0.],[0.,1.]]))
    row = checks.rows[0]
    assert row['max_residual'] == 100 and row['raw_max_location']['array_index'] == [0,0]
    assert row['max_violation'] == pytest.approx(.89)
    assert row['max_violation_location']['array_index'] == row['location']['array_index'] == [1,1]
    assert row['raw_max_time']['value'] == 10 and row['time']['value'] == 11
    assert row['failure_locations'] == []


def test_upper_reports_raw_margin_separately_from_excess_over_tolerance():
    checks = Checks()
    checks.upper('bound',[-10.,1.1],1.,.05)
    row = checks.rows[0]
    assert row['max_residual'] == 11 and row['max_error'] == pytest.approx(.1)
    assert row['max_violation'] == pytest.approx(.05)
    assert row['raw_max_location']['array_index'] == [0]
    assert row['location']['array_index'] == [1]


def test_failure_list_is_bounded_but_main_worst_is_never_lost():
    checks = Checks(max_failure_locations=2)
    checks.equal('many',[1.,4.,3.],0)
    assert [entry['location']['array_index'] for entry in checks.rows[0]['failure_locations']] == [[1],[2]]
    assert checks.rows[0]['location']['array_index'] == [1]


def test_equal_axes_lengths_do_not_turn_static_assets_into_time_series():
    checks = Checks()
    with checks.context(timestamps=np.array([100,101,102]),fields=['capacity'],axes=('bus',),time_support='fixed_asset'):
        checks.equal('static',np.array([0.,2.,0.]),0)
    row = checks.rows[0]
    assert row['location']['array_index'] == [1]
    assert row['time'] == {'status':'not_applicable','support':'fixed_asset','value':None}


def test_aggregate_without_explicit_axis_does_not_guess_time():
    checks = Checks()
    checks.equal('period_energy',np.array([1.,0.]),0,timestamps=[100,101],time_support='whole_period_per_bus')
    assert checks.rows[0]['time']['status'] == 'not_applicable'


def test_declared_time_axis_must_match_residual_rank_and_length():
    checks = Checks()
    with pytest.raises(ValueError,match='timestamps'):
        checks.equal('time',np.zeros((2,3)),0,axes=('time','bus'),timestamps=[1,2,3])
    with pytest.raises(ValueError,match='axes'):
        checks.equal('rank',np.zeros((2,3)),0,axes=('time',),timestamps=[1,2])


def test_nonfinite_residual_fails_with_location_and_json_safe_diagnostic():
    checks = Checks()
    checks.equal('nan',np.array([0.,np.nan]),0)
    row = checks.rows[0]
    assert row['status'] == 'FAIL' and row['nonfinite_count'] == 1
    assert row['location']['array_index'] == [1]
    json.dumps(row,allow_nan=False)


@pytest.mark.parametrize('tolerance,relative,scale',[(np.inf,0,1),(-1,0,1),(0,-1,1),(0,1,np.nan)])
def test_diagnostics_cannot_weaken_invalid_tolerances(tolerance,relative,scale):
    with pytest.raises(ValueError): Checks().equal('bad',[0],tolerance,relative_tolerance=relative,scale=scale)


def test_context_restores_and_required_metadata_keys_exist_on_every_state():
    checks = Checks()
    with checks.context(stage='water',fields=['rain'],relation_class='P',engineering_simplification='bucket'):
        checks.equal('water',[0],0)
    checks.status('not_supported','UNSUPPORTED')
    assert checks.rows[0]['stage'] == 'water' and checks.rows[1]['stage'] == 'unspecified'
    required = {'name','status','passed','max_residual','location','time','fields','stage','relation_class','engineering_simplification'}
    assert all(required.issubset(row) for row in checks.rows)


@pytest.mark.parametrize('value',[np.nan,2,0,np.array([True,np.nan]),np.array([1,0])])
def test_condition_rejects_numeric_truthiness_and_nan(value):
    with pytest.raises(ValueError,match='booleans'):
        Checks().condition('not_boolean',value)


def test_input_error_is_distinct_from_physical_infeasibility_and_markdown_status():
    report = input_error_report('bad_world',ValueError('Missing terminal SOC'))
    assert report['status'] == 'FAIL' and report['error_category'] == 'INPUT_OR_GENERATION_ERROR'
    assert report['checks'][0]['error_category'] == 'INPUT_OR_GENERATION_ERROR'
    assert 'INPUT_OR_GENERATION_ERROR' in _markdown(report)


def test_failed_generation_marker_prevents_validation_of_stale_artifacts(tmp_path,monkeypatch):
    import scripts.validate_world_physics as validator
    failure = {'schema_version':'generation_failure_v1','status':'FAIL','category':'PHYSICALLY_INFEASIBLE','stage':'Stage14','message':'Latest solver run failed'}
    (tmp_path/'generation_failure.json').write_text(json.dumps(failure),encoding='utf-8')
    def forbidden(*args,**kwargs): raise AssertionError('Stale files must not be opened')
    monkeypatch.setattr(validator,'load_world_config',forbidden)
    monkeypatch.setattr(validator,'_npz',forbidden)
    report = validator.validate_world(tmp_path)
    assert report['status'] == 'FAIL' and report['passed'] is False
    assert report['error_category'] == 'PHYSICALLY_INFEASIBLE'
    assert report['checks'][0]['name'] == 'latest_generation_failed'


def test_failed_generation_marker_prevents_packaging_even_before_output_creation(tmp_path):
    from world_generator.dataset.builder import build_dataset,package_world
    world = tmp_path/'world'; world.mkdir()
    (world/'generation_failure.json').write_text('{}',encoding='utf-8')
    output = tmp_path/'dataset'
    with pytest.raises(ValueError,match='generation_failure.json'):
        build_dataset([world],output,show_progress=False)
    assert not output.exists()
    with pytest.raises(ValueError,match='generation_failure.json'):
        package_world(world,tmp_path/'samples')


@pytest.mark.parametrize('value',[-1,1001,.5,True])
def test_validation_config_rejects_non_diagnostic_or_invalid_budget(value):
    with pytest.raises(ValueError): ValidationConfig(max_failure_locations=value)


def test_diagnostic_config_roundtrips_without_changing_physical_settings(tmp_path):
    original = WorldConfig()
    changed = replace(original,validation=ValidationConfig(max_failure_locations=0))
    a,b = original.to_dict(),changed.to_dict()
    a.pop('validation'); b.pop('validation')
    assert a == b
    path = tmp_path/'config.yaml'; dump_config_snapshot(changed,path)
    assert load_world_config(path) == changed
