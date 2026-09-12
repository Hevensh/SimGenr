"""G executable protocol guards, including real output fault injection."""
from dataclasses import replace
import json

import numpy as np
import pytest

from scripts import validate_mechanism_interventions as protocol


@pytest.fixture(scope="module")
def report():
    return protocol.run_interventions((42,123))


def test_two_seed_protocol_runs_nine_actual_mechanism_pairs(report):
    assert report["counts"] == {"PASS":18,"FAIL":0,"UNSUPPORTED":3,"NOT_RUN":0}, report
    supported=[row for row in report["cases"] if row["seed"] is not None]
    assert {(row["seed"],row["case_id"]) for row in supported} == {
        (seed,case_id) for seed in (42,123) for case_id,_ in protocol.CASES}
    for row in supported:
        assert row["actual_functions"] and row["checks"]
        assert all(check["status"] == "PASS" for check in row["checks"])
        assert row["relation_class"] and set(row["relation_class"]) <= {"P","E","S"}
        assert row["treatment"] and row["expected_relationship"] and row["limitations"]


def test_paired_assets_time_axes_rng_and_quantities_are_explicit(report):
    for row in report["cases"]:
        if row["status"] != "PASS":
            continue
        controls=row["controls"]
        assert controls["static_assets_before_sha256"] == controls["static_assets_after_sha256"]
        assert controls["time_axis_before_sha256"] == controls["time_axis_after_sha256"]
        assert controls["cell_size_km"] == 1
        rng=controls["module_rng"]
        if rng["derived_seed"] is not None:
            assert rng["initial_state_sha256"]
            assert rng["seed_derivation"] == "core.random_state.derive_module_seed"
            assert rng["derived_seed"] == protocol.derive_module_seed(row["seed"],rng["module_name"])
            assert rng["final_state_sha256"][0] == rng["final_state_sha256"][1]
        for side in ("before","after"):
            for metric in row[side].values():
                assert np.isfinite(metric["value"])
                assert metric["unit"] in {"MW","MWh","m/s","m3","K","Wh/m2"}


def test_two_seed_source_noise_is_real_but_each_pair_keeps_it_fixed(report):
    by_seed={row["seed"]:row for row in report["cases"] if row["case_id"] == "temperature_step_memory_load"}
    assert by_seed[42]["before"]["second_day_requested_energy"]["value"] != by_seed[123]["before"]["second_day_requested_energy"]["value"]
    for row in by_seed.values():
        assert row["after"]["max_pre_step_delta"]["value"] == 0
        assert any(check["name"] == "load_random_residual_held_fixed" and check["status"] == "PASS" for check in row["checks"])


def test_unsupported_physics_never_becomes_a_green_test(report):
    unsupported=[row for row in report["cases"] if row["status"] == "UNSUPPORTED"]
    assert len(unsupported) == 3
    assert report["coverage_status"] == "PARTIAL_WITH_EXPLICIT_UNSUPPORTED"
    for row in unsupported:
        assert row["reason"] and row["actual_functions"] == [] and row["checks"] == []
    markdown=protocol.render_markdown(report)
    assert "UNSUPPORTED" in markdown
    assert "未支持项不计入 PASS" in markdown
    assert "Wh/m2" in markdown and "MWh" in markdown and "固定资产哈希" in markdown
    json.dumps(report,allow_nan=False)


def test_exact_repeat_is_reproducible(report):
    repeated=protocol.run_interventions((42,123))
    assert protocol._digest(repeated) == protocol._digest(report)


def test_corrupt_pv_kernel_output_is_detected_by_the_protocol(monkeypatch):
    original=protocol.generate_source_load_forecast
    def broken(*args,**kwargs):
        result=original(*args,**kwargs)
        power=result.p_gen_available_mw.copy();power[:,2]=0
        return replace(result,p_gen_available_mw=power)
    monkeypatch.setattr(protocol,"generate_source_load_forecast",broken)
    monkeypatch.setattr(protocol,"CASES",(("clear_cloud_to_radiation_to_pv",protocol._cloud_to_pv),))
    result=protocol.run_interventions((42,))
    row=next(row for row in result["cases"] if row["case_id"] == "clear_cloud_to_radiation_to_pv")
    assert row["status"] == "FAIL"
    assert next(check for check in row["checks"] if check["name"] == "pv_energy_increases")["status"] == "FAIL"
    assert result["supported_execution_status"] == "FAIL"


def test_kernel_exception_is_failed_execution_not_unsupported(monkeypatch):
    def broken(context):
        raise RuntimeError("deliberate protocol execution failure")
    monkeypatch.setattr(protocol,"CASES",(("kernel_failure",broken),))
    result=protocol.run_interventions((123,))
    assert result["counts"]["FAIL"] == 1
    row=next(row for row in result["cases"] if row["case_id"] == "kernel_failure")
    assert row["status"] == "FAIL" and row["error_type"] == "RuntimeError"


def test_fixture_failure_records_failed_setup_and_unexecuted_cases(monkeypatch):
    def broken():
        raise RuntimeError("deliberate fixture failure")
    monkeypatch.setattr(protocol,"_daily_fixture",broken)
    result=protocol.run_interventions((42,123))
    assert result["counts"] == {"PASS":0,"FAIL":2,"UNSUPPORTED":3,"NOT_RUN":18}
    assert result["supported_execution_status"] == "FAIL"
    assert "NOT_RUN" in protocol.render_markdown(result)
    for row in result["cases"]:
        if row["status"] == "NOT_RUN":
            assert row["checks"] == [] and row["actual_functions"] == []


@pytest.mark.parametrize("seeds",[(),(42,42),(-1,),(.5,)])
def test_invalid_seed_declarations_are_rejected(seeds):
    with pytest.raises(ValueError,match="distinct nonnegative integers"):
        protocol.run_interventions(seeds)


def test_cli_writes_numeric_json_and_readable_markdown_inside_repo(report,tmp_path,monkeypatch):
    assert protocol.ROOT in tmp_path.resolve().parents, "Test temporary files must stay in the repository"
    monkeypatch.setattr(protocol,"run_interventions",lambda seeds:report)
    assert protocol.main(["--seeds","42","123","--output-dir",str(tmp_path)]) == 0
    saved=json.loads((tmp_path/"mechanism_interventions.json").read_text(encoding="utf-8"))
    assert saved == report
    assert (tmp_path/"mechanism_interventions.md").read_text(encoding="utf-8").startswith("# 成对机制")


def test_cli_failed_execution_keeps_failure_report_and_returns_nonzero(report,tmp_path,monkeypatch):
    assert protocol.ROOT in tmp_path.resolve().parents
    failed=dict(report,counts={"PASS":17,"FAIL":1,"UNSUPPORTED":3,"NOT_RUN":0},supported_execution_status="FAIL")
    monkeypatch.setattr(protocol,"run_interventions",lambda seeds:failed)
    assert protocol.main(["--output-dir",str(tmp_path)]) == 1
    saved=json.loads((tmp_path/"mechanism_interventions.json").read_text(encoding="utf-8"))
    assert saved["counts"]["FAIL"] == 1 and saved["supported_execution_status"] == "FAIL"


def test_cli_rejects_any_output_path_outside_repository_before_execution(monkeypatch):
    def should_not_run(seeds):
        raise AssertionError("outside-path rejection must precede kernel execution")
    monkeypatch.setattr(protocol,"run_interventions",should_not_run)
    with pytest.raises(SystemExit) as error:
        protocol.main(["--output-dir",str(protocol.ROOT.parent/"forbidden_protocol_output")])
    assert error.value.code == 2
