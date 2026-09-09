"""A failed attempt must retain its cause even if no output schema was written."""
import json

from scripts.validate_world_matrix import summarize_case


def test_matrix_retains_infeasibility_marker_without_reading_stale_outputs(tmp_path):
    world = tmp_path / "failed_world"
    world.mkdir()
    (world / "generation_failure.json").write_text(json.dumps({
        "schema_version": "generation_failure_v1", "status": "FAIL",
        "category": "PHYSICAL_INFEASIBILITY", "stage": "stage14_dc_opf",
        "message": "Isolated requested load cannot be served with load shedding disabled",
        "error_type": "PhysicalInfeasibilityError",
    }), encoding="utf-8")
    result = summarize_case(world)
    assert result["passed_supported_checks"] is False
    assert result["category"] == "PHYSICAL_INFEASIBILITY"
    assert result["failure_names"] == ["latest_generation_failed"]
