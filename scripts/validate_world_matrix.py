"""Compare already-generated seeds/scales without claiming regional calibration.

This runner never regenerates or repairs data. Each world's physical checks run
independently; cross-world statistics describe scenarios, not accuracy scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_world_physics import validate_world, _safe_json
from world_generator.core.config import load_world_config
from world_generator.core.output_layout import WorldDataLayout


def summarize_case(world: Path) -> dict[str, object]:
    layout = WorldDataLayout(world / "data")
    report = validate_world(world)
    if "summary" not in report:
        return {"world": world.name, "passed_supported_checks": False,
                "category": report.get("error_category", "INPUT_OR_GENERATION_ERROR"),
                "status_counts": report.get("validation_scope", {}).get("counts", {}),
                "failure_names": [row["name"] for row in report["checks"]],
                "message": report.get("limitations", "World could not be evaluated")}
    config = load_world_config(layout.config_snapshot)
    summary = report["summary"]
    with np.load(layout.topology / "static_maps.npz", allow_pickle=False) as payload:
        elevation = np.asarray(payload["elevation"], dtype=float)
    counts = {state: sum(row.get("status", "PASS" if row["passed"] else "FAIL") == state for row in report["checks"])
              for state in ("PASS", "FAIL", "UNSUPPORTED", "NOT_RUN")}
    return {"world": world.name, "seed": config.seed,
            "grid_shape": [config.world.height, config.world.width], "cell_size_km": config.world.cell_size_km,
            "physical_extent_km": [config.world.height * config.world.cell_size_km, config.world.width * config.world.cell_size_km],
            "area_km2": config.world.height * config.world.width * config.world.cell_size_km**2,
            "hours": summary["hours"], "planning_mode": config.planning.mode,
            "hydrology_enabled": config.hydrology_dynamic.enabled,
            "passed_supported_checks": report["passed"], "status_counts": counts,
            "elevation_quantiles_m": np.quantile(elevation, [.05, .5, .95]).tolist(),
            "population_persons": summary["population_persons"],
            "requested_load_mwh": summary["source_load"].get("requested_load_mwh"),
            "peak_exogenous_load_mw": summary["peak_exogenous_load_mw"],
            "unserved_mwh": summary["total_unserved_mwh"],
            "capacity_factors": summary["capacity_factors_descriptive_only"],
            "hydrology": summary["dynamic_hydrology"],
            "failure_names": [row["name"] for row in report["checks"] if row.get("status") == "FAIL" or row.get("passed") is False]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worlds", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(PROJECT_ROOT.parent):
        parser.error("Validation output must stay inside the workspace")
    cases = []
    for path in args.worlds:
        try:
            cases.append(summarize_case(path.resolve()))
        except (ValueError, KeyError, FileNotFoundError, IndexError) as exc:
            cases.append({"world": path.name, "passed_supported_checks": False,
                          "category": "INPUT_OR_GENERATION_ERROR", "message": str(exc)})
    result = {"schema_version": "world_matrix_v1", "relation_class": "S",
              "scope": "independent mechanism checks and descriptive seed/resolution/scenario sensitivity; no calibration",
              "comparison_rules": ["Equal physical extent does not imply equal pixels, nodes, population or weather realization.",
                                   "Seed comparisons are not paired causal interventions; use validate_mechanism_interventions.py.",
                                   "No resolution is called converged from a two-grid comparison.",
                                   "Explicit shortfall can coexist with a supported-constraint PASS."],
              "passed_supported_checks": all(case["passed_supported_checks"] for case in cases), "cases": cases}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_safe_json(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"World matrix: {len(cases)} cases; supported checks {'PASS' if result['passed_supported_checks'] else 'FAIL'}")
    return 0 if result["passed_supported_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
