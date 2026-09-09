"""Independent C export checks, including corrupted ledgers and zero assets."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_world_physics import Checks, check_land_accounting, check_thermal_land
from world_generator.core.config import CityConfig, WorldConfig, WorldGridConfig


def _example():
    config = replace(WorldConfig(), world=WorldGridConfig(2, 2, 2.0),
                     city=replace(CityConfig(), city_count=1, total_population=100.0))
    water = np.array([[True, False], [False, False]])
    protected = np.array([[False, False], [True, False]])
    active = np.array([[0.0, 1.0], [0.0, 0.0]])
    static = {"river": water, "lake": np.zeros((2, 2), bool),
              "protected": protected, "protected_mask": protected.copy(),
              "land_cover_type": np.array([[1, 3], [4, 2]]), "landform": np.ones((2, 2)),
              "allocatable_land_fraction": active.copy(), "population_density": active * 25,
              "energy_available_area_km2": active * 2.0,
              "energy_wind_project_area_km2": active * 0.8,
              "energy_pv_project_area_km2": active * 0.4,
              "energy_unallocated_area_km2": active * 0.8,
              "wind_land_eligible": active.copy(), "pv_land_eligible": active.copy()}
    fractions = {"water": water.astype(float), "wetland": np.array([[0, 0], [0, 1.0]]),
                 "residential": active * 0.4, "commercial": active * 0,
                 "industrial": active * 0, "agriculture": active * 0.1,
                 "park_green": active * 0, "natural": protected.astype(float),
                 "energy_reserve": active * 0.5}
    static.update({f"land_use_fraction_{name}": value for name, value in fractions.items()})
    columns = ("project_id", "technology_code", "candidate_id", "row", "col", "reserved_area_km2",
               "capacity_mw", "capacity_density_mw_km2", "capacity_equivalent_area_km2")
    candidates = {"energy_project_land_columns": np.array(columns),
                  "energy_project_land_ledger": np.array([[0, 1, 0, 0, 1, 0.8, 2.4, 3, 0.8],
                                                          [1, 2, 0, 0, 1, 0.4, 14, 35, 0.4]]),
                  "energy_project_area_by_cell_km2": np.stack([active * 0.8, active * 0.4]),
                  "wind_candidates": np.array([[0, 0, 1, 1, 0, 2.4, 1]]),
                  "pv_candidates": np.array([[0, 0, 1, 1, 0, 14, 1]])}
    metadata = {"land_accounting_version": "land_use_v1", "cities": [{"population": 100}],
                "city_population_budget": {"configured_population_persons": 100,
                    "target_population_persons": 100, "allocated_population_persons": 100,
                    "allocatable_land_area_km2": 4, "placed_city_count": 1}}
    return static, candidates, metadata, config


def _run(args):
    checks = Checks()
    summary = check_land_accounting(checks, *args)
    return {row["name"]: row for row in checks.rows}, summary


def test_land_ledgers_close_in_square_kilometres_and_keep_independent_protection():
    rows, summary = _run(_example())
    assert all(row["passed"] for row in rows.values())
    assert summary["domain_area_km2"] == 16
    assert summary["allocatable_area_km2"] == 4
    assert summary["energy_reserved_area_km2"] == 2
    assert summary["target_population_persons"] == 100


@pytest.mark.parametrize("case,expected", [
    ("fraction", "land_use_fraction_cell_closure"),
    ("protected", "allocatable_land_hard_exclusions"),
    ("project", "energy_project_area_integration"),
    ("double_count", "energy_projects_no_double_allocation"),
    ("setback", "wind_project_full_envelope_eligible"),
    ("capacity", "energy_project_capacity_area_bound"),
    ("people", "population_target_not_silently_dropped"),
])
def test_corrupt_land_accounting_names_the_failed_relationship(case, expected):
    static, candidates, metadata, config = _example()
    if case == "fraction":
        static["land_use_fraction_residential"][0, 1] += 0.1
    elif case == "protected":
        static["allocatable_land_fraction"][1, 0] = 1
    elif case == "project":
        candidates["energy_project_land_ledger"][0, 5] += 0.2
    elif case == "double_count":
        candidates["energy_project_area_by_cell_km2"][1, 0, 1] += 2
    elif case == "setback":
        static["wind_land_eligible"][0, 1] = 0
    elif case == "capacity":
        candidates["energy_project_land_ledger"][0, 6] = 100
    elif case == "people":
        metadata["cities"] = []
        metadata["city_population_budget"]["allocated_population_persons"] = 0
        metadata["city_population_budget"]["placed_city_count"] = 0
    rows, _ = _run((static, candidates, metadata, config))
    assert not rows[expected]["passed"]


def test_zero_energy_projects_preserve_unused_area_instead_of_losing_land():
    static, candidates, metadata, config = _example()
    for kind in ("wind", "pv"):
        static[f"energy_{kind}_project_area_km2"][:] = 0
        candidates[f"{kind}_candidates"] = np.empty((0, 7))
    static["energy_unallocated_area_km2"] = static["energy_available_area_km2"].copy()
    candidates["energy_project_land_ledger"] = np.empty((0, 9))
    candidates["energy_project_area_by_cell_km2"] = np.empty((0, 2, 2))
    rows, summary = _run((static, candidates, metadata, config))
    assert all(row["status"] != "FAIL" for row in rows.values())
    assert rows["energy_cell_area_budget"]["status"] == "PASS"
    assert rows["wind_project_config_density"]["status"] == "NOT_RUN"
    assert rows["pv_project_config_density"]["status"] == "NOT_RUN"
    assert summary["unallocated_energy_area_km2"] == 2


def test_partial_new_land_schema_cannot_silently_skip_area_validation():
    args = _example()
    del args[0]["land_use_fraction_natural"]
    with pytest.raises(ValueError, match="land_use_fraction_natural"):
        _run(args)


def _thermal_example():
    static, _, metadata, config = _example()
    active = static["allocatable_land_fraction"]
    static.update({"thermal_allocated_area_km2": active * 0.4,
                   "energy_unallocated_after_thermal_area_km2": active * 0.4,
                   "thermal_land_eligible": active.copy()})
    columns = ("bus_id", "row", "col", "reserved_area_km2", "capacity_mw", "capacity_density_mw_km2",
               "capacity_equivalent_area_km2", "land_capacity_upper_bound_mw", "requested_capacity_mw")
    nodes = {"thermal_land_columns": np.array(columns),
             "thermal_land_ledger": np.array([[99, 0, 1, 0.4, 80, 200, 0.4, 80, 100]]),
             "thermal_project_area_by_cell_km2": (active * 0.4)[None, ...]}
    metadata["grid_buses"] = [{"bus_id": 99, "kind": "thermal_bus", "row": 0, "col": 1, "capacity_mw": 80}]
    config = SimpleNamespace(world=config.world, power_grid=SimpleNamespace(thermal_capacity_density_mw_km2=200))
    return static, nodes, metadata, config


def test_thermal_area_uses_only_renewable_remainder_and_may_limit_capacity():
    checks = Checks()
    summary = check_thermal_land(checks, *_thermal_example())
    assert all(row["passed"] for row in checks.rows)
    assert summary["thermal_stage09_capacity_mw"] == 80
    assert summary["thermal_project_area_km2"] == 0.4


@pytest.mark.parametrize("case,failed", [("double_count", "thermal_cannot_reuse_wind_pv_land"),
                                          ("capacity", "thermal_stage09_capacity_within_land"),
                                          ("eligibility", "thermal_project_full_envelope_eligible")])
def test_thermal_export_corruptions_cannot_hide_behind_site_suitability(case, failed):
    args = _thermal_example()
    if case == "double_count":
        args[1]["thermal_project_area_by_cell_km2"][0, 0, 1] = 2
    elif case == "capacity":
        args[1]["thermal_land_ledger"][0, 4] = 1000
    else:
        args[0]["thermal_land_eligible"][0, 1] = 0
    checks = Checks()
    check_thermal_land(checks, *args)
    assert not next(row for row in checks.rows if row["name"] == failed)["passed"]
