"""Explicit D water fields: prefixes serialize groups; names define support."""
from __future__ import annotations

HYDROLOGY_SCHEMA_VERSION = "hydrology_v1"
HYDROLOGY_MODE = "bucket_routing_v1"
HYDROLOGY_STATE_UNITS = {
    "soil_storage_mm": "mm", "groundwater_storage_mm": "mm",
    "channel_storage_m3": "m3", "lake_storage_m3": "m3", "lake_water_level_m": "m", "lake_wetted_area_m2": "m2",
}
HYDROLOGY_FLUX_UNITS = {
    **{name: "mm" for name in (
        "precipitation_mm", "infiltration_mm", "soil_evapotranspiration_mm", "percolation_mm",
        "groundwater_overflow_mm", "baseflow_mm", "surface_runoff_mm", "potential_et_mm")},
    **{name: "m3" for name in (
        "open_water_evaporation_m3", "routing_inflow_m3", "routing_outflow_m3", "lake_mixing_inflow_m3",
        "lake_mixing_outflow_m3", "lake_overflow_m3", "boundary_inflow_m3", "boundary_outflow_m3", "actual_et_m3", "cell_budget_residual_m3")},
    "discharge_m3_s": "m3/s",
}
HYDROLOGY_STATIC_UNITS = {
    "impervious_fraction": "1", "pervious_fraction": "1", "soil_capacity_mm": "mm",
    "groundwater_capacity_mm": "mm", "lake_id": "1",
    "closed_sink_mask": "1", "lake_bed_elevation_m": "m", "lake_spill_elevation_m": "m", "lake_capacity_m3": "m3",
    "routing_receiver_flat_index": "1",
}
HYDROLOGY_BUDGET_UNITS = {name: "m3" for name in (
    "initial_storage_m3", "precipitation_m3", "boundary_inflow_m3", "actual_et_m3",
    "boundary_outflow_m3", "final_storage_m3", "residual_m3")}
HYDROLOGY_GROUP_UNITS = {
    "state": HYDROLOGY_STATE_UNITS, "flux": HYDROLOGY_FLUX_UNITS,
    "static": HYDROLOGY_STATIC_UNITS, "budget": HYDROLOGY_BUDGET_UNITS,
}


def hydrology_field_schema() -> dict[str, dict[str, str]]:
    schema = {}
    supports = {"state": "T+1,H,W", "flux": "T,H,W", "static": "H,W", "budget": "T"}
    time_kinds = {"state": "interval_boundary_state", "flux": "interval_accumulation", "static": "static", "budget": "interval_water_account"}
    for group, names in HYDROLOGY_GROUP_UNITS.items():
        for name, unit in names.items():
            time_kind = time_kinds[group]
            if name == "discharge_m3_s": time_kind = "interval_mean_rate"
            if group == "budget" and name == "initial_storage_m3": time_kind = "interval_start_state"
            if group == "budget" and name == "final_storage_m3": time_kind = "interval_end_state"
            if "residual" in name: time_kind = "interval_balance_residual"
            spatial_support = "whole_domain" if group == "budget" else ("whole_cell_equivalent_depth" if unit == "mm" else "cell")
            schema[f"{group}__{name}"] = {"unit": unit, "time_kind": time_kind, "shape": supports[group], "spatial_support": spatial_support}
    return schema
