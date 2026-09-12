"""E explicit field support; generation assumptions differ from energy accounts."""
SOURCE_LOAD_SCHEMA_VERSION = "source_load_v1"
SOURCE_LOAD_MODE = "exogenous_realization"
SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY = {
    "hub_wind_speed_mps": ("wind_bus",), "wind_air_density_kg_m3": ("wind_bus",),
    "pv_poa_w_m2": ("pv_bus",), "pv_module_temperature_c": ("pv_bus",),
    "load_effective_temperature_c": ("load_bus",), "load_log_residual": ("load_bus",),
}
SOURCE_LOAD_DIAGNOSTIC_UNITS = dict(zip(SOURCE_LOAD_DIAGNOSTIC_APPLICABILITY, ("m/s", "kg/m3", "W/m2", "degC", "degC", "1")))
SOURCE_LOAD_STATIC_UNITS = {
    "nameplate_capacity_mw": "MW", "reference_load_mw": "MW", "initial_effective_temperature_c": "degC",
    "weather_sample_row": "index", "weather_sample_col": "index", "capacity_factor_valid": "1",
}
SOURCE_LOAD_ENERGY_POWER_FIELDS = {
    "requested_load_energy_mwh": "p_load_mw", "available_generation_energy_mwh": "p_gen_available_mw",
    "planned_generation_energy_mwh": "p_gen_scheduled_mw",
}
SOURCE_LOAD_CF_POWER_FIELDS = {
    "available_capacity_factor": "p_gen_available_mw", "planned_capacity_factor": "p_gen_scheduled_mw",
}
SOURCE_LOAD_BASE_INTERVAL_FIELDS = ("p_load_mw", "p_gen_available_mw", "p_gen_scheduled_mw", "q_load_mvar")


def source_load_field_schema() -> dict[str, dict[str, str]]:
    schema = {}
    for name, unit in SOURCE_LOAD_STATIC_UNITS.items():
        schema[name] = {"unit": unit, "shape": "N", "time_kind": "initial_boundary_state" if name == "initial_effective_temperature_c" else "fixed_node_attribute", "generation_relation_type": "S"}
    for name, unit in SOURCE_LOAD_DIAGNOSTIC_UNITS.items():
        kind = "interval_end_state" if name == "load_effective_temperature_c" else "interval_representative_diagnostic"
        schema[f"diag__{name}"] = {"unit": unit, "shape": "T,N", "time_kind": kind, "generation_relation_type": "S"}
    for name in SOURCE_LOAD_ENERGY_POWER_FIELDS:
        schema[name] = {"unit": "MWh", "shape": "T,N", "time_kind": "interval_accumulation", "generation_relation_type": "P"}
        schema[f"period_{name}"] = {"unit": "MWh", "shape": "N", "time_kind": "declared_period_accumulation", "generation_relation_type": "P", "information_role": "realized_period_audit_or_target_not_static_asset_input"}
    for name in SOURCE_LOAD_CF_POWER_FIELDS:
        schema[name] = {"unit": "1", "shape": "T,N", "time_kind": "interval_mean_ratio", "generation_relation_type": "P"}
    return schema
