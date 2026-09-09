"""E independent dimensional ledgers and weather-to-asset diagnostics.

The exported exogenous realization is distinct from network-delivered power.
Requests above a load bus's design nameplate are retained and reported.
"""
from __future__ import annotations

import numpy as np

from world_generator.core.contracts import entity_ids, integer_labels


def check_source_load_contracts(checks, source, hourly, config, metadata):
    ids = entity_ids(source["bus_ids"], "source/load bus IDs")
    kinds = np.asarray(source["bus_kinds"])
    bounds = np.asarray(source["time_bounds_hours"], dtype=float)
    weather_bounds = np.asarray(hourly["time_bounds_hours"], dtype=float)
    if bounds.shape != weather_bounds.shape or not np.array_equal(source["timestamps"], hourly["timestamps"]):
        raise ValueError("Source/load intervals and timestamps must match shared hourly weather")
    checks.equal("source_load_weather_interval_support", bounds - weather_bounds, 1e-9, "h")
    dt = np.diff(bounds, axis=1)
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("Source/load integration requires positive finite intervals")
    shape = (len(bounds), len(ids))
    power_fields = ("p_load_mw", "p_gen_available_mw", "p_gen_scheduled_mw")
    if kinds.shape != (len(ids),):
        raise ValueError("Source/load bus kind axis must match IDs")
    for name in power_fields:
        if source[name].shape != shape or not np.isfinite(source[name]).all():
            raise ValueError(f"Source/load power field {name} must be finite [T,N]")
        checks.upper("source_load_nonnegative_" + name, -source[name], 0, 1e-6, "MW")
    checks.upper("source_load_schedule_below_available", source["p_gen_scheduled_mw"], source["p_gen_available_mw"], 2e-4, "MW")
    nameplate = np.asarray(source["nameplate_capacity_mw"], dtype=float)
    reference = np.asarray(source["reference_load_mw"], dtype=float)
    if nameplate.shape != (len(ids),) or reference.shape != nameplate.shape or not np.isfinite(nameplate).all() or not np.isfinite(reference).all() or np.any(nameplate < 0) or np.any(reference < 0):
        raise ValueError("Source/load static capacities must be finite nonnegative [N]")
    assets = {int(bus["bus_id"]): bus for bus in metadata["refined_grid_buses"]}
    if set(ids) != set(assets):
        raise ValueError("Source/load bus IDs must match original refined assets")
    checks.equal("source_load_nameplate_static_asset_anchor", nameplate - [assets[int(key)]["capacity_mw"] for key in ids], 2e-4, "MW")
    checks.condition("source_load_kind_static_asset_anchor", tuple(kinds) == tuple(assets[int(key)]["kind"] for key in ids))
    generators = np.isin(kinds, ["wind_bus", "pv_bus", "thermal_bus"])
    loads = kinds == "load_bus"
    checks.upper("source_load_available_nameplate_bound", source["p_gen_available_mw"][:, generators], nameplate[None, generators], 2e-4, "MW")
    checks.equal("source_load_no_generation_at_other_buses", source["p_gen_available_mw"][:, ~generators], 1e-6, "MW")
    checks.equal("source_load_no_demand_at_nonload_buses", source["p_load_mw"][:, ~loads], 1e-6, "MW")
    checks.equal("source_load_reference_at_load_buses_only", reference[~loads], 1e-6, "MW")
    energies = {}
    for prefix, power in zip(("requested_load", "available_generation", "planned_generation"), power_fields):
        expected = np.asarray(source[power], dtype=float) * dt
        actual = source[prefix + "_energy_mwh"]
        if actual.shape != shape:
            raise ValueError(f"Source/load {prefix} interval energy must be [T,N]")
        checks.equal("source_load_interval_energy_" + prefix, actual - expected, 1e-7, "MWh")
        checks.equal("source_load_period_energy_" + prefix, source["period_" + prefix + "_energy_mwh"] - expected.sum(axis=0), 1e-6, "MWh")
        energies[prefix + "_mwh"] = float(expected.sum())
    valid = generators & (nameplate > 0)
    checks.condition("source_load_capacity_factor_valid_mask", np.array_equal(source["capacity_factor_valid"], valid))
    for prefix, power in (("available", "p_gen_available_mw"), ("planned", "p_gen_scheduled_mw")):
        expected = np.divide(source[power], nameplate[None, :], out=np.zeros(shape), where=valid[None, :])
        checks.equal("source_load_" + prefix + "_capacity_factor", source[prefix + "_capacity_factor"] - expected, 1e-7, "1")

    rows = integer_labels(source["weather_sample_row"], "source weather rows")
    cols = integer_labels(source["weather_sample_col"], "source weather columns")
    if rows.shape != (len(ids),) or cols.shape != rows.shape or np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= hourly["dynamic"].shape[2]) or np.any(cols >= hourly["dynamic"].shape[3]):
        raise ValueError("Source/load weather sampling requires in-grid row/col for every bus")
    checks.condition("source_load_weather_location_static_anchor", all(rows[i] == assets[int(key)]["row"] and cols[i] == assets[int(key)]["col"] for i, key in enumerate(ids)))
    channels = {str(name): index for index, name in enumerate(hourly["channel_names"])}
    temperature = np.asarray(hourly["dynamic"][:, channels["temperature"], rows, cols], dtype=float)
    wind = np.asarray(hourly["dynamic"][:, channels["wind_speed"], rows, cols], dtype=float)
    cfg = config.source_load
    diagnostics = {
        "hub_wind_speed_mps": kinds == "wind_bus", "wind_air_density_kg_m3": kinds == "wind_bus",
        "pv_poa_w_m2": kinds == "pv_bus", "pv_module_temperature_c": kinds == "pv_bus",
        "load_effective_temperature_c": loads, "load_log_residual": loads,
    }
    for name, mask in diagnostics.items():
        values = source["diag__" + name]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Source/load diagnostic {name} must be finite [T,N]")
        checks.equal("source_load_diagnostic_applicability_" + name, values[:, ~mask], 1e-9, "native")
    wind_mask = kinds == "wind_bus"
    expected_hub_wind = wind[:, wind_mask] * (cfg.wind_hub_height_m / cfg.wind_reference_height_m) ** cfg.wind_shear_exponent
    checks.equal("source_load_hub_height_wind", source["diag__hub_wind_speed_mps"][:, wind_mask] - expected_hub_wind, 2e-5, "m/s")
    pressure = np.asarray(hourly["dynamic"][:, channels["pressure"], rows, cols], dtype=float) if "pressure" in channels else None
    if not np.any(wind_mask):
        surface_density = np.full(shape, cfg.wind_reference_density_kg_m3)
    elif "diagnostic__air_density_kg_m3" in hourly:
        density_map = np.asarray(hourly["diagnostic__air_density_kg_m3"], dtype=float)
        if density_map.shape != (shape[0], *hourly["dynamic"].shape[2:]) or not np.isfinite(density_map).all() or np.any(density_map <= 0):
            raise ValueError("Shared weather air density must be positive finite [T,H,W]")
        surface_density = density_map[:, rows, cols]
    elif cfg.wind_density_fallback == "dry_air_then_reference":
        surface_density = pressure * 100 / (287.05 * (temperature + 273.15)) if pressure is not None else np.full(shape, cfg.wind_reference_density_kg_m3)
    else:
        raise ValueError("Source/load declared density requires the shared weather diagnostic")
    hub_density = surface_density
    if np.any(wind_mask) and cfg.wind_density_height_mode == "isothermal_surface_to_hub" and pressure is None and "diagnostic__air_density_kg_m3" in hourly:
        raise ValueError("Source/load surface-to-hub density requires surface pressure")
    if cfg.wind_density_height_mode == "isothermal_surface_to_hub" and pressure is not None:
        hub_density = surface_density * np.exp(-9.80665 * surface_density * cfg.wind_hub_height_m / (100 * pressure))
    checks.equal("source_load_shared_weather_density", source["diag__wind_air_density_kg_m3"][:, wind_mask] - hub_density[:, wind_mask], 1e-7, "kg/m3")
    wind_fraction = np.clip(hub_density[:, wind_mask] / cfg.wind_reference_density_kg_m3 *
                            (expected_hub_wind**3 - cfg.wind_cut_in_mps**3) /
                            (cfg.wind_rated_mps**3 - cfg.wind_cut_in_mps**3), 0, 1)
    wind_fraction = np.where((expected_hub_wind >= cfg.wind_cut_in_mps) & (expected_hub_wind < cfg.wind_cut_out_mps), wind_fraction, 0)
    expected_wind_power = nameplate[None, wind_mask] * (1 - cfg.wind_system_loss_fraction) * wind_fraction
    checks.equal("source_load_wind_engineering_curve", source["p_gen_available_mw"][:, wind_mask] - expected_wind_power, 2e-4, "MW", "continuous density scaling; nominal rated speed at reference density; actual hub speed controls shutdown")
    pv = kinds == "pv_bus"
    poa = np.asarray(source["diag__pv_poa_w_m2"][:, pv], dtype=float)
    ghi = np.asarray(hourly["dynamic"][:, channels["irradiance"], rows, cols], dtype=float)[:, pv]
    checks.upper("source_load_pv_poa_nonnegative", -poa, 0, 1e-6, "W/m2")
    checks.equal("source_load_zero_ghi_zero_poa", poa[ghi == 0], 1e-6, "W/m2")
    module_wind = wind[:, pv] * (cfg.pv_module_height_m / cfg.wind_reference_height_m) ** cfg.pv_wind_shear_exponent
    module_temperature = temperature[:, pv] + poa / (cfg.pv_heat_loss_constant + cfg.pv_heat_loss_wind * module_wind)
    checks.equal("source_load_pv_module_temperature", source["diag__pv_module_temperature_c"][:, pv] - module_temperature, 2e-5, "degC", "Faiman engineering closure at configured module-height wind")
    expected_pv = np.clip(nameplate[None, pv] * cfg.pv_dc_ac_ratio * poa / 1000 *
                          np.maximum(1 + cfg.pv_temperature_coefficient_per_c * (module_temperature - 25), 0) *
                          (1 - cfg.pv_system_loss_fraction) * cfg.pv_inverter_efficiency, 0, nameplate[None, pv])
    checks.equal("source_load_pv_ac_conversion", source["p_gen_available_mw"][:, pv] - expected_pv, 2e-4, "MW", "GHI already includes cloud effects; no second cloud multiplier")
    initial = np.asarray(source["initial_effective_temperature_c"])
    if initial.shape != (len(ids),) or not np.isfinite(initial).all():
        raise ValueError("Source/load thermal initial boundary must be finite [N]")
    checks.equal("source_load_thermal_initial_applicability", initial[~loads], 1e-9, "degC")
    expected_initial = temperature[0, loads] if cfg.load_initial_temperature_mode == "first_hour" else cfg.load_initial_temperature_c
    checks.equal("source_load_thermal_initial_configuration", initial[loads] - expected_initial, 1e-5, "degC")
    effective = source["diag__load_effective_temperature_c"][:, loads]
    previous = np.concatenate((initial[None, loads], effective[:-1]), axis=0)
    decay = np.exp(-dt / cfg.load_thermal_memory_hours) if cfg.load_thermal_memory_hours > 0 else np.zeros_like(dt)
    checks.equal("source_load_thermal_memory_recursion", effective - decay * previous - (1 - decay) * temperature[:, loads], 1e-5, "degC", "T end states plus explicit pre-window boundary")
    # Design peak is an asset descriptor, not permission to remove requested load.
    utilization = np.divide(source["p_load_mw"][:, loads], nameplate[None, loads], out=np.zeros((shape[0], int(loads.sum()))), where=nameplate[None, loads] > 0)
    return {**energies, "max_requested_load_design_utilization": float(np.max(utilization, initial=0)),
            "requested_load_exceeds_design_count": int(np.sum(source["p_load_mw"][:, loads] > nameplate[None, loads])),
            "semantics": "exogenous requested/available/planned only; delivery, curtailment and unserved belong to operation"}
