"""Independent water ledgers reconstructed from exported D states and fluxes.

This checks conservation of the declared coarse model, not hydraulic validity.
Relative tolerances use each cell's own throughflow/inventory, never a domain
maximum that could conceal a small cell's water injection.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from world_generator.core.datatypes import HydrologyTimeSeriesStore


def check_dynamic_hydrology(checks, arrays, hourly, static, config):
    store = HydrologyTimeSeriesStore.from_arrays(arrays)
    s, f, m, b = store.states, store.fluxes, store.static_maps, store.budgets
    cfg = config.hydrology_dynamic
    shape = (config.world.height, config.world.width)
    if s["soil_storage_mm"].shape[1:] != shape:
        raise ValueError("Dynamic hydrology grid differs from the configured world")
    checks.condition("hydrology_weather_time_axis", np.array_equal(store.timestamps, hourly["timestamps"]))
    checks.equal("hydrology_weather_time_bounds", store.time_bounds_hours - hourly["time_bounds_hours"], 1e-9, "h")
    dt = np.diff(store.time_bounds_hours, axis=1)[:, 0, None, None]
    mm_volume = config.world.cell_size_km ** 2 * 1000.0
    cell_area = config.world.cell_size_km ** 2 * 1e6
    channels = list(hourly["channel_names"])
    rain = np.asarray(hourly["dynamic"][:, channels.index("precipitation")], dtype=float)
    ghi = np.asarray(hourly["dynamic"][:, channels.index("irradiance")], dtype=float)
    if rain.shape != f["precipitation_mm"].shape:
        raise ValueError("Dynamic hydrology forcing shape differs from hourly weather")

    def water_equal(name, residual, scale=0.0):
        checks.equal("hydrology_" + name, residual, cfg.budget_absolute_tolerance_m3, "m3",
                     relative_tolerance=cfg.budget_relative_tolerance, scale=scale)

    water = np.asarray(static["river"], bool) | np.asarray(static["lake"], bool)
    lake = np.asarray(static["lake"], bool)
    pervious, impervious = m["pervious_fraction"], m["impervious_fraction"]
    expected_impervious = sum(static["land_use_fraction_" + name] * coefficient
                              for name, coefficient in cfg.impervious_fraction_by_use.items())
    expected_impervious = np.where(water, 0.0, expected_impervious)
    checks.equal("hydrology_land_partition", pervious + impervious + water - 1, 2e-6, "1")
    checks.equal("hydrology_impervious_scenario_rule", impervious - expected_impervious, 2e-6, "1")
    checks.equal("hydrology_soil_capacity_support", m["soil_capacity_mm"] - cfg.soil_capacity_mm * pervious, 1e-8, "mm")
    checks.equal("hydrology_groundwater_capacity_support", m["groundwater_capacity_mm"] - cfg.groundwater_capacity_mm * ~water, 1e-8, "mm")
    for name, capacity in (("soil_storage_mm", "soil_capacity_mm"), ("groundwater_storage_mm", "groundwater_capacity_mm"), ("lake_storage_m3", "lake_capacity_m3")):
        checks.upper("hydrology_capacity_" + name, s[name], m[capacity], 1e-7)
    checks.equal("hydrology_soil_initial_state", s["soil_storage_mm"][0] - cfg.initial_soil_fraction * m["soil_capacity_mm"], 1e-8, "mm")
    checks.equal("hydrology_groundwater_initial_state", s["groundwater_storage_mm"][0] - cfg.initial_groundwater_fraction * m["groundwater_capacity_mm"], 1e-8, "mm")
    water_equal("channel_initial_state", s["channel_storage_m3"][0] - cfg.initial_channel_storage_mm * mm_volume * ~lake)
    checks.equal("hydrology_precipitation_forcing", f["precipitation_mm"] - rain, 1e-8, "mm")
    expected_pet = ghi * cfg.pet_shortwave_absorptivity * cfg.pet_latent_energy_fraction * dt * 3600 / cfg.latent_heat_vaporization_j_kg
    checks.equal("hydrology_pet_scenario_rule", f["potential_et_mm"] - expected_pet, 1e-8, "mm")
    checks.upper("hydrology_infiltration_rain_bound", f["infiltration_mm"], rain * pervious, 1e-8, "mm")
    checks.upper("hydrology_infiltration_rate_bound", f["infiltration_mm"], cfg.infiltration_capacity_mm_h * pervious * dt, 1e-8, "mm")
    checks.upper("hydrology_soil_et_energy_bound", f["soil_evapotranspiration_mm"], expected_pet * pervious, 1e-8, "mm")
    water_equal("open_water_et_energy_bound_excess", np.maximum(f["open_water_evaporation_m3"] - expected_pet * mm_volume * water, 0))
    soil_change = np.diff(s["soil_storage_mm"], axis=0)
    groundwater_change = np.diff(s["groundwater_storage_mm"], axis=0)
    water_equal("soil_balance", (soil_change - f["infiltration_mm"] + f["soil_evapotranspiration_mm"] + f["percolation_mm"]) * mm_volume,
                s["soil_storage_mm"][:-1] * mm_volume)
    water_equal("groundwater_balance", (groundwater_change - f["percolation_mm"] + f["baseflow_mm"] + f["groundwater_overflow_mm"]) * mm_volume,
                s["groundwater_storage_mm"][:-1] * mm_volume)
    water_equal("dry_soil_cannot_gain", np.where(rain == 0, np.maximum(soil_change, 0), 0) * mm_volume)
    checks.equal("hydrology_surface_runoff_partition", f["surface_runoff_mm"] + f["infiltration_mm"] - rain * ~lake, 1e-8, "mm")
    water_equal("actual_et_partition", f["actual_et_m3"] - f["soil_evapotranspiration_mm"] * mm_volume - f["open_water_evaporation_m3"])
    checks.equal("hydrology_discharge_rate_support", f["discharge_m3_s"] - (f["routing_outflow_m3"] + f["boundary_outflow_m3"]) / (dt * 3600), 1e-9, "m3/s")

    receiver = m["routing_receiver_flat_index"].astype(np.int64).ravel()
    internal = receiver >= 0
    outgoing = f["routing_outflow_m3"].reshape(len(dt), -1)
    incoming = np.zeros_like(outgoing)
    for hour in range(len(dt)):
        np.add.at(incoming[hour], receiver[internal], outgoing[hour, internal])
    water_equal("routing_incidence", f["routing_inflow_m3"] - incoming.reshape(f["routing_inflow_m3"].shape), f["routing_inflow_m3"])
    water_equal("routing_outflow_internal_only", outgoing[:, ~internal])
    water_equal("boundary_outflow_external_only", f["boundary_outflow_m3"].reshape(len(dt), -1)[:, receiver != -1])
    rows, cols = np.indices(shape)
    edge = (rows == 0) | (cols == 0) | (rows == shape[0] - 1) | (cols == shape[1] - 1)
    direction = np.asarray(static["flow_direction"])
    if direction.shape != shape or not np.isin(direction, np.arange(-1, 8)).all():
        raise ValueError("Static D8 directions must be integer [H,W] codes -1..7")
    expected_receiver = np.where(edge, -1, -2)
    offsets = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    for code, (dr, dc) in enumerate(offsets):
        selected = direction == code
        rr, cc = rows[selected] + dr, cols[selected] + dc
        if np.any((rr < 0) | (rr >= shape[0]) | (cc < 0) | (cc >= shape[1])):
            raise ValueError("Static D8 direction points outside the grid; use an edge outlet")
        expected_receiver[selected] = rr * shape[1] + cc
    checks.condition("hydrology_receivers_match_static_d8", np.array_equal(receiver, expected_receiver.ravel()))
    water_equal("boundary_inflow_edge_only", f["boundary_inflow_m3"][:, ~edge])
    checks.condition("hydrology_external_receivers_on_boundary", bool(np.all(edge.ravel()[receiver == -1])))
    checks.condition("hydrology_closed_sink_mask", np.array_equal(m["closed_sink_mask"].astype(bool).ravel(), receiver == -2))
    water_equal("lake_mixing_global_cancellation", (f["lake_mixing_inflow_m3"] - f["lake_mixing_outflow_m3"]).sum(axis=(1, 2)),
                f["lake_mixing_inflow_m3"].sum(axis=(1, 2)))
    for name in ("lake_mixing_inflow_m3", "lake_mixing_outflow_m3", "lake_overflow_m3"):
        water_equal(name + "_lake_footprint_only", f[name][:, ~lake])
    checks.condition("hydrology_lake_identity", np.array_equal(m["lake_id"] > 0, lake))
    components, _ = label(lake, structure=np.ones((3, 3), dtype=np.int8))
    checks.equal("hydrology_lake_bed_static_anchor", m["lake_bed_elevation_m"][lake] - static["hydrology_elevation"][lake], 1e-9, "m")
    level = s["lake_water_level_m"]
    depth = np.maximum(level - m["lake_bed_elevation_m"], 0)
    water_equal("lake_volume_geometry", s["lake_storage_m3"] - depth * cell_area * lake, s["lake_storage_m3"])
    checks.equal("hydrology_lake_wetted_area_geometry", s["lake_wetted_area_m2"] - (depth > 0) * cell_area * lake, 1e-6, "m2")
    checks.upper("hydrology_lake_spill_level", level[:, lake], m["lake_spill_elevation_m"][lake], 1e-9, "m")
    water_equal("lake_capacity_geometry", m["lake_capacity_m3"] - np.maximum(m["lake_spill_elevation_m"] - m["lake_bed_elevation_m"], 0) * cell_area * lake, m["lake_capacity_m3"])
    for lake_id in np.unique(m["lake_id"][lake]):
        selected = m["lake_id"] == lake_id
        component_ids = np.unique(components[selected])
        checks.condition(f"hydrology_lake_{lake_id}_connected_identity", len(component_ids) == 1 and np.array_equal(selected, components == component_ids[0]))
        static_spill = np.min(np.asarray(static["hydrology_elevation"][selected], dtype=float) + np.asarray(static["water_depth"][selected], dtype=float))
        checks.equal(f"hydrology_lake_{lake_id}_spill_static_anchor", m["lake_spill_elevation_m"][selected] - static_spill, 1e-9, "m")
        water_equal(f"lake_{lake_id}_mixing_cancellation", (f["lake_mixing_inflow_m3"][:, selected] - f["lake_mixing_outflow_m3"][:, selected]).sum(axis=1), f["lake_mixing_inflow_m3"][:, selected].sum(axis=1))
        # Overflow transfers lake inventory into its outlet's channel inventory.
        # It belongs in this lake-only ledger, but not in the whole-cell ledger.
        lake_inputs = (rain * mm_volume + f["boundary_inflow_m3"] + f["routing_inflow_m3"])[:, selected].sum(axis=1)
        lake_outputs = (f["open_water_evaporation_m3"] + f["lake_overflow_m3"])[:, selected].sum(axis=1)
        lake_start = s["lake_storage_m3"][:-1, selected].sum(axis=1)
        lake_end = s["lake_storage_m3"][1:, selected].sum(axis=1)
        water_equal(f"lake_{lake_id}_pool_balance", lake_start + lake_inputs - lake_end - lake_outputs, np.maximum(lake_start + lake_inputs, lake_end + lake_outputs))
        checks.equal(f"hydrology_lake_{lake_id}_common_level", np.ptp(level[:, selected], axis=1), 1e-9, "m")
        water_equal(f"lake_{lake_id}_initial_volume", s["lake_storage_m3"][0, selected].sum() - cfg.initial_lake_storage_fraction * m["lake_capacity_m3"][selected].sum(), m["lake_capacity_m3"][selected].sum())

    storage = (s["soil_storage_mm"] + s["groundwater_storage_mm"]) * mm_volume + s["channel_storage_m3"] + s["lake_storage_m3"]
    inputs = rain * mm_volume + f["boundary_inflow_m3"] + f["routing_inflow_m3"] + f["lake_mixing_inflow_m3"]
    outputs = f["actual_et_m3"] + f["boundary_outflow_m3"] + f["routing_outflow_m3"] + f["lake_mixing_outflow_m3"]
    residual = storage[:-1] + inputs - storage[1:] - outputs
    scale = np.maximum(storage[:-1] + inputs, storage[1:] + outputs)
    water_equal("cell_water_balance", residual, scale)
    water_equal("reported_cell_residual", f["cell_budget_residual_m3"] - residual, scale)
    totals = {"initial_storage_m3": storage[:-1].sum(axis=(1, 2)), "final_storage_m3": storage[1:].sum(axis=(1, 2)),
              "precipitation_m3": rain.sum(axis=(1, 2)) * mm_volume}
    totals.update({name: f[name].sum(axis=(1, 2)) for name in ("boundary_inflow_m3", "actual_et_m3", "boundary_outflow_m3")})
    lhs = totals["initial_storage_m3"] + totals["precipitation_m3"] + totals["boundary_inflow_m3"]
    rhs = totals["final_storage_m3"] + totals["actual_et_m3"] + totals["boundary_outflow_m3"]
    totals["residual_m3"] = lhs - rhs
    water_equal("domain_interval_balance", lhs - rhs, np.maximum(lhs, rhs))
    for name, expected in totals.items():
        water_equal("reported_budget_" + name, b[name] - expected, np.maximum(lhs, rhs) if name == "residual_m3" else expected)
    window_input = storage[0].sum() + totals["precipitation_m3"].sum() + totals["boundary_inflow_m3"].sum()
    window_output = storage[-1].sum() + totals["actual_et_m3"].sum() + totals["boundary_outflow_m3"].sum()
    water_equal("whole_window_balance", window_input - window_output, max(window_input, window_output))
    return {"mode": "bucket_routing_v1", "initial_storage_m3": float(storage[0].sum()), "final_storage_m3": float(storage[-1].sum()),
            "precipitation_m3": float(totals["precipitation_m3"].sum()), "actual_et_m3": float(totals["actual_et_m3"].sum()),
            "boundary_inflow_m3": float(totals["boundary_inflow_m3"].sum()), "boundary_outflow_m3": float(totals["boundary_outflow_m3"].sum()),
            "max_cell_residual_m3": float(np.max(np.abs(residual))), "whole_window_residual_m3": float(window_input - window_output)}
