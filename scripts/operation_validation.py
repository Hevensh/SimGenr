"""F independent exported DC/island/reserve and information-boundary checks.

These checks reconstruct identities from stored arrays, without solving an OPF.
Reserve is an island aggregate held option, not a claim about AC deliverability.
"""
from __future__ import annotations

import json
import numpy as np
from scripts.validation_checks import validation_context


@validation_context(stage="F_operation",fields=["storage.op__*","flow.*","source_load.*","electrical.*"],time_support="explicit_check_axis_or_aggregate",engineering_simplification="Lossless DC and island-aggregate reserve with configured service/ramp rules; no AC/frequency or reserve activation proof")
def check_operation_contracts(checks, storage, flow, electrical, source, config, thermal_ledger=None):
    from world_generator.core.datatypes import StorageDispatchStore, PowerFlowStore
    StorageDispatchStore.from_arrays(storage)
    PowerFlowStore.from_arrays(flow)
    op = {k[4:]: np.asarray(v) for k, v in storage.items() if k.startswith("op__")}
    cfg = config.storage
    ids = op["bus_ids"].astype(int)
    index = {int(key): i for i, key in enumerate(ids)}
    kinds = op["bus_kinds"]
    hours = len(storage["timestamps"])
    sites = np.asarray([index[int(i)] for i in storage["site_bus_ids"]], dtype=int)
    thermal = np.asarray([index[int(i)] for i in storage["thermal_bus_ids"]], dtype=int)
    renew = np.isin(kinds, ["wind_bus", "pv_bus"])
    therm = kinds == "thermal_bus"
    meta = json.loads(str(storage["operation_metadata_json"]))
    tol = 2e-3  # float32 legacy powers / MWh, not a relaxation of solver limits

    def eq(name, value, unit="MW", tolerance=tol, **annotations):
        checks.equal("operation_" + name, value, tolerance, unit, **annotations)

    def upper(name, value, limit, unit="MW", tolerance=tol, **annotations):
        checks.upper("operation_" + name, value, limit, tolerance, unit, **annotations)

    checks.condition("operation_bus_axis", np.array_equal(ids, flow["bus_ids"]))
    checks.condition("operation_branch_axis", np.array_equal(storage["branch_ids"], flow["branch_ids"]))
    eq("served_plus_unserved", op["served_load_mw"] + op["unserved_load_mw"] - op["requested_load_mw"],axes=("time","bus"),timestamps=storage["timestamps"],fields=["op__served_load_mw","op__unserved_load_mw","op__requested_load_mw"],time_support="hour_interval_mean")
    unserved_total = op["unserved_load_mw"].sum(axis=1)
    checks.equal("operation_unserved_not_hidden_in_summary", storage["dispatched_unserved_mw"] - unserved_total,
                 1e-7, "MW", relative_tolerance=5e-7, scale=unserved_total)
    upper("dispatch_within_available", op["generator_dispatch_mw"], op["generation_available_mw"])
    eq("renewable_energy_partition", (op["generation_available_mw"] - op["generator_dispatch_mw"]) * renew - op["renewable_curtailment_mw"])
    eq("thermal_available_partition", (op["generation_available_mw"] - op["generator_dispatch_mw"]) * therm - op["thermal_unused_available_mw"])
    eq("thermal_array_dispatch", op["thermal_dispatch_mw"] - op["generator_dispatch_mw"][:, thermal])
    eq("thermal_array_availability", op["thermal_available_mw"] - op["generation_available_mw"][:, thermal])
    eq("thermal_backdown_kind", op["thermal_backdown_mw"][:, ~therm])
    eq("nongenerator_zero_dispatch", op["generator_dispatch_mw"][:, ~(renew | therm)])
    eq("nonload_zero_request", op["requested_load_mw"][:, kinds != "load_bus"])

    # Independently map unchanged exogenous columns; an expanded thermal plant
    # retains the original hourly availability fraction, including exact zero.
    original = {int(key): i for i, key in enumerate(source["bus_ids"])}
    expected_request = np.zeros((hours, len(ids)))
    expected_available = np.zeros_like(expected_request)
    installed = dict(zip(storage["thermal_bus_ids"].astype(int), op["thermal_installed_capacity_mw"]))
    for key, position in index.items():
        if key not in original:
            continue
        j = original[key]
        expected_request[:, position] = source["p_load_mw"][:, j]
        expected_available[:, position] = source["p_gen_available_mw"][:, j]
        if key in installed:
            base = float(source["nameplate_capacity_mw"][j])
            if base <= 0 and installed[key] > 0:
                raise ValueError("No original availability fraction for expanded zero-base thermal plant")
            expected_available[:, position] *= installed[key] / base if base > 0 else 0
    eq("exogenous_request_preserved", op["requested_load_mw"] - expected_request)
    eq("exogenous_availability_preserved", op["generation_available_mw"] - expected_available)

    charge = storage["charge_mw"].astype(float)
    discharge = storage["discharge_mw"].astype(float)
    soc = storage["soc_mwh"].astype(float)
    power = storage["site_power_capacity_mw"].astype(float)
    energy = storage["site_energy_capacity_mwh"].astype(float)
    sr = op["storage_reserve_mw"]
    tr = op["thermal_reserve_mw"]
    capacity = op["thermal_installed_capacity_mw"]
    pgen = op["generator_dispatch_mw"].astype(float).copy()
    served = op["served_load_mw"].astype(float).copy()
    for i, bus in enumerate(sites):
        pgen[:, bus] += discharge[:, i]
        served[:, bus] += charge[:, i]
    eq("physical_generator_plus_storage", pgen - flow["dispatched_generation_mw"])
    eq("exogenous_served_plus_charge", served - flow["served_load_mw"])
    eq("local_shed_preserved", op["unserved_load_mw"] - flow["unserved_load_mw"])
    eq("initial_soc_declaration", soc[0] - op["initial_soc_mwh"], "MWh")
    if not cfg.cyclic_state_of_charge:
        eq("storage_first_step_ramp_boundary", np.maximum(np.abs(discharge[0] - charge[0] - op["previous_storage_net_mw"])
           - cfg.storage_power_ramp_fraction_per_hour * power, 0))
        eq("thermal_first_step_ramp_boundary", np.maximum(np.abs(op["thermal_dispatch_mw"][0] - op["previous_thermal_mw"])
           - cfg.thermal_ramp_fraction_per_hour * capacity, 0))
    upper("thermal_with_reserve_availability", op["thermal_dispatch_mw"] + tr, cfg.thermal_operating_limit_ratio * op["thermal_available_mw"])
    upper("thermal_reserve_response", tr, cfg.thermal_ramp_fraction_per_hour * capacity * cfg.reserve_response_hours)
    upper("storage_reserve_inverter", charge + discharge + sr, power)
    upper("storage_reserve_energy", sr * cfg.reserve_duration_hours / cfg.discharge_efficiency,
          soc[1:] - cfg.minimum_soc_fraction * energy, "MWh")
    upper("storage_reserve_response", sr, cfg.storage_power_ramp_fraction_per_hour * power * cfg.reserve_response_hours)
    eq("no_charge_while_reserve_held", np.minimum(charge, sr), tolerance=1e-4)
    upper("thermal_land_capacity", capacity, op["thermal_land_limit_mw"])
    if thermal_ledger is not None:
        bounds = {int(row[0]): row[7] for row in thermal_ledger}
        eq("thermal_land_static_anchor", op["thermal_land_limit_mw"] - [bounds[int(i)] for i in storage["thermal_bus_ids"]])

    branch_rows = {int(row[0]): row for row in electrical["electrical_branches"]}
    ordered = np.asarray([branch_rows[int(i)] for i in storage["branch_ids"]]).reshape(-1, 10)
    enabled = op["branch_in_service"].astype(bool)
    eq("service_mask_flow_agreement", enabled.astype(int) - flow["op__branch_in_service"], "1", 0)
    eq("inactive_branch_zero_flow", np.where(enabled, 0, flow["line_flow_mw"]))
    expected_island = np.zeros((hours, len(ids)), dtype=int)
    expected_requirement = np.zeros((hours, len(ids)))
    balance, reserve_balance, island_times = [], [], []
    for t in range(hours):
        parent = list(range(len(ids)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for row, active in zip(ordered, enabled[t]):
            if active:
                a, b = root(index[int(row[1])]), root(index[int(row[2])])
                parent[b] = a
        groups = {}
        for i in range(len(ids)):
            groups.setdefault(root(i), []).append(i)
        for members in groups.values():
            members = np.asarray(members, dtype=int)
            expected_island[t, members] = ids[members].min()
            load = expected_request[t, members].sum()
            need = cfg.reserve_load_fraction * load + (cfg.reserve_contingency_mw if load > 0 else 0)
            # Kernel anchors reserve accounting at the lowest persistent bus ID.
            anchor = members[np.argmin(ids[members])]
            expected_requirement[t, anchor] = need
            balance.append(flow["bus_p_injection_mw"][t, members].sum())
            island_times.append(storage["timestamps"][t])
            provided = sr[t, np.isin(sites, members)].sum() + tr[t, np.isin(thermal, members)].sum()
            reserve_balance.append(need - provided - op["reserve_shortfall_mw"][t, members].sum())
    eq("island_labels_connectivity", op["island_id"] - expected_island, "id", 0)
    eq("island_labels_flow_agreement", flow["op__island_id"] - expected_island, "id", 0)
    eq("each_active_island_balance", np.asarray(balance),axes=("time",),timestamps=np.asarray(island_times),fields=["op__island_id","bus_p_injection_mw"],time_support="explicit_island_hour_observations_repeated_hour_values")
    eq("reserve_requirement_configuration", op["reserve_requirement_mw"] - expected_requirement,relation_class="S",axes=("time","bus"),timestamps=storage["timestamps"],fields=["op__reserve_requirement_mw","op__requested_load_mw","op__island_id"],time_support="hour_interval_reserve_rule")
    upper("each_island_reserve_adequacy_or_shortfall", np.asarray(reserve_balance), 0,axes=("time",),timestamps=np.asarray(island_times),fields=["op__storage_reserve_mw","op__thermal_reserve_mw","op__reserve_shortfall_mw","op__reserve_requirement_mw"],time_support="explicit_island_hour_observations_repeated_hour_values")
    upper("reserve_shortfall_within_requirement", op["reserve_shortfall_mw"], expected_requirement)
    angle_flow = np.zeros_like(flow["line_flow_mw"], dtype=float)
    for e, row in enumerate(ordered):
        a, b = index[int(row[1])], index[int(row[2])]
        angle_flow[:, e] = enabled[:, e] * row[3] ** 2 / row[6] * (flow["bus_angle_rad"][:, a] - flow["bus_angle_rad"][:, b])
    eq("dc_angle_flow_relation", flow["line_flow_mw"] - angle_flow,axes=("time","branch"),timestamps=storage["timestamps"],fields=["line_flow_mw","bus_angle_rad","op__branch_in_service","electrical_branches"],time_support="hour_interval_DC_state")
    if getattr(config, "planning", None) and config.planning.mode != "full_window_planning":
        for name in ("storage_power_expansion_mw", "storage_energy_expansion_mwh", "thermal_capacity_expansion_mw", "line_capacity_expansion_mva"):
            eq("fixed_assets_zero_" + name, storage[name], "MWh" if "mwh" in name else "MW", 1e-6)
    return {"mode": getattr(config.planning, "mode", "unspecified"), "network_model": "lossless_DC",
            "exogenous_unserved_mwh": float(op["unserved_load_mw"].sum()),
            "renewable_curtailment_mwh": float(op["renewable_curtailment_mw"].sum()),
            "thermal_unused_available_mwh": float(op["thermal_unused_available_mw"].sum()),
            "reserve_shortfall_mw_hours": float(op["reserve_shortfall_mw"].sum()),
            "max_simultaneous_islands": max(len(np.unique(row)) for row in expected_island),
            "inactive_branch_hours": int((~enabled).sum()), "metadata": meta}


@validation_context(stage="F_asset_planning_boundary",fields=["initial_assets.*","frozen_assets.*","asset_boundary.json"],time_support="fixed_asset_and_information_boundary",relation_class="S",engineering_simplification="Configured fixed/preplanned/oracle information policy; content hashes and fixed identity checks")
def check_frozen_asset_contracts(checks, world_dir, storage, flow, electrical, config):
    from world_generator.operation.stage_cache import load_asset_boundary_checkpoint
    boundary = load_asset_boundary_checkpoint(world_dir, expected_timestamps=storage["timestamps"],
        expected_bus_ids=flow["bus_ids"], expected_branch_ids=flow["branch_ids"])
    meta, frozen = boundary["metadata"], boundary["frozen_assets"]
    checks.condition("asset_planning_mode_matches_config", meta["mode"] == config.planning.mode)
    for name in ("electrical_buses", "electrical_branches"):
        checks.equal("asset_frozen_runtime_" + name, frozen[name] - electrical[name], 2e-3,
                     "serialized_electrical_units", "No test-period capacity or connectivity rewrite after freezing")
    checks.condition("asset_frozen_storage_site_order", np.array_equal(frozen["storage_site_ids"], storage["site_ids"]))
    checks.condition("asset_frozen_storage_bus_order", np.array_equal(frozen["storage_bus_ids"], storage["site_bus_ids"]))
    for original, runtime, unit in (("storage_power_mw", "site_power_capacity_mw", "MW"),
                                    ("storage_energy_mwh", "site_energy_capacity_mwh", "MWh")):
        checks.equal("asset_frozen_" + original, frozen[original] - storage[runtime], 2e-3, unit)
    if meta["mode"] == "fixed_assets":
        initial = boundary["initial_assets"]
        checks.condition("asset_fixed_identity_unchanged", set(initial) == set(frozen) and
                         all(initial[name].shape == frozen[name].shape and
                             np.allclose(initial[name], frozen[name], atol=2e-3, rtol=0) for name in initial))
    return meta
