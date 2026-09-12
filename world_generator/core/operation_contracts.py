"""F operation appendix names, dimensions and interval/boundary support."""
from __future__ import annotations

import json
import numpy as np

OPERATION_SCHEMA_VERSION = "operation_v1"
OPERATION_TEXT_FIELDS = ("operation_schema_version", "operation_field_schema_json", "operation_metadata_json")
PLANNING_MODES = ("fixed_assets", "preplanned", "full_window_planning")


def operation_field_schema(store_kind: str) -> dict[str, dict[str, object]]:
    """Explicit support is authoritative even when T equals N, E, S or K."""
    if store_kind == "power_flow":
        definitions = {
            "requested_load_mw": ("T,N", "MW", "interval_mean"),
            "generation_available_mw": ("T,N", "MW", "interval_mean"),
            "renewable_curtailment_mw": ("T,N", "MW", "interval_mean"),
            "thermal_backdown_mw": ("T,N", "MW", "interval_mean"),
            "thermal_unused_available_mw": ("T,N", "MW", "interval_mean"),
            "island_id": ("T,N", "id", "interval_topology_state"),
            "branch_in_service": ("T,E", "1", "interval_topology_state"),
        }
    elif store_kind == "storage_dispatch":
        definitions = {name:("T,N", "MW", "interval_mean") for name in (
            "requested_load_mw", "served_load_mw", "unserved_load_mw", "generation_available_mw",
            "generator_dispatch_mw", "renewable_curtailment_mw", "thermal_backdown_mw",
            "thermal_unused_available_mw", "reserve_requirement_mw", "reserve_shortfall_mw")}
        definitions.update({
            "bus_ids": ("N", "id", "fixed_entity_identity"),
            "bus_kinds": ("N", "category", "fixed_entity_identity"),
            "island_id": ("T,N", "id", "interval_topology_state"),
            "branch_in_service": ("T,E", "1", "interval_topology_state"),
            "storage_reserve_mw": ("T,S", "MW", "interval_available_reserve"),
            "thermal_reserve_mw": ("T,K", "MW", "interval_available_reserve"),
            "thermal_available_mw": ("T,K", "MW", "interval_mean"),
            "thermal_dispatch_mw": ("T,K", "MW", "interval_mean"),
            "initial_soc_mwh": ("S", "MWh", "initial_boundary_state"),
            "previous_storage_net_mw": ("S", "MW", "previous_interval_mean"),
            "previous_thermal_mw": ("K", "MW", "previous_interval_mean"),
            "thermal_land_limit_mw": ("K", "MW", "fixed_asset_limit"),
            "thermal_installed_capacity_mw": ("K", "MW", "fixed_asset_capacity"),
        })
    else:
        raise ValueError(f"Unknown operation store kind: {store_kind}")
    result = {name: {"shape": shape, "unit": unit, "time_kind": support,
                   "relation_class": "S" if support in {"fixed_entity_identity","fixed_asset_limit","fixed_asset_capacity","initial_boundary_state","previous_interval_mean"} or name == "reserve_requirement_mw" else "P",
                   "information_role": "realized_operation_or_boundary_not_planning_input"
                   if support not in {"fixed_entity_identity", "fixed_asset_limit", "fixed_asset_capacity"} else "fixed_asset_attribute"}
            for name, (shape, unit, support) in definitions.items()}
    for name,spec in result.items():
        if name in {"bus_ids","bus_kinds","island_id","branch_in_service"}:
            spec["relation_note"] = "identity_or_topology_definition; no constitutive physical law implied"
        elif "reserve" in name:
            spec["generation_relation_type"] = "S"
            spec["accounting_relation_type"] = "P"
            spec["relation_note"] = "Engineering reserve service rule subject to physical capacity, energy and ramp bounds; island aggregate is not network deliverability"
        elif spec["relation_class"] == "S":
            spec["relation_note"] = "Configured, inherited or frozen scenario boundary/asset input; not a derived physical law"
        else:
            spec["relation_note"] = "Physical power accounting or realized quantity; dispatch choice comes from engineering optimization"
    return result


def operation_appendix_arrays(store_kind: str, arrays: dict[str, np.ndarray], metadata: dict[str, object],
                              *, timestamps: np.ndarray, bus_ids: np.ndarray | None = None,
                              branch_ids: np.ndarray, site_ids: np.ndarray | None = None,
                              thermal_bus_ids: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if not arrays and not metadata:
        return {}
    schema = operation_field_schema(store_kind)
    if set(arrays) != set(schema):
        raise ValueError(f"Incomplete {store_kind} operation appendix: missing={sorted(set(schema)-set(arrays))}, extra={sorted(set(arrays)-set(schema))}")
    if metadata.get("schema_version") != OPERATION_SCHEMA_VERSION or metadata.get("store_kind") != store_kind:
        raise ValueError("Operation metadata must declare the matching schema_version and store_kind")
    from world_generator.core.contracts import entity_ids, interval_bounds_hours
    stamps = np.asarray(timestamps)
    bounds = interval_bounds_hours(stamps, 1.0, stamp_unit="hour")
    if "duration_hours" in metadata and metadata["duration_hours"] != 1.0:
        raise ValueError("Operation appendix currently supports one-hour intervals")
    ids = entity_ids(arrays["bus_ids"] if store_kind == "storage_dispatch" else bus_ids, "operation bus_ids")
    branches = entity_ids(branch_ids, "operation branch_ids")
    sites = entity_ids(site_ids, "operation site_ids") if site_ids is not None else np.empty(0)
    thermal = entity_ids(thermal_bus_ids, "operation thermal_bus_ids") if thermal_bus_ids is not None else np.empty(0)
    sizes = {"T":len(stamps),"N":len(ids),"E":len(branches),"S":len(sites),"K":len(thermal)}
    if not set(thermal).issubset(set(ids)):
        raise ValueError("Thermal IDs must be a subset of operation bus IDs")
    result = {}
    for name, spec in schema.items():
        value = np.asarray(arrays[name])
        shape = tuple(sizes[dim] for dim in str(spec["shape"]).split(","))
        if value.shape != shape:
            raise ValueError(f"Operation {name} must have explicit shape {shape}, got {value.shape}")
        if name == "bus_kinds":
            if value.dtype.kind not in "US" or not np.isin(value, ("load_bus","wind_bus","pv_bus","thermal_bus","transit_bus","storage_bus")).all():
                raise ValueError("Operation bus_kinds contains unknown or non-string labels")
        else:
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"Operation {name} must be finite numeric values")
            if name in {"bus_ids","island_id"} and (np.any(value < 0) or np.any(value != np.floor(value))):
                raise ValueError(f"Operation {name} must contain nonnegative integer IDs")
            if name == "branch_in_service" and not np.isin(value, (0,1)).all():
                raise ValueError("branch_in_service must be boolean")
            if spec["unit"] in {"MW","MWh"} and name != "previous_storage_net_mw" and np.any(value < -1e-7):
                raise ValueError(f"Operation {name} must be nonnegative")
        result[f"op__{name}"] = value
    result.update({"operation_schema_version":np.asarray(OPERATION_SCHEMA_VERSION),
                   "operation_field_schema_json":np.asarray(json.dumps(schema,sort_keys=True)),
                   "operation_metadata_json":np.asarray(json.dumps(metadata,sort_keys=True)),
                   "operation_time_bounds_hours":bounds})
    return result


def decode_operation_appendix(payload: dict[str, np.ndarray], store_kind: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    modern = any(name.startswith("op__") or name in (*OPERATION_TEXT_FIELDS,"operation_time_bounds_hours") for name in payload)
    if not modern:
        return {}, {}
    required = {f"op__{name}" for name in operation_field_schema(store_kind)} | set(OPERATION_TEXT_FIELDS) | {"operation_time_bounds_hours"}
    actual = {name for name in payload if name.startswith("op__") or name in (*OPERATION_TEXT_FIELDS,"operation_time_bounds_hours")}
    if actual != required or str(payload["operation_schema_version"]) != OPERATION_SCHEMA_VERSION:
        raise ValueError("Incomplete or unsupported operation appendix")
    if json.loads(str(payload["operation_field_schema_json"])) != operation_field_schema(store_kind):
        raise ValueError("Operation field schema differs from explicit supported fields")
    metadata = json.loads(str(payload["operation_metadata_json"]))
    if not isinstance(metadata, dict):
        raise ValueError("Operation metadata must be a mapping")
    return {name: np.asarray(payload[f"op__{name}"]) for name in operation_field_schema(store_kind)}, metadata


def validate_operation_time_bounds(payload: dict[str, np.ndarray], checked: dict[str, np.ndarray]) -> None:
    if "operation_time_bounds_hours" in checked:
        if set(payload) != set(checked):
            raise ValueError("Unknown or missing fields in modern operation store")
        declared = np.asarray(payload["operation_time_bounds_hours"])
        expected = checked["operation_time_bounds_hours"]
        if declared.shape != expected.shape or not np.array_equal(declared, expected):
            raise ValueError("Operation interval bounds differ from timestamps")
        kind = json.loads(str(checked["operation_metadata_json"]))["store_kind"]
        validate_operation_serialized_shapes(checked,kind)


def validate_storage_operation_states(payload: dict[str, np.ndarray]) -> None:
    hours, sites = len(payload["timestamps"]), len(payload["site_ids"])
    for name in ("charge_mw","discharge_mw","emergency_discharge_mw"):
        value = np.asarray(payload[name])
        if value.shape != (hours,sites) or not np.isfinite(value).all() or np.any(value < -1e-7):
            raise ValueError(f"{name} must be nonnegative interval mean [T,S]")
    soc = np.asarray(payload["soc_mwh"])
    if soc.shape != (hours+1,sites) or not np.isfinite(soc).all():
        raise ValueError("soc_mwh must contain T+1 finite boundary states")
    if not np.allclose(soc[0],payload["op__initial_soc_mwh"],rtol=1e-6,atol=1e-5):
        raise ValueError("Initial SOC declaration differs from first SOC boundary")


def operation_store_field_schema(store_kind: str) -> dict[str, dict[str, object]]:
    """Full serialized field support, including retained historical channels."""
    base = {
        "timestamps":("T","h","interval_start"), "branch_ids":("E","id","fixed_entity_identity"),
    }
    if store_kind == "power_flow":
        base.update({"bus_ids":("N","id","fixed_entity_identity"),"slack_bus_id":("scalar","id","legacy_reference_identity")})
        base.update({name:("T,N","rad" if name=="bus_angle_rad" else "MW","interval_mean") for name in (
            "bus_angle_rad","bus_p_injection_mw","served_load_mw","dispatched_generation_mw","unserved_load_mw","curtailed_generation_mw")})
        base.update({"line_flow_mw":("T,E","MW","interval_mean"),"line_loading_ratio":("T,E","1","interval_mean")})
    elif store_kind == "storage_dispatch":
        base.update({name:("S",unit,"fixed_asset_attribute") for name,unit in {
            "site_ids":"id","site_bus_ids":"id","site_power_capacity_mw":"MW","site_energy_capacity_mwh":"MWh",
            "storage_power_expansion_mw":"MW","storage_energy_expansion_mwh":"MWh"}.items()})
        base.update({name:("T,S","MW","interval_mean") for name in ("charge_mw","discharge_mw","emergency_discharge_mw")})
        base.update({"soc_mwh":("T+1,S","MWh","boundary_state"),"target_soc_mwh":("T,S","MWh","interval_end_target"),
                     "cycle_boundary_soc_mwh":("S","MWh","world_terminal_target"),
                     "thermal_bus_ids":("K","id","fixed_entity_identity"),"thermal_capacity_expansion_mw":("K","MW","fixed_asset_attribute"),
                     "line_capacity_expansion_mva":("E","MVA","fixed_asset_attribute"),
                     "baseline_line_loading_ratio":("T,E","1","interval_mean"),"dispatched_line_loading_ratio":("T,E","1","interval_mean")})
        base.update({name:("scalar","1","fixed_operation_parameter") for name in (
            "minimum_soc_fraction","maximum_soc_fraction","preferred_soc_lower_fraction","preferred_soc_upper_fraction")})
        base.update({name:("T","MW","interval_mean") for name in (
            "total_load_mw","renewable_available_mw","baseline_thermal_mw","scheduled_thermal_mw",
            "baseline_unserved_mw","dispatched_unserved_mw","baseline_curtailed_mw","dispatched_curtailed_mw")})
    else:
        raise ValueError("Unknown operation store kind")
    result = {name:{"shape":shape,"unit":unit,"time_kind":support} for name,(shape,unit,support) in base.items()}
    result.update({f"op__{name}":spec for name,spec in operation_field_schema(store_kind).items()})
    result.update({name:{"shape":"scalar","unit":"text","time_kind":"metadata"} for name in OPERATION_TEXT_FIELDS})
    result["operation_time_bounds_hours"] = {"shape":"T,2","unit":"h","time_kind":"interval_bounds"}
    return result


def validate_operation_serialized_shapes(payload: dict[str,np.ndarray], store_kind: str) -> None:
    schema = operation_store_field_schema(store_kind)
    if set(schema) != set(payload):
        raise ValueError("Modern operation store does not match complete explicit field support")
    sizes = {"T":len(payload["timestamps"]),"T+1":len(payload["timestamps"])+1,"2":2,
             "N":len(payload["op__bus_ids"] if store_kind=="storage_dispatch" else payload["bus_ids"]),
             "E":len(payload["branch_ids"]),"S":len(payload.get("site_ids",[])),"K":len(payload.get("thermal_bus_ids",[]))}
    for name,spec in schema.items():
        expected = () if spec["shape"] == "scalar" else tuple(sizes[dim] for dim in str(spec["shape"]).split(","))
        value = np.asarray(payload[name])
        if value.shape != expected:
            raise ValueError(f"Serialized operation {name} must have shape {expected}")
        if spec["unit"] not in {"text","category"} and (value.dtype.kind not in "biuf" or not np.isfinite(value).all()):
            raise ValueError(f"Serialized operation {name} must be finite numeric values")
