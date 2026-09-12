from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


class SimGenrDataset:
    """Lazy reader for complete multimodal world samples."""

    def __init__(
        self,
        root: str | Path,
        *,
        sample_ids: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
        as_torch: bool = False,
        cache_size: int = 1,
        verify_checksums: bool = False,
        device: object | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        entries = list(self.manifest["samples"])
        if sample_ids is not None:
            allowed = set(sample_ids)
            entries = [entry for entry in entries if entry["sample_id"] in allowed]
        if seeds is not None:
            allowed_seeds = {int(seed) for seed in seeds}
            entries = [entry for entry in entries if entry.get("seed") is not None and int(entry["seed"]) in allowed_seeds]
        self.entries = entries
        self.as_torch = bool(as_torch)
        self.cache_size = max(int(cache_size), 0)
        self.verify_checksums = bool(verify_checksums)
        self.device = device
        self.transform = transform
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample_id = str(entry["sample_id"])
        if sample_id in self._cache:
            sample = self._cache.pop(sample_id)
            self._cache[sample_id] = sample
        else:
            sample = self._load_sample(entry)
            if self.as_torch:
                sample = _to_torch(sample, self.device)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.cache_size > 0:
                self._cache[sample_id] = sample
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return sample

    def preload(self) -> "SimGenrDataset":
        """Load every selected world into the configured cache and device."""
        if self.cache_size < len(self.entries):
            raise ValueError("cache_size must cover every selected world before preload()")
        for index in range(len(self.entries)):
            self[index]
        return self

    def cached_tensor_bytes(self) -> int:
        return _unique_tensor_storage_bytes(self._cache)

    def sample_ids(self) -> list[str]:
        return [str(entry["sample_id"]) for entry in self.entries]

    def _load_sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        sample_path = self.root / "samples" / str(entry["file"]["name"])
        if self.verify_checksums:
            _verify_file(sample_path, entry["file"])
        payload = _load_npz(sample_path)
        metadata = json.loads(str(payload["metadata_json"]))
        hydrology = _extract_group(payload, "hydrology")
        source_load = _extract_group(payload, "source_load")
        operation_detail = _extract_group(payload,"operation_detail")
        power_flow_detail = _extract_group(payload,"power_flow_detail")
        asset_planning = _extract_group(payload,"asset_planning")
        operation_mode = metadata.get("operation_detail",{}).get("mode","legacy_no_appendix")
        if operation_mode not in {"legacy_no_appendix","operation_v1"} or (operation_mode == "operation_v1") != bool(operation_detail) or bool(operation_detail) != bool(power_flow_detail) or bool(operation_detail) != bool(asset_planning):
            raise ValueError("Dataset F operation/asset appendices must be complete and explicitly declared")
        if operation_detail:
            from world_generator.core.datatypes import StorageDispatchStore, PowerFlowStore
            storage_store = StorageDispatchStore.from_arrays(operation_detail)
            power_store = PowerFlowStore.from_arrays(power_flow_detail)
            if not np.array_equal(storage_store.timestamps,payload["dynamic__timestamps"]) or not np.array_equal(power_store.timestamps,storage_store.timestamps):
                raise ValueError("Dataset runtime operation time differs from weather")
            if not np.array_equal(storage_store.branch_ids,power_store.branch_ids) or not np.array_equal(storage_store.operation_arrays["bus_ids"],power_store.bus_ids):
                raise ValueError("Dataset runtime stores do not share fixed entity IDs")
            _validate_asset_planning_group(asset_planning,storage_store.operation_arrays["bus_ids"],storage_store.branch_ids)
        source_declaration = metadata.get("source_load", {})
        if not isinstance(source_declaration, dict):
            raise ValueError("Dataset source/load metadata must be a mapping")
        source_mode = source_declaration.get("mode", "legacy_no_diagnostics")
        if source_mode not in {"legacy_no_diagnostics", "exogenous_realization"} or (source_mode == "exogenous_realization") != bool(source_load):
            raise ValueError("Dataset source/load appendix differs from its declared mode")
        if source_load:
            from world_generator.core.datatypes import SourceLoadForecastStore
            source_store = SourceLoadForecastStore.from_arrays(source_load)
            if not np.array_equal(source_store.timestamps, payload["dynamic__timestamps"]):
                raise ValueError("Dataset source/load timestamps differ from weather")
            if tuple(source_store.metadata["weather_grid_shape"]) != payload["dynamic__weather"].shape[-2:]:
                raise ValueError("Dataset source/load sampling grid differs from weather")
        hydrology_declaration = metadata.get("hydrology", {})
        if not isinstance(hydrology_declaration, dict):
            raise ValueError("Dataset hydrology metadata must be a mapping")
        hydrology_mode = hydrology_declaration.get("mode", "legacy_static_only")
        if hydrology_mode not in {"bucket_routing_v1", "static_only", "legacy_static_only"} or (hydrology_mode == "bucket_routing_v1") != bool(hydrology):
            raise ValueError("Dataset dynamic hydrology arrays do not match their declared mode")
        if hydrology:
            from world_generator.core.datatypes import HydrologyTimeSeriesStore
            store = HydrologyTimeSeriesStore.from_arrays(hydrology)
            if not np.array_equal(store.timestamps, payload["dynamic__timestamps"]):
                raise ValueError("Dataset hydrology timestamps differ from weather")
            if store.states["soil_storage_mm"].shape[1:] != payload["dynamic__weather"].shape[-2:]:
                raise ValueError("Dataset hydrology grid differs from weather")
        return {
            "sample_id": str(entry["sample_id"]),
            "seed": int(entry["seed"]) if entry.get("seed") is not None else None,
            "partition": str(entry.get("partition", "unspecified")),
            "static": _extract_group(payload, "static"),
            "land": _extract_group(payload, "land"),
            "hydrology": hydrology,
            "source_load": source_load,
            "operation_detail":operation_detail,
            "power_flow_detail":power_flow_detail,
            "asset_planning":asset_planning,
            "dynamic": _extract_group(payload, "dynamic"),
            "graph": _extract_group(payload, "graph"),
            "operation": _extract_group(payload, "operation"),
            "metadata": metadata,
            "config_yaml": str(payload["config_yaml"]),
            "sample_path": sample_path,
        }


class TemporalWindowDataset:
    """View worlds as history/target windows, including optional exogenous data.

    `future.weather` is realized weather; a caller must not present it as NWP
    available at forecast issue time. The final graph and dispatch can also
    depend on the complete operation period; consult sample metadata.
    """

    def __init__(
        self,
        worlds: SimGenrDataset,
        *,
        history_hours: int = 24,
        forecast_hours: int = 6,
        stride: int = 6,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache_size: int = 0,
    ) -> None:
        self.worlds = worlds
        self.history_hours = int(history_hours)
        self.forecast_hours = int(forecast_hours)
        self.stride = int(stride)
        self.transform = transform
        self.cache_size = max(int(cache_size), 0)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        if min(self.history_hours, self.forecast_hours, self.stride) <= 0:
            raise ValueError("history_hours, forecast_hours, and stride must be positive")
        self.windows: list[tuple[int, int]] = []
        for world_index, entry in enumerate(worlds.entries):
            hours = int(entry["hours"])
            last_start = hours - self.history_hours - self.forecast_hours
            self.windows.extend((world_index, start) for start in range(0, last_start + 1, self.stride))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index in self._cache:
            window = self._cache.pop(index)
            self._cache[index] = window
            return window
        world_index, start = self.windows[index]
        world = self.worlds[world_index]
        history_end = start + self.history_hours
        forecast_end = history_end + self.forecast_hours
        hours = int(world["dynamic"]["timestamps"].shape[0])
        history_operation, future_operation, static_operation = _split_operation(
            world["operation"], start, history_end, forecast_end, hours
        )
        history_hydrology, future_hydrology, static_hydrology = _split_hydrology(
            world.get("hydrology", {}), start, history_end, forecast_end, hours
        )
        history_source, future_source, static_source = _split_source_load(
            world.get("source_load", {}), start, history_end, forecast_end, hours
        )
        history_detail,future_detail,static_detail = _split_operation_detail(world.get("operation_detail",{}),start,history_end,forecast_end,hours,"storage_dispatch")
        history_power,future_power,static_power = _split_operation_detail(world.get("power_flow_detail",{}),start,history_end,forecast_end,hours,"power_flow")
        window = {
            "sample_id": world["sample_id"],
            "seed": world["seed"],
            "window_start": start,
            "static": world["static"],
            "land": world.get("land", {}),
            "graph": world["graph"],
            "asset_planning": world.get("asset_planning",{}),
            "history": {
                **_slice_weather(world["dynamic"], start, history_end, hours),
                "operation": history_operation,
                "hydrology": history_hydrology,
                "source_load": history_source,
                "operation_detail":history_detail,
                "power_flow_detail":history_power,
            },
            "future": {
                **_slice_weather(world["dynamic"], history_end, forecast_end, hours),
                "operation": future_operation,
                "hydrology": future_hydrology,
                "source_load": future_source,
                "operation_detail":future_detail,
                "power_flow_detail":future_power,
            },
            "operation_static": static_operation,
            "hydrology_static": static_hydrology,
            "source_load_static": static_source,
            "operation_detail_static":static_detail,
            "power_flow_detail_static":static_power,
            "metadata": world["metadata"],
            "prepared_static": world.get("prepared_static"),
        }
        if self.transform is not None:
            window = self.transform(window)
        if self.cache_size > 0:
            self._cache[index] = window
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return window

    def preload(self) -> "TemporalWindowDataset":
        """Build and retain every transformed temporal window."""
        if self.cache_size < len(self.windows):
            raise ValueError("cache_size must cover every temporal window before preload()")
        for index in range(len(self.windows)):
            self[index]
        return self

    def cached_tensor_bytes(self) -> int:
        return _unique_tensor_storage_bytes(self._cache)


def make_dataloader(
    dataset: object,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs: Any,
) -> object:
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required for make_dataloader; use dataset[index] for NumPy loading") from exc
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        **kwargs,
    )


def collate_multimodal(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    result = _collate_values(samples)
    if isinstance(result, dict) and "graph" in result:
        result["graph"] = [sample["graph"] for sample in samples]
    return result


def channel_index(channel_names: Iterable[object], name: str) -> int:
    names = [str(value) for value in channel_names]
    if name not in names:
        raise KeyError(f"Unknown channel {name!r}; available channels: {names}")
    return names.index(name)


def edge_path(graph: dict[str, Any], edge_position: int) -> tuple[Any, Any]:
    start = int(graph["path_ptr"][edge_position])
    end = int(graph["path_ptr"][edge_position + 1])
    return graph["path_row"][start:end], graph["path_col"][start:end]


def _slice_weather(dynamic: dict[str, Any], start: int, end: int, hours: int) -> dict[str, Any]:
    # Explicit field contract: H==T cannot make a static elevation map temporal.
    temporal = {"timestamps", "weather", "weather_class", "time_bounds_hours"}
    result = {}
    for name, values in dynamic.items():
        if name in temporal or name.startswith("diagnostic__"):
            if not hasattr(values, "shape") or not values.shape or values.shape[0] != hours:
                raise ValueError(f"Weather time series {name!r} does not match the time axis")
            result[name] = values[start:end]
    return result


def _split_source_load(
    source: dict[str, Any], start: int, history_end: int, forecast_end: int, hours: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """E names define support; thermal initial state and period totals follow each window."""
    from world_generator.core.source_load_contracts import (
        SOURCE_LOAD_STATIC_UNITS, SOURCE_LOAD_DIAGNOSTIC_UNITS, SOURCE_LOAD_ENERGY_POWER_FIELDS,
        SOURCE_LOAD_CF_POWER_FIELDS, SOURCE_LOAD_BASE_INTERVAL_FIELDS,
    )
    if not source:
        return {}, {}, {}
    initial = "initial_effective_temperature_c"
    intervals = set(SOURCE_LOAD_BASE_INTERVAL_FIELDS) | set(SOURCE_LOAD_ENERGY_POWER_FIELDS) | set(SOURCE_LOAD_CF_POWER_FIELDS)
    intervals |= {f"diag__{name}" for name in SOURCE_LOAD_DIAGNOSTIC_UNITS} | {"timestamps", "time_bounds_hours"}
    period = {f"period_{name}" for name in SOURCE_LOAD_ENERGY_POWER_FIELDS}
    static = (set(SOURCE_LOAD_STATIC_UNITS) - {initial}) | {"bus_ids", "bus_kinds", "source_channels", "data_semantics", "source_load_schema_version", "source_load_field_schema_json", "source_load_metadata_json"}
    if set(source) != intervals | period | static | {initial}:
        raise ValueError("Window source/load appendix has missing or unknown fields")
    history, future, shared = {}, {}, {}
    for name in intervals:
        values = source[name]
        if not hasattr(values, "shape") or not values.shape or values.shape[0] != hours:
            raise ValueError(f"Source/load interval field {name} must have T entries")
        history[name], future[name] = values[start:history_end], values[history_end:forecast_end]
    for name in SOURCE_LOAD_ENERGY_POWER_FIELDS:
        history[f"period_{name}"] = history[name].sum(axis=0)
        future[f"period_{name}"] = future[name].sum(axis=0)
    thermal = source["diag__load_effective_temperature_c"]
    history[initial] = source[initial] if start == 0 else thermal[start - 1]
    future[initial] = thermal[history_end - 1]
    shared.update({name: source[name] for name in static})
    return history, future, shared


def _split_hydrology(
    hydrology: dict[str, Any], start: int, history_end: int, forecast_end: int, hours: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Explicit D support: H==T cannot make a lake geometry map temporal."""
    from world_generator.core.hydrology_contracts import HYDROLOGY_GROUP_UNITS

    if not hydrology:
        return {}, {}, {}
    states = {f"state__{name}" for name in HYDROLOGY_GROUP_UNITS["state"]} | {"state_time_hours"}
    intervals = {f"{group}__{name}" for group in ("flux", "budget") for name in HYDROLOGY_GROUP_UNITS[group]}
    intervals |= {"timestamps", "time_bounds_hours"}
    static = {f"static__{name}" for name in HYDROLOGY_GROUP_UNITS["static"]}
    static |= {"mode", "schema_version", "field_schema_json", "hydrology_metadata_json"}
    if set(hydrology) != states | intervals | static:
        raise ValueError("Window hydrology has incomplete or unknown fields")
    history, future, shared = {}, {}, {}
    for name, values in hydrology.items():
        if name in states:
            if not hasattr(values, "shape") or not values.shape or values.shape[0] != hours + 1:
                raise ValueError(f"Hydrology boundary state {name} must have T+1 entries")
            history[name], future[name] = values[start:history_end + 1], values[history_end:forecast_end + 1]
        elif name in intervals:
            if not hasattr(values, "shape") or not values.shape or values.shape[0] != hours:
                raise ValueError(f"Hydrology interval field {name} must have T entries")
            history[name], future[name] = values[start:history_end], values[history_end:forecast_end]
        else:
            shared[name] = values
    return history, future, shared


def _validate_asset_planning_group(group:dict[str,Any], bus_ids:np.ndarray, branch_ids:np.ndarray) -> None:
    from world_generator.operation.asset_planning import asset_snapshot_sha256
    from world_generator.core.operation_contracts import PLANNING_MODES
    if "boundary_metadata_json" not in group:
        raise ValueError("Dataset assets lack planning information boundary")
    metadata = json.loads(str(group["boundary_metadata_json"]))
    if metadata.get("schema_version") != "asset_planning_v1" or metadata.get("mode") not in PLANNING_MODES:
        raise ValueError("Dataset assets have unknown planning mode/schema")
    for label in ("initial_assets","frozen_assets"):
        arrays = {name.removeprefix(label+"__"):np.asarray(value) for name,value in group.items() if name.startswith(label+"__")}
        if not arrays or asset_snapshot_sha256(arrays) != metadata.get(label+"_sha256"):
            raise ValueError("Dataset asset snapshot hash mismatch")
        if label == "frozen_assets" and (not np.array_equal(arrays.get("bus_ids"),bus_ids) or not np.array_equal(arrays.get("branch_ids"),branch_ids)):
            raise ValueError("Dataset frozen asset IDs differ from operation IDs")


def _split_operation_detail(payload:dict[str,Any],start:int,history_end:int,forecast_end:int,hours:int,store_kind:str) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    if not payload:
        return {},{},{}
    from world_generator.core.operation_contracts import operation_store_field_schema
    schema = operation_store_field_schema(store_kind)
    if set(payload) != set(schema):
        raise ValueError("Operation detail lacks explicit complete field support")
    history,future,shared = {},{},{}
    boundary_fields = {"op__initial_soc_mwh","op__previous_storage_net_mw","op__previous_thermal_mw"}
    for name,value in payload.items():
        if name in boundary_fields:
            continue
        support = schema[name]["shape"]
        if str(support).split(",")[0] == "T":
            if value.shape[0] != hours:
                raise ValueError(f"Operation detail {name} differs from weather time axis")
            history[name],future[name] = value[start:history_end],value[history_end:forecast_end]
        elif str(support).split(",")[0] == "T+1":
            if value.shape[0] != hours+1:
                raise ValueError(f"Operation detail {name} lacks terminal boundary")
            history[name],future[name] = value[start:history_end+1],value[history_end:forecast_end+1]
        else:
            shared[name] = value
    if store_kind == "storage_dispatch":
        def boundaries(index:int) -> dict[str,Any]:
            return {"op__initial_soc_mwh":payload["soc_mwh"][index],
                    "op__previous_storage_net_mw":payload["op__previous_storage_net_mw"] if index==0 else payload["discharge_mw"][index-1]-payload["charge_mw"][index-1],
                    "op__previous_thermal_mw":payload["op__previous_thermal_mw"] if index==0 else payload["op__thermal_dispatch_mw"][index-1]}
        history.update(boundaries(start)); future.update(boundaries(history_end))
    return history,future,shared


def _split_operation(
    operation: dict[str, Any],
    start: int,
    history_end: int,
    forecast_end: int,
    hours: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    history: dict[str, Any] = {}
    future: dict[str, Any] = {}
    static: dict[str, Any] = {}
    temporal_fields = {
        "timestamps", "node_dynamic", "line_dynamic",
        "exogenous_p_load_mw", "exogenous_p_gen_available_mw", "exogenous_p_renewable_available_mw",
        "charge_mw", "discharge_mw", "emergency_discharge_mw", "target_soc_mwh",
        "total_load_mw", "renewable_available_mw", "baseline_thermal_mw", "scheduled_thermal_mw",
        "baseline_unserved_mw", "dispatched_unserved_mw", "baseline_curtailed_mw", "dispatched_curtailed_mw",
    }
    for name, values in operation.items():
        if name in temporal_fields:
            if not hasattr(values, "shape") or len(values.shape) == 0 or values.shape[0] != hours:
                raise ValueError(f"Operation time series {name!r} does not match the weather time axis")
            history[name] = values[start:history_end]
            future[name] = values[history_end:forecast_end]
        elif name == "soc_mwh":
            if values.shape[0] != hours + 1:
                raise ValueError("soc_mwh must include the terminal boundary state")
            history[name] = values[start : history_end + 1]
            future[name] = values[history_end : forecast_end + 1]
        else:
            static[name] = values
    return history, future, static


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: _decode_array(payload[name]) for name in payload.files}


def _decode_array(values: np.ndarray) -> Any:
    if values.dtype.kind in {"U", "S"}:
        if values.ndim == 0:
            return str(values.item())
        return tuple(str(value) for value in values.tolist())
    return values.copy()


def _extract_group(payload: dict[str, Any], group: str) -> dict[str, Any]:
    prefix = f"{group}__"
    return {name[len(prefix) :]: values for name, values in payload.items() if name.startswith(prefix)}


def _to_torch(value: Any, device: object | None = None) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required when as_torch=True") from exc
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        return tensor.to(device) if device is not None else tensor
    if isinstance(value, dict):
        return {key: _to_torch(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_torch(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_torch(item, device) for item in value)
    return value


def _tensor_bytes(value: Any) -> int:
    try:
        import torch
    except ImportError:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _unique_tensor_storage_bytes(value: Any) -> int:
    try:
        import torch
    except ImportError:
        return 0
    seen: set[tuple[str, int]] = set()
    total = 0

    def visit(item: Any) -> None:
        nonlocal total
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), int(storage.data_ptr()))
            if key not in seen:
                seen.add(key)
                total += int(storage.nbytes())
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return total


def _collate_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        if any(not isinstance(value, dict) or set(value) != set(first) for value in values):
            # Mixed legacy/new samples may lack exogenous targets. Preserve
            # those records separately instead of dropping fields or inventing zeros.
            return values
        return {key: _collate_values([value[key] for value in values]) for key in first}
    if _is_torch_tensor(first):
        import torch

        if all(tuple(value.shape) == tuple(first.shape) for value in values):
            return torch.stack(values, dim=0)
        return values
    if isinstance(first, np.ndarray):
        if all(value.shape == first.shape for value in values):
            return np.stack(values, axis=0)
        return values
    if isinstance(first, (int, float, np.number)):
        return np.asarray(values)
    return values


def _is_torch_tensor(value: Any) -> bool:
    # An actual tensor implies torch is already loaded. NumPy-only users must
    # not initialize the heavyweight optional runtime just to collate a string.
    torch = sys.modules.get("torch")
    return torch is not None and hasattr(torch, "Tensor") and isinstance(value, torch.Tensor)


def _verify_file(path: Path, item: dict[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
