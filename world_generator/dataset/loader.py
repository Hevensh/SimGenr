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
        return {
            "sample_id": str(entry["sample_id"]),
            "seed": int(entry["seed"]) if entry.get("seed") is not None else None,
            "partition": str(entry.get("partition", "unspecified")),
            "static": _extract_group(payload, "static"),
            "land": _extract_group(payload, "land"),
            "dynamic": _extract_group(payload, "dynamic"),
            "graph": _extract_group(payload, "graph"),
            "operation": _extract_group(payload, "operation"),
            "metadata": json.loads(str(payload["metadata_json"])),
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
        window = {
            "sample_id": world["sample_id"],
            "seed": world["seed"],
            "window_start": start,
            "static": world["static"],
            "land": world.get("land", {}),
            "graph": world["graph"],
            "history": {
                **_slice_weather(world["dynamic"], start, history_end, hours),
                "operation": history_operation,
            },
            "future": {
                **_slice_weather(world["dynamic"], history_end, forecast_end, hours),
                "operation": future_operation,
            },
            "operation_static": static_operation,
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
