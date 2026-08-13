from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.core.output_layout import WorldDataLayout


STAGE_FILES = {
    "weather": ("daily_weather.npz", "hourly_weather_week.npz"),
    "energy": ("source_load_candidates.npz", "energy_sites.json"),
    "buses": ("grid_nodes.npz", "bus_sites.json"),
    "topology": (
        "static_maps.npz",
        "grid_topology.npz",
        "refined_grid_topology.npz",
        "grid_electrical.npz",
        "grid_edges.json",
        "refined_bus_sites.json",
        "refined_grid_edges.json",
        "grid_electrical.json",
    ),
    "operation": (
        "source_load_forecast.npz",
        "power_flow_hourly.npz",
        "grid_upgrade_plan.npz",
        "source_load_forecast.json",
        "power_flow_hourly.json",
        "grid_upgrade_plan.json",
    ),
    "grid_update": ("grid_update_loop.json",),
    "storage_planning": ("storage_need.npz", "storage_plan.npz", "storage_need.json", "storage_plan.json"),
    "storage_dispatch": (
        "storage_dispatch.npz",
        "storage_dispatch_forecast.npz",
        "storage_dispatch_power_flow.npz",
        "storage_dispatch_electrical.npz",
        "storage_dispatch.json",
    ),
}

STAGE12_CHECKPOINT_FILES = (
    "actions.json",
    "bus_index.json",
    "power_flow_hourly.npz",
    "grid_upgrade_plan.npz",
    "grid_electrical.npz",
    "refined_grid_topology.npz",
    "refined_grid_edges.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Move legacy flat world outputs into stage-based data folders.")
    parser.add_argument("roots", nargs="*", default=["outputs"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    worlds = _find_worlds([Path(root) for root in args.roots])
    moved = 0
    for world_dir in tqdm(worlds, desc="Migrating outputs", unit="world", dynamic_ncols=True):
        moved += migrate_world(world_dir, dry_run=args.dry_run)
    action = "Would move" if args.dry_run else "Moved"
    print(f"{action} {moved} files across {len(worlds)} worlds.")


def migrate_world(world_dir: Path, *, dry_run: bool = False) -> int:
    layout = WorldDataLayout(world_dir / "data")
    if not layout.root.exists():
        return 0
    moved = 0
    for attribute, filenames in STAGE_FILES.items():
        destination_dir = getattr(layout, attribute)
        for filename in filenames:
            moved += _move(layout.root / filename, destination_dir / filename, dry_run=dry_run)

    legacy_stage12 = layout.root / "stage_12"
    for filename in STAGE12_CHECKPOINT_FILES:
        moved += _move(legacy_stage12 / filename, layout.grid_update / filename, dry_run=dry_run)
    figure_stage12 = world_dir / "figures" / "stage_12_grid_update"
    for filename in STAGE12_CHECKPOINT_FILES:
        moved += _move(figure_stage12 / filename, layout.grid_update / filename, dry_run=dry_run)

    if not dry_run and legacy_stage12.is_dir() and not any(legacy_stage12.iterdir()):
        legacy_stage12.rmdir()
    return moved


def _find_worlds(roots: list[Path]) -> list[Path]:
    worlds: set[Path] = set()
    for root in roots:
        if (root / "data").is_dir():
            worlds.add(root.resolve())
        if root.is_dir():
            worlds.update(path.parent.resolve() for path in root.rglob("data") if path.is_dir())
    return sorted(worlds)


def _move(source: Path, destination: Path, *, dry_run: bool) -> int:
    if not source.is_file():
        return 0
    if dry_run:
        return 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise FileExistsError(f"Destination differs from source: {destination}")
        source.unlink()
    else:
        shutil.move(str(source), str(destination))
    return 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
