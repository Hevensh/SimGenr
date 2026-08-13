from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.core.config import load_world_config
from world_generator.core.output_layout import WorldDataLayout
from world_generator.dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and package a seed-split SimGenr dataset.")
    parser.add_argument("--config", default="configs/small_debug.yaml")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--world-output", default="outputs/dataset_seed_1_50")
    parser.add_argument("--dataset-output", default="datasets/seed_1_50")
    parser.add_argument("--force", action="store_true", help="Regenerate worlds even when complete checkpoints exist.")
    parser.add_argument("--with-figures", action="store_true", help="Also render PNG/WebP inspection outputs.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed seed instead of recording it.")
    parser.add_argument("--plan-only", action="store_true", help="Print the split and exit without generating data.")
    args = parser.parse_args()

    if args.seed_start < 0 or args.seed_end < args.seed_start:
        parser.error("Require 0 <= seed-start <= seed-end")

    seeds = list(range(args.seed_start, args.seed_end + 1))
    partitions = prime_validation_test_split(seeds)
    split_summary = _split_summary(partitions)
    print(json.dumps(split_summary, indent=2))
    if args.plan_only:
        return

    config_path = Path(args.config).resolve()
    config = load_world_config(config_path)
    world_output = Path(args.world_output).resolve()
    dataset_output = Path(args.dataset_output).resolve()
    world_output.mkdir(parents=True, exist_ok=True)

    world_dirs: list[Path] = []
    failures: list[dict[str, int | str]] = []
    progress = tqdm(seeds, desc="Generating worlds", unit="seed", dynamic_ncols=True)
    for seed in progress:
        world_dir = world_output / f"{config.output.world_name}_seed{seed}"
        progress.set_postfix(seed=seed, split=partitions[seed])
        if not args.force and _world_is_complete(world_dir):
            world_dirs.append(world_dir)
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_static_world.py"),
            "--config",
            str(config_path),
            "--output",
            str(world_output),
            "--seed",
            str(seed),
        ]
        if not args.with_figures:
            command.append("--no-figures")
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append({"seed": seed, "partition": partitions[seed], "returncode": exc.returncode})
            if args.fail_fast:
                raise
            tqdm.write(f"Seed {seed} failed and was excluded; continuing with the next seed.")
            continue
        if _world_is_complete(world_dir):
            world_dirs.append(world_dir)
        else:
            failures.append({"seed": seed, "partition": partitions[seed], "returncode": "incomplete_output"})

    manifest = build_dataset(world_dirs, dataset_output, partitions=partitions)
    report = {
        "requested_seeds": seeds,
        "completed_seeds": [int(path.name.rsplit("seed", 1)[1]) for path in world_dirs],
        "failures": failures,
        "split_plan": split_summary,
        "actual_split_counts": manifest["split_counts"],
    }
    (dataset_output / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Dataset written to: {dataset_output}")
    print(f"Split counts: {manifest['split_counts']}")
    if failures:
        print(f"Excluded failed seeds: {[item['seed'] for item in failures]}")


def prime_validation_test_split(seeds: list[int]) -> dict[int, str]:
    primes = [seed for seed in seeds if _is_prime(seed)]
    prime_partition = {
        seed: ("val" if index % 2 == 0 else "test")
        for index, seed in enumerate(primes)
    }
    return {seed: prime_partition.get(seed, "train") for seed in seeds}


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _split_summary(partitions: dict[int, str]) -> dict[str, list[int]]:
    return {
        split: [seed for seed, assigned in partitions.items() if assigned == split]
        for split in ("train", "val", "test")
    }


def _world_is_complete(world_dir: Path) -> bool:
    layout = WorldDataLayout(world_dir / "data")
    required = (
        layout.existing(layout.topology, "static_maps.npz", layout.root / "static_maps.npz"),
        layout.existing(layout.weather, "hourly_weather_week.npz", layout.root / "hourly_weather_week.npz"),
        layout.existing(
            layout.storage_dispatch, "storage_dispatch_forecast.npz", layout.root / "storage_dispatch_forecast.npz"
        ),
        layout.existing(
            layout.storage_dispatch,
            "storage_dispatch_power_flow.npz",
            layout.root / "storage_dispatch_power_flow.npz",
        ),
        layout.existing(layout.storage_dispatch, "storage_dispatch.npz", layout.root / "storage_dispatch.npz"),
        layout.existing(
            layout.storage_dispatch,
            "storage_dispatch_electrical.npz",
            layout.root / "storage_dispatch_electrical.npz",
        ),
        layout.config_snapshot,
    )
    stage12_candidates = (layout.grid_update, layout.root / "stage_12", world_dir / "figures" / "stage_12_grid_update")
    stage12_dir = next((path for path in stage12_candidates if path.exists()), layout.grid_update)
    stage12_files = (
        "refined_grid_topology.npz",
        "grid_electrical.npz",
        "power_flow_hourly.npz",
        "bus_index.json",
        "refined_grid_edges.json",
    )
    return all(path.exists() for path in required) and all((stage12_dir / name).exists() for name in stage12_files)


if __name__ == "__main__":
    main()
