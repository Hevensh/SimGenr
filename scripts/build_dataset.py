from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world_generator.dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Package generated worlds into aligned multimodal samples.")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output", default="datasets/small_debug_preview")
    parser.add_argument("--worlds", nargs="+", default=["small_debug_seed42", "small_debug_seed123"])
    args = parser.parse_args()

    input_root = Path(args.input_root)
    world_dirs = [input_root / name for name in args.worlds]
    manifest = build_dataset(world_dirs, Path(args.output))
    print(f"Packaged {manifest['sample_count']} worlds into {args.output}")


if __name__ == "__main__":
    main()
