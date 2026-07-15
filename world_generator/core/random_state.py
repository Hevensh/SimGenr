from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np


MODULE_NAMES = (
    "terrain",
    "hydrology",
    "weather",
    "city",
    "energy",
    "grid",
    "operation",
    "evolution",
)


@dataclass(frozen=True)
class RngRegistry:
    seed: int
    module_seeds: dict[str, int]

    def generator(self, module_name: str) -> np.random.Generator:
        if module_name not in self.module_seeds:
            raise KeyError(f"Unknown RNG module: {module_name}")
        return np.random.default_rng(self.module_seeds[module_name])


def build_rng_registry(world_seed: int) -> RngRegistry:
    module_seeds = {
        name: derive_module_seed(world_seed, name) for name in MODULE_NAMES
    }
    return RngRegistry(seed=world_seed, module_seeds=module_seeds)


def derive_module_seed(world_seed: int, module_name: str) -> int:
    payload = f"{world_seed}:{module_name}".encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)
