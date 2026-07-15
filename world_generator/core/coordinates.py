from __future__ import annotations

import numpy as np


def normalized_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    return np.meshgrid(x, y)


def grid_to_normalized(row: np.ndarray, col: np.ndarray, height: int, width: int) -> np.ndarray:
    x = col.astype(np.float32) / max(width - 1, 1)
    y = row.astype(np.float32) / max(height - 1, 1)
    return np.stack([x, y], axis=-1)


def normalized_to_grid(pos: np.ndarray, height: int, width: int) -> np.ndarray:
    xy = np.asarray(pos, dtype=np.float32)
    col = np.clip(np.rint(xy[..., 0] * (width - 1)), 0, width - 1)
    row = np.clip(np.rint(xy[..., 1] * (height - 1)), 0, height - 1)
    return np.stack([row, col], axis=-1).astype(np.int32)
