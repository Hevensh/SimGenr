from __future__ import annotations

from pathlib import Path

import numpy as np

from world_generator.visualization.common import add_deduped_legend, draw_built_environment_texture, draw_elevation_with_water_overlay
from world_generator.visualization.refined_topology_figures import _draw_buses


ELECTRICAL_FILES = [
    "electrical_overview.png",
    "branch_rating_overlay.png",
    "branch_voltage_overlay.png",
]


def save_electrical_figures(static_maps: dict[str, np.ndarray], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_electrical_overview(static_maps, output_dir / "electrical_overview.png")
    _save_branch_overlay(static_maps, output_dir / "branch_rating_overlay.png", field="rate", title="Branch thermal rating")
    _save_branch_overlay(static_maps, output_dir / "branch_voltage_overlay.png", field="voltage", title="Branch nominal voltage")
    return ELECTRICAL_FILES


def _save_electrical_overview(static_maps: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _draw_electrical_panel(static_maps, axes[0], "rate")
    axes[0].set_title("Branch rating over refined topology")
    _draw_electrical_panel(static_maps, axes[1], "voltage")
    axes[1].set_title("Nominal voltage classes")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_branch_overlay(static_maps: dict[str, np.ndarray], path: Path, *, field: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _draw_electrical_panel(static_maps, ax, field)
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_electrical_panel(static_maps: dict[str, np.ndarray], ax: object, field: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    image = draw_elevation_with_water_overlay(ax, static_maps)
    draw_built_environment_texture(ax, static_maps)
    buses = np.atleast_2d(static_maps["refined_grid_buses"])
    branches = np.atleast_2d(static_maps["electrical_branches"])
    bus_by_id = {int(row[0]): row for row in buses}
    segments = []
    values = []
    widths = []
    for branch in branches:
        from_bus = bus_by_id.get(int(branch[1]))
        to_bus = bus_by_id.get(int(branch[2]))
        if from_bus is None or to_bus is None:
            continue
        segments.append([(float(from_bus[2]), float(from_bus[1])), (float(to_bus[2]), float(to_bus[1]))])
        values.append(float(branch[8] if field == "rate" else branch[3]))
        widths.append(1.4 if branch[9] >= 0.5 else 2.0)
    if segments:
        cmap = "viridis" if field == "rate" else "plasma"
        collection = LineCollection(
            segments,
            cmap=cmap,
            linewidths=widths,
            alpha=0.86,
            zorder=3,
        )
        collection.set_array(np.asarray(values, dtype=np.float32))
        ax.add_collection(collection)
        label = "MVA" if field == "rate" else "kV"
        plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, label=label)
    else:
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _draw_buses(ax, static_maps)
    add_deduped_legend(ax, static_maps)
    ax.set_xticks([])
    ax.set_yticks([])
