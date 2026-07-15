from __future__ import annotations

from pathlib import Path

import numpy as np


def save_single_map(
    values: np.ndarray,
    path: Path,
    title: str,
    cmap: object,
    norm: object | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap, norm=norm, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_elevation_with_water_overlay(
    static_maps: dict[str, np.ndarray],
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = draw_elevation_with_water_overlay(ax, static_maps)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_elevation_with_water_overlay(ax: object, static_maps: dict[str, np.ndarray]) -> object:
    elevation = static_maps["elevation"]
    river_centerline = static_maps.get("river_centerline", static_maps["river"]).astype(bool)
    river = static_maps["river"].astype(bool)
    lake = static_maps["lake"].astype(bool)
    water_overlay = np.zeros((*elevation.shape, 4), dtype=np.float32)
    river_bank = river & ~river_centerline
    lake_shore = mask_outer_ring(lake)
    lake_core = lake
    water_overlay[river_bank] = [0.62, 0.82, 1.0, 0.44]
    water_overlay[lake_shore] = [0.30, 0.78, 0.92, 0.56]
    water_overlay[river_centerline] = [0.0, 0.38, 0.95, 0.78]
    water_overlay[lake_core] = [0.0, 0.58, 0.78, 0.78]

    image = ax.imshow(elevation, cmap=land_terrain_cmap(), vmin=0.0, origin="upper")
    ax.imshow(water_overlay, origin="upper")
    return image


def draw_grid_edge_lines(
    ax: object,
    buses: np.ndarray,
    edges: np.ndarray,
    *,
    color: str = "#151515",
    redundant_color: str = "#006f8f",
    linewidth: float = 1.7,
    alpha: float = 0.76,
) -> None:
    if buses.size == 0 or edges.size == 0:
        return
    bus_by_id = {int(row[0]): row for row in np.atleast_2d(buses)}
    edge_rows = np.atleast_2d(edges)
    labels_used = {False: False, True: False}
    for redundant in (False, True):
        for edge in edge_rows:
            is_redundant = bool(edge[5] >= 0.5)
            if is_redundant != redundant:
                continue
            from_bus = bus_by_id.get(int(edge[1]))
            to_bus = bus_by_id.get(int(edge[2]))
            if from_bus is None or to_bus is None:
                continue
            line_color = redundant_color if is_redundant else color
            line_width = linewidth * (0.88 if is_redundant else 1.0)
            label = "redundant line" if is_redundant else "confirmed line"
            ax.plot(
                [float(from_bus[2]), float(to_bus[2])],
                [float(from_bus[1]), float(to_bus[1])],
                color=line_color,
                linewidth=line_width,
                alpha=alpha,
                solid_capstyle="round",
                label=label if not labels_used[is_redundant] else "_nolegend_",
                zorder=2,
            )
            labels_used[is_redundant] = True


def add_deduped_legend(
    ax: object,
    static_maps: dict[str, np.ndarray] | None = None,
    *,
    loc: str | None = None,
    fontsize: int = 7,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        legend_loc = loc or _least_occupied_corner(static_maps) or "upper right"
        ax.legend(unique_handles, unique_labels, loc=legend_loc, fontsize=fontsize, framealpha=0.82)


def _least_occupied_corner(static_maps: dict[str, np.ndarray] | None) -> str | None:
    if static_maps is None:
        return None
    masks = []
    for name in ("load_bus_map", "source_bus_map", "thermal_bus_map", "transit_bus_map"):
        values = static_maps.get(name)
        if values is not None:
            masks.append(values >= 0)
    if not masks:
        return None
    node_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        node_mask |= mask
    rows, cols = np.where(node_mask)
    if rows.size == 0:
        return None
    height, width = node_mask.shape
    row_limit = max(int(np.ceil(height * 0.30)), 1)
    col_limit = max(int(np.ceil(width * 0.30)), 1)
    counts = {
        "upper left": int(((rows < row_limit) & (cols < col_limit)).sum()),
        "upper right": int(((rows < row_limit) & (cols >= width - col_limit)).sum()),
        "lower left": int(((rows >= height - row_limit) & (cols < col_limit)).sum()),
        "lower right": int(((rows >= height - row_limit) & (cols >= width - col_limit)).sum()),
    }
    return min(counts, key=counts.get)


def mask_outer_ring(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    expanded = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            expanded |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return expanded & ~mask


def save_land_cover(values: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    draw_land_cover(ax, values, fig)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_land_cover(ax: object, values: np.ndarray, fig: object) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = [
        "#e6d77a",  # plain
        "#b6c96a",  # hill
        "#9a7b5f",  # mountain
        "#1f78b4",  # water
        "#7fcdbb",  # wetland
        "#7b4fa3",  # protected
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)
    image = ax.imshow(values, cmap=cmap, norm=norm, origin="upper")
    ax.set_title("Land cover")
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[1, 2, 3, 4, 5, 6])
    colorbar.ax.set_yticklabels(["plain", "hill", "mountain", "water", "wetland", "protected"])


def city_centers_from_id_map(city_id_map: np.ndarray, urban_density: np.ndarray) -> list[tuple[float, float, int]]:
    centers = []
    for city_id in sorted(int(v) for v in np.unique(city_id_map) if v >= 0):
        mask = city_id_map == city_id
        if not mask.any():
            continue
        weighted = urban_density * mask
        total = float(weighted.sum())
        rows, cols = np.indices(city_id_map.shape)
        if total <= 1e-8:
            row = float(rows[mask].mean())
            col = float(cols[mask].mean())
        else:
            row = float((rows * weighted).sum() / total)
            col = float((cols * weighted).sum() / total)
        centers.append((col, row, city_id))
    return centers


def draw_city_centers(ax: object, centers: list[tuple[float, float, int]]) -> None:
    if not centers:
        return
    cols = [item[0] for item in centers]
    rows = [item[1] for item in centers]
    ax.scatter(cols, rows, s=42, c="cyan", edgecolors="black", linewidths=0.6, marker="o")
    for col, row, city_id in centers:
        ax.text(col + 0.8, row + 0.8, str(city_id), color="black", fontsize=7, weight="bold")


def land_terrain_cmap() -> object:
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.00, (0.24, 0.62, 0.32)),
        (0.28, (0.58, 0.78, 0.38)),
        (0.52, (0.86, 0.82, 0.47)),
        (0.74, (0.62, 0.48, 0.34)),
        (0.92, (0.73, 0.70, 0.65)),
        (1.00, (0.94, 0.94, 0.91)),
    ]
    return LinearSegmentedColormap.from_list("land_terrain_no_water", colors, N=256)
