"""Thermal project area budgets after Stage8 renewable reservations.

The geometric area identities are physical accounting constraints. Project radius,
installed MW/km2 and residential-score cutoffs are scenario assumptions. Subcell
quadrature resolves circular envelopes, not cadastral parcels or fuel/water rights.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import distance_transform_edt

from world_generator.core.config import PowerGridConfig, WorldGridConfig
from world_generator.core.datatypes import (
    EnergyCandidateState, GridBus, HydrologyState, LandUseState, StaticLandState,
)
from world_generator.land.land_generator import hard_allocatable_land_fraction


def thermal_land_eligibility(
    land: StaticLandState, hydrology: HydrologyState, land_use: LandUseState,
    energy: EnergyCandidateState, grid: WorldGridConfig, config: PowerGridConfig,
) -> np.ndarray:
    """Whole-cell exclusions, with distances in km between raster cell centres.

    The configured residential score threshold defines residential cores. Setbacks apply
    to every project cell, including cells away from a legal project centre. No
    cores or renewable sites means no corresponding setback restriction.
    """
    shape = (grid.height, grid.width)
    hard_land = np.asarray(hard_allocatable_land_fraction(land, hydrology))
    residential = np.asarray(land_use.residential)
    if hard_land.shape != shape or residential.shape != shape:
        raise ValueError("Thermal eligibility maps must match the world grid")
    if not np.isfinite(hard_land).all() or not np.isfinite(residential).all():
        raise ValueError("Thermal eligibility maps must be finite")
    eligible = hard_land > 0.0
    threshold = config.thermal_residential_score_threshold
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Thermal residential score threshold must be finite and within [0, 1]")
    residential_cores = residential > threshold
    renewable_sites = np.zeros(shape, dtype=bool)
    for candidate in energy.wind_candidates + energy.pv_candidates:
        row, col = _site_indices(candidate.row, candidate.col, shape)
        renewable_sites[row, col] = True
    for mask, setback in (
        (residential_cores, config.thermal_residential_buffer_km),
        (renewable_sites, config.thermal_min_renewable_distance_km),
    ):
        if not np.isfinite(setback) or setback < 0.0:
            raise ValueError("Thermal land setbacks must be finite and nonnegative")
        if mask.any():
            eligible &= distance_transform_edt(~mask, sampling=grid.cell_size_km) >= setback
    return eligible


def allocate_thermal_land(
    thermal_buses: list[GridBus], available_area_km2: np.ndarray,
    eligible: np.ndarray, grid: WorldGridConfig, config: PowerGridConfig,
) -> tuple[list[GridBus], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Reserve disjoint shares of the remaining per-cell energy area budget.

    Each quadrature point awards at most 1/n**2 of a cell's remaining budget to
    its nearest eligible thermal centre within the scenario project radius. Ties
    use the input bus order. The fractional residual is assumed uniform within
    each cell because Stage8 exports areas, not the subcell parcel geometry.
    Nameplate is clipped only after adequacy sizing; no compensating rescale is
    applied. A site with no area or zero requested nameplate is omitted.
    """
    area = validate_remaining_area(available_area_km2, grid)
    mask = np.asarray(eligible, dtype=bool)
    if mask.shape != area.shape:
        raise ValueError("Thermal eligibility must match the available area grid")
    density = float(config.thermal_capacity_density_mw_km2)
    radius = float(config.thermal_project_radius_km)
    if not np.isfinite(density) or density <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Thermal capacity density and project radius must be positive and finite")
    subdivisions = config.thermal_project_area_subcells_per_axis
    if not np.isfinite(subdivisions) or int(subdivisions) != subdivisions or subdivisions < 1:
        raise ValueError("Thermal project quadrature needs a positive integer subdivision count")
    subdivisions = int(subdivisions)
    active = []
    seen = set()
    for bus in thermal_buses:
        row, col = _site_indices(bus.row, bus.col, area.shape)
        if bus.kind != "thermal_bus" or not np.isfinite(bus.capacity_mw) or bus.capacity_mw < 0:
            raise ValueError("Thermal land allocation requires nonnegative finite thermal nameplates")
        if not np.isfinite(bus.bus_id) or int(bus.bus_id) != bus.bus_id or bus.bus_id in seen:
            raise ValueError("Thermal land bus IDs must be unique finite integers")
        seen.add(bus.bus_id)
        if bus.capacity_mw > 0 and mask[row, col] and area[row, col] > 0:
            active.append(bus)
    cube = np.zeros((len(active), *area.shape), dtype=np.float64)
    rows, cols = np.indices(area.shape)
    offsets = (np.arange(subdivisions) + .5) / subdivisions - .5
    for subrow in offsets:
        for subcol in offsets:
            nearest = np.full(area.shape, np.inf)
            owner = np.full(area.shape, -1, dtype=np.int32)
            for index, bus in enumerate(active):
                distance = np.hypot(rows + subrow - bus.row, cols + subcol - bus.col) * grid.cell_size_km
                take = mask & (distance <= radius) & (distance < nearest)
                nearest[take], owner[take] = distance[take], index
            for index in range(len(active)):
                take = owner == index
                cube[index, take] += area[take] / subdivisions**2
    installed = []
    ledger = []
    keep = []
    for index, bus in enumerate(active):
        reserved = float(cube[index].sum())
        upper = reserved * density
        if not np.isfinite(upper):
            raise ValueError("Thermal land-supported capacity must remain finite")
        capacity = min(float(bus.capacity_mw), upper)
        if capacity <= 0.0:
            continue
        installed.append(replace(bus, capacity_mw=capacity))
        keep.append(index)
        ledger.append((bus.bus_id, bus.row, bus.col, reserved, capacity, density,
                       capacity / density, upper, bus.capacity_mw))
    cube = cube[keep]
    allocated = cube.sum(axis=0)
    unallocated = area - allocated
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, grid.cell_size_km**2)
    if np.any(unallocated < -tolerance):
        raise ValueError("Thermal project allocation exceeds the Stage8 remaining area budget")
    unallocated = np.maximum(unallocated, 0.0)  # Roundoff only, after the budget check.
    maps = {
        "thermal_land_eligible": mask.copy(),
        "thermal_allocated_area_km2": allocated,
        "energy_unallocated_after_thermal_area_km2": unallocated,
    }
    return installed, np.asarray(ledger, dtype=np.float64).reshape(-1, 9), cube, maps


def validate_remaining_area(values: np.ndarray, grid: WorldGridConfig) -> np.ndarray:
    """Check an area map in km2, never reinterpret a suitability as a fraction."""
    area = np.asarray(values, dtype=np.float64)
    if area.shape != (grid.height, grid.width):
        raise ValueError("Thermal remaining energy area must match the world grid")
    if not np.isfinite(grid.cell_size_km) or grid.cell_size_km <= 0:
        raise ValueError("Thermal project area requires positive finite cell spacing")
    if not np.isfinite(area).all() or np.any(area < 0) or np.any(area > grid.cell_size_km**2 + 1e-9):
        raise ValueError("Thermal remaining energy area must be finite and within each cell's km2 area")
    return area


def _site_indices(row: int, col: int, shape: tuple[int, int]) -> tuple[int, int]:
    if not all(np.isfinite(value) and int(value) == value for value in (row, col)):
        raise ValueError("Thermal project site coordinates must be finite integers")
    row, col = int(row), int(col)
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError("Thermal project site lies outside the world grid")
    return row, col
