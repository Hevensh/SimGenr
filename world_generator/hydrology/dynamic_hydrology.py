"""Optional hourly water accounting on the unchanged static D8 skeleton.

Soil/groundwater depths are whole-cell equivalent mm. Channel/lake inventories
and exchanged volumes are m3. All spatial integration uses the local cell area;
static flow accumulation is used only to validate the D8 graph, never as rainfall
area. No random generator is accepted or consumed by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label

from world_generator.core.config import HydrologyDynamicConfig, WorldGridConfig
from world_generator.core.contracts import interval_bounds_hours, validate_weather_arrays
from world_generator.core.datatypes import (
    HydrologyState, HydrologyTimeSeriesStore, LandUseState, TerrainFeatures, WeatherStore,
)
from world_generator.core.hydrology_contracts import (
    HYDROLOGY_BUDGET_UNITS, HYDROLOGY_FLUX_UNITS, HYDROLOGY_STATE_UNITS,
)
from world_generator.hydrology.hydrology_generator import D8_OFFSETS, _compute_flow_accumulation


@dataclass(frozen=True)
class _Lake:
    cells: np.ndarray
    bed: np.ndarray
    spill: float
    cell_area_m2: float
    capacity: float
    outlet: int

    def geometry(self, volume: float) -> tuple[float, np.ndarray, np.ndarray]:
        """Invert V(h)=A sum(max(h-z_i,0)); the wetted-area curve is stepped."""
        datum = float(self.bed.min())
        relative_bed = self.bed - datum
        if volume <= 0.0:
            level = 0.0
        elif volume >= self.capacity:
            level = self.spill - datum
        else:
            sorted_bed = np.sort(relative_bed)
            cumulative = np.cumsum(sorted_bed)
            at_bed = self.cell_area_m2 * (np.arange(1, self.bed.size + 1) * sorted_bed - cumulative)
            count = max(1, min(self.bed.size, int(np.searchsorted(at_bed, volume, side="right"))))
            level = float((volume / self.cell_area_m2 + cumulative[count - 1]) / count)
            # Compute depths relative to the lowest bed. Large absolute datums
            # must not turn sub-millimetre rain into artificial water gains.
            for _ in range(2):
                wet = level > relative_bed
                if not wet.any():
                    break
                reconstructed = float(np.maximum(level - relative_bed, 0.0).sum() * self.cell_area_m2)
                level += (volume - reconstructed) / (self.cell_area_m2 * wet.sum())
            level = min(level, self.spill - datum)
        depths = np.maximum(level - relative_bed, 0.0)
        return datum + level, depths * self.cell_area_m2, (depths > 0.0).astype(float) * self.cell_area_m2


def generate_dynamic_hydrology(
    terrain: TerrainFeatures, hydrology: HydrologyState, land_use: LandUseState,
    hourly_weather: WeatherStore, grid: WorldGridConfig, config: HydrologyDynamicConfig,
    *, boundary_inflow_m3: np.ndarray | None = None,
) -> HydrologyTimeSeriesStore | None:
    """Run the optional bucket/routing model, returning T+1 states and T fluxes.

    Disabled mode returns before inspecting any inputs. Boundary inflow must be
    nonnegative interval m3 located on domain-edge cells. The hourly channel
    reservoir advances synchronously, so newly received water cannot traverse
    a second D8 edge in the same routing substep.
    """
    if not config.enabled:
        return None
    shape = (grid.height, grid.width)
    n = grid.height * grid.width
    cell_area_m2 = grid.cell_size_km**2 * 1e6
    mm_to_m3 = grid.cell_size_km**2 * 1000.0
    rain, irradiance, bounds = _forcing(hourly_weather, shape)
    steps = len(bounds)
    receiver, distances, closed, edge = _routing_graph(terrain, hydrology, grid, config)
    lake_mask = _bool_map(hydrology.lake, shape, "lake")
    river_mask = _bool_map(hydrology.river, shape, "river") & ~lake_mask
    water = lake_mask | river_mask
    impervious, pervious = _land_fractions(land_use, water, config)
    lakes, lake_id, bed_map, spill_map, capacity_map = _lake_geometry(hydrology, receiver, grid)
    soil_capacity = config.soil_capacity_mm * pervious
    groundwater_capacity = config.groundwater_capacity_mm * (~water)
    soil = (soil_capacity * config.initial_soil_fraction).ravel()
    groundwater = (groundwater_capacity * config.initial_groundwater_fraction).ravel()
    channel = np.full(n, config.initial_channel_storage_mm * mm_to_m3)
    channel[lake_mask.ravel()] = 0.0  # Lake initial state is configured separately.
    lake_volume = np.zeros(n)
    for basin in lakes:
        lake_volume[basin.cells] = basin.geometry(basin.capacity * config.initial_lake_storage_fraction)[1]
    incoming_boundary = np.zeros((steps, *shape)) if boundary_inflow_m3 is None else np.asarray(boundary_inflow_m3, dtype=float)
    if incoming_boundary.shape != (steps, *shape) or not np.isfinite(incoming_boundary).all() or np.any(incoming_boundary < 0):
        raise ValueError("Boundary inflow must be finite nonnegative interval m3 with shape [T,H,W]")
    if np.any(incoming_boundary[:, ~edge] > 0):
        raise ValueError("Boundary inflow may enter only domain-edge cells")

    states = {name: np.zeros((steps + 1, *shape), dtype=float) for name in HYDROLOGY_STATE_UNITS}
    fluxes = {name: np.zeros((steps, *shape), dtype=float) for name in HYDROLOGY_FLUX_UNITS}
    budgets = {name: np.zeros(steps, dtype=float) for name in HYDROLOGY_BUDGET_UNITS}
    _record_state(states, 0, soil, groundwater, channel, lake_volume, lakes, shape)
    substeps = config.routing_substeps_per_hour
    residence_hours = distances / (config.routing_velocity_m_s * 3600.0)
    release_fraction = -np.expm1(-(1.0 / substeps) / residence_hours)
    release_fraction[receiver == -2] = 0.0
    internal = receiver >= 0
    external = receiver == -1
    lake_flat = lake_mask.ravel()
    river_flat = river_mask.ravel()
    pervious_flat = pervious.ravel()
    soil_cap_flat = soil_capacity.ravel()
    groundwater_cap_flat = groundwater_capacity.ravel()
    for hour in range(steps):
        flux = {name: values[hour].ravel() for name, values in fluxes.items()}
        initial = (soil + groundwater) * mm_to_m3 + channel + lake_volume
        precipitation = rain[hour].ravel()
        boundary = incoming_boundary[hour].ravel()
        pet = (irradiance[hour].ravel() * config.pet_shortwave_absorptivity
               * config.pet_latent_energy_fraction * 3600.0 / config.latent_heat_vaporization_j_kg)
        flux["precipitation_mm"][:] = precipitation
        flux["potential_et_mm"][:] = pet
        flux["boundary_inflow_m3"][:] = boundary
        infiltration = np.minimum.reduce((precipitation * pervious_flat,
                                          config.infiltration_capacity_mm_h * pervious_flat,
                                          np.maximum(soil_cap_flat - soil, 0.0)))
        soil += infiltration
        soil_et = np.minimum(soil, pet * pervious_flat)
        soil -= soil_et
        percolation = np.maximum(soil - config.soil_field_capacity_fraction * soil_cap_flat, 0.0) * -np.expm1(-1.0 / config.soil_percolation_time_hours)
        soil -= percolation
        groundwater += percolation
        groundwater_overflow = np.maximum(groundwater - groundwater_cap_flat, 0.0)
        groundwater -= groundwater_overflow
        baseflow = groundwater * -np.expm1(-1.0 / config.baseflow_time_hours)
        groundwater -= baseflow
        runoff = precipitation * ~lake_flat - infiltration
        channel += (runoff + groundwater_overflow + baseflow) * mm_to_m3
        channel[~lake_flat] += boundary[~lake_flat]
        lake_volume[lake_flat] += precipitation[lake_flat] * mm_to_m3 + boundary[lake_flat]
        for name, values in (("infiltration_mm", infiltration), ("soil_evapotranspiration_mm", soil_et),
                             ("percolation_mm", percolation), ("groundwater_overflow_mm", groundwater_overflow),
                             ("baseflow_mm", baseflow), ("surface_runoff_mm", runoff)):
            flux[name][:] = values
        _equilibrate_lakes(lakes, lake_volume, channel, flux)
        # Explicit open-water evaporation: only mapped rivers and currently wet
        # lake cells. No ET is invented from a dry lake or conceptual flood store.
        river_et = np.minimum(channel[river_flat], pet[river_flat] * mm_to_m3)
        channel[river_flat] -= river_et
        flux["open_water_evaporation_m3"][river_flat] += river_et
        for basin in lakes:
            wetted_area = basin.geometry(float(lake_volume[basin.cells].sum()))[2]
            evaporated = np.minimum(lake_volume[basin.cells], pet[basin.cells] * wetted_area / 1000.0)
            lake_volume[basin.cells] -= evaporated
            flux["open_water_evaporation_m3"][basin.cells] += evaporated
        _equilibrate_lakes(lakes, lake_volume, channel, flux)
        for _ in range(substeps):
            released = channel * release_fraction
            channel -= released
            arriving = np.zeros(n)
            np.add.at(arriving, receiver[internal], released[internal])
            flux["routing_outflow_m3"][internal] += released[internal]
            flux["routing_inflow_m3"][:] += arriving
            flux["boundary_outflow_m3"][external] += released[external]
            channel[~lake_flat] += arriving[~lake_flat]
            lake_volume[lake_flat] += arriving[lake_flat]
            _equilibrate_lakes(lakes, lake_volume, channel, flux)
        flux["discharge_m3_s"][:] = (flux["routing_outflow_m3"] + flux["boundary_outflow_m3"]) / 3600.0
        flux["actual_et_m3"][:] = soil_et * mm_to_m3 + flux["open_water_evaporation_m3"]
        final = (soil + groundwater) * mm_to_m3 + channel + lake_volume
        inputs = (precipitation * mm_to_m3 + boundary + flux["routing_inflow_m3"] + flux["lake_mixing_inflow_m3"])
        outputs = (flux["actual_et_m3"] + flux["boundary_outflow_m3"]
                   + flux["routing_outflow_m3"] + flux["lake_mixing_outflow_m3"])
        residual = initial + inputs - final - outputs
        flux["cell_budget_residual_m3"][:] = residual
        _check_balance(residual, initial + inputs, final + outputs, config, f"cell budget at hour {hour}")
        for name, value in (("initial_storage_m3", initial.sum()), ("precipitation_m3", precipitation.sum() * mm_to_m3),
                            ("boundary_inflow_m3", boundary.sum()), ("actual_et_m3", flux["actual_et_m3"].sum()),
                            ("boundary_outflow_m3", flux["boundary_outflow_m3"].sum()), ("final_storage_m3", final.sum())):
            budgets[name][hour] = value
        lhs = budgets["initial_storage_m3"][hour] + budgets["precipitation_m3"][hour] + budgets["boundary_inflow_m3"][hour]
        rhs = budgets["final_storage_m3"][hour] + budgets["actual_et_m3"][hour] + budgets["boundary_outflow_m3"][hour]
        budgets["residual_m3"][hour] = lhs - rhs
        _check_balance(np.array([lhs - rhs]), np.array([lhs]), np.array([rhs]), config, f"domain budget at hour {hour}")
        _record_state(states, hour + 1, soil, groundwater, channel, lake_volume, lakes, shape)
    static_maps = {
        "impervious_fraction": impervious, "pervious_fraction": pervious,
        "soil_capacity_mm": soil_capacity, "groundwater_capacity_mm": groundwater_capacity,
        "lake_id": lake_id, "closed_sink_mask": closed,
        "lake_bed_elevation_m": bed_map, "lake_spill_elevation_m": spill_map,
        "lake_capacity_m3": capacity_map, "routing_receiver_flat_index": receiver.reshape(shape),
    }
    store = HydrologyTimeSeriesStore(
        timestamps=np.asarray(hourly_weather.timestamps).copy(), time_bounds_hours=bounds,
        state_time_hours=np.concatenate((bounds[:, 0], bounds[-1:, 1])),
        states=states, fluxes=fluxes, static_maps=static_maps, budgets=budgets,
        metadata={
            "mode": "bucket_routing_v1", "schema_version": "hydrology_v1", "config": asdict(config),
            "cell_area_km2": grid.cell_size_km**2, "time_step_hours": 1.0,
            "lake_group_count": len(lakes), "lake_group_rule": "8-neighbour static footprint with lowest static spill elevation",
            "routing_rule": "synchronous linear reservoirs; at most one D8 edge per substep",
            "channel_storage_interpretation": "conceptual routing inventory, including pending lake overflow; no validated bankfull/flood capacity",
            "surface_runoff_interpretation": "non-lake precipitation minus infiltration; includes direct river rain, excludes groundwater/baseflow",
            "lake_overflow_interpretation": "diagnostic lake-to-channel transfer at outlet; lake-group redistribution is in lake_mixing pairs",
            "routing_flux_interpretation": "internal D8 exchanges only; domain outflow is separate",
            "pet_model": "absorbed shortwave times scenario latent-energy share / latent heat; not FAO reference ET or a full energy balance",
            "soil_depth_support": "whole-cell equivalent mm; soil capacity scales with pervious fraction",
            "groundwater_depth_support": "whole-cell equivalent mm; capacity scales with non-open-water fraction",
            "limitations": ["no snow or freeze/thaw", "no 3D groundwater", "no backwater or floodplain inundation",
                            "static river footprint uses whole-cell open water", "lake wet area is a stepped raster hypsometry",
                            "no atmospheric moisture return coupling", "all effective process parameters are uncalibrated scenario priors"],
        },
    )
    store.as_arrays()  # Apply the public serialization contract before returning.
    return store


def _forcing(weather: WeatherStore, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weather.time_unit != "hour":
        raise ValueError("Dynamic hydrology requires hourly weather intervals")
    validate_weather_arrays(weather.dynamic, weather.weather_class, weather.timestamps, weather.channel_names, "hour")
    if weather.dynamic.shape[2:] != shape:
        raise ValueError("Dynamic hydrology weather grid does not match the world")
    if not {"precipitation", "irradiance"}.issubset(weather.channel_names):
        raise ValueError("Dynamic hydrology needs precipitation and irradiance channels")
    rain = np.asarray(weather.dynamic[:, weather.channel_names.index("precipitation")], dtype=float)
    irradiance = np.asarray(weather.dynamic[:, weather.channel_names.index("irradiance")], dtype=float)
    if np.any(rain < 0) or np.any(irradiance < 0):
        raise ValueError("Precipitation and irradiance must be nonnegative")
    bounds = interval_bounds_hours(weather.timestamps, 1.0)
    declared = getattr(weather, "time_bounds_hours", None)
    if declared is not None:
        declared = np.asarray(declared, dtype=float)
        if declared.shape != bounds.shape or not np.allclose(declared, bounds, rtol=0, atol=1e-9):
            raise ValueError("Declared weather time bounds disagree with the hourly intervals")
    return rain, irradiance, bounds


def _land_fractions(land_use: LandUseState, water: np.ndarray, config: HydrologyDynamicConfig) -> tuple[np.ndarray, np.ndarray]:
    if set(land_use.use_fractions) != set(config.impervious_fraction_by_use):
        raise ValueError("Enabled dynamic hydrology requires all nine C land-use fractions")
    total = np.zeros(water.shape)
    impervious = np.zeros(water.shape)
    for name, coefficient in config.impervious_fraction_by_use.items():
        values = np.asarray(land_use.use_fractions[name], dtype=float)
        if values.shape != water.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Hydrology land-use fractions must be finite [H,W] in [0,1]")
        total += values
        impervious += values * coefficient
    if not np.allclose(total, 1.0, rtol=0, atol=2e-6):
        raise ValueError("Hydrology land-use fractions must close to one per cell")
    if not np.allclose(land_use.use_fractions["water"], water.astype(float), rtol=0, atol=1e-6):
        raise ValueError("Land-use water fraction must agree with the static river/lake footprint")
    impervious[water] = 0.0
    land_fraction = (~water).astype(float)
    if np.any(impervious > land_fraction + 2e-6):
        raise ValueError("Impervious area exceeds non-water area")
    return np.minimum(impervious, land_fraction), np.maximum(land_fraction - impervious, 0.0)


def _routing_graph(terrain: TerrainFeatures, hydrology: HydrologyState, grid: WorldGridConfig,
                   config: HydrologyDynamicConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (grid.height, grid.width)
    direction = np.asarray(hydrology.flow_direction)
    if direction.shape != shape or not np.isfinite(direction).all() or np.any(direction != np.floor(direction)) or np.any((direction < -1) | (direction > 7)):
        raise ValueError("D8 directions must be integer [H,W] codes -1..7")
    elevation = np.asarray(terrain.elevation)
    if elevation.shape != shape or not np.isfinite(elevation).all():
        raise ValueError("Dynamic hydrology requires finite terrain elevations")
    _compute_flow_accumulation(elevation, direction)  # Existing graph order/cycle/bounds checks.
    rows, cols = np.indices(shape)
    edge = (rows == 0) | (cols == 0) | (rows == shape[0] - 1) | (cols == shape[1] - 1)
    closed = (direction == -1) & ~edge
    if config.interior_sink_policy == "reject" and np.any(closed & ~np.asarray(hydrology.lake, dtype=bool)):
        raise ValueError("Interior non-lake D8 sink requires explicit closed storage or a corrected flow graph")
    receiver = np.where(closed, -2, -1).astype(np.int64)
    distances = np.full(shape, .5 * grid.cell_size_km * 1000.0)
    for row, col in np.argwhere(direction >= 0):
        dr, dc = D8_OFFSETS[int(direction[row, col])]
        rr, cc = int(row + dr), int(col + dc)
        if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
            raise ValueError("D8 receiver points outside the raster; use an edge -1 outlet")
        receiver[row, col] = rr * shape[1] + cc
        distances[row, col] = np.hypot(dr, dc) * grid.cell_size_km * 1000.0
    return receiver.ravel(), distances.ravel(), closed, edge


def _lake_geometry(hydrology: HydrologyState, receiver: np.ndarray, grid: WorldGridConfig) -> tuple:
    shape = (grid.height, grid.width)
    mask = _bool_map(hydrology.lake, shape, "lake")
    ids, count = label(mask, structure=np.ones((3, 3), dtype=np.int8))
    bed = np.asarray(hydrology.hydrology_elevation, dtype=float)
    depth = np.asarray(hydrology.water_depth, dtype=float)
    if bed.shape != shape or depth.shape != shape or not np.isfinite(bed).all() or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Lake geometry requires finite bed elevations and nonnegative water depths")
    bed_map, spill_map, capacity_map = (np.zeros(shape) for _ in range(3))
    lakes = []
    area = grid.cell_size_km**2 * 1e6
    flat_ids = ids.ravel()
    for lake_id in range(1, count + 1):
        cells = np.flatnonzero(flat_ids == lake_id)
        beds = bed.ravel()[cells]
        spill = float(np.min(beds + depth.ravel()[cells]))
        capacity_cells = np.maximum(spill - beds, 0.0) * area
        capacity = float(capacity_cells.sum())
        if capacity <= 0.0:
            raise ValueError("Mapped lake must define a positive volume below its shared spill elevation")
        exits = []
        for cell in cells:
            next_cell = int(receiver[cell])
            if next_cell >= 0 and flat_ids[next_cell] == lake_id:
                continue
            cursor = next_cell
            while cursor >= 0 and flat_ids[cursor] != lake_id:
                cursor = int(receiver[cursor])
            if cursor < 0:  # This exit's original D8 path never re-enters this lake.
                exits.append(int(cell))
        if not exits:
            raise ValueError("Lake has no acyclic outgoing or closed-sink D8 path")
        outlet = min(exits, key=lambda cell: (receiver[cell] == -2, float(bed.ravel()[cell] + depth.ravel()[cell]), cell))
        basin = _Lake(cells, beds, spill, area, capacity, outlet)
        lakes.append(basin)
        bed_map.ravel()[cells] = beds
        spill_map.ravel()[cells] = spill
        capacity_map.ravel()[cells] = capacity_cells
    _check_collapsed_lake_graph(lakes, flat_ids, receiver)
    return lakes, ids.astype(np.int32), bed_map, spill_map, capacity_map


def _check_collapsed_lake_graph(lakes: list[_Lake], ids: np.ndarray, receiver: np.ndarray) -> None:
    """Reject cycles introduced by merging multiple lake cells into one pool."""
    size = receiver.size
    graph = np.full(size + len(lakes), -1, dtype=np.int64)
    for cell in np.flatnonzero(ids == 0):
        target = receiver[cell]
        graph[cell] = target if target < 0 or ids[target] == 0 else size + ids[target] - 1
    for index, basin in enumerate(lakes):
        target = receiver[basin.outlet]
        graph[size + index] = target if target < 0 or ids[target] == 0 else size + ids[target] - 1
    visited = np.zeros(graph.size, dtype=np.int8)
    for start in range(graph.size):
        cursor, path = start, []
        while cursor >= 0 and visited[cursor] == 0:
            visited[cursor] = 1
            path.append(cursor)
            cursor = int(graph[cursor])
        if cursor >= 0 and visited[cursor] == 1:
            raise ValueError("Lake-pool aggregation produces a drainage cycle; a backwater model is required")
        visited[path] = 2


def _equilibrate_lakes(lakes: list[_Lake], volume: np.ndarray, channel: np.ndarray,
                       flux: dict[str, np.ndarray]) -> None:
    for basin in lakes:
        before = volume[basin.cells].copy()
        total = float(before.sum())
        overflow = max(total - basin.capacity, 0.0)
        _, after, _ = basin.geometry(min(total, basin.capacity))
        volume[basin.cells] = after
        channel[basin.outlet] += overflow
        flux["lake_overflow_m3"][basin.outlet] += overflow
        exchanged = after - before
        exchanged[np.flatnonzero(basin.cells == basin.outlet)[0]] += overflow
        flux["lake_mixing_inflow_m3"][basin.cells] += np.maximum(exchanged, 0.0)
        flux["lake_mixing_outflow_m3"][basin.cells] += np.maximum(-exchanged, 0.0)


def _record_state(states: dict[str, np.ndarray], index: int, soil: np.ndarray, groundwater: np.ndarray,
                   channel: np.ndarray, lake_volume: np.ndarray, lakes: list[_Lake], shape: tuple[int, int]) -> None:
    for name, values in (("soil_storage_mm", soil), ("groundwater_storage_mm", groundwater),
                         ("channel_storage_m3", channel), ("lake_storage_m3", lake_volume)):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Nonfinite or negative dynamic water state: {name}")
        states[name][index] = values.reshape(shape)
    for basin in lakes:
        level, _, wet_area = basin.geometry(float(lake_volume[basin.cells].sum()))
        states["lake_water_level_m"][index].ravel()[basin.cells] = level
        states["lake_wetted_area_m2"][index].ravel()[basin.cells] = wet_area


def _bool_map(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all() or np.any((values != 0) & (values != 1)):
        raise ValueError(f"Hydrology {name} mask must be boolean [H,W]")
    return values.astype(bool)


def _check_balance(residual: np.ndarray, lhs: np.ndarray, rhs: np.ndarray,
                   config: HydrologyDynamicConfig, context: str) -> None:
    tolerance = config.budget_absolute_tolerance_m3 + config.budget_relative_tolerance * np.maximum(np.abs(lhs), np.abs(rhs))
    failed = ~np.isfinite(residual) | (np.abs(residual) > tolerance)
    if np.any(failed):
        index = int(np.flatnonzero(failed)[0])
        raise ValueError(f"Water {context} fails at flat cell {index}: residual={residual[index]} m3, tolerance={tolerance[index]} m3")
