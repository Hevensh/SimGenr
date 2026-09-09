"""D: independent local/global balances, travel delay and lake geometry."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from world_generator.core.config import HydrologyDynamicConfig, WorldGridConfig
from world_generator.core.datatypes import HydrologyState, LandUseState, TerrainFeatures, WeatherStore
from world_generator.hydrology.dynamic_hydrology import generate_dynamic_hydrology, _Lake


def _config(**kwargs):
    defaults = dict(enabled=True, initial_soil_fraction=0., initial_groundwater_fraction=0.,
                    infiltration_capacity_mm_h=0., initial_lake_storage_fraction=0.)
    defaults.update(kwargs)
    return HydrologyDynamicConfig(**defaults)


def _world(direction, rain, *, irradiance=0., cell_size=1., lake=None, bed=None, depth=None):
    direction = np.asarray(direction, dtype=np.int8)
    shape = direction.shape
    rain = np.asarray(rain, dtype=float)
    if rain.ndim == 1:
        rain = np.broadcast_to(rain[:, None, None], (rain.size, *shape)).copy()
    assert rain.shape[1:] == shape
    zero = np.zeros(shape)
    false = np.zeros(shape, bool)
    lake = false if lake is None else np.asarray(lake, dtype=bool)
    bed = zero if bed is None else np.asarray(bed, dtype=float)
    depth = zero if depth is None else np.asarray(depth, dtype=float)
    terrain = TerrainFeatures(bed.copy(), *(zero.copy() for _ in range(5)))
    water = HydrologyState(direction, np.ones(shape), false, false, lake, depth, bed,
                           np.zeros(shape, np.int32), zero, zero)
    fractions = {name: zero.copy() for name in HydrologyDynamicConfig().impervious_fraction_by_use}
    fractions['water'] = lake.astype(float)
    fractions['natural'] = (~lake).astype(float)
    land_use = LandUseState(*(zero.copy() for _ in range(6)), np.zeros(shape, np.int16), use_fractions=fractions)
    weather = WeatherStore(np.stack((rain, np.broadcast_to(irradiance, rain.shape)), axis=1),
                           np.zeros(rain.shape, np.int8), np.arange(rain.shape[0]),
                           ('precipitation', 'irradiance'), time_unit='hour')
    return terrain, water, land_use, weather, WorldGridConfig(*shape, cell_size)


def _check_independent_balance(store, grid):
    s, f = store.states, store.fluxes
    factor = grid.cell_size_km**2 * 1000.
    total = (s['soil_storage_mm'] + s['groundwater_storage_mm']) * factor + s['channel_storage_m3'] + s['lake_storage_m3']
    incoming = f['precipitation_mm'] * factor + f['boundary_inflow_m3'] + f['routing_inflow_m3'] + f['lake_mixing_inflow_m3']
    outgoing = f['actual_et_m3'] + f['boundary_outflow_m3'] + f['routing_outflow_m3'] + f['lake_mixing_outflow_m3']
    np.testing.assert_allclose(total[:-1] + incoming, total[1:] + outgoing, rtol=1e-11, atol=1e-5)
    global_lhs = total[0].sum() + (f['precipitation_mm'] * factor + f['boundary_inflow_m3']).sum()
    global_rhs = total[-1].sum() + f['actual_et_m3'].sum() + f['boundary_outflow_m3'].sum()
    assert global_lhs == pytest.approx(global_rhs, rel=1e-11, abs=1e-5)
    np.testing.assert_allclose(f['actual_et_m3'], f['soil_evapotranspiration_mm'] * factor + f['open_water_evaporation_m3'])
    np.testing.assert_allclose(f['discharge_m3_s'] * 3600, f['routing_outflow_m3'] + f['boundary_outflow_m3'])
    assert np.all(s['soil_storage_mm'] >= 0) and np.all(s['soil_storage_mm'] <= store.static_maps['soil_capacity_mm'] + 1e-10)
    assert np.all(s['groundwater_storage_mm'] >= 0) and np.all(s['groundwater_storage_mm'] <= store.static_maps['groundwater_capacity_mm'] + 1e-10)
    for name in ('channel_storage_m3', 'lake_storage_m3'):
        assert np.all(s[name] >= 0)
    np.testing.assert_allclose(f['routing_inflow_m3'].sum(axis=(1, 2)), f['routing_outflow_m3'].sum(axis=(1, 2)), atol=1e-6)


def test_disabled_is_static_only_and_does_not_read_inputs_or_consume_rng():
    before = np.random.get_state()
    assert generate_dynamic_hydrology(None, None, None, None, None, HydrologyDynamicConfig()) is None
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize('cell_size', [.5, 2.])
def test_one_cell_rain_uses_local_area_and_exact_reservoir_recession(cell_size):
    args = _world([[-1]], [1., 0., 0.], cell_size=cell_size)
    store = generate_dynamic_hydrology(*args, _config())
    amount = cell_size**2 * 1000.
    retained = np.exp(-3600. * .5 / (cell_size * 1000. / 2.))
    np.testing.assert_allclose(store.states['channel_storage_m3'][1:, 0, 0], amount * retained**np.arange(1, 4))
    assert store.budgets['precipitation_m3'][0] == amount
    # Deliberately changing static catchment accumulation cannot change volume.
    changed = list(args); changed[1] = replace(args[1], flow_accumulation=np.full((1, 1), 1e9))
    other = generate_dynamic_hydrology(*changed, _config())
    np.testing.assert_array_equal(store.states['channel_storage_m3'], other.states['channel_storage_m3'])
    _check_independent_balance(store, args[-1])


def test_three_cell_routing_cannot_instantly_traverse_the_whole_graph():
    rain = np.zeros((4, 1, 3)); rain[0, 0, 0] = 1.
    args = _world([[2, 2, -1]], rain)
    store = generate_dynamic_hydrology(*args, _config())
    assert store.fluxes['boundary_outflow_m3'][:2].sum() == 0
    assert store.states['channel_storage_m3'][1, 0, 1] > 0
    assert store.states['channel_storage_m3'][1, 0, 2] == 0
    assert store.fluxes['boundary_outflow_m3'][2].sum() > 0
    _check_independent_balance(store, args[-1])


def test_closed_interior_sink_retains_water_or_is_explicitly_rejected():
    rain = np.zeros((3, 3, 3)); rain[0, 1, 1] = 2.
    args = _world(np.full((3, 3), -1), rain)
    store = generate_dynamic_hydrology(*args, _config())
    np.testing.assert_array_equal(store.states['channel_storage_m3'][1:, 1, 1], [2000., 2000., 2000.])
    assert store.static_maps['routing_receiver_flat_index'][1, 1] == -2
    assert store.static_maps['closed_sink_mask'][1, 1]
    assert store.fluxes['boundary_outflow_m3'].sum() == 0
    _check_independent_balance(store, args[-1])
    with pytest.raises(ValueError, match='Interior non-lake'):
        generate_dynamic_hydrology(*args, _config(interior_sink_policy='reject'))


def test_soil_infiltration_capacity_and_overflow_are_whole_cell_mm():
    args = _world([[-1]], [10., 0.])
    coefficients = {name: 0. for name in HydrologyDynamicConfig().impervious_fraction_by_use}
    cfg = _config(soil_capacity_mm=5., initial_soil_fraction=.4, infiltration_capacity_mm_h=20.,
                  soil_field_capacity_fraction=1., impervious_fraction_by_use=coefficients)
    store = generate_dynamic_hydrology(*args, cfg)
    assert store.states['soil_storage_mm'][0, 0, 0] == 2.
    assert store.fluxes['infiltration_mm'][0, 0, 0] == 3.
    assert store.fluxes['surface_runoff_mm'][0, 0, 0] == 7.
    assert store.states['soil_storage_mm'][1, 0, 0] == 5.
    _check_independent_balance(store, args[-1])


def test_groundwater_overflow_baseflow_and_zero_rain_do_not_create_water():
    args = _world([[-1]], [0., 0., 0.])
    coefficients = {name: 0. for name in HydrologyDynamicConfig().impervious_fraction_by_use}
    cfg = _config(soil_capacity_mm=10., initial_soil_fraction=1., soil_field_capacity_fraction=0.,
                  soil_percolation_time_hours=1., groundwater_capacity_mm=1., initial_groundwater_fraction=1.,
                  baseflow_time_hours=1., impervious_fraction_by_use=coefficients)
    store = generate_dynamic_hydrology(*args, cfg)
    assert store.fluxes['percolation_mm'][0, 0, 0] == pytest.approx(10 * (1 - np.exp(-1)))
    assert store.fluxes['groundwater_overflow_mm'][0, 0, 0] == pytest.approx(10 * (1 - np.exp(-1)))
    assert store.fluxes['baseflow_mm'][0, 0, 0] == pytest.approx(1 - np.exp(-1))
    assert np.all(np.diff(store.budgets['final_storage_m3']) <= 0)
    _check_independent_balance(store, args[-1])


def test_et_is_energy_requested_but_limited_by_available_water():
    args = _world([[-1]], [0., 0.], irradiance=1e6)
    cfg = _config(initial_soil_fraction=.01, initial_groundwater_fraction=0.)
    store = generate_dynamic_hydrology(*args, cfg)
    initial_soil = store.states['soil_storage_mm'][0].sum()
    assert store.fluxes['potential_et_mm'][0].sum() > initial_soil
    assert store.fluxes['soil_evapotranspiration_mm'].sum() == pytest.approx(initial_soil)
    assert store.states['soil_storage_mm'][-1].sum() == 0
    _check_independent_balance(store, args[-1])


def test_land_use_impervious_mix_does_not_treat_energy_reservation_as_sealed():
    args = list(_world([[-1, -1]], [1.]))
    fractions = {name: np.zeros((1, 2)) for name in args[2].use_fractions}
    fractions['residential'][0, 0] = 1.; fractions['energy_reserve'][0, 1] = 1.
    args[2] = replace(args[2], use_fractions=fractions)
    store = generate_dynamic_hydrology(*args, _config(infiltration_capacity_mm_h=20.))
    np.testing.assert_allclose(store.static_maps['impervious_fraction'], [[.65, .05]])
    np.testing.assert_allclose(store.fluxes['infiltration_mm'][0], [[.35, .95]])
    _check_independent_balance(store, args[-1])


def test_lake_shared_hypsometry_initially_empty_and_overflow_waits_in_channel():
    rain = np.zeros((4, 1, 4)); rain[:3, 0, 1:3] = np.array([500., 500., 1000.])[:, None]
    args = _world([[2, 2, 2, -1]], rain, lake=[[False, True, True, False]],
                  bed=[[2., 0., 1., 2.]], depth=[[0., 2., 1., 0.]])
    store = generate_dynamic_hydrology(*args, _config())
    volumes = store.states['lake_storage_m3'].sum(axis=(1, 2))
    np.testing.assert_allclose(volumes, [0., 1e6, 2e6, 3e6, 3e6])
    np.testing.assert_allclose(store.states['lake_water_level_m'][:, 0, 1:3], np.array([0., 1., 1.5, 2., 2.])[:, None] * np.ones((1, 2)))
    np.testing.assert_allclose(store.states['lake_wetted_area_m2'].sum(axis=(1, 2)), [0, 1e6, 2e6, 2e6, 2e6])
    assert store.fluxes['lake_overflow_m3'][2].sum() == pytest.approx(1e6)
    assert store.fluxes['boundary_outflow_m3'][:3].sum() == 0
    assert store.fluxes['boundary_outflow_m3'][3].sum() > 0
    assert store.fluxes['lake_mixing_inflow_m3'].sum() > 0
    _check_independent_balance(store, args[-1])


def test_lake_geometry_volume_level_area_monotonicity_and_vertical_datum():
    for datum in (0., 5000.):
        basin = _Lake(np.array([0, 1, 2]), np.array([0., 1., 2.]) + datum, datum + 3., 1e6, 6e6, 2)
        volume = np.array([0, 1e-4, 1., 5e5, 1e6, 3e6, 6e6])
        values = [basin.geometry(v) for v in volume]
        assert np.all(np.diff([value[0] for value in values]) >= 0)
        assert np.all(np.diff([value[2].sum() for value in values]) >= 0)
        np.testing.assert_allclose([value[1].sum() for value in values], volume, atol=1e-9)
        for level, cells, wet in values:
            np.testing.assert_allclose(cells, np.maximum(level - basin.bed, 0.) * 1e6, atol=1e-5)
            assert np.all(wet >= 0)


def test_closed_lake_retains_excess_in_channel_and_respects_configured_initial_volume():
    lake = np.zeros((3, 3), bool); lake[1, 1] = True
    rain = np.zeros((2, 3, 3)); rain[0, 1, 1] = 1500.
    args = _world(np.full((3, 3), -1), rain, lake=lake, depth=lake.astype(float))
    store = generate_dynamic_hydrology(*args, _config(initial_lake_storage_fraction=.25))
    assert store.states['lake_storage_m3'][0, 1, 1] == 250000.
    assert store.states['lake_storage_m3'][-1, 1, 1] == 1e6
    assert store.states['channel_storage_m3'][-1, 1, 1] == 750000.
    assert store.fluxes['lake_overflow_m3'].sum() == 750000.
    assert store.fluxes['boundary_outflow_m3'].sum() == 0
    _check_independent_balance(store, args[-1])


def test_open_water_evaporation_cannot_exceed_river_or_dry_lake_inventory():
    args = list(_world([[-1]], [1.], irradiance=1e6))
    args[1] = replace(args[1], river=np.ones((1, 1), bool))
    fractions = {name: np.zeros((1, 1)) for name in args[2].use_fractions}
    fractions['water'][:] = 1.
    args[2] = replace(args[2], use_fractions=fractions)
    store = generate_dynamic_hydrology(*args, _config(initial_channel_storage_mm=2.))
    assert store.fluxes['open_water_evaporation_m3'].sum() == 3000.
    assert store.states['channel_storage_m3'][-1].sum() == 0
    assert store.fluxes['surface_runoff_mm'].sum() == 1.
    _check_independent_balance(store, args[-1])
    dry_lake = _world([[-1]], [0.], irradiance=1e6, lake=[[True]], depth=[[1.]])
    empty = generate_dynamic_hydrology(*dry_lake, _config())
    assert empty.fluxes['actual_et_m3'].sum() == 0
    assert empty.states['lake_storage_m3'].sum() == 0


def test_small_rain_on_high_flat_lake_conserves_water_without_datum_cancellation():
    direction = np.full((1, 22), 2); direction[0, -1] = -1
    lake = np.zeros((1, 22), bool); lake[0, 1:-1] = True
    rain = np.zeros((6, 1, 22)); rain[:4, 0, 1:-1] = 1e-5
    args = _world(direction, rain, lake=lake, bed=np.full((1, 22), 5000.), depth=lake.astype(float))
    store = generate_dynamic_hydrology(*args, _config())
    assert store.states['lake_storage_m3'][-1].sum() == pytest.approx(.8, abs=1e-9)
    _check_independent_balance(store, args[-1])


@pytest.mark.parametrize('substeps', [1, 4, 12])
def test_boundary_inflow_and_substepped_routing_close_the_same_water_account(substeps):
    args = _world([[2, -1]], [0., 0., 0.])
    boundary = np.zeros((3, 1, 2)); boundary[0, 0, 0] = 100.
    store = generate_dynamic_hydrology(*args, _config(routing_substeps_per_hour=substeps), boundary_inflow_m3=boundary)
    assert store.budgets['boundary_inflow_m3'].sum() == 100.
    _check_independent_balance(store, args[-1])


def test_invalid_boundaries_cyclic_routes_and_inconsistent_inputs_fail():
    args = _world(np.full((3, 3), -1), [0.])
    for row, col, value, message in [(0, 0, -1., 'nonnegative'), (1, 1, 1., 'domain-edge')]:
        boundary = np.zeros((1, 3, 3)); boundary[0, row, col] = value
        with pytest.raises(ValueError, match=message):
            generate_dynamic_hydrology(*args, _config(), boundary_inflow_m3=boundary)
    cyclic = _world([[2, 6]], [0.])
    with pytest.raises(ValueError, match='cycle'):
        generate_dynamic_hydrology(*cyclic, _config())
    incomplete = list(args); incomplete[2] = replace(args[2], use_fractions={})
    with pytest.raises(ValueError, match='nine C'):
        generate_dynamic_hydrology(*incomplete, _config())
    inconsistent = list(args)
    values = dict(vars(args[3])); values['time_bounds_hours'] = np.array([[0., 2.]])
    inconsistent[3] = SimpleNamespace(**values)
    with pytest.raises(ValueError, match='time bounds'):
        generate_dynamic_hydrology(*inconsistent, _config())
