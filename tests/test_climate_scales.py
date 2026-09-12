"""Climate perturbation support is measured in km, with explicit legacy mode."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtri

from world_generator.climate import climate_generator as climate_module
from world_generator.core.config import ClimateConfig, WorldConfig, WorldGridConfig, dump_config_snapshot, load_world_config
from world_generator.core.datatypes import HydrologyState, TerrainFeatures


def _flat_states(shape):
    zero = np.zeros(shape, dtype=np.float32)
    terrain = TerrainFeatures(zero, zero, zero, np.ones(shape, np.float32), zero, zero)
    hydrology = HydrologyState(
        np.full(shape, -1, np.int8), np.ones(shape, np.float32), zero.astype(bool),
        zero.astype(bool), zero.astype(bool), zero, zero, zero.astype(np.int32),
        np.full(shape, np.inf, np.float32), zero,
    )
    return terrain, hydrology


def test_climate_passes_grid_spacing_and_both_declared_lengths(monkeypatch):
    calls = []

    def field(shape, rng, correlation_length_km, cell_size_km, boundary):
        calls.append((shape, correlation_length_km, cell_size_km, boundary))
        return np.zeros(shape)

    def forbidden_pixel_smoothing(*args, **kwargs):
        raise AssertionError("Physical climate must not consume pixel smoothing")

    monkeypatch.setattr(climate_module, "gaussian_field", field)
    monkeypatch.setattr(climate_module, "_smooth_noise", forbidden_pixel_smoothing)
    config = replace(ClimateConfig(), climate_correlation_length_km=18.0,
                     wind_direction_correlation_length_km=7.0, spatial_boundary="periodic")
    for cell_size, shape in [(2.0, (3, 5)), (1.0, (6, 10)), (1.0, (9, 15))]:
        terrain, hydrology = _flat_states(shape)
        grid = WorldGridConfig(height=shape[0], width=shape[1], cell_size_km=cell_size)
        climate_module.generate_climate_baseline(terrain, hydrology, grid, config, np.random.default_rng(9))
        assert calls[-2:] == [(shape, 18.0, cell_size, "periodic"), (shape, 7.0, cell_size, "periodic")]


def test_physical_wind_direction_ignores_legacy_pixel_steps():
    terrain, _ = _flat_states((32, 40))
    config = replace(ClimateConfig(), wind_direction_correlation_length_km=5.0)
    first = climate_module._terrain_steered_wind_direction(
        terrain, 1.0, 0.0, replace(config, wind_direction_smoothing_steps=0),
        np.random.default_rng(123), cell_size_km=0.5,
    )
    second = climate_module._terrain_steered_wind_direction(
        terrain, 1.0, 0.0, replace(config, wind_direction_smoothing_steps=80),
        np.random.default_rng(123), cell_size_km=0.5,
    )
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(np.hypot(*first), 1.0, atol=1e-7)
    angle = np.rad2deg(np.arctan2(first[1], first[0]))
    assert np.max(np.abs(angle)) <= config.wind_direction_noise_degrees + 1e-5


class _ImpulseRng:
    """Deterministic impulse reveals the actual climate convolution support."""

    def standard_normal(self, shape):
        result = np.zeros(shape)
        result[shape[0] // 2, shape[1] // 2] = 1.0
        return result


def test_climate_kernel_response_keeps_km_width_across_resolution_and_extent():
    config = replace(ClimateConfig(), spatial_boundary="periodic")
    length_km = 8.0
    widths = []
    responses = []
    for size, cell_size in [(129, 1.0), (257, 0.5), (257, 1.0), (513, 0.5)]:
        bounded = climate_module._spatial_noise(
            (size, size), _ImpulseRng(), config, cell_size, length_km, legacy_steps=999,
        )
        # Invert the fixed CDF: L describes the latent Gaussian, not the bounded marginal.
        response = ndtri(bounded)[size // 2]
        x_km = (np.arange(size) - size // 2) * cell_size
        widths.append(np.sqrt(np.sum(response * x_km ** 2) / np.sum(response)))
        sampled = response[size // 2: size // 2 + int(10 / cell_size) + 1: int(1 / cell_size)]
        responses.append(sampled / response.max())
    # Shared helper truncates at 3 sigma, so second moments are slightly smaller.
    np.testing.assert_allclose(widths, length_km / 2.0, rtol=0.02)
    for profile in responses[1:]:
        np.testing.assert_allclose(profile, responses[0], rtol=1e-10, atol=1e-12)


def test_legacy_climate_exactly_reuses_old_pixel_stencil():
    config = replace(ClimateConfig(), spatial_scale_mode="legacy")
    actual = climate_module._spatial_noise((19, 21), np.random.default_rng(42), config, 0.5, 20.0, legacy_steps=6)
    expected = climate_module._smooth_noise((19, 21), np.random.default_rng(42), steps=6)
    np.testing.assert_array_equal(actual, expected)
    # Physical-km settings do not silently modify explicitly selected legacy results.
    alternative = climate_module._spatial_noise((19, 21), np.random.default_rng(42), config, 10.0, 2.0, legacy_steps=6)
    np.testing.assert_array_equal(actual, alternative)
    unsmoothed = climate_module._spatial_noise((19, 21), np.random.default_rng(42), config, 0.5, 20.0, legacy_steps=0)
    assert not np.array_equal(actual, unsmoothed)


@pytest.mark.parametrize("field", ["climate_correlation_length_km", "wind_direction_correlation_length_km"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_climate_lengths_fail(field, value):
    with pytest.raises(ValueError):
        replace(ClimateConfig(), **{field: value})


@pytest.mark.parametrize("override", [
    {"spatial_scale_mode": "auto"}, {"spatial_boundary": "open"},
    {"wind_direction_smoothing_steps": -1}, {"wind_direction_smoothing_steps": 1.5},
    {"wind_direction_smoothing_steps": True},
])
def test_invalid_climate_mode_boundary_and_legacy_steps_fail(override):
    with pytest.raises(ValueError):
        replace(ClimateConfig(), **override)


def test_climate_scales_and_legacy_mode_survive_config_snapshot(tmp_path):
    for mode in ("physical", "legacy"):
        climate = replace(ClimateConfig(), spatial_scale_mode=mode, climate_correlation_length_km=23.0,
                          wind_direction_correlation_length_km=9.0, spatial_boundary="periodic",
                          wind_direction_smoothing_steps=3)
        config = replace(WorldConfig(), climate=climate)
        path = tmp_path / f"climate_{mode}.yaml"
        dump_config_snapshot(config, path)
        assert load_world_config(path) == config


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1)])
def test_physical_noise_handles_singleton_axes_without_domain_normalization(shape):
    values = climate_module._spatial_noise(shape, np.random.default_rng(42), ClimateConfig(), 1.0, 12.0, legacy_steps=6)
    assert values.shape == shape
    assert np.isfinite(values).all()
    assert np.all((values > 0.0) & (values < 1.0))
