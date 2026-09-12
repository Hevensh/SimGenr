from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from world_generator.core.contracts import positive_finite


@dataclass(frozen=True)
class WorldGridConfig:
    height: int = 64
    width: int = 64
    cell_size_km: float = 2.0
    origin_x_km: float = 0.0
    origin_y_km: float = 0.0
    latitude_center_degrees: float = 35.0

    def __post_init__(self) -> None:
        if not isinstance(self.height, int) or not isinstance(self.width, int) or min(self.height, self.width) < 1:
            raise ValueError("world height/width must be positive integers")
        positive_finite(self.cell_size_km, "world.cell_size_km")
        if not -90 <= self.latitude_center_degrees <= 90:
            raise ValueError("world.latitude_center_degrees must be in [-90,90]")


@dataclass(frozen=True)
class ContractConfig:
    """S: interface choices; P: unit conversions are defined in core.contracts."""

    time_step_hours: float = 1.0
    export_field_contracts: bool = True

    def __post_init__(self) -> None:
        positive_finite(self.time_step_hours, "contracts.time_step_hours")
        if self.time_step_hours != 1.0:
            raise ValueError("The current generator supports exactly 1 h intervals; conversion helpers support other durations")


@dataclass(frozen=True)
class TerrainConfig:
    algorithm: str = "multiscale_v2"
    octaves: int = 6
    mountain_ridges: int = 4
    basins: int = 2
    lowland_elevation_min_m: float = 100.0
    lowland_elevation_max_m: float = 800.0
    relief_mean_m: float = 2500.0
    relief_std_m: float = 500.0
    plainness: float = 0.35
    ridge_strength: float = 0.42
    basin_strength: float = 0.22
    boundary_falloff: float = 0.08
    continent_wavelength_km: float = 192.0
    erosion_wavelength_km: float = 64.0
    ridge_wavelength_km: float = 36.0
    detail_wavelength_km: float = 18.0
    domain_warp_wavelength_km: float = 96.0
    domain_warp_amplitude_km: float = 14.0


@dataclass(frozen=True)
class HydrologyConfig:
    algorithm: str = "conditioned_v2"
    river_threshold_quantile: float = 0.98
    river_max_dilation_cells: int = 1
    river_dilation_min_strength: float = 0.72
    lake_max_elevation_quantile: float = 0.48
    lake_slope_threshold: float = 0.08
    lake_min_accumulation_quantile: float = 0.92
    lake_min_cells: int = 6
    lake_extra_dilation_cells: int = 1
    lake_expansion_slope_multiplier: float = 1.25
    flow_smoothing_steps: int = 0
    river_depth_min_m: float = 0.5
    river_depth_max_m: float = 12.0
    lake_depth_m: float = 18.0
    flood_water_decay_km: float = 4.0
    river_min_catchment_km2: float = 32.0
    river_width_reference_catchment_km2: float = 128.0
    river_width_reference_m: float = 30.0
    river_width_exponent: float = 0.35
    lake_min_depth_m: float = 6.0
    lake_max_area_km2: float = 80.0
    lake_min_area_km2: float = 2.0
    depression_fill_epsilon_m: float = 0.02


@dataclass(frozen=True)
class HydrologyDynamicConfig:
    """S: optional, uncalibrated hourly bucket and routing parameters."""

    enabled: bool = False
    soil_capacity_mm: float = 150.0
    initial_soil_fraction: float = 0.35
    infiltration_capacity_mm_h: float = 20.0
    soil_field_capacity_fraction: float = 0.65
    soil_percolation_time_hours: float = 48.0
    groundwater_capacity_mm: float = 500.0
    initial_groundwater_fraction: float = 0.10
    baseflow_time_hours: float = 240.0
    initial_channel_storage_mm: float = 0.0
    initial_lake_storage_fraction: float = 0.0
    routing_velocity_m_s: float = 0.5
    routing_substeps_per_hour: int = 1
    interior_sink_policy: str = "closed_storage"
    pet_shortwave_absorptivity: float = 0.77
    pet_latent_energy_fraction: float = 0.65
    latent_heat_vaporization_j_kg: float = 2.45e6
    budget_absolute_tolerance_m3: float = 1e-5
    budget_relative_tolerance: float = 1e-10
    impervious_fraction_by_use: dict[str, float] = field(default_factory=lambda: {
        "water": 0.0, "wetland": 0.0, "residential": 0.65, "commercial": 0.85,
        "industrial": 0.8, "agriculture": 0.05, "park_green": 0.02,
        "natural": 0.02, "energy_reserve": 0.05,
    })

    def __post_init__(self) -> None:
        import math
        if not isinstance(self.enabled, bool):
            raise ValueError("hydrology_dynamic.enabled must be boolean")
        for name in ("soil_capacity_mm", "soil_percolation_time_hours", "groundwater_capacity_mm", "baseflow_time_hours", "routing_velocity_m_s", "latent_heat_vaporization_j_kg"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"hydrology_dynamic.{name} must be positive and finite")
        for name in ("infiltration_capacity_mm_h", "initial_channel_storage_mm", "budget_absolute_tolerance_m3", "budget_relative_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"hydrology_dynamic.{name} must be finite and nonnegative")
        if self.budget_absolute_tolerance_m3 == self.budget_relative_tolerance == 0:
            raise ValueError("At least one water-budget tolerance must be positive")
        for name in ("initial_soil_fraction", "soil_field_capacity_fraction", "initial_groundwater_fraction", "initial_lake_storage_fraction", "pet_shortwave_absorptivity", "pet_latent_energy_fraction"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"hydrology_dynamic.{name} must be finite in [0,1]")
        if isinstance(self.routing_substeps_per_hour, bool) or not isinstance(self.routing_substeps_per_hour, int) or not 1 <= self.routing_substeps_per_hour <= 60:
            raise ValueError("routing_substeps_per_hour must be an integer in [1,60]")
        if self.interior_sink_policy not in {"closed_storage", "reject"}:
            raise ValueError("interior_sink_policy must be closed_storage or reject")
        required = {"water", "wetland", "residential", "commercial", "industrial", "agriculture", "park_green", "natural", "energy_reserve"}
        if not isinstance(self.impervious_fraction_by_use, dict) or set(self.impervious_fraction_by_use) != required:
            raise ValueError("impervious_fraction_by_use must explicitly cover all nine land uses")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.impervious_fraction_by_use.values()):
            raise ValueError("Impervious coefficients must be finite in [0,1]")


@dataclass(frozen=True)
class LandConfig:
    protected_fraction: float = 0.10
    water_buffer_km: float = 1.5
    steep_slope_threshold: float = 0.22
    high_flood_threshold: float = 0.70
    vegetation_noise_weight: float = 0.25
    random_patch_count: int = 5
    # S: independent geomorphology/cover classification and hard allocation.
    hill_elevation_m: float = 1800.0
    mountain_elevation_m: float = 2800.0
    hill_slope_threshold: float = 0.12
    mountain_slope_threshold: float = 0.22
    wetland_flood_threshold: float = 0.45
    bare_vegetation_threshold: float = 0.12
    woodland_vegetation_threshold: float = 0.45
    allocatable_max_slope: float = 0.35

    def __post_init__(self) -> None:
        import math
        for name in ("hill_elevation_m", "mountain_elevation_m", "hill_slope_threshold", "mountain_slope_threshold", "allocatable_max_slope"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("protected_fraction", "wetland_flood_threshold", "bare_vegetation_threshold", "woodland_vegetation_threshold"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.hill_elevation_m > self.mountain_elevation_m or self.hill_slope_threshold > self.mountain_slope_threshold:
            raise ValueError("Landform hill thresholds must not exceed mountain thresholds")
        if self.bare_vegetation_threshold > self.woodland_vegetation_threshold:
            raise ValueError("Bare cover threshold must not exceed woodland threshold")


@dataclass(frozen=True)
class ClimateConfig:
    base_temperature_c: float = 28.0
    latitude_temperature_gradient_c: float = 12.0
    lapse_rate_c_per_km: float = 5.8
    water_moderation_km: float = 8.0
    annual_amplitude_c: float = 13.0
    coastal_amplitude_reduction_c: float = 5.0
    prevailing_wind_degrees: float = 255.0
    prevailing_wind_speed_mps: float = 5.5
    wind_speed_terrain_factor: float = 0.35
    wind_direction_terrain_steering: float = 1.20
    wind_direction_noise_degrees: float = 28.0
    wind_direction_smoothing_steps: int = 6
    humidity_water_decay_km: float = 10.0
    precipitation_base_mm_year: float = 760.0
    orographic_precipitation_factor: float = 0.55
    rain_shadow_factor: float = 0.28
    irradiance_base_w_m2: float = 195.0
    climate_noise_weight: float = 0.12
    # S: latent Gaussian covariance e-folding distances, not fitted climatology.
    # Existing positional fields stay in place. Pixel steps apply only in legacy.
    spatial_scale_mode: str = "physical"
    climate_correlation_length_km: float = 20.0
    wind_direction_correlation_length_km: float = 12.0
    spatial_boundary: str = "reflect"

    def __post_init__(self) -> None:
        _validate_scale_config(self)
        if self.spatial_boundary not in {"reflect", "periodic"}:
            raise ValueError("Climate spatial_boundary must be reflect or periodic")
        for name in ("climate_correlation_length_km", "wind_direction_correlation_length_km", "water_moderation_km", "humidity_water_decay_km"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.wind_direction_smoothing_steps, bool) or not isinstance(self.wind_direction_smoothing_steps, int) or self.wind_direction_smoothing_steps < 0:
            raise ValueError("wind_direction_smoothing_steps must be a nonnegative integer (legacy mode only)")


@dataclass(frozen=True)
class WeatherConfig:
    days: int = 365
    wet_day_probability: float = 0.30
    wet_day_persistence: float = 0.55
    precipitation_gamma_shape: float = 1.5
    start_day_of_year: int = 0
    hourly_week_days: int = 7
    hourly_temperature_diurnal_c: float = 4.5
    hourly_temperature_noise_c: float = 0.75
    hourly_wind_variability: float = 0.14
    hourly_cloud_variability: float = 0.16
    hourly_storm_event_rate: float = 0.18
    hourly_precipitation_burstiness: float = 2.2
    hourly_cloud_system_count: int = 9
    hourly_cloudlet_count: int = 11
    hourly_cloud_radius_min_km: float = 8.0
    hourly_cloud_radius_max_km: float = 22.0
    hourly_cloud_motion_km_per_hour: float = 1.15
    hourly_raining_cloud_fraction: float = 0.42
    advection_rho: float = 0.92
    innovation_smoothing_steps: int = 5
    synoptic_shift_cells_per_day: float = 1.4
    temperature_synoptic_c: float = 2.4
    humidity_synoptic_weight: float = 0.18
    cloud_synoptic_weight: float = 0.28
    wind_synoptic_weight: float = 0.22
    precipitation_event_scale: float = 2.4
    irradiance_cloud_sensitivity: float = 0.68
    pressure_base_hpa: float = 1013.0
    pressure_synoptic_hpa: float = 9.0
    seasonal_second_harmonic: float = 0.18
    seasonal_third_harmonic: float = 0.08
    temperature_seasonal_lag_days: float = 24.0
    solar_seasonal_lag_days: float = 4.0
    low_frequency_temperature_c: float = 1.4
    low_frequency_irradiance_weight: float = 0.06
    # Appended to preserve old positional arguments. Daily files are driver
    # anchors; aggregate_daily_weather(hourly) provides realized diagnostics.
    hourly_generation_mode: str = "primitive_hourly"
    spatial_scale_mode: str = "physical"
    innovation_correlation_length_km: float = 10.0
    spatial_boundary: str = "open"
    temporal_scale_mode: str = "physical"
    synoptic_memory_hours: float = 288.0
    hourly_memory_hours: float = 8.0
    synoptic_advection_speed_km_per_hour: float = 0.12
    hourly_wind_direction_std_degrees: float = 15.0
    # Old *_steps, *_cells_per_day and advection_rho are honored only in
    # explicit legacy modes; they remain readable in old configuration files.

    def __post_init__(self) -> None:
        _validate_scale_config(self)
        if self.hourly_generation_mode not in {"primitive_hourly", "daily_conditioned"}:
            raise ValueError("hourly_generation_mode must be primitive_hourly or daily_conditioned")
        if self.spatial_boundary not in {"open", "reflect", "periodic"}:
            raise ValueError("Weather spatial_boundary must be open, reflect or periodic")
        if self.temporal_scale_mode not in {"physical", "legacy"}:
            raise ValueError("temporal_scale_mode must be physical or legacy")
        for name in ("days", "hourly_week_days", "innovation_smoothing_steps"):
            value = getattr(self, name)
            if int(value) != value or value < (0 if name.endswith("steps") else 1):
                raise ValueError(f"{name} must be an integer in its supported range")
        if int(self.start_day_of_year) != self.start_day_of_year:
            raise ValueError("start_day_of_year must be an integer day index")
        for name in ("innovation_correlation_length_km", "synoptic_memory_hours", "hourly_memory_hours", "precipitation_gamma_shape", "pressure_base_hpa"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("synoptic_advection_speed_km_per_hour", "synoptic_shift_cells_per_day", "hourly_wind_direction_std_degrees", "hourly_temperature_diurnal_c", "hourly_temperature_noise_c", "hourly_wind_variability", "hourly_cloud_variability", "pressure_synoptic_hpa", "hourly_precipitation_burstiness"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.wet_day_probability <= 1 or not 0 <= self.wet_day_persistence < 1 or not 0 <= self.advection_rho < 1:
            raise ValueError("Wet probability must be [0,1], persistence and advection_rho [0,1)")
        if not 0 <= self.irradiance_cloud_sensitivity <= 1:
            raise ValueError("irradiance_cloud_sensitivity must be in [0,1]")


def _validate_scale_config(config: object) -> None:
    import math

    if any(not math.isfinite(value) for value in vars(config).values() if isinstance(value, (int, float))):
        raise ValueError("Climate/weather parameters must be finite")
    if config.spatial_scale_mode not in {"physical", "legacy"}:
        raise ValueError("spatial_scale_mode must be physical or legacy")


@dataclass(frozen=True)
class CityConfig:
    scaling_mode: str = "fixed"
    city_count: int = 8
    min_city_distance_km: float = 8.0
    total_population: float = 1_200_000.0
    reference_effective_area_km2: float = 1_400.0
    city_count_area_exponent: float = 0.80
    population_area_exponent: float = 0.90
    min_scaled_city_count: int = 1
    core_water_min_distance_km: float = 1.0
    water_access_optimal_km: float = 4.0
    water_access_sigma_km: float = 3.0
    waterfront_optimal_km: float = 0.8
    waterfront_sigma_km: float = 0.9
    max_core_flood_risk: float = 0.58
    edge_buffer_km: float = 3.0
    edge_buffer_min_factor: float = 0.40
    urban_radius_min_km: float = 2.5
    urban_radius_max_km: float = 8.0
    city_size_alpha: float = 1.25


@dataclass(frozen=True)
class LandUseConfig:
    industrial_edge_preference: float = 0.55
    park_waterfront_weight: float = 0.55
    agriculture_max_urban_density: float = 0.22
    agriculture_slope_soft_limit: float = 0.14
    agriculture_slope_hard_limit: float = 0.24
    residential_core_weight: float = 0.18
    residential_load_anchor_weight: float = 0.14
    load_residential_weight: float = 0.48
    load_commercial_weight: float = 0.34
    load_industrial_weight: float = 0.18
    # S: disjoint area budgets, separate from the legacy suitability maps.
    built_population_density_persons_km2: float = 6000.0
    maximum_built_fraction: float = 0.80
    agriculture_fraction_of_remaining: float = 0.45
    park_fraction_of_remaining: float = 0.10
    energy_reserve_fraction_of_remaining: float = 0.50

    def __post_init__(self) -> None:
        import math
        if not math.isfinite(self.built_population_density_persons_km2) or self.built_population_density_persons_km2 <= 0:
            raise ValueError("built_population_density_persons_km2 must be positive and finite")
        for name in ("maximum_built_fraction", "agriculture_fraction_of_remaining", "park_fraction_of_remaining", "energy_reserve_fraction_of_remaining"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class EnergyConfig:
    wind_candidate_count: int = 8
    pv_candidate_count: int = 8
    load_node_count: int = 18
    min_source_distance_km: float = 5.0
    wind_min_source_distance_km: float = 5.5
    min_cross_source_distance_km: float = 1.5
    min_load_node_distance_km: float = 2.0
    load_spread_radius_km: float = 6.0
    load_spread_penalty_weight: float = 0.68
    source_spread_radius_km: float = 5.0
    source_spread_penalty_weight: float = 0.42
    source_relax_iterations: int = 3
    source_neighbor_repulsion_weight: float = 0.06
    wind_selection_radius_km: float = 1.5
    pv_selection_radius_km: float = 1.2
    wind_cluster_radius_km: float = 2.0
    pv_cluster_radius_km: float = 1.5
    wind_cluster_capacity_max_multiplier: float = 4.0
    pv_cluster_capacity_max_multiplier: float = 2.3
    wind_min_city_distance_km: float = 3.0
    wind_city_half_distance_km: float = 12.0
    wind_flatness_weight: float = 0.55
    wind_roughness_penalty_weight: float = 0.45
    pv_min_city_distance_km: float = 1.5
    wind_capacity_min_mw: float = 4.0
    wind_capacity_max_mw: float = 18.0
    pv_capacity_min_mw: float = 20.0
    pv_capacity_max_mw: float = 75.0
    load_capacity_min_mw: float = 10.0
    load_capacity_max_mw: float = 120.0
    wind_capacity_density_mw_km2: float = 3.0
    pv_capacity_density_mw_km2: float = 35.0
    per_capita_peak_load_kw: float = 1.5
    project_area_subcells_per_axis: int = 4
    urban_exclusion_density_threshold: float = 0.18
    wind_max_project_slope: float = 0.35
    pv_max_project_slope: float = 0.24

    def __post_init__(self) -> None:
        import math
        if isinstance(self.project_area_subcells_per_axis, bool) or not isinstance(self.project_area_subcells_per_axis, int) or not 1 <= self.project_area_subcells_per_axis <= 32:
            raise ValueError("project_area_subcells_per_axis must be an integer in [1, 32]")
        for name in ("wind_capacity_density_mw_km2", "pv_capacity_density_mw_km2"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("wind_cluster_radius_km", "pv_cluster_radius_km", "wind_min_city_distance_km", "pv_min_city_distance_km", "wind_max_project_slope", "pv_max_project_slope"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.urban_exclusion_density_threshold) or not 0 <= self.urban_exclusion_density_threshold <= 1:
            raise ValueError("urban_exclusion_density_threshold must be in [0, 1]")


@dataclass(frozen=True)
class SourceLoadConfig:
    """Transparent scenario priors; calibrate against regional observations."""

    calendar_start_date: str = "2025-01-01"
    wind_reference_height_m: float = 10.0
    wind_hub_height_m: float = 90.0
    wind_shear_exponent: float = 0.14
    wind_cut_in_mps: float = 3.0
    wind_rated_mps: float = 11.4
    wind_cut_out_mps: float = 25.0
    wind_system_loss_fraction: float = 0.08
    pv_dc_ac_ratio: float = 1.2
    pv_tilt_degrees: float = 30.0
    pv_azimuth_degrees: float = 180.0
    pv_ground_albedo: float = 0.2
    pv_inverter_efficiency: float = 0.96
    pv_system_loss_fraction: float = 0.10
    pv_temperature_coefficient_per_c: float = -0.004
    pv_heat_loss_constant: float = 25.0
    pv_heat_loss_wind: float = 6.84
    load_heating_balance_c: float = 16.0
    load_cooling_balance_c: float = 22.0
    load_heating_sensitivity_per_c: float = 0.016
    load_cooling_sensitivity_per_c: float = 0.025
    load_thermal_memory_hours: float = 8.0
    load_residual_std_fraction: float = 0.035
    load_residual_ar1: float = 0.85
    load_common_variance_fraction: float = 0.55
    load_spatial_correlation_km: float = 20.0
    load_residential_fraction: float = 0.55
    load_commercial_fraction: float = 0.30
    load_industrial_fraction: float = 0.15
    # E/S: the rated wind is nominal at this reference density. Safety cut-in
    # and cut-out use physical hub wind; density changes the rated onset.
    wind_reference_density_kg_m3: float = 1.225
    wind_density_height_mode: str = "isothermal_surface_to_hub"
    wind_density_fallback: str = "dry_air_then_reference"
    pv_module_height_m: float = 2.0
    pv_wind_shear_exponent: float = 0.14
    load_initial_temperature_mode: str = "first_hour"
    load_initial_temperature_c: float = 20.0
    thermal_dispatch_half_distance_km: float = 24.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate once at construction and again through legacy wrappers."""
        import math

        if any(not math.isfinite(value) for value in vars(self).values() if isinstance(value, (int, float))):
            raise ValueError("Source/load parameters must be finite")
        if not 0 <= self.wind_cut_in_mps < self.wind_rated_mps < self.wind_cut_out_mps:
            raise ValueError("Wind thresholds must satisfy 0 <= cut-in < rated < cut-out")
        for name in ("wind_reference_height_m", "wind_hub_height_m", "pv_dc_ac_ratio", "pv_heat_loss_constant", "load_spatial_correlation_km", "wind_reference_density_kg_m3", "pv_module_height_m", "thermal_dispatch_half_distance_km"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.wind_density_height_mode not in {"isothermal_surface_to_hub", "surface_proxy"}:
            raise ValueError("wind_density_height_mode must be isothermal_surface_to_hub or surface_proxy")
        if self.wind_density_fallback not in {"dry_air_then_reference", "error"}:
            raise ValueError("wind_density_fallback must be dry_air_then_reference or error")
        if self.load_initial_temperature_mode not in {"first_hour", "configured"}:
            raise ValueError("load_initial_temperature_mode must be first_hour or configured")
        if self.load_initial_temperature_c <= -273.15:
            raise ValueError("Initial load temperature must exceed absolute zero")
        if self.pv_temperature_coefficient_per_c > 0:
            raise ValueError("This PV temperature-loss model requires a nonpositive temperature coefficient")
        for name in ("wind_system_loss_fraction", "pv_system_loss_fraction", "load_common_variance_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not 0 < self.pv_inverter_efficiency <= 1 or not 0 <= self.pv_ground_albedo <= 1:
            raise ValueError("PV efficiency and ground albedo must be physical fractions")
        if not 0 <= self.pv_tilt_degrees <= 90 or not 0 <= self.pv_azimuth_degrees < 360:
            raise ValueError("PV tilt must be 0..90 degrees and azimuth 0..<360 degrees")
        if not -1 < self.load_residual_ar1 < 1:
            raise ValueError("Load AR(1) coefficient must be strictly inside (-1, 1)")
        for name in ("pv_heat_loss_wind", "load_thermal_memory_hours", "load_residual_std_fraction", "load_heating_sensitivity_per_c", "load_cooling_sensitivity_per_c", "load_residential_fraction", "load_commercial_fraction", "load_industrial_fraction"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.load_heating_balance_c > self.load_cooling_balance_c:
            raise ValueError("Heating balance temperature must not exceed cooling balance temperature")
        if self.load_residential_fraction + self.load_commercial_fraction + self.load_industrial_fraction <= 0:
            raise ValueError("At least one load sector fraction must be positive")


@dataclass(frozen=True)
class PowerGridConfig:
    nominal_voltage_kv: float = 220.0
    thermal_scale_with_load: bool = True
    thermal_planning_reserve_margin: float = 1.15
    thermal_candidate_count: int = 4
    thermal_load_optimal_km: float = 5.0
    thermal_load_sigma_km: float = 4.0
    thermal_residential_buffer_km: float = 4.0
    thermal_capacity_min_mw: float = 80.0
    thermal_capacity_max_mw: float = 260.0
    thermal_adequacy_load_fraction: float = 0.68
    thermal_wind_capacity_credit: float = 0.10
    thermal_solar_capacity_credit: float = 0.12
    thermal_min_distance_km: float = 8.0
    thermal_min_generation_distance_km: float = 2.5
    thermal_min_renewable_distance_km: float = 3.0
    thermal_spread_radius_km: float = 8.0
    thermal_spread_penalty_weight: float = 0.45
    thermal_relax_iterations: int = 2
    thermal_neighbor_repulsion_weight: float = 0.12
    candidate_knn: int = 8
    redundancy_ratio: float = 0.35
    load_cluster_radius_km: float = 6.0
    generation_cluster_radius_km: float = 4.0
    load_cluster_min_loads: int = 3
    load_cluster_generation_links: int = 2
    load_cluster_external_degree_penalty: float = 0.70
    generation_cluster_external_degree_penalty: float = 0.55
    flow_topup_max_edges: int = 6
    flow_topup_target_loading: float = 0.95
    flow_topup_min_relief: float = 0.02
    line_water_penalty: float = 2.8
    line_protected_penalty: float = 3.5
    line_slope_penalty: float = 1.6
    line_terrain_cost_penalty: float = 1.2
    transit_flood_penalty: float = 4.0
    transit_water_penalty: float = 8.0
    transit_protected_penalty: float = 8.0
    transit_slope_penalty: float = 1.8
    transit_terrain_cost_penalty: float = 1.4
    line_multiplier_step: float = 0.125
    min_line_multiplier: float = 0.125
    max_upgrade_factor: float = 4.0
    merged_line_target_loading: float = 0.80
    near_parallel_max_distance_km: float = 2.0
    near_parallel_max_angle_deg: float = 20.0
    near_parallel_min_length_km: float = 4.0
    near_parallel_corridor_radius_km: float = 2.0
    near_parallel_min_cost_improvement: float = 0.05
    low_utilization_peak_ratio: float = 0.45
    downgrade_max_network_loading: float = 0.90
    thermal_capacity_density_mw_km2: float = 200.0
    thermal_project_radius_km: float = 2.0
    thermal_project_area_subcells_per_axis: int = 4
    thermal_residential_score_threshold: float = 0.22

    def __post_init__(self) -> None:
        import math
        for name in ("thermal_capacity_density_mw_km2", "thermal_project_radius_km"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if isinstance(self.thermal_project_area_subcells_per_axis, bool) or not isinstance(self.thermal_project_area_subcells_per_axis, int) or not 1 <= self.thermal_project_area_subcells_per_axis <= 32:
            raise ValueError("thermal_project_area_subcells_per_axis must be an integer in [1, 32]")
        if not math.isfinite(self.thermal_residential_score_threshold) or not 0 <= self.thermal_residential_score_threshold <= 1:
            raise ValueError("thermal_residential_score_threshold must be finite in [0, 1]")


@dataclass(frozen=True)
class StorageConfig:
    cyclic_state_of_charge: bool = True
    allow_load_shedding: bool = True
    load_shedding_cost: float = 10000.0
    congestion_trigger_ratio: float = 0.90
    congestion_decay_hops: float = 2.0
    minimum_duration_hours: float = 2.0
    maximum_duration_hours: float = 8.0
    congestion_storage_max_active_fraction: float = 0.35
    map_spread_radius_km: float = 4.0
    site_max_count: int = 5
    site_min_need_score: float = 0.55
    site_coverage_decay_hops: float = 1.8
    site_max_service_hops: int = 4
    site_min_residual_fraction: float = 0.12
    capacity_reserve_margin: float = 1.15
    initial_soc_fraction: float = 0.50
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    minimum_soc_fraction: float = 0.20
    maximum_soc_fraction: float = 0.90
    preferred_soc_lower_fraction: float = 0.40
    preferred_soc_upper_fraction: float = 0.70
    normal_dispatch_c_rate: float = 0.20
    storage_power_ramp_fraction_per_hour: float = 0.25
    thermal_ramp_fraction_per_hour: float = 0.30
    thermal_operating_limit_ratio: float = 0.90
    line_operating_limit_ratio: float = 0.90
    max_thermal_expansion_mw_per_bus: float = 100.0
    max_storage_power_expansion_fraction: float = 1.0
    max_storage_energy_expansion_fraction: float = 1.0
    max_line_expansion_fraction: float = 1.0
    thermal_capacity_cost: float = 1000.0
    storage_power_cost: float = 350.0
    storage_energy_cost: float = 240.0
    line_capacity_cost_per_mva_km: float = 4.0
    thermal_dispatch_cost: float = 1.0
    storage_cycle_cost: float = 0.50
    emergency_discharge_cost: float = 4.0
    soc_band_penalty: float = 2.0
    renewable_dispatch_credit: float = 0.05
    # S: aggregate upward reserve scenario; no network deliverability claim.
    reserve_load_fraction: float = 0.0
    reserve_contingency_mw: float = 0.0
    reserve_duration_hours: float = 1.0
    reserve_response_hours: float = 0.25
    reserve_shortfall_cost: float = 1000.0
    reserve_offer_cost: float = 0.01
    # S: pre-window dispatch used when the boundary is not cyclic.
    initial_thermal_mw: float = 0.0
    initial_storage_net_mw: float = 0.0

    def __post_init__(self) -> None:
        import math
        for name, value in vars(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"storage.{name} must be finite")
        for name in ("reserve_load_fraction", "reserve_contingency_mw", "reserve_offer_cost", "initial_thermal_mw",
                     "thermal_ramp_fraction_per_hour", "storage_power_ramp_fraction_per_hour", "normal_dispatch_c_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"storage.{name} must be nonnegative")
        for name in ("reserve_duration_hours", "reserve_response_hours", "reserve_shortfall_cost"):
            if getattr(self, name) <= 0:
                raise ValueError(f"storage.{name} must be positive")


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"
    world_name: str = "small_debug"


@dataclass(frozen=True)
class PlanningConfig:
    """S: explicit asset information boundary, separate from dispatch policy."""

    mode: str = "full_window_planning"
    design_days: int = 7
    design_start_day_of_year: int = 0
    design_seed: int = 190731
    design_weather_overrides: dict[str, Any] = field(default_factory=dict)
    design_source_load_overrides: dict[str, Any] = field(default_factory=dict)
    fixed_storage_sites: tuple[dict[str, Any], ...] = ()
    line_faults: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        import math

        if self.mode not in {"fixed_assets", "preplanned", "full_window_planning"}:
            raise ValueError("planning.mode must be fixed_assets, preplanned or full_window_planning")
        object.__setattr__(self, "fixed_storage_sites", tuple(dict(site) for site in self.fixed_storage_sites))
        object.__setattr__(self, "line_faults", tuple(dict(fault) for fault in self.line_faults))
        for name in ("design_days", "design_start_day_of_year", "design_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "design_days" else 0):
                raise ValueError(f"planning.{name} must be an integer in its supported range")
        if self.design_start_day_of_year >= 365:
            raise ValueError("planning.design_start_day_of_year must be 0..364")
        if self.mode != "fixed_assets" and self.fixed_storage_sites:
            raise ValueError("planning.fixed_storage_sites is only applicable to fixed_assets mode")
        if set(self.design_weather_overrides) & {"days", "hourly_week_days", "start_day_of_year"}:
            raise ValueError("Use planning design_days/start_day fields to define the design clock")
        WeatherConfig(**self.design_weather_overrides)
        SourceLoadConfig(**self.design_source_load_overrides)
        site_ids = []
        for index, site in enumerate(self.fixed_storage_sites):
            if set(site) - {"site_id", "bus_id", "power_mw", "energy_mwh", "initial_soc_fraction"}:
                raise ValueError("Unknown fixed storage site property")
            for name in ("bus_id",):
                if name not in site or isinstance(site[name], bool) or not isinstance(site[name], int) or site[name] < 0:
                    raise ValueError("Fixed storage bus_id must be a nonnegative integer")
            site_id = site.get("site_id", index)
            if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id < 0:
                raise ValueError("Fixed storage site_id must be a nonnegative integer")
            site_ids.append(site_id)
            for name in ("power_mw", "energy_mwh"):
                if name not in site or not math.isfinite(site[name]) or site[name] <= 0:
                    raise ValueError("Fixed storage power/energy must be positive and finite")
            if "initial_soc_fraction" in site and not 0 <= site["initial_soc_fraction"] <= 1:
                raise ValueError("Fixed storage initial SOC fraction must be [0,1]")
        if len(set(site_ids)) != len(site_ids):
            raise ValueError("Fixed storage site IDs must be unique")
        for fault in self.line_faults:
            if set(fault) != {"branch_id", "start_offset_hours", "duration_hours"}:
                raise ValueError("Line faults need branch_id, start_offset_hours and duration_hours")
            for name, value in fault.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "duration_hours" else 0):
                    raise ValueError("Line fault identifiers/offsets must be nonnegative integers and duration positive")


@dataclass(frozen=True)
class ValidationConfig:
    """Diagnostic presentation only; physical tolerances remain with each mechanism."""
    max_failure_locations: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.max_failure_locations,bool) or not isinstance(self.max_failure_locations,int) or not 0 <= self.max_failure_locations <= 1000:
            raise ValueError("validation.max_failure_locations must be an integer in [0,1000]")


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 42
    world: WorldGridConfig = field(default_factory=WorldGridConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    land: LandConfig = field(default_factory=LandConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    city: CityConfig = field(default_factory=CityConfig)
    land_use: LandUseConfig = field(default_factory=LandUseConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    source_load: SourceLoadConfig = field(default_factory=SourceLoadConfig)
    power_grid: PowerGridConfig = field(default_factory=PowerGridConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    contracts: ContractConfig = field(default_factory=ContractConfig)
    hydrology_dynamic: HydrologyDynamicConfig = field(default_factory=HydrologyDynamicConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        # P: capacities cannot be negative; do not silently clip bad scenarios.
        import math

        for section_name in ("energy", "power_grid", "storage"):
            section = asdict(getattr(self, section_name))
            for name, value in section.items():
                if "capacity" in name and isinstance(value, (int, float)):
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{section_name}.{name} must be finite and nonnegative")
        for kind in ("wind", "pv", "load"):
            if getattr(self.energy, f"{kind}_capacity_min_mw") > getattr(self.energy, f"{kind}_capacity_max_mw"):
                raise ValueError(f"energy.{kind} capacity minimum exceeds maximum")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_world_config(path: str | Path) -> WorldConfig:
    data = _load_mapping(Path(path))
    return WorldConfig(
        seed=int(data.get("seed", 42)),
        world=WorldGridConfig(**data.get("world", {})),
        terrain=TerrainConfig(**data.get("terrain", {})),
        hydrology=HydrologyConfig(**data.get("hydrology", {})),
        land=LandConfig(**data.get("land", {})),
        climate=ClimateConfig(**data.get("climate", {})),
        weather=WeatherConfig(**data.get("weather", {})),
        city=CityConfig(**data.get("city", {})),
        land_use=LandUseConfig(**data.get("land_use", {})),
        energy=EnergyConfig(**data.get("energy", {})),
        source_load=SourceLoadConfig(**data.get("source_load", {})),
        power_grid=PowerGridConfig(**data.get("power_grid", {})),
        storage=StorageConfig(**data.get("storage", {})),
        output=OutputConfig(**data.get("output", {})),
        contracts=ContractConfig(**data.get("contracts", {})),
        planning=PlanningConfig(**data.get("planning", {})),
        validation=ValidationConfig(**data.get("validation", {})),
        hydrology_dynamic=HydrologyDynamicConfig(**data.get("hydrology_dynamic", {})),
    )


def dump_config_snapshot(config: WorldConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except ImportError:
        import json

        path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        return loaded
    except ImportError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        while stack[-1][0] >= indent:
            stack.pop()
        key, value = _split_yaml_pair(line.strip())
        parent = stack[-1][1]
        if value is None:
            section: dict[str, Any] = {}
            parent[key] = section
            stack.append((indent, section))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _split_yaml_pair(line: str) -> tuple[str, str | None]:
    if ":" not in line:
        raise ValueError(f"Invalid config line: {line}")
    key, value = line.split(":", 1)
    value = value.strip()
    return key.strip(), value if value else None


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")
