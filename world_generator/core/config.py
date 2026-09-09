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
class LandConfig:
    protected_fraction: float = 0.10
    water_buffer_km: float = 1.5
    steep_slope_threshold: float = 0.22
    high_flood_threshold: float = 0.70
    vegetation_noise_weight: float = 0.25
    random_patch_count: int = 5


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


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"
    world_name: str = "small_debug"


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
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" "):
            key, value = _split_yaml_pair(line)
            if value is None:
                current = {}
                root[key] = current
            else:
                root[key] = _parse_scalar(value)
                current = None
        else:
            if current is None:
                raise ValueError(f"Nested value without section: {raw_line}")
            key, value = _split_yaml_pair(line.strip())
            current[key] = _parse_scalar(value)
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
