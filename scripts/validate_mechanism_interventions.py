"""Run paired interventions through real B/D/E/F kernels on fixed small worlds.

This is a mechanism protocol, not a calibration or forecast-quality benchmark.
No observed data are downloaded, fitted, or represented by the fixtures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_generator.core.config import HydrologyDynamicConfig, SourceLoadConfig, StorageConfig, WeatherConfig, WorldGridConfig
from world_generator.core.random_state import derive_module_seed
from world_generator.core.datatypes import (
    GridBus, GridEdge, HydrologyState, LandUseState, RefinedGridTopologyState,
    SourceLoadForecastStore, StoragePlanStore, StorageSite, TerrainFeatures, WeatherStore,
)
from world_generator.grid.electrical_builder import build_grid_electrical
from world_generator.hydrology.dynamic_hydrology import generate_dynamic_hydrology
from world_generator.operation.power_flow import solve_dc_power_flow
from world_generator.operation.source_load_forecast import generate_source_load_forecast
from world_generator.operation.storage_dispatch import dispatch_storage_week
from world_generator.weather.physics import clear_sky_transmissivity, diagnose_moist_air, extraterrestrial_hourly_irradiance, latitude_grid
from world_generator.weather.weather_generator import WEATHER_CHANNELS, aggregate_daily_weather, generate_hourly_weather_week


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _paired_rng(seed: int, stream: int) -> tuple[np.random.Generator, np.random.Generator]:
    module_name = {10: "weather_hourly", 30: "source_load"}[stream]
    module_seed = derive_module_seed(seed, module_name)
    return tuple(np.random.default_rng(module_seed) for _ in range(2))


def _check(name: str, passed: bool, *, residual: float | None = None, unit: str = "1") -> dict:
    return {"name": name, "status": "PASS" if bool(passed) else "FAIL", "residual": residual, "unit": unit}


def _metrics(**values: tuple[float, str]) -> dict:
    return {name: {"value": float(value), "unit": unit} for name, (value, unit) in values.items()}


def _row(case_id: str, seed: int, stage: str, relations: list[str], treatment: dict,
         expected: str, before: dict, after: dict, checks: list[dict], controls: dict,
         functions: list[str], limitations: str = "") -> dict:
    return {"case_id": case_id, "seed": seed, "stage": stage, "relation_class": relations,
            "treatment": treatment, "expected_relationship": expected, "before": before, "after": after,
            "checks": checks, "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "controls": controls, "actual_functions": functions, "limitations": limitations}


def _controls(assets: Any, before_hash: str, before_times: np.ndarray, after_times: np.ndarray,
              *, rngs: tuple[np.random.Generator, np.random.Generator] | None = None,
              seed: int, stream: int | None, initial_rng_hash: str | None = None) -> tuple[dict, list[dict]]:
    current_hash = _digest(assets)
    control = {"static_assets_before_sha256": before_hash, "static_assets_after_sha256": current_hash,
               "time_axis_before_sha256": _digest(before_times), "time_axis_after_sha256": _digest(after_times),
               "time_unit": "h", "time_support": "hourly interval [t,t+1)",
               "hours": int(len(before_times)), "cell_size_km": 1.0,
               "module_rng": {"world_seed": seed,
                              "module_name": {10: "weather_hourly", 30: "source_load"}[stream] if stream is not None else None,
                              "derived_seed": derive_module_seed(seed, {10: "weather_hourly", 30: "source_load"}[stream]) if stream is not None else None,
                              "seed_derivation": "core.random_state.derive_module_seed" if stream is not None else None,
                              "initial_state_sha256": initial_rng_hash,
                              "mode": "same initial module stream" if rngs else "deterministic kernel; no RNG argument"}}
    checks = [_check("static_assets_unchanged", current_hash == before_hash),
              _check("identical_hour_axis", np.array_equal(before_times, after_times))]
    if rngs is not None:
        hashes = [_digest(rng.bit_generator.state) for rng in rngs]
        control["module_rng"]["final_state_sha256"] = hashes
        checks.append(_check("same_rng_consumption", hashes[0] == hashes[1]))
    return control, checks


def _daily_fixture() -> tuple[WeatherStore, WeatherConfig, WorldGridConfig]:
    grid = WorldGridConfig(height=1, width=5, cell_size_km=1., latitude_center_degrees=35.)
    config = WeatherConfig(days=2, hourly_week_days=2, start_day_of_year=172,
                           hourly_generation_mode="daily_conditioned", hourly_wind_variability=0.,
                           hourly_wind_direction_std_degrees=0., hourly_cloud_variability=0.,
                           hourly_temperature_diurnal_c=0., hourly_temperature_noise_c=0., pressure_synoptic_hpa=0.)
    values = np.zeros((2, len(WEATHER_CHANNELS), 1, 5))
    values[:, 0] = 4.; values[:, 2] = 4.; values[:, 3] = 20.; values[:, 6] = .6
    q = np.full((2, 1, 5), .005); p0 = np.full_like(q, 1000.)
    elevation = np.zeros((1, 5))
    rh, p, rho = diagnose_moist_air(values[:, 3], q, elevation, p0)
    values[:, 4], values[:, 5] = rh, p
    days = np.array([172, 173])
    for index, day in enumerate(days):
        clear = extraterrestrial_hourly_irradiance(latitude_grid(grid, (1, 5)), float(day)) * clear_sky_transmissivity(elevation)
        values[index, 8] = clear.mean(axis=0) * (1 - config.irradiance_cloud_sensitivity * .6)
    daily = WeatherStore(values, np.zeros((2, 1, 5), int), days, WEATHER_CHANNELS, "day", 172,
                         diagnostics={"specific_humidity_kg_kg": q, "sea_level_pressure_hpa": p0, "air_density_kg_m3": rho},
                         static_elevation_m=elevation)
    return daily, config, grid


def _assets(grid: WorldGridConfig):
    kinds = ("load_bus", "wind_bus", "pv_bus", "thermal_bus", "load_bus")
    capacities = (100., 20., 20., 1000., 100.)
    buses = tuple(GridBus(10*(i+1), kind, 0, i, float(i), 0., capacity, .5, 0., "protocol", i)
                  for i, (kind, capacity) in enumerate(zip(kinds, capacities)))
    edges = tuple(GridEdge(100+i, 40, bus.bus_id, float(abs(3-i)), 1., False, (0, 3), (0, i))
                  for i, bus in enumerate(buses) if bus.bus_id != 40)
    zero = np.zeros((grid.height, grid.width))
    topology = RefinedGridTopologyState(zero, zero, zero, buses, edges)
    return topology, build_grid_electrical(topology)


def _source_pair(context: dict, first: WeatherStore, second: WeatherStore):
    rngs = _paired_rng(context["seed"], 30)
    initial = _digest(rngs[0].bit_generator.state)
    assets = (context["grid"], context["topology"], context["electrical"])
    asset_hash = _digest(assets)
    results = [generate_source_load_forecast(weather, context["topology"], context["electrical"], rng,
                                            config=context["source_config"], grid=context["grid"])
               for weather, rng in zip((first, second), rngs)]
    controls, checks = _controls(assets, asset_hash, results[0].timestamps, results[1].timestamps,
                                 rngs=rngs, seed=context["seed"], stream=30, initial_rng_hash=initial)
    residual_equal = np.array_equal(results[0].diagnostics["load_log_residual"], results[1].diagnostics["load_log_residual"])
    checks.append(_check("load_random_residual_held_fixed", residual_equal))
    return *results, controls, checks


def _reverse_wind(context: dict) -> dict:
    daily, config, grid = context["daily"], context["weather_config"], context["grid"]
    values = daily.dynamic.copy(); values[:, :2] = 0.
    changed = replace(daily, dynamic=values)
    rngs = _paired_rng(context["seed"], 10)
    initial = _digest(rngs[0].bit_generator.state)
    assets = (grid, daily.static_elevation_m)
    asset_hash = _digest(assets)
    before, after = [generate_hourly_weather_week(item, config, rng, grid=grid) for item, rng in zip((daily, changed), rngs)]
    b, a = aggregate_daily_weather(before), aggregate_daily_weather(after)
    controls, checks = _controls(assets, asset_hash, before.timestamps, after.timestamps, rngs=rngs,
                                 seed=context["seed"], stream=10, initial_rng_hash=initial)
    vector_b = float(np.hypot(b.dynamic[:, 0], b.dynamic[:, 1]).mean())
    vector_a = float(np.hypot(a.dynamic[:, 0], a.dynamic[:, 1]).mean())
    scalar_b, scalar_a = float(b.dynamic[:, 2].mean()), float(a.dynamic[:, 2].mean())
    checks += [_check("mean_speed_preserved", abs(scalar_a - scalar_b) < 1e-6, residual=scalar_a-scalar_b, unit="m/s"),
               _check("opposing_vectors_cancel", vector_a < 1e-6, residual=vector_a, unit="m/s"),
               _check("direction_really_reverses", np.all(after.dynamic[:12,0] > 0) and np.all(after.dynamic[12:24,0] < 0))]
    return _row("reverse_wind_scalar_vector", context["seed"], "B", ["P","S"],
                {"field":"daily mean (u,v)","before":[4.,0.],"after":[0.,0.],"unit":"m/s","held_scalar_mean_mps":4.},
                "The hourly opposing vectors cancel while scalar mean speed remains 4 m/s.",
                _metrics(vector_mean_speed=(vector_b,"m/s"),scalar_mean_speed=(scalar_b,"m/s")),
                _metrics(vector_mean_speed=(vector_a,"m/s"),scalar_mean_speed=(scalar_a,"m/s")),checks,controls,
                ["generate_hourly_weather_week","aggregate_daily_weather"],
                "Analytic daily-conditioned construction; no claim about regional wind turning frequency.")


def _cloud_to_pv(context: dict) -> dict:
    daily, config, grid = context["daily"], context["weather_config"], context["grid"]
    values = daily.dynamic.copy(); values[:, 6] = 0.
    values[:, 8] /= 1 - config.irradiance_cloud_sensitivity * .6
    clear = replace(daily, dynamic=values)
    rngs = _paired_rng(context["seed"], 10)
    weather_initial = _digest(rngs[0].bit_generator.state)
    weather_b, weather_a = [generate_hourly_weather_week(item,config,rng,grid=grid) for item,rng in zip((daily,clear),rngs)]
    before, after, controls, checks = _source_pair(context, weather_b, weather_a)
    controls["weather_rng_final_sha256"] = [_digest(rng.bit_generator.state) for rng in rngs]
    controls["weather_module_rng"] = {"world_seed": context["seed"], "module_name": "weather_hourly",
                                      "derived_seed": derive_module_seed(context["seed"], "weather_hourly"),
                                      "seed_derivation": "core.random_state.derive_module_seed",
                                      "initial_state_sha256": weather_initial,
                                      "final_state_sha256": controls["weather_rng_final_sha256"]}
    checks.append(_check("weather_rng_consumption_unchanged", controls["weather_rng_final_sha256"][0] == controls["weather_rng_final_sha256"][1]))
    ghi_b, ghi_a = float(weather_b.dynamic[:,8].sum()/5), float(weather_a.dynamic[:,8].sum()/5)
    pv_b, pv_a = float(before.p_gen_available_mw[:,2].sum()), float(after.p_gen_available_mw[:,2].sum())
    night = weather_a.dynamic[:,8,0,2] == 0
    checks += [_check("clouds_cleared", np.all(weather_a.dynamic[:,6] == 0)),
               _check("clear_sky_increases_ghi", ghi_a > ghi_b), _check("pv_energy_increases", pv_a > pv_b),
               _check("no_twilight_model_has_zero_night_pv", np.all(after.p_gen_available_mw[night,2] == 0))]
    return _row("clear_cloud_to_radiation_to_pv",context["seed"],"B→E",["E","S"],
                {"fields":["daily cloud fraction","jointly feasible daily GHI"],"cloud_before":.6,"cloud_after":0.,"cloud_unit":"1"},
                "Recompute coupled hourly cloud/GHI first; clear-sky energy and PV output increase.",
                _metrics(mean_cell_ghi_energy=(ghi_b,"Wh/m2"),pv_available_energy=(pv_b,"MWh")),
                _metrics(mean_cell_ghi_energy=(ghi_a,"Wh/m2"),pv_available_energy=(pv_a,"MWh")),checks,controls,
                ["generate_hourly_weather_week","generate_source_load_forecast"],
                "Joint feasible GHI anchor changed with cloud; no second cloud multiplier in PV. Night-zero uses the no-twilight solar approximation.")


def _solar_ablation(context: dict) -> dict:
    weather = context["hourly"]
    values = weather.dynamic.copy(); values[:,8] = 0.
    before,after,controls,checks = _source_pair(context,weather,replace(weather,dynamic=values))
    b,a=float(before.p_gen_available_mw[:,2].sum()),float(after.p_gen_available_mw[:,2].sum())
    checks += [_check("positive_baseline_pv",b>0),_check("no_solar_driver_no_pv",a==0),
               _check("load_unchanged_by_direct_ghi_ablation",np.array_equal(before.p_load_mw,after.p_load_mw))]
    return _row("solar_driver_ablation",context["seed"],"E",["P","E","S"],
                {"field":"hourly GHI","after":0.,"unit":"W/m2"},"Zero irradiance drives zero photovoltaic available energy.",
                _metrics(pv_available_energy=(b,"MWh")),_metrics(pv_available_energy=(a,"MWh")),checks,controls,
                ["generate_source_load_forecast"],"Conversion-driver ablation, not a claim that daytime zero GHI and unchanged cloud describe a complete atmospheric realization.")


def _temperature_memory(context: dict) -> dict:
    weather=context["hourly"]
    values=weather.dynamic.copy(); values[24:,3] += 20.
    diagnostics={name: array.copy() for name,array in weather.diagnostics.items()}
    rh,p,rho=diagnose_moist_air(values[:,3],diagnostics["specific_humidity_kg_kg"],weather.static_elevation_m,diagnostics["sea_level_pressure_hpa"])
    values[:,4],values[:,5]=rh,p;diagnostics["air_density_kg_m3"]=rho
    changed=replace(weather,dynamic=values,diagnostics=diagnostics)
    before,after,controls,checks=_source_pair(context,weather,changed)
    difference=after.p_load_mw-before.p_load_mw
    lag=float(after.diagnostics["load_effective_temperature_c"][24,0]-before.diagnostics["load_effective_temperature_c"][24,0])
    before_energy=float(before.p_load_mw[24:].sum());after_energy=float(after.p_load_mw[24:].sum())
    checks += [_check("no_future_temperature_leak",np.array_equal(before.p_load_mw[:24],after.p_load_mw[:24])),
               _check("hot_period_load_increases",after_energy>before_energy),_check("thermal_memory_lags_step",0<lag<20),
               _check("fixed_ghi_hot_pv_derates",after.p_gen_available_mw[24:,2].sum()<before.p_gen_available_mw[24:,2].sum())]
    return _row("temperature_step_memory_load",context["seed"],"B diagnostics→E",["P","E","S"],
                {"field":"primitive temperature from hour 24 onward","increment":20.,"unit":"K","recomputed":["RH","surface pressure","air density"]},
                "Load responds after the step through thermal memory; previous hours and residual innovations stay fixed.",
                _metrics(second_day_requested_energy=(before_energy,"MWh"),max_pre_step_delta=(0.,"MW")),
                _metrics(second_day_requested_energy=(after_energy,"MWh"),max_pre_step_delta=(float(np.abs(difference[:24]).max()),"MW"),first_step_effective_temperature_increment=(lag,"K")),
                checks,controls,["diagnose_moist_air","generate_source_load_forecast"],
                "Controlled temperature response with unchanged specific humidity/sea-level pressure/GHI, not a calibrated heat-wave frequency model.")


def _wind_to_source(context: dict) -> dict:
    weather=context["hourly"]
    values=weather.dynamic.copy();values[:,:2]*=1.4;values[:,2]=np.hypot(values[:,0],values[:,1])
    before,after,controls,checks=_source_pair(context,weather,replace(weather,dynamic=values))
    b,a=float(before.p_gen_available_mw[:,1].sum()),float(after.p_gen_available_mw[:,1].sum())
    checks += [_check("hub_wind_increases",np.all(after.diagnostics["hub_wind_speed_mps"][:,1]>before.diagnostics["hub_wind_speed_mps"][:,1])),
               _check("subrated_wind_energy_increases",a>b),_check("temperature_load_unchanged",np.array_equal(before.p_load_mw,after.p_load_mw))]
    return _row("shared_wind_to_wind_generation",context["seed"],"E",["E","S"],
                {"fields":["wind_u","wind_v"],"multiplier":1.4,"recomputed":"wind_speed=hypot(u,v)","unit":"m/s"},
                "Below rated speed and cut-out, stronger shared near-surface wind increases hub wind and wind energy.",
                _metrics(wind_available_energy=(b,"MWh"),hub_speed=(float(before.diagnostics["hub_wind_speed_mps"][:,1].mean()),"m/s")),
                _metrics(wind_available_energy=(a,"MWh"),hub_speed=(float(after.diagnostics["hub_wind_speed_mps"][:,1].mean()),"m/s")),
                checks,controls,["generate_source_load_forecast"],"Conditional conversion experiment; does not claim monotonic power above cut-out or rerun atmospheric advection.")


def _hydrology(context: dict) -> dict:
    grid=context["grid"];shape=(grid.height,grid.width);zero=np.zeros(shape)
    terrain=TerrainFeatures(*(zero.copy() for _ in range(6)))
    direction=np.full(shape,2,dtype=np.int8);direction[:,-1]=-1
    hydro=HydrologyState(direction,np.ones(shape),zero.astype(bool),zero.astype(bool),zero.astype(bool),zero.copy(),zero.copy(),zero.astype(int),zero.copy(),zero.copy())
    fractions={name:np.full(shape,float(name=="natural")) for name in HydrologyDynamicConfig().impervious_fraction_by_use}
    land=LandUseState(*(zero.copy() for _ in range(6)),zero.astype(int),use_fractions=fractions)
    original=context["hourly"];rain=original.dynamic.copy();rain[:,7]=0.;rain[0,7]=1.
    wet=replace(original,dynamic=rain);dry_values=rain.copy();dry_values[:,7]=0.;dry=replace(original,dynamic=dry_values)
    cfg=HydrologyDynamicConfig(enabled=True,initial_soil_fraction=0.,initial_groundwater_fraction=0.,initial_channel_storage_mm=0.,initial_lake_storage_fraction=0.)
    assets=(grid,terrain,hydro,land);asset_hash=_digest(assets)
    before=generate_dynamic_hydrology(terrain,hydro,land,wet,grid,cfg)
    after=generate_dynamic_hydrology(terrain,hydro,land,dry,grid,cfg)
    disabled=generate_dynamic_hydrology(terrain,hydro,land,wet,grid,replace(cfg,enabled=False))
    controls,checks=_controls(assets,asset_hash,before.timestamps,after.timestamps,seed=context["seed"],stream=None)
    def account(store):
        s,f=store.states,store.fluxes;factor=grid.cell_size_km**2*1000.
        stock=(s["soil_storage_mm"]+s["groundwater_storage_mm"])*factor+s["channel_storage_m3"]+s["lake_storage_m3"]
        local=stock[:-1]+f["precipitation_mm"]*factor+f["boundary_inflow_m3"]+f["routing_inflow_m3"]+f["lake_mixing_inflow_m3"]-stock[1:]-f["actual_et_m3"]-f["boundary_outflow_m3"]-f["routing_outflow_m3"]-f["lake_mixing_outflow_m3"]
        total=stock[0].sum()+f["precipitation_mm"].sum()*factor+f["boundary_inflow_m3"].sum()-stock[-1].sum()-f["actual_et_m3"].sum()-f["boundary_outflow_m3"].sum()
        return stock,float(np.abs(local).max()),float(total)
    b,bl,bg=account(before);a,al,ag=account(after)
    checks += [_check("wet_budget_local",bl<1e-5,residual=bl,unit="m3"),_check("wet_budget_domain",abs(bg)<1e-5,residual=bg,unit="m3"),
               _check("zero_rain_empty_initial_no_water_creation",np.all(a==0)),_check("dry_budget_local",al<1e-5,residual=al,unit="m3"),
               _check("disabled_kernel_returns_static_only",disabled is None)]
    return _row("zero_rain_water_budget_and_disable",context["seed"],"D",["P","S"],
                {"field":"rainfall","before":"1 mm first interval in each cell","after":"0 mm all intervals","secondary_ablation":"enabled=False"},
                "With zero initial stocks and no boundary inflow, removing rain keeps every water inventory zero; disabled mode is explicit static-only.",
                _metrics(rain_volume=(before.budgets["precipitation_m3"].sum(),"m3"),peak_storage=(b.sum(axis=(1,2)).max(),"m3"),budget_residual=(bg,"m3")),
                _metrics(rain_volume=(after.budgets["precipitation_m3"].sum(),"m3"),peak_storage=(a.sum(axis=(1,2)).max(),"m3"),budget_residual=(ag,"m3")),
                checks,controls,["generate_dynamic_hydrology"],"Local conversion is 1000 m3 per mm per km2. Snow, withdrawals, groundwater 3D and backwater remain outside this bucket model.")


def _dispatch_config(**overrides):
    values=dict(cyclic_state_of_charge=False,minimum_soc_fraction=0.,maximum_soc_fraction=1.,
                preferred_soc_lower_fraction=0.,preferred_soc_upper_fraction=1.,normal_dispatch_c_rate=2.,
                thermal_ramp_fraction_per_hour=1.,storage_power_ramp_fraction_per_hour=1.,
                thermal_operating_limit_ratio=1.,line_operating_limit_ratio=1.)
    values.update(overrides)
    return replace(StorageConfig(),**values)


def _fault(context: dict) -> dict:
    source,_,_,_=_source_pair(context,context["hourly"],context["hourly"])
    top,electrical=context["topology"],context["electrical"]
    plan=StoragePlanStore(source.timestamps.copy(),np.array([],int),np.array([],int),np.zeros((48,0)),())
    baseline=solve_dc_power_flow(source,top,electrical)
    intact=np.ones((48,len(electrical.branch_params)),bool);fault=intact.copy()
    branch_index=next(i for i,b in enumerate(electrical.branch_params) if b.to_bus==50)
    fault[24:,branch_index]=False
    assets=(top,electrical,plan);asset_hash=_digest(assets)
    b,a=[dispatch_storage_week(top,electrical,baseline,plan,_dispatch_config(),source_forecast=source,fixed_capacity=True,
                               branch_in_service=mask,thermal_land_limits_mw={40:1000.}) for mask in (intact,fault)]
    before,after=b[0],a[0];flow=a[2]
    controls,checks=_controls(assets,asset_hash,before.timestamps,after.timestamps,seed=context["seed"],stream=None)
    controls["source_realization_sha256"]=_digest(source.as_arrays())
    ens_b,ens_a=before.operation_arrays["unserved_load_mw"],after.operation_arrays["unserved_load_mw"]
    checks += [_check("intact_demand_supplied",ens_b.sum()<1e-6),_check("local_failure_sheds_only_disconnected_load",ens_a[24:,4].sum()>0 and np.all(ens_a[:,:4]==0)),
               _check("failed_branch_flow_zero",np.all(flow.line_flow_mw[24:,branch_index]==0)),
               _check("fixed_assets_no_expansion",all(np.all(getattr(after,name)==0) for name in ("thermal_capacity_expansion_mw","line_capacity_expansion_mva","storage_power_expansion_mw","storage_energy_expansion_mwh")))]
    return _row("line_fault_local_unserved_energy",context["seed"],"F",["P","S"],
                {"field":"branch_in_service","persistent_branch_id":int(flow.branch_ids[branch_index]),"after_hour":24,"before":True,"after":False},
                "The disconnected load island sheds local demand; the intact load remains supplied and failed-line flow is zero.",
                _metrics(unserved_energy=(ens_b.sum(),"MWh")),_metrics(unserved_energy=(ens_a.sum(),"MWh"),unserved_intact_load=(ens_a[:,0].sum(),"MWh")),
                checks,controls,["solve_dc_power_flow","dispatch_storage_week"],"Deterministic specified outage under fixed assets; not an estimated outage probability or all-contingency N-1 test.")


def _reserve_fixture():
    buses=(GridBus(10,"wind_bus",0,0,0.,0.,1.,1.,0.,"protocol",0),GridBus(20,"load_bus",0,1,1.,0.,1.,1.,0.,"protocol",1))
    edge=GridEdge(7,10,20,1.,1.,False,(0,0),(0,1));zero=np.zeros((1,2))
    top=RefinedGridTopologyState(zero,zero,zero,buses,(edge,));electrical=build_grid_electrical(top)
    load=np.array([[0.,.01]]);gen=np.array([[.01,0.]])
    source=SourceLoadForecastStore(np.array([0]),np.array([10,20]),("wind_bus","load_bus"),load,gen,gen.copy(),np.zeros_like(load),())
    site=StorageSite(4,20,0,1,(20,),4.,2.,1.,1.)
    plan=StoragePlanStore(np.array([0]),np.array([20]),np.array([4]),np.zeros((1,1)),(site,))
    return top,electrical,source,plan


def _reserve(context: dict, response_case: bool=False) -> dict:
    top,electrical,source,plan=_reserve_fixture();baseline=solve_dc_power_flow(source,top,electrical)
    assets=(top,electrical,plan);asset_hash=_digest(assets)
    config=_dispatch_config(reserve_contingency_mw=2.,reserve_duration_hours=2.,reserve_response_hours=1.,storage_power_ramp_fraction_per_hour=.25)
    configs=(replace(config,reserve_response_hours=.1),config) if response_case else (config,config)
    initial=(2.,2.) if response_case else (0.,2.)
    results=[dispatch_storage_week(top,electrical,baseline,plan,cfg,source_forecast=source,fixed_capacity=True,initial_soc_mwh_by_site_id={4:energy})[0]
             for cfg,energy in zip(configs,initial)]
    before,after=results
    controls,checks=_controls(assets,asset_hash,before.timestamps,after.timestamps,seed=context["seed"],stream=None)
    reserve_b=float(before.operation_arrays["storage_reserve_mw"].sum());reserve_a=float(after.operation_arrays["storage_reserve_mw"].sum())
    for index,(result,cfg) in enumerate(zip(results,configs)):
        reserve=result.operation_arrays["storage_reserve_mw"]
        balance=np.diff(result.soc_mwh,axis=0)-cfg.charge_efficiency*result.charge_mw+result.discharge_mw/cfg.discharge_efficiency
        energy_violation=float(np.maximum(reserve*cfg.reserve_duration_hours/cfg.discharge_efficiency-result.soc_mwh[1:],0).max())
        response_violation=float(np.maximum(reserve-cfg.storage_power_ramp_fraction_per_hour*result.site_power_capacity_mw*cfg.reserve_response_hours,0).max())
        checks += [_check(f"pair_{index}_soc_balance",np.abs(balance).max()<1e-7,residual=float(np.abs(balance).max()),unit="MWh"),
                   _check(f"pair_{index}_reserve_energy_bound",energy_violation<1e-7,residual=energy_violation,unit="MWh"),
                   _check(f"pair_{index}_reserve_response_bound",response_violation<1e-7,residual=response_violation,unit="MW"),
                   _check(f"pair_{index}_no_charge_while_holding_reserve",np.minimum(result.charge_mw,reserve+result.discharge_mw).max()<1e-7)]
    expected_before=.1 if response_case else 0.
    checks += [_check("expected_before_reserve",abs(reserve_b-expected_before)<1e-7,residual=reserve_b-expected_before,unit="MW"),
               _check("expected_after_energy_limited_reserve",abs(reserve_a-.95)<1e-7,residual=reserve_a-.95,unit="MW")]
    return _row("reserve_response_boundary" if response_case else "initial_soc_reserve_boundary",context["seed"],"F",["P","S"],
                {"field":"reserve_response_hours" if response_case else "initial_soc_mwh_by_site_id", "before":.1 if response_case else 0.,"after":1. if response_case else 2.,"unit":"h" if response_case else "MWh"},
                "Held reserve obeys both response ramp and efficiency-adjusted energy duration; stronger boundary support lowers explicit shortfall.",
                _metrics(held_reserve=(reserve_b,"MW"),reserve_shortfall=(before.operation_arrays["reserve_shortfall_mw"].sum(),"MW")),
                _metrics(held_reserve=(reserve_a,"MW"),reserve_shortfall=(after.operation_arrays["reserve_shortfall_mw"].sum(),"MW")),
                checks,controls,["dispatch_storage_week"],"Island-aggregate reserve only. The one-hour experiment does not activate reserve or certify its network deliverability.")


CASES: tuple[tuple[str,Callable[[dict],dict]], ...] = (
    ("reverse_wind_scalar_vector",_reverse_wind),("zero_rain_water_budget_and_disable",_hydrology),
    ("clear_cloud_to_radiation_to_pv",_cloud_to_pv),("solar_driver_ablation",_solar_ablation),
    ("temperature_step_memory_load",_temperature_memory),("shared_wind_to_wind_generation",_wind_to_source),
    ("line_fault_local_unserved_energy",_fault),("initial_soc_reserve_boundary",_reserve),
    ("reserve_response_boundary",lambda context:_reserve(context,True)),
)


def run_interventions(seeds: tuple[int,...]=(42,123)) -> dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(not isinstance(seed,(int,np.integer)) or seed<0 for seed in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    rows=[]
    for seed in seeds:
        try:
            daily,weather_config,grid=_daily_fixture();topology,electrical=_assets(grid)
            hourly=generate_hourly_weather_week(daily,weather_config,_paired_rng(int(seed),10)[0],grid=grid)
            context={"seed":int(seed),"daily":daily,"weather_config":weather_config,"hourly":hourly,"grid":grid,
                     "topology":topology,"electrical":electrical,
                     "source_config":SourceLoadConfig(pv_tilt_degrees=0.,load_initial_temperature_mode="configured",load_initial_temperature_c=20.)}
        except Exception as error:
            rows.append({"case_id":"fixture_setup","seed":int(seed),"status":"FAIL",
                         "error_type":type(error).__name__,"error":str(error),"checks":[]})
            rows.extend({"case_id":case_id,"seed":int(seed),"status":"NOT_RUN",
                         "reason":"Shared fixture setup failed; the intervention kernel was not executed.",
                         "checks":[],"actual_functions":[]} for case_id,_ in CASES)
            continue
        for case_id,function in CASES:
            try:
                rows.append(function(context))
            except Exception as error:
                rows.append({"case_id":case_id,"seed":int(seed),"status":"FAIL","error_type":type(error).__name__,"error":str(error),"checks":[]})
    for case_id,reason in (
        ("reserve_network_deliverability","Reserve activation is not re-optimized under line limits; only island aggregate reserve is implemented."),
        ("ac_voltage_and_frequency_security","The implemented operating kernel is lossless DC with fixed voltage magnitude and no frequency dynamics."),
        ("twilight_pv_generation","The low-order solar driver omits twilight; strict night-zero is a model boundary, not a universal irradiance claim."),
    ):
        rows.append({"case_id":case_id,"seed":None,"status":"UNSUPPORTED","reason":reason,"checks":[],"actual_functions":[],"relation_class":["P","E"]})
    counts={status:sum(row["status"]==status for row in rows) for status in ("PASS","FAIL","UNSUPPORTED","NOT_RUN")}
    return {"schema_version":"mechanism_interventions_v1","seeds":[int(seed) for seed in seeds],"cases":rows,"counts":counts,
            "supported_execution_status":"FAIL" if counts["FAIL"] else "PASS",
            "coverage_status":"PARTIAL_WITH_EXPLICIT_UNSUPPORTED",
            "scope":"Controlled synthetic mechanisms; no observed-data calibration, regional fit, forecast skill, or reliability probability claim.",
            "pairing_rule":"Static assets and interval axis fixed within each pair; same initial per-module RNG stream and checked final consumption; deterministic D/F kernels take no RNG."}


def render_markdown(report: dict) -> str:
    lines=["# 成对机制干预报告","",report["scope"],"",f"种子：{report['seeds']}。计数：{report['counts']}。未支持项不计入 PASS。","",
           "| 场景 | 种子 | 状态 | 处理 / 预期 |", "|---|---:|---|---|"]
    for row in report["cases"]:
        description=row.get("expected_relationship",row.get("reason",row.get("error","")))
        lines.append(f"| {row['case_id']} | {row['seed'] if row['seed'] is not None else '—'} | {row['status']} | {description.replace('|','/')} |")
    for row in report["cases"]:
        lines += ["",f"## {row['case_id']} / seed={row['seed']}","",f"状态：{row['status']}。"]
        if row["status"] in ("UNSUPPORTED","NOT_RUN"):
            lines += [row["reason"]];continue
        if "error" in row:
            lines += [f"运行错误：{row['error_type']}: {row['error']}"];continue
        lines += [f"阶段：{row['stage']}；关系：{', '.join(row['relation_class'])}。", "", "处理：`"+json.dumps(row["treatment"],ensure_ascii=False)+"`。", "",
                  "| 指标 | 基线 | 处理后 | 单位 |","|---|---:|---:|---|"]
        for name in sorted(set(row["before"])|set(row["after"])):
            b,a=row["before"].get(name),row["after"].get(name)
            lines.append(f"| {name} | {b['value'] if b else '—'} | {a['value'] if a else '—'} | {(a or b)['unit']} |")
        control=row["controls"]
        lines += ["",f"固定资产哈希：`{control['static_assets_before_sha256']}`；时间轴哈希：`{control['time_axis_before_sha256']}`。",
                  f"时间支持：{control['time_support']}；{control['hours']} 小时；网格边长 {control['cell_size_km']} km。",
                  "模块随机流：`"+json.dumps(control["module_rng"],ensure_ascii=False)+"`。",
                  "实际函数："+", ".join(row["actual_functions"])+"。",
                  "检查："+", ".join(f"{item['name']}={item['status']}" for item in row["checks"])+"。",
                  "范围："+row["limitations"]]
    return "\n".join(lines)+"\n"


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,123])
    parser.add_argument("--output-dir",default="outputs/validation/mechanism_interventions")
    args=parser.parse_args(argv)
    output=Path(args.output_dir)
    output=(ROOT/output).resolve() if not output.is_absolute() else output.resolve()
    if output!=ROOT and ROOT not in output.parents:
        parser.error("output-dir must remain inside this repository")
    report=run_interventions(tuple(args.seeds))
    output.mkdir(parents=True,exist_ok=True)
    (output/"mechanism_interventions.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"mechanism_interventions.md").write_text(render_markdown(report),encoding="utf-8")
    print(json.dumps({"counts":report["counts"],"supported_execution_status":report["supported_execution_status"],"coverage_status":report["coverage_status"],"output_dir":str(output)},ensure_ascii=False))
    return 1 if report["counts"]["FAIL"] else 0


if __name__=="__main__":
    raise SystemExit(main())
