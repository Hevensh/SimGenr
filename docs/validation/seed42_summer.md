# Physics validation: small_debug_seed42

Result: **PASS**

Unserved energy: **0.0035516 MWh**; curtailed generation: **0.000126004 MWh**.
A physical-constraint PASS does not imply zero supply shortfall.

| Check | Result | Maximum error / violation | Tolerance |
|---|---|---:|---:|
| finite_static | PASS | — | — |
| finite_daily | PASS | — | — |
| finite_hourly | PASS | — | — |
| finite_source | PASS | — | — |
| finite_storage | PASS | — | — |
| finite_dispatch | PASS | — | — |
| finite_flow | PASS | — | — |
| finite_electrical | PASS | — | — |
| nonempty_consecutive_hourly_timestamps | PASS | — | — |
| timestamps_match_source | PASS | — | — |
| timestamps_match_storage | PASS | — | — |
| timestamps_match_dispatch | PASS | — | — |
| timestamps_match_flow | PASS | — | — |
| population_mass | PASS | 0.00136532 persons | 4.672755103592814 |
| population_nonnegative | PASS | -0 persons/km2 | 1e-07 |
| complete_daily_alignment_271 | PASS | — | — |
| complete_daily_alignment_272 | PASS | — | — |
| complete_daily_alignment_273 | PASS | — | — |
| complete_daily_alignment_274 | PASS | — | — |
| complete_daily_alignment_275 | PASS | — | — |
| complete_daily_alignment_276 | PASS | — | — |
| complete_daily_alignment_277 | PASS | — | — |
| daily_hourly_wind_u | PASS | 1.58946e-07 channel native unit | 0.0002 |
| daily_hourly_wind_v | PASS | 7.94729e-08 channel native unit | 0.0002 |
| daily_hourly_wind_speed | PASS | 1.27157e-06 channel native unit | 0.0002 |
| daily_hourly_temperature | PASS | 3.97364e-07 channel native unit | 0.0002 |
| daily_hourly_humidity | PASS | 1.24176e-08 channel native unit | 0.0002 |
| daily_hourly_pressure | PASS | 1.52588e-05 channel native unit | 0.0002 |
| daily_hourly_cloud | PASS | 1.24176e-08 channel native unit | 0.0002 |
| daily_hourly_precipitation | PASS | 8.64267e-07 mm/day | 0.0002 |
| daily_hourly_irradiance | PASS | 5.72205e-06 channel native unit | 0.0002 |
| nighttime_ghi_zero | PASS | 0 W/m2 | 1e-06 |
| irradiance_below_toa | PASS | 0 W/m2 | 0.0002 |
| hourly_precipitation_nonnegative | PASS | -0  | 1e-06 |
| hourly_irradiance_nonnegative | PASS | -0  | 1e-06 |
| hourly_wind_speed_nonnegative | PASS | 0  | 1e-06 |
| hourly_humidity_lower_bound | PASS | 0  | 1e-06 |
| hourly_humidity_upper_bound | PASS | 0  | 1e-06 |
| hourly_cloud_lower_bound | PASS | -0  | 1e-06 |
| hourly_cloud_upper_bound | PASS | 0  | 1e-06 |
| wind_vector_magnitude | PASS | 9.53674e-07 m/s | 1e-05 |
| exogenous_load_nonnegative | PASS | -0 MW | 1e-06 |
| exogenous_availability_nonnegative | PASS | -0 MW | 1e-06 |
| wind_bus_ac_capacity | PASS | 0 MW | 0.0001 |
| pv_bus_ac_capacity | PASS | 0 MW | 0.0001 |
| unique_final_bus_ids | PASS | — | — |
| dispatch_flow_bus_order | PASS | — | — |
| kcl_incidence | PASS | 6.86646e-05 MW | 0.002 |
| kcl_global_balance | PASS | 6.38962e-05 MW | 0.002 |
| injection_matches_served_dispatch | PASS | 3.05176e-05 MW | 0.002 |
| requested_load_equals_served_plus_unserved | PASS | 9.35135e-06 MW | 0.002 |
| served_load_nonnegative | PASS | -0 MW | 1e-06 |
| unserved_load_nonnegative | PASS | -0 MW | 1e-06 |
| unserved_load_below_requested | PASS | 0 MW | 0.002 |
| line_operating_limit | PASS | 3.05176e-05 MW | 0.002 |
| soc_terminal_boundary | PASS | — | — |
| soc_energy_conservation | PASS | 6.62553e-06 MWh | 0.002 |
| soc_lower_bound | PASS | 0 MWh | 0.002 |
| soc_upper_bound | PASS | 1.52588e-06 MWh | 0.002 |
| soc_cycle_closure | PASS | 0 MWh | 0.002 |
| storage_charge_nonnegative | PASS | -0 MW | 1e-06 |
| storage_discharge_nonnegative | PASS | -0 MW | 1e-06 |
| emergency_part_of_total_discharge | PASS | 0 MW | 1e-05 |
| storage_no_simultaneous_charge_discharge | PASS | 0 MW | 0.0001 |
| storage_shared_inverter_capacity | PASS | 0 MW | 0.002 |
| storage_normal_charge_c_rate | PASS | 0 MW | 0.002 |
| storage_normal_discharge_c_rate | PASS | 3.72529e-07 MW | 0.002 |
| storage_ramp_including_emergency | PASS | 9.53674e-07 MW/hour | 0.002 |
| thermal_operating_limit | PASS | 0 MW | 0.002 |
| thermal_ramp | PASS | 2.28882e-05 MW/hour | 0.002 |

Checks validate exported physical identities and constraints, not empirical realism or forecast accuracy. A PASS can include explicitly reported unserved energy when load shedding is allowed; it does not imply supply adequacy. Capacity factors are descriptive only. Final graph and dispatch use perfect foresight.
