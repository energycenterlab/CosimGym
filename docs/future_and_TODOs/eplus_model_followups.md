# EnergyPlus model: follow-ups

Feature: `eplus_model` (plan: `docs/changes_reports/eplus_model_plan.md`).

## EnergyPlus 26.x IDFs (deferred, user decision 2026-11-27)

The four `2_zones_*` IDFs are version 26.1 and cannot run on `eclab-cloud1` (Ubuntu 20.04, glibc 2.31):
the 26.x Linux builds are Ubuntu 24.04 only (26.2 also AlmaLinux 9.6). They also need the missing
`ran2026_rr_rack_workloads_energyplus_year.csv`. Waiting on a newer host or a container worker
(Docker 28 is available here; the registry/worker protocol in the plan allows a non-native worker).
Until then these IDFs are marked "not runnable here" in the validation matrix, not skipped silently.

## Other follow-ups
- Diagnose why `bui0_fmu_test` gets SIGTERM in SETUP on this host, then compare FMU vs API numerically.
- Decide the default of `burn_in_days` (now 0; 14 days gives 0.04 % on BUI0 but costs start-up time).
- Exercise `skip_sizing`, PythonPlugin, `ExternalInterface:To:Variable`, meters in discovery; `.epJSON`/`.imf`; multiple RunPeriods.
- Run the whole regression suite with the new scenarios.
