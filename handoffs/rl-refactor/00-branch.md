# Handoff 00 — Branch + scaffolding (Phase 0 DONE)

## State
- **Branch:** `rl-config-refactor` (off `main` @ bd14fa1). All work here. No commits to `main`.
- **Handoff dir:** `handoffs/rl-refactor/` created.
- **Env:** conda env `cosim_gym` at `/media/space/rando/Environments/cosim_gym`.
  Activate with: `source /opt/anaconda3/etc/profile.d/conda.sh && conda activate cosim_gym`
  (interactive shell starts in `base` — must activate explicitly).
- **Deps verified:** pydantic 2.13.4, gymnasium 1.2.3, stable_baselines3 2.8.0.
- **Services:** `cosim_redis` (6379) + `cosim_minio` (9000) already UP & healthy. No need to
  `docker compose up` again unless they stop.

## Plan reference
`/media/space/rando/.claude/plans/ultraplan-read-carefully-plan-for-rl-co-reactive-shell.md`
Design rationale: `plan_for_rl_config_refactor.md` (repo root).

## Next phase: Phase 1 — New config schema + converter + migrate YAMLs
First action: define new Pydantic RL models (see plan "New schema" + design doc §4.1) in
`src/utils/config_dataclasses.py` (current RL models at ~228–431), `extra='forbid'`. Then
converter `scripts/convert_rl_config.py`, migrate 9 RL scenarios, add parse-gate.

## Open risks
- None yet.

## Context-budget rule
Handoff at ~50% context. This is the durable source of truth if auto-summary drops detail.
