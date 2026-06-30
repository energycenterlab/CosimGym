# Handoff 01 — New config schema + converter + migrate YAMLs (Phase 1 DONE)

Branch `rl-config-refactor`. Activate env:
`source /opt/anaconda3/etc/profile.d/conda.sh && conda activate cosim_gym`
**Run sims from `src/`** (CWD matters: `from core...` needs CWD=src). Redis+minio already up.

## What changed
- **`src/utils/config_dataclasses.py`** — replaced the 10 legacy RL models (lines ~228–430)
  with the new four-axis schema, all `extra='forbid'`:
  - `ObservationSpec`, `ActionSpec`, `ResetConfig`, `EnvironmentConfig` (the MDP; obs/actions
    are now keyed MAPPINGS, not parallel arrays). `EnvironmentConfig._coerce_null_specs`
    accepts `key:` (null) as shorthand for a default spec. Non-empty obs+actions enforced.
  - `Hyperparameters` (all-Optional, `.as_kwargs()` returns only set fields), `AgentConfig`
    (model_name/backend/algorithm/policy/hyperparameters/params).
  - `PhaseConfig` (episodes+episode_length, `.total_steps` property, checkpoint none-normalize),
    `EvalConfig`, `RunConfig` (train/eval/test; validates ≥1 phase + test-only needs checkpoint).
  - `CheckpointConfig` (`.best_path` resolves against dir), `ExperimentConfig`.
  - New root `ReinforcementLearningConfig` = seed + environment + agent + run + experiment.
  - **Note:** `ActionSpec.bins` is NOT hard-required for discrete at config level — the
    "bins required when discretizing a CONTINUOUS catalog var" check is deferred to Phase 2
    in `RL_Federate._prepare_act_dict` (needs catalog type). A naturally-integer action may use
    `discrete` with no bins. (simple_DQN_test / simple_test_rlagent rely on this.)
  - `_FederateConfigBase.rl_config` (flat dict) still present — removed in Phase 2.
- **`scripts/convert_rl_config.py`** — one-shot legacy→new converter. Splices only the
  top-level `reinforcement_learning_config:` block (federations + comments preserved verbatim).
  Idempotent. Maps parallel arrays→mappings, reset_observation_defaults→per-obs reset_default,
  additional_observations→role:extra, training/test/checkpointing→run/experiment,
  exploration+replay_buffer+non-core HP→agent.params. Drops unused `action_space_remapping`.
  Infers backend/algorithm from model_name. Usage:
  `python scripts/convert_rl_config.py <file...> [--out X] [--dry-run]`.
- **`scripts/validate_scenarios.py`** — parse-gate: validates every `src/scenarios/*.yaml`
  against `ScenarioConfig`. `python scripts/validate_scenarios.py`.
- **9 RL scenarios migrated in place** (bui_hp_{DQN,SAC}{,_rollingreset}, pv_batt_{DQN,SAC},
  simple_DQN_test, simple_SACsb3_test, simple_test_rlagent).

## Verification done
- Parse-gate: **19/20 pass**. All 9 RL + all valid base scenarios validate under new schema.
  The 1 fail = `Adelaide_test.yaml` — pre-existing YAML **syntax** error (unindented mapping),
  untouched by me, fails `yaml.safe_load` regardless of schema.
- `extra='forbid'` confirmed: a typo'd key (`observationz`) is rejected.
- Diff scope (proof base path is untouched): `git diff --name-only` =
  config_dataclasses.py + 9 RL YAMLs only (+ a stray tracked .pyc). New files: 2 scripts +
  handoffs. **No base-runtime file (BaseFederate/federate_launcher/ScenarioManager) and no
  base scenario YAML changed.** Base federates take the `rl_task=None` path, unchanged.

## ⚠ PRE-EXISTING BLOCKERS (NOT caused by this refactor — present on `main`)
Full co-sim runs currently fail on this machine for reasons independent of the RL schema:
1. **Duplicate `core_name` check** — `ScenarioManager.py:1301` (added in commit `6a391f6`
   "broker and core fixed by claude need to debug...") raises when ≥2 federates in a federation
   share a `core_name`. Trips `simple_test`, `bui_hp_test_base`, and the `bui_hp_*` RL scenarios
   (all use `core_name: fed1` repeatedly). Sharing one core is a valid HELICS pattern, so the
   CHECK appears over-strict — likely a ScenarioManager bug, not a scenario bug.
2. **FMU federate exits code 1** — `bui0_fmu_test` gets past setup but both federates die
   immediately (code 1). No traceback captured because the manager never drains federate
   stderr (`ScenarioManager` PIPEs federate stdout/stderr but only spawns a drain thread for
   BROKERS, line ~1473–1485; federate failure path at ~1583 logs only the exit code). Latent
   issue: undrained federate PIPE can also deadlock on chatty federates.

These block the plan's verification milestones (Phase 3 RL runs, Phase 4 bui0fmu example).
**DECISION PENDING from maintainer** — see below.

## Next phase: Phase 2 — migrate runtime read sites
First action: `ScenarioManager._get_rl_pubsubs` / `_create_RL_federation` etc. — rewrite to
iterate `environment.observations.items()` / `environment.actions.items()` (mapping), collapse
additional-obs blocks into one `role=='extra'` pass, delete length-mismatch warnings. Then
federate_launcher (drop `rl_config` dict), BaseFederate reset reads, RL_Federate obs/act
builders + run dispatch. See plan §Phase 2 + design doc §5.1–5.6.

## Open risks
- The two pre-existing blockers above gate end-to-end testing. May need a small ScenarioManager
  fix (relax/scope the core_name check; drain federate stderr) BEFORE Phase 3 verification can
  pass — arguably a precondition, though strictly outside the RL-config refactor.
- Stray tracked `src/utils/__pycache__/config_dataclasses.cpython-312.pyc` churns on every edit;
  add `__pycache__` to .gitignore + `git rm --cached` in Phase 6.
