# Handoff 02 — Phase 2: runtime read-site migration (DONE)

Branch `rl-config-refactor`. **Run sims from repo ROOT** (see 01b). Redis+minio up.

## What changed (runtime now reads the new four-axis schema)
### `src/core/RL_Federate.py`
- `_prepare_obs_dict`: iterates `rl_task.environment.observations.items()` (mapping). Skips
  `role=='extra'` (not in policy obs space). Honors `ObservationSpec.bounds` override (else
  catalog), `.history` for shape. Catalog specs via `observed_models` (covers all obs keys).
- `_prepare_act_dict`: iterates `rl_task.environment.actions.items()`. Reads `ActionSpec`
  `.space`/`.bounds`/`.bins` per key. **Enforces** the deferred check: float (continuous)
  catalog var + discrete/multidiscrete space + `bins is None` → raises with a clear message.
- `__init__`: `episode_length`/`n_episodes` from `rl_task.run.train`; `force_reset_observation_defaults`
  from `rl_task.environment.reset.force_defaults`.
- `run()`: training gate = `rl_task.run.train is not None and rl_task.run.mode != 'offline'`;
  test gate = `rl_task.run.test is not None`. (`rl_task.agent` still valid for `model_name`.)

### `src/core/ScenarioManager.py`
- Setup branch (~303): branches on `run.mode` guarded by `run.train is not None`; test on
  `run.test is not None`.
- Deleted `_resolve_observation_causality` (positional-array helper, obsolete).
- `_get_rl_pubsubs`: single pass over `environment.observations.items()` using per-spec
  `.causality`; collapsed the old main+additional duplicated blocks; deleted length-mismatch
  warnings. Actions from `environment.actions` keys.
- `_get_rl_controlled_models`: `environment.actions` keys.
- `_build_rl_reset_observation_defaults`: explicit defaults from per-spec `.reset_default`;
  iterates `environment.observations` keys.
- `_create_RL_federation`: required_inputs/controlled/observed models from `environment.*`
  mappings; `additional_observed_models` = the `role=='extra'` subset (if any).
- `_modify_config_for_testing`/`_modify_config_for_online_training`: `run.test.total_steps`,
  `run.train.total_steps` (PhaseConfig `.total_steps` property), `run.mode`, `run.train`.

### `src/core/federate_launcher.py`
- Base-federate `rl_config` dict now built from NEW schema: `run.train` (episode_length,
  episodes) + `environment.reset` (mode→reset_type, period→reset_period, rolling_window).
  Same dict shape, so `BaseFederate.__init__` (94–103) is unchanged.

## Two MORE pre-existing bare-co-sim bugs found + fixed (were blocking ALL RL scenarios)
1. **Port allocator** (`ScenarioManager._get_n_available_tcp_ports`, ~1149): break condition
   was `len(available) >= n - len(exclude_ports)` — wrongly subtracted the excluded count from
   the number needed. With an explicit port (e.g. federation_1 `23404`) reserved + the
   hierarchy broker that RL's injected `rl_federation` triggers, it returned one port short →
   "Could not find enough free local TCP ports". Fixed to `>= n` (exclude only *skips* ports).
2. (from 01b) core_name auto-unique + federate stderr capture.

## Scenario edit
- `src/scenarios/simple_DQN_test.yaml`: set the 3 spring observations to `causality: next_step`.
  The original was implicitly all-`same_step`, which forms a `spring_federate ↔ rl_agent`
  same_step cycle (agent observes spring same-step and publishes force back same-step). The
  cycle detector (correctly) rejects it. `next_step` breaks the loop (agent acts on prior-step
  state — standard MDP). Other RL scenarios (bui_hp_*) already have a `next_step` obs.

## Verification
- Core modules import clean. Parse-gate 19/20 (only pre-existing `Adelaide_test.yaml` YAML
  syntax error).
- **Phase 2 integration proven** via `simple_DQN_test` from root: gets fully past setup (RL
  federation built from mappings, ports allocated, causality validated), federates launch,
  RL_Federate builds obs/act dicts, agent is instantiated — then fails with
  `'NoneType' object has no attribute 'batch_size'` *inside `rl_simple_DQN`*. That is the
  expected Phase 3 boundary: agent classes still read OLD paths (`rl_task.agent.hyperparameters`
  shape, `rl_task.training.*`, `rl_task.checkpointing.*`).
- Base regression OK: `pv_batt_test_base` still completes (~2.6s) after the ScenarioManager edits.

## Next: Phase 3 — restructure agent code (the NoneType error is the entry point)
Agents still on old schema (break at init/loop). Files + current old reads:
- `src/models/base_agent_rl.py`: reward via `rl_task.agent.reward_function` →
  `rl_task.environment.reward`; `rl_task.training/test.total_steps` → `rl_task.run.train/test.total_steps`
  (PhaseConfig `.total_steps` property); `rl_task.checkpointing.directory` →
  `rl_task.experiment.checkpoint.dir` (and `.best_path`).
- `src/models/model_catalog/RL_agents/rl_simple_SACsb3.py`: `agent.hyperparameters`
  (now `.as_kwargs()` → only set fields, None omitted), `training.replay_buffer`/
  `train_frequency`/`gradient_steps` → `agent.params` + `agent.hyperparameters`;
  `training.total_steps`/`test.total_steps` → `run.*`; `checkpointing.single_best_checkpoint`
  → `experiment.checkpoint.best`/`.best_path`; `rl_task.seed` unchanged.
- `src/models/model_catalog/RL_agents/rl_simple_DQN.py`: `agent.hyperparameters` (`gradient_clip`
  now in `agent.params`), `training.exploration` → `agent.params['exploration']`,
  `training.replay_buffer` → `agent.params['replay_buffer']`, `training.total_steps` →
  `run.train.total_steps`, `test.total_steps` → `run.test.total_steps`, `checkpointing` →
  `experiment.checkpoint`.
- Plan also asks: extract reusable components into
  `src/models/model_catalog/RL_agents/components/` (replay_buffer, reward_loader,
  checkpoint_manager, env_loop) and make `base_agent_rl.RLAgent` compose them. Per maintainer:
  keep one-class-per-algorithm; components are for reuse/extension by custom agents.

Verify Phase 3 with: base smoke (5/6 approved) + `simple_DQN_test`, `bui_hp_SAC`, `bui_hp_DQN`
train+test from ROOT (re-baselined numbers, not bit-identical — None-default HP).

## Open risks / notes
- `agent.params` is a free dict; agents must read exploration/replay_buffer sub-dicts defensively
  (`.get`), since None-default HP means some keys absent.
- Old reset modes 'soft'/'random' branches still in `BaseFederate` (860,875) are now dead
  (converter maps to full). Prune in Phase 6.
- Stray tracked `src/utils/__pycache__/*.pyc`; gitignore in Phase 6.
- Many of my test runs left `logs/` dirs under repo root (`./logs/...`) — harmless.
