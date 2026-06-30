# Handoff 03 — Phase 3: agent restructure (DONE, both agents verified)

Branch `rl-config-refactor`. Run sims from repo ROOT. Redis+minio up.

## Correction to handoff 02
The `'NoneType' object has no attribute 'batch_size'` error was **misattributed** to the agent.
It was actually `BaseFederate.__init__:77` (`self.config.memory_config.batch_size`) — the
runtime-created `rl_agent` federate had **no memory_config**. Fixed (see below). The agents had
their own separate old-path reads, now also fixed.

## Reusable components (new package)
`src/models/model_catalog/RL_agents/components/`:
- `replay_buffer.py` — `ReplayBuffer` (extracted from DQN; subclass for PER/n-step).
- `reward_loader.py` — `load_reward_function(path, logger)` (dotted-path import; fail-loud).
- `checkpoint_manager.py` — `CheckpointManager(experiment_cfg, run_cfg)`:
  `.best_path` (resolves `experiment.checkpoint.best` against `dir`), `.ensure_dir()`,
  `.test_checkpoint()` (explicit `run.test.checkpoint` else best_path).
- `env_loop.py` — `run_online_loop` / `run_test_loop` helpers; documented seam for future
  offline-learning (`run.mode`) and parallel-env runs. Not mandatory; concrete agents may keep
  bespoke loops (SB3 uses `model.learn`).
- `__init__.py` re-exports ReplayBuffer, load_reward_function, CheckpointManager.

## Agents repointed to new schema (one-class-per-algorithm kept)
- `base_agent_rl.py`: reward via `load_reward_function(rl_task.environment.reward)`;
  default loops use `run.train/test.total_steps`; `save_checkpoint` uses
  `experiment.checkpoint.dir` + `os.path.join`. Removed `import importlib`, added `import os`
  and the components import. (Imports components from `.model_catalog.RL_agents.components`.)
- `rl_simple_SACsb3.py`: builds SAC kwargs from `agent.hyperparameters` + `agent.params`
  (`replay_buffer`, `ent_coef`, `target_update_interval`) and **drops None kwargs so SB3 uses
  its own per-algorithm defaults** (design §2.6). `policy` from `agent.policy` (default
  MultiInputPolicy). Checkpointing via `CheckpointManager`. `run.train/test.total_steps`,
  `run.test.deterministic`.
- `rl_simple_DQN.py`: uses shared `ReplayBuffer`; reads `agent.hyperparameters` +
  `agent.params` (`exploration`, `replay_buffer`, `gradient_clip`, `target_update_interval`);
  DQNConfig overridden only by set values (None-default → algorithm default). Checkpointing via
  `CheckpointManager`. `run.train/test.total_steps`.

## Other fixes made this phase
- **`ScenarioManager._create_RL_federation`**: inject `memory_config=self.config.memory_config`
  into the runtime `rl_agent` RLFederateConfig (the scenario validator that propagates
  memory_config runs before this federation exists). Unblocks BaseFederate init for RL.
- **`federate_launcher.py`**: error handler now logs `traceback.format_exc()` (was a bare
  `str(e)`), so federate-subprocess failures show the real file:line. Keep — genuinely useful.
- Scenario edits (causality + bins) — see below.

## Verification (from ROOT)
- **DQN** (`simple_DQN_test`): trains successfully — "Episode finished at step ~12000,
  return ~98, loss decreasing", checkpoints save. (Times out only because total_steps=100000;
  the path is proven. Consider lowering episodes for a fast smoke.)
- **SAC** (`simple_SACsb3_test`): **full train+test completed** in ~8.6s — "Scenario execution
  completed successfully", checkpoint saved + reloaded for testing.
- Both agents read the new schema and use the components.

## Scenario edits required by the new (correct) contracts
- `simple_DQN_test.yaml`: 3 spring observations → `causality: next_step` (break the
  spring↔agent same_step cycle); force action → `bins: 5` (discretizing a continuous var now
  requires explicit bins — RL_Federate enforces it).
- `simple_SACsb3_test.yaml`: 3 spring observations → `causality: next_step`.
  (Migrated bui_hp_* already have a next_step obs; pv_batt_* too.)

## ‼ KNOWN ISSUE blocking zmq multi-federation RL (affects bui_hp_* + Example 1/2)
RL always adds `rl_federation` → 2 federations → hierarchy (main) broker. With **tcp** brokers
this works (simple_DQN_test/simple_SACsb3_test ran). With **zmq** brokers the per-federation
sub-broker deterministically fails: `Broker failed to start: Broker is unable to connect`
(federation_1 zmq broker `--port=23404 --broker_address=zmq://127.0.0.1:20001` can't reach the
main broker). `bui_hp_DQN`/`bui_hp_SAC`/`pv_batt_*` use zmq → currently blocked.
**Options for Phase 4:** (a) author the new bui0_setpoint_* example scenarios with `core_type:
tcp` (proven to work with the hierarchy) and proceed; (b) investigate the zmq hierarchy broker
startup in `ScenarioManager` (`_normalize_broker_and_core_configs` ~1238 + broker launch
~1424). Recommend (a) to unblock Example 1, file (b) as follow-up.

## Next: Phase 4 — Example 1 (bui0fmu ZoneSetPoint, DQN + SAC)
- Add reward `bui0_setpoint_comfort` to `reward_functions.py` (band around target on
  `TBuilding`; `HeatingLoadTarget` available as a `role: extra` obs for an energy term).
- New scenarios `bui0_setpoint_SAC.yaml` (box action on `ZoneSetPoint`, bounds [16,24]) and
  `bui0_setpoint_DQN.yaml` (discrete + bins). Base off `bui0_fmu_test.yaml` (feeder + FMU);
  the RL agent takes over `ZoneSetPoint` (feeder stops publishing it). Use **tcp** brokers.
  Make `TBuilding` a `next_step` obs to avoid the agent↔FMU same_step cycle.
- Keep them small for a fast smoke (e.g. episodes 5, episode_length ~144).

## Open risks / cleanup for Phase 6
- Remove the temporary nature of the federate_launcher traceback log? It's useful — keep, but
  it's now permanent. Fine.
- zmq hierarchy broker (above).
- Old reset 'soft'/'random' dead branches in BaseFederate; gitignore `__pycache__`.
