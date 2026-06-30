# RL Config Refactor — Branch Summary

**Branch:** `rl-config-refactor` (6 commits, ~2900 lines added, ~830 removed)

---

## What Changed

### 1. New 4-Axis RL Config Schema

The old `reinforcement_learning_config` was a flat tree of ~70 enumerated fields across 10 Pydantic models (`RLHyperparametersConfig`, `RLExplorationConfig`, `RLReplayBufferConfig`, `RLTrainingConfig`, etc.). Most fields were unused. Observations and actions were specified as fragile positional parallel arrays. `extra='ignore'` silently dropped typos.

**New schema** — four orthogonal axes:

```yaml
reinforcement_learning_config:
  seed: 42
  environment:         # (a) the MDP — observations/actions/reward/reset
    observations:
      federation.federate.instance.variable:
        causality: next_step
        history: 0
        reset_default: 20.0
        role: state          # state | extra
        bounds: [0.0, 40.0]
    actions:
      federation.federate.instance.variable:
        space: box           # box | discrete | multidiscrete | multibinary
        bounds: [16.0, 24.0]
        bins: 9              # required when discretizing continuous
    reward: models.model_catalog.RL_agents.reward_functions.my_reward
    reset:
      mode: full             # full | rolling | none
      force_defaults: false
  agent:               # (b) the solver
    model_name: rl_simple_SACsb3
    backend: stable_baselines3
    algorithm: SAC
    policy: MultiInputPolicy
    hyperparameters:          # all-Optional; omit → backend default
      learning_rate: 0.001
      gamma: 0.99
      batch_size: 64
      net_arch: [256, 256]
      train_frequency: 1
      gradient_steps: 1
    params: {}                # backend-specific escape hatch
  run:                 # (c) the schedule
    mode: online             # online | offline | mixed
    train:
      episodes: 100
      episode_length: 2880
    test:
      episodes: 1
      episode_length: 2880
      deterministic: true
      checkpoint: null
  experiment:          # (d) infra
    name: my_experiment
    checkpoint:
      dir: src/models/model_catalog/RL_agents/checkpoints
      best: best_model.pth
    logging: null
    offline: null
```

**Key changes vs. old format:**

| Before | After |
|---|---|
| `agent.env.observations: [list]` + separate `observation_causality: [list]` + `additional_observations: [list]` + `include_prev_obs: [list]` + `reset_observation_defaults: {dict}` | `environment.observations: { key: ObservationSpec }` — all per-observation config on the spec |
| `agent.env.actions: [list]` + `action_spaces_type: [list]` + `action_boundaries: [list]` + `action_bins: [list]` | `environment.actions: { key: ActionSpec }` — space/bounds/bins on the spec |
| `agent.reward_function` | `environment.reward` — reward belongs to the MDP, not the solver |
| `training.episode_length` + `training.n_episodes` + `training.total_steps` + `training.mode` | `run.train: { episodes, episode_length }` → `total_steps` property |
| `training.reset_mode` + `training.rolling_window` + `training.reset_period` + `agent.env.force_reset_observation_defaults` | `environment.reset: { mode, rolling_window, force_defaults }` |
| `checkpointing.directory` + `checkpointing.single_best_checkpoint` | `experiment.checkpoint: { dir, best }` → `best_path` property |
| `test.total_steps` + `test.checkpoint_path` | `run.test: { episodes, episode_length, checkpoint }` → `total_steps` property |
| `training.exploration.*` + `training.replay_buffer.*` | `agent.params: { exploration: {...}, replay_buffer: {...} }` — backend-specific |
| `extra='ignore'` on all models | `extra='forbid'` on all models — typos fail loudly |

### 2. All 9 Existing RL Scenarios Migrated

A converter script (`scripts/convert_rl_config.py`) performs old→new YAML transformation. All scenarios parse under the new schema.

### 3. Runtime Migrated

- **ScenarioManager** — reads keyed mappings instead of positional arrays; single pass for all observations (no more separate `additional_observations` block); per-obs `reset_default` replaces global dict.
- **federate_launcher** — translates new schema to the flat `rl_config` dict that base (non-RL) federates use for episode sync.
- **RL_Federate** — reads `ObservationSpec`/`ActionSpec` per-entry; `role: extra` observations excluded from policy obs space; validates `bins` required when discretizing continuous variables.
- **base_agent_rl** — reads `run.train.total_steps`, `run.test.total_steps`, `environment.reward`, `experiment.checkpoint`.

### 4. Reusable RL Components Package

`src/models/model_catalog/RL_agents/components/`:

| Component | What it does |
|---|---|
| `ReplayBuffer` | Simple FIFO experience replay (subclass for PER/n-step) |
| `load_reward_function` | Dotted-path import of `environment.reward` callable |
| `CheckpointManager` | Path resolution + dir creation from `experiment.checkpoint` |
| `env_loop` | Generic online/test interaction loops (shared skeleton) |

Custom agents import what they need — no mandatory base beyond `RLAgent`.

### 5. Three Working RL Backends

| Agent class | Backend | Algorithm | Catalog key |
|---|---|---|---|
| `RL_Simple_SACsb3` | SB3 | SAC | `rl_simple_SACsb3` |
| `RL_Simple_DQN` | PyTorch | DQN | `rl_simple_DQN` |
| `RL_Simple_RLlib` | RLlib 2.55.1 | PPO | `rl_simple_rllib` |

### 6. New Example Scenarios

| Scenario | What it demonstrates |
|---|---|
| `bui0_setpoint_SAC.yaml` | RL (SAC/SB3) controls EnergyPlus FMU zone heating set-point |
| `bui0_setpoint_DQN.yaml` | Same MDP, DQN with discretized action (9 bins). Proves same declarative MDP, two algorithms by editing only `agent` + action `space/bins` |
| `simple_rllib_test.yaml` | RLlib PPO on spring-mass-damper (standalone RLModule, no ray workers) |

### 7. Validation Suite

`tests/test_rl_config.py` — 51 tests:
- Parse gate: every scenario YAML validates under `ScenarioConfig`
- RL scenarios: observations/actions non-empty, agent has model_name
- `extra='forbid'` rejection tests (6 models)
- Structural validators: `total_steps`, checkpoint resolution, null coercion, reset defaults, etc.

### 8. Infrastructure Fixes (Pre-existing)

- **core_name uniqueness**: auto-assigns unique HELICS core names instead of crashing on duplicates
- **Port allocator**: fixed off-by-one in `exclude_ports` logic
- **Subprocess logging**: federate stderr captured to `.stdio.log` instead of lost in undrained PIPE
- **federate_launcher**: full traceback on crash (was just one-line error)

---

## Impact on Usage

### Writing New Scenarios

Old parallel-array format is gone. All RL scenarios use keyed mappings:

```yaml
# BEFORE (old)
agent:
  env:
    observations: [fed.model.0.temp, fed.model.0.power]
    observation_causality: [next_step, same_step]
    additional_observations: [fed.model.0.load]
    action_spaces_type: [box]
    actions: [fed.model.0.setpoint]
    action_boundaries: [[16.0, 24.0]]
  reward_function: my_module.my_reward

# AFTER (new)
environment:
  observations:
    fed.model.0.temp:
      causality: next_step
    fed.model.0.power:
      causality: same_step
    fed.model.0.load:
      role: extra
  actions:
    fed.model.0.setpoint:
      space: box
      bounds: [16.0, 24.0]
  reward: my_module.my_reward
```

Shorthand: `fed.model.0.var: null` or `fed.model.0.var:` → default `ObservationSpec`/`ActionSpec`.

### Writing New Agents

1. Subclass `RLAgent` (from `src/models/base_agent_rl.py`)
2. Read config from `self.rl_task` (a `ReinforcementLearningConfig` instance):
   - `self.rl_task.agent.hyperparameters` — universal core (all-Optional)
   - `self.rl_task.agent.params` — backend-specific escape hatch (dict)
   - `self.rl_task.run.train.total_steps` — total training steps
   - `self.rl_task.run.test.total_steps` — total test steps
   - `self.rl_task.experiment.checkpoint` — checkpoint config
3. Compose reusable components:
   ```python
   from .components import CheckpointManager, ReplayBuffer, load_reward_function
   self.ckpt = CheckpointManager(self.rl_task.experiment, self.rl_task.run)
   ```
4. Add entry to `catalog.yaml`, run `catalog_loader.main()` to reload Redis
5. Reference via `agent.model_name: your_catalog_key` in scenario YAML

### Running Simulations

No change to invocation:
```bash
conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # Redis
python src/test_script_rl.py                       # runs active scenario
```

### Hyperparameter Defaults

All `Hyperparameters` fields default to `None`. Unset fields are omitted when forwarded to the backend — the backend (SB3/RLlib/custom) applies its own per-algorithm tuned default. Pin values in YAML only where reproducibility matters.

### Strictness

All RL config models use `extra='forbid'`. A typo in any RL YAML key will raise a Pydantic `ValidationError` at parse time instead of being silently ignored.

### Converting Old Scenarios

```bash
python scripts/convert_rl_config.py src/scenarios/my_old_scenario.yaml
```

Idempotent; writes in-place. Prints a diff summary.

---

## Features Not Yet Implemented — But Ready to Plug In

### 1. Offline Learning (`run.mode: offline | mixed`)

**Schema ready:** `RunConfig.mode` accepts `"offline"` and `"mixed"`. `ExperimentConfig.offline` field exists (currently `Optional[Dict]`).

**Seam:** `ScenarioManager.run()` already branches on `run.mode` (lines dispatching to `_offline_learning()`). `env_loop.py` documents the offline swap point: replace env interaction with a dataset iterator feeding the same `agent.update()`.

**To implement:** write `_offline_learning()` in ScenarioManager, add a dataset loader to `components/`, wire `experiment.offline` config (dataset_path, n_epochs, etc.).

### 2. Parallel / Vectorized Environments

**Schema ready:** `run.train` expresses total work in (episodes × episode_length); nothing assumes a single env.

**Seam:** `env_loop.py` keeps step logic side-effect-free w.r.t. agent internals. Comment documents: "a vectorized/parallel variant would fan `step_episode` over multiple env handles."

**To implement:** vectorized env wrapper around multiple `HelicsGymEnv` instances (each backed by its own HELICS federation), feed batched transitions to agent.

### 3. `role: extra` Observations (Reward-Only Variables)

**Schema ready:** `ObservationSpec.role` accepts `"state"` or `"extra"`. `RL_Federate._prepare_obs_dict()` filters out `role: extra` from the policy observation space. ScenarioManager subscribes to all observations regardless of role.

**Partially wired:** extra-role observations are still returned in the raw obs dict by `HelicsGymEnv`, which causes `KeyError` in SB3/RLlib because the obs space doesn't include them. Used in `bui0_setpoint_SAC.yaml` for `HeatingLoadTarget` but works there only because the reward reads it from the raw dict before the wrapper strips it.

**To implement:** split the obs dict in `HelicsGymEnv._get_obs()` into policy obs (returned to agent) and full obs (passed to reward). Currently the reward function receives the same dict as the agent.

### 4. Eval Phase (`run.eval`)

**Schema ready:** `EvalConfig` model defined with `every_steps`, `episodes`, `deterministic`.

**Not wired:** no runtime code reads `run.eval`. Agents currently train → test with no intermediate evaluation.

**To implement:** in the agent's training loop (or `env_loop`), periodically pause training, run `eval.episodes` deterministic rollouts, log metrics, resume.

### 5. Generic Backend Adapters

**Schema ready:** `AgentConfig.backend` and `AgentConfig.algorithm` are parsed. Three backends (SB3 SAC, PyTorch DQN, RLlib PPO) share the component layer.

**Not wired:** dispatch is still catalog `model_name` → specific class. No generic "given backend=stable_baselines3, algorithm=PPO, construct the right SB3 class."

**To implement:** adapter layer that maps `(backend, algorithm)` → agent class constructor, forwarding `hyperparameters` + `params`. The per-algorithm classes remain as the "full control" path.

### 6. Prioritized Experience Replay / N-Step Returns

**Schema ready:** `ReplayBuffer` is standalone and subclassable. `agent.params.replay_buffer` can carry `prioritized`, `alpha`, `beta`, `n_step` fields.

**Not wired:** `ReplayBuffer` is simple FIFO only.

**To implement:** subclass `ReplayBuffer` with sum-tree priority sampling; read config from `agent.params.replay_buffer`.

### 7. Experiment Logging (TensorBoard / W&B)

**Schema ready:** `ExperimentConfig.logging` field exists (`Optional[Dict]`).

**Not wired:** no runtime reads `experiment.logging`. Agents log via Python `logging` module only.

**To implement:** logging adapter that reads `experiment.logging.backend` and sets up TensorBoard `SummaryWriter` or W&B `wandb.init()`.

### 8. Early Stopping

**Was in old schema** (`RLEarlyStoppingConfig`), removed in refactor.

**Seam:** can be re-added as fields in `run.train` or `run.eval` (metric, patience, min_delta). The training loop in each agent already tracks `best_reward`.

---

## Known Carry-Over Issues

| Issue | Risk | Notes |
|---|---|---|
| `rl_config` dict bridge in BaseFederate | Low | Works; remove when BaseFederate reads typed config |
| Dead reset branches (`soft`/`random` in BaseFederate) | None | Unreachable from new schema, safe to delete |
| zmq multi-federation hierarchy broker | Medium | Fails deterministically; all examples use `core_type: tcp` |
| numpy/matplotlib pins + LD_PRELOAD | Low | ray 2.55.1 dependency cascade; document in Installation_Setup.md |
| Tracked `.pyc` files in git | Cosmetic | `git rm -r --cached '**/__pycache__'` |
| `role: extra` not end-to-end | Low | Don't use `role: extra` until env splits policy vs reward obs |
