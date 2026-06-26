# Reinforcement Learning Configuration

Add a `reinforcement_learning_config:` block to any scenario YAML to enable RL training or
evaluation. When present, `ScenarioManager` injects a synthetic `rl_agent` federate at runtime
and wires its pub/sub connections from `environment.observations` and `environment.actions`.

The block has **four orthogonal axes**:

| Axis | Key | Owns |
|---|---|---|
| (a) Environment | `environment` | The MDP — observations, actions, reward, reset. Framework-agnostic. |
| (b) Agent | `agent` | The solver — catalog class, backend, algorithm, hyperparameters. |
| (c) Run | `run` | The schedule — train / eval / test phases. Single source of truth for length. |
| (d) Experiment | `experiment` | Infra — name, checkpoints, logging, offline data. |

All RL models use Pydantic `extra='forbid'`: **a typo in any key raises a `ValidationError` at
parse time** instead of being silently dropped.

`run` must define at least one of `train` or `test`.

---

## Structure

```yaml
reinforcement_learning_config:
  seed: 42                            # optional — global RNG seed

  environment:                        # (a) the MDP — required
    observations:                     # MAPPING keyed by dotted variable path
      federation_1.spring_federate.0.position:
        causality: next_step
      federation_1.spring_federate.0.velocity:
        causality: next_step
    actions:                          # MAPPING keyed by dotted variable path
      federation_1.spring_federate.0.force:
        space: discrete
        bounds: [-10.0, 10.0]
        bins: 21
    reward: models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward
    reset:
      mode: full                      # full | rolling | none

  agent:                              # (b) the solver — required
    model_name: rl_simple_DQN         # catalog key → concrete agent class
    backend: custom_torch
    algorithm: DQN
    hyperparameters:                  # universal core — all Optional, omit → backend default
      learning_rate: 0.001
      gamma: 0.99
      batch_size: 64
    params:                           # backend-specific escape hatch (free-form dict)
      target_update_interval: 100
      exploration: { strategy: epsilon_greedy, epsilon_decay_steps: 5000 }
      replay_buffer: { buffer_size: 100000, prefill_steps: 1000 }

  run:                                # (c) the schedule — required
    mode: online                      # online | offline | mixed
    train:
      episodes: 1000
      episode_length: 100
    test:
      episodes: 1
      episode_length: 100
      deterministic: true

  experiment:                         # (d) infra — optional
    name: spring_DQN
    checkpoint:
      dir: src/models/model_catalog/RL_agents/checkpoints
      best: best_spring_dqn.pth
```

---

## (a) `environment` — the MDP

### `observations` — keyed mapping

Each observation is a dotted key → `ObservationSpec`. **No parallel arrays** — every per-observation
setting lives on its own spec.

```yaml
observations:
  federation_1.spring_federate.0.position:
    causality: next_step        # same_step | next_step   (default same_step)
    history: 0                  # frame-stack depth, 0 = current only (planned; not yet wired)
    reset_default: 0.0          # value forced at episode reset
    role: state                 # state | extra
    space: null                 # override; else derived from the model catalog
    bounds: null                # [low, high] override; else from catalog
```

Shorthand: `federation_1.spring_federate.0.position:` (null value) → default `ObservationSpec`.

- **`role: state`** — part of the policy observation space (default).
- **`role: extra`** — visible to the reward function / logging but excluded from the policy's
  obs space. *Not yet end-to-end* (see Known Limitations); prefer `state` for now.

### `actions` — keyed mapping

Each action is a dotted key → `ActionSpec`.

```yaml
actions:
  federation_1.spring_federate.0.force:
    space: box                  # box | discrete | multidiscrete | multibinary (default box)
    bounds: [-10.0, 10.0]       # [min, max] override; else from catalog
    bins: 21                    # required when discretizing a continuous variable
```

| `space` | Meaning | Notes |
|---|---|---|
| `box` | Continuous | uses `bounds` |
| `discrete` | Integer index | **`bins` required** — discretizes `bounds` into `bins` levels |
| `multidiscrete` / `multibinary` | Vector actions | advanced |

`bins` is validated at runtime where the catalog type/bounds are known: discretizing a
continuous variable without `bins` raises an error.

### `reward`

```yaml
reward: models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward
```

Dotted import path to a callable in `reward_functions.py`. Reward belongs to the MDP, not the
solver. Optional `termination:` (dotted path → `terminated(obs, action, t) -> bool`) reserved
but not yet wired.

### `reset`

Single home for episode-reset semantics (was spread across four places in the old schema).

```yaml
reset:
  mode: full                    # full | rolling | none      (default full)
  period: null                  # defaults to run.train.episode_length
  rolling_window: null          # required when mode == rolling
  force_defaults: false         # always apply reset_default even when valid values exist
```

`mode: rolling` requires `rolling_window`. Because FMUs/distributed federates cannot be cheaply
`reset()`, rolling treats the run as one long timeline and starts a new logical trajectory
segment without rebooting the physics.

### Key naming convention

```
<federation_name>.<federate_name>.<instance_id>.<variable_name>
```

Example `federation_1.spring_federate.0.position`: `federation_1` federation key,
`spring_federate` federate key, `0` zero-based instance index, `position` a published variable.

---

## (b) `agent` — the solver

```yaml
agent:
  model_name: rl_simple_DQN     # required — catalog.yaml key → concrete agent class
  backend: custom_torch         # informational now; reserved for adapter dispatch
  algorithm: DQN                # informational
  policy: null                  # e.g. MultiInputPolicy (SB3)
  hyperparameters:              # universal core — see below
    learning_rate: 0.001
    gamma: 0.99
  params: {}                    # backend-specific dict, forwarded by the agent class
```

The agent class is selected solely by `model_name`. `backend`/`algorithm` are documentation
today (each catalog class hard-codes its library + algorithm). `params` is the escape hatch for
anything not in the typed core — the agent class reads what it needs (e.g. DQN reads
`params.exploration`, `params.replay_buffer`, `params.target_update_interval`).

### `hyperparameters` — universal core

All fields are **Optional and default to `None`**. Unset fields are omitted when forwarded to
the backend, so the backend applies its own per-algorithm tuned default. Pin a value only where
reproducibility matters.

```yaml
hyperparameters:
  learning_rate: null
  gamma: null
  batch_size: null
  net_arch: null                # [hidden, hidden]
  train_frequency: null
  gradient_steps: null
```

---

## (c) `run` — the schedule

Single source of truth for "how long". Each phase derives `total_steps = episodes × episode_length`.

```yaml
run:
  mode: online                  # online | offline | mixed   (default online)
  train:                        # PhaseConfig
    episodes: 1000
    episode_length: 100
  eval:                         # optional periodic eval (schema present, runtime not yet wired)
    every_steps: 10000
    episodes: 10
    deterministic: true
  test:                         # PhaseConfig
    episodes: 1
    episode_length: 100
    deterministic: true
    checkpoint: null            # null → use the best produced by train
```

- A **test-only** run (no `train`) **requires** `test.checkpoint`.
- `checkpoint` accepts `"none"`, `"null"`, or `""` as equivalent to `null`.

---

## (d) `experiment` — infrastructure

```yaml
experiment:
  name: spring_DQN
  checkpoint:
    dir: src/models/model_catalog/RL_agents/checkpoints
    best: best_spring_dqn.pth   # resolved against dir unless absolute / already under dir
  logging: null                 # schema present; runtime not yet wired
  offline: null                 # only when run.mode in {offline, mixed}
```

`experiment.checkpoint.best_path` resolves `best` against `dir` automatically.

---

## Complete RL scenario example

See `src/scenarios/simple_DQN_test.yaml` for a runnable spring-mass-damper DQN scenario, and
`src/scenarios/bui0_setpoint_DQN.yaml` / `bui0_setpoint_SAC.yaml` for the same EnergyPlus FMU
MDP solved by two different algorithms (only the `agent` block and the action `space`/`bins`
change).

```yaml
reinforcement_learning_config:
  seed: 42
  environment:
    observations:
      federation_1.spring_federate.0.position: { causality: next_step }
      federation_1.spring_federate.0.velocity: { causality: next_step }
    actions:
      federation_1.spring_federate.0.force:
        space: discrete
        bounds: [-10.0, 10.0]
        bins: 21
    reward: models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward
    reset:
      mode: full
  agent:
    model_name: rl_simple_DQN
    backend: custom_torch
    algorithm: DQN
    hyperparameters:
      learning_rate: 0.001
      gamma: 0.99
      batch_size: 64
    params:
      target_update_interval: 100
      exploration:
        strategy: epsilon_greedy
        initial_epsilon: 1.0
        final_epsilon: 0.05
        epsilon_decay_steps: 5000
      replay_buffer:
        buffer_size: 100000
        prefill_steps: 1000
  run:
    mode: online
    train:
      episodes: 1000
      episode_length: 100
    test:
      episodes: 1
      episode_length: 100
      deterministic: true
  experiment:
    name: spring_DQN
    checkpoint:
      dir: src/models/model_catalog/RL_agents/checkpoints
      best: best_spring_dqn.pth
```

> **Key rule:** For each variable listed in `environment.actions`, the corresponding federate
> subscription must **omit** `targets`. `ScenarioManager` fills it at runtime with the RL
> agent's publication address.

---

## Available agents

| Catalog key | Backend | Algorithm |
|---|---|---|
| `rl_simple_SACsb3` | Stable-Baselines3 | SAC |
| `rl_simple_DQN` | custom PyTorch | DQN |
| `rl_simple_rllib` | Ray RLlib (standalone RLModule) | PPO |

Add a new agent by subclassing `RLAgent` (`src/models/base_agent_rl.py`), composing the reusable
components in `RL_agents/components/` (`ReplayBuffer`, `CheckpointManager`, `load_reward_function`,
`env_loop`), and adding a `catalog.yaml` entry.

---

## Known Limitations

| Limitation | Notes |
|---|---|
| `role: extra` not end-to-end | Excluded from obs space but `HelicsGymEnv` still returns it → `KeyError` in SB3/RLlib. Use `role: state`. |
| `run.eval` not wired | `EvalConfig` parses but no runtime reads it; agents go train → test. |
| `experiment.logging` not wired | Agents log via Python `logging` only. |
| `history` (frame-stacking) | Parsed, not yet implemented. |
| `run.mode: offline / mixed` | Schema + seam ready; `_offline_learning()` not implemented. |

See `handoffs/rl-refactor/SUMMARY.md` for the full future-work list and implementation seams.
