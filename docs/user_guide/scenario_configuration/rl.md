# Reinforcement Learning Configuration

Add a `reinforcement_learning_config:` block to any scenario YAML to enable RL training or evaluation. When present, ScenarioManager injects a synthetic `rl_agent` federate at runtime and wires its pub/sub connections based on the `env.observations` and `env.actions` lists.

At least one of `training` or `test` must be present.

---

## Structure

```yaml
reinforcement_learning_config:
  seed: 42                    # optional — global RNG seed

  agent:                      # required
    model_name: "rl_simple_DQN"
    reward_function: "models.model_catalog.RL_agents.reward_functions.my_reward"
    algorithm: "DQN"          # optional — passed to agent class
    library: "stable_baselines3"  # optional
    env:                      # required
      observations: [...]
      actions: [...]
      action_spaces_type: [...]
    hyperparameters:          # optional
      learning_rate: 0.001

  training:                   # optional (omit for test-only)
    mode: "online"
    episode_length: 100
    n_episodes: 1000

  test:                       # optional (omit for training-only)
    enabled: true
    checkpoint_path: "src/models/model_catalog/RL_agents/checkpoints/best_model.pth"
    n_episodes: 10

  checkpointing:              # optional
    enabled: true
    directory: "src/models/model_catalog/RL_agents/checkpoints"

  logging:                    # optional
    backend: "tensorboard"
    log_dir: "logs"
```

---

## `agent`

```yaml
agent:
  model_name: "rl_simple_DQN"    # required — key in catalog.yaml pointing to the agent class
  algorithm: "DQN"               # optional — algorithm name passed to the agent
  library: "stable_baselines3"   # optional — RL library identifier
  reward_function: "models.model_catalog.RL_agents.reward_functions.spring_reward"  # optional
  hyperparameters:               # optional
    learning_rate: 0.0003
    gamma: 0.99
    batch_size: 64
  env:                           # required
    ...
```

---

## `agent.env` — Environment definition

Defines the observation and action spaces that the RL agent sees.

```yaml
env:
  observations:                          # required — list of variable keys
    - federation_1.spring_federate.0.position
    - federation_1.spring_federate.0.velocity
    - federation_1.spring_federate.0.acceleration

  additional_observations: []            # optional — extra observations not used as RL inputs

  actions:                               # required — list of variable keys the agent controls
    - federation_1.spring_federate.0.force

  action_spaces_type:                    # required — one entry per action
    - "discrete"                         # "discrete" | "box"

  action_bins: [21]                      # optional — number of bins for discrete actions
  action_boundaries:                     # optional — [min, max] per action for box/discrete
    - [-10.0, 10.0]

  action_space_remapping: null           # optional — remap discrete index to physical value

  observation_causality: null            # optional — "same_step"|"next_step" per observation
  additional_observation_causality: null # optional — causality for additional_observations

  reset_observation_defaults: null       # optional — default values to use at episode reset
  force_reset_observation_defaults: false # optional — always use defaults even when valid values exist

  include_prev_obs: null                 # optional — [n] previous obs to append to state per variable
```

### Key naming convention

All observation and action keys use dot notation:
```
<federation_name>.<federate_name>.<instance_id>.<variable_name>
```

Example: `federation_1.spring_federate.0.position`

- `federation_1` — federation name (dict key under `federations:`)
- `spring_federate` — federate name (dict key under `federate_configs:`)
- `0` — model instance index (zero-based)
- `position` — variable name (must be in the model's `publishes` list)

### Action spaces

| Type | Meaning | Related fields |
|---|---|---|
| `"discrete"` | Integer action space — agent picks an index | `action_bins` sets number of bins |
| `"box"` | Continuous action space | `action_boundaries` sets `[min, max]` |

All action and observation spaces use `gym.Dict` internally. If your RL library requires `Box` or `Discrete`, wrap the environment in the agent class.

---

## `training`

```yaml
training:
  mode: "online"             # "online" | "offline" | "mixed" (default: "online")
  episode_length: 100        # required — simulation steps per episode
  n_episodes: 1000           # required — total number of training episodes
  reset_mode: "full"         # "full" | "partial" (default: "full")
  rolling_window: null       # optional — window size for rolling-reset mode
  warmup_steps: 0            # steps before gradient updates start (default: 0)
  train_frequency: 1         # environment steps between gradient updates (default: 1)
  gradient_steps: 1          # gradient updates per training call (default: 1)
  eval_frequency: 10000      # steps between evaluations (default: 10000)
  n_eval_episodes: 10        # episodes per evaluation (default: 10)
  eval_deterministic: true   # use deterministic policy for eval (default: true)
  log_interval: 100          # steps between log prints (default: 100)
  verbose: 1                 # verbosity level (default: 1)

  exploration:               # optional — controls epsilon-greedy or noise-based exploration
    strategy: "epsilon_greedy"
    initial_epsilon: 1.0
    final_epsilon: 0.05
    epsilon_decay_steps: 100000
    noise_std: 0.1
    noise_std_decay: 0.9999
    noise_std_min: 0.01
    ou_theta: 0.15
    ou_sigma: 0.2

  replay_buffer:             # optional — experience replay settings
    buffer_size: 1000000
    prioritized: false
    alpha: 0.6
    beta: 0.4
    beta_annealing_steps: 100000
    n_step: 1
    prefill_steps: 0

  early_stopping:            # optional
    enabled: false
    metric: "episode_reward"
    patience: 100
    min_delta: 0.01
    mode: "max"
```

### Derived fields (set automatically)

- `reset_period`: set to `episode_length` if not specified
- `total_steps`: set to `n_episodes × episode_length` if not specified

---

## `test`

```yaml
test:
  enabled: true                    # default: false
  checkpoint_path: "src/models/model_catalog/RL_agents/checkpoints/best_model.pth"
  n_episodes: 10                   # optional — number of test episodes
  episode_length: 100              # optional — steps per test episode
  total_steps: null                # optional — total test steps (overrides n_episodes × episode_length)
  deterministic: true              # use deterministic policy (default: true)
  render: false                    # optional — render environment (default: false)
  save_trajectories: false         # save episode trajectories to disk (default: false)
  trajectories_path: "results/test_trajectories.pkl"
```

`checkpoint_path` accepts `"none"`, `"null"`, or `""` as equivalent to `null` (no checkpoint loaded).

---

## `checkpointing`

```yaml
checkpointing:
  enabled: true
  directory: "src/models/model_catalog/RL_agents/checkpoints"
  save_frequency: 10000      # steps between checkpoint saves
  save_best: true            # always save the best model seen so far
  best_metric: "episode_reward"
  best_mode: "max"           # "max" or "min"
  keep_last_n: 5             # number of recent checkpoints to keep
  save_replay_buffer: false  # also save the replay buffer (large!)
  single_best_checkpoint: "best_sac_model.pth"  # optional — filename for the single best checkpoint
```

If `single_best_checkpoint` is a relative path, it is automatically joined with `directory`.

---

## `logging`

```yaml
logging:
  backend: "tensorboard"     # "tensorboard" | "wandb" (default: "tensorboard")
  log_dir: "logs"
  experiment_name: null      # optional label for the run
  project_name: "cosim_gym"
  tags: []                   # optional list of tags
  log_gradients: false
  log_weights: false
  wandb_entity: null         # W&B entity/team name (for wandb backend)
  wandb_mode: "online"       # "online" | "offline" | "disabled"
```

---

## `hyperparameters`

Common fields shared across algorithms:

```yaml
hyperparameters:
  learning_rate: 0.0003
  gamma: 0.99              # discount factor
  batch_size: 64
  net_arch: [64, 64]       # hidden layer sizes
  activation_fn: "relu"    # "relu" | "tanh" | "elu"
  optimizer: "adam"
  gradient_clip: null      # gradient norm clipping threshold
  # PPO-specific
  n_epochs: null
  gae_lambda: null
  clip_range: null
  normalize_advantages: true
  vf_coef: null
  ent_coef: null
  # DQN-specific
  target_update_interval: null
  # SAC-specific
  tau: null
  use_sde: false           # use state-dependent exploration
  algorithm_kwargs: {}     # pass-through dict for any other algorithm arguments
```

---

## Complete RL scenario example

```yaml
name: "spring_DQN"
start_time: "2024-01-01T00:00:00"
end_time:   "2024-01-01T01:00:00"
log_level: DEBUG

memory_config:
  batch_size: 100
  attrs: ["position", "velocity", "force"]

reinforcement_learning_config:
  seed: 42

  agent:
    model_name: "rl_simple_DQN"
    reward_function: "models.model_catalog.RL_agents.reward_functions.spring_reward"
    env:
      observations:
        - federation_1.spring_federate.0.position
        - federation_1.spring_federate.0.velocity
      actions:
        - federation_1.spring_federate.0.force
      action_spaces_type: ["discrete"]
      action_bins: [21]
      action_boundaries: [[-10.0, 10.0]]
    hyperparameters:
      learning_rate: 0.001
      gamma: 0.99
      batch_size: 64
      target_update_interval: 100

  training:
    mode: "online"
    episode_length: 100
    n_episodes: 1000
    exploration:
      strategy: "epsilon_greedy"
      initial_epsilon: 1.0
      final_epsilon: 0.05
      epsilon_decay_steps: 5000
    replay_buffer:
      buffer_size: 100000
      prefill_steps: 1000

  test:
    enabled: false

  checkpointing:
    enabled: true
    directory: "src/models/model_catalog/RL_agents/checkpoints"
    save_best: true
    single_best_checkpoint: "best_spring_dqn.pth"

federations:
  federation_1:
    broker_config:
      core_type: "tcp"
      port: 23404
    federate_configs:
      spring_federate:
        type: "base"
        timing_configs:
          real_period: 1
        connections:
          publishes:
            - key: "position"
              type: "double"
              units: "m"
            - key: "velocity"
              type: "double"
              units: "m/s"
          subscribes:
            - key: "force"     # target omitted — ScenarioManager wires the RL agent here
              type: "double"
              units: "N"
        model_configs:
          instantiation:
            model_name: "spring_mass_damper"
            n_instances: 1
          parameters:
            mass: 5.0
            stiffness: 10.0
            damping: 2.0
          init_state:
            position: 0.0
            velocity: 0.0
            force: 0.0
```

> **Key rule:** For each variable listed in `env.actions`, the corresponding federate subscription must **omit** `targets`. ScenarioManager fills them at runtime with the RL agent's publication address.
