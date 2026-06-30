# Reinforcement Learning Integration

CosimGym treats RL integration as a first-class feature rather than a duct-taped add-on. We achieve this by mapping HELICS network topologies natively onto **Gymnasium** environment spaces.

## The Paradigm Shift

In standard Gym definitions:
- The environment computes transition dynamics directly inside `step(action)`.

In CosimGym:
- The `HelicsGymEnv` (hosted inside a specialized `RL_Federate`) runs pure publish/subscribe networking commands.
- An `action` is **published** to a physics `BaseFederate`.
- An `observation` is **subscribed** from the federates' output sensors over the local network.
- The `step(action)` command essentially halts the agent process and advances the global HELICS time clock until the physics federates respond with the next state snapshot.

## Adding RL to your Scenario

To execute a training loop, you do not write a Python pipeline. Instead, inject the `reinforcement_learning_config` block directly into the scenario YAML. It is organized into **four orthogonal axes** — `environment` (the MDP), `agent` (the solver), `run` (the schedule), and `experiment` (infra). The agent is referenced by its catalog `model_name`; its observation and action spaces are wired from `environment.observations` / `environment.actions`, which are **keyed mappings** (one spec per variable, no parallel arrays):

```yaml
reinforcement_learning_config:
  environment:
    observations:
      federation_1.spring_federate.0.position: { causality: next_step }
      federation_1.spring_federate.0.velocity: { causality: next_step }
    actions:
      federation_1.spring_federate.0.force:
        space: box                     # box | discrete | multidiscrete | multibinary
        bounds: [-10.0, 10.0]
    reward: models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward
    reset:
      mode: full                       # full | rolling | none
  agent:
    model_name: rl_simple_SACsb3       # key in catalog.yaml
    backend: stable_baselines3
    algorithm: SAC
    hyperparameters:                   # all Optional → omit to use the backend default
      learning_rate: 0.0003
      gamma: 0.99
      batch_size: 256
  run:
    mode: online
    train:
      episodes: 1000
      episode_length: 96               # simulation steps per episode
  experiment:
    checkpoint:
      dir: src/models/model_catalog/RL_agents/checkpoints
      best: best_sac_sb3_model.pth
```

> For the complete field reference (environment, agent, run, experiment), see [Reinforcement Learning Configuration](scenario_configuration/rl.md). Every RL key is `extra='forbid'` — typos fail loudly at parse time.

### Supported RL Backends

CosimGym ships three working agents, all sharing the reusable component layer in
`RL_agents/components/`:

| Catalog key | Backend | Algorithm |
|---|---|---|
| `rl_simple_SACsb3` | Stable-Baselines3 | SAC |
| `rl_simple_DQN` | custom PyTorch | DQN |
| `rl_simple_rllib` | Ray RLlib (standalone RLModule) | PPO |

Because the internal translation implements a standard `gymnasium.Env`, adding a new backend
means subclassing `RLAgent` and adding a `catalog.yaml` entry — not rewriting the pipeline.

## Understanding Episode Logic & Resets

Because the models executing the physics might be compiled FMUs or distributed systems, "resetting" an episode is complex. Reset semantics live in one place — `environment.reset`:

```yaml
environment:
  reset:
    mode: rolling          # full | rolling | none
    rolling_window: 96     # required when mode == rolling
```

With **`mode: rolling`**, instead of killing physics states the agent treats the scenario as an arbitrarily long continuous timeline. A "reset" merely starts a new logical trajectory segment without forcing the physical simulators to reboot over the network. `mode: full` forces the configured `reset_default` values; `mode: none` never resets. 