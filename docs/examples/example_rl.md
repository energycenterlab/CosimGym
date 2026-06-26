# Example 2: Reinforcement Learning Training

This example shows how the same physics scenario becomes a Deep Q-Network (DQN) training pipeline by adding one YAML block.

## Scenario Source
Found at `src/scenarios/simple_DQN_test.yaml`.

## Key Differences from the Base Case

### 1. The controlled input omits its target
In the base case, `spring_federate`'s `force` subscription pointed at `input_federate`. For RL, the agent supplies the force, so the subscription **omits `targets`** — `ScenarioManager` wires the RL agent's publication automatically:

```yaml
subscribes:
  - key: "force"
    type: "double"
    units: "N"
    # no targets — filled at runtime with the rl_agent publication
```

### 2. The `reinforcement_learning_config` block
Added at the top level of the YAML, organized into four axes (`environment`, `agent`, `run`, `experiment`). Observations and actions are **keyed mappings** using dot notation `<federation>.<federate>.<instance>.<variable>`:

```yaml
reinforcement_learning_config:
  environment:
    observations:
      federation_1.spring_federate.0.position: { causality: next_step }
      federation_1.spring_federate.0.velocity: { causality: next_step }
      federation_1.spring_federate.0.acceleration: { causality: next_step }
    actions:
      federation_1.spring_federate.0.force:
        space: discrete
        bins: 5
    reward: models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward
  agent:
    model_name: rl_simple_DQN
    backend: custom_torch
    algorithm: DQN
    hyperparameters:
      gamma: 0.99
      learning_rate: 0.001
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
    train:
      episodes: 1000
      episode_length: 100        # HELICS steps before the episode resets
    test:
      episodes: 1
      episode_length: 100
      deterministic: true
```

> Full reference for every RL field: [Reinforcement Learning Configuration](../user_guide/scenario_configuration/rl.md).

### 3. The reward
The reward is computed by the function named in `environment.reward` (here `spring_oscillation_reward` in `src/models/model_catalog/RL_agents/reward_functions.py`). It penalizes spring oscillation, so the agent learns to apply a damping force.

> **Same MDP, different solver:** `bui0_setpoint_DQN.yaml` and `bui0_setpoint_SAC.yaml` show one declarative `environment` block solved by two algorithms — only the `agent` block and the action `space`/`bins` change.

## Execution
RL scenarios run from the dedicated entry point, which also sets thread-limiting env vars:

```python
# src/test_script_rl.py
main('simple_DQN_test')
```

```bash
conda activate cosim_gym
python src/test_script_rl.py
```

HELICS fast-forwards the physics, pauses for the Gymnasium `step()` to query the policy, then resumes. Open the dashboard's **Learning Metrics** tab to watch the episode-reward moving average improve as the agent learns.
