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
Added at the top level of the YAML. The agent is referenced by its catalog `model_name`; observations and actions use dot notation `<federation>.<federate>.<instance>.<variable>`:

```yaml
reinforcement_learning_config:
  agent:
    model_name: "rl_simple_DQN"
    reward_function: "models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward"
    env:
      observations:
        - federation_1.spring_federate.0.position
        - federation_1.spring_federate.0.velocity
        - federation_1.spring_federate.0.acceleration
      actions: [federation_1.spring_federate.0.force]
      action_spaces_type: ["discrete"]
    hyperparameters:
      gamma: 0.99
      learning_rate: 0.001
      batch_size: 64
      target_update_interval: 100

  training:
    mode: "online"
    episode_length: 100        # HELICS steps before the episode resets
    n_episodes: 1000
    exploration:
      strategy: "epsilon_greedy"
      initial_epsilon: 1.0
      final_epsilon: 0.05
      epsilon_decay_steps: 5000
    replay_buffer:
      buffer_size: 100000
      prefill_steps: 1000
```

> Full reference for every RL field: [Reinforcement Learning Configuration](../user_guide/scenario_configuration/rl.md).

### 3. The reward
The reward is computed by the function named in `agent.reward_function` (here `spring_oscillation_reward` in `src/models/model_catalog/RL_agents/reward_functions.py`). It penalizes spring oscillation, so the agent learns to apply a damping force.

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
