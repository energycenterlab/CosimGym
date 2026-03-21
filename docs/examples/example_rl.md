# Example 2: Reinforcement Learning Training

In this example, we examine how a YAML file transforms a static physics scenario into an active Deep Q-Network (DQN) training pipeline.

## Scenario Source
Found at `src/scenarios/simple_DQN_test.yaml`.

## Key Differences from Base Case

### 1. Removing the Static Controller
Instead of hardcoding `inputs4spring` to drive the signal, the agent needs to provide the force. The `spring_mass_damper` input subscription is mutated:
```yaml
subscribes:
  - key: input_var
    target: "rl_federation/rl_simple_DQN_1/action_0" 
```
It is now "listening" for a publication coming from an `RL_Federate`.

### 2. Injecting the RL Configuration
At the top of the YAML, the `reinforcement_learning_config` defines the agent:

```yaml
reinforcement_learning_config:
  memory_config:
    - type: "buffer" 
  agent:
    class_name: "DQNAgent"
    module: "src.models.RL_agents.rl_simple_DQN"
    algorithm: "DQN"
  training:
    mode: "online"
    total_timesteps: 1000000 
    episode_length: 500  # Number of HELICS steps before gym signals Terminated
```

### 3. The Objective (Reward Function)
The agent needs a reward. In CosimGym, rewards are evaluated inside the agent's Python code or injected via mappings. In `rl_simple_DQN.py`, the `compute_reward()` function evaluates the current distance of the `spring_mass_damper_1_displacement` from zero (or a target).
- **Goal:** Minimize spring oscillation.
- **Action Space:** Discrete force variables applied to the damper.

## Execution and Analysis
```bash
python src/test_script.py --scenario simple_DQN_test
```
You will notice the simulation taking significantly longer or resetting multiple times depending on the `rolling_reset` settings. HELICS will continuously fast-forward physics, pause, wait for Python Gymnasium `step()` logic to infer a neural network predict, and then resume.

Check the dashboard **Learning Metrics** tab to visualize the moving average of Episode Rewards climbing towards zero as the SAC/DQN agent learns to synthesize perfect critical damping forces automatically.