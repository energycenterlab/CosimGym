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

To execute a training loop, you do not write a Python pipeline. Instead, inject the `reinforcement_learning_config` block directly into the scenario YAML. The agent is referenced by its catalog `model_name`; its observation and action spaces are wired from `env`:

```yaml
reinforcement_learning_config:
  agent:
    model_name: "rl_simple_SACsb3"     # key in catalog.yaml
    reward_function: "models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward"
    env:
      observations: [federation_1.spring_federate.0.position, federation_1.spring_federate.0.velocity]
      actions:      [federation_1.spring_federate.0.force]
      action_spaces_type: ["box"]      # "discrete" | "box"
    hyperparameters:
      learning_rate: 0.0003
      gamma: 0.99
      batch_size: 256
  training:
    mode: "online"
    episode_length: 96                 # simulation steps per episode
    n_episodes: 1000
  checkpointing:
    enabled: true
    directory: "src/models/model_catalog/RL_agents/checkpoints"
    save_frequency: 5000
```

> For the complete field reference (training, exploration, replay buffer, test, logging), see [Reinforcement Learning Configuration](scenario_configuration/rl.md).

### Supported RL Libraries

CosimGym includes out-of-the-box wrappers for:
- **Stable-Baselines3** (e.g., DQN, SAC)
- Example implementations: `rl_simple_DQN.py`, `rl_simple_SACsb3.py` located in `src/models/model_catalog/RL_agents/`.

Because the internal translation implements a standard `gymnasium.Env`, it is relatively trivial to bolt on libraries like **Ray RLlib** with minor modifications.

## Understanding Episode Logic & Resets

Because the models executing the physics might be compiled FMUs or distributed systems, "resetting" an episode is complex.

CosimGym implements **`rolling_reset`**: Instead of killing physics states, the agent considers the scenario as an arbitrarily long continuous timeline. A "reset" for the Gym environment merely signifies the start of a logical new trajectory segment without forcing physical simulators to reboot completely over the network. 