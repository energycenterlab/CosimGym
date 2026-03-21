# Architecture

CosimGym relies on a layered software architecture designed to separate mathematical definitions from complex multiprocessing execution pipelines. 

## High-Level Execution Flow

When you run a simulation in CosimGym, here is what happens under the hood:

1. **Configuration Reading:**
   The `ScenarioManager` loads raw YAML and parses it through strict Pydantic definitions (`config_dataclasses.py`), ensuring topology rules and variable typings are valid.
2. **Central State Distribution:**
   The `ScenarioManager` pushes both the scenario configuration and the static `ModelCatalog` into a **Redis Data Store**. This prevents messy file-system dependencies; spawned federates just pull their metadata directly from Redis via `RedisCatalog`.
3. **Broker Startup:**
   For every Federation declared, the manager launches a decoupled `helics_broker` subprocess. It also links multiple brokers automatically if there's a multi-federation architecture.
4. **Federate Launching:**
   The manager then triggers `federate_launcher.py` logic which spawns multiple async processes.
   - Standard physics endpoints instantiate as standard `BaseFederate` objects.
   - If Reinforcement Learning is configured, a final `RL_Federate` process is created, which boots up a tailored Gymnasium env wrapped into Stable-Baselines3 algorithms.
5. **Execution Loop:**
   For simulation duration, HELICS manages internal time stepping, ensuring synchronized `publish` / `subscribe` across processes. If an RL agent is present, episode resets are coordinated via a global `reset_mode` signal across federates.
6. **Graceful Teardown:**
   Once the endpoint is reached, or the agent completes testing epochs, the `ScenarioManager` harvests logging output, kills the subprocesses gracefully, and shuts down the brokers.

## Diagram

Below is an illustration representing how the hierarchy of scenarios coordinates independent executable nodes.

<div style="text-align: center;">
<img src="../../images/cosimgymarchitecture.png" alt="Cosim-GYM Architecture" width="800">
<p><i>The CosimGym distributed architecture utilizing HELICS Brokers.</i></p>
</div>
