# Core Concepts

Understanding CosimGym requires familiarity with two main domains: **Co-Simulation** and **Reinforcement Learning (RL)**.

## Co-Simulation with HELICS

CosimGym is built on top of [HELICS](https://helics.org/) (Hierarchical Engine for Large-scale Infrastructure Co-Simulation).

### Publish / Subscribe Data Exchange
In HELICS, models exchange data via a **publish/subscribe** mechanism.
- **Publications:** Outputs from a model (e.g., a temperature sensor sending a reading).
- **Subscriptions:** Inputs to a model (e.g., an HVAC controller reading the temperature).

Models do not interact directly; instead, they interact with a central **Broker** that routes equations and variables correctly, decoupling the internal logic of the models from the network topology.

### Time Synchronization
Co-simulation involves advancing time iteratively.
- The environment steps forward in discrete **time steps**. 
- At each step, every participating model halts its internal computation to broadcast its latest output values and retrieve updated inputs simultaneously.

## Reinforcement Learning in Live Simulations

Reinforcement learning typically follows the Gymnasium loop:
1. Environment (`Env`) starts at an initial state (`reset()`).
2. Agent reads the current observation (`obs`).
3. Agent computes an `action`.
4. The environment advances time using the action (`step(action)`).
5. The environment returns `obs, reward, terminated, truncated, info`.

### The Bridge

To merge these two domains, CosimGym uses the internal `HelicsGymEnv` wrapper. 
It translates the continuous time-stepped simulation into discrete Episodes:
1. RL action outputs are mapped to **Publications** heading to the simulated actuators.
2. Simulated sensor **Subscriptions** are bundled into a Gymnasium **Observation Space**.
3. During `step()`, the simulation advances its time clock to the next decision timestep, pausing until new observations are ready to be passed back to the agent.