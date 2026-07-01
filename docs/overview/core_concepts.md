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

## Digital-Twin Interfaces & Live Streaming

By default a federation is a **closed box**: federates only exchange data over HELICS, and results are only written to disk once, at the end of a run. Two opt-in, MQTT-backed mechanisms open it up while it runs:
- **`streaming.stream: true`** on any federate mirrors its inputs/outputs to MQTT each step, for a live dashboard or external observer — the co-simulation itself is unaffected.
- An **interface federate** (`type: interface`) has no physics model; it relays its wired HELICS connections to and from an external adapter (real sensors/actuators, an operator, or another system), bidirectionally. Because it registers the same HELICS keys a physics-model federate would, swapping simulated hardware for real hardware is a change to *one* federate's block — the config-only sim-to-real pattern.

See [Digital-Twin Interfaces & Live Streaming](../user_guide/digital_twin_interfaces.md) for the full reference.