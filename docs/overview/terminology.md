# Framework Terminology

CosimGym leverages a defined set of hierarchical entities to map complex deployments into simple YAML instructions.

## Scenario
A **Scenario** is the top-level orchestrating entity.
- Represents the entire experiment.
- Has a specific `start_time` and `end_time` (or an explicit number of steps for RL).
- Holds multiple **Federations**, global settings (Redis state distribution, `memory_config` result storage), and an optional overarching **RL Agent Configuration**.

## Federation
A **Federation** is a grouped hierarchy of models operating under a single HELICS broker.
- Think of it as a logical "cluster". For example, one federation might handle "Weather and Demographics" while another parallel federation handles "Electrical Grid".
- Contains its own `BrokerConfig` (ports, type).
- Contains one or more **Federates**.

## Federate
A **Federate** is a distinct operating system process (usually spawned dynamically by `ScenarioManager`).
- It represents an active node in the simulation that connects to the Broker.
- Iterates through the HELICS time loop.
- It can represent a traditional mathematical simulation (`BaseFederate`) or the agent handler (`RL_Federate`).
- **Hosts Model instances.**

## Model
A **Model** is the fundamental computational engine residing *inside* a Federate.
- Models must inherit from `BaseModel`.
- Examples include a generic `SpringMassDamper`, a data reader `WeatherCSVReader`, or an FMU.
- It defines `initialize()`, `step()`, and `finalize()`.
- Multiple parameter variations of the same model class can be instantiated simultaneously inside a Federate.

## RL Agent
A specific model intended for Artificial Intelligence endpoints.
- When an RL agent configuration exists, CosimGym dynamically spins up an isolated `RL_Federation` and wires a discrete Gym episode loop, enabling algorithms like `DQN` or `SAC` to interoperate directly with the standard physics Models.

## Interface Federate
A `Federate` (`type: interface`) whose "model" is a network bridge instead of physics.
- Hosts an **Adapter** (e.g. MQTT) instead of a `Model`, relaying its HELICS connections to/from the external world.
- Enables **digital-twin** setups: real sensor-in-the-loop, actuator-out, and runtime output/parameter override — see [Digital-Twin Interfaces & Live Streaming](../user_guide/digital_twin_interfaces.md).

## Adapter
A pluggable transport (`InterfaceAdapter`, e.g. `MqttAdapter`) resolved from the same Model Catalog mechanism as physics Models, via the `interface_adapter` category. Used by both the `streaming.stream` outbound mirror and the Interface Federate.