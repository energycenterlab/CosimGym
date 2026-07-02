# Scenario Configuration

A CosimGym simulation is fully described by a single YAML file. No Python scripting is required. The file is validated against Pydantic v2 models at load time, so errors are caught before anything starts.

---

## File location

Place scenario YAML files in `src/scenarios/`. Pass the filename (without extension) to `read_scenario_config`:

```python
from src.utils.config_reader import read_scenario_config
config = read_scenario_config("my_scenario")   # resolves to src/scenarios/my_scenario.yaml
```

---

## Top-level structure

```yaml
name: "my_scenario"                  # required — unique identifier
start_time: "2024-01-01T00:00:00"   # required — ISO 8601
end_time:   "2024-01-02T00:00:00"   # required — ISO 8601
log_level: INFO                      # optional — ERROR|WARNING|INFO|DEBUG (default: INFO)

memory_config:                       # required — controls result storage
  batch_size: 100
  attrs: "all"                       # or a list of variable names

synchronization:                     # optional — time-offset and startup sync settings
  auto_offset:
    enabled: true

reinforcement_learning_config:       # optional — include only for RL scenarios
  environment: ...                    # the MDP (observations/actions/reward/reset)
  agent: ...                          # the solver (model_name/hyperparameters/params)
  run: ...                            # the schedule (train/eval/test)
  experiment: ...                     # infra (name/checkpoint/logging)

federations:                         # required — one or more named federations
  my_federation:
    broker_config: ...
    federate_configs:
      my_federate:
        type: "base"
        ...
```

Unknown top-level keys (e.g. `version`, `scenario_description`, `seed`) are silently ignored.

---

## Hierarchy

```
ScenarioConfig
└── federations: Dict[name → FederationConfig]
    ├── broker_config: BrokerConfig
    └── federate_configs: Dict[name → FederateConfig]
        ├── type: "base" | "rl" | "interface"   ← discriminator
        ├── timing_configs: FedTimingConfig
        ├── flags: FedFlags
        ├── streaming: StreamingConfig      ← optional, all types (outbound MQTT mirror)
        ├── connections: FedConnections
        │   ├── publishes: [FedPublication]
        │   └── subscribes: [FedSubscription]
        ├── model_configs: ModelConfig      ← required for type "base"
        └── interface_config: InterfaceConfig  ← only for type "interface"
```

---

## Minimal working example — plain co-simulation

```yaml
name: "spring_demo"
start_time: "2024-01-01T00:00:00"
end_time:   "2024-01-01T01:00:00"

memory_config:
  attrs: "all"

federations:
  main:
    federate_configs:
      physics:
        type: "base"
        timing_configs:
          real_period: 60        # one step = 60 real-world seconds
        connections:
          publishes:
            - key: "position"
              type: "double"
              units: "m"
          subscribes:
            - key: "force"
              type: "double"
              units: "N"
              targets:
                '0': [driver.0/force]
        model_configs:
          instantiation:
            model_name: "spring_mass_damper"
          parameters:
            mass: 5.0
            stiffness: 10.0
          init_state:
            position: 0.0
            force: 0.0

      driver:
        type: "base"
        timing_configs:
          real_period: 60
        connections:
          publishes:
            - key: "force"
              type: "double"
              units: "N"
        model_configs:
          instantiation:
            model_name: "constant_input"
          init_state:
            force: 5.0
```

---

## Sections

| Document | Covers |
|---|---|
| [General](general.md) | Top-level `ScenarioConfig` fields |
| [Federation](federation.md) | `FederationConfig`, `BrokerConfig`, cross-federation subscriptions |
| [Federate](federate.md) | `FederateConfig`, timing, flags, connections, model_configs |
| [Synchronization](synchronization.md) | Auto time-offset, startup sync, causality |
| [RL](rl.md) | `reinforcement_learning_config` — environment, agent, run, experiment |
| [Digital-Twin Interfaces](../digital_twin_interfaces.md) | `streaming`, `type: interface`, `interface_config` — MQTT mirror, sensor/actuator bridge, output/param override |
