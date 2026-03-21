# Scenario Configuration

The entire logic of your simulation runs via a strictly enforced YAML schema. This prevents manual scripting of the HELICS orchestration.

## The YAML Structure
A Scenario YAML file (e.g. `src/scenarios/simple_test.yaml`) maps to the Pydantic classes located in `src/utils/config_dataclasses.py`.

A simplified layout of a simulation looks like this:

```yaml
name: "case0: simple_test"
start_time: "2024-01-01 00:00:00"
end_time: "2024-01-02 00:00:00"

# Optional: Add reinforcement_learning_config here 

federations:
  - name: "federation1"
    broker_config:
      core_type: "zmq"
      port: 23405
      log_level: 3
    federate_configs:
      - name: "Fed_simple"
        type: "base"
        timing_config:
          step_size: 900
        model_configs:
          - model_name: "spring_mass_damper"
            instantiation:
              n_instances: 1
            parameters:
              mass: 10.0
        connections:
          subscribes:
            # Inputs
          publishes:
            # Outputs
```

### Hierarchy Breakdown

1. **`ScenarioConfig`** (Top Level): Defines the overall time window and global configs for metrics memory (InfluxDB) or reinforcement learning parameters.
2. **`FederationConfig`**: Configures the HELICS broker settings (`zmq` vs `tcp`) and groups multiple federate nodes.
3. **`FederateConfig`**: 
    - `type`: usually `'base'` for physics or `'rl'` if wrapping an AI agent.
    - `timing_config`: Defines `step_size` (in seconds).
    - `model_configs`: An array pointing strings to registered entries in the Model Catalog (e.g. `spring_mass_damper`).
4. **`ModelConfig` & `Connections`**: Under `connections`, you explicitly map HELICS publish/subscribe topic strings. If an actor model publishes to `'federation1/agent_actions/damper'`, the physics federate *must* subscribe using that same exact string.

### Strict Validation

When the scenario is executed, `config_validator.py` ensures that all requested `model_name` keys actually exist in your deployed Redis catalog, and that your `parameters` adhere to expected types bounds.

For a full tree of allowable parameters and features, check the `src/utils/config_dataclasses.py` datastructures directly.