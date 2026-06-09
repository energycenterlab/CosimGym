# Federation Configuration

A federation is a group of federates sharing one HELICS broker. Each entry under the top-level `federations:` dict is a `FederationConfig`.

---

## Structure

```yaml
federations:
  federation_1:                    # dict key becomes the federation name
    broker_config:                 # optional — ScenarioManager fills missing values
      core_type: "zmq"
      port: 23404
      federates: 2
      log_level: INFO
    federate_configs:              # required — dict of federate name → FederateConfig
      my_federate:
        type: "base"
        ...
      another_federate:
        type: "base"
        ...
```

The federation name (dict key) is injected into the config automatically. Each federate's `name` and `id` are similarly injected from the dict key — you do not need to repeat them.

---

## `broker_config`

All fields are optional. ScenarioManager auto-fills anything not specified.

```yaml
broker_config:
  core_type: "zmq"       # transport: "zmq" (default) | "tcp" | "ipc"
  port: 23404            # TCP port for the broker (auto-assigned if omitted)
  federates: 2           # expected federate count (auto-counted if omitted)
  log_level: INFO        # broker log verbosity
  host: "localhost"      # broker host (single-machine default)
  address: ~             # explicit broker address string (advanced)
  broker_address: ~      # parent broker address for hierarchy (advanced, set by ScenarioManager)
  sub_brokers: ~         # number of sub-brokers (multi-federation, set by ScenarioManager)
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `core_type` | string | auto | `"zmq"` for single-machine; `"tcp"` for multi-machine or multi-federation |
| `port` | int | auto | ScenarioManager assigns ports sequentially starting from a base port |
| `federates` | int | auto | If set, validated against the count of `federate_configs` entries |
| `log_level` | LogLevel | `INFO` | Broker-level log verbosity |
| `host` | string | `"localhost"` | Relevant for TCP core type |
| `broker_address` | string | set at runtime | Address of parent/hierarchy broker (multi-federation) |
| `sub_brokers` | int | set at runtime | Number of sub-brokers under a hierarchy broker |

> **Tip:** You can leave `broker_config` entirely empty (or omit it). ScenarioManager will assign a `core_type` and `port` automatically based on the scenario topology.

---

## `federate_configs`

Dict of federate name → `FederateConfig`. The dict key is the federate name and is used to construct the federate's `id` (`<federation_name>_<federate_name>`).

```yaml
federate_configs:
  spring_federate:          # dict key = federate name
    type: "base"
    timing_configs:
      real_period: 60
    ...

  controller_federate:
    type: "base"
    timing_configs:
      real_period: 60
    ...
```

For full `FederateConfig` options, see [Federate](federate.md).

---

## Multi-federation scenarios

When `federations:` contains more than one entry, ScenarioManager automatically:

1. Switches all federations to `core_type: "tcp"`
2. Assigns unique ports to each federation broker
3. Starts a **hierarchy broker** (`helics_broker --sub_brokers=N`) that sits above all per-federation brokers
4. Sets each federation broker's `broker_address` to connect to the hierarchy broker

No extra YAML configuration is needed. Just define multiple federations:

```yaml
federations:
  physics_federation:
    federate_configs:
      spring:
        type: "base"
        ...

  control_federation:
    federate_configs:
      controller:
        type: "base"
        ...
```

---

## Cross-federation subscriptions

A subscription's `targets` field uses different formats depending on whether the publisher is in the same or a different federation.

### Same federation
```yaml
targets:
  '0': [other_federate.0/pub_key]
  '1': [other_federate.1/pub_key]
```
Format: `<federate_name>.<instance_id>/<pub_key>`

### Cross-federation
```yaml
targets:
  '0': [physics_federation.spring.0/position]
```
Format: `<federation_name>.<federate_name>.<instance_id>/<pub_key>`

The instance ID is zero-based and matches the model instance number defined by `n_instances` in the publisher's `model_configs.instantiation`.

---

## Validation

At load time, the config is validated:

- If `broker_config.federates` is set, it must match the count of entries in `federate_configs`
- All federate `id` values within a federation must be unique
- All federate `name` values within a federation must be unique
- For `type: "base"` federates, `model_configs.instantiation.n_instances` must be ≥ 1
