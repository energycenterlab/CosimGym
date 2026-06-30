# General Scenario Configuration

These fields live at the top level of the scenario YAML.

---

## Full example

```yaml
name: "pv_batt_base"
start_time: "2024-01-01T00:00:00"
end_time:   "2024-01-03T00:00:00"
log_level: DEBUG

memory_config:
  batch_size: 1000
  attrs: "all"

synchronization:
  auto_offset:
    enabled: true
    offset_step: 0.1
    override_existing_offsets: false
  validate_causality_cycles: true

reinforcement_learning_config: ...   # omit for plain co-simulation

federations:
  federation_1: ...
```

---

## Fields

### `name`
- **Required:** yes
- **Type:** string
- **Meaning:** Unique scenario identifier. Used as directory name under `results/` and in logs.

> **Migration note:** The old key `scenario_name` was renamed to `name`. Update any YAML files still using `scenario_name`.

---

### `start_time` / `end_time`
- **Required:** yes
- **Type:** string — ISO 8601 datetime
- **Format:** `"YYYY-MM-DDTHH:MM:SS"` (no timezone suffix)
- **Example:** `"2024-01-01T00:00:00"`

These bound the simulation clock. The total number of HELICS ticks is derived from the time span and the minimum `real_period` across all federates.

---

### `log_level`
- **Required:** no
- **Default:** `INFO`
- **Accepted values:** `CRITICAL` | `ERROR` | `WARNING` | `INFO` | `DEBUG` | `NOTSET`

Scenario-level log level applied to all federates and the ScenarioManager process. Individual federates can override this with their own `log_level` field.

---

### `memory_config`
- **Required:** yes

Controls what simulation variables are stored to disk at the end of a run.

```yaml
memory_config:
  batch_size: 100      # number of timesteps buffered in memory before flushing
  attrs: "all"         # record every variable
  # OR
  attrs:               # record only named variables
    - "position"
    - "velocity"
    - "force"
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `batch_size` | int | `100` | In-memory buffer size before write |
| `attrs` | `"all"` or list of strings | `["all"]` | Variables to record; `"all"` records everything |

This `memory_config` is automatically propagated to every federate that does not define its own. To override for a specific federate, add a `memory_config` block inside that federate's config.

---

### `synchronization`
- **Required:** no
- **Default:** all sub-fields use their defaults (auto_offset enabled, startup_sync enabled)

Controls time-offset computation and startup input validation. See [Synchronization](synchronization.md) for full documentation.

```yaml
synchronization:
  auto_offset:
    enabled: true
    offset_step: 0.1
    override_existing_offsets: false
  default_startup_sync:
    enabled: true
    missing_inputs_policy: "warn"
    invalid_inputs_policy: "warn"
  default_subscription_causality: "same_step"
  validate_causality_cycles: true
```

---

### `federations`
- **Required:** yes
- **Type:** dict — keys are federation names, values are `FederationConfig` objects

```yaml
federations:
  federation_1:
    broker_config: ...
    federate_configs: ...
  federation_2:
    broker_config: ...
    federate_configs: ...
```

The dict key becomes the federation's `name`. It is injected automatically — you do not need to repeat it inside the federation block. For multi-federation scenarios, ScenarioManager automatically creates a hierarchy broker. See [Federation](federation.md).

---

### `reinforcement_learning_config`
- **Required:** no (omit entirely for plain co-simulation)

Configures the MDP, solver, run schedule, and experiment infra across four axes (`environment`, `agent`, `run`, `experiment`). When present, ScenarioManager injects a synthetic `rl_agent` federate into the appropriate federation at runtime. See [RL](rl.md).

```yaml
reinforcement_learning_config:
  environment:
    observations:
      federation_1.spring_federate.0.position: { causality: next_step }
    actions:
      federation_1.spring_federate.0.force:
        space: discrete
        bounds: [-10.0, 10.0]
        bins: 21
    reward: models.model_catalog.RL_agents.reward_functions.spring_oscillation_reward
  agent:
    model_name: rl_simple_DQN
  run:
    train:
      episodes: 500
      episode_length: 100
```

---

### `multi_computer` / `multi_computer_config`
- **Required:** no
- **Default:** `multi_computer: false`

Distributes federates across multiple machines via SSH. Not fully implemented in the current release.

```yaml
multi_computer: true
multi_computer_config:
  ssh_user: "ubuntu"
  ssh_key_path: "/home/user/.ssh/id_rsa"
  hostnames:
    - "192.168.1.10"
    - "192.168.1.11"
```

---

## Ignored fields

The following fields are accepted but silently ignored (useful for documentation inside the YAML):

- `version`
- `scenario_description`
- `seed` (at scenario level — seed inside `reinforcement_learning_config` is used)
- Any other unknown key
