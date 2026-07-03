# Federate Configuration

Each entry in `federate_configs` is a `FederateConfig`. The `type` field is the discriminator: `"base"` for physics federates, `"rl"` for the RL agent federate, `"interface"` for a digital-twin bridge federate (see [`type: "interface"`](#type-interface-interface-federate-digital-twin-bridge) below, and the full [Digital-Twin Interfaces & Live Streaming](../digital_twin_interfaces.md) reference).

---

## Full example — `type: "base"`

```yaml
federate_configs:
  spring_federate:
    type: "base"              # required — "base" | "rl" | "interface"
    log_level: DEBUG          # optional — overrides scenario-level log_level
    core_type: "zmq"          # optional — default "zmq"
    core_name: "fed1"         # optional — HELICS core identifier

    timing_configs:
      real_period: 60         # required — seconds of real-world time per simulation step

    flags:
      terminate_on_error: true
      wait_for_current_time_update: true

    startup_sync:             # optional — overrides scenario-level default_startup_sync
      enabled: true
      missing_inputs_policy: "warn"

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
            '0': [driver_federate.0/force]
          causality: "same_step"

    model_configs:
      instantiation:
        model_name: "spring_mass_damper"
        n_instances: 2
        prefix: "spring"
        parallel_execution: false
      parameters:
        mass: [5.0, 5.0]
        stiffness: [10.0, 20.0]
      init_state:
        position: [0.0, 100.0]
        velocity: 0.0
        force: 0.0
      user_defined:
        solver: "rk4"
```

---

## Common fields (both types)

| Field | Required | Default | Type | Meaning |
|---|---|---|---|---|
| `type` | yes | — | `"base"` \| `"rl"` \| `"interface"` | Federate class discriminator |
| `log_level` | no | scenario `log_level` | LogLevel | Federate-level log verbosity |
| `core_type` | no | `"zmq"` | string | HELICS transport: `"zmq"` \| `"tcp"` \| `"ipc"` |
| `core_name` | no | `null` | string | HELICS core name (advanced) |
| `broker_address` | no | set at runtime | string | Explicit broker address (set by ScenarioManager) |
| `startup_sync` | no | scenario default | StartupSyncConfig | Per-federate startup sync override |
| `streaming` | no | `{stream: false}` | StreamingConfig | Opt-in outbound MQTT mirror of this federate's I/O. All types. See [Digital-Twin Interfaces](../digital_twin_interfaces.md). |
| `override_enabled` | no | `false` | bool | `base`/`rl` federates only: opt in to accepting output/param overrides from an interface federate's `bridges`. |

> `name` and `id` are injected automatically from the dict key and federation name. Do not set them manually.

---

## `timing_configs`

```yaml
timing_configs:
  real_period: 60        # required — real-world seconds per federate step
  time_offset: 0.0       # optional — fractional HELICS units; set by auto_offset if enabled
  timeout: 30            # optional — max seconds to wait for HELICS grant (default: 30)
  int_max_iterations: 10000  # optional — max HELICS iterations per step (default: 10000)
  rt_lag: 1.0            # optional — realtime lag tolerance (seconds); requires flags.realtime: true
  rt_lead: 1.0           # optional — realtime lead tolerance (seconds); requires flags.realtime: true
```

| Field | Required | Default | Meaning |
|---|---|---|---|
| `real_period` | **yes** | — | Seconds of real-world simulation time per step. The only required timing field. |
| `time_offset` | no | `0.0` | Fractional HELICS time units added to this federate's requests. Computed automatically by `auto_offset` unless you set it manually. |
| `timeout` | no | `30` | Seconds before a HELICS time-grant request is considered failed. |
| `int_max_iterations` | no | `10000` | Max HELICS iteration count per time step. |
| `rt_lag` / `rt_lead` | no | `null` | Tunable wall-clock pacing tolerance (HELICS `time_rt_lag`/`time_rt_lead`), only applied when `flags.realtime: true`. Used by interface federates to give an external process a wall-clock window to publish/react each step. |

ScenarioManager normalizes all federates to the same tick size (the minimum `real_period`). A federate with `real_period: 120` steps every 2 ticks; one with `real_period: 60` steps every tick.

---

## `flags`

All HELICS federate flags. All default to `false` except `terminate_on_error` which defaults to `true`.

```yaml
flags:
  terminate_on_error: true         # kill federate on any HELICS error (recommended)
  wait_for_current_time_update: true  # wait for all publishers at current tick before stepping
  uninterruptible: false           # prevent HELICS from interrupting at non-requested times
  observer: false                  # receive-only federate (no publications)
  source_only: false               # publish-only federate (no subscriptions)
  only_update_on_change: false     # only update subscription value when it changes
  only_transmit_on_change: false   # only transmit publication when value changes
  realtime: false                  # enforce wall-clock pacing
  debugging: false                 # enable HELICS debug output
  slow_responding: false           # suppress slow-response warnings
  single_thread_federate: false    # run federate in single thread
  ignore_time_mismatch_warnings: false
  strict_config_checking: false
  force_logging_flush: false
  dumplog: false
  restrictive_time_policy: false
  rollback: false
  forward_compute: false
  event_triggered: false
```

Commonly used flags:
- `terminate_on_error: true` — recommended for all federates
- `wait_for_current_time_update: true` — ensures a federate waits for all upstream publishers before stepping (useful when `time_offset` order is tight)
- `observer: true` — for logging-only federates that only subscribe

---

## `connections`

Defines the HELICS pub/sub interfaces for this federate.

```yaml
connections:
  publishes:
    - key: "position"     # variable name used inside the model
      type: "double"      # HELICS type: "double" | "string" | "complex" | "vector" | "boolean" | "integer"
      units: "m"          # unit of measurement (must match subscriber's units)

  subscribes:
    - key: "force"
      type: "double"
      units: "N"
      targets:
        '0': [driver.0/force]       # targets for model instance 0
        '1': [driver.1/force]       # targets for model instance 1
      causality: "same_step"        # "same_step" (default) | "next_step"
      multi_input_handling: null    # aggregation strategy when multiple targets (advanced)

  endpoints: []   # not currently implemented — leave empty or omit
```

### Publication fields

| Field | Required | Type | Meaning |
|---|---|---|---|
| `key` | yes | string | Variable name. Must match the model's output variable name. |
| `type` | yes | string | HELICS data type (`"double"`, `"string"`, `"vector"`, etc.) |
| `units` | yes | string | Physical unit. Must be consistent with subscribers. |

### Subscription fields

| Field | Required | Type | Meaning |
|---|---|---|---|
| `key` | yes | string | Variable name. Must match the model's input variable name. |
| `type` | yes | string | HELICS data type. |
| `units` | yes | string | Physical unit. |
| `targets` | yes* | dict or list | What to subscribe to. *Omit for RL-controlled inputs (filled by ScenarioManager). |
| `causality` | no | string | `"same_step"` (default) or `"next_step"`. See [Synchronization](synchronization.md). |
| `multi_input_handling` | no | string/dict | Aggregation when multiple targets feed one input (advanced). |

### `targets` format

For a federate with `n_instances: N`, each instance needs its own target list keyed by string instance index:

```yaml
targets:
  '0': [federate_name.0/pub_key]
  '1': [federate_name.1/pub_key]
```

For a single instance (`n_instances: 1`):
```yaml
targets:
  '0': [federate_name.0/pub_key]
```

Or use a list (same target applied to all instances):
```yaml
targets: [federate_name.0/pub_key]
```

Cross-federation target format: `<federation_name>.<federate_name>.<instance_id>/<pub_key>`

**RL-controlled subscriptions:** When an RL agent controls a variable, omit `targets` on the corresponding subscription. ScenarioManager wires the RL agent's output to that subscription automatically.

---

## `model_configs` (required for `type: "base"`)

Specifies the model to instantiate and its configuration.

```yaml
model_configs:
  instantiation:
    model_name: "spring_mass_damper"  # required — key in catalog.yaml
    n_instances: 2                     # optional — number of parallel model instances (default: 1)
    prefix: "spring"                   # optional — instance naming prefix (default: "model")
    parallel_execution: false          # optional — step instances in parallel worker processes (default: false)
    max_parallel_workers: null         # optional — cap on worker processes (default: min(n_instances, cpu_count))

  parameters:                          # optional — model parameters (overrides catalog defaults)
    mass: [5.0, 5.0]                   # scalar applies to all instances; list assigns per-instance
    stiffness: [10.0, 20.0]

  init_state:                          # optional — initial values for model state variables
    position: [0.0, 100.0]
    velocity: 0.0

  user_defined:                        # optional — arbitrary dict passed to the model
    solver: "rk4"
    integrator: "fixed-step"
```

### `instantiation` fields

| Field | Required | Default | Meaning |
|---|---|---|---|
| `model_name` | **yes** | — | Key in `src/models/model_catalog/catalog.yaml`. Must exist. |
| `n_instances` | no | `1` | Number of model instances. Instances are named `<prefix>.0`, `<prefix>.1`, etc. |
| `prefix` | no | `"model"` | Prefix for instance names. |
| `parallel_execution` | no | `false` | If true, this federate's model instances are stepped concurrently in persistent worker **processes** (see below). |
| `max_parallel_workers` | no | `null` | Cap on worker processes. `null` → `min(n_instances, cpu_count())`. Must be `>= 1` if set. |

### Parallel model-instance execution (`parallel_execution`)

By default a federate steps its `n_instances` model instances **sequentially** each tick. When a model's `step()` is CPU-heavy, this is a bottleneck. Set `parallel_execution: true` to fan the per-tick `step()` compute out to a pool of **persistent worker processes** (`src/core/parallel_executor.py`), each owning a stateful shard of the instances that lives for the whole run. The main federate process keeps all HELICS I/O, storage and publishing.

- **Processes, not threads** — the target models are pure-Python (GIL-bound), so threads give no speedup. Workers rebuild their model shard from config (live model objects aren't picklable).
- **When it helps** — only when per-instance `step()` is genuinely CPU-heavy. For cheap steps, inter-process overhead can outweigh the gain (measured: speedup climbs as step cost grows, toward the worker-count ceiling). Light models (e.g. `rc_building`) see little/no benefit.
- **Cleanup** — workers are `daemon` processes shut down on every exit path (normal end, exception, SIGINT/SIGTERM) via an escalating `close()` (sentinel → join → terminate → kill) plus `atexit`/signal handlers. No orphan processes.
- **Not supported with** (raises `NotImplementedError`): `override_enabled: true` (digital-twin param/output overrides act on the main process's non-stepping model copies) and `type: rl` federates.

Benchmark scenarios: `src/scenarios/benchmark_parallel_seq.yaml` vs `benchmark_parallel_par.yaml` (identical except `parallel_execution`), using the CPU-heavy `heavy_compute_dummy` model.

### Per-instance vs scalar values in `parameters` / `init_state`

When `n_instances > 1`, you can assign different values per instance using lists:

```yaml
parameters:
  mass: [5.0, 10.0]    # instance 0 → 5.0 kg, instance 1 → 10.0 kg
  damping: 2.0          # scalar → same value for all instances
```

The list length must equal `n_instances`.

---

## `type: "rl"` — RL agent federate

The RL federate is usually **injected at runtime by ScenarioManager** and does not need to be written manually in the YAML. However, if you need to override its configuration:

```yaml
federate_configs:
  rl_agent:
    type: "rl"
    timing_configs:
      real_period: 60
    connections:
      publishes: []
      subscribes: []
    # model_configs is optional for type "rl"
    controlled_models: {}         # optional — maps model key to model name
    observed_models: {}           # optional
    additional_observed_models: {} # optional
```

For `type: "rl"`, `model_configs` is optional (unlike `type: "base"` where it is required).

---

## `streaming` (opt-in outbound MQTT mirror, all types)

Mirrors this federate's inputs/outputs to MQTT each step, alongside normal HELICS traffic — for live dashboards/observers. Does not change the co-simulation itself.

```yaml
federate_configs:
  spring_federate:
    type: "base"
    streaming:
      stream: true                # default: false (opt-in)
      # stream_topic_prefix: cosim/${sim_id}/spring   # default: cosim/<sim_id>/<federate_name>
      # every_n_ticks: 1
    ...
```

Requires Mosquitto running (`docker compose -f src/docker-compose.yaml up -d`). See [Digital-Twin Interfaces & Live Streaming](../digital_twin_interfaces.md).

---

## `type: "interface"` — interface federate (digital-twin bridge)

An interface federate has no physics model — instead of `model_configs`, it declares `interface_config`, and relays its wired HELICS connections to/from an external adapter (MQTT by default).

```yaml
federate_configs:
  dt_bridge:
    type: "interface"
    timing_configs:
      real_period: 1
      rt_lag: 1.0
      rt_lead: 1.0
    flags:
      realtime: true        # wall-clock pacing so an external process has time to react

    interface_config:
      adapter:
        name: mqtt_adapter                # catalog key (interface_adapter category)
        params: { host: localhost, port: 11883, qos: 0, client_id: cosim_dt }

      streams:                            # co-sim -> external (HELICS subscribe, MQTT publish)
        - helics_key: plant.spring_federate.0/position
          topic: cosim/${sim_id}/spring/position
          every_n_ticks: 1

      bridges:                            # external -> co-sim, or override registry
        - helics_key: plant.spring_federate.0/force
          topic: cosim/${sim_id}/sensor/force
          bounds: [-10, 10]
          scope: input          # input | output | param
          mode: replace          # replace external value | passthrough (real source + override)
          # source_key: plant.driver.0/force   # required for mode: passthrough, scope: input only
```

| Field | Meaning |
|---|---|
| `adapter.name` | Catalog key resolved from the `interface_adapter` category (e.g. `mqtt_adapter`), dynamic-imported like a physics model. |
| `streams[].helics_key` | A HELICS key this federate subscribes to; its value is relayed out to `topic`. |
| `bridges[].scope: input` | Registers a normal HELICS global publication at `helics_key`. `mode: replace` publishes only once an external value arrives; `mode: passthrough` (needs `source_key`) relays a real HELICS source until an external value shows up. |
| `bridges[].scope: output` \| `param` | No HELICS registration — the target federate already computes this value/parameter. Instead writes the bounds-clipped external value into a Redis-backed override registry; the target opts in with `override_enabled: true`. |
| `bridges[].bounds` | `[min, max]` clip applied to any external value before use. |

Because a physics-model federate and an interface federate register identical HELICS key names, swapping simulated hardware for real hardware ("config-only sim-to-real") is a change to **one federate's block** — every subscriber is untouched. See the worked example (`m5_bk4_demo_a_full_sim.yaml` / `m5_bk4_demo_b_digital_twin.yaml`) and full reference in [Digital-Twin Interfaces & Live Streaming](../digital_twin_interfaces.md).

---

## `memory_config` (per-federate override)

Each federate inherits `memory_config` from the scenario level. Override it for a specific federate:

```yaml
federate_configs:
  verbose_federate:
    type: "base"
    memory_config:
      batch_size: 500
      attrs:
        - "position"
        - "velocity"
      sink: parquet   # json (default) | parquet | none — see general.md
    ...
```

`sink: parquet` is useful here to isolate a single high-frequency federate onto the non-blocking writer while others keep the simpler `json` default. Not supported for `type: "rl"` federates yet. See [General Scenario Configuration](general.md#memory_config) for full `sink` semantics.
