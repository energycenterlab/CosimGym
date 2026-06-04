# Synchronization & Causality

CosimGym provides two independent synchronization mechanisms that together ensure federates exchange data in the correct order at every simulation step and start from a consistent state.

---

## The problem: ordering in a co-simulation

When multiple federates advance time simultaneously, the HELICS broker grants them time as soon as they request it. Without explicit ordering, a federate could step and call its model with stale or empty inputs because its upstream neighbor hasn't published yet. Two problems can arise:

1. **Sequential data chain** — A feeds B feeds C. If all three request time at once, B may step before A publishes.
2. **Algebraic loop** — A subscribes to B and B subscribes to A. Both need each other's output at the same logical step: an unresolvable circular dependency.

These two problems require different solutions.

---

## Mechanism 1 — Automatic time-offset sequencing (`auto_offset`)

This is the primary mechanism for correct ordering. It runs **before any federate process is spawned**, as part of `ScenarioManager._scenario_setup_timing_vars()`.

### How it works

**Step 1 — Build the dependency graph**

`_build_federate_dependency_graph()` reads every federate's `subscribes` list and constructs a directed graph where an edge `A → B` means "B subscribes to A's output". Subscriptions marked `causality: "next_step"` are **excluded** — those edges are already broken by design (see below).

**Step 2 — Detect Strongly Connected Components (Tarjan's algorithm)**

`_compute_sccs()` finds groups of federates with mutual `same_step` dependencies. A group of more than one node is an algebraic loop that cannot be resolved by offsets. If `validate_causality_cycles: true` (the default), a `RuntimeError` is raised at launch describing the cycle and instructing you to break it with `causality: "next_step"`.

**Step 3 — Topological sort**

Once cycles are confirmed absent, Kahn's algorithm assigns a **stage** to each federate (or SCC):

- Stage 0 — no upstream dependencies (sources: weather readers, CSV loaders)
- Stage 1 — depends on stage-0 output
- Stage 2 — depends on stage-1 output
- …

**Step 4 — Assign fractional HELICS time offsets**

```
time_offset = stage × offset_step     (default offset_step = 0.1)
```

Stage 0 → offset `0.0`, stage 1 → offset `0.1`, stage 2 → offset `0.2`, etc.

These values are written into each federate's `timing_configs.time_offset` before the config is serialized to Redis. HELICS then enforces them: a federate with offset `0.1` is only granted time `t=1.1` after the stage-0 federate has already been granted and completed `t=1.0`. This correct sequencing is maintained at **every tick** for the entire simulation, without any manual intervention.

One safety guard applies: if `offset_step × max_stage ≥ 1.0`, the offsets would span a full time tick (illegal in HELICS). The engine auto-clamps: `offset_step = 0.9 / max_stage`.

### YAML configuration

```yaml
synchronization:
  auto_offset:
    enabled: true              # default: true
    offset_step: 0.1           # fractional HELICS time units per dependency stage
    override_existing_offsets: false  # if true, overwrites explicit time_offset values in federate configs
  validate_causality_cycles: true    # raises RuntimeError if same_step cycles are detected
```

To disable auto-offset and manage offsets manually, set `enabled: false` and specify `time_offset` directly in each federate's `timing_configs`.

---

## Mechanism 2 — Startup input synchronization (`startup_sync`)

This is a one-shot check that runs inside each federate **once**, after all federates have entered executing mode (the HELICS collective barrier) and after each federate has published its `init_state` outputs. Its purpose is narrow: confirm that the federate's input buffer is populated and valid before the first `_step()` call.

It does **not** use time offsets, does not retry, and has no effect on the ordering of subsequent steps. Those are handled entirely by the offset mechanism above.

### Execution order at startup

```
helicsFederateEnterExecutingMode()   ← HELICS barrier: all federates block here until all are ready
_publish_init_state()                ← every federate pushes its declared init_state onto the bus
_enforce_startup_input_sync()        ← one-shot input validity check
                                        ↓
                                     while ts < stop_time:   ← normal simulation loop begins
```

The barrier + `_publish_init_state()` sequence is what makes the check viable: by the time any federate calls `_enforce_startup_input_sync`, all upstream federates have already published their initial values.

### What it checks

`_enforce_startup_input_sync()` calls `_receive_inputs(force_read_all=True)` — bypassing the HELICS `is_updated` flag, since at t=0 that flag may not be propagated yet — then runs two independent checks:

**Missing inputs** — are all required input variable names present in the input buffer?

"Required" is determined in priority order:
1. `startup_sync.required_inputs` explicit list (if set in YAML)
2. The keys declared in `model.state.inputs` (the model's own schema)
3. Fallback: names derived from subscription topic strings

**Invalid inputs** — for present inputs, are the values usable?

| Check | Config flag | What it catches |
|---|---|---|
| Updated flag | `require_updated_inputs: true` | HELICS never flagged the value as updated (stale default) |
| Finite numeric | `require_finite_numeric: true` | Value is `nan`, `inf`, or `-inf` |
| Sentinel value | `invalid_numeric_sentinels: [-1.0e49]` | Value matches a known HELICS uninitialized default |

### Policy enforcement

Each failure type has an independent policy:

- `"error"` — raises `RuntimeError`, kills the federate process
- `"warn"` — logs a warning and continues into the simulation loop
- `"ignore"` — logs at INFO level and continues

### YAML configuration

Scenario-level defaults apply to all federates that don't have their own `startup_sync` block:

```yaml
synchronization:
  default_startup_sync:
    enabled: true
    force_read_all_subscriptions: true
    require_updated_inputs: true
    require_finite_numeric: true
    invalid_numeric_sentinels: [-1.0e49]
    missing_inputs_policy: "warn"   # "error" | "warn" | "ignore"
    invalid_inputs_policy: "warn"
```

Per-federate override (inside any `federate_configs` entry):

```yaml
federate_configs:
  my_federate:
    startup_sync:
      enabled: false    # disable entirely for this federate
```

The RL agent federate (injected at runtime by `ScenarioManager`) automatically receives a `startup_sync` with `required_inputs` set to its full observation list, ensuring it never starts a training episode before all observation sources have published.

---

## Handling feedback loops: `causality`

A `causality` flag on a subscription controls whether its value is applied at the current tick or deferred one tick. This is the mechanism used to break algebraic loops.

```yaml
subscribes:
  - key: "SOC"
    type: "double"
    targets: ['battery_federate.0/SOC']
    causality: "next_step"   # value received at tick t is applied at tick t+1
```

| Value | Behaviour |
|---|---|
| `same_step` (default) | Value is applied immediately; creates a `same_step` dependency edge used by auto-offset |
| `next_step` | Value is deferred; edge is excluded from the dependency graph; breaks cycles |

A scenario with a `same_step` cycle — for example, a controller reading a model output and the model reading the controller action at the same tick — will be caught by `validate_causality_cycles` at launch. The fix is to mark one of the two subscriptions as `causality: "next_step"`, accepting a one-tick lag on that signal.

---

## Summary

| Problem | Solved by | Runs when |
|---|---|---|
| Sequential execution order each tick | `auto_offset` (HELICS fractional time offsets via topological sort) | Baked into config before launch; active every tick |
| Algebraic loops (A→B→A, same step) | `validate_causality_cycles` + `causality: "next_step"` | Detected at launch; user breaks the cycle manually |
| Valid initial inputs before step 0 | `startup_sync` | Once per federate, before the simulation loop |
