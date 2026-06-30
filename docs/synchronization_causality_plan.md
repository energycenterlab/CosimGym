# Synchronization & Causality — Overview, Bug Fix, and Generalization Plan

Status: design doc. Covers points 1–4 of the request. Point 5 (algebraic-loop /
iterative federate) lives in a separate doc: `algebraic_loop_federate_plan.md`.

---

## 1. How synchronization works today (definitive overview)

### 1.1 Time model

- HELICS time is **unitless integer ticks**. There is no wall-clock in the solver.
- `ScenarioManager._scenario_setup_timing_vars()` (`src/core/ScenarioManager.py:1097`)
  computes `min_real_period = min(real_period over all federates)`. That minimum
  becomes **tick 1**.
- Each federate gets `time_period = int(real_period / min_real_period)`
  (`ScenarioManager.py:1118`). A federate with `real_period=600` and a scenario
  minimum of `600` advances `time_period = 1` tick per step. A slower federate
  (`real_period=1200`) advances `2` ticks per step.
- `n_steps = duration / real_period` → `time_stop` per federate.

### 1.2 Offsets (tie-breaking within a tick)

- `time_offset` (fractional tick, e.g. `0.1`) shifts a federate's first grant so
  two federates that would otherwise be granted the **same** tick get a
  deterministic order. Applied to HELICS via
  `helics_property_time_offset` (`BaseFederate.py:204`).
- `time_offset_explicit` (`config_dataclasses.py:66`) is set automatically by a
  model-validator when YAML specifies `time_offset` (`config_dataclasses.py:71`).
  It tells auto-offset "the user pinned this, don't overwrite unless
  `override_existing_offsets`".
- `synchronization.auto_offset` (`ScenarioManager._apply_auto_time_offsets()`,
  `:997`) topologically orders federates by their **same_step** dependency graph,
  assigns each a stage, and sets `time_offset = stage * offset_step`. This makes a
  producer get granted **before** its same-step consumer inside one tick, so the
  consumer reads a fresh value.

### 1.3 Causality (per-subscription)

Each `FedSubscription` carries `causality ∈ {same_step, next_step}`
(`config_dataclasses.py:161`).

- **`same_step`**: the received value is written straight into `self.inputs` and
  consumed by the model **this** step (`BaseFederate._receive_inputs`,
  `:700`). The producer must be granted earlier in the same tick (that is what
  offsets guarantee). Semantics: *instantaneous* coupling.
- **`next_step`**: the received value is staged into `self._deferred_inputs`
  (`:698`) and only promoted into `self.inputs` at the **start of the next step**
  via `_apply_deferred_inputs()` (`:664`), which runs *before* `_receive_inputs`
  in the loop (`BaseFederate.run`, `:420-421`). Semantics: *one-tick-delayed*
  coupling. This is what makes a feedback edge safe — it removes the same-tick
  ordering requirement.

### 1.4 Per-step loop order

`BaseFederate.run()` (`:406`):

```
while ts < stop:
    request_time_advance()      # ts += 1, grant next tick
    _apply_deferred_inputs()    # promote last step's next_step values
    _receive_inputs()           # read same_step now; stage new next_step
    model._step()
    _publish_outputs()
    _reset()                    # episode reset (train)
    update_storage()
```

`RLFederate.step()` (`RL_Federate.py:644`) reorders for the agent:

```
_action_to_publish(action)
_publish_outputs()              # publish action FIRST
request_time_advance()
_apply_deferred_inputs()
_receive_inputs()
obs = _inputs_to_observations(use_staged_next_step=True)  # consume next_step now
reward = compute_reward(obs, action)
```

The `use_staged_next_step=True` flag (`RL_Federate.py:627`) is important: for a
`next_step` observation, the agent uses the **just-read staged** value for the
transition rather than waiting an extra step. So the storage transition is aligned
even though the plant federate sees the value one tick later. (See §3.2.)

### 1.5 Cycle validation

- `_build_federate_dependency_graph(include_next_step=False)` (`:904`) builds a
  directed graph **producer → consumer** using **only same_step edges**
  (`next_step` edges are skipped, `:922`). Nodes are `(federation, federate)`.
- `_validate_causality_cycles()` (`:937`) runs Tarjan SCC (`_compute_sccs`,
  `:958`). Any SCC with `>1` node is a same-step cycle that non-iterative HELICS
  `requestTime` cannot resolve → **`RuntimeError`** (`:951`).
- The same graph drives `_apply_auto_time_offsets()`.

So `next_step` is the **declarative escape valve**: marking one edge of every cycle
`next_step` removes it from the graph and breaks the SCC.

### 1.6 Where causality is resolved (two independent code paths — the root of the bug)

| Path | Default when value absent |
| --- | --- |
| `ScenarioManager._normalize_subscription_causality` (`:888`) | scenario `default_subscription_causality`, else `same_step` |
| `BaseFederate._normalize_subscription_causality` (`:513`) | hard-coded `same_step` |
| Pydantic field `FedSubscription.causality` (`:161`) | `same_step` |

These three do **not** agree, and that is the bug.

---

## 2. The bug (why `default_subscription_causality: next_step` is ignored)

### 2.1 Mechanism

`FedSubscription.causality` has a Pydantic **field default of `"same_step"`**
(`config_dataclasses.py:161`). So a YAML subscription that omits `causality` does
**not** arrive as `None` — it arrives as the string `"same_step"`.

`ScenarioManager._normalize_subscription_causality(raw)` only substitutes the
scenario default when `raw` is falsy:

```python
causality = (raw_value or default_causality or "same_step").lower()
```

Since `raw_value == "same_step"` is truthy, the scenario
`default_subscription_causality: "next_step"` is **never applied**. The documented
"set a scenario-wide default" knob is dead for every subscription that doesn't
spell out its own causality.

### 2.2 The concrete cycle in `bui0_heatingpower_DQN.yaml`

- `_get_rl_pubsubs` retargets `building_federate`'s `OthEquRadWatt` subscription to
  read from the agent (`ScenarioManager.py:408`). That subscription declares no
  `causality` → Pydantic gives it `same_step`. Edge: **rl_agent → building**
  (same_step).
- The agent's observation `HeatingLoadTarget` is declared `causality: same_step`
  and is produced by `building_federate`. Edge: **building → rl_agent** (same_step).
- Two same_step edges in opposite directions ⇒ SCC `{building_federate,
  rl_agent}` ⇒

```
Detected same_step dependency cycles that cannot be resolved with non-iterative
HELICS time requests: [[('federation_1','building_federate'),('rl_federation','rl_agent')]].
```

### 2.3 Second, latent half of the bug (runtime vs validation disagreement)

Even if the ScenarioManager graph were fixed to honor the scenario default, nothing
**writes the resolved causality back** into the `FedSubscription`. The config is
serialized to Redis as-is (`_upload_config_on_redis`, `:637`), and at runtime
`BaseFederate._normalize_subscription_causality` (`:513`) applies its **own**
hard-coded `same_step` fallback. So validation could decide "next_step" while the
running federate behaves "same_step" — a silent divergence. Any fix must resolve
causality **once** and **persist it** so graph, offsets, and runtime all read the
same concrete value.

---

## 3. Bug fix

### 3.1 Immediate, declarative unblock (no code change)

Mark the loop's incoming-action edge `next_step` on the plant subscription:

```yaml
# building_federate.connections.subscribes, key OthEquRadWatt
- key: "OthEquRadWatt"
  type: "double"
  units: "W"
  causality: "next_step"
```

This removes the `rl_agent → building` edge from the same_step graph, breaks the
SCC, and gives exactly the sequentiality wanted: **agent acts at T → plant consumes
at T+1 → plant emits `HeatingLoadTarget` at T+1 → agent reads it same-step at T+1.**

### 3.2 Proper fix (code) — single causality resolution pass, persisted

Goal: make `default_subscription_causality` actually work, and guarantee
validation/offsets/runtime never disagree.

**Step A — make "unset" distinguishable.** Change the field to an explicit sentinel
plus an explicitness flag, mirroring the existing `time_offset` / `time_offset_explicit`
pattern:

```python
class FedSubscription(BaseModel):
    causality: Optional[str] = None            # None = "not specified by author"
    causality_explicit: bool = False           # set by validator below

    @model_validator(mode='before')
    @classmethod
    def _mark_causality_explicit(cls, data):
        if isinstance(data, dict) and data.get('causality') is not None:
            data['causality_explicit'] = True
        return data
```

(Mirror the same idea for `ObservationSpec.causality` only if you want the scenario
default to reach RL observations; otherwise leave RL obs explicit-by-convention.)

**Step B — one resolution pass that writes back.** Add
`ScenarioManager._resolve_subscription_causalities()` that runs **before**
`_validate_causality_cycles` and **before** `_upload_config_on_redis`. For every
subscription in every federation (including the injected `rl_federation`):

```python
sub.causality = self._normalize_subscription_causality(
    sub.causality if sub.causality_explicit else None
)
```

After this pass every `FedSubscription.causality` is a concrete `same_step` /
`next_step` string that already reflects the scenario default. Then:

- `_build_federate_dependency_graph` reads the concrete value (no behavior change).
- `BaseFederate` reads the concrete value from Redis. Its local fallback becomes
  dead code but harmless; optionally drop it.

**Step C — keep `_normalize_subscription_causality` as the single source of the
default** so there is one place that knows the precedence
`explicit > scenario default > same_step`.

This is the recommended fix: it is small, mirrors an existing pattern
(`time_offset_explicit`), and closes the validation/runtime gap in §2.3.

---

## 4. Generalized sequentiality plan (points 1–4)

### 4.1 Point 1 — let the co-simulation run while avoiding cycles

Two complementary mechanisms already exist; formalize them:

- **same_step edge** = instantaneous, ordered by offsets, must be acyclic.
- **next_step edge** = one-tick delay, removed from the cycle graph, always safe.

Rule: *the same_step subgraph must be a DAG*. Every cycle in the full coupling graph
must contain **at least one** `next_step` edge. Validation (§1.5) enforces it; auto
relaxation (§4.3) can satisfy it automatically.

### 4.2 Point 2 — split observations: pre-action state vs post-action consequence

This is a **causality + bookkeeping** distinction, already partly implemented:

- **Pre-action / state observations** (the state the policy conditions on): these are
  values that exist *before* the agent acts this tick. In MDP terms they are `s_t`.
  In the loop they are the values read at the start of the agent step (or carried as
  `obs_before_action`, `RL_Federate.py:679`).
- **Post-action / consequence observations** (the result of the action): values the
  plant produces *because of* the action, i.e. part of `s_{t+1}` and the reward
  signal. The loop variable `HeatingLoadTarget` is exactly this.

Mapping to the framework:

- A consequence variable that closes a loop back to the agent should be **same_step**
  *on the agent side* (agent wants the immediate result) **and** the action it
  depends on should be **next_step into the plant** (plant consumes the action next
  tick). That is the §3.1 wiring and it is the canonical "action affects next state"
  MDP shape.
- Storage already separates the two: `observations_before_action` and
  `observations_after_action` (`RL_Federate.py:286-287`, written in
  `update_storage`, `:295`). The transition recorded is
  `(obs_before, action, reward, obs_after)`.

Proposed declarative surface (additive, optional): give `ObservationSpec` an explicit
**`phase`** hint so intent is self-documenting and validation can check wiring:

```yaml
observations:
  federation_1.building_federate.0.TBuilding:
    role: state
    phase: pre_action      # part of s_t; default
    causality: next_step
  federation_1.building_federate.0.HeatingLoadTarget:
    role: state
    phase: post_action     # consequence of the action; closes the loop
    causality: same_step
```

`phase` would be advisory metadata + a validation cross-check
(`post_action` + `same_step` is the expected consequence pattern; `post_action`
that does **not** participate in a relaxed loop is suspicious and warns). It does
**not** change the runtime read path — `use_staged_next_step` already aligns the
transition. Implement `phase` only if the self-documentation/validation value is
wanted; the mechanics work without it.

### 4.3 Point 3 — auto-fix cycles by relaxing one link (the "last" / feedback edge)

Replace the hard `RuntimeError` with an opt-in **auto-relax** policy:

```yaml
synchronization:
  cycle_policy: relax        # one of: error (current default) | relax
```

Algorithm `_auto_relax_cycles()` (runs after resolution §3.2B, before validation):

1. Build same_step graph. Compute SCCs (reuse `_compute_sccs`).
2. For each SCC with `>1` node, choose **one edge to demote** to `next_step`:
   - Build a topo-ish order of the SCC nodes using the same deterministic key as
     `_apply_auto_time_offsets._cycle_sort_key` (`:1033`): explicit offset, then
     name. The intended forward chain is `n0 → n1 → ... → nk`.
   - The **back-edge** is the edge from the latest node to an earlier node
     (`nk → n_j`, `j ≤ k`) — i.e. the link that *closes* the loop. Demote that one.
     "Relax the last link" = relax the feedback edge of the chain, leaving the
     forward sequential chain intact.
   - Tie-break deterministically; prefer demoting an edge **into** the chain's
     source (the federate the user clearly wants to run first).
3. Write `sub.causality = "next_step"` back on the chosen subscription, log a
   `WARNING` naming the demoted edge, and recompute SCCs. Repeat until the same_step
   graph is a DAG (each iteration strictly removes ≥1 edge → terminates).
4. Re-run `_validate_causality_cycles` as a safety net (should now pass).

Author override: a per-subscription `relax_priority: low|normal|high` (or
`pin_same_step: true`) lets the user protect an edge that must stay instantaneous,
forcing the relaxer to pick a different edge in that SCC. If every edge of an SCC is
pinned `same_step`, fall back to `error` for that SCC with a clear message.

Default stays `error` so existing scenarios are unaffected; `relax` is opt-in.

### 4.4 Point 4 — works for arbitrary meshes / multiple federates / any wiring

The machinery above is already topology-agnostic because it is **graph-based**, not
pattern-based:

- Nodes = every `(federation, federate)` across every federation, including the
  runtime-injected `rl_federation` (`_build_federate_dependency_graph`, `:905`).
- Cross-federation targets resolve through `_resolve_target_federate_node` (`:857`),
  so meshes spanning federations are handled identically.
- SCC + condensation already supports **multiple independent cycles**, **nested
  chains**, and **diamonds**: `_apply_auto_time_offsets` condenses each SCC to a
  super-node and topologically stages the condensation (`:1013-1068`). Auto-relax
  reuses the same SCC machinery, so it scales to any number of cycles.
- For a **chain of N simulators with one feedback** (`s0→s1→...→sN→s0`), §4.3 demotes
  only the `sN→s0` feedback edge, preserving the N-long same-step pipeline and
  delaying just the loop closure by one tick — the desired behavior.

Validation/coverage to add:
- Unit tests over synthetic graphs: single 2-cycle, 3-chain+feedback, two disjoint
  cycles, diamond with one cycle, fully meshed triple. Assert (a) which edge gets
  relaxed, (b) resulting graph is a DAG, (c) offsets stage correctly.
- A test asserting `default_subscription_causality: next_step` actually flips an
  un-annotated subscription (regression for §2).
- A test asserting validation and `BaseFederate` runtime read the **same** resolved
  causality (regression for §2.3).

---

## 5. Implementation order (suggested)

1. **§3.2** causality resolution pass + `causality_explicit` (fixes the live bug,
   closes validation/runtime gap). Smallest, highest value.
2. Regression tests for §2 and §2.3.
3. **§4.3** opt-in `cycle_policy: relax` with deterministic back-edge selection +
   override knob.
4. Graph/mesh tests (§4.4).
5. **§4.2** optional `phase` metadata + advisory validation.
6. Point 5 (iterative/algebraic-loop federate) — see
   `algebraic_loop_federate_plan.md`. Independent, speculative.

## Key code references

| Concern | Location |
| --- | --- |
| Cycle error | `ScenarioManager.py:937` `_validate_causality_cycles` |
| Same_step graph | `ScenarioManager.py:904` `_build_federate_dependency_graph` |
| SCC (Tarjan) | `ScenarioManager.py:958` `_compute_sccs` |
| Auto offsets / staging | `ScenarioManager.py:997` `_apply_auto_time_offsets` |
| Causality default (SM) | `ScenarioManager.py:888` |
| RL sub/pub wiring | `ScenarioManager.py:381` `_get_rl_pubsubs` (retarget at `:408`) |
| next_step staging | `BaseFederate.py:664` `_apply_deferred_inputs`, `:679` `_receive_inputs` |
| Causality default (runtime) | `BaseFederate.py:513` |
| Field default `same_step` | `config_dataclasses.py:161` |
| `time_offset_explicit` pattern to mirror | `config_dataclasses.py:66,71` |
| Obs before/after storage | `RL_Federate.py:286,295,679` |
| next_step transition alignment | `RL_Federate.py:616` `_inputs_to_observations` |
