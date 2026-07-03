# Speculative Plan — Iterative / Algebraic-Loop Federate (Point 5)

Status: **speculative design sketch.** This is intentionally separate from the main
synchronization plan (`synchronization_causality_plan.md`) because it changes the
HELICS execution model, not just config wiring. Do not start this before points 1–4
are stable.

---

## 1. Motivation

`next_step` relaxation (main plan §3–4) breaks a same-step cycle by inserting a
**one-tick delay**. That is correct for MDP "action → next state" coupling, but it is
**wrong** for a true *algebraic loop*: two (or more) federates that must converge to a
mutually-consistent value **within the same instant** (no physical delay). Examples:

- Electrical network ↔ power-electronics controller solving for a consistent
  voltage/current at one timestamp.
- Two thermo-fluid federates sharing a pressure/flow boundary that must balance now.
- Any co-simulation where a one-step lag introduces non-physical oscillation or
  instability.

HELICS supports this natively via **iterative time requests**
(`helicsFederateRequestTimeIterative`): federates re-evaluate at the *same* time and
exchange values repeatedly until the loop **converges**, then the broker grants the
time and everyone advances. We currently never use it (`BaseFederate.request_time_advance`
calls plain `helicsFederateRequestTime`, `:488`).

## 2. Goal

Allow a **strongly-connected component** of federates to run as an *iterated
algebraic loop* at each tick, **seamlessly**:

- Detect automatically when iteration is required (SCC that is *not* relaxed and is
  marked iterative).
- Run the iterative convergence loop only for that SCC; the rest of the scenario keeps
  using plain time requests, unchanged.
- Detect convergence and **stop** iterating (max-iters guard + tolerance).
- No disruption to existing scenarios (feature is opt-in, default off).

## 3. Detection — when is iteration needed?

Add a declarative marker rather than guessing:

```yaml
synchronization:
  cycle_policy: iterate            # error | relax | iterate
  iteration:
    max_iterations: 20
    convergence:
      mode: relative               # absolute | relative
      tolerance: 1e-4
```

Resolution order for a same_step SCC (extends main-plan §4.3):

1. If any subscription in the SCC is pinned `same_step` **and** the SCC is marked
   `iterate` → treat the SCC as an **iterative group**.
2. Else if `cycle_policy: relax` → relax the back-edge (main plan §4.3).
3. Else → `error` (current behavior).

A per-federate or per-federation flag `iterative_group: <id>` can scope iteration to
exactly the SCC the author intends, so the system does not iterate accidentally. The
automatic part is: *given the SCC graph, the set of federates in one un-relaxed,
iterate-marked SCC forms one convergence group*. That mapping is computed from the
existing SCC machinery (`_compute_sccs`) — no new graph code.

## 4. Mechanism — HELICS iterative requests

### 4.1 New federate variant

Introduce `IterativeFederate(BaseFederate)` selected via federate `type: "iterative"`
(parallel to `base` / `rl`, dispatched in `federate_launcher.py`). It overrides the
time-advance + I/O portion of the loop:

```
def request_time_advance_iterative(self):
    target = self.time_granted + self.time_period
    it = 0
    while it < self.max_iterations:
        granted, iter_result = h.helicsFederateRequestTimeIterative(
            self.federate, target, h.helics_iteration_request_iterate_if_needed
        )
        self._receive_inputs(force_read_all=True)   # read peers' latest guesses
        self._iterate_models()                       # recompute with new boundary values
        self._publish_outputs()                      # publish refined guess
        if iter_result == h.helics_iteration_result_next_step:
            break                                    # broker says converged → time granted
        if self._converged():                        # local tolerance check
            # request a final non-iterative grant to advance
            ...
            break
        it += 1
    self.time_granted = granted
```

Key points:

- During iteration **time does not advance**; only the iteration index does. All
  members of the SCC re-exchange boundary values at the same timestamp.
- Members must publish **on every iteration** (so peers see refined guesses) and read
  with `force_read_all=True` (values may be "unchanged" by HELICS's update flag yet
  still part of the fixed-point search).
- Non-iterative federates outside the SCC are untouched: they call plain
  `helicsFederateRequestTime` and simply wait for the group to settle before the tick
  is granted globally.

### 4.2 Convergence / stop condition

Two independent stop guards (both required):

- **Tolerance**: track each boundary (loop) variable's value across iterations; stop
  when `max |x_k - x_{k-1}| < tol` (absolute) or normalized by `|x_k|` (relative).
  Convergence config from §3.
- **Max-iterations**: hard cap. On exceed → policy choice: `warn_and_advance`
  (accept last guess, log non-convergence) or `error`. Default `warn_and_advance` to
  avoid hanging a long study; surface a metric/counter.

Each iterative federate evaluates its own tolerance and signals
`helics_iteration_request_no_iteration` once locally satisfied; HELICS grants the step
only when **all** members signal convergence (`helics_iteration_result_next_step`).
This is the standard HELICS fixed-point handshake — convergence is a **group**
decision, not unilateral.

### 4.3 Model contract

Models inside an iterative group must be **re-evaluable at the same timestamp without
advancing internal state** — i.e. `step()` for iteration must be a pure boundary
solve, with state commit deferred to the granted step. Add an optional
`BaseModel.iterate(t, inputs)` hook distinct from `step(t, inputs)`:

- `iterate()` recomputes outputs from inputs **without** committing irreversible state.
- `step()` (called once the tick is granted) commits.

Models that cannot do this (e.g. the EnergyPlus FMU, which cannot re-enter a timestep)
**must not** be placed in an iterative group — validation should reject that
combination with a clear message (catalog flag `supports_iteration: false`).

## 5. Seamless integration constraints

- **Opt-in only.** `cycle_policy` default stays `error`; `type: iterative` is never
  injected automatically. Existing scenarios behave identically.
- **No change to non-iterative federates.** Plain federates keep
  `helicsFederateRequestTime`. HELICS handles mixed iterative/non-iterative cores at
  the broker level.
- **RL interaction.** An RL agent generally should *not* be inside an algebraic loop
  (its "action" is not part of a physics fixed point). Keep the agent outside the
  iterative SCC; the SCC converges within a tick, then the agent observes the
  converged result via normal same_step/next_step rules. Validation should warn if an
  `rl` federate lands in an `iterate`-marked SCC.
- **Catalog/validation gating.** New catalog field `supports_iteration`. New scenario
  validation: every member of an iterative group must be `type: iterative` (or `base`
  with an `iterate()`-capable model) and `supports_iteration: true`.

## 6. Risks / open questions

- **Convergence not guaranteed.** Fixed-point iteration can diverge or oscillate for
  stiff couplings; may need relaxation/damping (`x_{k+1} = αx̂ + (1-α)x_k`). Expose
  `relaxation_factor`.
- **Performance.** Each tick may cost up to `max_iterations` model evaluations for the
  whole group. Document the cost; the group should be as small as possible.
- **Determinism / logging.** Iteration must not pollute storage with intermediate
  guesses — only the **converged** values get recorded. `update_storage` must be
  called once, after convergence, not per iteration.
- **Reset / episode boundaries** (RL studies): iteration interplay with `_reset()` and
  `_clear_deferred_inputs()` needs care; simplest is to forbid iterative groups in
  resettable RL federations initially.
- **HELICS flags.** `uninterruptible`, `restrictive_time_policy`, and granted-time
  semantics for iterative cores need verification against the installed HELICS version
  before committing to the API surface above.

## 7. Suggested phasing

1. Spike: a 2-federate toy algebraic loop (e.g. two linear boundary equations) using
   `helicsFederateRequestTimeIterative` directly, outside the framework, to validate
   the HELICS handshake and convergence detection.
2. `IterativeFederate` + `BaseModel.iterate()` hook + `federate_launcher` dispatch.
3. `cycle_policy: iterate` detection wired to the existing SCC machinery; validation
   gating (`supports_iteration`, no-RL-in-loop, no-FMU-in-loop).
4. Convergence config (tolerance/max-iter/relaxation) + once-only storage.
5. Tests: convergence on a well-posed loop, max-iter cutoff on a divergent loop,
   mixed iterative + plain federates in one scenario, validation rejections.

This plan deliberately reuses the SCC detection from points 1–4 so the "which
federates iterate together" question is answered by the **same graph code** that
answers "which cycles must be relaxed" — the two policies (`relax`, `iterate`) are
just different resolutions of the same detected SCC.
