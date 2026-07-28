# Scaling Study — Synthesis (Phases 0–5)

> **Canonical index is `README.md`** (trusted primitives, status, gaps, Phase 1b).
> This file is the narrative roll-up of Phases 0–5; if it disagrees with README,
> README wins. Not updated for Phase 1b (instance-count crossover) — see README.
>
> **⚠ Written before Phase D (data exchange).** Every conclusion below was measured
> with **zero subscriptions**, so it describes compute + sync only. Phase D
> (`phaseD_exchange.md`) shows coupling is the *dominant* per-tick cost once a
> federation is realistically wired — at 65 536 edges it is ~88% of tick time. Read
> the "best config" advice below as necessary but not sufficient, and see
> `phaseD_exchange.md` §8 for what changes.

One page tying `phase01.md`, `phase2_ceiling.md`, `phase3_federations.md`,
`phase4_distribution.md`, `phase5_validation.md` together. Read those for
detail and evidence; this is the "so what" for the two questions the plan
(`docs/future_and_TODOs/scaling_study_plan.md`) set out to answer.

## 1. What is the best config for a scenario? (per the framework)

Given a target: total model instances, the model's per-instance `work`,
and a machine set (`machines.json`):

1. **Instances-per-federate (M) vs. sequential/parallel**: the crossover
   law `(M − ⌈M/W⌉)·c(work) > O_par` (`O_par ≈ 0.039–0.044 s/tick`,
   Phase 1/5) decides seq vs. par. For `heavy_compute_dummy`
   (`c = 1.51e-5 + 1.34e-7·work`), the M=16/W=8 crossover measured at
   work ≈ 24,000 (three independent estimates agreed to ~4%, Phase 1).
   **Rule of thumb: parallel workers only pay off once each instance's
   step is expensive enough that skipping most of them sequentially saves
   more than the fixed per-tick worker-dispatch overhead** — for cheap
   models (or trivial `work`), sequential is *always* better; don't turn
   on `parallel_execution` by default.
2. **Federations (F)**: sharding does **not** raise single-machine
   throughput — Phase 3 showed `tick_mean_s` flat-to-improving vs. F at
   fixed total federate count. Only shard when you need to (a) place
   federates across more than one machine, (b) isolate fault domains, or
   (c) cross a genuine per-broker federate ceiling (none was found up to
   N=200 in this repo, see §2). Sharding costs a small, linear, one-time
   **setup** tax (~0.3–0.4 s per federation added, from broker spawn +
   hierarchy registration) — never a per-tick cost.
3. **Placement (local vs. distributed)**: only helps once compute
   genuinely dominates sync **and** the local machine's own cores would
   otherwise be the bottleneck. The idealized ceiling is
   `Σcores / local_cores` (1.571× for manager 112c + cloud1/cloud5 32c
   each). Phase 4 measured distribution beating even that idealized
   ceiling (ratio 0.548 at N=88) — but traced this to the shared
   manager's **background co-user load** (load avg ~67 at measurement
   time), not a real compute effect; on a genuinely idle manager, expect
   to approach but not beat 1.571×, and to see **no** benefit (ratio ≈1)
   at low/trivial work. **On this shared machine, "distribute" is a
   reasonable default hedge against unpredictable co-user load even when
   the workload alone wouldn't justify it** — but that is an artifact of
   this specific infrastructure being shared, not a property of the
   framework.
4. **`cost_model.py recommend`**: works and is a fast way to get a
   starting config (Phase 5, §A.4: 1,504 candidates evaluated in under a
   second), but two caveats found in Phase 5: (a) its absolute predicted
   `T_sim` should be read as an **optimistic lower bound**, not a point
   estimate, once the recommended config combines nontrivial N with
   nontrivial per-instance work — measured error there was 2.3–3.3× under
   the true value in every such cell tested, because no training sweep
   ever varied N and work together (an interaction term the additive
   `compute + sync` model structurally lacks); (b) it can prefer an
   **unmeasured** `core_type` (e.g. `tcp`) purely because the params file
   has no fitted data for it (defaults sync cost to 0) — always check
   `notes` in the params JSON for which core_types actually have fitted
   data before trusting a core_type recommendation. **Practical use: treat
   `recommend()`'s config choice (F/N/M/mode/W/placement) as trustworthy,
   its absolute T_sim number as a floor, and always smoke-test the
   recommended config with a short real run (ticks≈30) before committing
   to a long one** — which is exactly what Phase 5 did, and in both tested
   cases the recommended config comfortably met its tick budget even with
   the ~2.5× prediction gap, because the budget had enough margin.

## 2. Max-scale story

- **Superseded headline (Phase D, 2026-07-28): 32 768 model instances in 8
  federates, WITH data exchange on (65 536 edges), no failure** — the stress
  ladder exhausted its rungs with 97% of host RAM still free, at 329 ms/tick and
  3.6 GB. That is 20× the 1 600-instance figure quoted below, which was measured
  with no subscriptions. **The instance axis is not a wall**: cost is exactly
  linear in M (`≈0.080 ms · M`) and memory is near-flat per instance, because all
  M instances share one federate process. See `phaseD_exchange.md` §8.1.

- **The ~33-federate zmq_ss ceiling documented in
  `generate_scale_benchmark.py` does not reproduce today, at any N tested
  (up to 200)** — Phase 2 found it passes cleanly through N=89 with the
  calibrated harness and N=65 (720 ticks) with the original real-traffic
  topology; the original ~33 ceiling is most likely a **transient LAN
  condition** at the time it was measured, not an intrinsic zmq_ss/HELICS
  limit (git history for that file independently notes "non-deterministic
  behaviour to be checked").
- **A real, deterministic bug was found and has since been fixed**: an SSH
  `ControlPath` built under a long `logs/<scenario_name>/<timestamp>/...`
  tree crossed the AF_UNIX 108-byte `sun_path` limit once N reached ~112
  (one extra digit in the scenario name's embedded federate count was
  enough to tip it over). Fix (confirmed present in `ScenarioManager.py`):
  build `ControlPath` under a short, hashed `tempfile.gettempdir()`-rooted
  path instead. **Verified fixed in Phase 5: N=200 distributed_nat now
  passes 6/6, zero SSH failures**, where it previously failed 6/6.
- **A second bug (federation-count port collision) was also found and
  fixed**: `gen_scenario.py` originally spaced federation broker ports 1
  apart, colliding with plain `zmq`'s reserved `port+1` reply socket for
  any F≥2. Fix (confirmed present, `gen_scenario.py:282`): stride
  federation ports by 10.
- **Max stable scale actually reached in this repo, today**: **200 total
  federates** (distributed_nat, cores-proportional across manager 112c +
  cloud1 32c + cloud5 32c, Phase 5 §B.1) and, independently, **1,600 total
  model instances** (N=8 federates × M=200 instances/federate, local,
  Phase 5 §B.2) — both well inside a conservative safety budget (40% of
  free RAM, ≈61 GB) with 71%+ of that budget unused at the point tested.
  Neither number is "the ceiling" — both scans stopped at the task's
  explicitly bounded range, not at a failure. The theoretical
  memory-derived ceiling on the manager alone is ≈207 federates at the
  same 40%-of-free-RAM budget (≈300 MB/federate, Phase 0d); spreading the
  same relative budget across all three machines' own (larger, in
  aggregate) RAM would plausibly support several hundred to ~1,000+
  federates before any single machine's own budget bound — not validated,
  a back-of-envelope figure only.
- **This is a shared machine.** Every phase from 2 onward measured
  `uptime`/`free -g` immediately before scaling up, and Phase 4 in
  particular shows co-user background load can *dominate* a placement
  comparison's outcome. Any of these numbers (ceiling, roofline ratio,
  recommend() budget checks) should be re-validated at the time of a real
  deployment decision, not assumed frozen from this report.

## Where the framework should be extended next (not done here, out of scope
for a "run and interpret" phase)

1. An `N × c(work)` interaction term — no matrix in
   `scripts/scaling_study/matrices/` varies N and work together; Phase 5
   showed the current additive model under-predicts by 2.3–3.3× once both
   are simultaneously large.
2. An explicit `F × broker_startup_overhead` one-time setup term (Phase 3
   already derived it by hand: `broker_setup_s ≈ -0.070 + 0.369·F`) — the
   plan's §1 formula has no F term at all today.
3. Fitted `s(tcp)`/`s(tcp_ss)` data, so `recommend()` stops defaulting
   unmeasured core types to zero cost and picking them by data-gap
   artifact rather than genuine measurement.
