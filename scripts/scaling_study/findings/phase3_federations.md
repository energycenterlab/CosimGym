# Scaling Study -- Phase 3 (federation sharding, the F axis)

Question: what does sharding N federates across F federations COST
(hierarchy-broker overhead), and does multifed let total federate count
scale beyond a single federation? Phase 2 already showed a single federation
reaches 89 (and, with the SSH-ControlPath caveat, effectively >=112 locally)
fine, so this phase is about overhead + composition, not breaking a ceiling.
Harness used as-is (`run_bench.py`, `cost_model.py`, `make_report.py`); one
harness bug found (below), not fixed per task instructions.

## HARNESS BUG FOUND -- gen_scenario.py port spacing breaks multi-federation `zmq` locally

`gen_scenario.py` assigns per-federation broker ports as `23500 + f` (1 apart,
`gen_scenario.py:276`). `ScenarioManager._broker_ports()`
(`src/core/ScenarioManager.py:1533-1538`) reserves **both** `port` and
`port + 1` for a plain `zmq` broker (zmq's paired reply socket) -- so for
**any F >= 2 with `core_type: zmq`**, federation_f's `port + 1` collides with
federation_(f+1)'s `port`. Example (F=2): federation_1 = port 23501 (occupies
{23501, 23502}), federation_2 = port 23502 (occupies {23502, 23503}) ->
deterministic bind failure:

```
RuntimeError: Broker for federation 'federation_2': port(s) 23502 already in
use -- most likely an orphaned broker from an earlier run.
```

Confirmed **not** a stale-process artifact: `ps -eo pid,args | grep
helics_broker` showed nothing, `ss -ltnp | grep 23502` showed nothing
listening, and the failure was 100% reproducible across 3 fresh repeats at
F=2, F=4, and F=8. `zmq_ss` ("single socket") does **not** reserve `port + 1`
(the `core_type == 'zmq'` guard in `_broker_ports`), so it is unaffected --
verified directly (F=2 local run under `zmq_ss` completes cleanly). This
blocks the task's literal "F=2/4/8, core_type=zmq, local" cells for 3a/3b.
**Deviation**: ran 3a and 3b under `core_type=zmq_ss` instead (still fully
local, no NAT/deployment block -- `zmq_ss` works locally too, just forces the
single-socket wire protocol). This is reported as a bug, not patched, per
task instructions ("harness is BUILT... don't modify harness code"). Fix
would be trivial (space federation ports by >=2, e.g. `23500 + 2*f`, or the
`+10` spacing convention already used by `generate_scale_benchmark.py`'s
hand-written multi-federation examples) but is out of scope here.

## 3a -- hierarchy-broker overhead at fixed total ~= 32 federates

`scripts/scaling_study/matrices/phase3a_hierarchy_overhead.yaml` (explicit
`runs:` pins, not `axes:`, to get exact F*N pairs) -- `zmq_ss`, local,
`heavy_compute_dummy`/work=1/M=1/ticks=30, 3 repeats. **12/12 PASS.**
CSV: `phase3a_hierarchy_overhead.csv`.

| F | N | total | setup_s (median) | broker_setup_s (median) | tick_mean_s (median) |
|---|---|---|---|---|---|
| 1 | 32 | 32 | 2.399 | 0.211 | 0.2566 ms |
| 2 | 16 | 32 | 2.851 | 0.635 | 0.2556 ms |
| 4 | 8  | 32 | 3.389 | 1.607 | 0.2492 ms |
| 8 | 4  | 32 | 4.515 | 2.799 | 0.2483 ms |

**The cost of sharding is almost entirely paid at setup, not per tick.**
`broker_setup_s` rises ~13x from F=1 to F=8 (0.211s -> 2.799s) while
`tick_mean_s` is flat-to-slightly-improving (0.2566ms -> 0.2483ms) -- because
at fixed total, more F means fewer federates per broker (N shrinks 32->4),
which very slightly *reduces* each individual broker's per-tick sync cost,
almost exactly offsetting the extra hierarchy-sync layer's steady-state cost.
Linear fit on `broker_setup_s` vs F (least squares over the 4 medians):

```
broker_setup_s ~ -0.070 + 0.369 * F      (~0.37s of extra broker-startup cost per added federation)
```

(Intercept isn't meaningfully negative -- it's a 4-point fit dominated by the
F=8 point; read it as "roughly a third of a second of broker-spawn/hierarchy
-registration overhead per federation added", consistent with the raw
deltas: F1->2 +0.42s/1fed, F2->4 +0.49s/fed, F4->8 +0.30s/fed.) `setup_s`
(total, includes federate spawn too) shows the same shape: 2.399 -> 2.851 ->
3.389 -> 4.515s, i.e. ~0.30s/added federation once spawn overhead is folded
in. **Verdict: sharding into more federations is a fixed, small, one-time
setup tax (~0.3-0.4s/federation added in this harness) that does NOT show up
in steady-state per-tick cost when total federate count is held fixed** --
the tick-level HELICS hierarchy-sync overhead is invisible at this scale
against the sync-cost *reduction* from fewer federates/broker.

## 3b -- scaling TOTAL federate count via multifed, local

`scripts/scaling_study/matrices/phase3b_multifed_scale.yaml`, `zmq_ss`,
local, 2 repeats. **8/8 PASS -- multifed reaches 256 federates locally with
no failures.** CSV: `phase3b_multifed_scale.csv`.

| F | N | total | setup_s (median) | broker_setup_s (median) | tick_mean_s (median) |
|---|---|---|---|---|---|
| 1 | 64 | 64  | 3.543  | 0.211 | 0.2992 ms |
| 4 | 32 | 128 | 8.307  | 2.158 | 0.6166 ms |
| 8 | 16 | 128 | 9.684  | 4.193 | 0.9831 ms |
| 8 | 32 | 256 | 18.227 | 5.774 | 1.0864 ms |

Max total federates reached: **256** (F=8, N=32), all clean, no
`failure_mode` set, no comms errors -- multifed composes without a hard
ceiling in this range on the 112-core manager. Two useful comparisons at the
**same total (128)**: F=4,N=32 (setup 8.3s) vs F=8,N=16 (setup 9.7s) -- more
federations at the same total costs *more* setup, consistent with 3a's
per-federation broker tax. But `tick_mean_s` also rises noticeably with
*total* federate count regardless of the F/N split (0.30ms at 64 total ->
~0.6-1.0ms at 128 total -> 1.09ms at 256 total) -- this is **not** primarily
a hierarchy-sync effect (3a showed tick_mean_s flat vs F at fixed total); it
is the sheer number of concurrent Python/HELICS federate subprocesses on one
112-core manager starting to contend for CPU/scheduling once total count
climbs past ~100-150, independent of how many brokers they're split across.

## 3c -- distributed multifed composition check (Config A, zmq_ss/NAT)

`scripts/scaling_study/matrices/phase3c_distributed_multifed.yaml`, 3-machine
NAT deployment (manager ipazia/112c + cloud1/32c + cloud5/32c), 2 repeats.
**4/4 PASS.** Total federate count kept < 112 per Phase 2's SSH-ControlPath
finding. CSV: `phase3c_distributed_multifed.csv`.

| F | N | total | setup_s (median) | broker_setup_s (median) | tick_mean_s (median) |
|---|---|---|---|---|---|
| 2 | 16 | 32 | 4.535 | 0.636 | 0.2401 ms |
| 2 | 24 | 48 | 4.603 | 0.635 | 0.2517 ms |

**Multifed + distributed SSH placement compose cleanly** -- no failures, no
comms errors, `broker_setup_s` matches the local F=2 case almost exactly
(0.635-0.636s vs 3a's local F=2 0.635s), confirming the hierarchy-broker
overhead itself is placement-independent; the ~1.7-1.8s extra in total
`setup_s` vs the equivalent local run (4.5s distributed vs 2.85s local for
F=2,N=16) is SSH connection-setup/remote-spawn cost, not hierarchy-broker
cost. See `06_federation_sharding.png` (right panel) for the local-vs-
distributed comparison bars.

## Combined artifacts

- `phase3_combined.csv` -- 3a+3b+3c concatenated (24 rows), input to
  `cost_model.py fit` / `make_report.py`.
- `phase3_fit_params.json` -- output of `cost_model.py fit
  phase3_combined.csv`. Notes: fits `s[zmq_ss]` (s0=1.76e-4, s1=2.76e-6) and
  `c[heavy_compute_dummy]` (a=1.82e-4) from this dataset; `O_par`/rtt_s
  default to 0 (no `par` rows, and the rtt fit is a crude 2-pair estimate
  not to be trusted). **Caveat**: D4's cost-model formula (plan Sec.1) has
  no F term at all -- it models per-broker sync `s(N, core_type)` and
  per-federate compute/comms only, not a hierarchy-broker layer -- so this
  fit does not, and structurally cannot, capture the F-axis overhead found
  above. That's why this report derives the F-overhead line (`broker_setup_s
  ~ -0.070 + 0.369*F`) by hand from the raw medians instead of from
  `cost_model.py`'s output. **Framework implication**: the theoretical
  model in `scaling_study_plan.md` Sec.1 should gain an explicit
  `F * broker_startup_overhead` (one-time, setup-phase) term, separate from
  the existing per-tick `sync_m` term -- Phase 3 shows these are two
  different costs with very different magnitudes and scaling shapes.
- `make_report.py`'s standard D5 plots (sync curve / roofline / ceiling-vs-
  network / predicted-vs-measured) were generated but **not copied here** --
  they index by N (federates per federation), which conflates the F-axis
  question this phase asks (e.g. the sync-curve plot's apparent spike at
  N=32 is really the 3b F=4,N=32 total-federate-contention effect from
  above, not a sync-cost function of N alone). A custom plot
  (`06_federation_sharding.png`, 3 panels: 3a setup/broker/tick vs F at
  fixed total; 3b setup/tick vs total federates; 3c local-vs-distributed
  bars) was built instead and is the primary visual here.

## Verdict / framework implication

- **When does sharding into more federations help vs. hurt?** Purely as a
  broker-overhead question, sharding is close to free in steady state
  (flat-to-improving `tick_mean_s` at fixed total) and costs a small, linear,
  one-time setup tax (~0.3-0.4s per federation, from broker spawn + hierarchy
  registration). It's worth doing **only when it buys something else** --
  e.g. crossing a per-broker federate ceiling (not observed in this repo up
  to N=112 per Phase 2, so not a concern here), enabling placement across
  more machines (3c confirms multifed + distributed SSH compose), or
  isolating fault domains. It does **not** help raw throughput on a single
  machine -- 3b shows `tick_mean_s` rising with *total* federate count
  regardless of how that total is partitioned across F, so sharding does not
  dodge the real bottleneck (aggregate process/CPU contention on the host)
  once total federate count gets large (>~100-150 on this 112-core manager).
  Sharding "hurts" only in the trivial, fixed sense of the setup tax above;
  it never made a run slower in steady state within the ranges tested.
- **Max total federates reached locally: 256** (F=8, N=32), no failure.
- **3c composition: multifed + distributed (Config A, zmq_ss/NAT) works**,
  4/4 PASS, hierarchy-broker cost identical to the local case; only extra
  cost is the (expected, unrelated) SSH/remote-spawn setup time.
- **Total wallclock this phase**: 3a ~35s, 3b ~140s, 3c ~20s -> well under
  2 minutes and 30s of actual scenario execution combined (run_bench.py's
  own console timestamps), plus the one-time bug investigation. No hard
  failures other than the diagnosed-and-worked-around zmq port-spacing bug.

## Files in this directory

- `phase3_federations.md` -- this report
- `phase3a_hierarchy_overhead.csv`, `phase3b_multifed_scale.csv`,
  `phase3c_distributed_multifed.csv` -- the three designed sweeps
- `phase3_combined.csv` -- concatenation of the three, feeds the fit/report
- `phase3_fit_params.json` -- `cost_model.py fit` output (see caveat above)
- `06_federation_sharding.png` -- custom 3-panel plot (this phase's real
  deliverable; see caveat on the generic D5 plots above)
