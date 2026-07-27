# Scaling Study — Phase 2 (federate sweep + zmq_ss ceiling investigation)

Marquee question: `src/scenarios/generate_scale_benchmark.py`'s module docstring
claims a hard ~33-federate `zmq_ss` ceiling on this LAN (33 OK, 49 flaky, 65
FAIL, all dying `[-101] lost comms` after a ~52s timeout; plain `zmq` scales to
89). **Verdict: that ceiling does NOT reproduce today, at any federate count
tested (up to 89 in the calibrated harness, up to 65 with the original
heavier real-model topology run at full duration). Every zmq_ss/NAT run in
this investigation succeeded.** A real, deterministic failure was found at
N≥112, but it is an unrelated SSH-ControlPath/AF_UNIX path-length bug, not a
HELICS/zmq_ss comms ceiling — see §4.

## 1. 2a — LOCAL zmq control (no NAT, single machine)

Matrix `scripts/scaling_study/matrices/phase2a_local_zmq.yaml`, CSV
`phase2a_local_zmq.csv`. `F=1, M=1, mode=seq, core_type=zmq, model=heavy_compute_dummy,
work=1, placement=local, ticks=30`, 3 repeats/cell. Total wall-clock ≈110s.

| N | rep0 | rep1 | rep2 |
|---|------|------|------|
| 17  | ok | ok | ok |
| 33  | ok | ok | ok |
| 49  | ok | ok | ok |
| 65  | ok | ok | ok |
| 89  | ok | ok | ok |
| 112 | ok | ok | ok |

**18/18 PASS.** Confirms the prior claim that plain local `zmq` has no
federate-count ceiling in this range — matches `generate_scale_benchmark.py`'s
docstring ("zmq 65/89 federates: OK OK OK").

## 2. 2b — Config A distributed zmq_ss (NAT, 3 machines) — THE test

Matrix `scripts/scaling_study/matrices/phase2b_distributed_nat_zmqss.yaml`, CSV
`phase2b_distributed_nat_zmqss.csv`. Same design as 2a but
`core_type=zmq_ss, placement=distributed_nat` (`gen_scenario.py` auto-forces
zmq_ss + the 3-machine NAT deployment: manager ipazia/112c, machine_a
cloud1/32c, machine_b cloud5/32c). 3 repeats/cell. Total wall-clock ≈70s.

| N | rep0 | rep1 | rep2 |
|---|------|------|------|
| 17 | ok | ok | ok |
| 33 | ok | ok | ok |
| 49 | ok | ok | ok |
| 65 | ok | ok | ok |
| 89 | ok | ok | ok |

**15/15 PASS.** This directly contradicts the documented claim of "49: FAIL
FAIL OK, 65: FAIL". No `[-101] lost comms`, no timeout, no flakiness observed
anywhere in this sweep — every run completed in a few seconds (`setup_s`
3.5–5.3s, `sim_wall_s` <0.01s since HELICS ticks aren't wall-clock-gated and
the model is trivial).

`04_ceiling_vs_network.png` (copied here) plots both datasets side by side,
one panel per `core_type` — all points green (success), no failure marker in
either panel, confirming visually that neither `zmq` (local) nor `zmq_ss`
(distributed NAT) shows any onset of failure through N=112 / N=89
respectively.

## 3. Sanity check — does the ORIGINAL (heavier, real-subscription) topology reproduce the ceiling?

`gen_scenario.py`'s federates are deliberately self-contained (no
subscriptions at all — see its docstring), so 2a/2b measure pure
broker-registration + time-advance sync cost, not cross-federate message
routing. The *original* ceiling claim was measured on
`generate_scale_benchmark.py`'s district topology, which has real
weather→PV/building/heatpump→PID cross-federate pub/sub traffic (4 federates
per site + 1 shared weather federate). To rule out "the harness's
zero-message design is why the ceiling vanished," I regenerated and ran that
**original** topology directly (`generate_scale_benchmark.py --sites N
--sink none`), same 3-machine NAT zmq_ss deployment, matching its original
measurement protocol:

| federates | duration | result |
|---|---|---|
| 49 (12 sites) | 1 day (24 ticks) | **PASS** — completed in 6.37s (setup 2.99s, sim 6.30s) |
| 65 (16 sites) | 1 day (24 ticks) | **PASS** — completed in 7.20s (setup 2.82s, sim 7.13s) |
| 65 (16 sites) | 1 month (720 ticks, the exact protocol the original ceiling note used) | **PASS** — completed in 8.19s (setup 2.80s, sim 8.11s) |

All three completed cleanly, no comms errors, setup time (~2.8–3.0s) an order
of magnitude under the claimed ~52s failure threshold. **The ceiling does not
reproduce even with the original real-traffic topology at the original
duration protocol.** (The `rsync collect failed ... No such file or
directory` lines seen in these runs are an expected `--sink none` artifact —
no results dir is created on the remotes when there's nothing to write, so
the post-run rsync-back step has nothing to collect; unrelated to comms.)

## 4. Pushing further — where does something actually break?

To find any real ceiling in this environment (not just fail to reproduce the
old one), I extended 2b's trivial-model sweep to N ∈ {112, 150, 200}
(`phase2b_extended_high_n.csv`, 2 repeats/cell, distributed_nat/zmq_ss).

**6/6 FAIL — but a different, unrelated bug:**

```
RuntimeError: [machine_b] failed to open ssh control master to eclabuser@130.192.238.13:
unix_listener: path "logs/scaling_F1_N112_M1_seq_zmq_ss_distributed_nat/20260724_110317/
ssh_control/cm-machine_b.j8bodJjj9UEephgO" too long for Unix domain socket
```

(verbatim, un-truncated version, captured by re-running outside run_bench's
120-char `failure_mode` truncation)

**Root cause, confirmed precisely:** OpenSSH's `unix_listener()` binds the
`ControlPath` via an internal temp-file-then-rename step that appends a
random ~17-char suffix (`.j8bodJjj9UEephgO`) to the path before the atomic
rename. `AF_UNIX` `sun_path` is capped at 108 bytes on Linux. Our
`ControlPath` is built as a *relative* path
`logs/<scenario_name>/<run_timestamp>/ssh_control/cm-<alias>`
(`src/core/ScenarioManager.py::_open_remote_control_masters`,
`src/core/remote_executor.py::RemoteExecutor._control_path`), and
`gen_scenario.py`'s `scenario_name` embeds every knob including `N` itself
(e.g. `scaling_F1_N112_M1_seq_zmq_ss_distributed_nat`). Measured directly:

| N | literal ControlPath length | + ssh's ~17-char temp suffix | vs 108-byte limit |
|---|---|---|---|
| 89  | 90 chars  | 107 | **fits** |
| 112 | 91 chars  | 108 | **overflows by 1** |

The one extra digit going from N=89 to N=112 (`"89"` → `"112"`, +1 char) is
enough to cross the boundary, given how close the base path already sits to
the 108-byte limit. This is **not** a HELICS/zmq_ss ceiling, not network- or
NAT-related, and not federate-count-dependent in any architectural sense —
it is a path-length artifact of this harness's `logs/<long-descriptive-scenario-name>/<timestamp>/ssh_control/cm-<alias>`
directory layout, that happens to bite once the scenario name's embedded `N`
crosses into 3 digits at N≈100–112 in *this* repo's specific relative-path
depth. A real deployment with `host:`-tagged federates and a shorter
scenario name, or an absolute `logs` path elsewhere, would hit this at a
different N (or not at all locally, since the `local` placement never opens
SSH control masters).

**No tuning attempted on this bug** — it is out of scope for the zmq_ss
ceiling investigation (per task: "if tuning is out of scope/too deep, at
minimum pinpoint and quote the root-cause error and state the hypothesis").
It is a distinct, fixable harness bug (shorten `ControlPath` — e.g. hash the
scenario name, or use `/tmp` instead of the repo-relative `logs/` dir) worth
a follow-up ticket, but it must **not** be conflated with the zmq_ss ceiling
question this phase was scoped to answer.

## 5. Verdict

- **Is the ~33-federate zmq_ss ceiling real, today, on this LAN? No.**
  Every zmq_ss/distributed_nat run tested — 15/15 in the calibrated
  no-subscription sweep (N up to 89) and 3/3 additional sanity runs with the
  original real-cross-federate-traffic topology (N=49, 65, including the
  original's exact 720-tick duration) — passed cleanly, with setup times
  (~2.8–5.3s) an order of magnitude below the previously reported ~52s
  failure threshold. No `[-101] lost comms`, no flakiness, in any run.
- **What was the original ceiling, then?** The original note itself already
  called N=49 "flaky" (2 fails, 1 pass out of 3) rather than a hard,
  deterministic wall — and git history for this file
  (`e9cee8c "distributed test with 201 seems to work, there are some non
  deterministic behaviour to be checked"`) independently confirms this
  LAN's distributed zmq_ss behavior has previously been observed as
  non-deterministic/flaky rather than a fixed architectural limit. The most
  likely explanation is that the original ~33 ceiling was a **transient LAN
  condition** at measurement time (shared network, momentary congestion,
  or a since-resolved host/firewall hiccup) rather than an intrinsic
  zmq_ss/HELICS/NAT limit — it does not reproduce under the same topology,
  core_type, deployment, and (for the sanity checks) even the same duration
  protocol today.
- **Disentangling zmq_ss vs NAT vs multi-machine:** per the plan, Config B
  (non-NAT multi-machine plain zmq/tcp control) is deferred, so this data
  alone cannot cleanly separate "NAT" from "multi-machine" as a contributing
  factor in general — but that distinction is moot here since **no ceiling
  was observed for zmq_ss+NAT at all** in the tested range; there is nothing
  to disentangle a cause from.
- **A real ceiling does exist in this specific harness/repo layout**, but at
  N≳112 and it is an **SSH ControlPath / AF_UNIX 108-byte path-length bug**
  in `RemoteExecutor`/`ScenarioManager`'s log-directory-nested control-socket
  path, unrelated to HELICS, zmq_ss, or federate-count scaling per se — see
  §4 for the exact mechanism and numbers.

## Files in this directory

- `phase2_ceiling.md` — this report
- `phase2a_local_zmq.csv`, `phase2b_distributed_nat_zmqss.csv` — the two
  designed sweeps (§1, §2)
- `phase2b_extended_high_n.csv` — the N∈{112,150,200} extension that surfaced
  the SSH-ControlPath bug (§4)
- `phase2_combined.csv` — 2a+2b concatenated, input to `make_report.py` for
  the plot below
- `04_ceiling_vs_network.png` — success/failure vs N, one panel per
  `core_type` (zmq local vs zmq_ss distributed_nat); all points green
