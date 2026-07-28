# Runbook — how to re-run or change these experiments

Every command below is copy-pasteable and was actually used to produce the results
in `findings/`. Run **all of them from the repository root**
(`/media/space/rando/CODE/CosimGym`), never from inside `scripts/`.

Background on what each experiment *means*: `EXPERIMENTS.md`.
Schemas (knob names, CSV columns): `CONTRACT.md`.

---

## 0. One-time setup per shell session

```bash
cd /media/space/rando/CODE/CosimGym

# Put the conda env's binaries on PATH. This matters for more than python:
# helics_broker lives here too, and scenarios fail instantly without it.
export PATH=/media/space/rando/Environments/cosim_gym/bin:$PATH
PY=/media/space/rando/Environments/cosim_gym/bin/python

# Redis / MQTT / MinIO must be up. Redis is mandatory: every federate reads its
# config from it. Safe to re-run if already running.
docker compose -f src/docker-compose.yaml up -d
```

> **Do not use `conda activate`** — it silently fails from scripts on this machine.
> `conda run -n cosim_gym python ...` works but has been killed early on long jobs,
> so the direct interpreter above is the reliable form.

### Verify the setup in 30 seconds

```bash
$PY -c "import helics; print('helics', helics.helicsGetVersion())"
which helics_broker
docker compose -f src/docker-compose.yaml ps | grep redis
```

---

## 1. Sanity check before any campaign

Always do this first. It catches a broken env in seconds instead of 40 minutes in.

```bash
# 1a. Generate one tiny scenario and check it parses.
$PY scripts/scaling_study/gen_scenario.py \
    --F 1 --N 2 --M 1 --model exchange_dummy \
    --exchange on --distance intra_fed --fanout 1to1 \
    --ticks 10 --out /tmp/smoke.yaml

# 1b. Run the hand-written end-to-end scenario (proves data really moves).
$PY -c "import sys; sys.path.insert(0,'src'); from core.ScenarioManager import main; main('src/scenarios/exchange_dummy_test.yaml')"

# 1c. Unit tests for the cost model.
$PY -m pytest tests/test_cost_model_comms.py -q
```

Expect `Scenario execution completed successfully.` from 1b and `6 passed` from 1c.

### Check the machine has room (mandatory before scaling up)

```bash
uptime          # load average — if already high, someone else is using the box
free -g         # keep at least 40% of memory free (project safety budget)
```

### Check nothing was left running by a previous campaign

**Do this every time.** Leftover processes hold TCP ports and make every new run
fail with `port(s) NNNNN already in use`.

```bash
ps -u $(id -u) -o pid=,args= | grep -E '[h]elics_broker|[f]ederate_launcher'
```
If that prints anything, clean it (see §7).

---

## 2. Re-run the experiments exactly as published

Order matters only in that the distributed run (2c) should not overlap the local
ones — they would compete for the same CPUs and corrupt each other's timings.

> ⚠️ **`run_bench.py` APPENDS to its CSV.** Delete the target file first or you
> will silently mix two campaigns into one fit. Each command below does this.

### 2a. Local, gate-limited (~10 min, 135 runs)

```bash
rm -f scripts/scaling_study/findings/phaseD_local.csv
$PY scripts/scaling_study/run_bench.py \
    --matrix scripts/scaling_study/matrices/phaseD_local.yaml \
    --repeats 5 \
    --out scripts/scaling_study/findings/phaseD_local.csv
```

### 2b. Local, wide range (~40 min, 177 runs) — the main result

```bash
rm -f scripts/scaling_study/findings/phaseD_local_wide.csv
$PY scripts/scaling_study/run_bench.py \
    --matrix scripts/scaling_study/matrices/phaseD_local_wide.yaml \
    --repeats 3 --timeout 600 \
    --out scripts/scaling_study/findings/phaseD_local_wide.csv
```
Peaks at 64 federate processes and ~20 GB RAM.

### 2c. Across machines (~25 min, 57 runs) — needs the cloud machines

First confirm they are reachable and idle:
```bash
for h in 130.192.238.9 130.192.238.13; do
  ssh -o BatchMode=yes eclabuser@$h 'hostname; uptime'
done
```
Then:
```bash
rm -f scripts/scaling_study/findings/phaseD_cross_machine.csv
$PY scripts/scaling_study/run_bench.py \
    --matrix scripts/scaling_study/matrices/phaseD_cross_machine.yaml \
    --repeats 3 --timeout 420 \
    --out scripts/scaling_study/findings/phaseD_cross_machine.csv
```
Expect ~8 failures: the wide-payload cells hit bottleneck B12 and time out. That is
a recorded result, not a broken run. **Clean up afterwards (§7)** — those timeouts
strand processes.

### 2d & 2e. Stress ladders (~5 min and ~20 min)

```bash
rm -f scripts/scaling_study/findings/stress_M.csv
$PY scripts/scaling_study/stress_ramp.py \
    --axis M --start 64 --factor 2 --steps 7 \
    --F 2 --N 4 --ticks 100 \
    --exchange on --distance cross_fed --fanout all2all \
    --timeout 900 --out scripts/scaling_study/findings/stress_M.csv

rm -f scripts/scaling_study/findings/stress_N.csv
$PY scripts/scaling_study/stress_ramp.py \
    --axis N --start 8 --factor 2 --steps 6 \
    --F 2 --M 4 --ticks 100 \
    --exchange on --distance cross_fed --fanout all2all \
    --timeout 900 --out scripts/scaling_study/findings/stress_N.csv
```
The N ladder is *expected* to end in a timeout at N=128 — that is the finding.
It will sit at that rung for the full 900 s before giving up. **Clean up after.**

### 2f. Re-fit and re-plot (seconds)

```bash
$PY scripts/scaling_study/cost_model.py fit \
    scripts/scaling_study/findings/phaseD_local_wide.csv \
    --out scripts/scaling_study/findings/phaseD_wide_fit_params.json

$PY scripts/scaling_study/plot_exchange.py \
    --bench scripts/scaling_study/findings/phaseD_local_wide.csv
```
Figures land in `findings/10…14_*.png`. The fit prints its coefficients, weighted
R² and median relative error into the JSON's `notes` field — **read that field**,
it records exactly which method was used and whether anything was clamped or
dropped for lack of data.

---

## 3. Change an experiment — which file to edit

### 3a. Change what is swept → edit a matrix file

`matrices/*.yaml` is the recipe. Each entry under `runs:` is one configuration.

```yaml
runs:
  - {F: 2, N: 4, M: 4, mode: seq, core_type: zmq, model: exchange_dummy,
     placement: local, ticks: 300,
     exchange: "on", distance: cross_fed, fanout: all2all,
     msg_width: 1, freq: 1, causality: same_step}
```

Knobs you can set (full definitions in `CONTRACT.md`):

| Knob | Values | Effect |
|---|---|---|
| `F` | int | federations |
| `N` | int | federates per federation |
| `M` | int | model instances per federate |
| `ticks` | int | simulated steps |
| `model` | `exchange_dummy`, `heavy_compute_dummy`, … | which model |
| `core_type` | `zmq`, `zmq_ss`, `tcp`, `tcp_ss` | transport |
| `placement` | `local`, `distributed_nat`, `distributed_direct` | which machines |
| `exchange` | `none`, `on` | wiring off/on |
| `distance` | `intra_fed`, `cross_fed`, `cross_machine` | who talks to whom |
| `fanout` | `1to1`, `1toN`, `Nto1`, `all2all` | wiring shape |
| `msg_width` | int | doubles per message (×8 = bytes) |
| `freq` | int | publish every k-th tick |
| `causality` | `same_step`, `next_step` | on critical path or not |

**Three rules when adding cells:**

1. **Every wired cell needs a control twin.** Add a matching row with
   `exchange: none` and the *same* `F, N, M, mode, core_type, model, placement,
   ticks`. Without it the fit silently drops your row — there is nothing to
   subtract. This is the single most common way to waste a campaign.
2. **Vary `N` and `M` independently** if you care about separating "total edges"
   from "edges per federate". Holding one fixed makes them collinear and the fit
   will confidently give you the wrong answer (this happened — see
   `EXPERIMENTS.md` §3).
3. **`n_edges`, `n_subs`, `max_fed_in`, `max_fed_out` are computed for you** by the
   generator. Do not put them in the matrix.

Alternative to `runs:` — a full cartesian product:
```yaml
axes:
  N: [2, 4, 8]
  M: [1, 4]
  ticks: [100]
```
Use sparingly: cartesian sweeps are what produced the un-fittable collinear data
in the earlier phase.

### 3b. Change *how* wiring is built → `gen_scenario.py`

Only needed for a genuinely new topology (e.g. a ring, or a many-to-many that is
not bipartite). Relevant functions:

| Function | Does |
|---|---|
| `compute_exchange_edges()` | picks publisher/subscriber sides and which targets which. **Edit here to add a new `fanout` pattern.** |
| `build_subscribe_block()` | emits the actual YAML `subscribes:` entry |
| `max_federate_links()` / `count_subscriptions()` | derive `max_fed_in`/`max_fed_out`/`n_subs` |
| `build_scenario()` | assembles everything, assigns broker ports |

After editing, **verify backward compatibility** — an unwired scenario must come
out byte-identical to before. Generate the reference **before** you start editing:

```bash
# BEFORE editing gen_scenario.py:
$PY scripts/scaling_study/gen_scenario.py --F 2 --N 4 --M 4 --work 1000 --out /tmp/ref.yaml

# ... make your changes ...

# AFTER editing, same arguments:
$PY scripts/scaling_study/gen_scenario.py --F 2 --N 4 --M 4 --work 1000 --out /tmp/new.yaml
diff /tmp/ref.yaml /tmp/new.yaml     # must be EMPTY
```
(The `.spec.json` sidecar may legitimately differ if you added a knob; the YAML
must not.) If you forgot to capture the reference first, get it from git without
disturbing your working tree:
```bash
git show HEAD:scripts/scaling_study/gen_scenario.py > /tmp/gen_ref.py
$PY /tmp/gen_ref.py --F 2 --N 4 --M 4 --work 1000 --out /tmp/ref.yaml
```

> Keep wiring **bipartite** (publishers and subscribers disjoint). CosimGym rejects
> `same_step` dependency cycles outright — `ScenarioManager._validate_causality_cycles()`
> raises — so a cyclic topology aborts before tick 1 and forces you into
> `next_step`, which then contaminates the causality comparison.

### 3c. Change the dummy model's behaviour → `exchange_dummy.py`

`src/models/model_catalog/physical_models/exchange_dummy.py`. Parameters:
`msg_width` (output vector length), `publish_every` (cadence — driven by the `freq`
knob), `iterations` (optional CPU burn, default 0).

**If you change `catalog.yaml` you must reload Redis, or the run fails with
`model 'X' not found in catalog:index`:**
```bash
$PY src/models/model_catalog/catalog_loader.py
```

Keep per-step work independent of `msg_width` (build the vector from a cached
template) or the model's own CPU cost will masquerade as communication cost.

### 3d. Change the cost model → `cost_model.py`

The `comms` section of `fit()`. If you add a regressor you must also update
`predict()`, `_empty_params()`, and the shape documented in `CONTRACT.md`.
Then re-run `tests/test_cost_model_comms.py`, which injects known coefficients into
synthetic data and asserts they are recovered within 10% — if that fails, the
design matrix is wrong.

### 3e. Change the stress ladder → CLI only

No file edit needed. `--axis M|N --start --factor --steps`, plus
`--guard-free-pct` (default 40) and `--guard-load` (default = core count).

### 3f. Change the figures → `plot_exchange.py`

One function per figure (`fig10_…` … `fig14_…`). When plotting anything against a
knob, group by **both** `N` and `M` — grouping by `M` alone once pulled a 22 ms
outlier into a series of ~300 µs points and dwarfed every real datum.

---

## 4. Add a completely new experiment

1. Write `matrices/<name>.yaml` — wired cells **plus their controls** (§3a).
2. Dry-run one cell before committing to the whole matrix:
   ```bash
   $PY scripts/scaling_study/gen_scenario.py --F 2 --N 4 --M 4 \
       --model exchange_dummy --exchange on --distance cross_fed \
       --fanout all2all --ticks 30 --out /tmp/check.yaml
   cat /tmp/check.yaml.spec.json      # confirm n_edges / max_fed_in look right
   ```
3. Run it with `--repeats 1 --timeout 120` first to prove it completes.
4. Then the real run with `--repeats 3` (or 5 for noisy cells).
5. Fit, plot, and write findings into `findings/`.
6. Update `findings/README.md` — it is the canonical index and wins over every
   other document.

---

## 5. Reading the output

One CSV row per run. Columns that matter most:

| Column | Meaning |
|---|---|
| `tick_mean_s` | **the** performance number — mean seconds per tick |
| `failure_mode` | empty = success; else `timeout`, `lost_comms_-101`, … |
| `n_edges`, `n_subs`, `max_fed_in`, `max_fed_out` | derived topology sizes |
| `peak_rss_mb`, `cpu_util_pct` | sampled while running |
| `setup_s`, `sim_wall_s` | startup vs simulation time |

Quick look at a campaign:
```bash
$PY -c "
import csv;rows=list(csv.DictReader(open('scripts/scaling_study/findings/phaseD_local_wide.csv')))
ok=[r for r in rows if not r['failure_mode']]
print(len(ok),'ok /',len(rows),'total')
for r in ok[:5]: print(r['N'],r['M'],r['n_edges'],r['tick_mean_s'])
"
```

---

## 6. Safety gates — do not skip these on the shared machine

The manager is shared with other users. From the study plan §8.7:

- Runs larger than **F=2, N=4, M=4, ticks=30** need explicit go-ahead.
- Any **distributed** run needs explicit go-ahead.
- Check `uptime` and `free -g` **immediately before** scaling up; abort if load is
  already high or the 40%-free-RAM budget would be breached.
- Smoke-test a shortened version before any long run.

`stress_ramp.py` enforces the memory/load part automatically and records a guard
abort as "we stopped", distinct from "the framework failed".

---

## 7. Cleanup and troubleshooting

### After ANY timeout or interrupted run — always

```bash
# Stranded federates (run_bench's reaper does NOT handle these)
pkill -u $(id -u) -f federate_launcher || true

# Stranded brokers
pkill -u $(id -u) -f helics_broker || true

# Verify clean — should print nothing
ps -u $(id -u) -o args | grep -E '[h]elics_broker|[f]ederate_launcher' || echo "clean"

# If a distributed run was involved, clean the remotes too
for h in 130.192.238.9 130.192.238.13; do
  ssh -o BatchMode=yes eclabuser@$h 'pkill -f federate_launcher; pkill -f helics_broker' || true
done
```

### Symptom → cause

| Symptom | Cause and fix |
|---|---|
| `port(s) NNNNN already in use — most likely an orphaned broker` | Leftovers from a previous run. Run the cleanup above. |
| `Unable to bind zmq pull socket giving up tcp://…` | Port blocks too close together (bottleneck B10). Fixed in `gen_scenario.py`; if you hand-write a multi-federation `zmq` scenario, leave ≥ `N+11` ports between federation broker ports, or use a `*_ss` core type. |
| Run hangs, simulation already finished, `disconnect Timer expired forcing disconnect` in federate logs | Bottleneck **B12**, open and undiagnosed. Stay under ~256 federates locally / ~1 kB per tick distributed. |
| `model 'exchange_dummy' not found in catalog:index` | Redis catalog stale — `$PY src/models/model_catalog/catalog_loader.py`. |
| Fit reports coefficients of 0.0 or "no usable wired rows" | Missing control twins (§3a rule 1), or all-zero columns. Read the `notes` field of the params JSON — it says which. |
| Timings noisy / inconsistent | Someone else is on the machine. Check `uptime`; re-run when idle. |

### Where the logs are

```
logs/<scenario_name>/<timestamp>/federates/federate_<name>.log        # CosimGym log
logs/<scenario_name>/<timestamp>/federates/federate_<name>.stdio.log  # raw HELICS output
results/<scenario_name>/<sim_id>/perf.json                            # timing summary
```
The `.stdio.log` files are where HELICS reports the real cause of a hang — check
them first when a run misbehaves.

### Shell gotchas on this machine

- The shell runs with `set -e`: a `pkill` that matches nothing returns 1 and aborts
  the rest of a compound command. Append `|| true`.
- Long runs: use `nohup … &` or a background runner; `conda run` has been killed
  early on multi-minute jobs.
