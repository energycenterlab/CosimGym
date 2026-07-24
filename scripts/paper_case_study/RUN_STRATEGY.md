# Paper Case-Study — Run & Reporting Strategy (CONCLUDED)

Status: **all scenarios generated and smoke-tested; fast tier already executed;
the long RL sweep is prepared and left for the author to run.**

## Machine (as measured)
- Shared server, **112 CPU cores**, Linux 5.4, **glibc 2.31**.
- HELICS 3.6.1 · gymnasium 1.2.2 · stable_baselines3 2.8.0 · matplotlib 3.9.4 · Python 3.12.
- Docker services (redis, mosquitto, minio) must be up:
  `docker compose -f src/docker-compose.yaml up -d`

## Scenario inventory (all smoke-tested PASS)

| stage | scenarios | count |
| --- | --- | --- |
| S1 | `cs_s1_baseline` | 1 |
| S2A | `cs_s2_sac`, `cs_s2_dqn` | 2 |
| S2B | `cs_s2_reset_{full,rolling,none}` | 3 |
| S3 | `cs_s3_fmu` (SAC on BUI0 EnergyPlus FMU) | 1 |
| S4a | `cs_s4_vert_{seq,par}_N{1,5,10,20,40}` | 10 |
| S4b | `cs_s4_topo` (2 federations) | 1 |
| S4c | `cs_s4_dist_{1,2,3}m` (SSH distributed) | 3 |
| S5 | `cs_s5_dt` (+ `s5_external_feeder.py`) | 1 |
| **total** | | **22** |

Plus **18 generated seed variants** (`<base>_s{42,43,44}.yaml`) for the RL stages,
produced by `make_seed_variants.py` (changes ONLY name, seed, checkpoint name).

## Execution rules (learned the hard way — do not ignore)
- **Run scenarios SEQUENTIALLY.** Concurrent runs cause HELICS broker/port and Redis
  contention; a concurrent batch produced spurious failures that vanished when serialised.
- **`core_type: tcp`, not `zmq`.** zmq allocates a paired socket at `port+1`, which collides
  with the framework's sequential broker port auto-assignment. This bit S4a and S4b.
- **Omit `broker_config.port`** so ports are auto-assigned.
- FMU scenarios need **whole-day-multiple horizons** and cannot be shortened below one day.

## What has already been RUN (fast tier — done)
`run_all.sh` reproduces it: S1, S4a (10 scenarios × 3 reps), S4b, S5, S6.
Deliverables present in `results/paper_case_study/`:
`fig_s1_traces`, `tab_s1_metrics`, `fig_s4_throughput`, `exec_metrics.csv`,
`s4b_hierarchy_broker_evidence.txt`, `s5_dt_acceptance_evidence.txt`, `tab_s6_loc`.

Headline numbers already obtained:
- PID baseline: **7.516 comfort degree-hours, 48.940 kWh** over 48 h.
- S4a: sequential **8→223 s**, parallel **9→36 s** → **6.2× at N=40**, crossover ≈ N=5;
  peak RSS parallel 22.6 GB vs sequential 0.72 GB.
- S4b: hierarchy broker `--sub_brokers=2`, dynamic ports, cross-federation link verified.
- S5: PID consumed externally fed MQTT `T_indoor` (19.2–20.4 °C), not a physics model.

## What is LEFT for the author to run
```bash
docker compose -f src/docker-compose.yaml up -d
bash scripts/paper_case_study/run_s2.sh          # S2A + S2B + S3, all seeds. HOURS.
```
`run_s2.sh` = generate seed variants → run every variant **sequentially** via
`run_profiled.py` (records wall-clock + peak RSS) → regenerate
`tab_s2_metrics`, `fig_s2_learning_curves`, `tab_s2_sample_eff`, `tab_s3_metrics`.

Cost driver: 5 RL bases × 3 seeds × 100 episodes × 2880 steps (+ S3 at 30×144).
Override seeds with `SEEDS="42 43" bash scripts/paper_case_study/run_s2.sh`.

For **S4c on real machines** (currently only loopback-smoked):
1. edit `deployment.machines.*.host` in `cs_s4_dist_{1,2,3}m.yaml` to the real LAN IPs,
2. set `deployment.manager_address` to this manager's LAN IP,
3. ensure passwordless SSH + a `cosim_gym` env + writable `workdir` on each remote,
4. `python run_profiled.py cs_s4_dist_1m cs_s4_dist_2m cs_s4_dist_3m --reps 3`
5. `python fig_s4_machines.py`   (drop `--loopback` once on real hosts)

## Analysis scripts (all built)
`metrics.py` (shared: comfort deadband [19.5,20.5] °C, energy ∫P dt; naming-agnostic
via `scenario_metrics()`), `fig_s1_traces.py`, `run_profiled.py`, `make_seed_variants.py`,
`tab_s2_metrics.py`, `fig_s2_learning_curves.py` (also emits `tab_s2_sample_eff`),
`tab_s3_metrics.py`, `fig_s4_throughput.py`, `fig_s4_machines.py`, `s6_loc_audit.py`,
`s5_external_feeder.py`. Drivers: `run_all.sh` (fast tier), `run_s2.sh` (RL sweep).

Every table prints `MISSING` for absent runs — numbers are never fabricated.

## Known constraints / honest caveats for the paper
- **S3 is not a one-line swap.** Replacing `simple_building` with the EnergyPlus FMU also
  removes the weather + heat-pump federates, changes the action from heat-pump modulation
  to a zone set-point, changes the reward, forces `real_period` 60→600 s and
  `reset.mode: none`. `tab_s3_metrics.md` emits this diff as a table — report it plainly.
- **S3 energy is THERMAL** (`HeatingLoadTarget`), not electrical — not comparable to S1/S2 kWh.
- **The `adelaide_test` FMU cannot run on this machine**: its native `.so` requires
  GLIBC_2.33, the host has 2.31. The MinIO object was restored (it was genuinely missing),
  and the download now works, so this is purely a binary-compatibility wall. BUI0 is used instead.
- **S4c loopback ≠ scaling evidence.** All three configs run on one physical host, so they
  prove the SSH/rsync/collection mechanism only. Do not plot them as a speedup curve.
- `fig_s5_dashboard.png` is a human screenshot (dashboard Live page during a `cs_s5_dt` run).

## ADDENDUM (2026-07-23) — S4c real-machine run now done

The "left for the author" S4c real-machine item above is superseded: the author
approved and this was run for real on cloud1/cloud5. See
`results/paper_case_study/s4c_real_analysis.md` for the full write-up (capacity-
scaling result, ceiling-characterization sweep reproducing the historical
`[-101] lost comms` failure fresh, honest pros/cons) and the new MANIFEST.md
section "S4c-real — REAL multi-machine deployment". New assets: generator
`src/scenarios/generate_scale_sharded.py`, scenarios `cs_s4c_shard_{1,2,3}m.yaml`,
figures `fig_s4c_capacity`/`fig_s4c_ceiling`. The original `cs_s4_dist_{1,2,3}m.yaml`
loopback trio and `fig_s4_machines.py` are untouched and still valid as the
mechanism-only demo.

## ADDENDUM (2026-07-23, later) — mass-scale (K-per-shard) study, CHECKPOINT only

A different, follow-on scaling axis: instead of adding one federate per
building (S4c-real above), hold federate count fixed at 4/machine and push
K = n_instances inside `building`/`heatpump`/`pid`. A background agent session
ran this for real (`src/scenarios/generate_mass_instances.py`, new,
`cs_mass_k*` / `cs_mass_shard_{1,2,3}m_k*.yaml`) but was **interrupted before
writing up the analysis**. A follow-on checkpoint pass consolidated the real
data already on disk (`logs/cs_mass_*/*/execution_metrics.json`) — it did
**NOT** launch any new real-machine run, per the explicit author instruction
"before running massive simulations ask me." See
`results/paper_case_study/mass_scale_bottleneck_analysis.md` for the full
write-up and the new MANIFEST.md section "Mass-scale (K-per-shard) bottleneck
study — CHECKPOINT".

Headline (all real, measured): single-machine `tcp` core reaches **K=100,000
buildings with zero failures** (~19.7 min wall-clock) — the biggest number in
this whole study. But the NAT-mandated `zmq_ss` core (needed for any real
multi-machine run) fails between K=7,000 (PASS) and K=10,000 (FAIL) even
**fully locally, no network involved** — a broadcast-fan-out timeout in the
weather→buildings publication, not the federate-count ceiling already
documented in S4c-real. Over real SSH, that per-shard ceiling drops further as
machines are added (2 machines: safe at 500/machine, fails at 700; 3 machines:
safe at 200/machine, fails at 500) — i.e. **mass-scale (K) sharding and
federate-count sharding do not compose the same way**: the latter scales
capacity ~linearly with machines (S4c-real), the former shrinks it. A labeled
"PROPOSED NEXT RUN — awaiting author go-ahead" section in the analysis doc
recommends the specific next real-hardware experiments (not executed).

One dead scenario file was deleted (`cs_mass_shard_3m_k5000.yaml` — generated,
never run, no log/result/csv evidence); the other 16 generated `cs_mass_*.yaml`
files all back a real reported number (PASS, FAIL, or explicitly-marked
INTERRUPTED/inconclusive) and were kept. Orphan-process check (read-only `ps`,
locally and via SSH on `machine_a`/`machine_b`, nothing started): clean, no
leftover `federate_launcher.py`/`helics_broker` processes from the interrupted
session.
