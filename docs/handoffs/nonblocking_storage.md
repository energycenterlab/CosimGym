# Handoff — `nonblocking_storage` (Plan 2)

**Plan file:** `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
**Branch:** `nonblocking_storage` (created off `main`, independent of `digitaltwin_interfaces`)

> **Process (same as Plan 1):** the agent never runs `git commit`. It stages
> (`git add`) and hands off; **you** run the commit. A milestone's box in the
> Progress Tracker only gets ticked once you've confirmed the commit landed.

## Last committed milestone

None yet — this is the first session on this branch. Branch point: `main`
`38948b3`.

## Staged, awaiting your commit: S0 — `sink` field on `MemoryConfig`

- **`src/utils/config_dataclasses.py`** — `MemoryConfig` gets a new
  `sink: Literal['json', 'parquet', 'none'] = 'json'` field. Default `'json'`
  = today's behavior, unchanged.
- **`src/core/BaseFederate.py`** (`store_local_file`) and
  **`src/core/RL_Federate.py`** (its own separate `store_local_file`
  override — `RL_Federate` doesn't call `super()` here, so both needed the
  same guard) now branch on `self.config.memory_config.sink`:
  - `'json'` (default): existing behavior, byte-for-byte unchanged.
  - `'none'`: skips writing the per-federate JSON file entirely (logs an
    info line instead) — free to add now since it needs no new writer.
  - `'parquet'`: raises a clear `NotImplementedError` pointing at S2, rather
    than silently falling back to JSON or doing nothing. Deliberate: nobody
    should think they got Parquet output when the writer doesn't exist yet.
- **`tests/test_rl_config.py`** — new `TestMemoryConfigSink` (4 tests):
  default `'json'`, explicit `'parquet'`/`'none'` accepted, invalid value
  (`'csv'`) rejected.

**Verified:**
- Config gate: `pytest tests/test_rl_config.py -q` → 58 passed, 1 skipped
  (this branch is off `main`, before Plan 1's scenario/test additions — don't
  be surprised the count differs from `digitaltwin_interfaces`' 78).
- Regression: `python src/test_script.py` (green, `dh_district_jan_base`,
  4 federates) and `OMP_NUM_THREADS=1 python src/test_script_rl.py` (green,
  `bui0_heatingpower_DQN`, 3 brokers/3 federates) — both use the default
  `sink: json` implicitly, confirming zero behavior change.
- **Runtime plumbing smoke test** (not just the pydantic schema): a scratch
  scenario (`spring_mass_damper`, single federate, 10 ticks) run three times
  with `memory_config.sink` set to each of the three values:
  - `'json'` (implicit default): unchanged.
  - `'none'`: `results/<scenario>/<sim_id>/federation_1/` has `metadata.json`
    (written by `ScenarioManager`, unaffected) but **no**
    `spring_federate_test_storage.json` — the federate's own log shows
    `"memory_config.sink='none' — skipping local file storage"`.
  - `'parquet'`: the federate subprocess raises the `NotImplementedError`
    exactly as designed — confirmed in the federate's `.log` file (full
    traceback rooted at `BaseFederate.store_local_file`), which the manager
    correctly reports as `✗ Federate failed with code 1`. This is the
    intended "fail loud, not silent" behavior for an unimplemented sink.

**Your action:** review the staged diff, commit (e.g. `feat(storage): S0 add
memory_config.sink field (json default unchanged, none, parquet stub)`),
then say continue — I'll tick S0 and move to S1.

## Next step (after you commit S0)

**S1 — Background writer thread + bounded queue fed from `update_storage`.**
Per the plan's Design section: the producer is the existing per-step
`update_storage` hook (`BaseFederate.py:761`, called each tick from `run()`);
it should enqueue rows onto a bounded in-process `queue.Queue` instead of (or
alongside) the current in-memory list appends, and a background thread drains
the queue. S1 itself doesn't need to write Parquet yet (that's S2) — it
just needs the threading/queueing plumbing in place, verified not to disturb
`sink: json`'s behavior (which should probably keep using the direct
in-memory `self.storage` path, since S1/S2 are additive for `sink: parquet`
specifically, not a replacement of the JSON path). Re-read the plan's
"Design" and "Locked decisions" subsections under Plan 2 before starting —
the rationale for one-thread-per-federate + in-process queue (not a
separate process, to avoid pickling cost) is spelled out there.

First concrete action: read `BaseFederate.update_storage` (`BaseFederate.py:761`)
and decide where a `sink == 'parquet'` branch would enqueue a row without
touching the `sink == 'json'` code path at all.

## Files touched so far (S0)

**Modified:** `src/utils/config_dataclasses.py`, `src/core/BaseFederate.py`,
`src/core/RL_Federate.py`, `tests/test_rl_config.py`.
**New:** none yet.

## State of the tree

On `nonblocking_storage`, 0 commits ahead of `main` (`38948b3`) — S0's
changes are `git add`ed but **uncommitted**, waiting on you.

## Blockers / deviations from the plan

1. **`RL_Federate.store_local_file` is a full override, not a `super()` call**
   — had to duplicate the `sink` guard there rather than putting it in one
   place. If S1/S2 need more logic here, consider whether it's worth
   refactoring the common part into a shared helper at that point (not done
   now — S0 is schema + minimal plumbing only, avoid scope creep).
2. **`memory_config.sink='parquet'` currently hard-fails** (`NotImplementedError`)
   rather than silently doing nothing or falling back to JSON. This is a
   deliberate choice, not a bug — revisit only if S2 wants a softer rollout
   path (e.g. a warning + JSON fallback while Parquet is being rolled out
   incrementally across scenarios).
3. **This branch does not include Plan 1's `digitaltwin_interfaces` work**
   (not merged into `main` yet) — the two plans are intentionally decoupled,
   per the plan file. `MemoryConfig` here has no `streaming`/`interface_config`
   siblings; don't expect them.

## How to verify current state

```bash
cd /media/space/rando/CODE/CosimGym
git status && git branch --show-current   # nonblocking_storage; S0 staged, not committed
git log --oneline -3                       # 38948b3 (main) at HEAD, no new commits yet
git diff --staged --stat                   # S0's staged changes

conda activate cosim_gym
docker compose -f src/docker-compose.yaml up -d   # redis, minio (no mosquitto on this branch)

# Regression — must match main behavior exactly (sink defaults to json):
python src/test_script.py
OMP_NUM_THREADS=1 python src/test_script_rl.py

# Config gate:
python -m pytest tests/test_rl_config.py -v   # 58 passed, 1 skipped

# Runtime plumbing check (sink=none / sink=parquet) — build a tiny scratch
# scenario (see any src/scenarios/*.yaml for the spring_mass_damper shape),
# set memory_config.sink accordingly, and run it via:
PYTHONPATH=src python -c "from core.ScenarioManager import main; main('/absolute/path/to/scratch.yaml')"
# sink: none  -> no <federate>_<mode>_storage.json written, federate log says so
# sink: parquet -> federate subprocess raises NotImplementedError (by design, S2 not built yet)
```

## One-line kickoff prompt for a fresh session

> "Read `/media/space/rando/.claude/plans/federate-in-the-co-simualtion-fancy-ladybug.md`
> (Plan 2 section) and `docs/handoffs/nonblocking_storage.md`. We're on branch
> `nonblocking_storage` (off `main`, independent of `digitaltwin_interfaces`).
> S0 (`sink` field on `MemoryConfig`) is implemented and verified but staged,
> not committed — review and commit it yourself first, then tell the agent
> to tick S0 and continue to S1 (background writer thread + bounded queue).
> Follow the per-milestone loop: implement, run the milestone's check +
> regression tests, stage (don't commit), update the handoff doc, then stop
> and wait for the user's commit + 'continue' signal before ticking the box
> and starting S2."
