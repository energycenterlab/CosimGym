# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file and all support documents live in `CLAUDE.md` →
**Working Rules**. Read that section first.

---

## Goal (this session)

Security requirement from outside the project: **never run a Redis server older than
7.4.11**. Answer whether CosimGym can satisfy it, then do the upgrade and prove the
framework still works.

## Current Progress — done, uncommitted

### The analysis

- The stack ran `redis/redis-stack-server:latest`, which resolves to `7.4.0-v8`
  (pushed 2025-11-03) and reports **`redis_version:7.4.7`** — below the floor. Redis Stack
  has no newer core line and is not moving, so the floor cannot be met by staying on it.
- The framework's whole config/catalog pipeline is **RedisJSON** (`client.json().set/get`
  in `src/utils/redis_client.py`, `RedisCatalog.py`, `catalog_loader.py`,
  `fmu_catalog_register.py`). So plain `redis:7.4.11` is a dead end — the official 7.4
  image carries no modules.
- **Redis Open Source 8.x bundles ReJSON in the base image**, so it is both above the floor
  and module-complete. That is the upgrade path taken.
- Full Redis surface the repo actually uses: `JSON.SET` / `JSON.GET` (legacy `.`-paths),
  `SET`, `GET`, `DEL`, `EXISTS`, `EXPIRE`, `PING`. No streams, no `FT.*` / `TS.*` / Bloom,
  no Lua, no RedisGears. The `publish` / `subscribe` calls in `src/` are HELICS and MQTT,
  not Redis pub/sub. `redis-py` in `cosim_gym` is already 8.0.0.

### Shipped

- `src/docker-compose.yaml` and `docker-compose.setup.yml`: image
  `redis/redis-stack-server:latest` → **`redis:8.2.10-alpine`**, command
  `redis-stack-server …` → `redis-server …`, plus a comment recording why the pin exists.
- `Makefile`: `teardown` removed a volume named `cosim_gym_redis_data`, which has never
  existed — the compose project is `src`, so the volume is `src_redis_data`. Fixed, because
  the upgrade needs a working volume-reset path.
- `docs/Installation_Setup.md`: note on the image pin, why `redis:7.4.x` is not a
  substitute, and the one-time volume reset for existing checkouts.
- `docs/KNOWN_ISSUES.md`: new §7 (see *What Didn't Work* below), `Last reviewed` bumped.
- `graphify-out/` regenerated.

### Verified on Redis 8.2.10

- `MODULE LIST` → `ReJSON 80209` (+ search, timeseries, bf, vectorset).
- Legacy JSON path semantics identical to Stack: `JSON.GET k .`, `JSON.GET k .inputs`,
  missing path → `ERR Path does not exist`, missing key → nil. `RedisClient.set_json` /
  `get_json` / `get_json_path` / `delete` all behave as before.
- `RedisCatalog.get_model_metadata` / `get_inputs_outputs` / `query` / `search_models`
  return full metadata for both a Python model and the BUI0 FMU entry.
- `catalog-loader` exits 0, 28 catalog keys uploaded; full `down` → `up -d` cycle clean.
- Persistence works: `rdb_last_bgsave_status:ok`, `/data/dump.rdb` written and owned by
  uid 999.
- Non-loopback reachability (the distributed-SSH path) unchanged: `protected-mode no`,
  `bind * -::*` on both images; a client on the host LAN IP pings and reads JSON.
- **Scenarios PASS** through `regression_suite.run_scenario`: `rc_building_test_base`,
  `bui0_fmu_test`, `fmu_horizon_smoketest`, `bui0_setpoint_DQN`, `bui0_setpoint_SAC`,
  `bui0_heatingpower_DQN`. Those are *every* scenario that can currently start — see below.
- An RDB written by redis-stack 7.4.7 was also confirmed to load into 8.2.10 with its
  ReJSON keys intact, so no data format barrier exists even where a volume does hold data.

## What Worked

Probing the images instead of trusting tag names: running `redis-stack-server:7.4.0-v8`
and reading `INFO server` is what revealed the core is 7.4.7 (not 7.4.0, and not 7.4.11).
Everything downstream of that — the module list, the legacy-path semantics, the RDB
round-trip, the `/data` ownership — was settled the same way, by running the container,
before touching a single line of the repo.

## What Didn't Work / Watch Out

- **The old volume breaks the official image.** `src_redis_data` carried empty `redis/`
  and `redisinsight/` directories left by Redis Stack. The official entrypoint sees an
  unknown file in `/data`, prints `Notice: Unknown file './redis' found in data dir.
  Permissions will not be modified.` and **skips its chown**, so the server (uid 999)
  cannot write and every write fails with
  `MISCONF Redis is configured to save RDB snapshots, but it's currently unable to persist
  to disk`. Removing the two empty dirs (or the whole volume) fixes it permanently. This is
  documented in `docs/Installation_Setup.md`; a fresh checkout never hits it.
- **`main` is broken independently of Redis, and badly.** Commit `208f9fb` made
  `BaseModel.reset` an `@abstractmethod` without implementing it in 13 catalog models, so
  26 of 32 regression scenarios cannot even instantiate their models
  (`TypeError: Can't instantiate abstract class … 'reset'`), and `pytest tests/` is
  **25 failed, 292 passed, 2 skipped** — FMU snapshot API (`_state_snapshots`,
  `_snapshot_target_ts`) gone from the class while the tests still reference it, plus the
  two pre-existing remote tests. Logged as `docs/KNOWN_ISSUES.md` §7. **Not touched** — it
  is the user's in-flight refactor, and that commit's own message says *"need to
  refactor!"*. It is also why the pre-merge gate could not be run as a whole.
- Consequently: **the six passing scenarios are the entire testable surface right now.**
  They do cover base co-sim + CSV models, the FMU/MinIO path, and three RL runs, which
  between them exercise every Redis code path in the framework — but multifed, distributed
  and parallel could not be re-verified because their models don't instantiate.
- Do not "fix" the abstract-`reset` breakage as a drive-by. It changes the shape of the
  refactor in progress.

## In-flight, uncommitted state

Branch `main`, HEAD `208f9fb`, nothing committed this session. Modified:

- `src/docker-compose.yaml`, `docker-compose.setup.yml` — the image/command change
- `Makefile` — volume name
- `docs/Installation_Setup.md`, `docs/KNOWN_ISSUES.md`
- `graphify-out/` regenerated (`graphify update .`)

The running stack is already on 8.2.10 and healthy; `src_redis_data` has been reset once
and now persists correctly.

## Next Steps

1. **Finish the `reset` refactor** (`docs/KNOWN_ISSUES.md` §7) — either implement `reset()`
   per model or give `BaseModel` a concrete no-op default and keep the abstract contract
   only where a reset means something. Until then the gate cannot go green.
2. **Re-run the full gate on Redis 8** once §7 is closed:
   `conda run -n cosim_gym python tests/regression_suite.py`. The axes still unverified on
   8.2.10 are multifed, distributed-SSH and parallel model execution.
3. Optional hardening now that the security conversation is open: Redis is still
   LAN-exposed and unauthenticated. Adding `requirepass` touches `RedisClient`, both
   compose files and the launcher URL format (`redis://:pass@host`) — sized up in
   `docs/future_and_TODOs/distributed_ssh_spawning_alternatives.md`.
4. Licensing footnote for the paper: Redis Stack was RSALv2/SSPL; Redis 8 is tri-licensed
   AGPLv3 / RSALv2 / SSPL. CosimGym talks to it over a socket as a separate process, so
   there is no copyleft reach into the framework — but the compose file is distributed, so
   the software section should say which server it pulls.
5. Nothing is committed without an explicit ask.
