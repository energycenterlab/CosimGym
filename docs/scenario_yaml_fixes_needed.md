# Scenario YAML — Fixes Needed (post RL-config-refactor)

**Date:** 2026-06-30
**Branch merged to main:** `rl-config-refactor`
**Status:** All scenarios adapted to the new 4-axis RL schema (`environment` / `agent` / `run` / `experiment`) and parse cleanly. The items below are **residual correctness bugs inside adapted files** — not schema-migration gaps. Tracked here for follow-up; merge proceeded because the base path and the majority of RL scenarios run clean.

## Validation done before merge

Ran from project root, `cosim_gym` env (`/media/space/rando/Environments/cosim_gym`):

| Scenario | Backend | Result |
| --- | --- | --- |
| `simple_SACsb3_test` | SB3 SAC | ✅ clean |
| `simple_rllib_test` | RLlib PPO | ✅ clean (cosmetic warning) |
| `bui0_heatingpower_DQN` | custom DQN | ✅ clean, real reward |
| `bui0_fmu_test` (base) | — | ✅ clean |
| `bui0_setpoint_DQN` | custom DQN | ⚠️ runs, reward ≡ 0 (bug B) |
| `bui0_setpoint_SAC` | SB3 SAC | ⚠️ runs, reward ≡ 0 (bug B) |
| `simple_DQN_test` | custom DQN | 🔴 crash at episode boundary (bug A) |
| `bui_hp_DQN`, `bui_hp_DQN_rollingreset`, `bui_hp_SAC` | — | ⏱ slow full-training run (>300s); no crash before timeout |

Not yet runtime-verified: `bui_hp_SAC_rollingreset`, `pv_batt_DQN`, `pv_batt_SAC`, `simple_test_rlagent`, and the base suite (`pandapower/pandapipes/rc_building/multi_building/...`). They parse OK; runtime smoke pending.

---

## Bug A 🔴 — `simple_DQN_test.yaml` crashes (missing `experiment` block)

**Symptom (live):**
```
rl_simple_DQN.py:234  self.model.save_model(checkpoint_path)
rl_simple_DQN.py:140  d = os.path.dirname(path)
TypeError: expected str, bytes or os.PathLike object, not NoneType
```
At the first episode boundary the DQN agent dies. The driver still prints "Completed" because federate death is not propagated to the exit code (separate observability gap).

**Root cause:** scenario has no `experiment` block → `CheckpointConfig.best = None` → `best_path = None` → `save_model(None)`. The sibling `simple_SACsb3_test.yaml` *has* the block; it was dropped when migrating the DQN file.

**Fix (YAML)** — add to `simple_DQN_test.yaml` under `reinforcement_learning_config:` (sibling of `run:`):
```yaml
  experiment:
    checkpoint:
      best: best_spring_dqn.pth
```

**Defense-in-depth (code, optional):** guard `DQNAgent.save_model` against a `None` path in `src/models/model_catalog/RL_agents/rl_simple_DQN.py`.

---

## Bug B ⚠️ — `bui0_setpoint_DQN.yaml` / `bui0_setpoint_SAC.yaml` silent zero reward

**Symptom:** scenarios run to completion but reward is `0.0` every step → agents learn nothing. These are the flagship "Example 1" demos.

**Root cause:** reward fn `bui0_setpoint_comfort` (`src/models/model_catalog/RL_agents/reward_functions.py`) reads
`obs['federation_1.feeder_federate.0.ZoneSetPoint']`, but in the setpoint scenarios the only observation is
`federation_1.building_federate.0.TBuilding` (ZoneSetPoint is the **action**, published by `building_federate`).
Lookup raises `KeyError`, swallowed by a bare `except Exception: return 0.0`.

Confirmed from runtime config dump:
`observations={'federation_1.building_federate.0.TBuilding': ...}`, `actions={'federation_1.building_federate.0.ZoneSetPoint': ...}`.

`bui0_heatingpower_DQN.yaml` is **unaffected** — it explicitly observes `feeder_federate.0.ZoneSetPoint`.

**Fix (code, preferred)** in `reward_functions.py::bui0_setpoint_comfort`:
- Drop the `ZoneSetPoint` lookup (the hard-coded `T_TARGET = 21.0` above the `try` is what setpoint scenarios want — comparing the chosen setpoint to itself is meaningless), **and**
- narrow the bare `except Exception` → `except KeyError` (or `(KeyError, TypeError, ValueError)`) so future misconfig fails loud instead of silent-zero.

**Alternative (YAML)** — if the lookup must stay, add `feeder_federate.0.ZoneSetPoint` to `environment.observations` with `role: extra` in both setpoint scenarios. But the code fix is cleaner.

---

## Bug C 🟡 nit — `reset.mode: none` warning spam

**Affected YAML:** `bui0_setpoint_DQN.yaml`, `bui0_setpoint_SAC.yaml`, `bui0_heatingpower_DQN.yaml` (all use `environment.reset.mode: none`, correct for the EnergyPlus FMU which cannot re-init mid-run).

**Symptom:** `BaseFederate._reset` only handles `full`/`rolling`; `none` falls to the catch-all →
`WARNING: Unknown reset type 'none' specified.` every reset cycle (~episodes × base-federates). Behavior correct, log misleading.

**Fix (code, not YAML)** in `src/core/BaseFederate.py` `_reset`, before the `else`:
```python
elif self.reset_type in (None, 'none'):
    return
```

---

## Bug D 🟡 nit — `simple_rllib_test.yaml` core_type mismatch

**Symptom (live):**
```
WARNING: Federate 'federation_1.spring_federate' requested core_type 'zmq'
but the scenario protocol is 'tcp'. Overriding it for consistency...
```
Runs fine (auto-override), but spurious warning every run. Copy-paste artifact.

**Fix (YAML)** in `simple_rllib_test.yaml` — set `spring_federate.core_type: "tcp"` (match broker) or delete the field so it inherits.

---

## Bug E 🟡 nit — `Adelaide_test.yaml` raw YAML syntax error

**Status:** pre-existing on `main`, **not** caused by the refactor (`git diff main...rl-config-refactor` shows no change to it).

**Symptom:** `YAMLError ... line 54, column 7` — file does not even parse.

**Fix:** repair the indentation at line 54, or delete the file if obsolete.

---

## Also worth fixing (not scenario YAML)

- **Federate-death not surfaced:** a federate subprocess crashing (bug A) leaves the manager printing "Completed" with exit code 0. Propagate non-zero federate exit to the driver.
- **Orphan cleanup on abnormal kill:** federates are spawned `preexec_fn=os.setsid` (own session); killing the parent (`Ctrl-C`, SIGKILL) leaves federates + brokers orphaned and still running (RL agents keep training, holding ports). Track child PIDs and tear down their sessions on shutdown/interrupt.

## Suggested smoke-test config

`bui_hp_*` / `pv_batt_*` train to completion (100 episodes × 144 steps) → minutes each. For a fast CI smoke pass, add short-run overrides (e.g. `run.train.episodes: 2`, `episode_length: 10`) or a dedicated `*_smoke.yaml` variant per backend.
