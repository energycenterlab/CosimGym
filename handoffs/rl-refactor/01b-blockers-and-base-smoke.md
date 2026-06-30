# Handoff 01b — Phase 1.5: blockers resolved + base-smoke baseline (DONE)

Branch `rl-config-refactor`.

## ‼ CRITICAL OPERATIONAL RULE (caused most of the false alarms)
**Run ALL simulations from the REPO ROOT**, never from `src/`. The federate subprocess loads
`src/core/mappings.yaml` via a path relative to CWD (`federate_launcher.py:79`), and logs are
written to `./logs/` relative to CWD. Running from `src/` makes federates die with
`FileNotFoundError: 'src/core/mappings.yaml'` (exit code 1) — which looks like an FMU/federate
bug but is just wrong CWD. Canonical invocation:
```
cd /media/space/rando/CODE/CosimGym
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate cosim_gym
python -c "import sys; sys.path.insert(0,'src'); from core.ScenarioManager import main; main('<scenario>')"
# logs land in ./logs/<scenario>/<ts>/
```
(`python src/test_script.py` also works — it's already root-relative.)

## Changes made (both in `src/core/ScenarioManager.py`)
1. **Core names auto-unique** (~line 1297). Per maintainer: each federate needs its own HELICS
   core; core_name must be globally unique; don't force YAML to specify it and don't fail on a
   wrong/duplicate value. New logic: keep the YAML core_name only if free, else fall back to the
   (unique-per-federation) federate name, else `{federation}_{federate}`, else suffix `_N`.
   Never raises; logs an info line when it repairs a collision. Replaces the old
   `raise ValueError("Duplicate HELICS core_name ...")`.
2. **Federate stdout/stderr captured to file** (~line 1553, `_create_local_federate`). Was
   `stdout=PIPE, stderr=PIPE` and never drained → tracebacks lost + latent deadlock. Now
   `stdout=open(<...>/federate_<name>.stdio.log,'wb'), stderr=STDOUT`. Uncaught federate
   exceptions are now visible in `logs/<scn>/<ts>/federates/federate_<name>.stdio.log`.

## "FMU bug" — there was none
`bui0_fmu_test` runs fine from root (completes ~2.5s), exactly as maintainer said. The earlier
code-1 was the wrong-CWD issue above.

## Base-smoke baseline (maintainer's 6 approved bare-co-sim scenarios, run from root)
| scenario | result |
|---|---|
| pv_batt_test_base | ✅ PASS (~3.9s, 5 feds) |
| bui0_fmu_test | ✅ PASS (~2.5s) |
| pandapipes_grid_test_base | ✅ PASS (~1.6s) |
| pandapower_grid_test_base | ✅ PASS (~1.5s) |
| rc_building_test_base | ✅ PASS (~2.6s) |
| fmu_feedthrough_test | ❌ FAIL — pre-existing **hardcoded macOS FMU path** in catalog/scenario: `/Users/pietrorandomazzarino/.../Feedthrough_FMI3.fmu`. Not RL-related, not ours. Fix later = make the FMU path relative. |

**5/6 pass** — solid bare-co-sim baseline before runtime migration. Use this set (minus
fmu_feedthrough until its path is fixed) as the base regression gate at the end of Phase 2/3.

## Scenarios that the maintainer says to IGNORE as tests
`simple_test`, `bui_hp_test_base` — "many problems"; do not use as gates. (They also hit the
old duplicate-core_name path before the fix.)

## Next: Phase 2 — migrate runtime read sites (mapping schema)
Surface already mapped (file:line current as of now):
- `RL_Federate._prepare_obs_dict` (407–446): `obs_list=rl_task.agent.env.observations`,
  `prev_obs=...include_prev_obs`, indexes `prev_obs[i]`. → iterate
  `rl_task.environment.observations.items()`; per-entry `history`/`causality`/`reset_default`/
  `role`; **filter `role=='extra'` OUT of obs space**, keep for reward. Space still derived from
  catalog via `_get_io_specs`; honor `ObservationSpec.space`/`bounds` overrides if set.
- `RL_Federate._prepare_act_dict` (449–522): reads `actions`, `action_spaces_type[i]`,
  `action_boundaries[i]`, `action_bins[i]`. → read `ActionSpec` per key (`space`/`bounds`/
  `bins`). **Enforce here**: if catalog var is float (continuous) and `space=='discrete'` and
  `bins is None` → raise (the check deferred from config, see handoff 01).
- `RL_Federate.__init__` (241–247): `training.episode_length/n_episodes`,
  `env.force_reset_observation_defaults` → `run.train.episode_length/episodes`,
  `environment.reset.force_defaults`. Note `self.reset_observation_defaults` still comes from
  injected `self.config.reset_observation_defaults` (built by ScenarioManager).
- `RL_Federate.run` (561, 568): `training.mode`/`training`/`test` → `run.mode`/`run.train`/
  `run.test`. `_compute_truncated` (591) keeps passing `rl_task`.
- `ScenarioManager._get_rl_pubsubs` (385–451): rewrite to iterate
  `environment.observations.items()` (read `.causality`, `.role`) + `environment.actions`
  (mapping keys). Collapse the additional-obs block into one `role=='extra'` pass. Delete the
  length-mismatch warnings (390–405) and `_resolve_observation_causality` (~380).
- `ScenarioManager._get_rl_controlled_models` (453–457): `rl_task.environment.actions` keys.
- `ScenarioManager._build_rl_reset_observation_defaults` (459–510): pull per-obs
  `reset_default` from `environment.observations[k].reset_default` (was
  `env.reset_observation_defaults` dict); iterate mapping keys; merge in extras by role.
- `ScenarioManager._create_RL_federation` (524–582) + `_modify_config_for_online_training`/
  `_for_testing` (591–624): `rl_task.environment.observations/actions` (mapping),
  `run.train.total_steps` / `run.test.total_steps` (PhaseConfig `.total_steps` property),
  `run.mode`.
- `federate_launcher.py` (106–121): drop the `rl_4_fed`/`rl_config` dict. For base federates,
  pass the reset/run facts from new schema: mode = train if `run.train` and `run.mode!='offline'`
  else test; episode_length=`run.train.episode_length`; n_episodes=`run.train.episodes`;
  reset_type=`environment.reset.mode`; reset_period=`environment.reset.period or episode_length`;
  rolling_window=`environment.reset.rolling_window`. Decide: keep a slim `rl_config` dict OR add
  a typed field. Simplest: keep building a small dict from the NEW paths so BaseFederate barely
  changes.
- `BaseFederate.__init__` (94–103): currently reads `self.config.rl_config.get(...)`. NOTE line
  95 (`self.mode = ...get('mode','test')`) is UNGUARDED — relies on launcher always setting
  rl_config. Keep that contract (launcher still passes a dict) OR guard for None. rolling-window
  reset logic at 860–873 reads `self.reset_type`/`self.rolling_window` — unchanged once sourced.
  Reset modes in new schema = full|rolling|none; old 'soft'/'random' branches (860,875) are now
  dead (converter maps them to full) — leave or prune in Phase 6.

Agents (`base_agent_rl.py`, `rl_simple_SACsb3.py`, `rl_simple_DQN.py`) still read OLD paths →
they break until Phase 3. Expected. Gate Phase 2 on parse-gate + a first-step RL launch.

## Open risks / notes
- Stray tracked `src/utils/__pycache__/*.pyc` churns; gitignore + `git rm --cached` in Phase 6.
- New `.stdio.log` file handle in `_create_local_federate` is never explicitly closed (process
  outlives it; OS reclaims at exit). Fine for now; could track+close in cleanup later.
- fmu_feedthrough hardcoded path: fix to relative when convenient (not RL scope).
