# Handoff 06 — Phase 6: Validation hardening + cleanup DONE

Branch `rl-config-refactor`.

## Delivered
- **`tests/test_rl_config.py`** — 51 tests (1 skipped: Adelaide_test.yaml broken YAML):
  - Parse gate: every scenario YAML validates under ScenarioConfig
  - RL scenarios: environment has non-empty observations/actions, agent has model_name
  - `extra='forbid'`: ObservationSpec, ActionSpec, AgentConfig, Hyperparameters, RunConfig,
    full ReinforcementLearningConfig all reject unknown keys
  - Structural validators: PhaseConfig.total_steps, test-only needs checkpoint,
    CheckpointConfig.best_path resolution, null spec coercion, ResetConfig defaults,
    Hyperparameters.as_kwargs omits None
- **CLAUDE.md** updated with new 4-axis RL config schema reference and available agents
- Dead code audit: no imports of old RL models (RLExplorationConfig etc.) remain

## Known carry-over (not addressed — out of scope or low risk)

### `rl_config` dict bridge (BaseFederate.py:93-101 + federate_launcher.py:119-127)
Base (non-RL) federates still consume a flat `rl_config` dict for episode/reset sync.
federate_launcher translates new schema → flat dict. Removing would require BaseFederate
to read typed config directly — invasive; works fine as-is.

### `rl_config: Optional[Dict]` on `_FederateConfigBase` (config_dataclasses.py:438)
Kept because BaseFederate reads it. Remove when the dict bridge is eliminated.

### Dead reset branches (BaseFederate.py:860 'soft', :875 'random')
New schema only allows `full|rolling|none`. These branches are unreachable from new config
but exist in BaseFederate for historical reasons. Safe to remove but untested.

### `role: extra` not wired end-to-end (RL_Federate.py)
Observations with `role: extra` are excluded from the obs space but HelicsGymEnv still
returns them in the obs dict. SB3/RLlib KeyError on unknown obs keys. Workaround: don't
use `role: extra` in scenarios until env splits policy vs reward obs.

### zmq multi-federation hierarchy broker
Fails deterministically. All new examples use `core_type: tcp`.

### numpy/matplotlib pins + LD_PRELOAD
ray 2.55.1 upgrades numpy/matplotlib beyond system GLIBCXX. Pinned:
- numpy<2.1 (2.0.2), matplotlib<3.10 (3.9.4)
- `conda env config vars set LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`
Document in Installation_Setup.md when merging to main.

### Tracked .pyc files
Multiple `__pycache__/*.pyc` tracked in git. Should be removed with
`git rm -r --cached '**/__pycache__'` and .gitignore already covers them.
