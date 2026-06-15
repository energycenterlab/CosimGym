# Plan: Refactor of the `reinforcement_learning_config` Schema

**Status:** Proposal only — nothing implemented. This document is a future-work plan.
**Scope:** The RL configuration block of a scenario, its Pydantic models in
`src/utils/config_dataclasses.py`, and every site that reads those models.
**Goals (from request):** maximum generalization, compatibility with the conventions of
mainstream RL libraries (Gymnasium, Stable-Baselines3, RLlib, CityLearn), simplicity of
use, and elimination of redundancy.

---

## 1. What exists today

The RL schema is defined by `ReinforcementLearningConfig` and ten sibling models in
`src/utils/config_dataclasses.py:228-430`. The runtime shape is:

```
reinforcement_learning_config
├── seed
├── agent: RLAgentConfig
│   ├── model_name              # catalog key → concrete agent class
│   ├── algorithm               # (string)
│   ├── library                 # (string)
│   ├── reward_function         # dotted import path
│   ├── hyperparameters: RLHyperparametersConfig   # ~18 flat fields
│   └── env: RLEnvironmentConfig                    # 13 fields, mostly parallel arrays
├── training: RLTrainingConfig
│   ├── (episode_length, n_episodes, total_steps, mode, reset_mode, reset_period,
│   │    rolling_window, warmup_steps, train_frequency, gradient_steps,
│   │    eval_frequency, n_eval_episodes, eval_deterministic, log_interval, verbose, ...)
│   ├── exploration: RLExplorationConfig
│   ├── replay_buffer: RLReplayBufferConfig
│   ├── offline_config: RLOfflineTrainingConfig
│   └── early_stopping: RLEarlyStoppingConfig
├── checkpointing: RLCheckpointingConfig
├── logging: RLLoggingConfig
└── test: RLTestConfig
```

The effort to enumerate "every standard RL parameter" is visible and well-intentioned, but
the schema has drifted ahead of the code: only a small subset of these fields is actually
consumed at runtime, and the parts that *are* consumed are split across blocks that do not
match how the libraries — or the code itself — think about the problem.

---

## 2. Findings (evidence-based)

### 2.1 Large dead surface — schema promises more than the code reads

A grep over `src/**/*.py` for every `rl_task.<...>` access shows that **only ~12 distinct
config paths are ever read** across all agents and the orchestrator:

```
rl_task.training.total_steps        rl_task.training.replay_buffer
rl_task.test.total_steps            rl_task.training.exploration
rl_task.training.train_frequency    rl_task.training.gradient_steps
rl_task.test.deterministic          rl_task.agent.hyperparameters
rl_task.seed                        rl_task.checkpointing.single_best_checkpoint
rl_task.agent.reward_function       rl_task.agent.env.*  (observations/actions/...)
```

Fields that **no code reads** (aspirational schema): `agent.algorithm`, `agent.library`,
all of `RLHyperparametersConfig` except `learning_rate/gamma/batch_size/gradient_clip/
target_update_interval` (so `net_arch`, `activation_fn`, `optimizer`, `n_epochs`,
`ent_coef`, `vf_coef`, `gae_lambda`, `clip_range`, `normalize_advantages`, `tau`,
`use_sde`, `algorithm_kwargs` are unused), all of `RLEarlyStoppingConfig`, all of
`RLOfflineTrainingConfig`, all of `RLLoggingConfig`, most of `RLCheckpointingConfig`
(`save_best`, `best_metric`, `keep_last_n`, `save_frequency`, `save_replay_buffer`,
`single_best_checkpoint` is the only one used), and `training.warmup_steps`,
`training.eval_frequency`, `training.n_eval_episodes`, `training.eval_deterministic`,
`training.verbose`, `training.log_interval`.

> A schema field that nothing reads is worse than absent: it tells the user a knob exists,
> they set it, and nothing happens silently. `model_config = extra='ignore'` makes this
> failure mode invisible.

### 2.2 `algorithm` / `library` are structurally dead

The agent is selected by `model_name` → catalog → a concrete Python class
(`RL_Federate._register_entities`, `src/core/RL_Federate.py:397-402`). That class hard-codes
both the library and the algorithm (e.g. `rl_simple_SACsb3.py` imports `SAC` from
`stable_baselines3`). So `agent.algorithm` and `agent.library` are pure documentation; they
cannot influence behavior. Compatibility with a second backend (RLlib) today means *writing a
whole new agent class*, not changing config.

### 2.3 Two parallel RL-info channels: `rl_task` vs `rl_config`

`_FederateConfigBase` carries **both** `rl_task: ReinforcementLearningConfig` and
`rl_config: Optional[Dict]` (`config_dataclasses.py:449,453`). RL federates use `rl_task`;
**base** federates get a *separately re-assembled dict* `rl_config` built in
`federate_launcher.py:106-117` by copying `training.reset_period/reset_mode/episode_length/
n_episodes/rolling_window` into a flat dict, which `BaseFederate.__init__` then reads
(`BaseFederate.py:94-103`). The same five facts therefore live in two shapes and are plumbed
twice. This is the source of the `reset_type` vs `reset_mode` naming drift.

### 2.4 The solver is fragmented across three blocks

To build a SAC model, `rl_simple_SACsb3.py:75-86` must reach into:

- `rl_task.agent.hyperparameters` (learning rate, gamma, batch size, target update),
- `rl_task.training.replay_buffer` (buffer size, prefill),
- `rl_task.training.train_frequency` / `gradient_steps`.

Algorithm knobs are split between `agent.*` and `training.*`. Conceptually they all belong to
the *solver*. Both RLlib (`AlgorithmConfig.training(...)`) and SB3 (single algorithm
constructor) keep them together.

### 2.5 Kitchen-sink typed hyperparameters + redundant escape hatch

`RLHyperparametersConfig` flat-lists params from PPO **and** DQN **and** SAC/TD3
(`n_epochs`, `clip_range`, `gae_lambda` are PPO-only; `target_update_interval`, `tau` are
off-policy-only; `ou_theta/ou_sigma` in `RLExplorationConfig` are DDPG-only). These are
mutually exclusive per algorithm, so any given scenario leaves most of them meaningless. The
model *also* has `algorithm_kwargs: Dict` — so there are already two mechanisms for the same
purpose, and the typed one will always lag the libraries.

### 2.6 Non-`None` defaults silently override library-tuned defaults

`RLHyperparametersConfig` sets concrete defaults (`learning_rate=0.0003`, `gamma=0.99`,
`batch_size=64`, `normalize_advantages=True`, ...). Because they are never `None`, when an
agent forwards them to SB3 it **overrides SB3's per-algorithm tuned defaults even for knobs
the user never touched**. Example: SB3 SAC's default `learning_rate` is `3e-4` but DQN's is
`1e-4`; a single shared default cannot be right for both, and the config forces one. The
correct pattern is `None`-default → omit from the constructor → let the backend apply its own
default.

### 2.7 Observation/action spec uses fragile positional parallel arrays

`RLEnvironmentConfig` (`config_dataclasses.py:330-345`) encodes the MDP as up to **eight
index-aligned lists**:

```yaml
observations:                  [a, b, c]
observation_causality:         [same_step, next_step, same_step]
additional_observations:       [...]
additional_observation_causality: [...]
include_prev_obs:              [0, 0, 0]
actions:                       [x]
action_spaces_type:            [box]
action_bins:                   [null]
action_boundaries:             [[0,1]]
```

Every consumer indexes these in lockstep: `RL_Federate._prepare_obs_dict` uses `prev_obs[i]`
(`RL_Federate.py:440`), `_prepare_act_dict` uses `act_spaces_type[i]`, `action_bins[i]`,
`action_boundaries[i]` (`RL_Federate.py:464-512`), `ScenarioManager._resolve_observation_
causality` indexes `causality_list[index]` (`ScenarioManager.py:380-383`). A length mismatch
is only **warned** about at runtime (`ScenarioManager.py:390-405`), never rejected, and
silently defaults to `same_step`. This is an entire class of order-coupling bugs that a keyed
mapping eliminates structurally. (Note: the type is `Union[List, Dict]` but only the list
branch is implemented anywhere.)

### 2.8 `additional_observations` is half-wired

Additional observations are subscribed, added to `required_inputs`, and given reset defaults
(`ScenarioManager.py:423-438, 552-576`), but they are **not** placed in the agent observation
space: `_prepare_obs_dict` builds the space only from `self.config.observed_models`
(main observations), and `_inputs_to_observations` iterates only
`observation_space.spaces.keys()` (`RL_Federate.py:407-446, 616`). So today they are received
but never surfaced to the policy *or* the reward function. The code comment confirms it:
`# TODO: check if additional works (never tried)` (`ScenarioManager.py:422`). The *intent* —
"variables the reward/logging can see but the policy should not" — is valuable but currently
non-functional and needs an explicit, working representation.

### 2.9 Reset / episode semantics live in four places

Reset behavior is spread across `env.reset_observation_defaults` +
`env.force_reset_observation_defaults` (`RLEnvironmentConfig`), `training.reset_mode/
reset_period/rolling_window` (`RLTrainingConfig`), the runtime-injected
`_FederateConfigBase.reset_observation_defaults`, and the base-federate `rl_config` dict
(§2.3). There is no single place that answers "how does an episode reset?".

### 2.10 "How long does it run" has no single source of truth

`RLTrainingConfig` derives `total_steps = n_episodes * episode_length`
(`config_dataclasses.py:321-327`), but `RLTestConfig.total_steps` is a **separate, required,
non-derived** field, while `RLTestConfig.n_episodes` and `RLTestConfig.episode_length` exist
and are unused. Meanwhile "evaluation" is expressed three different ways: `training.
eval_frequency/n_eval_episodes/eval_deterministic`, the standalone `test` block, and
`early_stopping.metric`. A reader cannot tell which one fires.

### 2.11 Inverted ownership: the MDP is nested under the solver

The environment definition (`env`: observations, actions, reward, reset) lives **under
`agent`** (`agent.env`). But the MDP is the framework-agnostic problem that CosimGym owns and
that any backend must solve; the agent is the swappable solver. Nesting problem-under-solver
is backwards and is exactly the split RLlib and CityLearn keep apart (`environment()` vs
`training()`; schema `observations/actions/reward_function` vs `agent`).

### 2.12 Weak validation

`ReinforcementLearningConfig` validates only "training or test present"
(`config_dataclasses.py:426-430`). Nothing checks: array lengths agree, `action_bins` is set
when discretizing a continuous variable, a `checkpoint` exists for a test-only run, or that
`backend`+`algorithm` are a known pair. Because all models use `extra='ignore'`, typos in key
names are dropped silently.

---

## 3. What the libraries actually do (research basis)

- **Gymnasium** — the unit of generalization is the *space* (`Box`, `Discrete`,
  `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple`) plus the `Env` API. There is no global
  "list of hyperparameters"; the environment defines observation/action *spaces* and the
  algorithm is separate. `build_space` in `RL_Federate.py:25-60` already speaks this language.
- **Stable-Baselines3** — one algorithm class per file (`SAC`, `PPO`, `DQN`, `TD3`, ...),
  configured by its **own** constructor kwargs, with `policy_kwargs` (e.g. `net_arch`,
  `activation_fn`) as the structured escape hatch. Run length is a single `learn(total_
  timesteps=...)`. Eval/checkpoint are `callbacks`, not core config.
- **RLlib** — `AlgorithmConfig` groups settings by *purpose* via builder methods:
  `.environment()`, `.training(lr, gamma, train_batch_size, ...)`, `.rollouts()`,
  `.evaluation(evaluation_interval, evaluation_num_episodes, ...)`, `.framework()`,
  `.resources()`, `.multi_agent()`. Type-safe universal core + algorithm-specific kwargs
  passed through `.training(...)`. ([AlgorithmConfig API](https://docs.ray.io/en/latest/rllib/algorithm-config.html))
- **CityLearn** — a `schema.json` separates `observations` and `actions` (each a **mapping**
  keyed by variable name with per-variable `active` flags), a pluggable `reward_function`
  class, a pluggable `agent`, and a `central_agent` toggle for the single-vs-multi-agent
  axis. ([CityLearn reward function](https://www.citylearn.net/overview/reward_function.html),
  [citylearn module](https://www.citylearn.net/api/citylearn.citylearn.html))

**Common denominator across all four:** separate (a) the *environment/problem* from (b) the
*algorithm/solver*, keep (c) *run schedule / evaluation* as its own axis, type only the
*universal* knobs and pass the rest through, and describe per-variable observation/action
spec as a **mapping**, not parallel arrays.

---

## 4. Proposed structure

Four top-level blocks under `reinforcement_learning_config`, each mapping to one of the axes
above. The YAML key name is kept for continuity; the inner shape changes.

```yaml
reinforcement_learning_config:
  seed: 42

  # (a) ENVIRONMENT — the MDP. Framework-agnostic. CosimGym owns this.
  environment:
    observations:                       # MAPPING keyed by dotted path (was 4 parallel arrays)
      federation_1.weather.0.T_ext:
        causality: same_step            # default same_step      (was observation_causality[i])
        history: 0                      # frame-stack depth       (was include_prev_obs[i])
        reset_default: 3.0              #                         (was reset_observation_defaults[k])
        role: state                     # state | extra           (replaces additional_observations)
      federation_1.building.0.T_indoor:
        causality: next_step
        reset_default: 18.0
        # space/bounds auto-derived from the model catalog; override only if needed:
        # space: box
        # bounds: [low, high]

    actions:                            # MAPPING keyed by dotted path (was 4 parallel arrays)
      federation_1.heatpump.0.modulation:
        space: box                      # box|discrete|multidiscrete|multibinary (was action_spaces_type[i])
        bounds: [0.0, 1.0]              # override catalog        (was action_boundaries[i])
        # bins: 5                       # only to discretize a continuous var (was action_bins[i])

    reward: models.model_catalog.RL_agents.reward_functions.building_heatpump_comfort
    termination: null                   # optional dotted path → terminated(obs, action, t) -> bool

    reset:                              # single home for episode-reset semantics (was 4 places)
      mode: full                        # full | rolling | none   (was training.reset_mode)
      period: 2880                      # defaults to run.train.episode_length (was training.reset_period)
      rolling_window: null              # (was training.rolling_window)
      force_defaults: false             # (was env.force_reset_observation_defaults)

  # (b) AGENT — the solver. Library-specific, mostly pass-through.
  agent:
    model_name: rl_simple_SACsb3        # catalog key → concrete agent class (unchanged)
    backend: stable_baselines3          # informational + future dispatch (replaces dead library/algorithm)
    algorithm: SAC
    policy: MultiInputPolicy
    hyperparameters:                    # SMALL universal core; every field None-default → omit when unset
      learning_rate: 0.001
      gamma: 0.99
      batch_size: 64
      net_arch: [256, 256]
      train_frequency: 100
      gradient_steps: 1
    params:                             # backend-specific escape hatch (replaces the kitchen sink)
      buffer_size: 100000
      learning_starts: 2880
      ent_coef: auto
      target_update_interval: 5

  # (c) RUN — what to execute. Single source of truth for length. (was training + test + eval fields)
  run:
    mode: online                        # online | offline | mixed
    train:
      episodes: 100
      episode_length: 2880
      # total_timesteps = episodes * episode_length  (derived; the only length source)
    eval:                               # optional periodic eval during training (was training.eval_*)
      every_steps: 10000
      episodes: 10
      deterministic: true
    test:                               # final test phase
      episodes: 1
      episode_length: 2880
      deterministic: true
      checkpoint: null                  # null → best produced by train

  # (d) EXPERIMENT — orthogonal infrastructure. (was checkpointing + logging + offline_config)
  experiment:
    name: bui_hp_SAC
    checkpoint:
      dir: src/models/model_catalog/RL_agents/checkpoints
      best: best_sac_sb3_model.pth
    logging:
      backend: tensorboard
      project: cosim_gym
    offline:                            # only when run.mode in {offline, mixed}
      dataset_path: ...
```

### 4.1 Pydantic model sketch

```python
# (a) ENVIRONMENT
class ObservationSpec(BaseModel):
    causality: Literal["same_step", "next_step"] = "same_step"
    history: int = 0
    reset_default: Optional[float] = None
    role: Literal["state", "extra"] = "state"        # extra = visible to reward/log, not to policy
    space: Optional[str] = None                       # override; else from catalog
    bounds: Optional[Tuple[float, float]] = None

class ActionSpec(BaseModel):
    space: Literal["box", "discrete", "multidiscrete", "multibinary"] = "box"
    bounds: Optional[Tuple[float, float]] = None
    bins: Optional[int] = None

class ResetConfig(BaseModel):
    mode: Literal["full", "rolling", "none"] = "full"
    period: Optional[int] = None                      # defaults to run.train.episode_length
    rolling_window: Optional[int] = None
    force_defaults: bool = False

class EnvironmentConfig(BaseModel):
    observations: Dict[str, ObservationSpec]
    actions: Dict[str, ActionSpec]
    reward: Optional[str] = None
    termination: Optional[str] = None
    reset: ResetConfig = Field(default_factory=ResetConfig)

# (b) AGENT
class Hyperparameters(BaseModel):                     # ALL Optional → omit when unset (§2.6)
    learning_rate: Optional[float] = None
    gamma: Optional[float] = None
    batch_size: Optional[int] = None
    net_arch: Optional[List[int]] = None
    train_frequency: Optional[int] = None
    gradient_steps: Optional[int] = None

class AgentConfig(BaseModel):
    model_name: str
    backend: Optional[str] = None
    algorithm: Optional[str] = None
    policy: Optional[str] = None
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    params: Dict[str, Any] = Field(default_factory=dict)

# (c) RUN
class PhaseConfig(BaseModel):
    episodes: int
    episode_length: int
    deterministic: bool = False
    checkpoint: Optional[str] = None
    @property
    def total_steps(self) -> int: return self.episodes * self.episode_length

class EvalConfig(BaseModel):
    every_steps: Optional[int] = None
    episodes: int = 10
    deterministic: bool = True

class RunConfig(BaseModel):
    mode: Literal["online", "offline", "mixed"] = "online"
    train: Optional[PhaseConfig] = None
    eval: Optional[EvalConfig] = None
    test: Optional[PhaseConfig] = None

# (d) EXPERIMENT
class CheckpointConfig(BaseModel):
    dir: str = "src/models/model_catalog/RL_agents/checkpoints"
    best: Optional[str] = None

class ExperimentConfig(BaseModel):
    name: Optional[str] = None
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: Optional[Dict[str, Any]] = None
    offline: Optional[Dict[str, Any]] = None

# ROOT
class ReinforcementLearningConfig(BaseModel):
    seed: Optional[int] = None
    environment: EnvironmentConfig
    agent: AgentConfig
    run: RunConfig
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
```

### 4.2 Old → new field map

| Old path | New path |
|---|---|
| `agent.env.observations[i]` + `observation_causality[i]` + `include_prev_obs[i]` + `reset_observation_defaults[k]` | `environment.observations.<key>.{causality,history,reset_default}` |
| `agent.env.additional_observations` | `environment.observations.<key>.role: extra` |
| `agent.env.actions[i]` + `action_spaces_type[i]` + `action_bins[i]` + `action_boundaries[i]` | `environment.actions.<key>.{space,bins,bounds}` |
| `agent.env.force_reset_observation_defaults` | `environment.reset.force_defaults` |
| `training.reset_mode / reset_period / rolling_window` | `environment.reset.{mode,period,rolling_window}` |
| `agent.reward_function` | `environment.reward` |
| `agent.hyperparameters.*` (used subset) | `agent.hyperparameters.*` |
| `agent.hyperparameters.*` (unused) + `algorithm_kwargs` + `training.exploration/replay_buffer` | `agent.params.*` |
| `agent.algorithm`, `agent.library` (dead) | `agent.algorithm`, `agent.backend` (now used for dispatch) |
| `training.{episode_length,n_episodes,total_steps}` | `run.train.{episode_length,episodes}` (+derived `total_steps`) |
| `training.{eval_frequency,n_eval_episodes,eval_deterministic}` | `run.eval.{every_steps,episodes,deterministic}` |
| `test.{total_steps,n_episodes,episode_length,deterministic,checkpoint_path}` | `run.test.{episodes,episode_length,deterministic,checkpoint}` |
| `training.mode` | `run.mode` |
| `checkpointing.{directory,single_best_checkpoint}` | `experiment.checkpoint.{dir,best}` |
| `logging.*` | `experiment.logging.*` |
| `offline_config.*` | `experiment.offline.*` |
| `early_stopping.*` | dropped until implemented (move to `run.eval` when wired) |
| `_FederateConfigBase.rl_config` (dict) | removed — base federate reads `environment.reset` + `run.train` (§2.3) |

**Net effect:** ~70 enumerated fields across 10 models collapse to ~30 meaningful fields
across 4 axes, the 8 positional arrays become 2 keyed mappings, and the "kitchen sink" is
replaced by a 6-field typed core plus `params` pass-through.

---

## 5. Drawbacks & code to fix after implementation

These are the breakages this refactor introduces; each must be updated in lockstep with the
schema change.

### 5.1 Observation/action space builders (positional → mapping)
- `RL_Federate._prepare_obs_dict` (`RL_Federate.py:407-446`) iterates `observations` list and
  indexes `prev_obs[i]`; rewrite to iterate `environment.observations.items()` and read
  per-entry `history`, `causality`, `reset_default`, `role`. Filter `role == "extra"` out of
  the observation space but keep them available to the reward function.
- `RL_Federate._prepare_act_dict` (`RL_Federate.py:449-522`) indexes `act_spaces_type[i]`,
  `action_bins[i]`, `action_boundaries[i]`; rewrite to read `ActionSpec` per key.

### 5.2 Orchestrator RL wiring (ScenarioManager)
- `_get_rl_pubsubs` (`ScenarioManager.py:385-451`), `_get_rl_controlled_models`
  (`:453-457`), `_build_rl_reset_observation_defaults` (`:459-510`), `_create_RL_federation`
  (`:524-582`) all consume the list form of `observations` / `additional_observations` /
  `observation_causality`. Rewrite against the mapping; the four near-duplicate
  `additional_observations` blocks collapse into a single pass filtered by `role`.
- `_resolve_observation_causality` (`:380-383`) becomes a per-entry read; delete the
  length-mismatch warnings (`:390-405`) — the class of bug no longer exists.

### 5.3 Federate launcher & BaseFederate (remove the `rl_config` channel)
- `federate_launcher.py:106-117` builds the flat `rl_config` dict from `training.reset_*`;
  repoint to `environment.reset.*` + `run.train.*`, or remove if BaseFederate reads the model
  directly.
- `BaseFederate.__init__` (`BaseFederate.py:94-103`) reads `self.config.rl_config.get(...)`;
  repoint to the new reset block. The rolling-window logic (`BaseFederate.py:860-869`) reads
  `self.rolling_window` → source it from `environment.reset.rolling_window`.
- Remove `_FederateConfigBase.rl_config` once both sites are migrated.

### 5.4 Agent classes (solver config relocation)
- `rl_simple_SACsb3.py:75-86,106-114,151,156` reads `rl_task.agent.hyperparameters`,
  `rl_task.training.replay_buffer`, `rl_task.training.train_frequency/gradient_steps`,
  `rl_task.training.total_steps`, `rl_task.test.total_steps`,
  `rl_task.checkpointing.single_best_checkpoint`. Repoint to `agent.hyperparameters` +
  `agent.params`, `run.train.total_steps` (derived), `run.test.total_steps`,
  `experiment.checkpoint.best`.
- `rl_simple_DQN.py:174-200,225,263` reads `rl_task.agent.hyperparameters`,
  `rl_task.training.exploration`, `rl_task.training.replay_buffer`,
  `rl_task.training.total_steps`, `rl_task.test.total_steps`,
  `rl_task.checkpointing.single_best_checkpoint`. The DQN agent's `exploration` (epsilon
  schedule) moves into `agent.params` (it is a DQN-specific solver knob); update accordingly.
- `base_agent_rl.py:73-75` reads `rl_task.agent.reward_function` → `environment.reward`;
  `:164,195` read `rl_task.training/test.total_steps`; `:208-211` read
  `rl_task.checkpointing.directory`. Repoint all.

### 5.5 RL_Federate run/init
- `RL_Federate.run` (`:561,568`) branches on `rl_task.training.mode` and `rl_task.test`;
  repoint to `run.mode` and `run.test`.
- `RL_Federate.__init__` (`:241-247`) reads `training.episode_length`, `training.n_episodes`,
  `env.force_reset_observation_defaults`; repoint to `run.train.*` and
  `environment.reset.force_defaults`. `_compute_terminated` (`:691-696`) uses
  `self.episode_length` — fine once repointed.

### 5.6 Validators to re-home
- `RLTrainingConfig._set_derived_fields` → derived `total_steps` becomes `PhaseConfig.
  total_steps` property.
- `RLCheckpointingConfig._build_checkpoint_path` → re-home onto `CheckpointConfig`.
- `RLTestConfig._normalize_none_like` → re-home onto `PhaseConfig.checkpoint`.

### 5.7 All scenario YAMLs must be migrated
~14 RL scenarios in `src/scenarios/` (`simple_DQN_test`, `bui_hp_SAC`, `bui_hp_DQN`,
`pv_batt_SAC`, `pv_batt_DQN`, the `*_rollingreset` variants, `simple_SACsb3_test`,
`simple_test_rlagent`, ...) use the old shape. Provide a one-shot converter script (old dict →
new dict) **or** a Pydantic `model_validator(mode='before')` compatibility shim that rewrites
legacy keys to the new shape for one release, logging a deprecation warning. The shim is the
lower-risk path and lets old and new YAMLs coexist during migration.

### 5.8 Behavioral-change risk (must call out to the user)
Switching `Hyperparameters` to `None`-defaults (§2.6) means scenarios that *relied on* the old
forced defaults (`learning_rate=0.0003`, `gamma=0.99`, `batch_size=64`, ...) will now get the
**library's** defaults instead. **Training results will not reproduce bit-for-bit after the
change.** Either (a) accept and re-baseline, or (b) keep the current explicit defaults in the
*scenario files* during migration so behavior is pinned by config rather than by schema.

### 5.9 Catalog check
Verify no `catalog.yaml` RL-agent entry references the renamed paths (grep shows none today,
but confirm after edits).

---

## 6. Suggested phasing (low-risk order)

1. **Add new models alongside old** with a `model_validator(mode='before')` shim that maps
   legacy keys → new keys. Both shapes validate. No runtime reads change yet. (Reversible.)
2. **Migrate read sites** (§5.1-5.6) to the new model, one component at a time, running the
   existing scenarios after each (they still validate via the shim).
3. **Migrate scenario YAMLs** to the new shape (§5.7); pin hyperparameters in-file to avoid
   §5.8 surprises.
4. **Make `Hyperparameters` None-default** and re-baseline training, or decide to keep pinned
   defaults in YAML.
5. **Remove the shim and the dead models** (`RLExplorationConfig`, `RLReplayBufferConfig`,
   `RLEarlyStoppingConfig`, `RLOfflineTrainingConfig`, `RLLoggingConfig`, the flat
   `RLHyperparametersConfig`, `_FederateConfigBase.rl_config`) once nothing references them.
6. **Add the validation** (§2.12): action `bins` required when discretizing, checkpoint
   present for test-only runs, known `backend`/`algorithm` pairs, `extra='forbid'` on the new
   models so typos fail loudly instead of being dropped.

---

## 7. Open questions for the maintainer (decide before step 1)

1. **Multi-agent axis.** CityLearn's `central_agent` and RLlib's `.multi_agent()` are first-
   class. The current code is single-agent (`# only single model for now`, `RL_Federate.py:396`).
   Should the new `agent` block reserve a `multi_agent` / per-controlled-model structure now,
   or stay single-agent and add it later? This affects whether `agent` is one block or a map.
2. **`extra='ignore'` vs `'forbid'`.** Ignoring unknown keys hides typos (§2.12). Switch the
   new RL models to `forbid`? (Recommended, but it is a stricter contract.)
3. **Backend dispatch.** Should `agent.backend` actually select a generic SB3/RLlib adapter
   (so one agent class serves many algorithms), or remain purely informational with one class
   per algorithm as today? This decides whether `agent.params` is forwarded generically.
4. **`history` / frame-stacking** is declared (`include_prev_obs`) but unimplemented
   (`RL_Federate.py:440`, `# TODO not implemented`). Keep it in the schema as a planned field,
   or drop until built?

---

## Sources

- [RLlib AlgorithmConfig API — Ray docs](https://docs.ray.io/en/latest/rllib/algorithm-config.html)
- [CityLearn — Reward Function](https://www.citylearn.net/overview/reward_function.html)
- [CityLearn — citylearn.citylearn module (central_agent, observations/actions)](https://www.citylearn.net/api/citylearn.citylearn.html)
