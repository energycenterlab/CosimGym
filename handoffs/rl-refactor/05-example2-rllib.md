# Handoff 05 — Phase 5: Example 2 (RLlib PPO agent) DONE

Branch `rl-config-refactor`. Run from repo ROOT.

## Delivered
- **`src/models/model_catalog/RL_agents/rl_simple_rllib.py`** — RLlib PPO agent using
  standalone `DefaultPPOTorchRLModule` (no ray workers, no env serialization). Manual GAE +
  PPO clipped-surrogate update. Composes existing components: `CheckpointManager`,
  `DictKeyNameWrapper`, `SB3ActionWrapper`, `FlattenObservation`.
- **`src/scenarios/simple_rllib_test.yaml`** — spring-mass-damper MDP with PPO (3 train
  episodes × 50 steps, 1 test × 50 steps, tcp brokers).
- **Catalog entry** `rl_simple_rllib` in `catalog.yaml`.
- `test_script_rl.py` updated with Example 2 comment.

## Verified
- `simple_rllib_test`: "Federation completed in 7.587 seconds", 3 train episodes logged,
  200 total HELICS steps (150 train + 50 test), train+test rl_storage.json written.
  No errors.
- Existing SB3 scenarios unbroken: `simple_SACsb3_test` still passes after gymnasium 1.2.2
  downgrade + numpy 2.0.2 pin.

## Key decisions / fixes this phase

### ray[rllib] 2.55.1 dependency cascade
Installing ray 2.55.1 upgraded numpy to 2.4.2 and matplotlib to 3.10.9, both needing
`GLIBCXX_3.4.29` which this server only has up to 3.4.28. Fixed by:
- `pip install "numpy<2.1"` → numpy 2.0.2 (fmpy 0.3.29 warns but FMU runs fine)
- `pip install "matplotlib<3.10"` → matplotlib 3.9.4
- `conda env config vars set LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` — conda's
  libstdc++ (6.0.34) has GLIBCXX_3.4.29. Auto-applies on `conda activate cosim_gym`.

### Standalone RLModule (not Algorithm)
RLlib's `register_env` + `Algorithm.build_algo()` pickles the env factory for ray workers.
Our HelicsGymEnv contains HELICS CFFI objects that can't be pickled. Attempted workarounds
(old API stack, `create_env_on_local_worker`) all still hit serialization.

**Solution:** build `DefaultPPOTorchRLModule` standalone via `RLModuleSpec.build()`. Drive
the env manually in our own loop (like the DQN agent does). Implement GAE + PPO clipped
surrogate loss in ~80 lines of PyTorch. No ray workers at all — only imports the neural
network module class.

### API notes for ray 2.55.1
- `PPOConfig.api_stack()` → `enable_rl_module_and_learner`, `enable_env_runner_and_connector_v2`
  (not `enable_rl_module`)
- `Algorithm.build()` → deprecated, use `build_algo()`
- `compute_single_action()` → deprecated on new API stack; use `get_module().forward_inference()`
- `forward_inference` returns `{"action_dist_inputs": tensor}`, NOT `{"actions": ...}`.
  For Gaussian: first half = mean, second half = log_std. Use `TorchDiagGaussian.from_logits()`
  to sample. No `deterministic_sample()` — take mean directly for deterministic action.
- `forward_exploration` returns same format; use `get_exploration_action_dist_cls()`.
- `DefaultPPOTorchRLModule` is the current class name (not `PPOTorchRLModule`).
- `RLModuleSpec(model_config={"fcnet_hiddens": [64, 64]})` works for flat Box obs.
  **Dict obs not supported** by default catalog encoder — must flatten first.

## Architecture

```
RL_Simple_RLlib(RLAgent)
  ├── env: FlattenObservation(SB3ActionWrapper(DictKeyNameWrapper(HelicsGymEnv)))
  ├── module: DefaultPPOTorchRLModule  (standalone, no ray workers)
  ├── optimizer: Adam
  ├── act() → forward_inference + sample from TorchDiagGaussian
  ├── _collect_rollout() → N env steps, returns obs/act/rew/done/logp/values
  ├── _compute_gae() → advantages + returns
  ├── _ppo_update() → clipped surrogate + value loss + entropy, minibatch SGD
  ├── online_training_loop() → collect → GAE → PPO update loop
  ├── testing_loop() → deterministic inference loop
  └── _save_module/_load_module → torch state_dict
```

## Next: Phase 6 — Validation hardening, cleanup
- Confirm dead models removed, tighten validators, document offline/parallel seams
- Update CLAUDE.md + docs with new schema
- Add pytest test_rl_config.py
- Carry-over risks: zmq hierarchy broker, role:extra env wiring, BaseFMUModel.reset signature,
  dead reset 'soft'/'random' branches, tracked .pyc files
- Document numpy/matplotlib pin + LD_PRELOAD workaround in Installation_Setup.md
