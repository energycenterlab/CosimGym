# Handoff 04 — Phase 4: Example 1 (BUI0 FMU set-point, SAC + DQN) DONE

Branch `rl-config-refactor`. Run from repo ROOT. Committed: `3960f93`.

## Delivered
- **Reward** `bui0_setpoint_comfort` in `reward_functions.py`: negative quadratic comfort
  penalty around 21 °C on `TBuilding`, with an optional energy term on `HeatingLoadTarget`
  (degrades gracefully to comfort-only when that obs is absent — see role:extra note).
- **`src/scenarios/bui0_setpoint_SAC.yaml`** — continuous box action on ZoneSetPoint [16,24].
- **`src/scenarios/bui0_setpoint_DQN.yaml`** — discrete action, 9 bins over [16,24].
  Identical MDP; differ only in `agent` + action `space`/`bins` (the headline simplicity demo).
- `test_script_rl.py` updated to point at the new examples.

## Verified (from ROOT, both green)
- SAC: "Federation completed ~14.6s", train (3×144) + test (144), train+test storage written,
  no failed federates. FMU driven by RL ZoneSetPoint.
- DQN: "Federation completed ~9.6s", episodes finish, train+test rl_storage written, no failures.

## Key decisions / fixes this phase
- **tcp brokers** (broker_config.core_type: tcp) — zmq multi-federation hierarchy still broken.
- **reset.mode: none** — the EnergyPlus FMU cannot re-initialize mid-run; `BaseFederate._reset`
  calls `model.reset(mode=...)` for full/rolling, but `BaseFMUModel.reset()` takes no args AND
  EnergyPlus can't restart. `none` makes episodes contiguous (agent still segments internally).
  The FMU happily ran 576 contiguous steps, so its RunPeriod is not the 1-day limit I feared.
- **DQNAgent.update hardened**: now requires `len(replay) >= max(min_replay_size, batch_size)`
  before sampling (a prefill < batch_size previously raised "Sample larger than population").
- Observation `bounds` override used to give SB3 finite obs bounds (catalog has none for the
  FMU outputs): `TBuilding.bounds: [0,40]`; action `bounds: [16,24]`.

## ‼ role:extra is NOT wired end-to-end (discovered here)
Setting an observation `role: extra` excludes it from the policy obs space (correct), but
`HelicsGymEnv` still RETURNS it in the obs dict passed to `model.predict`, so SB3 KeyErrors
(`observation_space.spaces[key]`). The env returns all subscribed inputs; it does not yet
split "policy obs (state)" from "reward-only obs (extra)". I removed the extra `HeatingLoadTarget`
obs from Example 1 to ship it. **Follow-up (Phase 6 or later):** in `HelicsGymEnv`
`_inputs_to_observations` / observation construction, return only state-role keys to the policy
and expose extra-role values to `compute_reward` via a side channel. Until then, do not use
`role: extra` in scenarios. (This is the long-standing §2.8 "additional_observations never
worked" gap.)

## Convergence note
Rewards are large-negative (~-30k/episode) — the cold January zone sits well off 21 °C and 3
episodes is far from convergence. Example 1 proves the PIPELINE (train→checkpoint→test, two
algorithms, FMU-in-the-loop), not policy quality. Bump episodes for real training.

## Next: Phase 5 — Example 2 (RLlib agent)
- Add `ray[rllib]` to environment.yml (Python 3.12-compatible; check vs gymnasium 1.2.3).
- New `rl_simple_rllib.py` (RLAgent subclass, composing components). Co-sim drives stepping, so
  use RLlib in an external/inference style via the `env_loop` seam, not RLlib's own rollout
  loop. Reference (don't import) `Rllib_wrapper_old.py`, `base_agent_rl_example_copilot.py`.
- Catalog entry `rl_simple_rllib`. New small scenario (tcp brokers; reuse the spring MDP for a
  light first RLlib test, or a bui0 variant).
- Verify a few train iters + a test rollout via HELICS.

## Carry-over risks (Phase 6)
- zmq multi-federation hierarchy broker (bui_hp_*/pv_batt_* still blocked on zmq).
- role:extra env wiring (above).
- BaseFMUModel.reset signature (`mode`/`ts` kwargs) if FMU full/rolling reset ever needed.
- Dead reset 'soft'/'random' branches; gitignore already covers pyc/logs but tracked pyc remain.
