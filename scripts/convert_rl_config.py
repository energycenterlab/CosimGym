#!/usr/bin/env python3
"""
convert_rl_config.py — one-shot migrator for the RL config refactor.

Rewrites a scenario YAML's `reinforcement_learning_config:` block from the legacy shape
(parallel arrays under agent.env, training/test/checkpointing/logging blocks) to the new
four-axis shape (environment / agent / run / experiment). See
plan_for_rl_config_refactor.md §4.2 for the field map.

Only the top-level `reinforcement_learning_config:` block is spliced out and replaced; the
rest of the file (federations, comments) is preserved verbatim.

Usage:
    python scripts/convert_rl_config.py src/scenarios/bui_hp_SAC.yaml          # in place
    python scripts/convert_rl_config.py src/scenarios/*.yaml                   # batch, in place
    python scripts/convert_rl_config.py src/scenarios/bui_hp_SAC.yaml --out /tmp/x.yaml
    python scripts/convert_rl_config.py src/scenarios/bui_hp_SAC.yaml --dry-run

Idempotent: a file already in the new shape (has environment/run keys) is left untouched.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import yaml

# Infer backend/algorithm from the legacy catalog model_name (informational only).
_BACKEND_BY_MODEL = {
    "rl_simple_SACsb3": ("stable_baselines3", "SAC"),
    "rl_simple_DQN": ("custom_torch", "DQN"),
    "rl_simple_agent": (None, None),
}

# Hyperparameter keys that belong to the new SMALL universal core. Everything else in the
# legacy `hyperparameters` block is a backend-specific knob → goes to agent.params.
_CORE_HP = {"learning_rate", "gamma", "batch_size", "net_arch"}

# Legacy env keys the code never reads → dropped (reported in the summary).
_DROPPED_ENV_KEYS = {"action_space_remapping"}


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    return list(v)


def _minimal_obs_spec(causality: Optional[str], history: int,
                      reset_default: Optional[float], role: str) -> Any:
    """Emit the smallest spec dict; None (YAML null) when everything is default."""
    spec: Dict[str, Any] = {}
    if causality and causality != "same_step":
        spec["causality"] = causality
    if history:
        spec["history"] = history
    if reset_default is not None:
        spec["reset_default"] = reset_default
    if role != "state":
        spec["role"] = role
    return spec or None


def _convert_env(agent: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    env = agent.get("env", {}) or {}

    obs_keys = _as_list(env.get("observations"))
    obs_caus = _as_list(env.get("observation_causality"))
    prev_obs = _as_list(env.get("include_prev_obs"))
    reset_defaults = env.get("reset_observation_defaults") or {}

    add_obs_keys = _as_list(env.get("additional_observations"))
    add_obs_caus = _as_list(env.get("additional_observation_causality"))

    observations: Dict[str, Any] = {}
    for i, key in enumerate(obs_keys):
        caus = obs_caus[i] if i < len(obs_caus) else None
        hist = prev_obs[i] if i < len(prev_obs) else 0
        observations[key] = _minimal_obs_spec(caus, hist, reset_defaults.get(key), "state")
    for i, key in enumerate(add_obs_keys):
        caus = add_obs_caus[i] if i < len(add_obs_caus) else None
        observations[key] = _minimal_obs_spec(caus, 0, reset_defaults.get(key), "extra")

    act_keys = _as_list(env.get("actions"))
    act_types = _as_list(env.get("action_spaces_type"))
    act_bins = _as_list(env.get("action_bins"))
    act_bounds = _as_list(env.get("action_boundaries"))

    actions: Dict[str, Any] = {}
    for i, key in enumerate(act_keys):
        spec: Dict[str, Any] = {}
        space = act_types[i] if i < len(act_types) else "box"
        if space != "box":
            spec["space"] = space
        if i < len(act_bounds) and act_bounds[i] is not None:
            spec["bounds"] = list(act_bounds[i])
        if i < len(act_bins) and act_bins[i] is not None:
            spec["bins"] = act_bins[i]
        actions[key] = spec or None

    for dropped in _DROPPED_ENV_KEYS:
        if dropped in env:
            warnings.append(f"dropped unused env key '{dropped}'")

    # reset semantics
    legacy_reset_mode = None  # filled from training below by caller
    reset: Dict[str, Any] = {}
    if env.get("force_reset_observation_defaults"):
        reset["force_defaults"] = True

    out: Dict[str, Any] = {"observations": observations, "actions": actions}
    if agent.get("reward_function"):
        out["reward"] = agent["reward_function"]
    out["_reset_partial"] = reset  # merged with training-derived reset by caller
    return out


def _convert_reset(training: Dict[str, Any], partial: Dict[str, Any],
                   warnings: List[str]) -> Dict[str, Any]:
    reset = dict(partial)
    mode = training.get("reset_mode")
    if mode:
        if mode in ("full", "rolling", "none"):
            if mode != "full":
                reset["mode"] = mode
        else:
            # legacy 'soft' / 'random' have no new equivalent → map to full + warn
            warnings.append(f"reset_mode '{mode}' unsupported → mapped to 'full'")
    if training.get("rolling_window") is not None:
        reset["rolling_window"] = training["rolling_window"]
    if training.get("reset_period") is not None:
        reset["period"] = training["reset_period"]
    return reset


def _convert_agent(agent: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"model_name": agent["model_name"]}
    backend, algorithm = _BACKEND_BY_MODEL.get(agent["model_name"], (None, None))
    if backend:
        out["backend"] = backend
    if algorithm:
        out["algorithm"] = algorithm

    hp = agent.get("hyperparameters") or {}
    core = {k: hp[k] for k in _CORE_HP if k in hp}
    return out, core, hp


def _convert(rl: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    agent_in = rl.get("agent", {}) or {}
    training = rl.get("training", {}) or {}
    test = rl.get("test", {}) or {}
    checkpointing = rl.get("checkpointing", {}) or {}
    logging_in = rl.get("logging") or {}

    # ENVIRONMENT
    env = _convert_env(agent_in, warnings)
    reset_partial = env.pop("_reset_partial")
    reset = _convert_reset(training, reset_partial, warnings)
    if reset:
        env["reset"] = reset

    # AGENT
    agent_out, core_hp, full_hp = _convert_agent(agent_in)
    # train_frequency / gradient_steps live under training in the legacy shape
    if training.get("train_frequency") is not None:
        core_hp["train_frequency"] = training["train_frequency"]
    if training.get("gradient_steps") is not None:
        core_hp["gradient_steps"] = training["gradient_steps"]
    if core_hp:
        agent_out["hyperparameters"] = core_hp

    # params = non-core hyperparameters + exploration + replay_buffer + misc solver knobs
    params: Dict[str, Any] = {k: v for k, v in full_hp.items() if k not in _CORE_HP}
    if training.get("exploration"):
        params["exploration"] = training["exploration"]
    if training.get("replay_buffer"):
        params["replay_buffer"] = training["replay_buffer"]
    if training.get("warmup_steps"):
        params["warmup_steps"] = training["warmup_steps"]
    if params:
        agent_out["params"] = params

    # RUN
    run: Dict[str, Any] = {}
    mode = training.get("mode", "online")
    if mode != "online":
        run["mode"] = mode
    if training:
        run["train"] = {
            "episodes": training.get("n_episodes"),
            "episode_length": training.get("episode_length"),
        }
    if test:
        tot = test.get("total_steps")
        phase: Dict[str, Any] = {"episodes": 1, "episode_length": tot}
        if test.get("deterministic"):
            phase["deterministic"] = True
        ckpt = test.get("checkpoint_path")
        if ckpt not in (None, "", "null", "none"):
            phase["checkpoint"] = ckpt
        run["test"] = phase

    # EXPERIMENT
    experiment: Dict[str, Any] = {}
    if checkpointing.get("single_best_checkpoint"):
        experiment["checkpoint"] = {"best": checkpointing["single_best_checkpoint"]}
    if logging_in:
        experiment["logging"] = logging_in

    new: Dict[str, Any] = {}
    if rl.get("seed") is not None:
        new["seed"] = rl["seed"]
    new["environment"] = env
    new["agent"] = agent_out
    new["run"] = run
    if experiment:
        new["experiment"] = experiment
    return new


# --------------------------------------------------------------------------------------
# Splice the top-level reinforcement_learning_config block in/out of the raw text.
# --------------------------------------------------------------------------------------

def _find_block(lines: List[str], key: str) -> Optional[tuple]:
    """Return (start, end) line indices of a top-level `key:` block, else None."""
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(key + ":") and (len(ln) == len(key) + 1 or ln[len(key) + 1] in " \t#\n"):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() == "":
            continue
        if ln[0] not in " \t#":          # next top-level key at column 0
            end = j
            break
    return start, end


def convert_file(path: str, out: Optional[str], dry_run: bool) -> bool:
    with open(path, "r") as f:
        text = f.read()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "reinforcement_learning_config" not in raw:
        print(f"  [skip] {path}: no reinforcement_learning_config")
        return False
    rl = raw["reinforcement_learning_config"] or {}
    if "environment" in rl and "run" in rl:
        print(f"  [skip] {path}: already new shape")
        return False

    warnings: List[str] = []
    new_rl = _convert(rl, warnings)
    block = yaml.safe_dump({"reinforcement_learning_config": new_rl},
                           sort_keys=False, default_flow_style=False, allow_unicode=True)

    lines = text.splitlines(keepends=True)
    span = _find_block(lines, "reinforcement_learning_config")
    if span is None:
        print(f"  [skip] {path}: block not found in text")
        return False
    s, e = span
    if not block.endswith("\n"):
        block += "\n"
    new_lines = lines[:s] + [block] + lines[e:]
    new_text = "".join(new_lines)

    target = out or path
    print(f"  [conv] {path} -> {target}")
    for w in warnings:
        print(f"         ! {w}")
    if dry_run:
        print("---- new reinforcement_learning_config ----")
        print(block.rstrip())
        print("-------------------------------------------")
        return True
    with open(target, "w") as f:
        f.write(new_text)
    return True


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="scenario YAML file(s)")
    ap.add_argument("--out", default=None, help="output path (single file only)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args(argv)

    if args.out and len(args.files) != 1:
        ap.error("--out only valid with a single input file")

    n = 0
    for path in args.files:
        if convert_file(path, args.out, args.dry_run):
            n += 1
    print(f"converted {n}/{len(args.files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
