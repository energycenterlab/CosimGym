"""
env_loop.py — interaction-loop helpers between an RL agent and the HELICS-driven env.

In CosimGym the co-simulation drives time stepping, so the RL library does NOT own the rollout
loop the usual way. These helpers express the online/test loops in one place so per-algorithm
agents can reuse them and so the future seams are explicit:

  * OFFLINE learning (run.mode in {offline, mixed}): replace the env interaction with a dataset
    iterator feeding the same agent.update() — branch here on run.mode.
  * PARALLEL env runs: today one env steps in lockstep with HELICS; a vectorized/parallel
    variant would fan `step_episode` over multiple env handles. Keep step logic side-effect
    free w.r.t. agent internals to make that swap possible.

Concrete agents may still implement bespoke loops (e.g. SB3's model.learn()); this module is the
shared default + documentation of the seam, not a mandatory base.
"""
from __future__ import annotations

from typing import Callable


def run_online_loop(agent, total_steps: int, on_step: Callable | None = None):
    """Generic online interaction loop: reset, then step `total_steps`, resetting on done.

    `agent` must provide: env (gym env), act(obs), _env_step(action), and obs attribute.
    `on_step(step, transition)` optional hook for learning/logging.
    """
    obs = agent.env.reset()
    # gym reset may return (obs, info)
    agent.obs = obs[0] if isinstance(obs, tuple) else obs
    for step in range(total_steps):
        action = agent.act(agent.obs)
        next_obs, reward, terminated, truncated, info = agent._env_step(action)
        if on_step is not None:
            on_step(step, agent.transition)
        if terminated or truncated:
            r = agent.env.reset()
            agent.obs = r[0] if isinstance(r, tuple) else r


def run_test_loop(agent, total_steps: int, deterministic: bool = True):
    """Generic deterministic test loop."""
    r = agent.env.reset()
    agent.obs = r[0] if isinstance(r, tuple) else r
    for step in range(total_steps):
        action = agent.act(agent.obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = agent.env.step(action)
        agent.obs = obs
        if terminated or truncated:
            r = agent.env.reset()
            agent.obs = r[0] if isinstance(r, tuple) else r
