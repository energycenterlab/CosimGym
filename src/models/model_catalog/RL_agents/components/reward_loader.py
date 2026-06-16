"""
reward_loader.py — resolve a reward function from a dotted import path.

The new schema stores the reward callable path at environment.reward. This helper centralizes
the import so base and custom agents share one implementation.
"""
from __future__ import annotations

import importlib
from typing import Callable, Optional


def load_reward_function(reward_path: Optional[str], logger=None) -> Optional[Callable]:
    """Import and return the callable at `reward_path` (e.g. 'pkg.mod.fn'), or None.

    Raises if the path is set but cannot be imported (fail loud — a misconfigured reward is a
    silent-zero-reward trap otherwise).
    """
    if not reward_path:
        return None
    try:
        module_path, fn_name = reward_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        fn = getattr(module, fn_name)
        if logger:
            logger.info(f"Loaded reward function '{reward_path}'")
        return fn
    except Exception as e:
        if logger:
            logger.error(f"Failed to load reward function '{reward_path}': {e}")
        raise
