"""
Reusable RL building blocks.

These components are intentionally standalone and import-light so that anyone writing a custom
agent class (the supported "full configurability in code" path) can reuse or subclass them
instead of re-implementing common machinery:

- replay_buffer.ReplayBuffer       — simple FIFO experience replay (subclass for PER / n-step)
- reward_loader.load_reward_function — resolve a dotted-path reward callable from config
- checkpoint_manager.CheckpointManager — resolve/save/load checkpoints from experiment config
- env_loop                         — thin online/test interaction-loop helpers; the seam where
                                     future offline-learning and parallel-env runs will plug in
"""

from .replay_buffer import ReplayBuffer
from .reward_loader import load_reward_function
from .checkpoint_manager import CheckpointManager

__all__ = ["ReplayBuffer", "load_reward_function", "CheckpointManager"]
