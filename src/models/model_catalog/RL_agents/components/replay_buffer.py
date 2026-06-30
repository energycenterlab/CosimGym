"""
replay_buffer.py — reusable experience replay buffer.

Extracted from rl_simple_DQN so any custom agent can reuse it. Deliberately minimal; subclass
to add prioritized replay, n-step returns, etc. (those are left as extension points, not wired).
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Simple FIFO replay buffer of (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
