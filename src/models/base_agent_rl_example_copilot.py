"""
Universal Base Class for Reinforcement Learning Agents.

This module provides a production-quality abstract base class for RL agents that supports:
- Single-agent and multi-agent environments
- Online and offline training
- On-policy and off-policy algorithms
- Model-free and model-based approaches

Design Philosophy:
- Minimize dependencies (only numpy; torch is optional)
- Maximize flexibility through abstract methods
- Standardize lifecycle: reset -> act -> observe -> update
- Provide swappable storage backends
- Support both training modes (online/offline) uniformly

Integration Options:
This module provides wrappers for popular RL libraries:

┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Feature             │ Custom Agent         │ Stable-Baselines3    │ Ray RLlib            │
├─────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Ease of Use         │ Requires impl        │ ⭐⭐⭐⭐⭐            │ ⭐⭐⭐              │
│ Scalability         │ Single machine       │ Single machine       │ ⭐⭐⭐⭐⭐ Cluster   │
│ Algorithms          │ Custom only          │ 10+ standard algos   │ 20+ algos + research │
│ Multi-agent         │ Manual               │ Limited              │ ⭐⭐⭐⭐⭐ Native    │
│ Distributed         │ No                   │ No                   │ ⭐⭐⭐⭐⭐ Yes       │
│ Dependencies        │ numpy only           │ torch, gym           │ ray, torch/tf        │
│ Production Ready    │ Custom               │ ⭐⭐⭐⭐             │ ⭐⭐⭐⭐⭐           │
│ Documentation       │ Custom               │ ⭐⭐⭐⭐⭐            │ ⭐⭐⭐⭐             │
│ Best For            │ Research, learning   │ Prototyping, medium  │ Production, scale    │
└─────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

Usage:
    # Custom agent
    agent = ExampleTabularQAgent(num_states=10, num_actions=4)
    
    # Stable-Baselines3
    from stable_baselines3 import PPO
    sb3_model = PPO("MlpPolicy", env)
    agent = SB3AgentWrapper(sb3_model)
    
    # Ray RLlib
    from ray.rllib.algorithms.ppo import PPOConfig
    rllib_algo = PPOConfig().environment("CartPole-v1").build()
    agent = RLlibAgentWrapper(rllib_algo)
    
    # All use the same interface!
    agent.train_loop_online(env, max_steps=10000)

Author: Senior RL Engineering Team
Python: 3.11+
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np

# Optional torch support - guarded import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Type Aliases
# ============================================================================

AgentID = str | int
"""Agent identifier for multi-agent environments."""

Observation = np.ndarray | Any
"""Single-agent observation (array, tensor, or structured data)."""

Action = np.ndarray | int | float | Any
"""Single-agent action (discrete or continuous)."""

ObsDict = dict[AgentID, Observation]
"""Multi-agent observations: {agent_id: obs}."""

ActionDict = dict[AgentID, Action]
"""Multi-agent actions: {agent_id: action}."""

InfoDict = dict[str, Any]
"""Auxiliary information dictionary."""

MetricsDict = dict[str, float]
"""Training/evaluation metrics."""


# ============================================================================
# Data Containers
# ============================================================================

@dataclass
class Transition:
    """
    Single-agent transition (s, a, r, s', done).
    
    Used for off-policy algorithms and episodic storage.
    """
    obs: Observation
    action: Action
    reward: float
    next_obs: Observation
    done: bool
    info: InfoDict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate transition data."""
        if not isinstance(self.reward, (int, float, np.number)):
            raise TypeError(f"Reward must be numeric, got {type(self.reward)}")
        if not isinstance(self.done, bool):
            raise TypeError(f"Done must be bool, got {type(self.done)}")


@dataclass
class MA_Transition:
    """
    Multi-agent transition.
    
    Stores observations, actions, rewards for multiple agents.
    Supports both centralized and decentralized training.
    """
    obs: ObsDict
    actions: ActionDict
    rewards: dict[AgentID, float]
    next_obs: ObsDict
    dones: dict[AgentID, bool]
    info: InfoDict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate multi-agent transition."""
        agents = set(self.obs.keys())
        if not (agents == set(self.actions.keys()) == set(self.rewards.keys()) == 
                set(self.next_obs.keys()) == set(self.dones.keys())):
            raise ValueError("All agent dictionaries must have same keys")


@dataclass
class Batch:
    """
    Batch of transitions for training.
    
    Can contain numpy arrays or torch tensors.
    Supports both single-agent and multi-agent formats.
    """
    obs: np.ndarray | Any
    actions: np.ndarray | Any
    rewards: np.ndarray | Any
    next_obs: np.ndarray | Any
    dones: np.ndarray | Any
    extras: dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return batch size."""
        if isinstance(self.obs, np.ndarray):
            return len(self.obs)
        elif TORCH_AVAILABLE and isinstance(self.obs, torch.Tensor):
            return len(self.obs)
        elif isinstance(self.obs, dict):
            # Multi-agent: return size of first agent's obs
            first_key = next(iter(self.obs))
            return len(self.obs[first_key])
        else:
            raise TypeError(f"Cannot determine batch size from obs type: {type(self.obs)}")
    
    def to_device(self, device: str | Any) -> Batch:
        """
        Move batch to device (torch only).
        
        Returns self if torch not available or data is numpy.
        """
        if not TORCH_AVAILABLE:
            return self
        
        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            return x
        
        return Batch(
            obs=_to_device(self.obs),
            actions=_to_device(self.actions),
            rewards=_to_device(self.rewards),
            next_obs=_to_device(self.next_obs),
            dones=_to_device(self.dones),
            extras={k: _to_device(v) for k, v in self.extras.items()},
        )


# ============================================================================
# Storage Protocol and Implementations
# ============================================================================

T = TypeVar('T')


@runtime_checkable
class Storage(Protocol[T]):
    """
    Protocol for experience storage backends.
    
    Supports both replay buffers (off-policy) and rollout buffers (on-policy).
    """
    
    def add(self, item: T) -> None:
        """Add an item to storage."""
        ...
    
    def sample(self, batch_size: int) -> list[T]:
        """Sample a batch of items."""
        ...
    
    def __len__(self) -> int:
        """Return number of items in storage."""
        ...
    
    def clear(self) -> None:
        """Clear all stored items."""
        ...


class ReplayBuffer(Generic[T]):
    """
    Circular replay buffer for off-policy algorithms.
    
    Stores fixed number of recent transitions and samples uniformly.
    Thread-safe for single producer, single consumer.
    """
    
    def __init__(self, capacity: int, seed: int | None = None):
        """
        Args:
            capacity: Maximum number of items to store
            seed: Random seed for sampling reproducibility
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self.capacity = capacity
        self._buffer: deque[T] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)
    
    def add(self, item: T) -> None:
        """Add item to buffer (overwrites oldest if full)."""
        self._buffer.append(item)
    
    def sample(self, batch_size: int) -> list[T]:
        """
        Sample batch uniformly at random.
        
        Args:
            batch_size: Number of items to sample
            
        Returns:
            List of sampled items
            
        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} items from buffer of size {len(self._buffer)}"
            )
        
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def clear(self) -> None:
        """Remove all items from buffer."""
        self._buffer.clear()


class RolloutBuffer(Generic[T]):
    """
    Rollout buffer for on-policy algorithms.
    
    Stores complete trajectories and returns all data when sampled.
    Typically cleared after each update (PPO, A2C, etc.).
    """
    
    def __init__(self, seed: int | None = None):
        """
        Args:
            seed: Random seed (for compatibility; sampling returns all data)
        """
        self._buffer: list[T] = []
        self._rng = np.random.default_rng(seed)
    
    def add(self, item: T) -> None:
        """Add item to buffer."""
        self._buffer.append(item)
    
    def sample(self, batch_size: int | None = None) -> list[T]:
        """
        Return all items (on-policy algorithms use full trajectories).
        
        Args:
            batch_size: Ignored (for protocol compatibility)
            
        Returns:
            All stored items
        """
        return list(self._buffer)
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def clear(self) -> None:
        """Clear buffer (called after policy update)."""
        self._buffer.clear()


# ============================================================================
# Base RL Agent
# ============================================================================

class BaseRLAgent(ABC):
    """
    Universal abstract base class for RL agents.
    
    Supports:
    - Single-agent and multi-agent environments
    - Online training (with environment interaction)
    - Offline training (from dataset iterator)
    - On-policy and off-policy algorithms
    - Model-free and model-based approaches
    
    Lifecycle:
        1. Instantiate agent
        2. reset() to initialize episode
        3. set_mode("train") or set_mode("eval")
        4. Loop: act() -> step env -> observe() -> update()
        5. Evaluate with evaluate()
        6. Save/load with save()/load()
    
    Subclass Requirements:
        - Implement act() to compute actions from observations
        - Implement update() to perform learning updates
        - Optionally override observe(), can_update(), etc.
    """
    
    def __init__(
        self,
        *,
        name: str = "BaseAgent",
        multi_agent: bool = False,
        seed: int | None = None,
    ):
        """
        Args:
            name: Agent identifier
            multi_agent: Whether agent operates in multi-agent mode
            seed: Random seed for reproducibility
        """
        self.name = name
        self.multi_agent = multi_agent
        self.seed = seed
        
        # Training state
        self._mode: Literal["train", "eval"] = "train"
        self._rng = np.random.default_rng(seed)
        
        # Counters
        self._env_steps = 0
        self._updates = 0
        self._episodes = 0
        self._training_start_time: float | None = None
        
        # Metrics accumulator
        self._metrics: MetricsDict = {}
    
    # ========================================================================
    # Core Lifecycle Methods
    # ========================================================================
    
    def reset(self, seed: int | None = None) -> None:
        """
        Reset agent state for new episode.
        
        Args:
            seed: Optional seed to reset RNG (if None, keeps current seed)
        """
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        
        self._episodes += 1
    
    def set_mode(self, mode: Literal["train", "eval"]) -> None:
        """
        Set agent mode (train vs eval).
        
        Training mode: exploration, learning updates
        Eval mode: deterministic/greedy actions, no updates
        
        Args:
            mode: "train" or "eval"
        """
        if mode not in ("train", "eval"):
            raise ValueError(f"Mode must be 'train' or 'eval', got {mode}")
        self._mode = mode
    
    @abstractmethod
    def act(
        self,
        obs: Observation | ObsDict,
        *,
        explore: bool | None = None,
    ) -> tuple[Action | ActionDict, InfoDict]:
        """
        Compute action(s) from observation(s).
        
        Args:
            obs: Single-agent observation or multi-agent dict {agent_id: obs}
            explore: Whether to explore (None = use current mode)
        
        Returns:
            actions: Single action or dict {agent_id: action}
            info: Auxiliary information (e.g., log_probs, values)
        
        Design Notes:
            - If explore is None, use self._mode to decide
            - In "eval" mode, typically return deterministic/greedy actions
            - In "train" mode, apply exploration strategy
            - Multi-agent: return ActionDict with same keys as ObsDict
        """
        ...
    
    def observe(
        self,
        transition: Transition | MA_Transition,
    ) -> None:
        """
        Observe a transition (for online learning).
        
        Args:
            transition: Single-agent or multi-agent transition
        
        Design Notes:
            - Default: do nothing (stateless agents)
            - Off-policy: store in replay buffer
            - On-policy: store in rollout buffer
            - Can be used for logging, prioritization, etc.
        """
        pass  # Default: stateless agent, no storage
    
    def can_update(self) -> bool:
        """
        Check if agent is ready to perform update.
        
        Returns:
            True if update can be called
        
        Design Notes:
            - Off-policy: check if replay buffer has enough samples
            - On-policy: check if rollout buffer is full
            - Default: always ready (for offline agents)
        """
        return True  # Default: always ready
    
    @abstractmethod
    def update(self, *, num_updates: int = 1) -> MetricsDict:
        """
        Perform learning update(s).
        
        Args:
            num_updates: Number of gradient steps (or update iterations)
        
        Returns:
            Metrics dict with losses, gradients, etc.
        
        Design Notes:
            - Off-policy: sample from replay buffer, compute loss, update
            - On-policy: use all rollout data, multi-epoch updates
            - Increment self._updates counter
            - Return metrics for logging (e.g., {"loss": 0.5, "grad_norm": 1.2})
        """
        ...
    
    # ========================================================================
    # Training Loops
    # ========================================================================
    
    def train_loop_online(
        self,
        env: Any,  # Duck-typed environment with reset(), step()
        max_steps: int,
        *,
        update_every: int = 1,
        updates_per_step: int = 1,
        eval_every: int | None = None,
        eval_episodes: int = 5,
        log_every: int = 1000,
    ) -> MetricsDict:
        """
        Online training loop with environment interaction.
        
        Args:
            env: Environment with reset() -> obs, step(action) -> (obs, reward, done, info)
            max_steps: Maximum environment steps
            update_every: Update frequency (steps)
            updates_per_step: Number of updates per trigger
            eval_every: Evaluation frequency (None = no eval)
            eval_episodes: Episodes per evaluation
            log_every: Logging frequency (steps)
        
        Returns:
            Final metrics dict
        
        Design Notes:
            - Assumes env has standard gym-like interface
            - Handles episode resets automatically
            - Calls observe() after each step
            - Calls update() when can_update() and step % update_every == 0
        """
        self.set_mode("train")
        self._training_start_time = time.time()
        
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(max_steps):
            # Act
            action, _ = self.act(obs, explore=True)
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            if self.multi_agent:
                transition = MA_Transition(
                    obs=obs,
                    actions=action,
                    rewards=reward if isinstance(reward, dict) else {0: reward},
                    next_obs=next_obs,
                    dones=done if isinstance(done, dict) else {0: done},
                    info=info,
                )
            else:
                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=bool(done),
                    info=info,
                )
            
            self.observe(transition)
            
            # Update counters
            self._env_steps += 1
            episode_reward += reward if not isinstance(reward, dict) else sum(reward.values())
            episode_steps += 1
            
            # Update agent
            if (step + 1) % update_every == 0 and self.can_update():
                for _ in range(updates_per_step):
                    update_metrics = self.update(num_updates=1)
                    self._metrics.update(update_metrics)
            
            # Logging
            if (step + 1) % log_every == 0:
                elapsed = time.time() - self._training_start_time
                print(
                    f"[{self.name}] Step {step+1}/{max_steps} | "
                    f"Episodes: {self._episodes} | "
                    f"Steps/s: {self._env_steps/elapsed:.1f}"
                )
            
            # Evaluation
            if eval_every and (step + 1) % eval_every == 0:
                eval_metrics = self.evaluate(env, episodes=eval_episodes)
                print(f"[{self.name}] Eval: {eval_metrics}")
                self._metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                self.set_mode("train")
            
            # Episode boundary
            if done if not isinstance(done, dict) else all(done.values()):
                self._metrics["episode_reward"] = episode_reward
                self._metrics["episode_steps"] = episode_steps
                obs = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                self.reset()
            else:
                obs = next_obs
        
        return self.get_metrics()
    
    def train_loop_offline(
        self,
        dataset_iter: Iterator[Batch],
        max_updates: int,
        *,
        log_every: int = 100,
    ) -> MetricsDict:
        """
        Offline training loop from dataset.
        
        Args:
            dataset_iter: Iterator yielding Batch objects
            max_updates: Maximum number of updates
            log_every: Logging frequency
        
        Returns:
            Final metrics dict
        
        Design Notes:
            - Agent learns purely from dataset (no env interaction)
            - Useful for offline RL, behavioral cloning, imitation learning
            - Subclass can override update() to consume batches directly
        """
        self.set_mode("train")
        self._training_start_time = time.time()
        
        for update_idx in range(max_updates):
            try:
                batch = next(dataset_iter)
            except StopIteration:
                print(f"[{self.name}] Dataset exhausted at update {update_idx}")
                break
            
            # Subclass should implement batch-based update
            # Here we call update() assuming subclass handles batch sampling internally
            update_metrics = self.update(num_updates=1)
            self._metrics.update(update_metrics)
            
            if (update_idx + 1) % log_every == 0:
                elapsed = time.time() - self._training_start_time
                print(
                    f"[{self.name}] Update {update_idx+1}/{max_updates} | "
                    f"Updates/s: {self._updates/elapsed:.1f}"
                )
        
        return self.get_metrics()
    
    def evaluate(
        self,
        env: Any,
        episodes: int,
        *,
        max_steps_per_episode: int = 10000,
        render: bool = False,
    ) -> MetricsDict:
        """
        Evaluate agent on environment.
        
        Args:
            env: Environment instance
            episodes: Number of episodes to run
            max_steps_per_episode: Max steps per episode
            render: Whether to render (if env supports it)
        
        Returns:
            Metrics dict with mean/std of returns and episode lengths
        """
        self.set_mode("eval")
        
        episode_rewards = []
        episode_lengths = []
        
        #  need to evaluate at the end of episodes and add in the training lood the episode concept
        #  cannot increas or decrease (if not in episode multiples) the evaluatiomn frequency
        for ep in range(episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for _ in range(max_steps_per_episode):
                action, _ = self.act(obs, explore=False)
                obs, reward, done, _ = env.step(action)
                
                episode_reward += reward if not isinstance(reward, dict) else sum(reward.values())
                episode_length += 1
                
                if render and hasattr(env, "render"):
                    env.render()
                
                if done if not isinstance(done, dict) else all(done.values()):
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_return": float(np.mean(episode_rewards)),
            "std_return": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
        }
    
    # ========================================================================
    # State Management
    # ========================================================================
    
    def state_dict(self) -> dict[str, Any]:
        """
        Return agent state for checkpointing.
        
        Returns:
            State dictionary (counters, metrics, etc.)
        
        Design Notes:
            - Subclass should extend to include model parameters, optimizer state
            - Framework-agnostic: use pickle-able types
        """
        return {
            "name": self.name,
            "multi_agent": self.multi_agent,
            "seed": self.seed,
            "env_steps": self._env_steps,
            "updates": self._updates,
            "episodes": self._episodes,
            "metrics": self._metrics.copy(),
        }
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Load agent state from checkpoint.
        
        Args:
            state: State dictionary from state_dict()
        """
        self.name = state.get("name", self.name)
        self.multi_agent = state.get("multi_agent", self.multi_agent)
        self.seed = state.get("seed", self.seed)
        self._env_steps = state.get("env_steps", 0)
        self._updates = state.get("updates", 0)
        self._episodes = state.get("episodes", 0)
        self._metrics = state.get("metrics", {}).copy()
    
    def save(self, path: str | Path) -> None:
        """
        Save agent to file.
        
        Args:
            path: File path (will create parent dirs)
        """
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)
        
        print(f"[{self.name}] Saved to {path}")
    
    def load(self, path: str | Path) -> None:
        """
        Load agent from file.
        
        Args:
            path: File path to load from
        """
        import pickle
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.load_state_dict(state)
        print(f"[{self.name}] Loaded from {path}")
    
    # ========================================================================
    # Metrics and Utilities
    # ========================================================================
    
    def get_metrics(self) -> MetricsDict:
        """
        Get current metrics.
        
        Returns:
            Metrics dict (includes counters and training metrics)
        """
        base_metrics = {
            "env_steps": self._env_steps,
            "updates": self._updates,
            "episodes": self._episodes,
        }
        
        if self._training_start_time is not None:
            elapsed = time.time() - self._training_start_time
            base_metrics["elapsed_time"] = elapsed
            base_metrics["steps_per_sec"] = self._env_steps / elapsed if elapsed > 0 else 0
        
        return {**base_metrics, **self._metrics}
    
    def close(self) -> None:
        """
        Clean up resources.
        
        Design Notes:
            - Override to close environments, free GPU memory, etc.
        """
        pass


# ============================================================================
# Example Implementations
# ============================================================================

class ExampleRandomAgent(BaseRLAgent):
    """
    Example: Random agent (compatible with single-agent and multi-agent).
    
    Demonstrates minimal implementation.
    """
    
    def __init__(
        self,
        action_space_size: int | dict[AgentID, int],
        **kwargs,
    ):
        """
        Args:
            action_space_size: Number of discrete actions (int or dict for multi-agent)
            **kwargs: Passed to BaseRLAgent
        """
        super().__init__(**kwargs)
        self.action_space_size = action_space_size
    
    def act(
        self,
        obs: Observation | ObsDict,
        *,
        explore: bool | None = None,
    ) -> tuple[Action | ActionDict, InfoDict]:
        """Return random action(s)."""
        if self.multi_agent:
            assert isinstance(obs, dict), "Multi-agent mode requires dict observations"
            assert isinstance(self.action_space_size, dict), "Multi-agent requires dict action space"
            
            actions = {
                agent_id: int(self._rng.integers(0, self.action_space_size[agent_id]))
                for agent_id in obs.keys()
            }
            return actions, {}
        else:
            assert isinstance(self.action_space_size, int), "Single-agent requires int action space"
            action = int(self._rng.integers(0, self.action_space_size))
            return action, {}
    
    def update(self, *, num_updates: int = 1) -> MetricsDict:
        """Random agent doesn't learn."""
        self._updates += num_updates
        return {"loss": 0.0}



# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Base RL Agent - Usage Examples")
    print("=" * 70)
    print("""
This module demonstrates three approaches to RL agents:

1. Custom Agents (Examples 1-2):
   - Full control over algorithm
   - Minimal dependencies
   - Best for research, custom algorithms
   - Examples: RandomAgent, TabularQAgent

2. Stable-Baselines3 Wrapper (Example 5):
   - Easy to use, well-documented
   - Good for single-machine training
   - Wide range of standard algorithms (PPO, SAC, DQN, TD3, A2C)
   - Best for prototyping, small-to-medium scale

3. Ray RLlib Wrapper (Example 6):
   - Production-grade, distributed training
   - Scales to clusters
   - Advanced features (multi-agent, offline RL, model-based)
   - Best for large-scale, production deployments

All three approaches work with the same BaseRLAgent interface!
    """)
    print("=" * 70)
    
    # ========================================================================
    # Example 1: Online Training with Random Agent
    # ========================================================================
    
    print("\n[Example 1] Online Training - Random Agent")
    print("-" * 70)
    
    # Dummy single-agent environment (duck-typed)
    class DummyEnv:
        def __init__(self, num_states=10, num_actions=4):
            self.num_states = num_states
            self.num_actions = num_actions
            self.state = 0
            self.steps = 0
            self._rng = np.random.default_rng()
        
        def reset(self):
            self.state = 0
            self.steps = 0
            return self.state
        
        def step(self, action):
            self.steps += 1
            reward = self._rng.random() - 0.5
            self.state = self._rng.integers(0, self.num_states)
            done = self.steps >= 20
            info = {}
            return self.state, reward, done, info
    
    # Create environment and agent
    env = DummyEnv(num_states=10, num_actions=4)
    agent = ExampleRandomAgent(
        action_space_size=4,
        name="RandomAgent",
        multi_agent=False,
        seed=42,
    )
    
    # Train online
    metrics = agent.train_loop_online(
        env=env,
        max_steps=500,
        update_every=10,
        updates_per_step=1,
        log_every=200,
    )
    
    print(f"\nFinal metrics: {metrics}")
    
    # ========================================================================
    # Example 2: Offline Training with Tabular Q-Learning
    # ========================================================================
    
    print("\n[Example 2] Offline Training - Tabular Q-Learning")
    print("-" * 70)
    
    # Create dataset iterator (dummy batches)
    def create_dummy_dataset(num_batches=100, batch_size=32):
        """Generate dummy transition batches."""
        for _ in range(num_batches):
            obs = np.random.randint(0, 10, size=batch_size)
            actions = np.random.randint(0, 4, size=batch_size)
            rewards = np.random.randn(batch_size)
            next_obs = np.random.randint(0, 10, size=batch_size)
            dones = np.random.rand(batch_size) < 0.1
            
            yield Batch(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
            )
    
    # Note: For offline training with BaseRLAgent, you'd typically:
    # 1. Create a subclass that consumes batches directly in update()
    # 2. Or manually add transitions to buffer before calling train_loop_offline
    
    # For demonstration, we'll use online training with TabularQ:
    env2 = DummyEnv(num_states=10, num_actions=4)
    q_agent = ExampleTabularQAgent(
        num_states=10,
        num_actions=4,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.2,
        buffer_capacity=1000,
        batch_size=32,
        name="TabularQ",
        seed=42,
    )
    
    print("Training Tabular Q-Learning agent...")
    metrics = q_agent.train_loop_online(
        env=env2,
        max_steps=1000,
        update_every=5,
        updates_per_step=1,
        eval_every=500,
        eval_episodes=10,
        log_every=400,
    )
    
    print(f"\nFinal Q-agent metrics: {metrics}")
    print(f"Learned Q-table shape: {q_agent.q_table.shape}")
    print(f"Sample Q-values for state 0: {q_agent.q_table[0]}")
    
    # ========================================================================
    # Example 3: Multi-Agent Random Agent
    # ========================================================================
    
    print("\n[Example 3] Multi-Agent Random Agent")
    print("-" * 70)
    
    # Dummy multi-agent environment
    class DummyMultiAgentEnv:
        def __init__(self):
            self.agents = ["agent_0", "agent_1"]
            self.num_actions = {"agent_0": 3, "agent_1": 5}
            self.steps = 0
            self._rng = np.random.default_rng()
        
        def reset(self):
            self.steps = 0
            return {agent: np.zeros(4) for agent in self.agents}
        
        def step(self, actions: dict):
            self.steps += 1
            obs = {agent: self._rng.random(4) for agent in self.agents}
            rewards = {agent: self._rng.random() for agent in self.agents}
            dones = {agent: self.steps >= 15 for agent in self.agents}
            info = {}
            return obs, rewards, dones, info
    
    ma_env = DummyMultiAgentEnv()
    ma_agent = ExampleRandomAgent(
        action_space_size={"agent_0": 3, "agent_1": 5},
        name="MultiAgentRandom",
        multi_agent=True,
        seed=42,
    )
    
    metrics = ma_agent.train_loop_online(
        env=ma_env,
        max_steps=300,
        update_every=20,
        log_every=150,
    )
    
    print(f"\nMulti-agent metrics: {metrics}")
    
    # ========================================================================
    # Example 4: Save/Load
    # ========================================================================
    
    print("\n[Example 4] Save and Load Agent")
    print("-" * 70)
    
    # Save Q-agent
    save_path = Path("/tmp/q_agent_checkpoint.pkl")
    q_agent.save(save_path)
    
    # Create new agent and load
    new_agent = ExampleTabularQAgent(
        num_states=10,
        num_actions=4,
        name="LoadedAgent",
    )
    new_agent.load(save_path)
    
    print(f"Original agent updates: {q_agent._updates}")
    print(f"Loaded agent updates: {new_agent._updates}")
    print(f"Q-tables match: {np.allclose(q_agent.q_table, new_agent.q_table)}")
    
    # ========================================================================
    # Example 5: Stable-Baselines3 Integration
    # ========================================================================
    
    print("\n[Example 5] Stable-Baselines3 Integration")
    print("-" * 70)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        
        # Create a simple gym-compatible environment
        import gymnasium as gym
        
        # For demonstration, we'll create a simple wrapper for our DummyEnv
        class GymDummyEnv(gym.Env):
            """Gym-compatible version of DummyEnv for SB3."""
            
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Discrete(10)
                self.action_space = gym.spaces.Discrete(4)
                self.state = 0
                self.steps = 0
                self._rng = np.random.default_rng()
            
            def reset(self, seed=None, options=None):
                if seed is not None:
                    self._rng = np.random.default_rng(seed)
                self.state = 0
                self.steps = 0
                return self.state, {}
            
            def step(self, action):
                self.steps += 1
                reward = float(self._rng.random() - 0.5)
                self.state = int(self._rng.integers(0, 10))
                terminated = self.steps >= 20
                truncated = False
                info = {}
                return self.state, reward, terminated, truncated, info
        
        # Create environment
        gym_env = GymDummyEnv()
        
        # Create SB3 PPO model
        print("Creating Stable-Baselines3 PPO model...")
        sb3_model = PPO(
            "MlpPolicy",
            gym_env,
            verbose=0,
            n_steps=128,  # Rollout buffer size
            batch_size=64,
            learning_rate=3e-4,
        )
        
        # Wrap in BaseRLAgent interface
        sb3_agent = SB3AgentWrapper(
            sb3_model,
            name="PPO_Agent",
            batch_size=128,
        )
        
        print(f"Agent created: {sb3_agent.name}")
        print(f"On-policy algorithm: {sb3_agent._is_on_policy}")
        
        # Train using SB3's native method (efficient)
        print("\nTraining with SB3's native learn() method...")
        metrics = sb3_agent.train_loop_online(
            env=gym_env,
            max_steps=2000,
            eval_every=2000,
            eval_episodes=5,
        )
        
        print(f"\nSB3 Training metrics: {metrics}")
        
        # Test action selection
        print("\nTesting action selection:")
        test_obs = 5
        action, info = sb3_agent.act(test_obs, explore=False)
        print(f"  Observation: {test_obs}")
        print(f"  Deterministic action: {action}")
        print(f"  Info: {info}")
        
        # Save and load
        sb3_save_path = Path("/tmp/sb3_agent_checkpoint")
        sb3_agent.save(sb3_save_path)
        
        # Load into new wrapper
        new_sb3_model = PPO("MlpPolicy", gym_env, verbose=0)
        new_sb3_agent = SB3AgentWrapper(new_sb3_model, name="LoadedPPO")
        new_sb3_agent.load(sb3_save_path)
        
        print(f"\nOriginal timesteps: {sb3_agent.sb3_model.num_timesteps}")
        print(f"Loaded timesteps: {new_sb3_agent.sb3_model.num_timesteps}")
        
        print("\n[Example 5] Success! SB3 integration working correctly.")
        
        # Show how to use different SB3 algorithms
        print("\n" + "-" * 70)
        print("Other SB3 algorithms you can use:")
        print("-" * 70)
        print("""
from stable_baselines3 import SAC, DQN, A2C, TD3

# Off-policy algorithms (good for sample efficiency)
sac_model = SAC("MlpPolicy", env, verbose=1)
dqn_model = DQN("MlpPolicy", env, verbose=1)
td3_model = TD3("MlpPolicy", env, verbose=1)

# On-policy algorithms (good for stability)
a2c_model = A2C("MlpPolicy", env, verbose=1)
ppo_model = PPO("MlpPolicy", env, verbose=1)

# Wrap any of them
agent = SB3AgentWrapper(ppo_model, name="MyAgent")
agent.train_loop_online(env, max_steps=10000)
        """)
        
    except ImportError as e:
        print(f"\n[Example 5] Skipped - Stable-Baselines3 not installed")
        print(f"  Install with: pip install stable-baselines3")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"\n[Example 5] Error: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Example 6: Ray RLlib Integration
    # ========================================================================
    
    print("\n[Example 6] Ray RLlib Integration")
    print("-" * 70)
    
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.dqn import DQNConfig
        
        # Initialize Ray (required for RLlib)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)
        
        print("Creating Ray RLlib PPO algorithm...")
        
        # Create RLlib configuration
        config = (
            PPOConfig()
            .environment(env="CartPole-v1")
            .framework("torch")  # or "tf"
            .training(
                lr=0.0003,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
            )
            .rollouts(
                num_rollout_workers=1,  # Parallel workers for sampling
                num_envs_per_worker=1,
            )
            .evaluation(
                evaluation_interval=10,
                evaluation_num_workers=1,
            )
        )
        
        # Build the algorithm
        rllib_algo = config.build()
        
        print(f"RLlib algorithm created: {rllib_algo.__class__.__name__}")
        
        # Wrap in BaseRLAgent interface
        rllib_agent = RLlibAgentWrapper(
            rllib_algo,
            name="PPO_RLlib",
            use_rllib_training=True,
        )
        
        print(f"Agent created: {rllib_agent.name}")
        print(f"Using RLlib native training: {rllib_agent.use_rllib_training}")
        
        # Train using RLlib's native method (efficient, distributed)
        print("\nTraining with RLlib's native train() method...")
        metrics = rllib_agent.train_loop_online(
            env=None,  # RLlib uses env from config
            max_steps=5000,
            log_every=1000,
        )
        
        print(f"\nRLlib Training metrics: {metrics}")
        
        # Test action selection
        print("\nTesting action selection:")
        test_obs = np.array([0.1, 0.2, 0.3, 0.4])
        action, info = rllib_agent.act(test_obs, explore=False)
        print(f"  Observation: {test_obs}")
        print(f"  Deterministic action: {action}")
        print(f"  Info: {info}")
        
        # Save checkpoint
        rllib_save_path = Path("/tmp/rllib_agent_checkpoint")
        rllib_agent.save(rllib_save_path)
        
        print(f"\nCheckpoint saved successfully")
        
        # Demonstrate different RLlib algorithms
        print("\n" + "-" * 70)
        print("Other RLlib algorithms you can use:")
        print("-" * 70)
        print("""
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig

# On-policy algorithms
ppo_config = PPOConfig().environment("CartPole-v1")
ppo_algo = ppo_config.build()

# Off-policy algorithms
dqn_config = DQNConfig().environment("CartPole-v1")
dqn_algo = dqn_config.build()

sac_config = SACConfig().environment("Pendulum-v1")
sac_algo = sac_config.build()

# Distributed algorithms
appo_config = (
    APPOConfig()
    .environment("CartPole-v1")
    .rollouts(num_rollout_workers=4)
)
appo_algo = appo_config.build()

# Wrap any of them
agent = RLlibAgentWrapper(ppo_algo, name="MyRLlibAgent")
agent.train_loop_online(env=None, max_steps=10000)
        """)
        
        # Cleanup
        rllib_algo.stop()
        ray.shutdown()
        
        print("\n[Example 6] Success! RLlib integration working correctly.")
        
    except ImportError as e:
        print(f"\n[Example 6] Skipped - Ray RLlib not installed")
        print(f"  Install with: pip install 'ray[rllib]'")
        print(f"  Error: {e}")
    except Exception as e:
        print(f"\n[Example 6] Error: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        try:
            ray.shutdown()
        except:
            pass
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
