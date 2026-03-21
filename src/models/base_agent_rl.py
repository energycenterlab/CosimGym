"""
base_agent_rl.py

Abstract base classes and type definitions for Reinforcement Learning agents using Gymnasium.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

from abc import ABC, abstractmethod
import importlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace, Discrete, MultiBinary, MultiDiscrete


Observation = np.ndarray | Any
"""Single-agent observation (array, tensor, or structured data)."""

Action = np.ndarray | int | float | Any
"""Single-agent action (discrete or continuous)."""

InfoDict = dict[str, Any]
"""Auxiliary information dictionary."""

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
    def __repr__(self):
        return (f"Transition(obs={self.obs}, action={self.action}, reward={self.reward}, "
                f"next_obs={self.next_obs}, done={self.done}, info={self.info})")



   

class RLAgent(ABC):
    def __init__(self, env, logger=None, rl_task=None):
        self.env = env
        self.logger = logger
        self.mode = None
        self.model = None
        self.rl_task = rl_task # complete set of rl configs from scenario yaml
        self.transition = None # current transition for online training
        self.obs = None
        self.best_model_checkpoint = None # path to best model checkpoint during training for testing loop if not specified in test config

        # --- dynamic reward function (loaded from YAML config) ---
        self._reward_fn: Optional[Callable] = None
        reward_path = (
            rl_task.agent.reward_function
            if rl_task is not None and hasattr(rl_task, 'agent') and rl_task.agent.reward_function
            else None
        )
        if reward_path:
            try:
                module_path, fn_name = reward_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                self._reward_fn = getattr(module, fn_name)
                if logger:
                    logger.info(f"RLAgent: loaded reward function '{reward_path}'")
            except Exception as e:
                if logger:
                    logger.error(f"RLAgent: failed to load reward function '{reward_path}': {e}")
                raise


    @abstractmethod
    def act(self, obs):
        self.logger.debug(f"RLAgent received observation: {obs}")
        action= self.env.action_space.sample()  # Replace with actual action selection logic based on the observation
        self.logger.debug(f"RLAgent selected action: {action}")
        return action
    
    def compute_reward(self, obs, action) -> float:
        """
        Compute reward for the current transition.

        If a reward function was specified in the scenario config
        (reinforcement_learning_config.agent.reward_function), it is called here.
        Reward functions receive observations with original key names (with dots).
        
        Subclasses may override this method to define the reward inline instead.
        """
        if self._reward_fn is not None:
            # Desanitize observation if DictKeyNameWrapper is in the env wrapper chain
            obs_for_reward = self._desanitize_obs_if_needed(obs)
            prev_obs_for_reward = self._desanitize_obs_if_needed(self.obs)
            return float(self._reward_fn(obs=obs_for_reward, action=action, prev_obs=prev_obs_for_reward))
        # fallback – subclass must override if no reward_function is configured
        self.logger.warning("compute_reward called but no reward_function configured and no override provided.")
        return 0.0
    
    def _desanitize_obs_if_needed(self, obs):
        """
        Desanitize observation if DictKeyNameWrapper is in the env wrapper chain.
        
        Traverses the entire wrapper stack to find DictKeyNameWrapper.
        
        Args:
            obs: Observation object (potentially with sanitized keys)
            
        Returns:
            Observation with original keys if DictKeyNameWrapper is found, otherwise original obs
        """
        if obs is None:
            return obs
        
        # Traverse the wrapper chain to find DictKeyNameWrapper
        current_env = self.env
        while current_env is not None:
            if isinstance(current_env, DictKeyNameWrapper):
                return current_env.desanitize_observation(obs)
            
            # Move to the next wrapper in the chain
            current_env = getattr(current_env, 'env', None)
        
        # No DictKeyNameWrapper found, return observation as-is
        return obs
    
   
    
    def _env_step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        transition = Transition(
                    obs=self.obs,
                    action=action,
                    reward=float(reward),
                    next_obs=obs,
                    done=bool(terminated), # or truncated not for now
                    info=info,
                )
        self.logger.debug(f"RLAgent stored transition: {transition}")
        self.obs = obs
        self.transition = transition
        return obs, reward, terminated, truncated, info

    def online_training_loop(self):
        # main loop for online training example to be overridden
        self.logger.debug(f"Starting online training loop for RLAgent: for a number of steps:{self.rl_task.training.total_steps}")
        obs = self.env.reset()  # Reset the environment at the start of training
        self.obs = obs
        while self.env.ts < self.rl_task.training.total_steps:
            self.logger.debug(f" Observation: {self.obs}")
            action = self.act(self.obs)
            next_obs, reward, terminated, truncated, info = self.env_step(action)

            # store transistion in buffer for learning

            # learning gradient update loop if buffer size > N and update frequency matched
                # for step in gradient steps:
                    # sample batch from buffer
                    # compute loss for variuos algorithms 
                    # perform optimization step


            if terminated or truncated:
                # evaluate policy if needed
                self.obs = self.env.reset()  # Reset the environment if the episode is done

            
   
    
    def offline_training_loop(self):
        # main loop for offline training
        
        return True
    
    def testing_loop(self):
        # main loop for testing the agent
        self.logger.debug(f"Starting testing loop for RLAgent: for a number of steps:{self.rl_task.test.total_steps}")
        obs = self.env.reset()  # Reset the environment at the start of training
        self.obs = obs
        while self.env.ts < self.rl_task.test.total_steps:
            self.logger.debug(f" Observation: {obs}")
            action = self.act(obs)
            next_obs, reward, terminated, truncated, info = self.env_step(action)
            obs = next_obs  # Update the current observation
       
        pass

    def save_checkpoint(self, name):
        # save model checkpoint
        if self.rl_task.checkpointing.directory:
            path = self.rl_task.checkpointing.directory + name
            self.best_model_checkpoint = path
            self.model.save(path) # TODO: credo funzioni solo con sb3 per ora, da capire come gestire i checkpoints in generale
            self.logger.debug(f"SAC model checkpoint saved at: {path}")
            del self.model # remove to demonstrate saving and loading
        else:
            self.logger.warning("Checkpointing directory not specified. Model will not be saved.")
        
    def load_checkpoint(self):
        # load model checkpoint
        if self.best_model_checkpoint:
            self.model = self.model.load(self.best_model_checkpoint, env=self.env) # TODO: credo funzioni solo con sb3 per ora, da capire come gestire i checkpoints in generale
            self.logger.debug(f"SAC model checkpoint loaded from: {self.best_model_checkpoint}")
        else:
            self.logger.warning("No checkpoint path specified. Model will not be loaded.")

    def reset(self):
        # reset agent state if needed
        pass


# ============================================================================
# GYMNASIUM WRAPPER FOR DICT ACTION SPACE TO BOX CONVERSION
# ============================================================================


class SB3ActionWrapper(gym.ActionWrapper):
    """
    Converts a Dict action space to a Box action space for compatibility with Stable Baselines3.
    
    Stable Baselines3 algorithms (SAC, PPO, DQN, etc.) typically do not support Dict action spaces.
    This wrapper flattens a Dict action space (supporting mixed types like Box and Discrete) 
    into a single Box space and handles the conversion between Box actions (output by SB3 models) 
    and Dict actions (required by the environment).
    
    Supported subspace types:
    - Box: Continuous actions (e.g., motor control, continuous parameters)
    - Discrete: Discrete actions (e.g., mode selection, strategy choice). Converted via rounding.
    - MultiDiscrete: Multiple independent discrete choices
    - MultiBinary: Binary actions
    
    Example:
        >>> from gymnasium.spaces import Dict, Box, Discrete
        >>> env = gym.make('some-env-with-dict-actions')
        >>> wrapped_env = DictToBoxActionWrapper(env)
        >>> # Now wrapped_env.action_space is a Box that SB3 can work with
        >>> model = SAC("MlpPolicy", wrapped_env, ...)
    
    Attributes:
        _dict_action_space: The original Dict action space
        _key_order: List of keys in the Dict (for consistent ordering)
        _key_metadata: Dict mapping keys to metadata (type, bounds, shape, etc.)
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: A gymnasium environment with a Dict action space
            
        Raises:
            ValueError: If the action space is not a Dict or contains unsupported subspace types
        """
        if not isinstance(env.action_space, DictSpace):
            raise ValueError(
                f"DictToBoxActionWrapper requires a Dict action space, "
                f"got {type(env.action_space).__name__}"
            )
        
        super().__init__(env)
        
        self._dict_action_space = env.action_space
        self._key_order = sorted(self._dict_action_space.spaces.keys())
        
        # Build the flattened Box action space and track metadata for each key
        self._key_metadata = {}
        flat_lows = []
        flat_highs = []
        current_idx = 0
        
        for key in self._key_order:
            subspace = self._dict_action_space.spaces[key]
            
            if isinstance(subspace, Box):
                # Box space: continuous actions
                subspace_size = int(np.prod(subspace.shape))
                self._key_metadata[key] = {
                    'type': 'Box',
                    'start_idx': current_idx,
                    'end_idx': current_idx + subspace_size,
                    'shape': subspace.shape,
                    'low': subspace.low,
                    'high': subspace.high,
                }
                flat_lows.extend(subspace.low.flatten())
                flat_highs.extend(subspace.high.flatten())
                current_idx += subspace_size
                
            elif isinstance(subspace, Discrete):
                # Discrete space: map to [0, n-1] in Box form
                n = subspace.n
                self._key_metadata[key] = {
                    'type': 'Discrete',
                    'start_idx': current_idx,
                    'end_idx': current_idx + 1,
                    'n': n,
                }
                flat_lows.append(0.0)
                flat_highs.append(float(n - 1))
                current_idx += 1
                
            elif isinstance(subspace, MultiDiscrete):
                # MultiDiscrete space: each element is discrete in [0, nvec[i]-1]
                nvec = subspace.nvec
                subspace_size = len(nvec)
                self._key_metadata[key] = {
                    'type': 'MultiDiscrete',
                    'start_idx': current_idx,
                    'end_idx': current_idx + subspace_size,
                    'nvec': nvec,
                }
                for n in nvec:
                    flat_lows.append(0.0)
                    flat_highs.append(float(n - 1))
                current_idx += subspace_size
                
            elif isinstance(subspace, MultiBinary):
                # MultiBinary space: each element is binary [0, 1]
                if isinstance(subspace.n, int):
                    subspace_size = subspace.n
                else:
                    subspace_size = int(np.prod(subspace.shape))
                self._key_metadata[key] = {
                    'type': 'MultiBinary',
                    'start_idx': current_idx,
                    'end_idx': current_idx + subspace_size,
                    'size': subspace_size,
                }
                for _ in range(subspace_size):
                    flat_lows.append(0.0)
                    flat_highs.append(1.0)
                current_idx += subspace_size
                
            else:
                raise ValueError(
                    f"DictToBoxActionWrapper does not support space type {type(subspace).__name__} "
                    f"for key '{key}'. Supported types: Box, Discrete, MultiDiscrete, MultiBinary"
                )
        
        # Create the flattened Box action space
        self.action_space = Box(
            low=np.array(flat_lows, dtype=np.float32),
            high=np.array(flat_highs, dtype=np.float32),
            dtype=np.float32
        )
    
    def action(self, action):
        """
        Convert a flat Box action to a Dict action.
        
        Args:
            action: A numpy array from the flattened Box space
            
        Returns:
            A dictionary with keys from the original Dict space and values reshaped appropriately.
            Discrete values are converted from continuous via rounding.
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        dict_action = {}
        
        for key in self._key_order:
            metadata = self._key_metadata[key]
            start_idx = metadata['start_idx']
            end_idx = metadata['end_idx']
            action_type = metadata['type']
            
            # Extract the slice for this key
            key_action = action[start_idx:end_idx]
            
            if action_type == 'Box':
                # Box: reshape and clip to bounds
                shape = metadata['shape']
                key_action = key_action.reshape(shape)
                key_action = np.clip(key_action, metadata['low'], metadata['high'])
                
            elif action_type == 'Discrete':
                # Discrete: round to nearest integer and clip to [0, n-1]
                key_action = int(np.round(key_action[0]))
                key_action = np.clip(key_action, 0, metadata['n'] - 1)
                
            elif action_type == 'MultiDiscrete':
                # MultiDiscrete: round each element and clip to valid range
                nvec = metadata['nvec']
                key_action = np.round(key_action).astype(int)
                for i, n in enumerate(nvec):
                    key_action[i] = np.clip(key_action[i], 0, n - 1)
                
            elif action_type == 'MultiBinary':
                # MultiBinary: round to nearest integer (0 or 1)
                size = metadata['size']
                key_action = np.round(key_action[:size]).astype(int)
                key_action = np.clip(key_action, 0, 1)
            
            dict_action[key] = key_action
        
        return dict_action


# ============================================================================
# WRAPPER FOR DICT KEY NAME SANITIZATION (DOTS TO UNDERSCORES)
# ============================================================================

class DictKeyNameWrapper(gym.ObservationWrapper):
    """
    Sanitizes Dict observation and action keys by replacing dots with underscores.
    
    Stable Baselines3 does not accept dictionary keys containing dots because they
    cannot be used as valid Python variable names or module paths. This wrapper
    converts:
    - Observation keys: 'federation_1.building.0.T_indoor' → 'federation_1_building_0_T_indoor'
    - Action keys (reverse): 'federation_1_building_0_T_indoor' → 'federation_1.building.0.T_indoor'
    
    Example:
        >>> env = gym.make('some-env')
        >>> wrapped_env = DictKeyNameWrapper(env)
        >>> # Now obs and action keys are sanitized (dots → underscores)
    
    Attributes:
        _obs_key_mapping: Maps sanitized obs keys back to original keys
        _action_key_mapping: Maps original action keys to sanitized keys
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: A gymnasium environment with Dict observation and/or action spaces
        """
        super().__init__(env)
        
        # Create mapping for observation space if it's a Dict
        self._obs_key_mapping = {}
        if isinstance(env.observation_space, DictSpace):
            for key in env.observation_space.spaces.keys():
                sanitized_key = key.replace('.', '/')
                self._obs_key_mapping[sanitized_key] = key
            
            # Update observation space with sanitized keys
            new_obs_spaces = {}
            for key, space in env.observation_space.spaces.items():
                sanitized_key = key.replace('.', '/')
                new_obs_spaces[sanitized_key] = space
            self.observation_space = DictSpace(new_obs_spaces)
        
        # # Create mapping for action space if it's a Dict
        # self._action_key_mapping = {}
        # if isinstance(env.action_space, DictSpace):
        #     for key in env.action_space.spaces.keys():
        #         sanitized_key = key.replace('.', '/')
        #         self._action_key_mapping[key] = sanitized_key
            
        #     # Update action space with sanitized keys
        #     new_action_spaces = {}
        #     for key, space in env.action_space.spaces.items():
        #         sanitized_key = key.replace('.', '/')
        #         new_action_spaces[sanitized_key] = space
        #     self.action_space = DictSpace(new_action_spaces)
    
    def observation(self, obs):
        """
        Convert observation dict keys from original format to sanitized format.
        
        Example:
            {'federation_1.building.0.T_indoor': 20.0} 
            → 
            {'federation_1/building/0/T_indoor': 20.0}
        
        Args:
            obs: Observation from the environment (original keys with dots)
            
        Returns:
            Observation with sanitized keys (dots replaced with slashes)
        """
        if not isinstance(obs, dict):
            return obs
        
        sanitized_obs = {}
        for key, value in obs.items():
            sanitized_key = key.replace('.', '/')
            sanitized_obs[sanitized_key] = value
        
        return sanitized_obs
    
    def desanitize_observation(self, obs):
        """
        Convert observation dict keys from sanitized format back to original format with dots.
        
        This is useful when you need to pass observations to reward functions or other
        logic that expects the original key names.
        
        Example:
            {'federation_1/building/0/T_indoor': 20.0} 
            → 
            {'federation_1.building.0.T_indoor': 20.0}
        
        Args:
            obs: Observation with sanitized keys (slashes)
            
        Returns:
            Observation with original keys (dots)
        """
        if not isinstance(obs, dict):
            return obs
        
        original_obs = {}
        for key, value in obs.items():
            # If this is a sanitized key, convert it back using the mapping
            if key in self._obs_key_mapping:
                original_key = self._obs_key_mapping[key]
            else:
                # Fallback: just replace slashes with dots
                original_key = key.replace('/', '.')
            original_obs[original_key] = value
        
        return original_obs
    
    # def action(self, action):
        """
        Convert action dict keys from sanitized format back to original format.
        
        Example:
            {'federation_1/building/0/modulation': 0.5}
            →
            {'federation_1.building.0.modulation': 0.5}
        
        Args:
            action: Action from the agent (sanitized keys with slashes)
            
        Returns:
            Action with original keys (slashes replaced with dots)
        """
        if not isinstance(action, dict):
            return action
        
        original_action = {}
        for key, value in action.items():
            # If this is a sanitized key, convert it back to original
            if key in self._obs_key_mapping.values():
                # This shouldn't happen, but handle it gracefully
                original_action[key] = value
            else:
                # Replace slashes back with dots using the reverse mapping
                # We need to find which original key corresponds to this sanitized key
                original_key = None
                for orig_key, san_key in self._action_key_mapping.items():
                    if san_key == key:
                        original_key = orig_key
                        break
                
                if original_key:
                    original_action[original_key] = value
                else:
                    # Fallback: if not in mapping, just replace slashes with dots
                    # This handles cases where the action space might differ
                    restored_key = key.replace('/', '.')
                    original_action[restored_key] = value
        
        return original_action