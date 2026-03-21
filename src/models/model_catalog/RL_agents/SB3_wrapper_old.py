

import time
from pathlib import Path
from typing import Any

from src.models.pre_defined_RL_agents.base_RL_agent import (
    BaseRLAgent,
    Action,
    ActionDict,
    InfoDict,
    MetricsDict,
    Observation,
    ObsDict,
    Transition,
)

try:
    from src.models.pre_defined_RL_agents.ma_RL_agent import MA_Transition
except ImportError:
    MA_Transition = None


class SB3AgentWrapper(BaseRLAgent):
    """
    Example: Wrapper for Stable-Baselines3 algorithms.
    
    Demonstrates how to integrate SB3 (PPO, SAC, DQN, etc.) with BaseRLAgent interface.
    
    Features:
    - Wraps any SB3 algorithm (PPO, SAC, DQN, A2C, TD3, etc.)
    - Maintains compatibility with BaseRLAgent lifecycle
    - Supports both online and offline training modes
    - Handles saving/loading of SB3 models
    
    Requirements:
        pip install stable-baselines3
    
    Usage:
        from stable_baselines3 import PPO
        
        # Create SB3 model
        sb3_model = PPO("MlpPolicy", env, verbose=1)
        
        # Wrap it
        agent = SB3AgentWrapper(sb3_model, name="PPO_Agent")
        
        # Use standard BaseRLAgent interface
        agent.train_loop_online(env, max_steps=10000)
    """
    
    def __init__(
        self,
        sb3_model: Any,
        *,
        name: str = "SB3Agent",
        batch_size: int = 2048,
        **kwargs,
    ):
        """
        Args:
            sb3_model: A Stable-Baselines3 model instance (PPO, SAC, DQN, etc.)
            name: Agent identifier
            batch_size: Rollout batch size for on-policy algorithms
            **kwargs: Passed to BaseRLAgent
        """
        super().__init__(name=name, **kwargs)
        
        # Check if SB3 is available
        try:
            from stable_baselines3.common.base_class import BaseAlgorithm
        except ImportError:
            raise ImportError(
                "Stable-Baselines3 not found. Install with: pip install stable-baselines3"
            )
        
        self.sb3_model = sb3_model
        self.batch_size = batch_size
        
        # Track if we're using on-policy (PPO, A2C) or off-policy (SAC, DQN, TD3)
        self._is_on_policy = self._check_on_policy(sb3_model)
        
        # For on-policy: track steps since last update
        self._steps_since_update = 0
    
    def _check_on_policy(self, model: Any) -> bool:
        """Check if SB3 model is on-policy algorithm."""
        model_class_name = model.__class__.__name__
        on_policy_algos = {"PPO", "A2C", "RecurrentPPO"}
        return model_class_name in on_policy_algos
    
    def act(
        self,
        obs: Observation | ObsDict,
        *,
        explore: bool | None = None,
    ) -> tuple[Action | ActionDict, InfoDict]:
        """
        Compute action using SB3 model.
        
        Args:
            obs: Observation from environment
            explore: Whether to use stochastic policy (None = use current mode)
        
        Returns:
            action: Action from SB3 model
            info: Additional info (can include state values, log probs, etc.)
        """
        if explore is None:
            explore = (self._mode == "train")
        
        # SB3's predict() method
        # deterministic=True for greedy/eval, False for stochastic/train
        action, _states = self.sb3_model.predict(
            obs,
            deterministic=not explore,
        )
        
        # Extract additional info if available (e.g., for logging)
        info: InfoDict = {}
        
        # Some SB3 algorithms provide value predictions
        # This is optional and depends on the algorithm
        try:
            # For PPO/A2C, we can get value estimates
            if hasattr(self.sb3_model.policy, 'predict_values'):
                import torch
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs).unsqueeze(0)
                    if hasattr(self.sb3_model.policy, 'obs_to_tensor'):
                        obs_tensor = self.sb3_model.policy.obs_to_tensor(obs)[0]
                    value = self.sb3_model.policy.predict_values(obs_tensor)
                    info['value'] = float(value.cpu().numpy())
        except Exception:
            pass  # Not all algorithms support this
        
        return action, info
    
    def observe(self, transition: Transition | MA_Transition) -> None:
        """
        SB3 handles experience collection internally.
        
        For online training with train_loop_online(), SB3's learn() method
        will collect experiences automatically. This method is a no-op.
        
        For offline training or custom workflows, you could implement
        custom replay buffer integration here.
        """
        # SB3 manages its own replay buffer / rollout buffer
        # No need to manually store transitions
        self._steps_since_update += 1
    
    def can_update(self) -> bool:
        """
        Check if ready to update.
        
        For on-policy algorithms (PPO, A2C): wait until rollout buffer is full
        For off-policy algorithms (SAC, DQN): can update anytime (if replay buffer has data)
        """
        if self._is_on_policy:
            # On-policy: update when we've collected enough steps
            return self._steps_since_update >= self.batch_size
        else:
            # Off-policy: check if replay buffer has minimum samples
            if hasattr(self.sb3_model, 'replay_buffer'):
                min_samples = getattr(self.sb3_model, 'learning_starts', 100)
                return self.sb3_model.replay_buffer.size() >= min_samples
            return True
    
    def update(self, *, num_updates: int = 1) -> MetricsDict:
        """
        Perform learning update using SB3's internal training.
        
        Note: For SB3, the standard workflow is to use model.learn() which
        handles both environment interaction and updates. This method provides
        a way to trigger updates in the BaseRLAgent framework.
        
        Args:
            num_updates: Number of gradient steps
        
        Returns:
            Metrics from training (loss, etc.)
        """
        # For SB3, we typically don't call update separately
        # Instead, we use the custom train loop below
        
        # However, if you want to support offline updates or fine-tuning:
        # You could implement custom gradient steps here using the model's components
        
        metrics: MetricsDict = {
            "sb3_num_timesteps": int(self.sb3_model.num_timesteps),
        }
        
        # Reset step counter for on-policy algorithms
        if self._is_on_policy:
            self._steps_since_update = 0
        
        self._updates += num_updates
        return metrics
    
    def train_loop_online(
        self,
        env: Any,
        max_steps: int,
        *,
        update_every: int = 1,
        updates_per_step: int = 1,
        eval_every: int | None = None,
        eval_episodes: int = 5,
        log_every: int = 1000,
    ) -> MetricsDict:
        """
        Online training using SB3's native learn() method.
        
        This overrides the base implementation to use SB3's optimized
        training loop, which handles environment interaction and updates
        more efficiently than the generic BaseRLAgent loop.
        
        Args:
            env: Gym-compatible environment
            max_steps: Total training timesteps
            Other args are mostly for compatibility (SB3 has its own scheduling)
        
        Returns:
            Final metrics
        """
        self.set_mode("train")
        self._training_start_time = time.time()
        
        # Use SB3's native learn() method
        # This is more efficient than manually stepping through the environment
        self.sb3_model.learn(
            total_timesteps=max_steps,
            log_interval=max(1, log_every // 1000),  # SB3 logs per episode by default
        )
        
        # Update our counters
        self._env_steps += max_steps
        self._updates = int(self.sb3_model.num_timesteps)
        
        # Optional evaluation
        if eval_every:
            eval_metrics = self.evaluate(env, episodes=eval_episodes)
            print(f"[{self.name}] Final Eval: {eval_metrics}")
            self._metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        
        return self.get_metrics()
    
    def state_dict(self) -> dict[str, Any]:
        """
        Return state including SB3 model path.
        
        Note: SB3 models are saved separately using save()/load() methods.
        This returns metadata for checkpoint tracking.
        """
        state = super().state_dict()
        state["sb3_model_class"] = self.sb3_model.__class__.__name__
        state["sb3_num_timesteps"] = int(self.sb3_model.num_timesteps)
        state["batch_size"] = self.batch_size
        return state
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load metadata from state dict."""
        super().load_state_dict(state)
        self.batch_size = state.get("batch_size", self.batch_size)
    
    def save(self, path: str | Path) -> None:
        """
        Save SB3 model and metadata.
        
        SB3 models are saved in their native format (.zip).
        Agent metadata is saved separately (.pkl).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save SB3 model
        model_path = path.with_suffix('.zip')
        self.sb3_model.save(model_path)
        
        # Save metadata
        import pickle
        meta_path = path.with_suffix('.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(self.state_dict(), f)
        
        print(f"[{self.name}] Saved SB3 model to {model_path}")
        print(f"[{self.name}] Saved metadata to {meta_path}")
    
    def load(self, path: str | Path) -> None:
        """
        Load SB3 model and metadata.
        
        Args:
            path: Base path (will load .zip for model and .pkl for metadata)
        """
        path = Path(path)
        
        # Load SB3 model
        model_path = path.with_suffix('.zip')
        if not model_path.exists():
            raise FileNotFoundError(f"SB3 model not found: {model_path}")
        
        # Need to know the model class to load
        # Load metadata first to get model class
        import pickle
        meta_path = path.with_suffix('.pkl')
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                state = pickle.load(f)
            self.load_state_dict(state)
        
        # Load the SB3 model (requires model class)
        # User should create the wrapper with correct model class first
        self.sb3_model = self.sb3_model.__class__.load(model_path)
        
        print(f"[{self.name}] Loaded SB3 model from {model_path}")
        print(f"[{self.name}] Loaded metadata from {meta_path}")



