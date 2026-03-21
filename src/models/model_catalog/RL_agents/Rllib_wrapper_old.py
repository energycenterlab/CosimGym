
from typing import Any
from pathlib import Path
import time

from .base_rl_agent import BaseRLAgent, Observation, ObsDict, Action, ActionDict, InfoDict, Transition, MA_Transition, MetricsDict


class RLlibAgentWrapper(BaseRLAgent):
    """
    Example: Wrapper for Ray RLlib algorithms.
    
    Demonstrates how to integrate RLlib (PPO, DQN, SAC, APPO, etc.) with BaseRLAgent interface.
    
    Features:
    - Wraps any RLlib algorithm (PPO, DQN, SAC, APPO, IMPALA, etc.)
    - Maintains compatibility with BaseRLAgent lifecycle
    - Supports distributed training
    - Handles saving/loading of RLlib checkpoints
    - Compatible with RLlib's extensive configuration system
    
    Requirements:
        pip install "ray[rllib]"
    
    Usage:
        from ray.rllib.algorithms.ppo import PPOConfig
        
        # Create RLlib algorithm
        config = (
            PPOConfig()
            .environment(env="CartPole-v1")
            .training(lr=0.0003, train_batch_size=4000)
            .rollouts(num_rollout_workers=2)
        )
        rllib_algo = config.build()
        
        # Wrap it
        agent = RLlibAgentWrapper(rllib_algo, name="PPO_RLlib")
        
        # Use standard BaseRLAgent interface
        agent.train_loop_online(env, max_steps=10000)
    """
    
    def __init__(
        self,
        rllib_algorithm: Any,
        *,
        name: str = "RLlibAgent",
        use_rllib_training: bool = True,
        **kwargs,
    ):
        """
        Args:
            rllib_algorithm: A Ray RLlib Algorithm instance (PPO, DQN, SAC, etc.)
            name: Agent identifier
            use_rllib_training: If True, use RLlib's native train() method
                               If False, use BaseRLAgent's step-by-step training
            **kwargs: Passed to BaseRLAgent
        """
        super().__init__(name=name, **kwargs)
        
        # Check if RLlib is available
        try:
            from ray.rllib.algorithms.algorithm import Algorithm
        except ImportError:
            raise ImportError(
                "Ray RLlib not found. Install with: pip install 'ray[rllib]'"
            )
        
        self.rllib_algorithm = rllib_algorithm
        self.use_rllib_training = use_rllib_training
        
        # Get algorithm name
        self._algo_name = rllib_algorithm.__class__.__name__
        
        # Track training iterations
        self._training_iteration = 0
    
    def act(
        self,
        obs: Observation | ObsDict,
        *,
        explore: bool | None = None,
    ) -> tuple[Action | ActionDict, InfoDict]:
        """
        Compute action using RLlib algorithm.
        
        Args:
            obs: Observation from environment
            explore: Whether to explore (None = use current mode)
        
        Returns:
            action: Action from RLlib policy
            info: Additional info (can include state values, action logits, etc.)
        """
        if explore is None:
            explore = (self._mode == "train")
        
        # RLlib's compute_single_action method
        # explore=True for stochastic, False for deterministic
        try:
            action = self.rllib_algorithm.compute_single_action(
                observation=obs,
                explore=explore,
            )
        except Exception as e:
            # Fallback for older RLlib API
            action = self.rllib_algorithm.compute_action(
                observation=obs,
                explore=explore,
            )
        
        # Extract additional info if available
        info: InfoDict = {}
        
        # Try to get action distribution info
        try:
            # Get policy for additional info
            policy = self.rllib_algorithm.get_policy()
            if policy is not None:
                # Note: This requires more complex handling for full distribution info
                # For simplicity, we just note it's available
                info['policy_name'] = policy.config.get('model', {}).get('custom_model', 'default')
        except Exception:
            pass  # Not all algorithms/configs support this
        
        return action, info
    
    def observe(self, transition: Transition | MA_Transition) -> None:
        """
        RLlib handles experience collection internally.
        
        For online training with train_loop_online(), RLlib's train() method
        will collect experiences automatically through its workers.
        This method is a no-op for standard RLlib usage.
        
        For custom workflows, you could implement offline data ingestion here.
        """
        # RLlib manages its own experience collection through RolloutWorkers
        # No need to manually store transitions in standard usage
        pass
    
    def can_update(self) -> bool:
        """
        Check if ready to update.
        
        For RLlib, this is typically always True since RLlib manages
        its own sampling and buffer logic internally.
        """
        return True
    
    def update(self, *, num_updates: int = 1) -> MetricsDict:
        """
        Perform learning update using RLlib's training.
        
        Args:
            num_updates: Number of training iterations
        
        Returns:
            Metrics from training
        """
        metrics: MetricsDict = {}
        
        for _ in range(num_updates):
            # RLlib's train() method performs one iteration of training
            result = self.rllib_algorithm.train()
            
            # Extract key metrics
            # RLlib returns a large dict of metrics
            metrics.update({
                "episode_reward_mean": result.get("episode_reward_mean", 0.0),
                "episode_len_mean": result.get("episode_len_mean", 0.0),
                "episodes_this_iter": result.get("episodes_this_iter", 0),
                "timesteps_this_iter": result.get("timesteps_this_iter", 0),
                "training_iteration": result.get("training_iteration", 0),
            })
            
            # Add algorithm-specific metrics
            if "info" in result and "learner" in result["info"]:
                learner_info = result["info"]["learner"]
                if "default_policy" in learner_info:
                    policy_info = learner_info["default_policy"]
                    if "learner_stats" in policy_info:
                        stats = policy_info["learner_stats"]
                        metrics.update({
                            "policy_loss": stats.get("total_loss", 0.0),
                            "vf_loss": stats.get("vf_loss", 0.0),
                            "entropy": stats.get("entropy", 0.0),
                        })
            
            self._training_iteration = result.get("training_iteration", self._training_iteration + 1)
            self._updates += 1
        
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
        Online training using RLlib's native training loop.
        
        This overrides the base implementation to use RLlib's optimized
        distributed training, which handles environment interaction through
        its RolloutWorkers.
        
        Args:
            env: Environment (not directly used; RLlib uses its own env config)
            max_steps: Total training timesteps
            Other args are mostly for compatibility
        
        Returns:
            Final metrics
        """
        self.set_mode("train")
        self._training_start_time = time.time()
        
        if self.use_rllib_training:
            # Use RLlib's native training loop
            # Train until we reach max_steps
            current_timesteps = 0
            iteration = 0
            
            while current_timesteps < max_steps:
                result = self.rllib_algorithm.train()
                current_timesteps = result.get("timesteps_total", 0)
                iteration = result.get("training_iteration", iteration + 1)
                
                # Logging
                if iteration % max(1, log_every // 1000) == 0:
                    print(
                        f"[{self.name}] Iteration {iteration} | "
                        f"Timesteps: {current_timesteps}/{max_steps} | "
                        f"Reward: {result.get('episode_reward_mean', 0.0):.2f}"
                    )
                
                # Store metrics
                self._metrics.update({
                    "episode_reward_mean": result.get("episode_reward_mean", 0.0),
                    "episode_len_mean": result.get("episode_len_mean", 0.0),
                })
                
                # Optional evaluation
                if eval_every and current_timesteps % eval_every < result.get("timesteps_this_iter", 1):
                    # RLlib handles evaluation through evaluation_config
                    # For custom evaluation, use the evaluate() method
                    if hasattr(env, 'reset'):  # If env is provided
                        eval_metrics = self.evaluate(env, episodes=eval_episodes)
                        print(f"[{self.name}] Eval: {eval_metrics}")
                        self._metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                        self.set_mode("train")
            
            self._env_steps = current_timesteps
            self._updates = iteration
        else:
            # Fallback to base class step-by-step training
            # (Not recommended for RLlib, but provided for compatibility)
            return super().train_loop_online(
                env, max_steps, 
                update_every=update_every,
                updates_per_step=updates_per_step,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                log_every=log_every,
            )
        
        return self.get_metrics()
    
    def state_dict(self) -> dict[str, Any]:
        """
        Return state including RLlib checkpoint path.
        
        Note: RLlib models are saved using checkpoints.
        This returns metadata for tracking.
        """
        state = super().state_dict()
        state["rllib_algo_name"] = self._algo_name
        state["training_iteration"] = self._training_iteration
        state["use_rllib_training"] = self.use_rllib_training
        return state
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load metadata from state dict."""
        super().load_state_dict(state)
        self._training_iteration = state.get("training_iteration", 0)
        self.use_rllib_training = state.get("use_rllib_training", True)
    
    def save(self, path: str | Path) -> None:
        """
        Save RLlib checkpoint and metadata.
        
        RLlib saves checkpoints as directories with multiple files.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save RLlib checkpoint
        checkpoint_dir = self.rllib_algorithm.save(str(path.parent))
        
        # Save metadata
        import pickle
        meta_path = path.with_suffix('.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(self.state_dict(), f)
        
        print(f"[{self.name}] Saved RLlib checkpoint to {checkpoint_dir}")
        print(f"[{self.name}] Saved metadata to {meta_path}")
    
    def load(self, path: str | Path) -> None:
        """
        Load RLlib checkpoint and metadata.
        
        Args:
            path: Path to checkpoint directory or metadata file
        """
        path = Path(path)
        
        # Load metadata if available
        import pickle
        meta_path = path.with_suffix('.pkl')
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                state = pickle.load(f)
            self.load_state_dict(state)
        
        # Load RLlib checkpoint
        # Path should be to checkpoint directory or parent containing checkpoint
        if path.is_dir():
            checkpoint_path = path
        else:
            checkpoint_path = path.parent
        
        # Find the actual checkpoint directory (RLlib creates subdirs)
        checkpoint_dirs = sorted(checkpoint_path.glob("checkpoint_*"))
        if checkpoint_dirs:
            actual_checkpoint = checkpoint_dirs[-1]  # Latest checkpoint
            self.rllib_algorithm.restore(str(actual_checkpoint))
            print(f"[{self.name}] Loaded RLlib checkpoint from {actual_checkpoint}")
        else:
            raise FileNotFoundError(f"No RLlib checkpoint found in {checkpoint_path}")
        
        if meta_path.exists():
            print(f"[{self.name}] Loaded metadata from {meta_path}")
    
    def close(self) -> None:
        """Clean up RLlib resources."""
        super().close()
        try:
            self.rllib_algorithm.stop()
            print(f"[{self.name}] RLlib algorithm stopped")
        except Exception as e:
            print(f"[{self.name}] Error stopping RLlib algorithm: {e}")