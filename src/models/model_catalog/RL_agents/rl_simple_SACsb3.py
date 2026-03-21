"""
rl_simple_SACsb3.py

Soft Actor-Critic (SAC) implementation using Stable Baselines3 for reinforcement learning scenarios.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from ...base_agent_rl import RLAgent, SB3ActionWrapper, DictKeyNameWrapper


class BestEpisodeCheckpointCallback(BaseCallback):
    """
    Save checkpoint whenever episode reward improves during learn().

    SB3's learn() returns the final model, not the best-by-reward model.
    This callback tracks episode returns from rollout rewards/dones and saves
    the best-performing checkpoint to the configured path.
    """

    def __init__(self, checkpoint_path, logger=None, verbose=0):
        super().__init__(verbose=verbose)
        self.checkpoint_path = checkpoint_path
        # BaseCallback exposes a read-only `logger` property from SB3 internals.
        # Keep application logger on a separate attribute.
        self.app_logger = logger
        self.current_episode_reward = 0.0
        self.best_episode_reward = -np.inf
        self.episode_count = 0
        self.saved_any = False

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        rewards_arr = np.asarray(rewards, dtype=float).reshape(-1)
        dones_arr = np.asarray(dones).reshape(-1)
        if rewards_arr.size == 0 or dones_arr.size == 0:
            return True

        # Single-env setup for this project (VecEnv wrapper still provides arrays).
        self.current_episode_reward += float(rewards_arr[0])

        if bool(dones_arr[0]):
            self.episode_count += 1
            ep_reward = float(self.current_episode_reward)
            if ep_reward > self.best_episode_reward:
                self.best_episode_reward = ep_reward
                self.model.save(self.checkpoint_path)
                self.saved_any = True
                if self.app_logger:
                    self.app_logger.info(
                        f"New best SAC checkpoint saved at episode {self.episode_count}: "
                        f"episode_reward={ep_reward:.6f} path={self.checkpoint_path}"
                    )
            self.current_episode_reward = 0.0

        return True


class RL_Simple_SACsb3(RLAgent):
    def __init__(self, env, logger=None, rl_task=None):
        super().__init__(env, logger, rl_task)
        
        hp = self.rl_task.agent.hyperparameters          # RLHyperparametersConfig
        replay_buf = self.rl_task.training.replay_buffer
        
        # Wrap the environment: first sanitize key names, then convert Dict actions to Box
        # IMPORTANT: Update self.env directly so the federate and reward functions can access the wrapped env
        self.env = DictKeyNameWrapper(self.env)  # Step 1: Convert 'federation_1.building.0.X' → 'federation_1/building/0/X'
        self.env = SB3ActionWrapper(self.env)    # Step 2: Convert Dict action space to Box

        self.model = SAC("MultiInputPolicy", self.env, learning_rate=hp.learning_rate, buffer_size=replay_buf.buffer_size, 
                         learning_starts=replay_buf.prefill_steps, batch_size=hp.batch_size, gamma=hp.gamma,
                           train_freq=self.rl_task.training.train_frequency, gradient_steps=self.rl_task.training.gradient_steps, ent_coef='auto',
                               target_update_interval=hp.target_update_interval, target_entropy='auto', seed=self.rl_task.seed)

    def _resolve_test_checkpoint(self):
        test_cfg = self.rl_task.test
        ckpt_cfg = self.rl_task.checkpointing

        if test_cfg is not None and getattr(test_cfg, "checkpoint_path", None):
            return test_cfg.checkpoint_path
        if ckpt_cfg is not None and getattr(ckpt_cfg, "single_best_checkpoint", None):
            return ckpt_cfg.single_best_checkpoint
        return None

    
    def act(self,  obs, deterministic=False):
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action

    
    def online_training_loop(self):
        
        self.logger.debug(f"Starting online training loop for RL_Simple_SACsb3: for a number of steps:{self.rl_task.training.total_steps}")
        checkpoint_path = self.rl_task.checkpointing.single_best_checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        best_callback = BestEpisodeCheckpointCallback(checkpoint_path, logger=self.logger)
        self.model.learn(
            total_timesteps=self.rl_task.training.total_steps,
            log_interval=4,
            callback=best_callback,
        )

        # Fallback safeguard: if no episode boundary was observed by callback,
        # save final model so testing always has a usable checkpoint.
        if not best_callback.saved_any:
            self.model.save(checkpoint_path)
            if self.logger:
                self.logger.warning(
                    "Best-checkpoint callback did not save any model during training. "
                    f"Saved final SAC model to {checkpoint_path} as fallback."
                )
        elif self.logger:
            self.logger.info(
                f"Training completed. Best episode reward={best_callback.best_episode_reward:.6f}. "
                f"Using checkpoint {checkpoint_path} for testing."
            )
        del self.model  # remove the model from memory after saving to free up resources
       

    def testing_loop(self):
        checkpoint_path = self._resolve_test_checkpoint()
        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint available for testing. "
                "Set reinforcement_learning_config.test.checkpoint_path or checkpointing.single_best_checkpoint."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for SAC testing: {checkpoint_path}")

        # Load the trained model
        self.model = SAC.load(checkpoint_path, env=self.env)
        self.logger.debug(f"Loaded model from checkpoint: {checkpoint_path}")
        # Initialize observation from environment reset
        self.obs, _ = self.env.reset()
        self.logger.debug(f" Starting testing loop for RL_Simple_SACsb3: for a number of steps:{self.rl_task.test.total_steps}")

        episode_reward = 0.0
        deterministic = True if self.rl_task.test is None else bool(self.rl_task.test.deterministic)
        
        for step in range(self.rl_task.test.total_steps):  # Run for a fixed number of steps or until termination
            # Get action from model using current observation
            self.logger.debug(f"Testing step {step}: Current observation: {self.obs}")
            
            action = self.act(self.obs, deterministic=deterministic)
            self.logger.debug(f"Testing step {step}: Predicted action: {action}")
            
            # Step the environment with the action
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                self.logger.info(f"Episode finished after {step+1} steps with total reward: {episode_reward}")
                episode_reward = 0.0
                self.obs, _ = self.env.reset()  # Reset for next episode



   
