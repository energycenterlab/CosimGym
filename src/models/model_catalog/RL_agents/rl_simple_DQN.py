"""
rl_simple_DQN.py

Simple Deep Q-Network (DQN) implementation for reinforcement learning tasks.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

from ...base_agent_rl import RLAgent


import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pprint as pp
from gymnasium.wrappers import FlattenObservation
pp = pp.PrettyPrinter(indent=4)


# ---------- Replay Buffer ----------
class ReplayBuffer:
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


# ---------- Q Network ----------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------- DQN Agent ----------
@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    min_replay_size: int = 1_000
    target_update_freq: int = 1_000  # in environment steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    grad_clip_norm: float = 10.0


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, device: str = "cpu", cfg: DQNConfig = DQNConfig()):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.cfg = cfg

        self.q = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.buffer_size)

        self.best_model = None
        self.step_count = 0

    def epsilon(self):
        # Linear decay
        t = min(self.step_count, self.cfg.eps_decay_steps)
        frac = t / self.cfg.eps_decay_steps
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, state, deterministic=False):
        eps = self.epsilon()
        if random.random() < eps and not deterministic:
            return random.randrange(self.n_actions)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q(s)  # [1, n_actions]
        return int(torch.argmax(q_values, dim=1).item())

    def push(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.cfg.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(states_t).gather(1, actions_t)

        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.q_target(next_states_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.cfg.gamma * max_next_q * (1.0 - dones_t)

        loss = nn.SmoothL1Loss()(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        return float(loss.item())

    def maybe_update_target(self):
        if self.step_count % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def load_best_model(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q.state_dict())
    
    def save_model(self, path, last=False):
        # Create parent directory if it doesn't exist
        self.best_model = self.q.state_dict()
        if last:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.q.state_dict(), path)




class RL_Simple_DQN(RLAgent):
    def __init__(self, env, logger=None, rl_task=None):
        super().__init__(env, logger, rl_task)
        self.logger.debug(f"Initialized RL_Simple_DQN with rl_task: {pp.pformat(self.rl_task)}")
        # Initialize DQN from the structured RL dataclasses
        hp = self.rl_task.agent.hyperparameters          # RLHyperparametersConfig
        # self.logger.debug(f"Extracted hyperparameters from RL task: {pp.pformat(hp)}")
        exploration = self.rl_task.training.exploration   # RLExplorationConfig
        # self.logger.debug(f"Extracted exploration config from RL task: {pp.pformat(exploration)}")
        replay_buf = self.rl_task.training.replay_buffer  # RLReplayBufferConfig
        # self.logger.debug(f"Extracted replay buffer config from RL task: {pp.pformat(replay_buf)}")
        # self.logger.debug(f'obs_dim : {self.env.observation_space}, n_actions: {self.env.action_space}')
        self.env = FlattenObservation(self.env)  # Ensure observations are flat vectors
        # self.logger.debug(f'After FlattenObservation — action_space: {self.env.action_space}, n={getattr(self.env.action_space, "n", "N/A")}')
        self.check_pointing_config = self.rl_task.checkpointing  # RLCheckpointingConfig

        self.model = DQNAgent(
                obs_dim=self.env.observation_space.shape[0],
                n_actions=self.env.action_space[list(self.env.action_space.keys())[0]].n,
                device="cpu",  # or "cuda" if GPU is available
                cfg=DQNConfig(
                    gamma=hp.gamma,
                    lr=hp.learning_rate,
                    batch_size=hp.batch_size,
                    buffer_size=replay_buf.buffer_size,
                    min_replay_size=replay_buf.prefill_steps,
                    target_update_freq=hp.target_update_interval,
                    eps_start=exploration.initial_epsilon,
                    eps_end=exploration.final_epsilon,
                    eps_decay_steps=exploration.epsilon_decay_steps,
                    grad_clip_norm=hp.gradient_clip,
                )
            )
    
            
        self.logger.debug("DQNAgent initialized successfully within RL_Simple_DQN.")
    
    def act(self, obs, deterministic=False):
        action = self.model.act(obs, deterministic=deterministic)
        self.logger.debug(f"RL_Simple_DQN act() called with obs: {obs}, selected action: {action}")
        return action
    
  
    def online_training_loop(self):
        self.logger.debug("Starting online training loop for RL_Simple_Agent:")
        # obs_dim = int(np.prod(self.env.observation_space.shape))
        # n_actions = len(self.env.action_space)

        # agent = DQNAgent(obs_dim, n_actions, device='cpu')

        obs, _ = self.env.reset()
        self.obs= obs
        
        episode_reward = 0.0
        best_ep_reward = float('-inf')

        for step in range(self.rl_task.training.total_steps):
            self.model.step_count += 1
            last = False

            # (flatten state if needed)
            s = np.array(self.obs, dtype=np.float32).reshape(-1)

            action = self.act(s)


            obs, reward, terminated, truncated, _ = self._env_step(action)
            # self.logger.debug("7777777777777777777777")

            # ns = np.array(next_state, dtype=np.float32).reshape(-1)
            self.model.push(self.transition.obs, action, reward, self.transition.next_obs, self.transition.done)

            loss = self.model.update()
            self.model.maybe_update_target()

            episode_reward += reward
            self.obs = obs

            if terminated:
                # simple logging
                if step == self.rl_task.training.total_steps - 1:
                    last=True
                              
                if episode_reward > best_ep_reward:
                    best_ep_reward = episode_reward
                self.logger.info(f"Episode finished at step {step} with return {episode_reward:.1f} (best so far: {best_ep_reward:.1f}), loss={loss}")
                self.model.save_model(self.check_pointing_config.single_best_checkpoint, last=last)
                print(f"step={step:7d}  eps={self.model.epsilon():.3f}  ep_return={episode_reward:.1f}  loss={loss}")
                self.obs, _ = self.env.reset()
                episode_reward = 0.0
        
    def testing_loop(self):
        self.model.load_best_model(self.check_pointing_config.single_best_checkpoint)
        self.env.reset()
        for step in range(self.rl_task.test.total_steps):
            obs, reward, terminated, truncated, _ = self.env.step(self.act(self.obs, deterministic=True))
            self.obs = obs
            if terminated:
                self.obs, _ = self.env.reset()
