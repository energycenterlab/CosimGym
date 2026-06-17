"""
rl_simple_rllib.py

RLlib PPO agent for CosimGym's HELICS-driven co-simulation loop.

Uses RLlib's PPO RLModule (neural network) standalone — no Algorithm, no
env runners, no ray serialization. HELICS drives stepping; this agent
collects transitions, computes GAE, and runs PPO clipped-surrogate
updates manually.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-06-17

"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from ...base_agent_rl import RLAgent, SB3ActionWrapper, DictKeyNameWrapper
from .components import CheckpointManager

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from gymnasium.wrappers import FlattenObservation


class RL_Simple_RLlib(RLAgent):

    def __init__(self, env, logger=None, rl_task=None):
        super().__init__(env, logger, rl_task)

        hp = self.rl_task.agent.hyperparameters
        params = self.rl_task.agent.params or {}
        self.checkpoints = CheckpointManager(
            self.rl_task.experiment, self.rl_task.run, logger=logger
        )

        self.env = DictKeyNameWrapper(self.env)
        self.env = SB3ActionWrapper(self.env)
        self.env = FlattenObservation(self.env)

        algorithm = (self.rl_task.agent.algorithm or "PPO").upper()
        if algorithm != "PPO":
            raise ValueError(
                f"rl_simple_rllib currently supports PPO only, got '{algorithm}'"
            )

        self.gamma = hp.gamma if hp.gamma is not None else 0.99
        self.lr = hp.learning_rate if hp.learning_rate is not None else 3e-4
        self.clip_param = params.get("clip_param", 0.2)
        self.vf_loss_coeff = params.get("vf_loss_coeff", 1.0)
        self.entropy_coeff = params.get("entropy_coeff", 0.0)
        self.gae_lambda = params.get("lambda", 0.95)
        self.num_epochs = params.get("num_epochs", 4)
        self.minibatch_size = params.get("minibatch_size", 64)
        self.max_grad_norm = params.get("grad_clip", 0.5)
        self.batch_size = params.get("train_batch_size", 128)

        hidden = list(hp.net_arch) if hp.net_arch else [64, 64]

        obs_space = self.env.observation_space
        act_space = self.env.action_space
        spec = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space,
            action_space=act_space,
            model_config={"fcnet_hiddens": hidden},
        )
        self.module = spec.build()
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)

        self.logger.info(
            f"RLlib PPO module initialized (standalone): "
            f"obs={obs_space.shape}, act={act_space.shape}, "
            f"hidden={hidden}, lr={self.lr}, gamma={self.gamma}"
        )

    def act(self, obs, deterministic=False):
        obs_t = torch.tensor(
            np.array(obs, dtype=np.float32).reshape(1, -1), dtype=torch.float32
        )
        with torch.no_grad():
            out = self.module.forward_inference({"obs": obs_t})
        dist_inputs = out["action_dist_inputs"]
        act_dim = self.env.action_space.shape[0]
        if deterministic:
            return dist_inputs[:, :act_dim].numpy().flatten()
        dist_cls = self.module.get_inference_action_dist_cls()
        dist = dist_cls.from_logits(dist_inputs)
        return dist.sample().numpy().flatten()

    def _collect_rollout(self, num_steps):
        """Collect a batch of transitions from the env."""
        obs_list, act_list, rew_list, done_list, logp_list, val_list = (
            [], [], [], [], [], []
        )
        obs = self.obs
        for _ in range(num_steps):
            obs_t = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)

            with torch.no_grad():
                fwd = self.module.forward_exploration({"obs": obs_t})
            dist_inputs = fwd["action_dist_inputs"]
            dist_cls = self.module.get_exploration_action_dist_cls()
            dist = dist_cls.from_logits(dist_inputs)
            action_t = dist.sample()
            logp = dist.logp(action_t)
            vf = fwd.get("vf_preds", torch.zeros(1))

            action = action_t.numpy().flatten()
            next_obs, reward, terminated, truncated, info = self._env_step(action)

            obs_list.append(obs.copy())
            act_list.append(action.copy())
            rew_list.append(float(reward))
            done_list.append(terminated or truncated)
            logp_list.append(logp.item())
            val_list.append(vf.item())

            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset()

        self.obs = obs

        with torch.no_grad():
            obs_t = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            fwd = self.module.forward_exploration({"obs": obs_t})
            last_val = fwd.get("vf_preds", torch.zeros(1)).item()

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(act_list, dtype=np.float32),
            "rewards": np.array(rew_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=bool),
            "old_logp": np.array(logp_list, dtype=np.float32),
            "values": np.array(val_list, dtype=np.float32),
            "last_value": last_val,
        }

    def _compute_gae(self, rewards, values, dones, last_value):
        """Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def _ppo_update(self, batch):
        """Run PPO clipped-surrogate update over minibatches."""
        obs_t = torch.tensor(batch["obs"], dtype=torch.float32)
        act_t = torch.tensor(batch["actions"], dtype=torch.float32)
        old_logp_t = torch.tensor(batch["old_logp"], dtype=torch.float32)
        adv_t = torch.tensor(batch["advantages"], dtype=torch.float32)
        ret_t = torch.tensor(batch["returns"], dtype=torch.float32)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = len(obs_t)
        total_loss_val = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)
                mb_idx = idx[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                fwd = self.module.forward_train({"obs": mb_obs})
                dist_inputs = fwd["action_dist_inputs"]
                vf_preds = fwd.get("vf_preds", torch.zeros(len(mb_idx)))

                dist_cls = self.module.get_train_action_dist_cls()
                dist = dist_cls.from_logits(dist_inputs)
                new_logp = dist.logp(mb_act)
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                vf_loss = F.mse_loss(vf_preds.squeeze(-1), mb_ret)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_loss_coeff * vf_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.module.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

                total_loss_val += loss.item()
                num_updates += 1

        return total_loss_val / max(num_updates, 1)

    def online_training_loop(self):
        total_steps = self.rl_task.run.train.total_steps
        checkpoint_path = self.checkpoints.ensure_dir()
        best_reward = float("-inf")
        steps_done = 0

        self.obs, _ = self.env.reset()
        self.logger.info(f"RLlib PPO training: target {total_steps} env steps")

        while steps_done < total_steps:
            collect_n = min(self.batch_size, total_steps - steps_done)
            rollout = self._collect_rollout(collect_n)
            advantages, returns = self._compute_gae(
                rollout["rewards"],
                rollout["values"],
                rollout["dones"],
                rollout["last_value"],
            )
            rollout["advantages"] = advantages
            rollout["returns"] = returns

            avg_loss = self._ppo_update(rollout)
            steps_done += collect_n

            ep_mask = rollout["dones"]
            ep_reward = rollout["rewards"].sum()
            info = (
                f"step={steps_done}/{total_steps}  "
                f"loss={avg_loss:.4f}  batch_reward={ep_reward:.2f}"
            )
            print(info)
            self.logger.info(info)

            if ep_reward > best_reward:
                best_reward = ep_reward
                if checkpoint_path:
                    self._save_module(checkpoint_path)

        if checkpoint_path:
            self._save_module(checkpoint_path)
            self.logger.info(f"Final RLlib checkpoint saved to {checkpoint_path}")

    def testing_loop(self):
        checkpoint_path = self.checkpoints.test_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_module(checkpoint_path)
            self.logger.info(f"Loaded RLlib checkpoint from {checkpoint_path}")

        self.obs, _ = self.env.reset()
        episode_reward = 0.0
        deterministic = True
        if self.rl_task.run.test is not None:
            deterministic = bool(
                getattr(self.rl_task.run.test, "deterministic", True)
            )

        for step in range(self.rl_task.run.test.total_steps):
            action = self.act(self.obs, deterministic=deterministic)
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            if terminated or truncated:
                self.logger.info(
                    f"Test episode done at step {step}: reward={episode_reward:.2f}"
                )
                episode_reward = 0.0
                self.obs, _ = self.env.reset()

    def _save_module(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.module.state_dict(), path)

    def _load_module(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.module.load_state_dict(state)
