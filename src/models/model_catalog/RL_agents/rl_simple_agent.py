"""
rl_simple_agent.py

Skeleton implementation for a simple RL agent providing a template for custom algorithms.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

from ...base_agent_rl import RLAgent

class RL_Simple_Agent(RLAgent):
    def __init__(self, env, logger=None, rl_task=None):
        super().__init__(env, logger, rl_task)
        self.logger.info(f"Initialized RL_Simple_Agent with rl_task: {rl_task}")
        # Initialize any additional attributes or models here based on algo_specs
    
    def act(self, obs):
        return super().act(obs)
    
    def compute_reward(self, obs, action):
        return super().compute_reward(obs, action)
    
  
    def online_training_loop(self):
        self.logger.info("Starting online training loop for RL_Simple_Agent:")
        # Implement the online training loop logic here
        return super().online_training_loop()
    
    def testing_loop(self):
        return super().testing_loop()