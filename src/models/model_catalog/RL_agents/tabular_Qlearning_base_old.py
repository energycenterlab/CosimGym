


from typing import Any
import numpy as np

from src.models.base_RL_agent import BaseRLAgent
from src.models.storage import ReplayBuffer
from src.models.types import (
    Observation,
    ObsDict,
    Action,
    ActionDict,
    InfoDict,
    Transition,
    MA_Transition,
    MetricsDict,
)


class ExampleTabularQAgent(BaseRLAgent):
    """
    Example: Tabular Q-learning agent (single-agent, discrete obs/action).
    
    Demonstrates:
    - Storage integration (ReplayBuffer)
    - Simple learning update
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        *,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        # Q-table
        self.q_table = np.zeros((num_states, num_actions))
        
        # Replay buffer
        self.buffer: ReplayBuffer[Transition] = ReplayBuffer(buffer_capacity, seed=self.seed)
    
    def act(
        self,
        obs: Observation | ObsDict,
        *,
        explore: bool | None = None,
    ) -> tuple[Action | ActionDict, InfoDict]:
        """Epsilon-greedy action selection."""
        if explore is None:
            explore = (self._mode == "train")
        
        # Assume obs is discrete state index
        state = int(obs)
        
        if explore and self._rng.random() < self.epsilon:
            action = int(self._rng.integers(0, self.num_actions))
        else:
            action = int(np.argmax(self.q_table[state]))
        
        return action, {"q_values": self.q_table[state].copy()}
    
    def observe(self, transition: Transition | MA_Transition) -> None:
        """Store transition in replay buffer."""
        if isinstance(transition, MA_Transition):
            raise NotImplementedError("Tabular Q-learning is single-agent only")
        
        self.buffer.add(transition)
    
    def can_update(self) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= self.batch_size
    
    def update(self, *, num_updates: int = 1) -> MetricsDict:
        """Sample batch and perform Q-learning update."""
        total_loss = 0.0
        
        for _ in range(num_updates):
            # Sample batch
            batch_transitions = self.buffer.sample(self.batch_size)
            
            # Update Q-values
            batch_loss = 0.0
            for trans in batch_transitions:
                s = int(trans.obs)
                a = int(trans.action)
                r = trans.reward
                s_next = int(trans.next_obs)
                done = trans.done
                
                # Q-learning target
                if done:
                    target = r
                else:
                    target = r + self.gamma * np.max(self.q_table[s_next])
                
                # TD error
                td_error = target - self.q_table[s, a]
                
                # Update
                self.q_table[s, a] += self.lr * td_error
                
                batch_loss += td_error ** 2
            
            total_loss += batch_loss / self.batch_size
            self._updates += 1
        
        return {"loss": total_loss / num_updates}
    
    def state_dict(self) -> dict[str, Any]:
        """Include Q-table in state."""
        state = super().state_dict()
        state["q_table"] = self.q_table.copy()
        return state
    
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load Q-table from state."""
        super().load_state_dict(state)
        if "q_table" in state:
            self.q_table = state["q_table"].copy()

