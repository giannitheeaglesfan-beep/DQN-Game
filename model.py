"""DQN network, experience replay buffer, and training agent.

RL concepts:
    - Experience replay (`ReplayBuffer`) breaks the correlation between
      consecutive transitions by sampling random past experience, which
      stabilizes training for a bootstrapped, off-policy method like DQN.
    - A separate target network (`target_net`) provides stable Q-value
      targets for a number of steps at a time; without it, chasing a
      constantly-moving target (the same network used for both prediction
      and bootstrapping) tends to diverge.
    - Epsilon-greedy action selection balances exploration (random actions)
      against exploitation (greedy actions w.r.t. current Q estimates).
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 8
ACTION_DIM = 4


class DQN(nn.Module):
    """MLP mapping an 8-float battle state to 4 Q-values, one per move."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity ring buffer of (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Wraps policy/target networks, action selection, and the Q-learning update rule."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        lr: float = 1e-3,
        buffer_capacity: int = 50_000,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.sync_target()
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: explore with probability epsilon, else act greedily."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch_size: int, gamma: float = 0.99) -> float | None:
        """Sample a batch and take one gradient step toward the Bellman target.

        Returns the loss value, or None if the buffer doesn't yet have enough
        transitions to sample a full batch.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + gamma * next_q_values * (1.0 - dones_t)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def sync_target(self) -> None:
        """Copy policy network weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.sync_target()
