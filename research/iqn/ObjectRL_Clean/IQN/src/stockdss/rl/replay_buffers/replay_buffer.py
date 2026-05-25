"""
Replay buffer for off-policy value-based RL agents such as DQN/IQN.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
