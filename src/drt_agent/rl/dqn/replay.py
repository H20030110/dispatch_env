from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray


class ReplayBuffer:
    """最小版经验回放。存 obs + mask，方便做 action masking 的 target。"""

    def __init__(self, capacity: int, obs_dim: int, num_actions: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)

        self.rng = np.random.default_rng(int(seed))

        self._ptr = 0
        self._size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.masks = np.zeros((capacity, num_actions), dtype=np.float32)
        self.next_masks = np.zeros((capacity, num_actions), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.dones[i] = 1.0 if done else 0.0
        self.masks[i] = mask
        self.next_masks[i] = next_mask

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        assert self._size > 0
        idx = self.rng.integers(0, self._size, size=int(batch_size))
        return Batch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            dones=self.dones[idx],
            masks=self.masks[idx],
            next_masks=self.next_masks[idx],
        )
