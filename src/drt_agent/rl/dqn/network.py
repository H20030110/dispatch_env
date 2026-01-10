from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """最小版 MLP Q 网络。"""

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: List[int]) -> None:
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
