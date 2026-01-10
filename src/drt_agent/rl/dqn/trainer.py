from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import trange

from drt_agent.common.utils import set_global_seed
from drt_agent.common.logger import JsonlLogger
from drt_agent.rl.dqn.network import QNetwork
from drt_agent.rl.dqn.replay import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    replay_size: int = 50000
    warmup_steps: int = 1000
    train_every: int = 4
    target_update_every: int = 500

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20000

    max_grad_norm: float = 10.0


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    frac = step / float(max(decay_steps, 1))
    return eps_start + frac * (eps_end - eps_start)


class DQNTrainer:
    """学习版 DQN Trainer（带 action mask）。

    你后续可以加：
    - Double DQN / Dueling
    - Prioritized Replay
    - n-step return
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes,
        cfg: DQNConfig,
        seed: int,
        device: Optional[str] = None,
        run_dir: str = "runs/dqn_run",
    ) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.cfg = cfg
        self.seed = int(seed)
        set_global_seed(self.seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q = QNetwork(obs_dim, num_actions, hidden_sizes).to(self.device)
        self.q_tgt = QNetwork(obs_dim, num_actions, hidden_sizes).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.opt = Adam(self.q.parameters(), lr=self.cfg.lr)
        self.replay = ReplayBuffer(capacity=self.cfg.replay_size, obs_dim=obs_dim, num_actions=num_actions, seed=self.seed)

        os.makedirs(run_dir, exist_ok=True)
        self.logger = JsonlLogger(os.path.join(run_dir, "train.jsonl"))
        self.run_dir = run_dir

        self.global_step = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        """epsilon-greedy + mask。"""
        assert obs.shape == (self.obs_dim,)
        assert mask.shape == (self.num_actions,)

        # 随机动作（从可行动作里采样）
        if np.random.rand() < epsilon:
            valid = np.where(mask > 0.5)[0]
            if len(valid) == 0:
                return 0
            return int(np.random.choice(valid))

        x = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
        q = self.q(x).squeeze(0).cpu().numpy()

        # mask invalid actions
        q = np.where(mask > 0.5, q, -1e9)
        return int(q.argmax())

    def add_transition(self, obs, action, reward, next_obs, done, mask, next_mask) -> None:
        self.replay.add(obs, action, reward, next_obs, done, mask, next_mask)

    def train_step(self) -> Dict[str, float]:
        batch = self.replay.sample(self.cfg.batch_size)

        obs = torch.tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.dones, dtype=torch.float32, device=self.device)

        masks = torch.tensor(batch.masks, dtype=torch.float32, device=self.device)
        next_masks = torch.tensor(batch.next_masks, dtype=torch.float32, device=self.device)

        # 当前 Q(s,a)
        q_all = self.q(obs)  # [B, A]
        q_sa = q_all.gather(1, actions).squeeze(-1)  # [B]

        with torch.no_grad():
            q_next_all = self.q_tgt(next_obs)  # [B, A]
            # mask invalid
            q_next_all = torch.where(next_masks > 0.5, q_next_all, torch.full_like(q_next_all, -1e9))
            q_next_max, _ = q_next_all.max(dim=1)
            target = rewards + self.cfg.gamma * (1.0 - dones) * q_next_max

        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        return {"loss": float(loss.item()), "q_mean": float(q_sa.mean().item())}

    def maybe_update_target(self) -> None:
        if self.global_step % self.cfg.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

    def save(self, name: str = "model.pt") -> str:
        path = os.path.join(self.run_dir, name)
        torch.save({"q": self.q.state_dict()}, path)
        return path
