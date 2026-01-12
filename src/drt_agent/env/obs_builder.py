from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from drt_agent.common.types import Order, Vehicle


@dataclass
class ObsSpec:
    obs_dim: int
    num_actions: int
    num_candidates: int
    num_modes: int = 3


def flatten_obs(
        order: Order,
        now: int,
        episode_horizon: int,
        num_nodes: int,
        vehicles: List[Vehicle],
        candidate_vehicle_ids: List[int],
        candidate_feats: np.ndarray,
        candidate_mask: np.ndarray,
) -> np.ndarray:
    """把结构化信息拼成固定维度向量。"""

    # ---------- order features (4 dim) ----------
    t_norm = float(now) / float(max(episode_horizon, 1))
    o_norm = float(order.origin) / float(max(num_nodes - 1, 1))
    d_norm = float(order.destination) / float(max(num_nodes - 1, 1))
    dist_norm = float(abs(order.destination - order.origin)) / float(max(num_nodes - 1, 1))

    order_feat = np.array([t_norm, o_norm, d_norm, dist_norm], dtype=np.float32)

    # ---------- global features (2 dim) ----------
    # [Fix] 这里的判断逻辑变了：schedule 为空才算 idle
    idle = sum(1 for v in vehicles if len(v.schedule) == 0)
    idle_ratio = float(idle) / float(max(len(vehicles), 1))

    # 平均负载率 (之前加的)
    total_load = sum(v.load for v in vehicles)
    total_cap = sum(v.capacity for v in vehicles)
    mean_load_ratio = float(total_load) / float(max(total_cap, 1))

    global_feat = np.array([idle_ratio, mean_load_ratio], dtype=np.float32)

    # ---------- candidates ----------
    cand_flat = candidate_feats.reshape(-1).astype(np.float32)
    cand_mask = candidate_mask.astype(np.float32)

    obs = np.concatenate([order_feat, global_feat, cand_flat, cand_mask], axis=0)
    return obs


def build_action_mask(
        candidate_mask: np.ndarray,
        num_modes: int,
) -> np.ndarray:
    """构造动作 mask。"""
    K = int(candidate_mask.shape[0])
    A = 2 + K * num_modes
    mask = np.ones((A,), dtype=np.float32)

    for k in range(K):
        for m in range(num_modes):
            a = 2 + k * num_modes + m
            mask[a] = float(candidate_mask[k] > 0.5)
    return mask