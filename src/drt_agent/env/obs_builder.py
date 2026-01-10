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
    num_modes: int = 3  # FAST/LOCAL/GLOBAL


def flatten_obs(
    order: Order,
    now: int,
    episode_horizon: int,
    num_nodes: int,
    vehicles: List[Vehicle],
    candidate_vehicle_ids: List[int],
    candidate_feats: np.ndarray,  # [K, d]
    candidate_mask: np.ndarray,   # [K]
) -> np.ndarray:
    """把结构化信息拼成固定维度向量（学习版示例）。

    你后续可以把它替换为更合理的编码方式（embedding、attention、统计特征等）。
    """
    # ---------- order features ----------
    # 归一化到 0~1
    t_norm = float(now) / float(max(episode_horizon, 1))
    o_norm = float(order.origin) / float(max(num_nodes - 1, 1))
    d_norm = float(order.destination) / float(max(num_nodes - 1, 1))
    dist_norm = float(abs(order.destination - order.origin)) / float(max(num_nodes - 1, 1))

    order_feat = np.array([t_norm, o_norm, d_norm, dist_norm], dtype=np.float32)

    # ---------- global features ----------
    idle = sum(1 for v in vehicles if v.busy_until <= now)
    idle_ratio = float(idle) / float(max(len(vehicles), 1))
    global_feat = np.array([idle_ratio], dtype=np.float32)

    # ---------- candidates ----------
    # 直接 flatten
    cand_flat = candidate_feats.reshape(-1).astype(np.float32)
    cand_mask = candidate_mask.astype(np.float32)

    # 把 mask 也放进 obs（可选）
    obs = np.concatenate([order_feat, global_feat, cand_flat, cand_mask], axis=0)
    return obs


def build_action_mask(
    candidate_mask: np.ndarray,
    num_modes: int,
) -> np.ndarray:
    """构造扁平离散动作的 mask。

    动作编码：
    0: REJECT
    1: HOLD
    2..: ACCEPT with (candidate_k, mode_m)
        action_id = 2 + k*num_modes + m
    """
    K = int(candidate_mask.shape[0])
    A = 2 + K * num_modes
    mask = np.ones((A,), dtype=np.float32)
    # accept actions 取决于 candidate 是否可行
    for k in range(K):
        for m in range(num_modes):
            a = 2 + k * num_modes + m
            mask[a] = float(candidate_mask[k] > 0.5)
    return mask
