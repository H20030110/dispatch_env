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
        candidate_mask: np.ndarray,  # [K]
) -> np.ndarray:
    """把结构化信息拼成固定维度向量。"""

    # ---------- order features ----------
    # 归一化到 0~1
    t_norm = float(now) / float(max(episode_horizon, 1))
    o_norm = float(order.origin) / float(max(num_nodes - 1, 1))
    d_norm = float(order.destination) / float(max(num_nodes - 1, 1))
    dist_norm = float(abs(order.destination - order.origin)) / float(max(num_nodes - 1, 1))

    order_feat = np.array([t_norm, o_norm, d_norm, dist_norm], dtype=np.float32)

    # ---------- global features ----------
    # [关键修复]：使用 schedule 长度判断空闲，替代 v.busy_until
    idle_count = sum(1 for v in vehicles if len(v.schedule) == 0)

    idle_ratio = float(idle_count) / float(max(len(vehicles), 1))
    global_feat = np.array([idle_ratio], dtype=np.float32)

    # ---------- candidates ----------
    # Flatten 候选车辆特征
    # 注意：env 里传入的 candidate_feats 已经是 numpy 数组了
    cand_flat = []

    # 如果维度不对，手动铺平；或者直接 reshape
    # 这里为了稳健，假设 candidate_feats 是 (K, 4)
    if candidate_feats.ndim == 2:
        cand_flat = candidate_feats.reshape(-1)
    else:
        # 异常兜底
        cand_flat = np.zeros(len(candidate_vehicle_ids) * 4, dtype=np.float32)

    cand_flat = cand_flat.astype(np.float32)
    cand_mask = candidate_mask.astype(np.float32)

    # 拼接所有特征
    # shapes: [4] + [1] + [K*4] + [K]
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
        # 如果第 k 个候选车辆不可行，则其对应的所有 mode 动作都 mask 掉
        is_feasible = float(candidate_mask[k] > 0.5)
        if is_feasible < 0.5:
            start_idx = 2 + k * num_modes
            end_idx = start_idx + num_modes
            mask[start_idx:end_idx] = 0.0

    return mask