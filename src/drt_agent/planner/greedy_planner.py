from __future__ import annotations

from typing import List, Tuple

import numpy as np

from drt_agent.common.types import Order, Vehicle
from drt_agent.planner.base import Planner


class GreedyPlanner(Planner):
    """学习版：极简候选生成器（toy）。

    - 可行性：车辆是否空闲
    - 特征：feasible, eta_pickup, delta_deadhead_km, cap_left
    """

    def __init__(self, vehicle_capacity: int):
        self.vehicle_capacity = vehicle_capacity

    def get_candidates(self, order, vehicles, now: int, K: int):
        feats = []
        vids = []
        mask = []

        # 1. 遍历所有车辆生成特征
        for v in vehicles:
            available_in = max(0, int(v.busy_until - now))

            # 简单的可行性检查
            feasible = 1.0 if (v.busy_until <= now and v.load < self.vehicle_capacity) else 0.0

            deadhead = self.sim.travel_time(v.node, order.origin) if hasattr(self, "sim") else 0
            trip = self.sim.travel_time(order.origin, order.destination) if hasattr(self, "sim") else 0

            # eta_complete
            eta_complete = int(now + available_in + deadhead + trip)

            vids.append(int(v.vehicle_id))
            feats.append([available_in, deadhead, trip, eta_complete])
            mask.append(feasible)

        # 转为 numpy
        if len(vids) > 0:
            vids = np.array(vids, dtype=np.int64)
            feats = np.array(feats, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)
        else:
            # 极端情况：没有车
            vids = np.array([], dtype=np.int64)
            feats = np.zeros((0, 4), dtype=np.float32)
            mask = np.array([], dtype=np.float32)

        # 2. 排序：按 eta_complete 排序 (不可行的排后面)
        # 构造排序键：feasible 的排前面(eta)，infeasible 的加个大数排后面
        sort_key = feats[:, 3] if len(feats) > 0 else []
        if len(mask) > 0:
            sort_key = sort_key + (1.0 - mask) * 1e9

        if len(vids) > 0:
            idx = np.argsort(sort_key)[:K]
            selected_vids = vids[idx]
            selected_feats = feats[idx]
            selected_mask = mask[idx]
        else:
            selected_vids = np.array([], dtype=np.int64)
            selected_feats = np.zeros((0, 4), dtype=np.float32)
            selected_mask = np.array([], dtype=np.float32)

        # 3. [关键修复] Padding 补齐到 K
        # 如果选出来的数量不足 K，必须补“空数据”，保证 obs/mask 维度固定
        current_count = len(selected_vids)
        if current_count < K:
            pad_size = K - current_count

            # vids 补 -1
            pad_vids = np.full(pad_size, -1, dtype=np.int64)
            selected_vids = np.concatenate([selected_vids, pad_vids])

            # feats 补 0 (维度要匹配: [pad_size, 4])
            pad_feats = np.zeros((pad_size, 4), dtype=np.float32)
            selected_feats = np.concatenate([selected_feats, pad_feats])

            # mask 补 0 (不可行)
            pad_mask = np.zeros(pad_size, dtype=np.float32)
            selected_mask = np.concatenate([selected_mask, pad_mask])

        # --- debug (可选) ---
        if getattr(self, "cfg", None) is not None and self.cfg.get("debug", False):
            if not hasattr(self, "_dbg_cnt"):
                self._dbg_cnt = 0
            if self._dbg_cnt < 3:
                print(f"[DBG][planner] Padded to K={K}, mask sum={selected_mask.sum()}")
                self._dbg_cnt += 1

        return selected_vids, selected_feats, selected_mask