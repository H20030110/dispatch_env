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

    import numpy as np

    def get_candidates(self, order, vehicles, now: int, K: int):
        feats = []
        vids = []
        mask = []

        # 订单固定 trip（与车辆无关）
        trip = self.sim.travel_time(order.origin, order.destination) if hasattr(self, "sim") else 0

        for v in vehicles:
            available_in = max(0, int(v.busy_until - now))

            feasible = 1.0 if (v.busy_until <= now and v.load < self.vehicle_capacity) else 0.0

            deadhead = self.sim.travel_time(v.node, order.origin) if hasattr(self, "sim") else 0
            trip = self.sim.travel_time(order.origin, order.destination) if hasattr(self, "sim") else 0

            # 先算 eta_complete（不做 infeasible 的 1e9 加罚）
            eta_complete = int(now + available_in + deadhead + trip)

            vids.append(int(v.vehicle_id))
            feats.append([available_in, deadhead, trip, eta_complete])
            mask.append(feasible)

        vids = np.array(vids, dtype=np.int64)
        feats = np.array(feats, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        # --- debug: 只打印前 3 次 ---
        if getattr(self, "cfg", None) is not None and self.cfg.get("debug", False):
            if not hasattr(self, "_dbg_cnt"):
                self._dbg_cnt = 0
            if self._dbg_cnt < 3:
                print("[DBG][planner] now=", now, "order=", order.order_id)
                print("[DBG][planner] feasible sum=", int(np.sum(mask)))
                print("[DBG][planner] mask=", np.array(mask, dtype=np.float32))
                self._dbg_cnt += 1

        # 按 eta_complete 排序取前 K
        idx = np.argsort(feats[:, 3] + (1.0 - mask) * 1e9)[:K]
        return vids[idx], feats[idx], mask[idx]



