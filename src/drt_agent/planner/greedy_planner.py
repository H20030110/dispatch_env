from __future__ import annotations
from typing import List, Tuple
import numpy as np

from drt_agent.planner.base import BasePlanner
from drt_agent.common.types import Order, Vehicle
# 引入新写的插入逻辑
from drt_agent.planner.insertion import calculate_insertion_cost


class GreedyPlanner(BasePlanner):
    def __init__(self, vehicle_capacity: int = 8):
        self.vehicle_capacity = vehicle_capacity
        self.sim = None

    def get_candidates(self, order: Order, vehicles: List[Vehicle], now: int, k: int = 5) -> Tuple[
        List[int], np.ndarray, np.ndarray]:

        candidates = []

        for v in vehicles:
            if not self.sim:
                # 如果没有 sim (测试用)，这就无法计算
                candidates.append((v.vehicle_id, float('inf'), []))
                continue

            # 调用插入算法
            cost, new_sched = calculate_insertion_cost(v, order, self.sim)

            # 如果 cost 是 inf，说明不可行
            candidates.append((v.vehicle_id, cost, new_sched))

        # 按 cost 排序 (越小越好)
        candidates.sort(key=lambda x: x[1])
        top_k = candidates[:k]

        cand_ids = []
        cand_feats = []
        cand_mask = []

        for vid, cost, new_sched in top_k:
            is_feasible = (cost < float('inf'))

            cand_ids.append(vid)
            cand_mask.append(1.0 if is_feasible else 0.0)

            # 特征: [Cost, Schedule长度, 剩余容量, 0]
            # 这里你可以根据需要丰富特征
            v = next(v for v in vehicles if v.vehicle_id == vid)
            sched_len = len(v.schedule)
            rem_cap = v.capacity - v.load

            cand_feats.append([cost if is_feasible else 9999, sched_len, rem_cap, 0])

            # 暂时把 new_sched 存哪里？
            # 我们的接口只返回 ID。Env 需要重新计算一遍插入，或者我们把 new_sched 缓存起来。
            # 为了简化，Phase 2 让 Env 再算一遍（或者 Env 只是简单 Append，等待 Phase 3 完善）。
            # ⚠️ 注意：目前 Env.step 里还是写的 "Append"。
            # 如果你想现在就生效，需要修改 Env.step。

        # Padding
        while len(cand_ids) < k:
            cand_ids.append(-1)
            cand_feats.append([0, 0, 0, 0])
            cand_mask.append(0.0)

        return cand_ids, np.array(cand_feats, dtype=np.float32), np.array(cand_mask, dtype=np.float32)