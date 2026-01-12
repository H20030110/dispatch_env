from __future__ import annotations

from typing import List, Tuple
import numpy as np

from drt_agent.common.types import Order, Vehicle
from drt_agent.planner.base import Planner
# [新增] 导入插入算法
from drt_agent.planner.insertion import find_best_insertion


class GreedyPlanner(Planner):
    """
    学习版 GreedyPlanner (升级为 Insertion 模式)。
    策略：对每个车辆，尝试寻找最佳插入位置。
    特征：feasible (是否可插入), delta_cost (增加时间), trip_time (订单原本时间), eta_complete
    """

    def __init__(self, vehicle_capacity: int):
        self.vehicle_capacity = vehicle_capacity

    def get_candidates(self, order, vehicles, now: int, K: int):
        feats = []
        vids = []
        mask = []

        # 获取 travel_time 函数句柄
        travel_func = self.sim.travel_time if hasattr(self, "sim") else (lambda u, v: int(abs(u - v) + 5))

        for v in vehicles:
            # [核心] 调用插入算法
            delta_cost, best_schedule = find_best_insertion(v, order, travel_func, now=now)

            # 判断可行性
            is_feasible = (delta_cost != float("inf"))

            # 特征 1: 增加的 Detour 时间 (delta_cost)
            # 如果不可行，给一个很大的惩罚值，方便排序排到后面
            feat_delta = delta_cost if is_feasible else 1e5

            # 特征 2: 订单本身的 Trip 时间
            feat_trip = travel_func(order.origin, order.destination)

            # 特征 3: 预计完成时间 (ETA)
            # 这里的计算稍微估算一下：车辆做完所有任务的时间
            finish_time_base = self.sim._predict_schedule_finish_time(v) if hasattr(self, "sim") else now
            feat_eta = finish_time_base + feat_delta

            # 特征 4: 车辆空闲时间 (available_in) - 估算
            feat_avail = max(0, finish_time_base - now)

            vids.append(int(v.vehicle_id))
            feats.append([feat_avail, feat_delta, feat_trip, feat_eta])
            mask.append(1.0 if is_feasible else 0.0)

        # 转 numpy
        if len(vids) > 0:
            vids = np.array(vids, dtype=np.int64)
            feats = np.array(feats, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)
        else:
            vids = np.array([], dtype=np.int64)
            feats = np.zeros((0, 4), dtype=np.float32)
            mask = np.array([], dtype=np.float32)

        # 排序：优先选 Detour (feat_delta) 最小的
        # 注意：这里我们改用 feat_delta (第2列，索引1) 进行排序，因为“顺路”最重要
        sort_key = feats[:, 1] if len(feats) > 0 else []
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

        # Padding 补齐
        current_count = len(selected_vids)
        if current_count < K:
            pad_size = K - current_count
            pad_vids = np.full(pad_size, -1, dtype=np.int64)
            selected_vids = np.concatenate([selected_vids, pad_vids])
            pad_feats = np.zeros((pad_size, 4), dtype=np.float32)
            selected_feats = np.concatenate([selected_feats, pad_feats])
            pad_mask = np.zeros(pad_size, dtype=np.float32)
            selected_mask = np.concatenate([selected_mask, pad_mask])

        return selected_vids, selected_feats, selected_mask