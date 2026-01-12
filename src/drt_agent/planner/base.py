from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import numpy as np

from drt_agent.common.types import Order, Vehicle


class BasePlanner(ABC):  # <--- 修改点：这里原来是 Planner，现在改为 BasePlanner
    """规划器接口：Env 用它来生成候选车辆与执行分配。

    学习阶段先用 GreedyPlanner 占位；后续你可以替换为：
    - 插入启发式（有时间窗/容量约束）
    - ALNS
    """

    @abstractmethod
    def get_candidates(
        self,
        order: Order,
        vehicles: List[Vehicle],
        now: int,
        k: int,
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """返回 Top-K 候选车辆，并给出 candidates_feat 与 mask。

        Returns:
            vehicle_ids: [k] 候选车辆 id（不足 k 则用 -1 填充）
            candidates_feat: shape [k, d]
            feasible_mask: shape [k] 0/1
        """
        raise NotImplementedError