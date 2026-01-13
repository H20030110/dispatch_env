from __future__ import annotations
import numpy as np
from drt_agent.common.types import Order


class CancellationModel:
    """
    乘客取消行为模型 (对应开题报告 3.2 节)
    基于 Logit 或 规则 判断订单是否取消。
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        # 敏感度参数 (可以后续在 config 里配)
        self.alpha_wait = 0.05  # 等待时间敏感度
        self.alpha_delay = 0.1  # 延误敏感度

    def predict_cancellation(self, order: Order, current_wait: int, expected_extra_wait: int) -> bool:
        """
        判断是否取消。
        :param current_wait: 已经等待了多久
        :param expected_extra_wait: 预计还要等多久 (ETA - now)
        """
        # 1. 硬约束：超过最大容忍时间直接取消
        total_wait = current_wait + expected_extra_wait
        if total_wait > order.max_wait_time:
            return True

        # 2. 软约束：概率性取消 (Logit 形式的简化版)
        # 效用函数 U = - alpha * wait
        # 转化成概率 P(cancel) = 1 / (1 + e^(-U - bias))
        # 这里用一个更简单的线性概率模型演示：

        prob = 0.0

        # 如果等待超过 10 分钟 (600s)，每多等 1 分钟增加 5% 取消率
        if total_wait > 3600:
            prob += (total_wait - 3600) / 60.0 * 0.05

        # 封顶 80%
        prob = min(prob, 0.8)

        return self.rng.random() < prob