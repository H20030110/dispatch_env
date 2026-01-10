from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from drt_agent.common.types import Event, EventType, Order, Vehicle


@dataclass
class SimResult:
    """用于把仿真推进到“下一个决策点”时的结果打包返回给 Env。"""
    now: int
    decision_order_id: Optional[int]
    completed_orders: List[int]


class Simulator:
    """学习版最小离散事件仿真器（DES）。

    - 事件：ORDER_ARRIVAL, ORDER_COMPLETE
    - 车辆：toy 模型（一次只服务一单）
    - 路径：toy 行驶时间=|dest-origin| + 常数（可替换为矩阵/路网）
    """

    def __init__(
        self,
        num_nodes: int,
        episode_horizon: int,
        vehicles: List[Vehicle],
        rng: np.random.Generator,
    ) -> None:
        self.num_nodes = num_nodes
        self.episode_horizon = episode_horizon
        self.vehicles: Dict[int, Vehicle] = {v.vehicle_id: v for v in vehicles}
        self.rng = rng

        self.now: int = 0
        self._event_heap: List[Event] = []
        self._seq = 0

        self.orders: Dict[int, Order] = {}
        self.pending_orders: List[int] = []   # toy: HOLD 订单池
        self.completed_orders: List[int] = []

    # -------------------------
    # 基础：事件队列
    # -------------------------
    def schedule(self, event: Event) -> None:
        self._seq += 1
        event.seq = self._seq
        heapq.heappush(self._event_heap, event)

    def pop_event(self) -> Optional[Event]:
        if not self._event_heap:
            return None
        return heapq.heappop(self._event_heap)

    # -------------------------
    # 初始化订单流（toy）
    # -------------------------
    def load_orders(self, orders: List[Order]) -> None:
        self.orders = {o.order_id: o for o in orders}
        for o in orders:
            self.schedule(Event(time=o.t_request, event_type=EventType.ORDER_ARRIVAL, order_id=o.order_id))

    # -------------------------
    # toy 行驶时间/服务时间
    # -------------------------
    def travel_time(self, u: int, v: int) -> int:
        # TODO(你填空): 换成 OD 时间矩阵 / time-dependent 矩阵 / 路网最短路
        return int(abs(v - u) + 5)

    def service_time_for_order(self, order: Order) -> int:
        # TODO(你填空): 上下客时间、绕行、拼车影响等
        return self.travel_time(order.origin, order.destination)

    # -------------------------
    # 执行动作效果（由 Env 调用）
    # -------------------------
    def assign_order_to_vehicle(self, order_id: int, vehicle_id: int) -> Tuple[bool, int]:
        """把订单分配给车辆（toy：车辆必须空闲）。

        Returns:
            success: 是否成功
            complete_time: 若成功，订单完成时间
        """
        order = self.orders[order_id]
        v = self.vehicles[vehicle_id]

        # toy 可行性：车辆必须空闲
        if v.busy_until > self.now:
            return False, self.now

        # 计算空驶到上车点时间 + 订单服务时间
        deadhead = self.travel_time(v.node, order.origin)  # 车当前位置 -> 上车点
        trip = self.travel_time(order.origin, order.destination)  # 上车点 -> 目的地
        dur = deadhead + trip
        complete_time = self.now + dur

        v.busy_until = complete_time
        v.node = order.destination
        v.load = 0

        # 安排完成事件
        self.schedule(Event(time=complete_time, event_type=EventType.ORDER_COMPLETE, order_id=order_id, vehicle_id=vehicle_id))
        return True, complete_time

    def defer_order(self, order_id: int, hold_delay: int) -> None:
        """HOLD：把订单推迟一段时间再回来决策（toy）。"""
        t2 = min(self.now + hold_delay, self.episode_horizon)
        self.schedule(Event(time=t2, event_type=EventType.ORDER_ARRIVAL, order_id=order_id))

    # -------------------------
    # 推进到下一个“决策点”
    # -------------------------
    def run_until_next_arrival(self) -> SimResult:
        """推进事件，直到遇到下一个 ORDER_ARRIVAL（需要决策）或 episode 结束。"""
        completed: List[int] = []

        while True:
            ev = self.pop_event()
            if ev is None:
                # 没事件了：episode 结束
                return SimResult(now=self.now, decision_order_id=None, completed_orders=completed)

            self.now = ev.time

            # 推进车辆状态（toy）
            for v in self.vehicles.values():
                v.update_time(self.now)

            if ev.event_type == EventType.ORDER_COMPLETE:
                if ev.order_id is not None:
                    completed.append(ev.order_id)
                    self.completed_orders.append(ev.order_id)
                continue

            if ev.event_type == EventType.ORDER_ARRIVAL:
                # 到了一个需要决策的订单
                return SimResult(now=self.now, decision_order_id=ev.order_id, completed_orders=completed)
