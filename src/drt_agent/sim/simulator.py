from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from drt_agent.common.types import Event, EventType, Order, Vehicle, Stop
# [新增] 导入插入算法
from drt_agent.planner.insertion import find_best_insertion


@dataclass
class SimResult:
    now: int
    decision_order_id: Optional[int]
    completed_orders: List[int]


class Simulator:
    """支持任务队列 + 插入逻辑的仿真器。"""

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
        self.completed_orders: List[int] = []

    def schedule(self, event: Event) -> None:
        self._seq += 1
        event.seq = self._seq
        heapq.heappush(self._event_heap, event)

    def pop_event(self) -> Optional[Event]:
        if not self._event_heap:
            return None
        return heapq.heappop(self._event_heap)

    def load_orders(self, orders: List[Order]) -> None:
        self.orders = {o.order_id: o for o in orders}
        for o in orders:
            self.schedule(Event(time=o.t_request, event_type=EventType.ORDER_ARRIVAL, order_id=o.order_id))

    def travel_time(self, u: int, v: int) -> int:
        return int(abs(v - u) + 5)

    def service_time_for_order(self, order: Order) -> int:
        return self.travel_time(order.origin, order.destination)

    # -------------------------------------------------
    # [核心升级] 使用 Insertion 逻辑分配订单
    # -------------------------------------------------


        # 在 Simulator 类里增加参数或读取配置（这里简化，直接硬编码或传参）
        # ...

    def assign_order_to_vehicle(self, order_id: int, vehicle_id: int) -> Tuple[bool, int]:
            order = self.orders[order_id]
            v = self.vehicles[vehicle_id]

            # 1. 尝试插入
            delta, new_schedule = find_best_insertion(v, order, self.travel_time, now=self.now)

            if new_schedule is None or delta == float("inf"):
                return False, 0

            # ---------------- Phase 5 新增逻辑 ----------------
            # 2. 检查乘客耐心 (Patience Check)
            # 找到新 schedule 里这个订单的 Pickup 节点，看它预计几点到
            pickup_time = 0
            for stop in new_schedule:
                if stop.order_id == order_id and stop.action == 0:  # Pickup
                    pickup_time = stop.arrival_time  # 这个字段在 insertion.py 里被计算并赋值了
                    break

            wait_time = pickup_time - self.now
            # 假设乘客耐心只有 15 分钟 (900秒)，超过就“当场取消/拒绝被派单”
            # 你也可以把这个 900 放到 config 里
            if wait_time > 900:
                return False, 0  # 模拟乘客因为等太久而拒绝
            # --------------------------------------------------

            # 3. 真正执行
            v.schedule = new_schedule
            finish_time = self._predict_schedule_finish_time_with_schedule(v, new_schedule)

            return True, finish_time

    def defer_order(self, order_id: int, hold_delay: int) -> None:
        t2 = min(self.now + hold_delay, self.episode_horizon)
        self.schedule(Event(time=t2, event_type=EventType.ORDER_ARRIVAL, order_id=order_id))

    def run_until_next_arrival(self) -> SimResult:
        new_completed: List[int] = []
        while True:
            ev = self.pop_event()
            if ev is None:
                return SimResult(now=self.now, decision_order_id=None, completed_orders=new_completed)

            delta = max(0, ev.time - self.now)
            if delta > 0:
                self._forward_vehicles(delta, new_completed)

            self.now = ev.time

            if ev.event_type == EventType.ORDER_ARRIVAL:
                return SimResult(now=self.now, decision_order_id=ev.order_id, completed_orders=new_completed)

    def _forward_vehicles(self, delta: int, completed_list: List[int]) -> None:
        for v in self.vehicles.values():
            time_budget = delta
            while time_budget > 0 and v.schedule:
                next_stop = v.schedule[0]
                dist_time = self.travel_time(v.node, next_stop.node)

                if v.t_to_next_stop <= 0:
                    v.t_to_next_stop = dist_time

                needed = v.t_to_next_stop
                if time_budget >= needed:
                    time_budget -= needed
                    v.t_to_next_stop = 0
                    v.node = next_stop.node
                    v.schedule.pop(0)

                    if next_stop.action == 0:  # Pickup
                        v.load += 1
                    else:  # Dropoff
                        v.load -= 1
                        completed_list.append(next_stop.order_id)
                        self.completed_orders.append(next_stop.order_id)
                else:
                    v.t_to_next_stop -= time_budget
                    time_budget = 0

    def _predict_schedule_finish_time(self, v: Vehicle) -> int:
        return self._predict_schedule_finish_time_with_schedule(v, v.schedule)

    def _predict_schedule_finish_time_with_schedule(self, v: Vehicle, schedule: List[Stop]) -> int:
        t = self.now
        curr_node = v.node
        if v.t_to_next_stop > 0:
            t += v.t_to_next_stop
            if schedule:
                curr_node = schedule[0].node
                stops_to_visit = schedule[1:]
            else:
                stops_to_visit = []
        else:
            stops_to_visit = schedule

        for s in stops_to_visit:
            t += self.travel_time(curr_node, s.node)
            curr_node = s.node
        return t