from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from drt_agent.common.types import Event, EventType, Order, Vehicle, Stop, StopType, OrderStatus
# 引入新写的行为模型 (确保你创建了 behavior_model.py)
from drt_agent.sim.behavior_model import CancellationModel


@dataclass
class SimResult:
    now: int
    decision_order_id: Optional[int]
    completed_orders: List[int]


class Simulator:
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

        # 行为模型
        self.behavior_model = CancellationModel(self.rng)

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

    def assign_schedule_to_vehicle(self, vehicle_id: int, new_schedule: List[Stop]) -> None:
        v = self.vehicles[vehicle_id]
        is_idle = (len(v.schedule) == 0)

        v.schedule = new_schedule

        for stop in new_schedule:
            if stop.stop_type == StopType.PICKUP and stop.order_id in self.orders:
                order = self.orders[stop.order_id]
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.ASSIGNED

        if is_idle and len(new_schedule) > 0:
            first_stop = new_schedule[0]
            dt = self.travel_time(v.location, first_stop.location)
            arrival_time = self.now + dt
            first_stop.estimated_arrival_time = arrival_time

            self.schedule(Event(time=arrival_time, event_type=EventType.VEHICLE_ARRIVE, vehicle_id=vehicle_id))

    def process_vehicle_arrival(self, vehicle_id: int) -> None:
        v = self.vehicles[vehicle_id]
        if not v.schedule: return

        # 1. 到站
        current_stop = v.schedule.pop(0)
        v.location = current_stop.location
        oid = current_stop.order_id

        # 2. 业务处理
        if current_stop.stop_type == StopType.PICKUP:
            if oid in self.orders:
                order = self.orders[oid]
                # 上车前检查是否取消
                if order.status == OrderStatus.CANCELLED:
                    pass
                else:
                    order.status = OrderStatus.PICKED_UP
                    v.passengers.add(oid)

        elif current_stop.stop_type == StopType.DROPOFF:
            if oid in v.passengers:
                v.passengers.remove(oid)
                self.orders[oid].status = OrderStatus.COMPLETED
                self.completed_orders.append(oid)
                self.schedule(Event(time=self.now, event_type=EventType.ORDER_COMPLETE, order_id=oid))

        # 3. 驶向下一站
        if v.schedule:
            next_stop = v.schedule[0]
            dt = self.travel_time(v.location, next_stop.location)
            arrival_time = self.now + dt
            next_stop.estimated_arrival_time = arrival_time

            self.schedule(Event(time=arrival_time, event_type=EventType.VEHICLE_ARRIVE, vehicle_id=vehicle_id))

            # 触发取消检测
            if next_stop.stop_type == StopType.PICKUP:
                self.check_cancellation(next_stop.order_id, arrival_time)
        else:
            v.next_free_time = self.now

    def check_cancellation(self, order_id: int, expected_arrival_time: int):
        if order_id not in self.orders: return
        order = self.orders[order_id]
        if order.status in [OrderStatus.PICKED_UP, OrderStatus.COMPLETED, OrderStatus.CANCELLED]:
            return

        current_wait = self.now - order.t_request
        extra_wait = max(0, expected_arrival_time - self.now)

        if self.behavior_model.predict_cancellation(order, current_wait, extra_wait):
            order.status = OrderStatus.CANCELLED

    def defer_order(self, order_id: int, hold_delay: int) -> None:
        t2 = min(self.now + hold_delay, self.episode_horizon)
        self.schedule(Event(time=t2, event_type=EventType.ORDER_ARRIVAL, order_id=order_id))

    def run_until_next_arrival(self) -> SimResult:
        completed_batch: List[int] = []
        while True:
            ev = self.pop_event()
            if ev is None:
                return SimResult(self.now, None, self.completed_orders)

            self.now = ev.time

            if ev.event_type == EventType.VEHICLE_ARRIVE:
                self.process_vehicle_arrival(ev.vehicle_id)
                continue

            if ev.event_type == EventType.ORDER_COMPLETE:
                if ev.order_id is not None: completed_batch.append(ev.order_id)
                continue

            if ev.event_type == EventType.ORDER_ARRIVAL:
                return SimResult(self.now, ev.order_id, completed_batch)