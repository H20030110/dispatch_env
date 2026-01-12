from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# 务必引入新的类型定义
from drt_agent.common.types import Event, EventType, Order, Vehicle, Stop, StopType, OrderStatus


@dataclass
class SimResult:
    """仿真步进结果：告知 Env 当前时间、需要决策的订单 ID、以及这段时间内完成的订单。"""
    now: int
    decision_order_id: Optional[int]
    completed_orders: List[int]


class Simulator:
    """
    离散事件仿真器 (DARP版)。
    核心变化：支持车辆按照 Schedule 序列行驶，支持中途上下客。
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
        self.completed_orders: List[int] = []

    # -------------------------
    # 基础：事件堆操作
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
    # 初始化
    # -------------------------
    def load_orders(self, orders: List[Order]) -> None:
        self.orders = {o.order_id: o for o in orders}
        for o in orders:
            self.schedule(Event(time=o.t_request, event_type=EventType.ORDER_ARRIVAL, order_id=o.order_id))

    # -------------------------
    # 物理模型
    # -------------------------
    def travel_time(self, u: int, v: int) -> int:
        """计算两点间通行时间。后续可替换为距离矩阵。"""
        # 简单曼哈顿距离 + 固定损耗
        return int(abs(v - u) + 5)

    # -------------------------
    # 核心：车辆调度接口 (供 Env/Planner 调用)
    # -------------------------
    def assign_schedule_to_vehicle(self, vehicle_id: int, new_schedule: List[Stop]) -> None:
        """
        赋予车辆新的路径。
        如果车辆原本是静止的 (IDLE)，需要触发它的第一次移动。
        """
        v = self.vehicles[vehicle_id]
        is_idle = (len(v.schedule) == 0)

        # 1. 更新车辆的日程表
        v.schedule = new_schedule

        # 2. 更新涉及订单的状态为 ASSIGNED (仅限 Pickup 点)
        for stop in new_schedule:
            if stop.stop_type == StopType.PICKUP and stop.order_id in self.orders:
                order = self.orders[stop.order_id]
                # 只有还未被接起的订单才更新状态
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.ASSIGNED

        # 3. 物理启动：如果车之前没活干，现在有活了，要让它动起来
        if is_idle and len(new_schedule) > 0:
            first_stop = new_schedule[0]
            # 计算从当前位置去第一个点的时间
            dt = self.travel_time(v.location, first_stop.location)
            arrival_time = self.now + dt

            # 更新该站点的预计到达时间 (ETA)
            first_stop.estimated_arrival_time = arrival_time

            # 注册到达事件
            self.schedule(Event(
                time=arrival_time,
                event_type=EventType.VEHICLE_ARRIVE,
                vehicle_id=vehicle_id
            ))

    # -------------------------
    # 核心：事件处理逻辑
    # -------------------------
    def process_vehicle_arrival(self, vehicle_id: int) -> None:
        """
        处理 VEHICLE_ARRIVE 事件：
        1. 车辆到达当前站点 -> 上下客 -> 更新状态
        2. 安排去下一个站点
        """
        v = self.vehicles[vehicle_id]

        # 异常保护：如果没有 schedule 但触发了 arrive，直接返回
        if not v.schedule:
            return

        # --- 1. 到达当前站点 ---
        current_stop = v.schedule.pop(0)  # 取出并移除当前站
        v.location = current_stop.location  # 车辆位置更新

        oid = current_stop.order_id

        # --- 2. 处理业务 (上下客) ---
        if current_stop.stop_type == StopType.PICKUP:
            if oid in self.orders:
                order = self.orders[oid]
                # 只有订单没取消才能上车
                if order.status != OrderStatus.CANCELLED:
                    order.status = OrderStatus.PICKED_UP
                    v.passengers.add(oid)
                    # 可以在这里记录实际 Pickup 时间，计算等待时长

        elif current_stop.stop_type == StopType.DROPOFF:
            if oid in v.passengers:
                v.passengers.remove(oid)
                if oid in self.orders:
                    self.orders[oid].status = OrderStatus.COMPLETED
                    self.completed_orders.append(oid)
                    # 触发一个完成事件，方便 Env 统计
                    self.schedule(Event(time=self.now, event_type=EventType.ORDER_COMPLETE, order_id=oid))

        # --- 3. 驶向下一站 ---
        if v.schedule:
            next_stop = v.schedule[0]
            dt = self.travel_time(v.location, next_stop.location)
            arrival_time = self.now + dt

            # 更新下一站 ETA
            next_stop.estimated_arrival_time = arrival_time

            self.schedule(Event(
                time=arrival_time,
                event_type=EventType.VEHICLE_ARRIVE,
                vehicle_id=vehicle_id
            ))
        else:
            # 没任务了，车停在当前位置，标记空闲时间
            v.next_free_time = self.now

    def defer_order(self, order_id: int, hold_delay: int) -> None:
        """推迟订单决策"""
        t2 = min(self.now + hold_delay, self.episode_horizon)
        self.schedule(Event(time=t2, event_type=EventType.ORDER_ARRIVAL, order_id=order_id))

    # -------------------------
    # 主循环
    # -------------------------
    def run_until_next_arrival(self) -> SimResult:
        """
        推进时间，直到发生 ORDER_ARRIVAL（需要决策）或时间结束。
        期间会自动处理所有的 VEHICLE_ARRIVE 和 ORDER_COMPLETE。
        """
        current_batch_completed: List[int] = []

        while True:
            ev = self.pop_event()
            if ev is None:
                # 没事件了，仿真结束
                return SimResult(self.now, None, self.completed_orders)

            self.now = ev.time

            # 分发事件
            if ev.event_type == EventType.VEHICLE_ARRIVE:
                self.process_vehicle_arrival(ev.vehicle_id)
                continue

            if ev.event_type == EventType.ORDER_COMPLETE:
                # 仅用于统计
                if ev.order_id is not None:
                    current_batch_completed.append(ev.order_id)
                continue

            if ev.event_type == EventType.ORDER_ARRIVAL:
                # 遇到新订单（或 Hold 回来的订单），暂停仿真，交出控制权给 Agent
                return SimResult(self.now, ev.order_id, current_batch_completed)