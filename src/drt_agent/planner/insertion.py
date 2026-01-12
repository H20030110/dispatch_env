from __future__ import annotations

import copy
from typing import List, Tuple, Callable

from drt_agent.common.types import Stop, Vehicle, Order


def calculate_schedule_cost(
        schedule: List[Stop],
        start_node: int,
        initial_load: int,
        capacity: int,
        travel_time_func: Callable[[int, int], int],
        start_time: int  # [新增] 需要知道当前时间才能算是否超时
) -> Tuple[bool, int]:
    """
    计算序列代价，并检查容量 + 时间窗约束。
    """
    t = start_time
    curr = start_node
    load = initial_load

    total_travel = 0

    for stop in schedule:
        # 1. 累加行驶时间
        travel = travel_time_func(curr, stop.node)
        t += travel
        total_travel += travel
        curr = stop.node

        # [更新] 记录预计到达时间到 Stop 对象（方便后续看，但这只在副本上生效）
        stop.arrival_time = t

        # 2. [核心] 检查时间窗 (Time Window Constraint)
        # 如果当前时间 t 超过了该站点的最晚允许时间，则不可行
        if t > stop.latest_time:
            return False, float("inf")

        # 3. 更新并检查载客量
        if stop.action == 0:  # Pickup
            load += 1
        else:  # Dropoff
            load -= 1

        if load > capacity:
            return False, float("inf")

    return True, total_travel


def find_best_insertion(
        vehicle: Vehicle,
        order: Order,
        travel_time_func: Callable[[int, int], int],
        now: int = 0  # [新增] 传入当前时间
) -> Tuple[float, List[Stop]]:
    """
    寻找最佳插入位置，同时满足容量和时间窗约束。
    """
    # 构造 Pickup 和 Dropoff 节点
    # Pickup: latest_time 暂时不限（或者限制为 deadline - trip_time）
    # Dropoff: latest_time = order.deadline
    stop_p = Stop(node=order.origin, action=0, order_id=order.order_id, latest_time=10 ** 9)
    stop_d = Stop(node=order.destination, action=1, order_id=order.order_id, latest_time=order.deadline)

    current_schedule = vehicle.schedule
    n = len(current_schedule)

    # 估算当前车辆状态
    # 如果车在路上，它到达下一站的时间已经被扣减了 t_to_next_stop
    # 我们做个简化：假设“基准时间”是 now + t_to_next_stop (如果 t_to_next_stop>0)
    # 并且起始位置是 schedule[0] 或 vehicle.node

    # 更严谨的做法：
    start_t = now
    if vehicle.t_to_next_stop > 0:
        start_t += vehicle.t_to_next_stop

    # 计算插入前的基准耗时（用于算增量）
    # 注意：这里要传入 start_t
    valid_base, base_cost = calculate_schedule_cost(
        current_schedule, vehicle.node, vehicle.load, vehicle.capacity, travel_time_func, start_t
    )
    if not valid_base:
        base_cost = 0

    best_cost = float("inf")
    best_schedule = None

    # 确定插入起始点
    start_idx = 1 if vehicle.t_to_next_stop > 0 else 0

    for i in range(start_idx, n + 1):
        for j in range(i + 1, n + 2):
            tmp_sched = copy.deepcopy(current_schedule)  # deepcopy 因为 stop 对象被修改了
            tmp_sched.insert(i, stop_p)
            tmp_sched.insert(j, stop_d)

            feasible, total_time = calculate_schedule_cost(
                tmp_sched, vehicle.node, vehicle.load, vehicle.capacity, travel_time_func, start_t
            )

            if feasible:
                if total_time < best_cost:
                    best_cost = total_time
                    best_schedule = tmp_sched

    if best_schedule is None:
        return float("inf"), []

    delta = best_cost - base_cost
    return delta, best_schedule