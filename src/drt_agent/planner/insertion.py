from __future__ import annotations
from typing import List, Tuple, Optional
import copy

from drt_agent.common.types import Order, Vehicle, Stop, StopType, OrderStatus


def calculate_insertion_cost(
        vehicle: Vehicle,
        order: Order,
        sim  # 传入 Simulator 实例用于计算时间
) -> Tuple[float, List[Stop]]:
    """
    尝试将 order 插入到 vehicle 的 schedule 中。
    返回: (cost, new_schedule)
    如果不可行（违反时间窗或容量），返回 (float('inf'), [])
    """

    # 1. 基础检查
    # 如果插入这单会导致超载（简单检查：假设当前负载+1 <= 容量）
    # 更严谨的检查需要在模拟路径时一步步推演 load
    if vehicle.load >= vehicle.capacity:
        # 注意：这只是当前瞬间的负载。严谨做法是检查路径上每个点的负载。
        # 这里为了简化，先做此检查。
        pass

    current_schedule = vehicle.schedule
    best_cost = float('inf')
    best_schedule = []

    # 车辆当前位置和时间
    if not current_schedule:
        # 空车：直接构造 [Pickup, Dropoff]
        start_loc = vehicle.location
        start_time = max(sim.now, vehicle.next_free_time)

        # 计算时间
        t_to_pickup = sim.travel_time(start_loc, order.origin)
        arr_pickup = start_time + t_to_pickup

        # 检查 Pickup 时间窗 (Deadline)
        if arr_pickup > order.pickup_deadline:
            return float('inf'), []

        t_to_drop = sim.travel_time(order.origin, order.destination)
        arr_drop = arr_pickup + t_to_drop

        # 成本 = 完成时间 - 当前时间 (或者仅仅是行驶时间)
        # 这里成本定义为：增加的总耗时
        cost = t_to_pickup + t_to_drop

        new_sched = [
            Stop(order.origin, StopType.PICKUP, order.order_id, arr_pickup),
            Stop(order.destination, StopType.DROPOFF, order.order_id, arr_drop)
        ]
        return cost, new_sched

    # 有任务的车：遍历所有可能的插入位置 (i, j)
    # 0 <= i <= len
    # i < j <= len + 1
    # 原序列: S1 -> S2 -> S3
    # 插入 P, D:
    # i=0: P -> S1 -> S2 -> S3 (然后 D 插在后面任意位置)

    n = len(current_schedule)

    # 为了性能，这里我们简化逻辑：
    # 重新构建一条完整的临时路径，并计算是否可行
    # 提示：Python 循环可能会慢，后续可用 Cython 或 Numba 优化

    # 预计算当前路径的结束时间，用于对比 Delta Cost
    old_end_time = current_schedule[-1].estimated_arrival_time

    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            # 构造新序列
            temp_schedule = current_schedule[:]  # Copy

            p_stop = Stop(order.origin, StopType.PICKUP, order.order_id)
            d_stop = Stop(order.destination, StopType.DROPOFF, order.order_id)

            # 插入 Dropoff (注意索引变化)
            temp_schedule.insert(j - 1, d_stop)
            # 插入 Pickup
            temp_schedule.insert(i, p_stop)

            # --- 验证可行性并计算成本 ---
            is_feasible, new_end_time = simulate_schedule(vehicle, temp_schedule, sim, order.pickup_deadline)

            if is_feasible:
                # Cost = 增加的完工时间 (延误)
                # 也可以加入 detour distance
                delta = new_end_time - old_end_time
                if delta < best_cost:
                    best_cost = delta
                    best_schedule = temp_schedule

    return best_cost, best_schedule


def simulate_schedule(vehicle: Vehicle, schedule: List[Stop], sim, pickup_deadline: int) -> Tuple[bool, int]:
    """
    模拟跑一遍 schedule，检查时间窗和容量。
    """
    curr_t = sim.now
    curr_loc = vehicle.location
    curr_load = vehicle.load  # 初始负载

    # 如果车辆正在前往 schedule[0] (Sim中已经有 ARRIVE 事件)，
    # 实际上我们修改 schedule 时，通常保留第一个正在进行的站点不动，
    # 或者如果改动了第一个站，需要 simulator 能处理。
    # 这里假设我们总是从“当前位置”开始推演。

    for stop in schedule:
        travel = sim.travel_time(curr_loc, stop.location)
        arr_time = curr_t + travel

        # 1. 检查 Pickup Deadline
        if stop.stop_type == StopType.PICKUP:
            # 如果是新订单的 Pickup
            if stop.estimated_arrival_time == 0:  # 标记是新点
                if arr_time > pickup_deadline:
                    return False, 0
            curr_load += 1
        elif stop.stop_type == StopType.DROPOFF:
            curr_load -= 1

        # 2. 检查容量
        if curr_load > vehicle.capacity:
            return False, 0

        stop.estimated_arrival_time = arr_time
        curr_t = arr_time
        curr_loc = stop.location

    return True, curr_t