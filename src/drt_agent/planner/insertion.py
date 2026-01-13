from __future__ import annotations
from typing import List, Tuple, Optional
import copy

from drt_agent.common.types import Order, Vehicle, Stop, StopType, OrderStatus


def calculate_insertion_cost(
        vehicle: Vehicle,
        order: Order,
        sim
) -> Tuple[float, List[Stop]]:
    if vehicle.load >= vehicle.capacity:
        pass

    current_schedule = vehicle.schedule
    best_cost = float('inf')
    best_schedule = []

    if not current_schedule:
        start_loc = vehicle.location
        start_time = max(sim.now, vehicle.next_free_time)

        t_to_pickup = sim.travel_time(start_loc, order.origin)
        arr_pickup = start_time + t_to_pickup

        if arr_pickup > order.pickup_deadline:
            return float('inf'), []

        t_to_drop = sim.travel_time(order.origin, order.destination)
        arr_drop = arr_pickup + t_to_drop

        cost = t_to_pickup + t_to_drop

        new_sched = [
            Stop(order.origin, StopType.PICKUP, order.order_id, arr_pickup),
            Stop(order.destination, StopType.DROPOFF, order.order_id, arr_drop)
        ]
        return cost, new_sched

    # Phase 3 修复：禁止插入到正在进行的第一个任务之前
    n = len(current_schedule)
    start_index = 1

    old_end_time = current_schedule[-1].estimated_arrival_time

    for i in range(start_index, n + 1):
        for j in range(i + 1, n + 2):
            temp_schedule = current_schedule[:]

            p_stop = Stop(order.origin, StopType.PICKUP, order.order_id)
            d_stop = Stop(order.destination, StopType.DROPOFF, order.order_id)

            temp_schedule.insert(j - 1, d_stop)
            temp_schedule.insert(i, p_stop)

            is_feasible, new_end_time = simulate_schedule(vehicle, temp_schedule, sim, order.pickup_deadline)

            if is_feasible:
                delta = new_end_time - old_end_time
                if delta < best_cost:
                    best_cost = delta
                    best_schedule = temp_schedule

    return best_cost, best_schedule


def simulate_schedule(vehicle: Vehicle, schedule: List[Stop], sim, pickup_deadline: int) -> Tuple[bool, int]:
    curr_t = sim.now
    curr_loc = vehicle.location
    curr_load = vehicle.load

    for stop in schedule:
        travel = sim.travel_time(curr_loc, stop.location)
        arr_time = curr_t + travel

        if stop.stop_type == StopType.PICKUP:
            if stop.estimated_arrival_time == 0:
                if arr_time > pickup_deadline:
                    return False, 0
            curr_load += 1
        elif stop.stop_type == StopType.DROPOFF:
            curr_load -= 1

        if curr_load > vehicle.capacity:
            return False, 0

        curr_t = arr_time
        curr_loc = stop.location

    return True, curr_t