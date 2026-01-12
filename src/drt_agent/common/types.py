from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List


class EventType(Enum):
    ORDER_ARRIVAL = auto()
    ORDER_COMPLETE = auto()
    ORDER_CANCEL = auto()  # [新增] 订单取消
    DECISION_POINT = auto()


class Decision(Enum):
    REJECT = 0
    HOLD = 1
    ACCEPT = 2


class PlanMode(Enum):
    FAST_INSERT = 0
    LOCAL_REPLAN = 1
    GLOBAL_REPLAN = 2


@dataclass(frozen=True)
class Order:
    order_id: int
    t_request: int
    origin: int
    destination: int
    pax: int = 1

    # [新增] 最晚送达时间 (绝对时间秒)
    deadline: int = 10 ** 9

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stop:
    """表示任务队列中的一个站点。"""
    node: int
    action: int  # 0: Pickup, 1: Dropoff
    order_id: int

    # [新增] 预计到达时间（用于校验时间窗）
    # 注意：这个值是动态计算的，存这里主要为了方便调试查看
    arrival_time: int = 0
    # [新增] 该站点的最晚允许时间（Pickup可以是无穷大，Dropoff则是订单deadline）
    latest_time: int = 10 ** 9


@dataclass
class Vehicle:
    vehicle_id: int
    capacity: int = 4
    node: int = 0
    load: int = 0

    schedule: List[Stop] = field(default_factory=list)
    t_to_next_stop: int = 0

    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_idle(self) -> bool:
        return len(self.schedule) == 0

    def update_time(self, now: int) -> None:
        pass


@dataclass(order=True)
class Event:
    time: int
    seq: int = field(default=0)
    event_type: "EventType" = field(default=None, compare=False)
    order_id: Optional[int] = field(default=None, compare=False)
    vehicle_id: Optional[int] = field(default=None, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)