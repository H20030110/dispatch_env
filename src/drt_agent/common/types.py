from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any


class EventType(Enum):
    """离散事件类型。"""
    ORDER_ARRIVAL = auto()
    ORDER_COMPLETE = auto()
    DECISION_POINT = auto()


class Decision(Enum):
    """决策动作类型。"""
    REJECT = 0
    HOLD = 1
    ACCEPT = 2


class PlanMode(Enum):
    """ACCEPT 时的具体执行模式。"""
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
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Vehicle:
    vehicle_id: int
    capacity: int = 8
    node: int = 0                # 当前所在节点
    busy_until: int = 0          # 忙到什么时候
    load: int = 0                # 载客数
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_idle(self) -> bool:
        return self.busy_until <= 0

    def update_time(self, now: int) -> None:
        """更新车辆状态到 now。"""
        if self.busy_until <= now:
            self.busy_until = now
            self.load = 0


@dataclass(order=True)
class Event:
    time: int
    seq: int = field(default=0)
    event_type: "EventType" = field(default=None, compare=False)
    order_id: Optional[int] = field(default=None, compare=False)
    vehicle_id: Optional[int] = field(default=None, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)