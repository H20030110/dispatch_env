from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Set

# ==========================================
# 1. 状态枚举
# ==========================================
class EventType(Enum):
    ORDER_ARRIVAL = auto()      # 新订单到达
    VEHICLE_ARRIVE = auto()     # 车辆到达站点
    ORDER_COMPLETE = auto()     # 【修复】订单完成事件 (之前漏了这个)
    # ORDER_CANCEL = auto()     # (预留给下一阶段)

class Decision(Enum):
    REJECT = 0
    HOLD = 1
    ACCEPT = 2

class PlanMode(Enum):
    FAST_INSERT = 0
    LOCAL_REPLAN = 1
    GLOBAL_REPLAN = 2

class OrderStatus(Enum):
    PENDING = 0     # 等待分配
    ASSIGNED = 1    # 已分配，车在路上
    PICKED_UP = 2   # 乘客已上车
    COMPLETED = 3   # 到达目的地
    CANCELLED = 4   # 订单取消

class StopType(Enum):
    PICKUP = 0      # 上车点
    DROPOFF = 1     # 下车点
    IDLE = 2        # 车辆待命/回库

# ==========================================
# 2. 站点 (Stop)
# ==========================================
@dataclass
class Stop:
    location: int
    stop_type: StopType
    order_id: Optional[int]
    estimated_arrival_time: int = 0

# ==========================================
# 3. 订单 (Order)
# ==========================================
@dataclass(frozen=False)
class Order:
    order_id: int
    t_request: int
    origin: int
    destination: int
    pax: int = 1
    status: OrderStatus = OrderStatus.PENDING
    max_wait_time: int = 1800
    pickup_deadline: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.pickup_deadline == 0:
            self.pickup_deadline = self.t_request + self.max_wait_time

# ==========================================
# 4. 车辆 (Vehicle)
# ==========================================
@dataclass
class Vehicle:
    vehicle_id: int
    capacity: int = 8
    location: int = 0
    next_free_time: int = 0
    schedule: List[Stop] = field(default_factory=list)
    passengers: Set[int] = field(default_factory=set)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def load(self) -> int:
        return len(self.passengers)

# ==========================================
# 5. 事件 (Event)
# ==========================================
@dataclass(order=True)
class Event:
    time: int
    seq: int = field(default=0)
    event_type: "EventType" = field(default=None, compare=False)
    order_id: Optional[int] = field(default=None, compare=False)
    vehicle_id: Optional[int] = field(default=None, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)