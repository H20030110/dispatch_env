from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml

# 引入核心组件
from drt_agent.common.types import Order, Vehicle, Decision, PlanMode, Stop, StopType
from drt_agent.sim.simulator import Simulator
from drt_agent.planner.greedy_planner import GreedyPlanner
# 【关键】引入插入算法
from drt_agent.planner.insertion import calculate_insertion_cost
from drt_agent.env.obs_builder import ObsSpec, flatten_obs, build_action_mask


@dataclass
class StepOutput:
    obs: np.ndarray
    action_mask: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class DispatchEnv:
    """
    DARP 调度环境 (Phase 2 完成版)
    支持：
    1. 车辆按 Schedule 行驶 (Simulator)
    2. 贪心插入算法 (Insertion Heuristic)
    3. 完整的统计 (Accept/Complete)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config

        self.rng = np.random.default_rng(int(self.cfg["seed"]))
        self.num_nodes = int(self.cfg["num_nodes"])
        self.episode_horizon = int(self.cfg["episode_horizon"])

        self.fleet_size = int(self.cfg["fleet_size"])
        self.vehicle_capacity = int(self.cfg["vehicle_capacity"])

        self.num_orders = int(self.cfg["num_orders"])
        self.interarrival_mean = float(self.cfg["interarrival_mean"])
        self.hold_delay = int(self.cfg["hold_delay"])

        self.hold_delay_min = int(self.cfg.get("hold_delay_min", 5))
        self.max_hold_per_order = int(self.cfg.get("max_hold_per_order", 5))

        self.K = int(self.cfg["num_candidates"])
        self.num_modes = 3  # PlanMode

        # reward weights
        self.r_accept_success = float(self.cfg["r_accept_success"])
        self.r_accept_fail = float(self.cfg["r_accept_fail"])
        self.r_reject = float(self.cfg["r_reject"])
        self.r_hold = float(self.cfg["r_hold"])
        self.r_complete = float(self.cfg["r_complete"])

        self.planner = GreedyPlanner(vehicle_capacity=self.vehicle_capacity)

        self.sim: Optional[Simulator] = None
        self.current_order_id: Optional[int] = None

        # order_feat(4) + global_feat(1) + cand_feats(K*4) + cand_mask(K)
        self.obs_dim = 4 + 1 + self.K * 4 + self.K
        self.num_actions = 2 + self.K * self.num_modes
        self.spec = ObsSpec(
            obs_dim=self.obs_dim,
            num_actions=self.num_actions,
            num_candidates=self.K,
            num_modes=self.num_modes,
        )

        self._hold_counts: Dict[int, int] = {}

    @staticmethod
    def load_from_yaml(path: str) -> "DispatchEnv":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return DispatchEnv(cfg)

    # -------------------------
    # 内部初始化逻辑
    # -------------------------
    def _generate_orders(self) -> List[Order]:
        orders: List[Order] = []
        t = 0
        for i in range(self.num_orders):
            dt = int(self.rng.exponential(self.interarrival_mean))
            t = min(t + dt, self.episode_horizon)

            o = int(self.rng.integers(0, self.num_nodes))
            d = int(self.rng.integers(0, self.num_nodes))
            while d == o:
                d = int(self.rng.integers(0, self.num_nodes))

            orders.append(Order(order_id=i, t_request=t, origin=o, destination=d, pax=1))
        return orders

    def _init_vehicles(self) -> List[Vehicle]:
        vehicles: List[Vehicle] = []
        for vid in range(self.fleet_size):
            # Phase 2: 使用 location 初始化
            loc = int(self.rng.integers(0, self.num_nodes))
            vehicles.append(Vehicle(vehicle_id=vid, capacity=self.vehicle_capacity, location=loc))
        return vehicles

    # -------------------------
    # RL 核心接口
    # -------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        vehicles = self._init_vehicles()
        self.sim = Simulator(
            num_nodes=self.num_nodes,
            episode_horizon=self.episode_horizon,
            vehicles=vehicles,
            rng=self.rng,
        )
        self.planner.sim = self.sim
        orders = self._generate_orders()
        self.sim.load_orders(orders)

        res = self.sim.run_until_next_arrival()
        self.current_order_id = res.decision_order_id

        # episode stats
        self.n_accept = 0
        self.n_reject = 0
        self.n_hold = 0
        self.n_accept_success = 0
        self.n_completed = 0
        self.n_accept_fail = 0

        self._hold_counts = {}

        obs, mask = self._build_obs_and_mask()
        return obs, mask

    def step(self, action_id: int) -> StepOutput:
        assert self.sim is not None, "Call reset() first."
        assert self.current_order_id is not None, "No current order to decide."

        order = self.sim.orders[self.current_order_id]

        decision, cand_k, mode = self._decode_action(action_id)

        reward = 0.0
        info: Dict[str, Any] = {"t": self.sim.now, "order_id": order.order_id}
        info["decision"] = decision.name

        if decision == Decision.REJECT:
            self.n_reject += 1
        elif decision == Decision.HOLD:
            self.n_hold += 1
        elif decision == Decision.ACCEPT:
            self.n_accept += 1

        vehicles = list(self.sim.vehicles.values())
        cand_vehicle_ids, cand_feats, cand_mask = self.planner.get_candidates(order, vehicles, self.sim.now, self.K)

        # --- 分支逻辑 ---
        if decision == Decision.REJECT:
            reward += self.r_reject
            # 拒绝，无事发生，处理下一单

        elif decision == Decision.HOLD:
            reward += self.r_hold
            hc = self._hold_counts.get(order.order_id, 0) + 1
            self._hold_counts[order.order_id] = hc

            if hc > self.max_hold_per_order:
                reward += self.r_reject
                self.n_reject += 1  # 统计为被动拒绝
            else:
                # 寻找未来空闲时间
                future_times = []
                for v in vehicles:
                    if not v.schedule:
                        future_times.append(self.sim.now)
                    else:
                        future_times.append(v.schedule[-1].estimated_arrival_time)

                ft = [t for t in future_times if t > self.sim.now]

                if not ft:
                    delay = self.hold_delay_min
                else:
                    delay = max(self.hold_delay_min, min(ft) - self.sim.now)

                # 限制最大 delay
                delay = min(delay, 3600)
                self.sim.defer_order(order.order_id, int(delay))
                info["hold_delay"] = int(delay)

        elif decision == Decision.ACCEPT:
            if cand_k is None or cand_k < 0 or cand_k >= self.K or cand_mask[cand_k] < 0.5:
                reward += self.r_accept_fail
                info["accept_success"] = False
                self.n_accept_fail += 1
            else:
                vid = int(cand_vehicle_ids[cand_k])
                vehicle = self.sim.vehicles[vid]

                # ==========================================
                # Phase 2: 调用插入算法
                # ==========================================
                cost, new_schedule = calculate_insertion_cost(vehicle, order, self.sim)

                if new_schedule:
                    self.sim.assign_schedule_to_vehicle(vid, new_schedule)
                    reward += self.r_accept_success
                    self.n_accept_success += 1
                    info["accept_success"] = True
                    info["vehicle_id"] = vid
                else:
                    reward += self.r_accept_fail
                    self.n_accept_fail += 1
                    info["accept_success"] = False

        # --- 推进仿真 ---
        res = self.sim.run_until_next_arrival()
        completed = res.completed_orders

        # 奖励完成订单
        reward += self.r_complete * float(len(completed))
        self.n_completed += len(completed)
        info["completed_count"] = len(completed)

        self.current_order_id = res.decision_order_id
        done = self.current_order_id is None

        # --- 结束处理 ---
        if done:
            # 统计汇总
            den = max(1, self.n_accept_success + self.n_accept_fail)
            info["stats"] = {
                "accept": int(self.n_accept),
                "reject": int(self.n_reject),
                "hold": int(self.n_hold),
                "accept_success": int(self.n_accept_success),
                "accept_fail": int(self.n_accept_fail),
                "completed": int(self.n_completed),
                "accept_success_rate": float(self.n_accept_success / den),
            }

            obs = np.zeros((self.obs_dim,), dtype=np.float32)
            mask = np.zeros((self.num_actions,), dtype=np.float32)
            return StepOutput(obs=obs, action_mask=mask, reward=float(reward), done=True, info=info)

        obs, mask = self._build_obs_and_mask()
        return StepOutput(obs=obs, action_mask=mask, reward=float(reward), done=False, info=info)

    # -------------------------
    # 辅助函数
    # -------------------------
    def _build_obs_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.sim is not None and self.current_order_id is not None
        now = self.sim.now
        order = self.sim.orders[self.current_order_id]
        vehicles = list(self.sim.vehicles.values())

        cand_vehicle_ids, cand_feats, cand_mask = self.planner.get_candidates(order, vehicles, now, self.K)

        obs = flatten_obs(
            order=order,
            now=now,
            episode_horizon=self.episode_horizon,
            num_nodes=self.num_nodes,
            vehicles=vehicles,
            candidate_vehicle_ids=cand_vehicle_ids,
            candidate_feats=cand_feats,
            candidate_mask=cand_mask,
        )

        action_mask = build_action_mask(candidate_mask=cand_mask, num_modes=self.num_modes)
        return obs, action_mask

    def _decode_action(self, action_id: int) -> Tuple[Decision, Optional[int], Optional[PlanMode]]:
        if action_id == 0:
            return Decision.REJECT, None, None
        if action_id == 1:
            return Decision.HOLD, None, None

        x = action_id - 2
        k = x // self.num_modes
        m = x % self.num_modes
        mode = PlanMode(m)
        return Decision.ACCEPT, int(k), mode