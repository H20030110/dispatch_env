from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml

# 注意：这里导入 PlanMode
from drt_agent.common.types import Order, Vehicle, Decision, PlanMode
from drt_agent.sim.simulator import Simulator
from drt_agent.planner.greedy_planner import GreedyPlanner
from drt_agent.env.obs_builder import ObsSpec, flatten_obs, build_action_mask


@dataclass
class StepOutput:
    obs: np.ndarray
    action_mask: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class DispatchEnv:
    """学习版最小环境：事件驱动 + 候选车辆 + action mask。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config

        self.rng = np.random.default_rng(int(self.cfg["seed"]))
        self.num_nodes = int(self.cfg["num_nodes"])
        self.episode_horizon = int(self.cfg["episode_horizon"])

        self.fleet_size = int(self.cfg["fleet_size"])
        self.vehicle_capacity = int(self.cfg["vehicle_capacity"])

        self.num_orders = int(self.cfg["num_orders"])
        self.interarrival_mean = float(self.cfg["interarrival_mean"])

        # HOLD 相关配置（防抖关键）
        self.hold_delay = int(self.cfg.get("hold_delay", 60))
        self.hold_delay_min = int(self.cfg.get("hold_delay_min", 5))
        self.max_hold_per_order = int(self.cfg.get("max_hold_per_order", 5))

        self.K = int(self.cfg["num_candidates"])
        self.num_modes = 3  # 对应 PlanMode 的枚举数量

        # reward weights
        self.r_accept_success = float(self.cfg["r_accept_success"])
        self.r_accept_fail = float(self.cfg["r_accept_fail"])
        self.r_reject = float(self.cfg["r_reject"])
        self.r_hold = float(self.cfg["r_hold"])
        self.r_complete = float(self.cfg["r_complete"])

        self.planner = GreedyPlanner(vehicle_capacity=self.vehicle_capacity)

        self.sim: Optional[Simulator] = None
        self.current_order_id: Optional[int] = None

        # 维度定义
        self.obs_dim = 4 + 1 + self.K * 4 + self.K
        self.num_actions = 2 + self.K * self.num_modes
        self.spec = ObsSpec(
            obs_dim=self.obs_dim,
            num_actions=self.num_actions,
            num_candidates=self.K,
            num_modes=self.num_modes,
        )

        # 统计每个订单被 HOLD 的次数
        self._hold_counts: Dict[int, int] = {}

    @staticmethod
    def load_from_yaml(path: str) -> "DispatchEnv":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return DispatchEnv(cfg)

    # -------------------------
    # 订单流与车辆初始化
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
            node = int(self.rng.integers(0, self.num_nodes))
            vehicles.append(Vehicle(vehicle_id=vid, capacity=self.vehicle_capacity, node=node, busy_until=0, load=0))
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
        self.planner.cfg = self.cfg

        orders = self._generate_orders()
        self.sim.load_orders(orders)

        res = self.sim.run_until_next_arrival()
        self.current_order_id = res.decision_order_id

        # 重置统计
        self.n_accept = 0
        self.n_reject = 0
        self.n_hold = 0
        self.n_accept_success = 0
        self.n_completed = 0
        self.n_accept_fail = 0
        self._hold_counts = {}

        # Debug 计数器重置
        if hasattr(self, "_dbg_cand_cnt"): delattr(self, "_dbg_cand_cnt")
        if hasattr(self, "_dbg_am_cnt"): delattr(self, "_dbg_am_cnt")

        obs, mask = self._build_obs_and_mask()
        return obs, mask

    def step(self, action_id: int) -> StepOutput:
        assert self.sim is not None, "Call reset() first."
        if self.current_order_id is None:
            # 如果已经 done 了还被调用，返回全 0
            return StepOutput(np.zeros(self.obs_dim), np.zeros(self.num_actions), 0.0, True, {})

        now = self.sim.now
        order = self.sim.orders[self.current_order_id]

        reward = 0.0
        info: Dict[str, Any] = {"t": now, "order_id": order.order_id}

        decision, cand_k, mode = self._decode_action(action_id)
        info["decision"] = decision.name
        info["cand_k"] = cand_k
        info["mode"] = mode.name if mode is not None else None

        # 统计动作
        if decision == Decision.REJECT:
            self.n_reject += 1
        elif decision == Decision.HOLD:
            self.n_hold += 1
        elif decision == Decision.ACCEPT:
            self.n_accept += 1

        # 执行动作
        vehicles = list(self.sim.vehicles.values())
        # 需要获取 candidates 来判断 accept 是否合法，或者做 reject/hold 的辅助判断
        cand_vehicle_ids, cand_feats, cand_mask = self.planner.get_candidates(order, vehicles, now, self.K)

        if decision == Decision.REJECT:
            reward += self.r_reject
            # toy: 拒绝即不再处理该订单

        elif decision == Decision.HOLD:
            reward += self.r_hold

            # 1. 检查单订单 HOLD 次数上限
            hc = self._hold_counts.get(order.order_id, 0) + 1
            self._hold_counts[order.order_id] = hc
            info["hold_count"] = hc

            if hc > self.max_hold_per_order:
                # 超过次数，强制转为拒绝
                reward += self.r_reject  # 叠加拒绝惩罚
                self.n_reject += 1  # 计入 reject 统计（虽然 action 是 hold）
            else:
                # 2. 智能等待：跳到下一个车辆释放的时间点，但至少 wait hold_delay_min
                now2 = self.sim.now
                vehicles2 = list(self.sim.vehicles.values())
                future_busy_until = [int(v.busy_until) for v in vehicles2 if int(v.busy_until) > now2]

                if not future_busy_until:
                    # 所有车都空闲（或 busy<=now），说明不需要等 -> 但如果策略选了 HOLD，说明还没准备好
                    # 这里为了防止死循环，强制 wait 一个 min 时间
                    delay = self.hold_delay_min
                else:
                    next_free_t = min(future_busy_until)
                    raw_delay = next_free_t - now2
                    # 限制 delay 范围
                    delay = max(self.hold_delay_min, raw_delay)
                    # 可选：不超过最大等待上限
                    # delay = min(delay, 3600)

                info["hold_delay_used"] = int(delay)
                self.sim.defer_order(order.order_id, hold_delay=int(delay))

        elif decision == Decision.ACCEPT:
            # 验证合法性
            if cand_k is None or cand_k < 0 or cand_k >= self.K or cand_mask[cand_k] < 0.5:
                # 选了无效候选
                reward += self.r_accept_fail
                info["accept_success"] = False
                self.n_accept_fail += 1
            else:
                vid = cand_vehicle_ids[cand_k]
                success, complete_time = self.sim.assign_order_to_vehicle(order.order_id, vehicle_id=vid)
                if success:
                    reward += self.r_accept_success
                    self.n_accept_success += 1
                    info["vehicle_id"] = int(vid)
                else:
                    reward += self.r_accept_fail
                    self.n_accept_fail += 1
                info["accept_success"] = bool(success)

        # 推进仿真
        res = self.sim.run_until_next_arrival()

        # 结算中途完成的订单
        completed = res.completed_orders
        reward += self.r_complete * float(len(completed))
        self.n_completed += len(completed)
        info["completed_in_between"] = completed

        # 更新状态
        self.current_order_id = res.decision_order_id
        done = self.current_order_id is None

        if done:
            # Episode 结束时的统计信息
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

        # 构建下一步 obs
        obs, mask = self._build_obs_and_mask()
        return StepOutput(obs=obs, action_mask=mask, reward=float(reward), done=False, info=info)

    def _build_obs_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.sim is not None and self.current_order_id is not None
        now = self.sim.now
        order = self.sim.orders[self.current_order_id]
        vehicles = list(self.sim.vehicles.values())

        cand_vehicle_ids, cand_feats, cand_mask = self.planner.get_candidates(order, vehicles, now, self.K)

        # [Debug] 仅当 debug=True 且前3次时打印
        if self.cfg.get("debug", False):
            if not hasattr(self, "_dbg_cand_cnt"):
                self._dbg_cand_cnt = 0
            if self._dbg_cand_cnt < 3:
                print(f"[DBG] now={now} order={order.order_id}")
                print(f"[DBG] cand_mask={cand_mask}")
                self._dbg_cand_cnt += 1

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

        # [Debug] 仅当 debug=True 且前3次时打印

        if self.cfg.get("debug", False):
            if not hasattr(self, "_dbg_am_cnt"):
                self._dbg_am_cnt = 0
            if self._dbg_am_cnt < 3:
                valid_actions = np.where(action_mask > 0.5)[0]
                print(f"[DBG] valid_actions count={len(valid_actions)} sample={valid_actions[:10]}")
                self._dbg_am_cnt += 1

        return obs, action_mask

    def _decode_action(self, action_id: int) -> Tuple[Decision, Optional[int], Optional[PlanMode]]:
        if action_id == 0:
            return Decision.REJECT, None, None
        if action_id == 1:
            return Decision.HOLD, None, None

        x = action_id - 2
        k = x // self.num_modes
        m = x % self.num_modes
        # 这里使用 PlanMode
        mode = PlanMode(m)
        return Decision.ACCEPT, int(k), mode