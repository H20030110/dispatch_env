from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml

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
    """学习版环境 (适配 Schedule 模式)。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(int(self.cfg["seed"]))

        self.num_nodes = int(self.cfg["num_nodes"])
        self.episode_horizon = int(self.cfg["episode_horizon"])
        self.fleet_size = int(self.cfg["fleet_size"])
        self.vehicle_capacity = int(self.cfg["vehicle_capacity"])

        self.num_orders = int(self.cfg["num_orders"])
        self.interarrival_mean = float(self.cfg["interarrival_mean"])

        self.hold_delay = int(self.cfg.get("hold_delay", 60))
        self.hold_delay_min = int(self.cfg.get("hold_delay_min", 5))
        self.max_hold_per_order = int(self.cfg.get("max_hold_per_order", 5))

        self.K = int(self.cfg["num_candidates"])
        self.num_modes = 3

        self.r_accept_success = float(self.cfg["r_accept_success"])
        self.r_accept_fail = float(self.cfg["r_accept_fail"])
        self.r_reject = float(self.cfg["r_reject"])
        self.r_hold = float(self.cfg["r_hold"])
        self.r_complete = float(self.cfg["r_complete"])

        self.planner = GreedyPlanner(vehicle_capacity=self.vehicle_capacity)
        self.sim: Optional[Simulator] = None
        self.current_order_id: Optional[int] = None

        # 4(order) + 2(global) + K*4 + K
        self.obs_dim = 4 + 2 + self.K * 4 + self.K
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

    def _generate_orders(self) -> List[Order]:
        orders: List[Order] = []
        t = 0

        # [新增配置] 最大允许延误时间 (buffer)，从 config 读取或硬编码
        # 意思是在“直达时间”的基础上，允许由于拼车导致的额外延误
        max_delay = int(self.cfg.get("max_delay", 600))  # 默认 10 分钟

        for i in range(self.num_orders):
            dt = int(self.rng.exponential(self.interarrival_mean))
            t = min(t + dt, self.episode_horizon)

            o = int(self.rng.integers(0, self.num_nodes))
            d = int(self.rng.integers(0, self.num_nodes))
            while d == o:
                d = int(self.rng.integers(0, self.num_nodes))

            # 估算直达时间 (toy: abs距离 + 5)
            # 注意：这里最好复用 Simulator.travel_time 的逻辑，但 Sim 可能还没初始化
            # 这里简单硬编码保持一致
            trip_time = int(abs(d - o) + 5)

            # deadline = 请求时间 + 直达时间 + 允许的绕路时间
            deadline = t + trip_time + max_delay

            orders.append(Order(
                order_id=i,
                t_request=t,
                origin=o,
                destination=d,
                pax=1,
                deadline=deadline  # [传递 deadline]
            ))
        return orders

    def _init_vehicles(self) -> List[Vehicle]:
        vehicles: List[Vehicle] = []
        for vid in range(self.fleet_size):
            node = int(self.rng.integers(0, self.num_nodes))
            # [Fix] 这里的 Vehicle 初始化不再传 busy_until
            vehicles.append(Vehicle(vehicle_id=vid, capacity=self.vehicle_capacity, node=node, load=0))
        return vehicles

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

        self.n_accept = 0
        self.n_reject = 0
        self.n_hold = 0
        self.n_accept_success = 0
        self.n_completed = 0
        self.n_accept_fail = 0
        self._hold_counts = {}

        if hasattr(self, "_dbg_cand_cnt"): delattr(self, "_dbg_cand_cnt")
        if hasattr(self, "_dbg_am_cnt"): delattr(self, "_dbg_am_cnt")

        obs, mask = self._build_obs_and_mask()
        return obs, mask

    def step(self, action_id: int) -> StepOutput:
        assert self.sim is not None, "Call reset() first."
        if self.current_order_id is None:
            return StepOutput(np.zeros(self.obs_dim), np.zeros(self.num_actions), 0.0, True, {})

        now = self.sim.now
        order = self.sim.orders[self.current_order_id]
        reward = 0.0
        info: Dict[str, Any] = {"t": now, "order_id": order.order_id}

        decision, cand_k, mode = self._decode_action(action_id)
        info["decision"] = decision.name
        info["cand_k"] = cand_k

        if decision == Decision.REJECT:
            self.n_reject += 1
            reward += self.r_reject

        elif decision == Decision.HOLD:
            self.n_hold += 1
            reward += self.r_hold

            hc = self._hold_counts.get(order.order_id, 0) + 1
            self._hold_counts[order.order_id] = hc

            if hc > self.max_hold_per_order:
                reward += self.r_reject
                self.n_reject += 1
            else:
                now2 = self.sim.now
                vehicles2 = list(self.sim.vehicles.values())

                # [Fix] 预测每辆车什么时候跑完 schedule
                future_finish_times = [self.sim._predict_schedule_finish_time(v) for v in vehicles2]
                # 过滤出“未来会释放”的时间点 (大于当前时间)
                valid_futures = [t for t in future_finish_times if t > now2]

                if not valid_futures:
                    delay = self.hold_delay_min
                else:
                    next_free_t = min(valid_futures)
                    raw_delay = next_free_t - now2
                    delay = max(self.hold_delay_min, raw_delay)

                self.sim.defer_order(order.order_id, hold_delay=int(delay))
                info["hold_delay_used"] = int(delay)

        elif decision == Decision.ACCEPT:
            self.n_accept += 1
            vehicles = list(self.sim.vehicles.values())
            cand_vehicle_ids, cand_feats, cand_mask = self.planner.get_candidates(order, vehicles, now, self.K)

            if cand_k is None or cand_k < 0 or cand_k >= self.K or cand_mask[cand_k] < 0.5:
                reward += self.r_accept_fail
                info["accept_success"] = False
                self.n_accept_fail += 1
            else:
                vid = cand_vehicle_ids[cand_k]
                # 尝试分配（现在 Simulator 内部会检查：如果太慢，success 就返回 False）
                success, complete_time = self.sim.assign_order_to_vehicle(order.order_id, vehicle_id=vid)

                if success:
                    reward += self.r_accept_success
                    self.n_accept_success += 1
                    info["vehicle_id"] = int(vid)
                else:
                    # [Phase 5 核心修改]
                    # 分配失败！可能是车满了，也可能是刚才 Simulator 判断乘客由于等待超时取消了。
                    # 之前是 + self.r_accept_fail (通常是 -0.5)
                    # 现在改为 -2.0 (重罚)，让智能体长记性
                    reward += -2.0
                    self.n_accept_fail += 1
                    info["accept_success"] = False

        res = self.sim.run_until_next_arrival()
        completed = res.completed_orders
        reward += self.r_complete * float(len(completed))
        self.n_completed += len(completed)
        info["completed_in_between"] = completed

        self.current_order_id = res.decision_order_id
        done = self.current_order_id is None

        if done:
            den = max(1, self.n_accept_success + self.n_accept_fail)
            info["stats"] = {
                "accept": int(self.n_accept),
                "reject": int(self.n_reject),
                "hold": int(self.n_hold),
                "acc_succ": int(self.n_accept_success),
                "acc_fail": int(self.n_accept_fail),
                "completed": int(self.n_completed),
                "rate": float(self.n_accept_success / den),
            }
            obs = np.zeros((self.obs_dim,), dtype=np.float32)
            mask = np.zeros((self.num_actions,), dtype=np.float32)
            return StepOutput(obs=obs, action_mask=mask, reward=float(reward), done=True, info=info)

        obs, mask = self._build_obs_and_mask()
        return StepOutput(obs=obs, action_mask=mask, reward=float(reward), done=False, info=info)

    def _build_obs_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.sim is not None and self.current_order_id is not None
        now = self.sim.now
        order = self.sim.orders[self.current_order_id]
        vehicles = list(self.sim.vehicles.values())
        cand_vehicle_ids, cand_feats, cand_mask = self.planner.get_candidates(order, vehicles, now, self.K)

        if self.cfg.get("debug", False):
            if not hasattr(self, "_dbg_cand_cnt"): self._dbg_cand_cnt = 0
            if self._dbg_cand_cnt < 3:
                print(f"[DBG] now={now} order={order.order_id} cand_mask={cand_mask}")
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