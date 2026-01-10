from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from drt_agent.env.dispatch_env import DispatchEnv
from drt_agent.common.types import Decision


@dataclass
class EpResult:
    ep: int
    steps: int
    total_reward: float
    completed: int
    accept: int
    reject: int
    hold: int
    acc_succ: int
    acc_fail: int
    acc_succ_rate: float


def greedy_pick_action_v2(env: DispatchEnv, mask: np.ndarray) -> int:
    valid = np.where(mask > 0.5)[0]

    # 没有 env.sim / current_order 时：退化策略（随便选一个合法动作）
    if env.sim is None or env.current_order_id is None:
        return int(valid[0]) if len(valid) else 1  # 1=HOLD

    now = env.sim.now
    order = env.sim.orders[env.current_order_id]
    vehicles = list(env.sim.vehicles.values())

    cand_vehicle_ids, cand_feats, cand_mask = env.planner.get_candidates(order, vehicles, now, env.K)

    # ===== 关键修复：没可行候选时先 HOLD（有限次数），再 REJECT =====
    if float(np.sum(cand_mask)) <= 0.5:
        # 连续 HOLD 次数上限（默认 5 次，你可以调大到 10）
        max_hold = int(getattr(env, "cfg", {}).get("greedy_max_hold_streak", 5))
        streak = int(getattr(env, "_greedy_hold_streak", 0))

        # HOLD 动作是否可用（mask[1] == HOLD）
        can_hold = (mask.shape[0] > 1 and mask[1] > 0.5)

        if can_hold and streak < max_hold:
            env._greedy_hold_streak = streak + 1
            return 1  # HOLD

        # 超过连续 HOLD 次数 or 不能 HOLD -> REJECT
        env._greedy_hold_streak = 0
        return 0  # REJECT
    else:
        # 一旦出现可行候选，清空 streak
        env._greedy_hold_streak = 0

    best_action: Optional[int] = None
    best_eta: Optional[float] = None

    # 在所有合法 action 里找 ACCEPT 且 cand 可行，选 eta 最小
    for aid in valid:
        decision, cand_k, mode = env._decode_action(int(aid))
        if decision != Decision.ACCEPT:
            continue
        if cand_k is None or cand_k < 0 or cand_k >= len(cand_feats):
            continue
        if cand_mask[cand_k] < 0.5:
            continue

        eta = float(cand_feats[cand_k, 3])  # 第4列 eta_complete
        if best_eta is None or eta < best_eta:
            best_eta = eta
            best_action = int(aid)

    if best_action is not None:
        return best_action

    # 理论上：cand_mask>0 时应该能找到 accept；兜底
    return 0  # REJECT


def run_one_episode(env: DispatchEnv, ep: int, seed0: int) -> EpResult:
    obs, mask = env.reset(seed=seed0 + ep)
    done = False

    total_reward = 0.0
    steps = 0
    completed_total = 0

    out = None
    while not done:
        action = greedy_pick_action_v2(env, mask)
        out = env.step(action)

        total_reward += float(out.reward)
        steps += 1
        completed_total += len(out.info.get("completed_in_between", []))

        obs, mask, done = out.obs, out.action_mask, out.done

    stats: Dict[str, Any] = (out.info.get("stats", {}) if out is not None else {})  # type: ignore[union-attr]

    return EpResult(
        ep=ep,
        steps=int(steps),
        total_reward=float(total_reward),
        completed=int(completed_total),
        accept=int(stats.get("accept", 0)),
        reject=int(stats.get("reject", 0)),
        hold=int(stats.get("hold", 0)),
        acc_succ=int(stats.get("accept_success", 0)),
        acc_fail=int(stats.get("accept_fail", 0)),
        acc_succ_rate=float(stats.get("accept_success_rate", 0.0)),
    )


def summarize(name: str, results: list[EpResult]) -> None:
    if not results:
        return
    tr = np.array([r.total_reward for r in results], dtype=np.float64)
    comp = np.array([r.completed for r in results], dtype=np.float64)
    rate = np.array([r.acc_succ_rate for r in results], dtype=np.float64)
    steps = np.array([r.steps for r in results], dtype=np.float64)

    print("\n" + "=" * 80)
    print(f"[{name}] SUMMARY over {len(results)} episodes")
    print(f"  total_reward: mean={tr.mean():.3f} std={tr.std(ddof=0):.3f} min={tr.min():.3f} max={tr.max():.3f}")
    print(f"  completed   : mean={comp.mean():.3f} std={comp.std(ddof=0):.3f} min={comp.min():.0f} max={comp.max():.0f}")
    print(f"  steps       : mean={steps.mean():.1f} std={steps.std(ddof=0):.1f} min={steps.min():.0f} max={steps.max():.0f}")
    print(f"  acc_succ_rate: mean={rate.mean():.3f} std={rate.std(ddof=0):.3f} min={rate.min():.3f} max={rate.max():.3f}")
    print("=" * 80 + "\n")


def maybe_write_csv(path: Optional[str], results: list[EpResult]) -> None:
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"[Greedy] wrote csv -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=1, help="每隔多少个 episode 打印一次（1=每个都打印）")
    ap.add_argument("--out_csv", type=str, default="", help="可选：保存每个 episode 的结果到 csv")
    ap.add_argument("--debug", action="store_true", help="强制打开 env/planner debug（否则用 yaml 里的 debug）")
    ap.add_argument("--no_debug", action="store_true", help="强制关闭 env/planner debug（否则用 yaml 里的 debug）")
    args = ap.parse_args()

    env = DispatchEnv.load_from_yaml(args.config)

    # 统一控制 debug（避免刷屏）
    if args.debug:
        env.cfg["debug"] = True
    if args.no_debug:
        env.cfg["debug"] = False

    results: list[EpResult] = []
    for ep in range(int(args.episodes)):
        r = run_one_episode(env, ep, int(args.seed0))
        results.append(r)

        if args.log_every > 0 and (ep % int(args.log_every) == 0 or ep == int(args.episodes) - 1):
            print(
                f"[Greedy] ep={r.ep} steps={r.steps} total_reward={r.total_reward:.3f} completed={r.completed}"
                f" | accept={r.accept} reject={r.reject} hold={r.hold}"
                f" acc_succ={r.acc_succ} acc_fail={r.acc_fail} acc_succ_rate={r.acc_succ_rate:.3f}"
            )

    summarize("Greedy", results)
    maybe_write_csv(args.out_csv if args.out_csv else None, results)


if __name__ == "__main__":
    main()
