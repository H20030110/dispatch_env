from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from drt_agent.env.dispatch_env import DispatchEnv


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


def run_one_episode(env: DispatchEnv, rng: np.random.Generator, ep: int, seed0: int) -> EpResult:
    obs, mask = env.reset(seed=seed0 + ep)
    done = False

    total_reward = 0.0
    steps = 0
    completed_total = 0

    out = None
    while not done:
        valid = np.where(mask > 0.5)[0]
        if len(valid) == 0:
            action = 0  # REJECT
        else:
            action = int(rng.choice(valid))

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

    print("\n" + "=" * 80)
    print(f"[{name}] SUMMARY over {len(results)} episodes")
    print(f"  total_reward: mean={tr.mean():.3f} std={tr.std(ddof=0):.3f} min={tr.min():.3f} max={tr.max():.3f}")
    print(f"  completed   : mean={comp.mean():.3f} std={comp.std(ddof=0):.3f} min={comp.min():.0f} max={comp.max():.0f}")
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
    print(f"[Random] wrote csv -> {path}")


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

    # 统一控制 debug（避免你屏幕刷爆）
    if args.debug:
        env.cfg["debug"] = True
    if args.no_debug:
        env.cfg["debug"] = False

    rng = np.random.default_rng(int(args.seed0))

    results: list[EpResult] = []
    for ep in range(int(args.episodes)):
        r = run_one_episode(env, rng, ep, int(args.seed0))
        results.append(r)

        if args.log_every > 0 and (ep % int(args.log_every) == 0 or ep == int(args.episodes) - 1):
            print(
                f"[Random] ep={r.ep} steps={r.steps} total_reward={r.total_reward:.3f} completed={r.completed}"
                f" | accept={r.accept} reject={r.reject} hold={r.hold}"
                f" acc_succ={r.acc_succ} acc_fail={r.acc_fail} acc_succ_rate={r.acc_succ_rate:.3f}"
            )

    summarize("Random", results)
    maybe_write_csv(args.out_csv if args.out_csv else None, results)


if __name__ == "__main__":
    main()
