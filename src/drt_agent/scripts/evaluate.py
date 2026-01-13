from __future__ import annotations

import argparse
import yaml
import numpy as np
import torch
from dataclasses import dataclass, asdict
import csv
from typing import List, Optional

from drt_agent.env.dispatch_env import DispatchEnv
from drt_agent.rl.dqn.network import QNetwork


@dataclass
class EvalResult:
    ep: int
    steps: int
    total_reward: float
    completed: int
    cancelled: int
    accept: int
    reject: int
    hold: int
    acc_succ_rate: float


def masked_argmax(q: np.ndarray, mask: np.ndarray) -> int:
    # 将不可行动作的 Q 值设为负无穷
    q2 = np.where(mask > 0.5, q, -1e9)
    # 如果所有动作都不可行（理论上不应发生），默认选 0
    return int(q2.argmax()) if (mask > 0.5).any() else 0


def run_eval_episode(env: DispatchEnv, net: QNetwork, device: torch.device, ep: int, seed: int) -> EvalResult:
    obs, mask = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0

    # 临时统计
    out = None

    while not done:
        with torch.no_grad():
            x = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
            q = net(x).squeeze(0).cpu().numpy()

        action = masked_argmax(q, mask)
        out = env.step(action)

        total_reward += float(out.reward)
        steps += 1
        obs, mask, done = out.obs, out.action_mask, out.done

    # 从 info 中提取最终统计 (依赖 DispatchEnv.step 的 info['stats'])
    stats = out.info.get("stats", {}) if out else {}

    return EvalResult(
        ep=ep,
        steps=steps,
        total_reward=total_reward,
        completed=int(stats.get("completed", 0)),
        cancelled=int(stats.get("cancelled", 0)),
        accept=int(stats.get("accept", 0)),
        reject=int(stats.get("reject", 0)),
        hold=int(stats.get("hold", 0)),
        acc_succ_rate=float(stats.get("accept_success_rate", 0.0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--model", type=str, required=True, help="Path to model.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=30000, help="Evaluation seed")
    ap.add_argument("--out_csv", type=str, default="", help="Save results to csv")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 构造环境
    env = DispatchEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    net = QNetwork(env.obs_dim, env.num_actions, list(cfg["hidden_sizes"])).to(device)
    print(f"[Eval] Loading model from {args.model} ...")
    ckpt = torch.load(args.model, map_location=device)
    net.load_state_dict(ckpt["q"])
    net.eval()

    results: List[EvalResult] = []

    print(f"[Eval] Starting evaluation for {args.episodes} episodes...")
    for i in range(args.episodes):
        res = run_eval_episode(env, net, device, i, args.seed0 + i)
        results.append(res)
        print(
            f"  Ep {res.ep}: R={res.total_reward:.1f}, Done={res.completed}, Cancel={res.cancelled}, Steps={res.steps}")

    # 汇总统计
    if results:
        avg_r = np.mean([r.total_reward for r in results])
        avg_done = np.mean([r.completed for r in results])
        avg_cancel = np.mean([r.cancelled for r in results])

        print("\n" + "=" * 60)
        print(f"SUMMARY ({len(results)} eps):")
        print(f"  Avg Reward   : {avg_r:.2f}")
        print(f"  Avg Completed: {avg_done:.1f}")
        print(f"  Avg Cancelled: {avg_cancel:.1f}")
        print("=" * 60 + "\n")

    # 保存 CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))
        print(f"[Eval] Saved detailed results to {args.out_csv}")


if __name__ == "__main__":
    main()