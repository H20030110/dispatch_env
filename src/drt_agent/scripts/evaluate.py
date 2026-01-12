from __future__ import annotations

import argparse
import yaml
import numpy as np
import torch

from drt_agent.env.dispatch_env import DispatchEnv
from drt_agent.rl.dqn.network import QNetwork


def masked_argmax(q: np.ndarray, mask: np.ndarray) -> int:
    q2 = np.where(mask > 0.5, q, -1e9)
    return int(q2.argmax()) if (mask > 0.5).any() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20000)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["debug"] = False
    env = DispatchEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = QNetwork(env.obs_dim, env.num_actions, list(cfg["hidden_sizes"])).to(device)
    print(f"[Eval] Loading model from: {args.model}")
    try:
        ckpt = torch.load(args.model, map_location=device)
        net.load_state_dict(ckpt["q"])
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    net.eval()
    stats_list = []

    print(f"\n[Eval] Start evaluation for {args.episodes} episodes...")
    print("-" * 80)

    for ep in range(args.episodes):
        obs, mask = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        ep_info = {}

        while not done:
            with torch.no_grad():
                x = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
                q = net(x).squeeze(0).cpu().numpy()

            action = masked_argmax(q, mask)
            out = env.step(action)
            total_reward += out.reward
            obs, mask, done = out.obs, out.action_mask, out.done

            if done:
                ep_info = out.info.get("stats", {})

        # [Fix] 兼容 acc_succ 和 accept_success 两种写法
        n_succ = ep_info.get('acc_succ', ep_info.get('accept_success', 0))
        n_fail = ep_info.get('acc_fail', ep_info.get('accept_fail', 0))
        # 重新计算 rate (因为 env 返回的 rate 可能是旧 key)
        den = max(1, n_succ + n_fail)
        rate = n_succ / den

        # 将修正后的数据存回去以便汇总
        ep_info['corrected_succ'] = n_succ
        ep_info['corrected_fail'] = n_fail
        ep_info['corrected_rate'] = rate

        print(f"[Eval] Ep {ep}: R={total_reward:.1f} | "
              f"Accept={ep_info.get('accept', 0)} "
              f"Reject={ep_info.get('reject', 0)} "
              f"Hold={ep_info.get('hold', 0)} | "
              f"Succ={n_succ} "
              f"Fail={n_fail} "
              f"Rate={rate:.2f}")

        stats_list.append(ep_info)

    avg_succ = np.mean([ep['corrected_succ'] for ep in stats_list])
    avg_fail = np.mean([ep['corrected_fail'] for ep in stats_list])
    avg_rate = np.mean([ep['corrected_rate'] for ep in stats_list])

    print("-" * 80)
    print(
        f"[Summary] Avg Reward: {np.mean([ep.get('completed', 0) * env.r_complete for ep in stats_list]):.1f} (approx)")
    print(f"[Summary] Avg Succ Rate: {avg_rate:.3f}")
    print(f"[Summary] Avg Fail Count: {avg_fail:.1f} (Should be low)")
    print("-" * 80)


if __name__ == "__main__":
    main()