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
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = DispatchEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = QNetwork(env.obs_dim, env.num_actions, list(cfg["hidden_sizes"])).to(device)
    ckpt = torch.load(args.model, map_location=device)
    net.load_state_dict(ckpt["q"])
    net.eval()

    for ep in range(args.episodes):
        obs, mask = env.reset(seed=20000 + ep)
        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                x = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
                q = net(x).squeeze(0).cpu().numpy()
            action = masked_argmax(q, mask)

            out = env.step(action)
            total_reward += out.reward
            obs, mask, done = out.obs, out.action_mask, out.done

        print(f"[Eval] ep={ep} total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
