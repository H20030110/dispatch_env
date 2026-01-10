from __future__ import annotations

import argparse
import os
import time
import yaml
import numpy as np

from drt_agent.env.dispatch_env import DispatchEnv
from drt_agent.rl.dqn.trainer import DQNTrainer, DQNConfig, linear_epsilon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--run_dir", type=str, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = DispatchEnv(cfg)

    run_dir = args.run_dir or os.path.join("runs", time.strftime("dqn_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    dqn_cfg = DQNConfig(
        gamma=float(cfg["gamma"]),
        lr=float(cfg["lr"]),
        batch_size=int(cfg["batch_size"]),
        replay_size=int(cfg["replay_size"]),
        warmup_steps=int(cfg["warmup_steps"]),
        train_every=int(cfg["train_every"]),
        target_update_every=int(cfg["target_update_every"]),
        eps_start=float(cfg["eps_start"]),
        eps_end=float(cfg["eps_end"]),
        eps_decay_steps=int(cfg["eps_decay_steps"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
    )

    trainer = DQNTrainer(
        obs_dim=env.obs_dim,
        num_actions=env.num_actions,
        hidden_sizes=list(cfg["hidden_sizes"]),
        cfg=dqn_cfg,
        seed=int(cfg["seed"]),
        run_dir=run_dir,
    )

    for ep in range(args.episodes):
        obs, mask = env.reset(seed=10000 + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            eps = linear_epsilon(trainer.global_step, dqn_cfg.eps_start, dqn_cfg.eps_end, dqn_cfg.eps_decay_steps)
            action = trainer.act(obs, mask, epsilon=eps)

            out = env.step(action)
            next_obs, next_mask, reward, done = out.obs, out.action_mask, out.reward, out.done

            trainer.add_transition(obs, action, reward, next_obs, done, mask, next_mask)

            obs, mask = next_obs, next_mask
            ep_reward += reward
            ep_steps += 1

            # optimize
            if len(trainer.replay) >= dqn_cfg.warmup_steps and trainer.global_step % dqn_cfg.train_every == 0:
                metrics = trainer.train_step()
            else:
                metrics = {}

            trainer.maybe_update_target()

            # log
            if metrics:
                trainer.logger.log({
                    "step": trainer.global_step,
                    "episode": ep,
                    "eps": float(eps),
                    "reward_step": float(reward),
                    **metrics
                })

            trainer.global_step += 1

        print(f"[Train] ep={ep} steps={ep_steps} ep_reward={ep_reward:.3f}")
        if (ep + 1) % 20 == 0:
            trainer.save("model.pt")

    final_path = trainer.save("model_final.pt")
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
