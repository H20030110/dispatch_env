from __future__ import annotations

import argparse
import os
import time
import yaml
import numpy as np
from tqdm import tqdm  # 如果没有安装 tqdm，可以 pip install tqdm

from drt_agent.env.dispatch_env import DispatchEnv
from drt_agent.rl.dqn.trainer import DQNTrainer, DQNConfig, linear_epsilon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--episodes", type=int, default=100)  # 默认跑100轮试试
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 初始化环境
    # 注意：确保 dispatch_env.py 已经应用了上一轮的 PlanMode 修复
    env = DispatchEnv(cfg)

    # 3. 准备运行目录
    run_dir = args.run_dir or os.path.join("runs", time.strftime("dqn_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Train] Run dir: {run_dir}")

    # 4. 初始化 DQN 配置
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
        eps_decay_steps=int(cfg.get("eps_decay_steps", 10000)),  # 兼容 yaml 缺失的情况
        max_grad_norm=float(cfg["max_grad_norm"]),
    )

    # 5. 初始化 Trainer
    # 关键检查：先 reset 一次拿到真实 obs 维度，防止 dispatch_env 里硬编码的 obs_dim 与实际不符
    obs, _ = env.reset(seed=args.seed)
    real_obs_dim = obs.shape[0]
    if real_obs_dim != env.obs_dim:
        print(f"[Warning] Env says obs_dim={env.obs_dim}, but real obs shape is {obs.shape}. Using real shape.")

    trainer = DQNTrainer(
        obs_dim=real_obs_dim,  # 使用真实维度
        num_actions=env.num_actions,
        hidden_sizes=list(cfg["hidden_sizes"]),
        cfg=dqn_cfg,
        seed=args.seed,
        run_dir=run_dir,
    )

    # 6. 开始训练循环
    print(f"[Train] Start training for {args.episodes} episodes...")

    # 使用 tqdm 显示总体进度
    pbar = tqdm(range(args.episodes), desc="Training")

    for ep in pbar:
        obs, mask = env.reset(seed=args.seed + 10000 + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        loss_list = []

        while not done:
            # 计算 epsilon
            eps = linear_epsilon(trainer.global_step, dqn_cfg.eps_start, dqn_cfg.eps_end, dqn_cfg.eps_decay_steps)

            # 选择动作
            action = trainer.act(obs, mask, epsilon=eps)

            # 执行动作
            out = env.step(action)
            # 兼容 StepOutput 对象
            next_obs, next_mask, reward, done = out.obs, out.action_mask, out.reward, out.done

            # 存入经验回放
            trainer.add_transition(obs, action, reward, next_obs, done, mask, next_mask)

            obs, mask = next_obs, next_mask
            ep_reward += reward
            ep_steps += 1

            # 训练一步
            if len(trainer.replay) >= dqn_cfg.warmup_steps and trainer.global_step % dqn_cfg.train_every == 0:
                metrics = trainer.train_step()
                loss_list.append(metrics["loss"])
            else:
                metrics = {}

            trainer.maybe_update_target()

            # 记录日志
            if metrics:
                trainer.logger.log({
                    "step": trainer.global_step,
                    "episode": ep,
                    "eps": float(eps),
                    "reward_step": float(reward),
                    **metrics
                })

            trainer.global_step += 1

        # 每个 episode 结束后的打印
        avg_loss = np.mean(loss_list) if loss_list else 0.0
        pbar.set_postfix({
            "R": f"{ep_reward:.1f}",
            "Steps": ep_steps,
            "Loss": f"{avg_loss:.3f}",
            "Eps": f"{eps:.2f}"
        })

        # 定期保存模型 (每 20 ep 或 最后一个 ep)
        if (ep + 1) % 20 == 0 or (ep + 1) == args.episodes:
            trainer.save("model.pt")

    # 保存最终模型
    final_path = trainer.save("model_final.pt")
    print(f"\n[Train] Done. Model saved to: {final_path}")
    print(f"[Train] Logs are in: {run_dir}/train.jsonl")


if __name__ == "__main__":
    main()