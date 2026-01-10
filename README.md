# DRT 调度智能体学习版最小代码骨架（从 0 开始）

这是一份**学习用**的最小可运行骨架，你可以每天“填一点点空”，逐步把它升级成更真实的网约公交调度智能体。

目标（最小闭环）：
- 有一个**事件驱动**的仿真器 Simulator（订单到达、完成等事件）
- 有一个 RL 环境 Env（reset/step，返回 observation + action_mask）
- 有一个可训练的 DQN Trainer（经验回放 + 目标网络 + 动作掩码）

> 注意：这里的仿真非常简化（车辆一次只服务一单、路径用节点差值近似），目的就是先把“智能体构建闭环”跑通。
> 你后续再逐步替换：更真实的路径规划（插入/ALNS）、取消模型、公平性指标等。

## 1. 安装与运行（建议用虚拟环境）
```bash
# 进入项目根目录（本文件所在目录）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 让 Python 能找到 src 目录
export PYTHONPATH=src       # Windows (PowerShell): $env:PYTHONPATH="src"
```

## 2. 先跑一个随机策略（确保环境没问题）
```bash
python -m drt_agent.scripts.run_random --episodes 2
```

你应该能看到每个 episode 的总 reward，以及完成单数等简易指标。

## 3. 跑 DQN 训练（最小版）
```bash
python -m drt_agent.scripts.train_dqn --episodes 50
```

会在 `runs/` 下保存训练日志与模型权重。

## 4. 代码结构（你每天“填空”主要改这里）
```
src/drt_agent/
  common/types.py        # Order / Vehicle / Event / 枚举
  sim/simulator.py       # 事件驱动仿真器（极简）
  env/dispatch_env.py    # RL 环境：reset/step、obs 与 mask
  planner/greedy_planner.py  # 候选车辆与可行性（先用 greedy 占位）
  rl/dqn/                # DQN: network / replay / trainer
  scripts/               # 运行与训练入口
```

## 5. 每天最推荐的“填空顺序”（不需要按开题时间表）
1) 看懂并能运行 `run_random.py`
2) 看懂 obs 是怎么拼出来的（`env/obs_builder.py`）
3) 训练跑通（`train_dqn.py`）并观察 reward 是否变化
4) 每天只加一个小功能：比如增加一个事件类型、增加一个特征、增加一个 reward 项、增加一个 mask 规则

## 6. 你后续怎么升级（路线提示）
- 把 `GreedyPlanner` 换成 “插入 + 可行性检查”
- 增加订单取消事件（ORDER_CANCEL）与取消损失
- 把 action 从 “扁平离散”升级为 “三段式动作（decision/vehicle/mode）”
- 把车辆从“一次一单”升级为“有路线/多停靠点”的表示

祝你填空顺利：先跑通闭环，再逐步加真实复杂度。
