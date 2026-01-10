import yaml
from drt_agent.env.dispatch_env import DispatchEnv


def test_env_runs_one_episode():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = DispatchEnv(cfg)
    obs, mask = env.reset(seed=123)

    done = False
    steps = 0
    while not done and steps < 10000:
        # always reject (action 0) should still terminate
        out = env.step(0)
        obs, mask, done = out.obs, out.action_mask, out.done
        steps += 1

    assert done
