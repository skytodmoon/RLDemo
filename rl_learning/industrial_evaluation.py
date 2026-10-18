"""Frozen-checkpoint evaluation for the existing TEP dashboard."""

from pathlib import Path

import numpy as np


def collect_tep_evaluation(agent, env, seeds):
    """Collect actual measurements without training or modifying the policy."""
    names = ("REACTOR_TEMP_NORM", "REACTOR_PRESS_NORM", "REACTOR_LEVEL_NORM",
             "SEPARATOR_TEMP_NORM", "SEPARATOR_LEVEL_NORM", "STRIPPER_LEVEL_NORM",
             "PRODUCT_FLOW_NORM", "PRODUCT_COMP_A_NORM", "FEED_TOTAL_NORM",
             "FEED_RATIO_D_NORM", "COOLING_TEMP_NORM", "RECYCLE_FLOW_NORM",
             "PURGE_RATE_NORM", "COMPRESSOR_WORK_NORM", "AGITATOR_SPEED_NORM")
    target = np.array([getattr(env, name) for name in names])
    states, flags, compositions, energy, production, margins = [], [], [], [], [], []
    violations, rewards, shutdowns = [], [], 0
    for seed in seeds:
        state, _ = env.reset(seed=int(seed))
        total, episode_violations = 0.0, 0
        for _ in range(env.max_steps):
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            actual = env.state.copy()
            states.append(actual)
            flags.append(info["product_quality"])
            compositions.append(float(actual[7]))
            energy.append(float(actual[13]))
            production.append(float(actual[6]))
            # Normalized distance to the four emergency shutdown boundaries.
            margin = min((env.REACTOR_TEMP_MAX - actual[0]) / (env.REACTOR_TEMP_MAX - 80),
                         (env.REACTOR_PRESS_MAX - actual[1]) / (env.REACTOR_PRESS_MAX - 2000),
                         (env.REACTOR_LEVEL_MAX - actual[2]) / (env.REACTOR_LEVEL_MAX - env.REACTOR_LEVEL_MIN),
                         (actual[2] - env.REACTOR_LEVEL_MIN) / (env.REACTOR_LEVEL_MAX - env.REACTOR_LEVEL_MIN))
            margins.append(float(margin))
            episode_violations += int(margin <= 0)
            total += reward
            if terminated or truncated:
                shutdowns += int(terminated)
                break
        violations.append(episode_violations)
        rewards.append(total)
    return {"states": np.array(states), "setpoints": np.tile(target, (len(states), 1)),
            "quality_flags": flags, "product_compositions": np.array(compositions),
            "safety_violations": violations, "emergency_shutdowns": shutdowns,
            "constraint_margins": np.array(margins), "energy_consumption": np.array(energy),
            "production_output": np.array(production), "rewards": np.array(rewards),
            "production_quality": np.array([int(f == "Pass") for f in flags])}


def evaluate_tep_checkpoint(checkpoint, seeds=range(20000, 20010), max_steps=100):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError("请先训练 SAC 并配置 TEP_SAC_CHECKPOINT，当前没有可评估的检查点。")
    from deep_rl_agent import SACAgent
    from tennessee_eastman_env import TennesseeEastmanEnv
    from tep_evaluator import evaluate_tep_sac

    agent = SACAgent(state_dim=15, action_dim=6)
    agent.load(str(path))
    agent.actor.eval()
    agent.critic.eval()
    env = TennesseeEastmanEnv(max_steps=max_steps)
    try:
        data = collect_tep_evaluation(agent, env, seeds)
        return evaluate_tep_sac(data), agent
    finally:
        env.close()
