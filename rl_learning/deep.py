"""Train and evaluate real policies in the repository's temperature environment.

Training uses centered observations and rewards divided by ten. Evaluation
always reports the original environment's rewards and temperature errors.
"""

import random
from pathlib import Path

import gymnasium as gym
import numpy as np

from industrial_temp_env import IndustrialTempEnv


class CenterTemperature(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-0.5, 1.0, shape=(1,), dtype=np.float32)

    def observation(self, observation):
        return ((observation - 25.0) / 10.0).astype(np.float32)


class ScaleTrainingReward(gym.RewardWrapper):
    def reward(self, reward):
        return reward / 10.0


def evaluate_temperature(policy, seeds, label):
    """A separate environment and common initial conditions for every policy."""
    records = []
    env = IndustrialTempEnv()
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            rng = np.random.default_rng(int(seed))
            total, errors, violations, comfort = 0.0, [], 0, 0
            for step in range(env.max_steps):
                if policy == "random":
                    action = int(rng.integers(env.action_space.n))
                elif policy == "rule":
                    action = 1 if obs[0] < env.target_temp - 0.05 else (2 if obs[0] > env.target_temp + 0.2 else 0)
                else:
                    action = int(policy(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                errors.append(info["temperature_error"])
                violations += int(info["boundary_violation"])
                comfort += int(info["temperature_error"] <= 0.5)
                if terminated or truncated:
                    break
            records.append({"policy": label, "eval_seed": int(seed), "return": float(total),
                            "mae": float(np.mean(errors)), "comfort_rate": comfort / (step + 1),
                            "violations": violations, "steps": step + 1})
    finally:
        env.close()
    return {"return_mean": float(np.mean([r["return"] for r in records])),
            "return_std": float(np.std([r["return"] for r in records])),
            "mae_mean": float(np.mean([r["mae"] for r in records])),
            "comfort_rate": float(np.mean([r["comfort_rate"] for r in records])),
            "violations": sum(r["violations"] for r in records), "episodes": records}


class ReinforcePolicy:
    """Monte Carlo REINFORCE with a leave-one-trajectory-out time baseline.

    Each update collects two complete trajectories. The baseline for a sample
    uses other trajectories, so it is independent of that sample's action.
    """

    def __init__(self):
        import torch
        self.torch = torch
        self.network = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(),
                                           torch.nn.Linear(32, 32), torch.nn.Tanh(),
                                           torch.nn.Linear(32, 3))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-3)
        self.curve = []

    def action(self, raw_observation):
        with self.torch.no_grad():
            centered = self.torch.as_tensor((raw_observation - 25.0) / 10.0).float()
            return int(self.network(centered).argmax().item())

    def learn(self, timesteps, seed):
        torch = self.torch
        env = ScaleTrainingReward(CenterTemperature(IndustrialTempEnv()))
        obs, _ = env.reset(seed=seed)
        steps = 0
        while steps < timesteps:
            logs, all_returns, batch_rewards = [], [], []
            # Two trajectories provide an independent baseline while allowing
            # enough policy updates under a small classroom interaction budget.
            for _ in range(2):
                episode_logs, rewards = [], []
                while True:
                    distribution = torch.distributions.Categorical(logits=self.network(torch.as_tensor(obs)))
                    action = distribution.sample()
                    episode_logs.append(distribution.log_prob(action))
                    obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                    rewards.append(reward)
                    steps += 1
                    if terminated or truncated:
                        break
                running, returns = 0.0, []
                for reward in reversed(rewards):
                    running = reward + 0.95 * running
                    returns.insert(0, running)
                logs.append(torch.stack(episode_logs))
                all_returns.append(torch.tensor(returns, dtype=torch.float32))
                batch_rewards.append(sum(rewards) * 10.0)
                obs, _ = env.reset()
            returns = torch.stack(all_returns)
            baseline = (returns.sum(dim=0, keepdim=True) - returns) / (len(logs) - 1)
            advantages = returns - baseline
            advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-8)
            discount_weights = 0.95 ** torch.arange(returns.shape[1], dtype=torch.float32)
            loss = -(torch.stack(logs) * advantages.detach() * discount_weights).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            self.curve.append({"timesteps": steps, "training_return": float(np.mean(batch_rewards))})
        env.close()
        return steps


def run_deep(algorithm="ppo", seeds=(0, 1, 2), timesteps=20000, eval_episodes=20, output=None,
             eval_seed_base=10000):
    import torch
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback

    if algorithm not in {"ppo", "dqn", "a2c", "reinforce"}:
        raise ValueError("algorithm must be ppo, dqn, a2c or reinforce")
    if (timesteps < 1 or eval_episodes < 1 or not seeds or len(set(seeds)) != len(seeds)
            or eval_seed_base < 0 or eval_seed_base + eval_episodes + len(seeds) >= 2**32):
        raise ValueError("Positive budgets and unique training seeds are required")
    # Evaluation seeds are generated from a separate stream and explicitly
    # checked against the seeds that initialize training runs.
    eval_seeds = []
    candidate = eval_seed_base
    while len(eval_seeds) < eval_episodes:
        if candidate not in seeds:
            eval_seeds.append(candidate)
        candidate += 1
    random_baseline = evaluate_temperature("random", eval_seeds, "random")
    rule_baseline = evaluate_temperature("rule", eval_seeds, "rule")
    runs = []

    class TrainingCurve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.curve = []

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.curve.append({"timesteps": self.num_timesteps,
                                       "training_return": float(info["episode"]["r"])})
            return True

    torch.set_num_threads(1)
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        curve = TrainingCurve()
        if algorithm == "reinforce":
            model = ReinforcePolicy()
            predict = model.action
            env = None
        else:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(ScaleTrainingReward(CenterTemperature(IndustrialTempEnv())))
            common = dict(policy="MlpPolicy", env=env, seed=seed, verbose=0, device="cpu",
                          gamma=0.95, learning_rate=5e-4, policy_kwargs=dict(net_arch=[32, 32]))
            if algorithm == "ppo":
                model = PPO(**common, n_steps=256, batch_size=64, n_epochs=10, ent_coef=0.01)
            elif algorithm == "a2c":
                model = A2C(**common, n_steps=32, ent_coef=0.01)
            else:
                model = DQN(**common, buffer_size=20000, learning_starts=500, batch_size=64,
                            train_freq=4, target_update_interval=500, exploration_fraction=0.5)

            def predict(obs):
                action, _ = model.predict(((obs - 25.0) / 10.0).astype(np.float32), deterministic=True)
                return int(action)

        before = evaluate_temperature(predict, eval_seeds, "untrained")
        if algorithm == "reinforce":
            actual_steps = model.learn(timesteps, seed)
            learning_curve = model.curve
        else:
            model.learn(total_timesteps=timesteps, callback=curve)
            actual_steps = model.num_timesteps
            # SB3's Monitor sees the scaled training reward.
            learning_curve = [dict(p, training_return=p["training_return"] * 10.0) for p in curve.curve]
        after = evaluate_temperature(predict, eval_seeds, "trained")
        if output:
            folder = Path(output)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{algorithm}_seed_{seed}"
            if algorithm == "reinforce":
                torch.save(model.network.state_dict(), path.with_suffix(".pt"))
            else:
                model.save(path)
        runs.append({"seed": seed, "actual_timesteps": actual_steps, "before": before,
                     "after": after, "curve": learning_curve})
        if env is not None:
            env.close()
    mean_before = float(np.mean([r["before"]["mae_mean"] for r in runs]))
    mean_after = float(np.mean([r["after"]["mae_mean"] for r in runs]))
    return {"lab": "deep", "algorithm": algorithm, "eval_seeds": eval_seeds,
            "config": {"requested_timesteps": timesteps, "eval_episodes": eval_episodes,
                       "eval_seed_base": eval_seed_base,
                       "reinforce_trajectories_per_update": 2 if algorithm == "reinforce" else None,
                       "gamma": 0.95, "reward_scale": 0.1, "observation_center": 25.0,
                       "policy_network": [32, 32], "torch_threads": 1,
                       "learning_rate": 0.003 if algorithm == "reinforce" else 0.0005,
                       "algorithm_parameters": {
                           "ppo": {"n_steps": 256, "batch_size": 64, "n_epochs": 10, "ent_coef": 0.01},
                           "a2c": {"n_steps": 32, "ent_coef": 0.01},
                           "dqn": {"buffer_size": 20000, "learning_starts": 500, "batch_size": 64,
                                   "train_freq": 4, "target_update_interval": 500, "exploration_fraction": 0.5},
                           "reinforce": {"trajectories_per_update": 2, "baseline": "leave_one_trajectory_out"},
                       }[algorithm]},
            "baselines": {"random": random_baseline, "rule": rule_baseline}, "runs": runs,
            "metrics": {"before_mae_mean": mean_before, "after_mae_mean": mean_after,
                        "after_mae_std_across_training_seeds": float(np.std([r["after"]["mae_mean"] for r in runs])),
                        "rule_mae_mean": rule_baseline["mae_mean"]},
            "checks": {"improves_over_untrained": mean_after < mean_before,
                       "beats_random_mae": mean_after < random_baseline["mae_mean"],
                       "mae_below_one_degree": mean_after < 1.0,
                       "no_boundary_violations": all(r["after"]["violations"] == 0 for r in runs)},
            "interpretation": "规则基线是独立参照，不要求 RL 必须胜过它。仅验证该简化温控模型，不验证 HVAC/TEP 或真实设备。"}
