"""Course experiment orchestration. Checks are explicit, never synthetic scores."""

import numpy as np

from .tabular import (GridWorld, bandit_experiment, learn_control, policy_iteration,
                      random_walk_prediction, rollout_grid, value_iteration)


def run_lab(name, seeds=(0, 1, 2)):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one unique seed")
    mdp = GridWorld()
    if name == "dp":
        vi, vp, iterations, residual = value_iteration(mdp)
        pi, pp, policy_iterations = policy_iteration(mdp)
        # From (0,0) to (3,3) there are six moves, each with reward -1.
        oracle = -sum(mdp.gamma ** t for t in range(2 * (mdp.size - 1)))
        return {"lab": name, "metrics": {"vi_start_value": float(vi[0]),
                "pi_start_value": float(pi[0]), "analytic_start_value": oracle,
                "bellman_residual": residual, "value_iterations": iterations,
                "policy_iterations": policy_iterations, "vi_policy": vp.tolist(),
                "pi_policy": pp.tolist()},
                "checks": {"matches_analytic_value": bool(abs(vi[0] - oracle) < 1e-8),
                           "vi_pi_agree": bool(np.allclose(vi, pi, atol=1e-8)),
                           "bellman_residual_small": residual < 1e-8}}
    if name == "prediction":
        runs = [dict(seed=seed, **random_walk_prediction(seed=seed)) for seed in seeds]
        return {"lab": name, "runs": runs,
                "checks": {"mc_matches_random_walk_oracle": bool(np.mean([r["mc_rmse"] for r in runs]) < 0.08),
                           "td_matches_random_walk_oracle": bool(np.mean([r["td_rmse"] for r in runs]) < 0.12)}}
    if name in {"control", "planning", "exploration"}:
        variants = {
            "control": [("q_learning", "q_learning", 0, 0), ("sarsa", "sarsa", 0, 0)],
            "planning": [("q_learning", "q_learning", 0, 0), ("dyna_q", "q_learning", 5, 0)],
            "exploration": [("no_bonus", "q_learning", 0, 0), ("count_bonus", "q_learning", 0, 0.5)],
        }[name]
        runs = []
        for seed in seeds:
            baseline = rollout_grid(mdp, None, seed=10000 + seed)
            for label, algorithm, planning, bonus in variants:
                q, counts, curve = learn_control(mdp, algorithm, seed=seed,
                                                planning_steps=planning, bonus_scale=bonus,
                                                real_step_budget=3000 if name != "control" else None)
                evaluation = rollout_grid(mdp, q.argmax(axis=1), seed=10000 + seed)
                runs.append({"seed": seed, "algorithm": label, "evaluation": evaluation,
                             "random_baseline": baseline, "visited_pairs": int((counts > 0).sum()),
                             "real_steps": int(counts.sum()), "planning_updates": int(counts.sum()) * planning,
                             "curve": curve})
        return {"lab": name, "runs": runs, "checks": {
            "all_policies_reach_goal": all(r["evaluation"]["success_rate"] == 1 for r in runs),
            "all_policies_use_shortest_path": all(r["evaluation"]["steps_mean"] == 6 for r in runs),
            "all_policies_beat_random": all(r["evaluation"]["return_mean"] > r["random_baseline"]["return_mean"] for r in runs)},
            "interpretation": "曲线用于比较样本效率与计数奖励；不预设 Dyna 或内在奖励总能改善结果。"}
    if name == "bandit":
        runs = [dict(seed=seed, strategy=strategy, **bandit_experiment(strategy, seed=seed))
                for seed in seeds for strategy in ("epsilon_greedy", "ucb", "thompson")]
        return {"lab": name, "runs": runs, "checks": {
            "all_strategies_prefer_best_arm": all(r["optimal_action_rate"] > 0.5 for r in runs),
            "regret_below_uniform_expectation": all(r["pseudo_regret"] < 3000 * (0.8 - 0.425) for r in runs)}}
    raise ValueError(f"Unknown lab: {name}")
