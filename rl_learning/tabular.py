"""Small reference algorithms with inspectable updates and exact oracles.

These experiments isolate course concepts before using neural networks in the
industrial environments. All randomness is supplied through a local Generator.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GridWorld:
    """4x4 deterministic MDP. Actions: up/right/down/left, reward -1 per move.

    The lower-right cell is terminal. Terminal self-transitions have reward 0.
    The maximum episode length belongs to the sampler, not to the MDP.
    """

    size: int = 4
    gamma: float = 0.95

    def __post_init__(self):
        self.n_states = self.size ** 2
        self.n_actions = 4
        self.goal = self.n_states - 1
        self.next_states = np.zeros((self.n_states, 4), dtype=int)
        self.rewards = -np.ones((self.n_states, 4))
        self.terminated = np.zeros((self.n_states, 4), dtype=bool)
        for state in range(self.n_states):
            row, col = divmod(state, self.size)
            for action, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                nr = int(np.clip(row + dr, 0, self.size - 1))
                nc = int(np.clip(col + dc, 0, self.size - 1))
                nxt = nr * self.size + nc
                if state == self.goal:
                    nxt = self.goal
                    self.rewards[state, action] = 0.0
                self.next_states[state, action] = nxt
                self.terminated[state, action] = nxt == self.goal

    def q_from_v(self, values):
        return self.rewards + self.gamma * (~self.terminated) * values[self.next_states]


def value_iteration(mdp, tolerance=1e-10, max_iterations=10000):
    values = np.zeros(mdp.n_states)
    for iteration in range(1, max_iterations + 1):
        updated = mdp.q_from_v(values).max(axis=1)
        if np.max(np.abs(updated - values)) < tolerance:
            values = updated
            break
        values = updated
    else:
        raise RuntimeError("Value iteration did not converge")
    residual = float(np.max(np.abs(mdp.q_from_v(values).max(axis=1) - values)))
    return values, mdp.q_from_v(values).argmax(axis=1), iteration, residual


def evaluate_policy_exact(mdp, policy):
    """Solve (I - gamma P_pi)V = r_pi, excluding terminal bootstrapping."""
    rows = np.arange(mdp.n_states)
    transition = np.zeros((mdp.n_states, mdp.n_states))
    transition[rows, mdp.next_states[rows, policy]] = ~mdp.terminated[rows, policy]
    return np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * transition,
                           mdp.rewards[rows, policy])


def policy_iteration(mdp, max_iterations=1000):
    policy = np.zeros(mdp.n_states, dtype=int)
    for iteration in range(1, max_iterations + 1):
        values = evaluate_policy_exact(mdp, policy)
        updated = mdp.q_from_v(values).argmax(axis=1)
        if np.array_equal(updated, policy):
            return values, policy, iteration
        policy = updated
    raise RuntimeError("Policy iteration did not converge")


def epsilon_greedy(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    choices = np.flatnonzero(np.isclose(q_values, q_values.max()))
    return int(rng.choice(choices))


def td_target(reward, next_value, terminated, gamma):
    """Truncation must not set terminated=True. See Gymnasium time-limit docs."""
    return reward + gamma * (1.0 - float(terminated)) * next_value


def learn_control(mdp, algorithm="q_learning", episodes=600, seed=0,
                  planning_steps=0, bonus_scale=0.0, max_steps=100, real_step_budget=None):
    if algorithm not in {"q_learning", "sarsa"}:
        raise ValueError("algorithm must be q_learning or sarsa")
    if planning_steps and algorithm != "q_learning":
        raise ValueError("Dyna planning is defined here for Q-learning")
    rng = np.random.default_rng(seed)
    q = np.zeros((mdp.n_states, mdp.n_actions))
    counts = np.zeros_like(q, dtype=int)
    model = {}
    model_keys = []
    records = []
    real_steps = 0
    for episode in range(episodes):
        state = 0
        epsilon = max(0.05, 0.3 * (1 - episode / episodes))
        action = epsilon_greedy(q[state], epsilon, rng)
        total_reward = 0.0
        for step in range(max_steps):
            nxt = mdp.next_states[state, action]
            reward = float(mdp.rewards[state, action])
            terminal = bool(mdp.terminated[state, action])
            counts[state, action] += 1
            real_steps += 1
            bonus = bonus_scale / np.sqrt(counts[state, action])
            next_action = epsilon_greedy(q[nxt], epsilon, rng)
            next_value = q[nxt, next_action] if algorithm == "sarsa" else q[nxt].max()
            target = td_target(reward + bonus, next_value, terminal, mdp.gamma)
            q[state, action] += 0.3 * (target - q[state, action])
            if planning_steps:
                key = (state, action)
                if key not in model:
                    model_keys.append(key)
                model[key] = (reward, nxt, terminal)
                for _ in range(planning_steps):
                    ps, pa = model_keys[int(rng.integers(len(model_keys)))]
                    pr, pn, pt = model[(ps, pa)]
                    q[ps, pa] += 0.3 * (td_target(pr, q[pn].max(), pt, mdp.gamma) - q[ps, pa])
            total_reward += reward
            state, action = nxt, next_action
            if terminal or (real_step_budget is not None and real_steps >= real_step_budget):
                break
        records.append({"episode": episode, "return": total_reward, "steps": step + 1,
                        "real_steps": real_steps, "success": bool(state == mdp.goal)})
        if real_step_budget is not None and real_steps >= real_step_budget:
            break
    return q, counts, records


def rollout_grid(mdp, policy, seed=10000, episodes=100, max_steps=100):
    """Evaluate a fixed deterministic policy or a uniform random policy."""
    rng = np.random.default_rng(seed)
    returns, lengths, successes = [], [], []
    for _ in range(episodes):
        state, total = 0, 0.0
        for step in range(max_steps):
            action = int(rng.integers(mdp.n_actions)) if policy is None else int(policy[state])
            total += mdp.rewards[state, action]
            state = mdp.next_states[state, action]
            if state == mdp.goal:
                break
        returns.append(total)
        lengths.append(step + 1)
        successes.append(state == mdp.goal)
    return {"return_mean": float(np.mean(returns)), "steps_mean": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes))}


def random_walk_prediction(episodes=3000, seed=0):
    """First-visit MC and TD(0), using exactly the same sampled episodes.

    States 0 and 6 are terminal. Only reaching 6 gives reward 1. For the fair
    random policy, the analytic values of states 1..5 are [1/6, ..., 5/6].
    """
    rng = np.random.default_rng(seed)
    mc, td = np.full(7, 0.5), np.full(7, 0.5)
    mc[[0, 6]], td[[0, 6]] = 0.0, 0.0
    visits = np.zeros(7, dtype=int)
    truth = np.arange(1, 6) / 6
    curve = []
    for episode in range(episodes):
        state, trajectory = 3, []
        while state not in (0, 6):
            nxt = state + int(rng.choice([-1, 1]))
            reward = float(nxt == 6)
            terminal = nxt in (0, 6)
            td[state] += 0.05 * (td_target(reward, td[nxt], terminal, 1.0) - td[state])
            trajectory.append(state)
            state = nxt
        final_return = float(state == 6)
        for visited in set(trajectory):
            visits[visited] += 1
            mc[visited] += (final_return - mc[visited]) / visits[visited]
        if (episode + 1) % 50 == 0:
            curve.append({"episode": episode + 1,
                          "mc_rmse": float(np.sqrt(np.mean((mc[1:6] - truth) ** 2))),
                          "td_rmse": float(np.sqrt(np.mean((td[1:6] - truth) ** 2)))})
    return {"truth": truth.tolist(), "mc_values": mc[1:6].tolist(),
            "td_values": td[1:6].tolist(), "mc_rmse": curve[-1]["mc_rmse"],
            "td_rmse": curve[-1]["td_rmse"], "curve": curve}


def bandit_experiment(strategy, steps=3000, seed=0):
    if strategy not in {"epsilon_greedy", "ucb", "thompson"}:
        raise ValueError("Unknown exploration strategy")
    rng = np.random.default_rng(seed)
    means = np.array([0.1, 0.3, 0.8, 0.5])
    # One reward tape per arm gives strategies matched potential outcomes.
    tape = rng.random((steps, len(means))) < means
    counts, successes, estimates = np.zeros(4, int), np.zeros(4), np.zeros(4)
    regret, curve = 0.0, []
    for t in range(steps):
        if strategy == "epsilon_greedy":
            action = epsilon_greedy(estimates, 0.1, rng)
        elif strategy == "ucb":
            unseen = np.flatnonzero(counts == 0)
            action = int(unseen[0]) if len(unseen) else int(np.argmax(
                estimates + np.sqrt(2 * np.log(t + 1) / counts)))
        else:
            action = int(np.argmax(rng.beta(successes + 1, counts - successes + 1)))
        reward = float(tape[t, action])
        counts[action] += 1
        successes[action] += reward
        estimates[action] += (reward - estimates[action]) / counts[action]
        regret += float(means.max() - means[action])
        if (t + 1) % 100 == 0:
            curve.append({"step": t + 1, "pseudo_regret": regret})
    return {"pseudo_regret": regret, "optimal_action_rate": float(counts[2] / steps),
            "counts": counts.tolist(), "curve": curve}
