import numpy as np
import pytest

torch = pytest.importorskip("torch")

from building_hvac_env import BuildingHVACEnv
from model_based_rl import MBPOAgent
from multi_agent_ppo import MultiAgentPPO, MultiAgentRolloutBuffer


def test_mappo_behavior_ratio_is_one_before_any_update():
    torch.manual_seed(0)
    agent = MultiAgentPPO()
    obs, _ = BuildingHVACEnv().reset(seed=0)
    action, zone_inputs, old_logs, _ = agent.select_action(obs)
    assert old_logs.shape == (4,)
    for zone, actor in enumerate(agent.actors):
        logs, _ = actor.evaluate_actions(torch.tensor(zone_inputs[zone]).unsqueeze(0),
                                         torch.tensor(action[zone * 4:(zone + 1) * 4]).unsqueeze(0))
        ratio = torch.exp(logs - old_logs[zone])
        assert ratio.item() == pytest.approx(1.0, abs=1e-5)


def test_mappo_timeout_bootstraps_but_gae_stops_at_episode_boundary():
    buffer = MultiAgentRolloutBuffer()
    inputs = [np.zeros(6, np.float32) for _ in range(4)]
    args = (inputs, np.zeros(16), np.full(16, 0.5), np.zeros(4))
    buffer.add(*args, reward=1, value=2, done=True, next_value=10, terminated=False)
    buffer.add(*args, reward=100, value=0, done=True, next_value=10, terminated=True)
    advantage, returns = buffer.compute_returns_and_advantages(gamma=0.9, lam=1.0)
    assert advantage[0].item() == pytest.approx(8)
    assert returns[0].item() == pytest.approx(10)
    assert returns[1].item() == pytest.approx(100)
    with pytest.raises(ValueError, match="per zone"):
        buffer.add(inputs, np.zeros(16), np.full(16, 0.5), 1.0, 1, 0, True)


def test_mappo_single_transition_update_is_finite_and_updates_actors():
    torch.manual_seed(0)
    agent = MultiAgentPPO()
    obs, _ = BuildingHVACEnv().reset(seed=0)
    before = [p.detach().clone() for p in agent.actors.parameters()]
    action, zone_obs, logs, value = agent.select_action(obs)
    agent.store_transition(zone_obs, obs, action, logs, 1.0, value, True,
                           next_value=5, terminated=False)
    stats = agent.update(num_epochs=1, batch_size=1)
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actors.parameters()))
    assert agent.update() == {}


def test_mappo_flushes_short_rollouts_at_episode_end(tmp_path):
    from multi_agent_ppo import train_mappo
    result = train_mappo(num_episodes=2, max_steps=2, update_interval=100,
                         num_update_epochs=1, save_path=str(tmp_path / "model.pt"), verbose=False)
    assert len(result["actor_losses"]) == 2


def test_learned_model_accepts_one_hot_actions_with_nondefault_action_count():
    agent = MBPOAgent(state_dim=2, action_dim=4)
    for i in range(128):
        state = np.array([0.1, 0.2], dtype=np.float32)
        agent.store_transition(state, i % 4, 1.0, state + 0.01, False)
    losses = agent.train_model(epochs=1)
    assert all(np.isfinite(value) for value in losses.values())


def test_planner_returns_first_action_of_best_particle_not_particle_index():
    class AlwaysFourth(torch.nn.Module):
        def forward(self, states):
            p = torch.zeros(len(states), 4)
            p[:, 3] = 1.0
            return p

    agent = MBPOAgent(2, 4)
    agent.policy = AlwaysFourth()
    agent.num_particles = 17
    assert agent.plan(np.zeros(2, np.float32)) == 3


def test_model_rollout_policy_gradient_learns_rewarded_action():
    class KnownReward(torch.nn.Module):
        def forward(self, states, actions):
            return actions[:, 1:2]

    torch.manual_seed(0)
    agent = MBPOAgent(2, 3)
    agent.reward_model = KnownReward()
    agent.horizon = 1
    agent.num_particles = 20
    for i in range(128):
        agent.store_transition(np.zeros(2), i % 3, 0, np.zeros(2), False)
    before = agent.policy(torch.zeros(1, 2))[0, 1].item()
    agent.train_policy(iterations=5)
    after = agent.policy(torch.zeros(1, 2))[0, 1].item()
    assert after > before
    assert all(p.grad is None for p in agent.transition_model.parameters())


def test_sac_training_preserves_bootstrap_at_time_limit(tmp_path, monkeypatch):
    import deep_rl_agent
    transitions = []

    class DummyAgent:
        def __init__(self, **kwargs):
            from collections import defaultdict
            self.stats = defaultdict(list)
            self.alpha = 0.2
        def select_action(self, state):
            return np.zeros(6, np.float32)
        def store_transition(self, state, action, reward, nxt, done):
            transitions.append(done)
        def update(self):
            return {}
        def save(self, path):
            pass

    monkeypatch.setattr(deep_rl_agent, "SACAgent", DummyAgent)
    deep_rl_agent.train_sac(num_episodes=1, max_steps=2, warmup_steps=0,
                            save_path=str(tmp_path / "unused.pt"), verbose=False)
    assert transitions == [False, False]


def test_deep_runner_records_training_and_independent_evaluation(tmp_path):
    pytest.importorskip("stable_baselines3")
    from rl_learning.deep import run_deep
    result = run_deep("ppo", seeds=[0], timesteps=256, eval_episodes=2, output=tmp_path)
    assert result["runs"][0]["actual_timesteps"] == 256
    assert not set(result["eval_seeds"]) & {0}
    assert len(result["runs"][0]["before"]["episodes"]) == 2
    assert len(result["runs"][0]["after"]["episodes"]) == 2
    assert (tmp_path / "ppo_seed_0.zip").is_file()


def test_cost_surrogate_changes_actor_and_has_finite_critic_targets():
    from constrained_rl import CPOAgent
    torch.manual_seed(0)
    agent = CPOAgent(2, 3, lam=10, c=0)
    for i in range(64):
        action = i % 2
        agent.store_transition(np.zeros(2), action, 0, float(action == 1),
                               np.zeros(2), i == 63)
    before = agent.actor(torch.zeros(1, 2))[0, 1].item()
    result = agent.train()
    assert all(np.isfinite(value) for value in result.values())
    assert agent.actor(torch.zeros(1, 2))[0, 1].item() < before
