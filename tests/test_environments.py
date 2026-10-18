import numpy as np
import pytest

from building_hvac_env import BuildingHVACEnv
from industrial_temp_env import IndustrialTempEnv
from safe_rl_env import SafeIndustrialTempEnv, SafeLayer
from tennessee_eastman_env import TennesseeEastmanEnv


@pytest.mark.parametrize("factory,action", [(IndustrialTempEnv, 1), (SafeIndustrialTempEnv, 0),
                                          (BuildingHVACEnv, np.zeros(16, np.float32)),
                                          (TennesseeEastmanEnv, np.zeros(6, np.float32))])
def test_seed_reproduces_interleaved_environments_without_global_rng(factory, action):
    left, right = factory(), factory()
    lo, _ = left.reset(seed=42)
    np.random.seed(999)
    np.random.normal(size=100)
    ro, _ = right.reset(seed=42)
    np.testing.assert_array_equal(lo, ro)
    for _ in range(5):
        lo, lr, lt, lx, _ = left.step(action)
        assert type(lt) is bool and type(lx) is bool
        np.random.normal(size=100)
        ro, rr, rt, rx, _ = right.step(action)
        np.testing.assert_array_equal(lo, ro)
        assert (lr, lt, lx) == (rr, rt, rx)
    lo, _ = left.reset(seed=42)
    ro, _ = right.reset(seed=42)
    np.testing.assert_array_equal(lo, ro)


@pytest.mark.parametrize("factory", [IndustrialTempEnv, SafeIndustrialTempEnv])
def test_temperature_horizon_is_truncation(factory):
    env = factory()
    env.reset(seed=0)
    for step in range(100):
        obs, reward, terminated, truncated, _ = env.step(1 if step % 3 == 0 else 0)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not terminated
        assert truncated == (step == 99)


def test_clipping_does_not_hide_attempted_safety_violation():
    env = IndustrialTempEnv()
    env.reset(seed=0, options={"initial_temp": 20.0})
    obs, reward, _, _, info = env.step(2)
    assert obs[0] == 20.0
    assert info["boundary_violation"] is True
    assert reward == -50.0


@pytest.mark.parametrize("initial", [float("nan"), 19.0, 36.0])
def test_invalid_initial_temperature_rejected(initial):
    with pytest.raises(ValueError):
        IndustrialTempEnv().reset(options={"initial_temp": initial})


@pytest.mark.parametrize("temperature", [20.0, 20.5, 34.5, 35.0])
@pytest.mark.parametrize("action", [0, 1, 2])
def test_safety_layer_moves_away_from_boundary(temperature, action):
    env = SafeIndustrialTempEnv()
    env.reset(seed=0)
    env.current_temp = temperature
    obs, _, _, _, _ = env.step(action)
    if temperature <= 21:
        assert obs[0] > temperature
    else:
        assert obs[0] < temperature
    layer = SafeLayer(env)
    layer.update_safety_constraints(np.array([temperature, 1], np.float32))
    expected = 1 if temperature <= 21 else (2 if action == 1 else action)
    assert layer.filter_action(action) == expected


def test_environment_does_not_reseed_numpy_global_generator():
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    BuildingHVACEnv(seed=42).reset(seed=42)
    TennesseeEastmanEnv(seed=42).reset(seed=42)
    np.testing.assert_array_equal(np.random.random(3), expected)
