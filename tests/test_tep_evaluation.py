import numpy as np

from rl_learning.industrial_evaluation import collect_tep_evaluation
from tennessee_eastman_env import TennesseeEastmanEnv


def test_tep_evaluation_uses_actual_margins_and_does_not_train_policy():
    class FixedPolicy:
        def select_action(self, state, deterministic=False):
            assert deterministic
            return np.zeros(6, np.float32)
        def update(self):
            raise AssertionError("Evaluation must not train")
        def store_transition(self, *args):
            raise AssertionError("Evaluation must not alter replay")

    env = TennesseeEastmanEnv(max_steps=4)
    data = collect_tep_evaluation(FixedPolicy(), env, [10000, 10001])
    repeat = collect_tep_evaluation(FixedPolicy(), env, [10000, 10001])
    np.testing.assert_array_equal(data["states"], repeat["states"])
    states = data["states"]
    expected = np.min(np.column_stack([
        (env.REACTOR_TEMP_MAX - states[:, 0]) / (env.REACTOR_TEMP_MAX - 80),
        (env.REACTOR_PRESS_MAX - states[:, 1]) / (env.REACTOR_PRESS_MAX - 2000),
        (env.REACTOR_LEVEL_MAX - states[:, 2]) / (env.REACTOR_LEVEL_MAX - env.REACTOR_LEVEL_MIN),
        (states[:, 2] - env.REACTOR_LEVEL_MIN) / (env.REACTOR_LEVEL_MAX - env.REACTOR_LEVEL_MIN),
    ]), axis=1)
    np.testing.assert_allclose(data["constraint_margins"], expected)
    assert data["safety_violations"] == [0, 0]
    assert len(data["rewards"]) == 2
