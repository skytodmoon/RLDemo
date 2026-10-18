import json

import numpy as np
import pytest

from rl_learning.__main__ import main
from rl_learning.experiments import run_lab
from rl_learning.reporting import save_report
from rl_learning.tabular import GridWorld, learn_control, policy_iteration, td_target, value_iteration


def test_dp_matches_every_states_shortest_path_value():
    mdp = GridWorld()
    vi, _, _, residual = value_iteration(mdp)
    pi, _, _ = policy_iteration(mdp)
    for state in range(mdp.n_states):
        row, col = divmod(state, mdp.size)
        distance = 2 * (mdp.size - 1) - row - col
        analytic = -sum(mdp.gamma ** i for i in range(distance))
        assert vi[state] == pytest.approx(analytic)
        assert pi[state] == pytest.approx(analytic)
    assert residual < 1e-8


def test_bootstrapping_uses_termination_not_sampler_timeout():
    assert td_target(1, 10, terminated=True, gamma=0.9) == 1
    assert td_target(1, 10, terminated=False, gamma=0.9) == 10


@pytest.mark.parametrize("name", ["dp", "prediction", "control", "planning", "bandit", "exploration"])
def test_labs_reproducible_and_pass_declared_checks(name):
    result = run_lab(name, seeds=[0])
    assert all(result["checks"].values()), result["checks"]
    assert result == run_lab(name, seeds=[0])
    json.dumps(result, allow_nan=False)
    if name == "planning":
        assert {r["real_steps"] for r in result["runs"]} == {3000}
        assert result["runs"][1]["planning_updates"] == 15000


def test_planning_respects_real_environment_budget():
    _, visits, curve = learn_control(GridWorld(), planning_steps=5, real_step_budget=100)
    assert visits.sum() == 100
    assert curve[-1]["real_steps"] == 100


def test_report_retains_failure_and_cli_exits_nonzero(tmp_path, monkeypatch):
    failure = {"lab": "dp", "checks": {"oracle": False}}
    report = save_report([failure], tmp_path, [0])
    assert report["passed"] is False
    assert json.loads((tmp_path / "report.json").read_text())["checks"][0]["passed"] is False
    assert "FAIL" in (tmp_path / "report.md").read_text()
    monkeypatch.setattr("rl_learning.__main__.run_lab", lambda *args: failure)
    assert main(["run", "--lab", "dp", "--output", str(tmp_path)]) == 1


def test_report_writes_raw_curves(tmp_path):
    save_report([run_lab("prediction", [0])], tmp_path, [0])
    assert "mc_rmse" in (tmp_path / "learning_curves.csv").read_text()
    assert "numpy" in json.loads((tmp_path / "report.json").read_text())["metadata"]["versions"]


@pytest.mark.parametrize("args", [["run", "--lab", "dp", "--seeds", "0", "0"],
                                   ["run", "--lab", "dp", "--timesteps", "0"],
                                   ["run", "--lab", "dp", "--seeds", "-1"]])
def test_cli_rejects_invalid_budgets_and_duplicate_seeds(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
