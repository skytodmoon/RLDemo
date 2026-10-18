import pytest

from app import app


@pytest.fixture
def client(tmp_path):
    app.config.update(TESTING=True, TEP_SAC_CHECKPOINT=str(tmp_path / "missing.pt"))
    return app.test_client()


def test_learning_page_and_curriculum(client):
    response = client.get("/learning")
    assert response.status_code == 200
    assert "把每一课变成可验证的实验" in response.get_data(as_text=True)
    lessons = client.get("/api/learning/curriculum").json["lessons"]
    assert [l["lesson"] for l in lessons] == list(range(1, 13))
    assert lessons[8]["coverage"] == "guided"


def test_learning_runs_real_computation(client):
    response = client.post("/api/learning/run", json={"lab": "dp", "seed": 42})
    assert response.status_code == 200
    assert response.json["passed"] is True
    assert response.json["result"]["metrics"]["bellman_residual"] < 1e-8


@pytest.mark.parametrize("body", [{"lab": "deep"}, {"lab": "dp", "seed": True},
                                  {"lab": "dp", "seed": -1}, {"lab": "dp", "seed": 2**32},
                                  {"lab": "dp", "seed": "42"}, [], None])
def test_web_rejects_expensive_or_invalid_requests(client, body):
    assert client.post("/api/learning/run", json=body).status_code == 400


@pytest.mark.parametrize("path", ["/", "/safe-rl", "/mbpo", "/building-hvac", "/tep-control"])
def test_rule_illustrations_are_identified(client, path):
    assert "没有训练或加载 RL 策略" in client.get(path).get_data(as_text=True)


@pytest.mark.parametrize("method,path", [("get", "/api/evaluate_tep_sac"), ("post", "/api/tep/optimize")])
def test_tep_without_checkpoint_does_not_grade_random_network(client, method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 409
    assert response.json["status"] == "checkpoint_required"


def test_optimization_recommendations_are_serializable_and_not_claimed_applied(client, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("rl_learning.industrial_evaluation.evaluate_tep_checkpoint",
                        lambda _: (SimpleNamespace(), object()))
    monkeypatch.setattr("tep_optimizer.TEPOptimizer.analyze_evaluation",
                        lambda *args: [SimpleNamespace(strategy="test", description="example")])
    response = client.post("/api/tep/optimize")
    assert response.status_code == 200
    assert response.json["applied_count"] == 0
    assert response.json["recommendations"][0]["strategy"] == "test"
