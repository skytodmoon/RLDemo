import pytest
from html.parser import HTMLParser

from app import app


class LearningCards(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.current = {"id": attrs.get("id"), "buttons": [], "results": []}
            self.cards.append(self.current)
        elif self.current is not None:
            if tag == "button" and "data-lab" in attrs:
                self.current["buttons"].append(attrs)
            if tag == "div" and "lab-result" in attrs.get("class", "").split():
                self.current["results"].append(attrs)

    def handle_endtag(self, tag):
        if tag == "article":
            self.current = None


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


def test_every_run_button_controls_a_result_inside_its_own_lesson(client):
    parser = LearningCards()
    parser.feed(client.get("/learning").get_data(as_text=True))
    assert len(parser.cards) == 12
    targets = []
    for card in parser.cards:
        if not card["buttons"]:
            continue
        assert len(card["buttons"]) == len(card["results"]) == 1
        button, region = card["buttons"][0], card["results"][0]
        assert button["aria-controls"] == region["id"]
        assert region["role"] == "status"
        assert region["aria-live"] == "polite"
        targets.append(region["id"])
    assert len(targets) == len(set(targets)) == 7


@pytest.mark.parametrize("lab", ["dp", "prediction", "control", "planning", "bandit", "exploration"])
def test_each_visible_experiment_returns_complete_json(client, lab):
    response = client.post("/api/learning/run", json={"lab": lab, "seed": 0})
    assert response.status_code == 200
    assert response.is_json
    assert response.json["result"]["lab"] == lab
    assert response.json["result"]["checks"]
    assert isinstance(response.json["passed"], bool)
    if lab != "dp":
        assert response.json["result"]["runs"]


def test_experiment_exception_returns_visible_json_error_and_allows_retry(client, monkeypatch):
    import rl_learning.web as web
    original = web.run_lab

    def broken(*args, **kwargs):
        raise RuntimeError("internal simulation detail")

    monkeypatch.setattr(web, "run_lab", broken)
    response = client.post("/api/learning/run", json={"lab": "bandit", "seed": 0})
    assert response.status_code == 500 and response.is_json
    assert "实验计算失败" in response.json["error"]
    assert "internal simulation detail" not in response.get_data(as_text=True)
    monkeypatch.setattr(web, "run_lab", original)
    assert client.post("/api/learning/run", json={"lab": "bandit", "seed": 0}).status_code == 200


def test_serialization_failure_is_also_json(client, monkeypatch):
    monkeypatch.setattr("rl_learning.web.run_lab", lambda *args, **kwargs:
                        {"checks": {"ok": True}, "unserializable": object()})
    response = client.post("/api/learning/run", json={"lab": "dp", "seed": 0})
    assert response.status_code == 500 and response.is_json
    assert "error" in response.json


def test_learning_request_script_is_served(client):
    response = client.get("/static/learning_request.js")
    assert response.status_code == 200
    assert "requestLab" in response.get_data(as_text=True)


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
