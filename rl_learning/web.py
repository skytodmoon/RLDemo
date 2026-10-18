"""Bounded, local teaching experiments for the existing Flask application."""

from flask import Blueprint, current_app, jsonify, render_template, request

from .curriculum import FAST_LABS, LESSONS
from .experiments import run_lab

learning = Blueprint("learning", __name__)


@learning.get("/learning")
def index():
    return render_template("learning.html", lessons=LESSONS, labs=FAST_LABS)


@learning.get("/api/learning/curriculum")
def curriculum():
    return jsonify(lessons=LESSONS, fast_labs=FAST_LABS)


@learning.post("/api/learning/run")
def run():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="请提供 JSON 对象"), 400
    name, seed = body.get("lab"), body.get("seed", 0)
    if name not in FAST_LABS:
        return jsonify(error="请选择基础实验；神经网络训练请使用页面上的 CLI 命令"), 400
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        return jsonify(error="seed 必须是 [0, 2**32) 内的整数"), 400
    try:
        result = run_lab(name, seeds=[seed])
        return jsonify(result=result, passed=all(result["checks"].values()),
                       note="单种子课堂实验；多种子与独立评估请运行 CLI 验证。")
    except Exception:
        current_app.logger.exception("Learning experiment failed: lab=%s seed=%s", name, seed)
        return jsonify(error="实验计算失败，请重试；若重复失败，请检查服务日志", lab=name, seed=seed), 500
