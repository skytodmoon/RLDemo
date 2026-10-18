"""Persist raw measurements, learning curves and a human-readable report."""

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def metadata(seeds):
    versions = {}
    for package in ("numpy", "gymnasium", "torch", "stable-baselines3", "Flask"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                           cwd=Path(__file__).resolve().parent).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=Path(__file__).resolve().parent).strip())
        root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                            cwd=Path(__file__).resolve().parent).decode().strip())
        names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                         cwd=root).decode().splitlines()
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in sorted(set(names))
                   if (name.endswith(".py") or name in {"start.sh", "requirements.txt", "requirements-train.txt"})
                   and (root / name).is_file()}
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
        sources = {}
    return {"created_at": datetime.now(timezone.utc).isoformat(), "seeds": list(seeds),
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "git_revision": revision, "working_tree_dirty": dirty,
            "source_sha256": sources}


def _write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_report(experiments, output, seeds):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    checks = [{"experiment": e.get("algorithm", e["lab"]), "check": name, "passed": bool(passed)}
              for e in experiments for name, passed in e["checks"].items()]
    report = {"schema_version": 1, "metadata": metadata(seeds), "experiments": experiments,
              "checks": checks, "passed": all(c["passed"] for c in checks),
              "limitations": ["小型实验与简化温控的通过不能证明真实工业安全性。",
                              "L9 不确定性、TRPO、MCTS、RND/NGU 等尚无自动验收；参见课程覆盖表。",
                              "统计检查针对所列种子与预算，不构成算法的一般性能保证。"]}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    curves, evaluations = [], []
    lines = ["# 强化学习课程验证报告", "", f"记录时间：{report['metadata']['created_at']}",
             f"训练/采样种子：{list(seeds)}", f"源码提交：{report['metadata']['git_revision']}",
             f"工作区有未提交修改：{report['metadata']['working_tree_dirty']}", "",
             "## 验证结果", "", "| 实验 | 检查 | 结果 |", "| --- | --- | --- |"]
    lines += [f"| {c['experiment']} | {c['check']} | {'PASS' if c['passed'] else 'FAIL'} |" for c in checks]
    for experiment in experiments:
        name = experiment.get("algorithm", experiment["lab"])
        lines += ["", f"## {name}", ""]
        if "metrics" in experiment:
            lines += ["```json", json.dumps(experiment["metrics"], ensure_ascii=False, indent=2), "```"]
        if experiment["lab"] == "deep":
            lines += ["", "| 训练种子 | 未训练 MAE °C | 训练后 MAE °C | 训练后平均回报 |",
                      "| --- | --- | --- | --- |"]
            for run in experiment["runs"]:
                lines.append(f"| {run['seed']} | {run['before']['mae_mean']:.4f} | {run['after']['mae_mean']:.4f} | {run['after']['return_mean']:.2f} |")
            lines += ["", f"规则基线 MAE：{experiment['baselines']['rule']['mae_mean']:.4f} °C。",
                      f"随机基线 MAE：{experiment['baselines']['random']['mae_mean']:.4f} °C。"]
        for run in experiment.get("runs", []):
            label = run.get("algorithm", run.get("strategy", name))
            for point in run.get("curve", []):
                step = point.get("timesteps", point.get("step", point.get("episode")))
                for metric, value in point.items():
                    if metric not in {"timesteps", "step", "episode"}:
                        curves.append(dict(lab=experiment["lab"], algorithm=label, seed=run["seed"],
                                           step=step, metric=metric, value=value))
            if experiment["lab"] == "deep":
                for phase in ("before", "after"):
                    evaluations += [dict(algorithm=name, train_seed=run["seed"], **episode)
                                    for episode in run[phase]["episodes"]]
            elif "evaluation" in run:
                lines.append(f"- seed={run['seed']}, {label}: {run['evaluation']}, real_steps={run['real_steps']}, planning_updates={run['planning_updates']}")
            elif "mc_rmse" in run:
                lines.append(f"- seed={run['seed']}: MC RMSE={run['mc_rmse']:.4f}, TD RMSE={run['td_rmse']:.4f}")
            elif "pseudo_regret" in run:
                lines.append(f"- seed={run['seed']}, {label}: pseudo-regret={run['pseudo_regret']:.2f}, optimal-action-rate={run['optimal_action_rate']:.3f}")
        if experiment["lab"] == "deep":
            for baseline in experiment["baselines"].values():
                evaluations += [dict(algorithm=name, train_seed="baseline", **episode) for episode in baseline["episodes"]]
    lines += ["", "## 适用范围", ""] + [f"- {limit}" for limit in report["limitations"]]
    lines += ["", "运行参数、版本、逐种子指标和原始曲线见 report.json；检查失败不会被隐藏。", ""]
    (folder / "report.md").write_text("\n".join(lines))
    _write_csv(folder / "learning_curves.csv", curves, ["lab", "algorithm", "seed", "step", "metric", "value"])
    _write_csv(folder / "evaluation.csv", evaluations,
               ["algorithm", "train_seed", "policy", "eval_seed", "return", "mae", "comfort_rate", "violations", "steps"])
    return report
