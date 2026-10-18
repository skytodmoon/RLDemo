import argparse
import json

from .curriculum import FAST_LABS, LESSONS
from .experiments import run_lab
from .reporting import save_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="强化学习课程实验与验证")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="查看 12 课的代码与实验对应关系")
    for command in ("run", "verify"):
        item = sub.add_parser(command)
        item.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        item.add_argument("--output", default="results/course")
        item.add_argument("--timesteps", type=int, default=20000)
        item.add_argument("--eval-episodes", type=int, default=20)
        item.add_argument("--eval-seed-base", type=int, default=10000,
                          help="调参用 10000，最终测试另选未使用的种子段，例如 20000")
        if command == "run":
            item.add_argument("--lab", choices=(*FAST_LABS, "deep"), required=True)
            item.add_argument("--algorithm", choices=("ppo", "dqn", "a2c", "reinforce"), default="ppo")
        else:
            item.add_argument("--include-deep", action="store_true", help="包含真正的神经网络训练")
            item.add_argument("--algorithms", nargs="+", choices=("ppo", "dqn", "a2c", "reinforce"), default=["ppo"])
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(LESSONS, ensure_ascii=False, indent=2))
        return 0
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(s < 0 or s >= 2**32 for s in args.seeds)):
        parser.error("种子必须互不重复，且在 [0, 2**32) 内")
    if args.timesteps < 1 or args.eval_episodes < 1:
        parser.error("训练步数和评估回合数必须为正数")
    if args.eval_seed_base < 0 or args.eval_seed_base + args.eval_episodes + len(args.seeds) >= 2**32:
        parser.error("评估种子范围必须在 [0, 2**32) 内")
    names = FAST_LABS if args.command == "verify" else [args.lab]
    experiments = []
    for name in names:
        if name == "deep":
            continue
        print(f"Running {name} ...", flush=True)
        experiments.append(run_lab(name, args.seeds))
    algorithms = []
    if args.command == "verify" and args.include_deep:
        algorithms = args.algorithms
    elif args.command == "run" and args.lab == "deep":
        algorithms = [args.algorithm]
    if algorithms:
        from .deep import run_deep
        for algorithm in algorithms:
            print(f"Training/evaluating {algorithm} ...", flush=True)
            experiments.append(run_deep(algorithm, args.seeds, args.timesteps,
                                        args.eval_episodes, output=args.output,
                                        eval_seed_base=args.eval_seed_base))
    report = save_report(experiments, args.output, args.seeds)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {sum(c['passed'] for c in report['checks'])}/{len(report['checks'])} checks")
    print(f"Report: {args.output}/report.md")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
