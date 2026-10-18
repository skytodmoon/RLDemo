#!/usr/bin/env python3
"""Manual TEP training/checkpoint evaluation command, not a pytest test.

No checkpoint means no performance score. Evaluation never updates the agent.
"""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="评估已保存的 TEP SAC 检查点")
    parser.add_argument("--checkpoint", default="sac_tep_model.pt")
    parser.add_argument("--train-episodes", type=int, default=0, help="显式指定后才先训练")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-base", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.train_episodes < 0 or args.max_steps < 1 or args.eval_episodes < 1 or args.eval_seed_base < 0:
        parser.error("训练回合数不得为负，步数/评估回合数必须为正，种子不得为负")
    if args.train_episodes:
        from deep_rl_agent import train_sac
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        train_sac(num_episodes=args.train_episodes, max_steps=args.max_steps, save_path=args.checkpoint)
    from rl_learning.industrial_evaluation import evaluate_tep_checkpoint
    from tep_evaluator import generate_evaluation_report
    try:
        result, _ = evaluate_tep_checkpoint(args.checkpoint,
            seeds=range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes), max_steps=args.max_steps)
    except FileNotFoundError as error:
        parser.exit(2, str(error) + "\n")
    print(generate_evaluation_report(result))
    print("范围：简化 TEP 模型、所列评估种子与检查点；不是实际设备性能或安全认证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
