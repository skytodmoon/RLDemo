# 强化学习课程实践路径

本项目把课程概念、代码、实验和证据连接起来。先在有精确答案的小问题上验证公式，再把同样的训练与评估流程用于工业温控，最后扩展到 HVAC 和化工过程。

网页入口是 `/learning`。`python -m rl_learning list` 与网页读取同一份 `rl_learning/curriculum.py`，避免学习目录与代码分离。课程 PPT、视频和测验不包含在本仓库中。

## 每次学习的操作

1. 用自己的话回答本课的问题，明确状态、动作、奖励、策略和假设。
2. 阅读表中列出的实现，定位更新目标与参数更新的位置。
3. 先预测一次实验的结果，再运行代码查看检查项和原始数据。
4. 只改变一个因素，例如步长、探索率、规划次数或训练预算。
5. 保存配置、曲线与结论，并写清哪些结果推翻了原先判断。

“已阅读”只表示浏览器中记录的阅读进度。实验 PASS 只表示相应机制在所列条件下通过检查，不代表已掌握整章，也不代表论文算法或真实设备已得到验证。

## 课程与项目的对应关系

| 课程 | 阅读入口 | 实验与观察 | 当前覆盖 |
| --- | --- | --- | --- |
| L1 强化学习与 MDP | `industrial_temp_env.py`、`rl_learning/tabular.py` | 写出两种环境的 MDP，检查种子、奖励和时间上限 | MDP 基础与环境契约自动检查 |
| L2 动态规划 | `value_iteration`、`policy_iteration` | `--lab dp`：比较解析最短路径价值、VI/PI 与 Bellman 残差 | 自动验证 |
| L3 无模型预测 | `random_walk_prediction` | `--lab prediction`：相同轨迹上的 first-visit MC 与 TD(0)，比较解析值与 RMSE | 自动验证 |
| L4 无模型控制、层次 RL | `learn_control`、`hierarchical_rl.py` | `--lab control`：SARSA/Q-learning，随机基线与最短路径；再阅读 option 终止逻辑 | 表格控制已验证，层次策略收益需扩展 |
| L5 函数近似、DQN | `rl_learning/deep.py` | `--lab deep --algorithm dqn`：经验回放、目标网络、独立温控评估 | DQN 可训练；线性近似实验待扩展 |
| L6 策略梯度、REINFORCE | `ReinforcePolicy` | `--lab deep --algorithm reinforce`：采样动作、完整轨迹、回报、独立基线与 log π 更新 | 可训练并记录真实结果 |
| L7 Actor-Critic、TRPO、PPO | `rl_learning/deep.py`、`multi_agent_ppo.py` | 比较 `a2c` 与 `ppo`；回归检查每个 actor 在更新前概率比为 1 | A2C/PPO 可训练，TRPO 未实现 |
| L8 模型学习、规划 | `learn_control`、`model_based_rl.py`、`rl_mpc_hybrid.py` | `--lab planning`：相同 3000 次真实交互，比较 0/5 次规划更新 | Dyna-Q 自动验证；MCTS 未实现，MBPO 为教学近似 |
| L9 连续规划、不确定性 | `model_based_rl.py`、`rl_mpc_hybrid.py`、`tennessee_eastman_env.py` | 划分转移数据，比较一步/多步预测误差，扩展模型集成与保守规划 | 阅读与研究练习，尚无自动验收 |
| L10 Bandit、遗憾与探索 | `bandit_experiment` | `--lab bandit`：ε-greedy、UCB、Thompson Sampling 的伪遗憾与最优动作比例 | 自动验证基础机制，不验证理论渐近界 |
| L11 深度探索 | `learn_control` | `--lab exploration`：相同预算下对比 0 与 `0.5/sqrt(N(s,a))` 的计数奖励 | 仅表格机制；RND、NGU、Go-Explore 未实现 |
| L12 应用与影响 | `rl_learning/deep.py`、`deep_rl_agent.py`、`tep_evaluator.py` | 多种子温控训练、独立评估与报告；再设计 HVAC/TEP 留出故障实验 | 温控完整流程；HVAC/TEP 无性能通过结论 |

## 从基础验证过渡到应用

```bash
# 使用基础依赖即可运行
python -m rl_learning verify --seeds 0 1 2 --output results/foundations

# 先在验证种子上检查训练配置
python -m rl_learning run --lab deep --algorithm ppo --seeds 3 4 5 --timesteps 40000 --eval-seed-base 10000 --output results/validation

# 固定配置后，另选评估种子作最终测试
python -m rl_learning verify --include-deep --algorithms ppo dqn a2c reinforce --seeds 0 1 2 --timesteps 40000 --eval-seed-base 20000 --output results/full
```

`results/full/report.md` 给出检查与逐种子指标；`report.json` 保留配置、版本、源码提交和原始数据；`learning_curves.csv` 与 `evaluation.csv` 用于重新作图和分析。每个神经网络训练种子保存一个模型检查点。训练算法可能略微超过请求步数以完成一个 rollout 或 REINFORCE 轨迹批次，报告同时记录实际步数。

## 项目中的实现边界

- 原有网页模拟器使用规则或简化规划，适合观察状态和控制行为。它们没有训练或加载 RL 网络，不能用页面排名来证明算法优劣。
- `main.py` 是保留的单次 PPO 演示。课程实验优先使用新 CLI，后者提供训练前后与基线对照。
- `CPOAgent` 是为兼容旧代码保留的名称，内部是简化拉格朗日 actor-critic，没有求解论文 CPO 的约束信任域问题。
- `MBPOAgent` 学习转移和奖励，使用模型采样与策略梯度；没有实现论文 MBPO 的集成模型、合成 replay 与 SAC 混合更新。
- MAPPO 中央 critic 现在默认读取 16 维状态，包含剩余能源预算。旧的 15 维 critic 检查点需要显式构建 15 维 agent 才能加载；新训练应使用 16 维。
- PINNs 是物理建模扩展，残差损失需要数值验证。它不替代 RL 的策略训练和评估。

## 后续研究练习

优先扩展 L9 的留出模型误差实验，以及 HVAC/TEP 的规则、未训练、训练后和故障场景对照。模型集成的方差只有在留出误差和校准实验支持时，才适合解释为可用的不确定性信号。开展新的算法比较前，先固定相同真实交互预算、评估初始条件和故障分布，并明确计算成本。
