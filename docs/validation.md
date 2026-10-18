# 强化学习实验验证方法

本项目区分软件回归检查与策略性能实验。前者验证环境和更新逻辑；后者检验特定训练预算下是否真的学到有效行为。软件测试通过不要求所有策略性能实验都通过。

## 软件回归检查

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# 前端请求回归（Node.js 22，无额外 npm 依赖）
node --test tests/js/*.test.cjs
```

`pytest.ini` 只收集 `tests/`。根目录的 `test_algorithms.py`、`test_hvac.py`、`test_tep_sac.py` 是历史演示/训练脚本，不能作为验收测试。若复用已有 Python 环境且第三方 pytest 插件版本冲突，可用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` 隔离无关插件；推荐独立虚拟环境。

回归检查包括：环境种子隔离、温控时间上限、安全过滤方向、裁剪前违规、解析 MDP 价值、相同预算下的 Dyna 更新、报告失败保留、MAPPO 概率比与 GAE、模型动作编码、规划动作索引、策略梯度实际改变动作概率、SAC 截断自举，以及 TEP 评估不修改策略。

页面回归检查覆盖全部六类基础实验的 JSON 响应，以及每个运行按钮与本课结果区域的对应关系。结果、指标和错误直接显示在按钮下方，不覆盖其他章节的结果。前端请求测试覆盖非 JSON 错误页、结果格式错误、断网后重试、30 秒超时和并发请求乱序返回。后台计算或序列化失败时返回 JSON 错误并记录日志，不将异常当成科学实验的 FAIL 判定。

## 可复现的性能验证

```bash
# 不依赖 PyTorch 的基础检查
python -m rl_learning verify --seeds 0 1 2 --output results/foundations

# 真正训练 PPO、DQN、A2C 和 REINFORCE
python -m rl_learning verify --include-deep --algorithms ppo dqn a2c reinforce --seeds 0 1 2 --timesteps 40000 --eval-seed-base 20000 --output results/full
```

程序不会把 FAIL 改成 PASS。检查失败仍写出全部报告，并返回退出码 1。降低预算可能导致性能失败，即使代码实现正确。不要为了让报告变绿而放宽阈值。

## 对照与数据划分

工业温控使用同一环境动力学和奖励公式。训练时将观察转换为 `(temp - 25)/10`，奖励除以 10；评估时使用原始摄氏温度与原始奖励单位。

每个训练种子都先评估未训练网络，再训练，然后用相同的独立评估种子测试。随机控制与规则控制使用相同初始温度集合。规则控制是显式阈值控制器，可能优于 RL；它不作为“RL 必须胜过”的验收条件。

默认评估种子从 10000 开始，并跳过与训练种子相同的数字。调参可以使用这一段，但调参后应冻结配置，使用新的种子段，例如 20000，作最终测试。不同算法使用同一组最终种子，便于配对比较。只更换种子不会改变环境动力学，因此这些实验不能证明跨模型、跨设备或分布外泛化。

PPO、A2C、DQN 使用 Stable-Baselines3。REINFORCE 是可读的 PyTorch 实现，每次采集两条完整轨迹，使用另一条轨迹同一时刻的回报作独立 baseline，并包含折扣时刻权重。它优化固定 100 步采样轨迹上的折扣回报，尾部没有额外学习价值函数。

## 指标与预先定义的阈值

| 实验 | 检查条件 | 能支持的结论 |
| --- | --- | --- |
| DP | VI/PI 一致，解析价值误差与 Bellman 残差 < 1e-8 | 小型确定 MDP 上更新与求解一致 |
| Prediction | 跨种子平均 MC RMSE < 0.08、TD RMSE < 0.12 | 在公平随机游走上逼近解析预测值 |
| Control / Planning / Exploration | 固定策略成功率 100%，6 步到达目标，回报高于随机 | 所列小型 MDP 上学到最短路径 |
| Bandit | 最优动作比例 > 50%，伪遗憾低于均匀随机期望 | 所列 Bernoulli bandit 上形成有效偏好 |
| Deep | 平均 MAE 优于未训练与随机控制，MAE < 1°C，所有评估回合无边界违规 | 所列简化温控初始条件下策略有效 |

Deep 的 MAE 取每回合各时间步绝对温差的平均，再跨评估回合和训练种子取平均。`after_mae_std_across_training_seeds` 是训练种子之间的标准差，不是置信区间。小样本和单个环境的通过不能当作算法普遍优越的证据。

舒适度为绝对温差 ≤ 0.5°C 的时间步比例。边界违规按裁剪前提出的温度是否越界统计，避免物理裁剪掩盖错误动作。所有神经网络评估使用确定性策略，并且不更新网络或经验回放。

## 终止、自举与随机数

温控任务可以持续控制，100 步是外部采样上限，因此环境返回 `truncated=True`。TD/SAC 在真正的 `terminated=True` 时停止自举，遇到截断仍保留下一状态价值。MAPPO 在截断处自举，同时停止 GAE 跨越 reset 传播。Gymnasium 的[时间上限说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)解释了这一区别。

所有 Gymnasium 环境使用 `self.np_random`；重置一个环境不再改变其他环境或智能体使用的 NumPy 全局随机序列。训练与评估分开、多次运行的方法也与 [Stable-Baselines3 的评估说明](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm)一致。

## TEP 与工业边界

TEP 仪表盘仅评估已保存检查点。先用 `deep_rl_agent.train_sac` 训练并保存模型，再设置：

```python
from app import app
app.config["TEP_SAC_CHECKPOINT"] = "sac_tep_model.pt"
```

GET `/api/evaluate_tep_sac` 不训练策略；缺少检查点返回 409。安全裕度来自仿真实际温度、压力和液位到紧急停机边界的归一化距离。POST `/api/tep/optimize` 生成后续训练建议，返回 `recommendations_only` 和 `applied_count=0`，不会声称建议已经带来性能提升。

当前 TEP 模型是简化仿真，且包含测量延迟；单步观察不一定满足完整 Markov 性。其综合分数保留了历史教学评分权重，不是经校准的工业安全认证。没有完成 HVAC/TEP 的多种子性能、故障留出和真实设备验证。
