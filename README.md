# 工业温控强化学习仿真系统

基于 OpenAI Gym 和 PPO 算法的工业设备温度控制系统 Demo。

## 项目概述

本项目实现了一个极简的工业温控仿真环境，模拟工业设备温度控制场景：
- **温度范围**：20°C - 35°C（安全约束）
- **目标温度**：25°C
- **控制设备**：加热器、冷却器
- **算法**：PPO（Proximal Policy Optimization）

## 技术架构

```
RLDemo/
├── app.py                      # Flask Web 服务（前端后端通信）
├── templates/
│   └── index.html              # 可视化前端页面
├── industrial_temp_env.py       # Gymnasium 自定义环境
├── main.py                     # PPO 训练脚本
├── simple_demo.py              # 简化版规则控制演示
└── setup.py                   # 项目配置
```

## 核心技术栈

### 后端技术

| 技术 | 说明 |
|------|------|
| **Python 3.12** | 编程语言 |
| **Gymnasium** | OpenAI Gym 替代品，强化学习环境标准接口 |
| **Stable-Baselines3** | PPO、A2C 等强化学习算法实现 |
| **Flask** | 轻量级 Web 框架 |
| **NumPy** | 数值计算 |

### 前端技术

| 技术 | 说明 |
|------|------|
| **HTML5/CSS3** | 页面结构和样式 |
| **JavaScript** | 异步通信和动态交互 |
| **Chart.js** | 图表可视化 |
| **Fetch API** | 与后端 REST API 交互 |

## 开发流程

### 阶段一：环境定义（Gymnasium 自定义环境）

参考《AIAgent 强化学习实战》自定义工业环境部分，创建自定义 Gymnasium 环境：

```python
# industrial_temp_env.py
class IndustrialTempEnv(gym.Env):
    def __init__(self):
        # 定义动作空间：3个离散动作 (IDLE/HEAT/COOL)
        self.action_space = spaces.Discrete(3)
        
        # 定义观测空间：当前温度
        self.observation_space = spaces.Box(low=20, high=35)
        
        # 温度参数
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.target_temp = 25.0
```

**核心方法**：
- `reset()` - 初始化环境，随机设置初始温度
- `step(action)` - 执行动作，返回 (观测, 奖励, 完成标志, 信息)
- `render()` - 可视化当前状态

### 阶段二：奖励函数设计

```python
def step(self, action):
    # 温度变化逻辑
    delta_temp = -0.1  # 自然散热
    if action == 1:    # 加热
        delta_temp += 0.3
    elif action == 2:  # 冷却
        delta_temp -= 0.2
    
    # 奖励计算：温度误差越小，奖励越高
    temp_error = abs(current_temp - target_temp)
    reward = 10.0 - temp_error * 2
    
    # 安全约束惩罚
    if current_temp < temp_min or current_temp > temp_max:
        reward -= 50.0
```

### 阶段三：PPO 算法训练

使用 Stable-Baselines3 训练 PPO 模型：

```python
# main.py
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    n_steps=128,
    batch_size=32,
    gamma=0.98,
    learning_rate=5e-4,
    policy_kwargs=dict(net_arch=[32, 32])
)

model.learn(total_timesteps=50000)
model.save("temp_control_model")
```

**PPO 核心参数**：
- `n_steps`：每次更新前收集的样本数
- `batch_size`：每次梯度更新的样本数
- `gamma`：折扣因子（0.98 表示考虑未来奖励）
- `learning_rate`：学习率（5e-4 平衡收敛速度和稳定性）

### 阶段四：Web 可视化

使用 Flask 构建 REST API：

```python
# app.py
@app.route('/api/step', methods=['POST'])
def api_step():
    temp, action = sim.step()
    return jsonify({
        'temp': temp,
        'action': action,
        'heater': sim.heater_on,
        'cooler': sim.cooler_on
    })
```

前端通过 Fetch API 调用后端 API，实时更新页面：

```javascript
async function stepSimulation() {
    const response = await fetch('/api/step', { method: 'POST' });
    const data = await response.json();
    updateDashboard(data);
}
```

## 运行指南

### 方式一：Web 可视化（推荐）

```bash
# 安装依赖
pip install flask gymnasium stable-baselines3 numpy

# 启动服务
python app.py

# 浏览器访问
http://127.0.0.1:5000
```

### 方式二：命令行训练

```bash
python main.py
```

### 方式三：简化演示（无需训练）

```bash
python simple_demo.py
```

## 系统特性

### 安全约束
- 温度下限：20°C
- 温度上限：35°C
- 超出范围立即惩罚

### 智能控制
- 自动学习温度调节策略
- 加热器/冷却器智能切换
- 快速收敛到目标温度

### 实时监控
- 圆形仪表盘显示实时温度
- 温度变化曲线图
- 历史记录表格

## 强化学习核心概念

### 1. 智能体（Agent）
能够感知环境并采取行动的实体。本项目中，智能体学习何时加热、何时冷却。

### 2. 环境（Environment）
智能体与之交互的外部系统。本项目中，工业温控环境模拟真实温度变化。

### 3. 状态空间（State Space）
智能体能够观察的所有可能状态。本项目中，状态仅为当前温度。

### 4. 动作空间（Action Space）
智能体可以采取的所有动作。本项目中，3 个离散动作：
- 0：IDLE（保持不变）
- 1：HEAT（加热）
- 2：COOL（冷却）

### 5. 奖励函数（Reward Function）
指导智能体学习的信号。本项目中：
- 靠近目标温度 → 正奖励
- 远离目标温度 → 负奖励
- 超出安全范围 → 大幅负奖励

### 6. PPO 算法
近端策略优化（Proximal Policy Optimization）：
- 稳定可靠，无需大量超参数调优
- 适用于连续和离散动作空间
- 广泛应用的深度强化学习算法

## 项目扩展

### 可扩展方向

1. **多区域温控**：扩展为多个温度传感器和执行器
2. **时变目标**：目标温度随时间变化
3. **干扰因素**：加入外部温度扰动（模拟环境变化）
4. **真实硬件**：连接实际传感器和执行器
5. **PPO 参数调优**：调整网络结构、学习率等超参数

### 真实工业应用

工业温控是强化学习在工业界最常见的应用之一：
- 数据中心冷却系统
- 化学反应器温度控制
- 制造业热处理工艺
- HVAC 暖通空调系统

## 学习资源

- [Gymnasium 文档](https://gymnasium.farama.org/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
- 《AIAgent 强化学习实战》

## 许可证

MIT License

## 作者

基于 OpenAI Gym 搭建的工业温控仿真 Demo，参考《AIAgent 强化学习实战》实现。
