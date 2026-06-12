# 工业智能控制仿真系统

基于 OpenAI Gym + PPO 强化学习和 PINNs 物理信息神经网络的工业控制仿真平台。

## 项目概述

本项目包含两个核心 Demo：

### 1. 工业温控强化学习 Demo
- **温度范围**：20°C - 35°C（安全约束）
- **目标温度**：25°C
- **控制设备**：加热器、冷却器
- **算法**：PPO（Proximal Policy Optimization）

### 2. PINNs 物理信息神经网络 Demo
- **求解问题**：一维热传导方程
- **方法**：结合物理机理与深度学习
- **特点**：无需大量标注数据，自动满足物理约束

## 技术架构

```
RLDemo/
├── app.py                      # Flask Web 服务
├── templates/
│   ├── index.html              # 温控 RL 可视化页面
│   └── pinns.html              # PINNs 可视化页面
├── industrial_temp_env.py       # Gymnasium 自定义环境
├── main.py                     # PPO 训练脚本
├── simple_demo.py              # 简化版规则控制演示
├── pinns_heat_eq.py            # PINNs 热传导方程求解器
└── setup.py                   # 项目配置
```

## 核心技术栈

### 后端技术

| 技术 | 说明 |
|------|------|
| **Python 3.12** | 编程语言 |
| **PyTorch** | 深度学习框架（PINNs 自动微分） |
| **Gymnasium** | OpenAI Gym 替代品，强化学习环境标准接口 |
| **Stable-Baselines3** | PPO、A2C 等强化学习算法实现 |
| **Flask** | 轻量级 Web 框架 |
| **NumPy** | 数值计算 |
| **Matplotlib** | 可视化工具 |

### 前端技术

| 技术 | 说明 |
|------|------|
| **HTML5/CSS3** | 页面结构和样式 |
| **JavaScript** | 异步通信和动态交互 |
| **Canvas API** | 图表可视化（PINNs） |
| **Chart.js** | 图表可视化（RL） |
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

## 开发流程

### 阶段五：PINNs 物理信息神经网络

物理信息神经网络（Physics-Informed Neural Networks）是一种将物理定律融入深度学习的方法：

```python
# pinns_heat_eq.py
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
    
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
```

**热传导方程求解**：
- 方程：∂u/∂t = α ∂²u/∂x²
- 边界条件：u(0,t) = 0, u(1,t) = 0  
- 初始条件：u(x,0) = sin(πx)

**损失函数设计**：
```python
def loss_fn(self, x_pde, t_pde, x_bc, t_bc, x_ic, t_ic):
    # PDE 损失：满足偏微分方程
    pde_loss = torch.mean((du_dt - alpha * d2u_dx2)**2)
    
    # 边界条件损失
    bc_loss = torch.mean(u_bc**2)
    
    # 初始条件损失
    ic_loss = torch.mean((u_ic - sin(pi*x_ic))**2)
    
    return pde_loss + bc_loss + ic_loss
```

**自动微分**：通过 PyTorch 的自动微分机制计算高阶导数，无需手动推导。

## 运行指南

### 方式一：Web 可视化（推荐）

```bash
# 安装依赖
pip install flask gymnasium stable-baselines3 numpy torch matplotlib

# 启动服务
python app.py

# 浏览器访问 - 温控 RL Demo
http://127.0.0.1:5000

# 浏览器访问 - PINNs Demo
http://127.0.0.1:5000/pinns
```

### 方式二：命令行训练（RL）

```bash
python main.py
```

### 方式三：简化演示（无需训练）

```bash
python simple_demo.py
```

### 方式四：PINNs 独立运行

```bash
python pinns_heat_eq.py
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

## PINNs 核心概念

### 1. 物理信息神经网络
将物理定律作为正则化项融入神经网络训练，使模型在学习数据的同时满足物理约束。

### 2. 自动微分（Automatic Differentiation）
通过深度学习框架（PyTorch）自动计算高阶导数，无需手动推导复杂的微分公式。

### 3. 损失函数组成
- **PDE 损失**：确保解满足偏微分方程
- **边界条件损失**：确保解满足边界约束
- **初始条件损失**：确保解满足初始状态

### 4. PINNs 优势
- **数据高效**：无需大量标注数据
- **物理一致性**：自动满足物理定律
- **泛化能力强**：可在未训练区域进行预测

### 5. 典型应用场景
- 流体力学仿真
- 热传导分析
- 结构力学计算
- 化学反应动力学

## 项目扩展

### RL 温控扩展方向

1. **多区域温控**：扩展为多个温度传感器和执行器
2. **时变目标**：目标温度随时间变化
3. **干扰因素**：加入外部温度扰动（模拟环境变化）
4. **真实硬件**：连接实际传感器和执行器
5. **PPO 参数调优**：调整网络结构、学习率等超参数

### PINNs 扩展方向

1. **更高维方程**：求解二维/三维热传导方程
2. **复杂边界条件**：非齐次边界、时变边界
3. **非线性方程**：考虑温度依赖的热传导系数
4. **多物理场耦合**：热-结构耦合、流-热耦合
5. **数据融合**：结合实验数据和物理模型

### 真实工业应用

**强化学习温控**：
- 数据中心冷却系统
- 化学反应器温度控制
- 制造业热处理工艺
- HVAC 暖通空调系统

**PINNs 仿真**：
- 航空航天热分析
- 半导体器件热设计
- 电池热管理
- 建筑能耗模拟

## 学习资源

### 强化学习
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
- 《AIAgent 强化学习实战》

### PINNs
- [PINNs 原始论文](https://arxiv.org/abs/1711.10561)
- [PyTorch 自动微分教程](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl)
- 《Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations》

## 许可证

MIT License

## 作者

基于 OpenAI Gym 搭建的工业温控仿真 Demo，参考《AIAgent 强化学习实战》实现。
