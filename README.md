# 工业智能控制仿真系统

基于 OpenAI Gym + PPO 强化学习和 PINNs 物理信息神经网络的工业控制仿真平台。

## 项目概述

本项目包含三个核心 Demo：

### 1. 工业温控强化学习 Demo
- **温度范围**：20°C - 35°C（安全约束）
- **目标温度**：25°C
- **控制设备**：加热器、冷却器
- **算法**：PPO（Proximal Policy Optimization）

### 2. PINNs 物理信息神经网络 Demo
- **求解问题**：一维热传导方程
- **方法**：结合物理机理与深度学习
- **特点**：无需大量标注数据，自动满足物理约束

### 3. Multi-Zone HVAC 多区域建筑能源管理 Demo 🆕
- **场景**：4区域建筑（办公室、服务器室、实验室、会议室）
- **变量**：温度、湿度、占用率（每区域3个，共12个状态变量）
- **耦合**：区域间热传导（热导矩阵）、外部天气干扰、占用率变化
- **算法**：MAPPO（Multi-Agent PPO，多智能体近端策略优化）
- **架构**：Centralized Training, Decentralized Execution (CTDE)

## 技术架构

```
RLDemo/
├── app.py                         # Flask Web 服务（含所有算法模拟器）
├── main.py                        # PPO 训练脚本（Stable-Baselines3）
├── simple_demo.py                 # 简化版规则控制演示
│
├── industrial_temp_env.py         # Gymnasium 自定义环境（基础版）
├── safe_rl_env.py                 # Gymnasium 自定义环境（安全层版）
│
├── constrained_rl.py              # CPO 约束策略优化（PyTorch NN）
├── model_based_rl.py              # MBPO 基于模型的策略优化（PyTorch NN）
├── hierarchical_rl.py             # 层级RL 选项架构（PyTorch NN）
├── rl_mpc_hybrid.py               # RL+MPC 混合控制（PyTorch NN）
│
├── building_hvac_env.py           # 🆕 多区域HVAC Gymnasium环境
├── multi_agent_ppo.py             # 🆕 MAPPO 多智能体PPO（PyTorch NN）
│
├── pinns_heat_eq.py               # PINNs 热传导方程求解器
│
├── safe_rl_demo.py                # 安全RL独立演示
├── constrained_rl_demo.py         # 约束RL独立演示
├── mbpo_demo.py                   # MBPO独立演示
├── hierarchical_rl_demo.py        # 层级RL独立演示
├── rl_mpc_hybrid_demo.py          # RL+MPC独立演示
│
├── learn_safe_rl.py               # 学习版：安全RL
├── learn_constrained_rl.py        # 学习版：约束RL
├── learn_mbpo.py                  # 学习版：MBPO
├── learn_hierarchical_rl.py       # 学习版：层级RL
├── learn_rl_mpc.py                # 学习版：RL+MPC
│
├── templates/
│   ├── index.html                 # 温控 RL 主页（算法选择器）
│   ├── pinns.html                 # PINNs 可视化页面
│   ├── safe_rl.html               # 安全RL页面
│   ├── constrained_rl.html        # 约束RL页面
│   ├── mbpo.html                  # MBPO页面
│   ├── hierarchical_rl.html       # 层级RL页面
│   ├── rl_mpc.html                # RL+MPC页面
│   └── building_hvac.html         # 🆕 多区域HVAC MAPPO页面
│
├── setup.py                       # 项目配置
└── start.sh                       # 启动脚本
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

# 浏览器访问 - 多区域 HVAC MAPPO Demo
http://127.0.0.1:5000/building-hvac
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

## 进阶强化学习算法

本项目实现了多种先进的强化学习算法，可通过前端页面切换对比：

### 1. 安全强化学习 (Safe RL)

**核心思想**：在标准 RL 基础上增加安全层，当系统接近安全边界时自动阻止危险动作。

```python
class SafeRLSimulator(BaseSimulator):
    def get_action(self):
        # 安全约束：接近边界时禁止危险动作
        if self.current_temp <= self.temp_min + 1.0:
            return 'heat' if temp_error < 0 else 'idle'
        if self.current_temp >= self.temp_max - 1.0:
            return 'cool' if temp_error > 0 else 'idle'
```

**特点**：双层架构保障系统安全，防止温度越界。

### 2. 约束强化学习 (Constrained RL)

**核心思想**：使用拉格朗日乘子法处理约束优化问题，在最大化奖励的同时确保约束满足。

```python
class ConstrainedRLSimulator(BaseSimulator):
    def get_action(self):
        distance_to_boundary = min(
            self.current_temp - self.temp_min,
            self.temp_max - self.current_temp
        )
        if distance_to_boundary < 2.0:
            # 收紧控制策略以满足约束
            ...
```

**典型算法**：CPO（约束策略优化）、TRPO 的约束版本。

### 3. 基于模型的策略优化 (MBPO)

**核心思想**：先学习环境动力学模型，再基于模型进行规划和策略优化。

```python
class MBPO_Simulator(BaseSimulator):
    def predict_next_temp(self, temp, action):
        # 学习到的动力学模型
        delta = -0.1
        if action == 'heat':
            delta += 0.3
        return max(self.temp_min, min(self.temp_max, temp + delta))
    
    def get_action(self):
        # 基于模型进行多步规划
        for action in ['heat', 'cool', 'idle']:
            temp = self.current_temp
            total_reward = 0
            for _ in range(self.horizon):
                temp = self.predict_next_temp(temp, action)
                total_reward += reward
```

**优势**：数据效率高，适合样本获取成本高的场景。

### 4. 层级强化学习 (Hierarchical RL)

**核心思想**：将复杂任务分解为高层选项（Options）和低层动作。

```python
class HierarchicalRLSimulator(BaseSimulator):
    def __init__(self):
        self.current_goal = 'maintain'  # 高层目标
    
    def get_action(self):
        # 高层策略：选择目标
        if abs(temp_error) > 3.0:
            self.current_goal = 'approach'
        
        # 低层策略：执行动作
        if self.current_goal == 'approach':
            return 'heat' if temp_error < 0 else 'cool'
```

**优势**：支持长期规划，可处理时序扩展问题。

### 5. RL+MPC 混合控制

**核心思想**：结合强化学习的学习能力和模型预测控制（MPC）的优化能力。

```python
class RLMPCSimulator(BaseSimulator):
    def get_action(self):
        rl_action = self.get_rl_action()      # RL 策略
        mpc_action = self.get_mpc_action()    # MPC 优化
        # 加权融合
        if np.random.random() < self.rl_weight:
            return rl_action
        else:
            return mpc_action
```

**优势**：兼顾学习能力和在线优化能力。

### 6. 多智能体 PPO (MAPPO) — 多区域 HVAC 控制 🆕

**核心思想**：多个智能体协作控制建筑的4个区域，每个区域有独立的策略网络，共享一个中央评论家网络评估全局状态。

```python
# building_hvac_env.py - 多区域HVAC环境
class BuildingHVACEnv(gym.Env):
    ZONE_NAMES = ["Office", "Server", "Lab", "Conference"]
    # 4区域 × 3变量（温度/湿度/占用率）= 12个状态变量
    # 4区域 × 4控制（加热/冷却/加湿/除湿）= 16维连续动作
    # 区域间热传导耦合矩阵
    COUPLING = np.array([
        [0.0, 0.8, 0.3, 0.6],  # Office <-> Server, Lab, Conference
        [0.8, 0.0, 0.5, 0.1],  # Server <-> Office, Lab
        [0.3, 0.5, 0.0, 0.1],  # Lab <-> Office, Server
        [0.6, 0.1, 0.1, 0.0],  # Conference <-> Office
    ])

# multi_agent_ppo.py - MAPPO算法
class MultiAgentPPO:
    # 集中训练、分散执行 (CTDE)
    actors = [DecentralizedActor() for _ in range(4)]  # 每区域独立策略
    critic = CentralizedCritic(state_dim=15)            # 全局共享评论家
```

**环境特点**：
- **多变量**：温度、湿度、占用率（每区域3个变量，共12维状态）
- **多耦合**：区域间热传导（热导矩阵）、天气干扰、占用率变化
- **连续动作**：16维连续动作空间（4控制 × 4区域，Beta分布）
- **多目标奖励**：舒适度（占用率加权）+ 能耗惩罚 + 预算约束
- **外部干扰**：室外温度（昼夜循环）、太阳辐射、占用率时间表

**算法特点**：
- **CTDE 架构**：集中训练（评论家看全局）+ 分散执行（演员只看局部）
- **Beta 分布**：连续动作使用 Beta(α, β) 分布，天然限制在 [0,1]
- **GAE 优势估计**：Generalized Advantage Estimation 稳定训练
- **PPO 裁剪目标**：ε=0.2 裁剪比率，防止策略更新过大
- **熵正则化**：鼓励探索，防止过早收敛

**运行方式**：
```bash
# 环境测试
python building_hvac_env.py

# MAPPO训练测试
python multi_agent_ppo.py
```

**关键特性**：
- 4区域建筑仿真（办公室/服务器室/实验室/会议室）
- 区域间热传导耦合
- 昼夜天气循环与太阳辐射
- 能量预算约束
- 每区域独立HVAC控制（加热/冷却/加湿/除湿）

---

## 独立演示 Demo

本项目为每种进阶算法提供了独立的演示脚本，可以单独运行查看效果：

### 1. 安全强化学习 Demo (`safe_rl_demo.py`)

**技术路线**：
```
基础策略 → 安全层过滤 → 执行动作
    ↓            ↓
  选择动作    阻止危险动作
```

**工程过程**：
1. 定义安全边界（20°C ~ 35°C）和安全裕度（2°C）
2. 实现基础控制策略（基于误差的比例控制）
3. 添加安全层：当温度接近边界时过滤危险动作
4. 实现主动保护机制：距离边界1°C时强制反向动作

**运行方式**：
```bash
python safe_rl_demo.py
```

**关键特性**：
- 双层架构：基础策略 + 安全层
- 安全裕度自适应调整
- 安全干预记录和统计

---

### 2. 约束强化学习 Demo (`constrained_rl_demo.py`)

**技术路线**：
```
目标优化 + 约束惩罚 → 拉格朗日乘子动态调整
    ↓
  最优动作选择
```

**工程过程**：
1. 定义目标函数（温度接近目标）和约束函数（温度不越界）
2. 初始化拉格朗日乘子（λ_lower, λ_upper）
3. 综合代价 = 目标代价 + λ×约束违反
4. 迭代优化：更新动作选择和乘子值

**运行方式**：
```bash
python constrained_rl_demo.py
```

**关键特性**：
- 拉格朗日乘子法约束优化
- 乘子动态自适应调整
- 约束违反惩罚机制
- 乘子衰减防止过度保守

---

### 3. MBPO 基于模型的策略优化 Demo (`mbpo_demo.py`)

**技术路线**：
```
学习动力学模型 → 多步前向模拟 → 选择最优动作
     ↓
  真实环境执行
```

**工程过程**：
1. 构建动力学模型（温度变化预测）
2. 实现模型预测函数（给定当前温度和动作，预测下一时刻温度）
3. 基于模型进行多步规划（Horizon=5）
4. 使用折扣奖励评估每个动作的长期影响
5. 选择累积奖励最大的动作

**运行方式**：
```bash
python mbpo_demo.py
```

**关键特性**：
- 环境动力学模型学习
- 多步前向规划（Rollout）
- 折扣奖励值估计
- 高数据效率

---

### 4. 层级强化学习 Demo (`hierarchical_rl_demo.py`)

**技术路线**：
```
高层策略（选择选项）→ 低层策略（执行动作）
     ↓                        ↓
  approach/maintain      heat/cool/idle
  /safe_guard
```

**工程过程**：
1. 定义选项空间：approach（接近目标）、maintain（维持温度）、safe_guard（安全保护）
2. 实现高层策略：根据状态选择合适的选项
3. 实现低层策略：根据当前选项执行具体动作
4. 实现选项终止机制：决定何时切换选项

**运行方式**：
```bash
python hierarchical_rl_demo.py
```

**关键特性**：
- 两层决策架构
- 选项切换机制
- 最小选项持续时间保证
- 支持长期规划

---

### 5. RL+MPC 混合控制 Demo (`rl_mpc_hybrid_demo.py`)

**技术路线**：
```
RL策略 ←─── 加权融合 ───→ MPC控制
    ↓                        ↓
  快速决策              精确优化
```

**工程过程**：
1. 实现 RL 策略（学习到的近似最优策略）
2. 实现 MPC 控制器（基于模型的在线优化）
3. 设计融合机制（加权随机选择）
4. 实现自适应权重：根据状态不确定性动态调整

**运行方式**：
```bash
python rl_mpc_hybrid_demo.py
```

**关键特性**：
- 双策略融合
- 自适应权重调整
- 兼顾学习能力和优化精度
- 不确定性感知

---

## Web前端演示

每个技术案例都提供了完整的前端可视化页面，可以直接在浏览器中运行和交互：

### 访问地址

```bash
# 温控RL主页
http://127.0.0.1:5000

# 各算法演示页面
http://127.0.0.1:5000/safe-rl         # 安全强化学习
http://127.0.0.1:5000/constrained-rl  # 约束强化学习
http://127.0.0.1:5000/mbpo            # MBPO
http://127.0.0.1:5000/hierarchical-rl # 层级强化学习
http://127.0.0.1:5000/rl-mpc          # RL+MPC混合控制
http://127.0.0.1:5000/building-hvac   # 🆕 多区域HVAC MAPPO

# PINNs演示
http://127.0.0.1:5000/pinns
```

### 前端特性

| 功能 | 说明 |
|------|------|
| **实时监控** | 圆形仪表盘显示当前温度 |
| **数据可视化** | Chart.js图表展示温度变化曲线 |
| **日志记录** | 显示运行日志和安全干预记录 |
| **统计信息** | 实时计算平均奖励、干预率等指标 |
| **交互控制** | 开始/停止/重置模拟按钮 |

### 页面功能

每个前端页面都包含：

- 🌡️ **实时监控面板**：圆形温度仪表盘、安全裕度显示
- 📊 **统计面板**：步数、奖励、干预次数等关键指标
- 📈 **图表展示**：温度历史曲线、乘子变化、预测轨迹等
- 📝 **运行日志**：实时更新决策过程和干预记录
- 🔧 **技术说明**：每个页面都包含详细的技术原理说明

---

## 学习版演示脚本

每个技术案例都提供了学习版脚本（`learn_*.py`），包含详细的代码注释和学习练习：

### 学习版脚本列表

| 脚本 | 技术主题 | 学习目标 |
|------|----------|----------|
| `learn_safe_rl.py` | 安全强化学习 | 双层架构、安全裕度、安全层过滤 |
| `learn_constrained_rl.py` | 约束强化学习 | 拉格朗日乘子、约束优化、动态调整 |
| `learn_mbpo.py` | MBPO | 动力学模型、多步规划、前向模拟 |
| `learn_hierarchical_rl.py` | 层级RL | 选项架构、时间抽象、任务分解 |
| `learn_rl_mpc.py` | RL+MPC | 混合架构、自适应权重、不确定性估计 |

### 学习版特色

```
┌─────────────────────────────────────┐
│  📚 学习目标                        │
│  列出3-4个核心学习点                │
├─────────────────────────────────────┤
│  🔧 技术要点                        │
│  关键概念和术语                     │
├─────────────────────────────────────┤
│  🏗️ 架构说明                        │
│  带注释的架构图                     │
├─────────────────────────────────────┤
│  ▶️ 演示流程                        │
│  带输出的运行演示                   │
├─────────────────────────────────────┤
│  💡 思考与练习                      │
│  3个问题 + 1个扩展练习              │
└─────────────────────────────────────┘
```

### 运行方式

```bash
# 查看学习版脚本列表
./start.sh -l

# 运行学习版演示
python learn_safe_rl.py
python learn_constrained_rl.py
python learn_mbpo.py
python learn_hierarchical_rl.py
python learn_rl_mpc.py
```

---

## 算法对比

| 算法 | 核心特点 | 适用场景 | Demo文件 | 前端 |
|------|----------|----------|----------|------|
| **规则控制** | 基于阈值的简单策略 | 简单场景，无需训练 | `simple_demo.py` | ✅ |
| **安全 RL** | 双层安全架构 | 需要保证安全约束的场景 | `safe_rl_demo.py` | ✅ |
| **约束 RL** | 拉格朗日乘子优化 | 多约束优化问题 | `constrained_rl_demo.py` | ✅ |
| **MBPO** | 基于模型的规划 | 数据效率要求高 | `mbpo_demo.py` | ✅ |
| **层级 RL** | 分层任务分解 | 长期规划任务 | `hierarchical_rl_demo.py` | ✅ |
| **RL+MPC** | 混合控制策略 | 需要在线优化的场景 | `rl_mpc_hybrid_demo.py` | ✅ |
| **MAPPO** 🆕 | 多智能体协作 + CTDE | 多区域耦合系统 | `multi_agent_ppo.py` | ✅ |

## 项目扩展

### RL 温控扩展方向

1. **多区域温控** ✅ 已实现：Multi-Zone HVAC MAPPO Demo（4区域、12状态变量、16维动作）
2. **时变目标**：目标温度随时间变化
3. **干扰因素** ✅ 已实现：天气干扰（昼夜温度循环、太阳辐射、占用率变化）
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
