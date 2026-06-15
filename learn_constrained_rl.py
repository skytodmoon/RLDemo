#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================
 学习版：约束强化学习 (Constrained RL)
=============================================

📚 学习目标：
1. 理解约束优化问题
2. 掌握拉格朗日乘子法
3. 学会动态调整乘子
4. 理解约束满足与目标优化的权衡

🔧 技术要点：
- 拉格朗日乘子 (Lagrange Multiplier)
- 约束违反 (Constraint Violation)
- 代价函数 (Cost Function)
- 乘子更新规则

💡 适用场景：
- 资源约束问题
- 安全约束控制
- 多目标优化

🚀 运行方式：python learn_constrained_rl.py
"""

import numpy as np
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================
# 第1部分：约束强化学习智能体实现
# =============================================

class ConstrainedRLAgent:
    """
    约束强化学习智能体
    
    核心思想：
    在最大化奖励的同时，确保满足约束条件
    
    数学模型：
    max J = E[sum(r_t)]      (目标函数：最大化奖励)
    s.t. E[sum(c_t)] <= d     (约束条件：累计约束违反 <= d)
    
    使用拉格朗日乘子法：
    L = J - λ*(E[sum(c_t)] - d)
    
    架构说明：
    ┌─────────────────────────────────────┐
    │  状态 (温度)                        │
    └────────────────────┬────────────────┘
                         ▼
    ┌─────────────────────────────────────┐
    │  综合代价 = 目标代价 + λ×约束代价   │
    │  (Cost = Objective + λ×Constraint) │
    └────────────────────┬────────────────┘
                         ▼
    ┌─────────────────────────────────────┐
    │  选择最小化综合代价的动作            │
    └────────────────────┬────────────────┘
                         ▼
    ┌─────────────────────────────────────┐
    │  执行动作 → 观察约束违反 → 更新λ    │
    └─────────────────────────────────────┘
    """
    
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化参数
        
        参数：
        - temp_min: 温度下限约束
        - temp_max: 温度上限约束
        - target_temp: 目标温度
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.target_temp = target_temp
        
        # 拉格朗日乘子：权衡目标和约束
        self.lambda_lower = 0.1  # 下限约束乘子
        self.lambda_upper = 0.1  # 上限约束乘子
        
        # 约束违反惩罚系数
        self.constraint_penalty = 10.0
        
        # 乘子学习率
        self.lambda_learning_rate = 0.01
        
        # 当前状态
        self.current_temp = None
        self.step_count = 0
    
    def reset(self, initial_temp=None):
        """重置环境"""
        if initial_temp is None:
            self.current_temp = np.random.uniform(22, 28)
        else:
            self.current_temp = initial_temp
        self.step_count = 0
        # 重置乘子
        self.lambda_lower = 0.1
        self.lambda_upper = 0.1
        print(f"🔄 环境已重置，初始温度: {self.current_temp:.1f}°C")
    
    def compute_cost(self, action):
        """
        计算动作的综合代价
        
        代价组成：
        1. 目标代价：与目标温度的误差
        2. 约束代价：违反约束的预期程度 × 拉格朗日乘子
        
        参数：
        - action: 候选动作
        
        返回：
        - total_cost: 综合代价
        - next_temp: 预测的下一时刻温度
        """
        # 预测执行动作后的温度
        delta_temp = -0.1
        if action == 'heat':
            delta_temp += 0.3
        elif action == 'cool':
            delta_temp -= 0.2
        next_temp = self.current_temp + delta_temp
        
        # 目标代价：温度误差
        target_error = abs(next_temp - self.target_temp)
        objective_cost = target_error
        
        # 约束代价：违反约束的程度 × 乘子
        lower_violation = max(0, self.temp_min - next_temp)
        upper_violation = max(0, next_temp - self.temp_max)
        constraint_cost = self.lambda_lower * lower_violation + self.lambda_upper * upper_violation
        
        # 综合代价
        total_cost = objective_cost + constraint_cost
        
        return total_cost, next_temp
    
    def select_action(self):
        """
        选择最优动作：最小化综合代价
        
        返回：
        - best_action: 最优动作
        - expected_temp: 预期温度
        """
        actions = ['heat', 'cool', 'idle']
        best_action = 'idle'
        min_cost = float('inf')
        expected_temp = self.current_temp
        
        for action in actions:
            cost, next_temp = self.compute_cost(action)
            if cost < min_cost:
                min_cost = cost
                best_action = action
                expected_temp = next_temp
        
        return best_action, expected_temp
    
    def update_lagrange_multipliers(self, violation_lower, violation_upper):
        """
        更新拉格朗日乘子
        
        更新规则：
        λ_{t+1} = max(0, λ_t + α × violation_t)
        
        参数：
        - violation_lower: 下限约束违反程度
        - violation_upper: 上限约束违反程度
        """
        # 增加违反方向的乘子
        self.lambda_lower = max(0, self.lambda_lower + self.lambda_learning_rate * violation_lower)
        self.lambda_upper = max(0, self.lambda_upper + self.lambda_learning_rate * violation_upper)
        
        # 乘子衰减（防止乘子过大）
        self.lambda_lower *= 0.995
        self.lambda_upper *= 0.995
    
    def step(self, action):
        """执行动作，更新状态"""
        delta_temp = -0.1
        
        if action == 'heat':
            delta_temp += 0.3
        elif action == 'cool':
            delta_temp -= 0.2
        
        # 原始温度（未约束）
        raw_temp = self.current_temp + delta_temp
        
        # 约束后的温度
        self.current_temp = max(self.temp_min, min(self.temp_max, raw_temp))
        self.step_count += 1
        
        # 计算约束违反程度
        violation_lower = max(0, self.temp_min - raw_temp)
        violation_upper = max(0, raw_temp - self.temp_max)
        
        # 计算奖励
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        # 约束违反惩罚
        reward -= self.constraint_penalty * (violation_lower + violation_upper)
        
        return reward, violation_lower, violation_upper

# =============================================
# 第2部分：可视化工具
# =============================================

def plot_results(results):
    """绘制完整结果图"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    lambda_lower = [r['lambda_lower'] for r in results]
    lambda_upper = [r['lambda_upper'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, [25]*len(steps), label='目标温度', color='#10B981', linestyle='--')
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='上限')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('温度控制效果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 拉格朗日乘子变化
    ax2.plot(steps, lambda_lower, label='λ_lower', color='#F59E0B')
    ax2.plot(steps, lambda_upper, label='λ_upper', color='#8B5CF6')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('拉格朗日乘子')
    ax2.set_title('乘子动态调整')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 约束违反
    violations_lower = [r['violation_lower'] for r in results]
    violations_upper = [r['violation_upper'] for r in results]
    ax3.stackplot(steps, violations_lower, violations_upper, 
                  labels=['下限违反', '上限违反'], colors=['#EF4444', '#F97316'])
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('违反程度')
    ax3.set_title('约束违反情况')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 奖励变化
    rewards = [r['reward'] for r in results]
    ax4.plot(steps, rewards, label='奖励', color='#10B981')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('奖励')
    ax4.set_title('奖励变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_rl_results.png')
    plt.close()

# =============================================
# 第3部分：演示和学习练习
# =============================================

def run_demo():
    """完整演示流程"""
    print("=" * 70)
    print(" 📚 约束强化学习 - 学习演示")
    print("=" * 70)
    print()
    print("🎯 学习目标：")
    print("  1. 理解拉格朗日乘子法")
    print("  2. 观察乘子如何动态调整")
    print("  3. 理解约束与目标的权衡")
    print()
    print("📋 实验设置：")
    print(f"  - 温度范围: {20}°C ~ {35}°C")
    print(f"  - 目标温度: {25}°C")
    print(f"  - 初始乘子: λ_lower={0.1}, λ_upper={0.1}")
    print("=" * 70)
    
    # 创建智能体
    agent = ConstrainedRLAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 设置极端初始条件
    print("\n🔬 测试场景：初始温度低于下限")
    agent.reset(initial_temp=18.0)
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = []
    
    while agent.step_count < 50:
        action, expected_temp = agent.select_action()
        reward, violation_lower, violation_upper = agent.step(action)
        
        # 更新乘子
        agent.update_lagrange_multipliers(violation_lower, violation_upper)
        
        results.append({
            'step': agent.step_count,
            'temperature': round(agent.current_temp, 2),
            'action': action,
            'reward': round(reward, 2),
            'lambda_lower': round(agent.lambda_lower, 4),
            'lambda_upper': round(agent.lambda_upper, 4),
            'violation_lower': round(violation_lower, 2),
            'violation_upper': round(violation_upper, 2)
        })
        
        # 每5步输出
        if agent.step_count % 5 == 0:
            print(f"\n步骤 {agent.step_count}:")
            print(f"  温度: {agent.current_temp:.1f}°C")
            print(f"  动作: {action}")
            print(f"  乘子: λ_lower={agent.lambda_lower:.3f}, λ_upper={agent.lambda_upper:.3f}")
            print(f"  约束违反: 下={violation_lower:.2f}, 上={violation_upper:.2f}")
    
    # 分析结果
    print("\n" + "=" * 70)
    print("📊 实验结果分析")
    print("=" * 70)
    
    temps = [r['temperature'] for r in results]
    print(f"\n📈 温度统计:")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  最低温度: {min(temps):.2f}°C")
    print(f"  最高温度: {max(temps):.2f}°C")
    
    print(f"\n📈 乘子变化:")
    print(f"  λ_lower: 初始={0.1} → 最终={results[-1]['lambda_lower']:.4f}")
    print(f"  λ_upper: 初始={0.1} → 最终={results[-1]['lambda_upper']:.4f}")
    
    print(f"\n✅ 约束满足情况:")
    total_violation = sum(r['violation_lower'] + r['violation_upper'] for r in results)
    print(f"  累计约束违反: {total_violation:.2f}")
    
    # 可视化
    print("\n📉 生成可视化图表...")
    plot_results(results)

def learning_exercises():
    """学习练习"""
    print("\n" + "=" * 70)
    print(" 💡 思考与练习")
    print("=" * 70)
    print()
    print("📝 问题1：")
    print("  如果初始乘子设置为0，会发生什么？")
    print("  提示：修改 self.lambda_lower = 0")
    print()
    print("📝 问题2：")
    print("  如果乘子学习率增大到0.1，会有什么变化？")
    print("  提示：修改 self.lambda_learning_rate = 0.1")
    print()
    print("📝 问题3：")
    print("  为什么需要乘子衰减（*=0.995）？")
    print("  提示：尝试注释掉衰减代码，观察结果")
    print()
    print("🔧 扩展练习：")
    print("  添加新的约束：温度变化速率不能超过0.5°C/步")
    print("  需要：新增乘子λ_rate，定义违反程度，更新代价函数")
    print()

if __name__ == "__main__":
    run_demo()
    learning_exercises()
    print("✅ 学习演示完成！")