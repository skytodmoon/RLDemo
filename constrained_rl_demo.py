#!/usr/bin/env python3
"""
约束强化学习演示 Demo
Constrained Reinforcement Learning Demo

技术路线：
1. 使用拉格朗日乘子法处理约束优化问题
2. 目标函数：最大化奖励（温度接近目标）
3. 约束条件：温度不越界（20°C <= temp <= 35°C）
4. 通过调整拉格朗日乘子平衡目标和约束

工程过程：
1. 定义目标函数和约束函数
2. 初始化拉格朗日乘子
3. 迭代优化：同时更新策略和乘子
4. 验证约束满足情况
"""

import numpy as np
import matplotlib.pyplot as plt

class ConstrainedRLAgent:
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化约束强化学习智能体
        
        参数：
            temp_min: 温度下限约束
            temp_max: 温度上限约束
            target_temp: 目标温度
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.target_temp = target_temp
        
        # 拉格朗日乘子：用于权衡目标和约束
        self.lambda_lower = 0.1  # 下限约束乘子
        self.lambda_upper = 0.1  # 上限约束乘子
        
        # 约束违反惩罚系数
        self.constraint_penalty = 10.0
        
        # 学习率
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
        # 重置拉格朗日乘子
        self.lambda_lower = 0.1
        self.lambda_upper = 0.1
        
    def compute_cost(self, action):
        """
        计算动作的代价（用于优化）
        
        参数：
            action: 候选动作
        
        返回：
            total_cost: 综合代价（目标代价 + 约束代价）
        """
        # 预测执行动作后的温度
        delta_temp = -0.1
        if action == 'heat':
            delta_temp += 0.3
        elif action == 'cool':
            delta_temp -= 0.2
        next_temp = self.current_temp + delta_temp
        
        # 目标代价：与目标温度的误差
        target_error = abs(next_temp - self.target_temp)
        objective_cost = target_error
        
        # 约束代价：违反约束的程度
        lower_violation = max(0, self.temp_min - next_temp)
        upper_violation = max(0, next_temp - self.temp_max)
        constraint_cost = self.lambda_lower * lower_violation + self.lambda_upper * upper_violation
        
        # 综合代价
        total_cost = objective_cost + constraint_cost
        
        return total_cost, next_temp
    
    def select_action(self):
        """
        基于约束优化选择最优动作
        
        返回：
            best_action: 最优动作
            expected_temp: 预期温度
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
        
        参数：
            violation_lower: 下限约束违反程度
            violation_upper: 上限约束违反程度
        """
        # 增加违反方向的乘子
        self.lambda_lower = max(0, self.lambda_lower + self.lambda_learning_rate * violation_lower)
        self.lambda_upper = max(0, self.lambda_upper + self.lambda_learning_rate * violation_upper)
        
        # 乘子衰减（防止乘子过大）
        self.lambda_lower *= 0.995
        self.lambda_upper *= 0.995
    
    def step(self, action):
        """
        执行动作，更新环境状态
        
        参数：
            action: 要执行的动作
        
        返回：
            reward: 奖励值
            violation_lower: 下限约束违反
            violation_upper: 上限约束违反
        """
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
        
        # 计算奖励（考虑约束违反）
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        # 约束违反惩罚
        reward -= self.constraint_penalty * (violation_lower + violation_upper)
        
        return reward, violation_lower, violation_upper
    
    def run_policy(self):
        """运行完整的约束强化学习策略"""
        results = []
        
        while self.step_count < 100:
            # 选择动作
            action, expected_temp = self.select_action()
            
            # 执行动作
            reward, violation_lower, violation_upper = self.step(action)
            
            # 更新拉格朗日乘子
            self.update_lagrange_multipliers(violation_lower, violation_upper)
            
            # 记录结果
            results.append({
                'step': self.step_count,
                'temperature': round(self.current_temp, 2),
                'target': self.target_temp,
                'action': action,
                'reward': round(reward, 2),
                'lambda_lower': round(self.lambda_lower, 4),
                'lambda_upper': round(self.lambda_upper, 4),
                'violation_lower': round(violation_lower, 2),
                'violation_upper': round(violation_upper, 2)
            })
        
        return results

def plot_results(results):
    """可视化演示结果"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    targets = [r['target'] for r in results]
    lambda_lower = [r['lambda_lower'] for r in results]
    lambda_upper = [r['lambda_upper'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度变化曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, targets, label='目标温度', color='#10B981', linestyle='--', linewidth=2)
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='温度下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='温度上限')
    ax1.fill_between(steps, 20, 35, color='#FEE2E2', alpha=0.3)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('约束强化学习温度控制')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 拉格朗日乘子变化
    ax2.plot(steps, lambda_lower, label='λ_lower', color='#F59E0B')
    ax2.plot(steps, lambda_upper, label='λ_upper', color='#8B5CF6')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('拉格朗日乘子')
    ax2.set_title('拉格朗日乘子动态调整')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 奖励变化
    rewards = [r['reward'] for r in results]
    ax3.plot(steps, rewards, label='奖励', color='#10B981')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('奖励值')
    ax3.set_title('奖励变化曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 约束违反情况
    violations_lower = [r['violation_lower'] for r in results]
    violations_upper = [r['violation_upper'] for r in results]
    ax4.stackplot(steps, violations_lower, violations_upper, 
                  labels=['下限违反', '上限违反'], colors=['#EF4444', '#F97316'])
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('违反程度')
    ax4.set_title('约束违反情况')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """打印演示摘要"""
    print("=" * 60)
    print("约束强化学习演示摘要")
    print("=" * 60)
    
    temps = [r['temperature'] for r in results]
    rewards = [r['reward'] for r in results]
    violations_lower = [r['violation_lower'] for r in results]
    violations_upper = [r['violation_upper'] for r in results]
    
    print(f"\n📊 统计信息:")
    print(f"  总步数: {len(results)}")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  温度标准差: {np.std(temps):.2f}°C")
    print(f"  最大温度: {max(temps):.2f}°C")
    print(f"  最小温度: {min(temps):.2f}°C")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    
    print("\n🔒 约束违反统计:")
    print(f"  下限违反次数: {sum(1 for v in violations_lower if v > 0)}")
    print(f"  上限违反次数: {sum(1 for v in violations_upper if v > 0)}")
    print(f"  平均下限违反: {np.mean(violations_lower):.3f}")
    print(f"  平均上限违反: {np.mean(violations_upper):.3f}")
    
    print("\n📈 拉格朗日乘子最终值:")
    print(f"  λ_lower: {results[-1]['lambda_lower']:.4f}")
    print(f"  λ_upper: {results[-1]['lambda_upper']:.4f}")
    
    print("\n🎯 技术要点:")
    print("  1. 拉格朗日乘子法：权衡目标优化与约束满足")
    print("  2. 动态调整：乘子随约束违反程度自适应调整")
    print("  3. 约束代价：违反约束时增加动作代价")
    print("  4. 乘子衰减：防止乘子过大导致过度保守")

if __name__ == "__main__":
    print("🚀 约束强化学习演示")
    print("=" * 60)
    print("技术路线：拉格朗日乘子法约束优化")
    print("目标：最大化奖励（温度接近25°C）")
    print("约束：20°C <= 温度 <= 35°C")
    print("=" * 60)
    
    # 创建智能体
    agent = ConstrainedRLAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 重置环境（设置极端初始温度测试约束机制）
    agent.reset(initial_temp=18.0)  # 低于下限
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = agent.run_policy()
    
    # 输出结果
    print_summary(results)
    
    # 可视化
    print("\n📈 生成可视化图表...")
    plot_results(results)
    
    print("\n✅ 演示完成！")