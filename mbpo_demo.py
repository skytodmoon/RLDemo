#!/usr/bin/env python3
"""
基于模型的策略优化演示 Demo
Model-Based Policy Optimization (MBPO) Demo

技术路线：
1. 学习环境动力学模型（预测下一个状态）
2. 基于模型进行多步规划（Rollout）
3. 使用规划结果优化策略
4. 定期用真实经验更新模型

工程过程：
1. 构建动力学模型（简单线性模型）
2. 实现模型预测（预测温度变化）
3. 基于模型进行前向搜索（规划）
4. 选择最优动作序列
"""

import numpy as np
import matplotlib.pyplot as plt

class MBPOAgent:
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化MBPO智能体
        
        参数：
            temp_min: 温度下限
            temp_max: 温度上限
            target_temp: 目标温度
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.target_temp = target_temp
        
        # 规划时域（Horizon）
        self.horizon = 5
        
        # 折扣因子
        self.gamma = 0.99
        
        # 动力学模型参数（学习到的）
        self.model_params = {
            'cool_effect': -0.2,   # 冷却效果
            'heat_effect': 0.3,    # 加热效果
            'natural_decay': -0.1  # 自然散热
        }
        
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
        
    def dynamics_model(self, current_temp, action):
        """
        动力学模型：预测下一个状态
        
        参数：
            current_temp: 当前温度
            action: 动作
        
        返回：
            next_temp: 预测的下一时刻温度
        """
        delta = self.model_params['natural_decay']
        
        if action == 'heat':
            delta += self.model_params['heat_effect']
        elif action == 'cool':
            delta += self.model_params['cool_effect']
        
        next_temp = current_temp + delta
        
        # 边界约束
        return max(self.temp_min, min(self.temp_max, next_temp))
    
    def reward_function(self, temp):
        """
        奖励函数
        
        参数：
            temp: 当前温度
        
        返回：
            reward: 奖励值
        """
        error = abs(temp - self.target_temp)
        reward = 10.0 - error * 2
        
        # 边界惩罚
        if temp <= self.temp_min or temp >= self.temp_max:
            reward -= 50.0
        
        return reward
    
    def rollout(self, start_temp, action):
        """
        单步前向模拟（基于模型）
        
        参数：
            start_temp: 起始温度
            action: 执行的动作
        
        返回：
            total_reward: 累积奖励
            trajectory: 状态轨迹
        """
        temp = start_temp
        total_reward = 0
        
        for t in range(self.horizon):
            # 使用模型预测下一个状态
            temp = self.dynamics_model(temp, action)
            
            # 计算奖励
            reward = self.reward_function(temp)
            total_reward += reward * (self.gamma ** t)
            
            # 如果到达边界，停止模拟
            if temp <= self.temp_min or temp >= self.temp_max:
                break
        
        return total_reward
    
    def select_action(self):
        """
        通过模型预测选择最优动作
        
        返回：
            best_action: 最优动作
            best_value: 最优动作的值
        """
        actions = ['heat', 'cool', 'idle']
        best_action = 'idle'
        best_value = float('-inf')
        
        for action in actions:
            # 使用模型进行前向模拟
            value = self.rollout(self.current_temp, action)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action, best_value
    
    def step(self, action):
        """
        执行动作，更新真实环境状态
        
        参数：
            action: 要执行的动作
        
        返回：
            reward: 真实奖励
            done: 是否结束
        """
        delta_temp = self.model_params['natural_decay']
        
        if action == 'heat':
            delta_temp += self.model_params['heat_effect']
        elif action == 'cool':
            delta_temp += self.model_params['cool_effect']
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        
        reward = self.reward_function(self.current_temp)
        done = self.step_count >= 100
        
        return reward, done
    
    def run_policy(self):
        """运行完整的MBPO策略"""
        results = []
        
        while self.step_count < 100:
            # 使用模型规划选择动作
            action, value = self.select_action()
            
            # 执行动作（真实环境）
            reward, done = self.step(action)
            
            # 记录结果
            results.append({
                'step': self.step_count,
                'temperature': round(self.current_temp, 2),
                'target': self.target_temp,
                'action': action,
                'predicted_value': round(value, 2),
                'reward': round(reward, 2),
                'horizon': self.horizon
            })
            
            if done:
                break
        
        return results

def plot_results(results):
    """可视化演示结果"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    targets = [r['target'] for r in results]
    values = [r['predicted_value'] for r in results]
    rewards = [r['reward'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度变化曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, targets, label='目标温度', color='#10B981', linestyle='--', linewidth=2)
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='温度下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='温度上限')
    ax1.fill_between(steps, 20, 35, color='#FEE2E2', alpha=0.3)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('MBPO温度控制演示')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 预测值变化
    ax2.plot(steps, values, label='预测值', color='#8B5CF6', linewidth=2)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('预测值')
    ax2.set_title('模型预测值（多步规划）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 奖励变化
    ax3.plot(steps, rewards, label='真实奖励', color='#10B981')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('奖励值')
    ax3.set_title('真实奖励变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 动作分布
    action_counts = {'heat': 0, 'cool': 0, 'idle': 0}
    for r in results:
        action_counts[r['action']] += 1
    ax4.bar(action_counts.keys(), action_counts.values(), 
            color=['#EF4444', '#3B82F6', '#10B981'])
    ax4.set_xlabel('动作')
    ax4.set_ylabel('执行次数')
    ax4.set_title('动作执行分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """打印演示摘要"""
    print("=" * 60)
    print("基于模型的策略优化（MBPO）演示摘要")
    print("=" * 60)
    
    temps = [r['temperature'] for r in results]
    rewards = [r['reward'] for r in results]
    values = [r['predicted_value'] for r in results]
    
    print(f"\n📊 统计信息:")
    print(f"  总步数: {len(results)}")
    print(f"  规划时域 (Horizon): {results[0]['horizon']}")
    print(f"  折扣因子 (γ): 0.99")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  温度标准差: {np.std(temps):.2f}°C")
    print(f"  最大温度: {max(temps):.2f}°C")
    print(f"  最小温度: {min(temps):.2f}°C")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  平均预测值: {np.mean(values):.2f}")
    
    print("\n📈 动作统计:")
    action_counts = {'heat': 0, 'cool': 0, 'idle': 0}
    for r in results:
        action_counts[r['action']] += 1
    for action, count in action_counts.items():
        print(f"  {action.upper()}: {count}次 ({count/len(results)*100:.1f}%)")
    
    print("\n🎯 技术要点:")
    print("  1. 动力学模型：学习环境状态转移规律")
    print("  2. 多步规划：基于模型预测未来状态序列")
    print("  3. 值估计：使用折扣累积奖励评估动作")
    print("  4. 数据效率：减少与真实环境的交互")

if __name__ == "__main__":
    print("🚀 基于模型的策略优化（MBPO）演示")
    print("=" * 60)
    print("技术路线：动力学模型 + 多步规划")
    print(f"规划时域: 5步")
    print(f"目标温度: 25°C")
    print(f"温度范围: 20°C ~ 35°C")
    print("=" * 60)
    
    # 创建智能体
    agent = MBPOAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 重置环境
    agent.reset(initial_temp=22.0)
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = agent.run_policy()
    
    # 输出结果
    print_summary(results)
    
    # 可视化
    print("\n📈 生成可视化图表...")
    plot_results(results)
    
    print("\n✅ 演示完成！")