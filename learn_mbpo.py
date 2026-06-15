#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================
 学习版：基于模型的策略优化 (MBPO)
=============================================

📚 学习目标：
1. 理解基于模型的强化学习
2. 掌握动力学模型学习
3. 学会多步前向规划
4. 理解数据效率优势

🔧 技术要点：
- 动力学模型 (Dynamics Model)
- 前向模拟 (Rollout)
- 规划时域 (Planning Horizon)
- 折扣奖励 (Discounted Reward)

💡 适用场景：
- 样本获取成本高的场景
- 需要长期规划的任务
- 模型可学习的环境

🚀 运行方式：python learn_mbpo.py
"""

import numpy as np
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================
# 第1部分：MBPO智能体实现
# =============================================

class MBPOAgent:
    """
    基于模型的策略优化智能体
    
    核心思想：
    1. 学习环境动力学模型
    2. 基于模型进行多步规划
    3. 选择最大化长期奖励的动作
    
    架构说明：
    ┌─────────────────────────────────────┐
    │  真实环境                           │
    └────────────────────┬────────────────┘
                         │ 经验数据
                         ▼
    ┌─────────────────────────────────────┐
    │  动力学模型学习                      │
    │  P(s', r | s, a)                   │
    └────────────────────┬────────────────┘
                         │ 预测模型
                         ▼
    ┌─────────────────────────────────────┐
    │  多步前向规划                        │
    │  对每个动作模拟H步                  │
    │  计算累计奖励                        │
    └────────────────────┬────────────────┘
                         │ 选择最优动作
                         ▼
    ┌─────────────────────────────────────┐
    │  执行动作 → 获取新经验 → 更新模型    │
    └─────────────────────────────────────┘
    """
    
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化参数
        
        参数：
        - temp_min: 温度下限
        - temp_max: 温度上限
        - target_temp: 目标温度
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.target_temp = target_temp
        
        # 规划时域（向前看多少步）
        self.horizon = 5
        
        # 折扣因子（未来奖励的权重衰减）
        self.gamma = 0.99
        
        # 动力学模型参数（模拟学习到的模型）
        self.dynamics_model = {
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
        print(f"🔄 环境已重置，初始温度: {self.current_temp:.1f}°C")
    
    def predict_next_state(self, current_temp, action):
        """
        使用动力学模型预测下一状态
        
        参数：
        - current_temp: 当前温度
        - action: 执行的动作
        
        返回：
        - next_temp: 预测的下一时刻温度
        """
        delta = self.dynamics_model['natural_decay']
        
        if action == 'heat':
            delta += self.dynamics_model['heat_effect']
        elif action == 'cool':
            delta += self.dynamics_model['cool_effect']
        
        next_temp = current_temp + delta
        
        # 边界约束
        return max(self.temp_min, min(self.temp_max, next_temp))
    
    def compute_reward(self, temp):
        """
        计算奖励
        
        参数：
        - temp: 当前温度
        
        返回：
        - reward: 奖励值
        """
        error = abs(temp - self.target_temp)
        reward = 10.0 - error * 2
        
        # 边界惩罚
        if temp <= self.temp_min or temp >= self.temp_max:
            reward -= 50.0
        
        return reward
    
    def rollout(self, start_temp, action):
        """
        单动作前向模拟
        
        参数：
        - start_temp: 起始温度
        - action: 执行的动作（固定不变）
        
        返回：
        - total_reward: 累计折扣奖励
        - trajectory: 状态轨迹
        """
        temp = start_temp
        total_reward = 0
        trajectory = [temp]
        
        for t in range(self.horizon):
            # 使用模型预测下一个状态
            temp = self.predict_next_state(temp, action)
            trajectory.append(temp)
            
            # 计算奖励并累加
            reward = self.compute_reward(temp)
            total_reward += reward * (self.gamma ** t)
            
            # 如果到达边界，停止模拟
            if temp <= self.temp_min or temp >= self.temp_max:
                break
        
        return total_reward, trajectory
    
    def select_action(self):
        """
        选择最优动作：比较每个动作的长期奖励
        
        返回：
        - best_action: 最优动作
        - best_value: 最优动作的值
        - predictions: 各动作的预测轨迹
        """
        actions = ['heat', 'cool', 'idle']
        best_action = 'idle'
        best_value = float('-inf')
        predictions = {}
        
        for action in actions:
            value, trajectory = self.rollout(self.current_temp, action)
            predictions[action] = trajectory
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action, best_value, predictions
    
    def step(self, action):
        """执行动作，更新真实状态"""
        delta_temp = self.dynamics_model['natural_decay']
        
        if action == 'heat':
            delta_temp += self.dynamics_model['heat_effect']
        elif action == 'cool':
            delta_temp += self.dynamics_model['cool_effect']
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        
        reward = self.compute_reward(self.current_temp)
        done = self.step_count >= 100
        
        return reward, done

# =============================================
# 第2部分：可视化工具
# =============================================

def plot_results(results):
    """绘制完整结果"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, [25]*len(steps), label='目标温度', color='#10B981', linestyle='--')
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='上限')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('MBPO温度控制')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 预测值变化
    values = [r['predicted_value'] for r in results]
    ax2.plot(steps, values, label='预测值', color='#8B5CF6', linewidth=2)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('预测值')
    ax2.set_title('多步预测值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 奖励变化
    rewards = [r['reward'] for r in results]
    ax3.plot(steps, rewards, label='奖励', color='#10B981')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('奖励')
    ax3.set_title('真实奖励')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 动作分布
    action_counts = {'heat': 0, 'cool': 0, 'idle': 0}
    for r in results:
        action_counts[r['action']] += 1
    ax4.bar(action_counts.keys(), action_counts.values(), 
            color=['#EF4444', '#3B82F6', '#10B981'])
    ax4.set_xlabel('动作')
    ax4.set_ylabel('次数')
    ax4.set_title('动作分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mbpo_results.png')
    plt.close()

def plot_rollout_comparison(predictions, current_step):
    """绘制不同动作的前向预测对比"""
    plt.figure(figsize=(8, 4))
    
    colors = {'heat': '#EF4444', 'cool': '#3B82F6', 'idle': '#10B981'}
    
    for action, trajectory in predictions.items():
        steps = list(range(len(trajectory)))
        plt.plot(steps, trajectory, label=action, color=colors[action], 
                 linestyle='--' if action != 'heat' else '-', linewidth=2)
    
    plt.axhline(y=25, color='#8B5CF6', linestyle=':', label='目标')
    plt.axhline(y=20, color='#EF4444', linestyle=':')
    plt.axhline(y=35, color='#EF4444', linestyle=':')
    
    plt.xlabel('预测步数')
    plt.ylabel('温度 (°C)')
    plt.title(f'第{current_step}步 - 各动作前向预测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mbpo_rollout.png')
    plt.close()

# =============================================
# 第3部分：演示和学习练习
# =============================================

def run_demo():
    """完整演示流程"""
    print("=" * 70)
    print(" 📚 基于模型的策略优化 (MBPO) - 学习演示")
    print("=" * 70)
    print()
    print("🎯 学习目标：")
    print("  1. 理解动力学模型的作用")
    print("  2. 掌握多步前向规划")
    print("  3. 理解折扣因子的影响")
    print()
    print("📋 实验设置：")
    print(f"  - 温度范围: {20}°C ~ {35}°C")
    print(f"  - 目标温度: {25}°C")
    print(f"  - 规划时域 (Horizon): {5}步")
    print(f"  - 折扣因子 (γ): {0.99}")
    print("=" * 70)
    
    # 创建智能体
    agent = MBPOAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 设置初始条件
    print("\n🔬 测试场景：初始温度较低")
    agent.reset(initial_temp=22.0)
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = []
    
    while agent.step_count < 30:  # 缩短演示
        action, value, predictions = agent.select_action()
        reward, done = agent.step(action)
        
        results.append({
            'step': agent.step_count,
            'temperature': round(agent.current_temp, 2),
            'action': action,
            'predicted_value': round(value, 2),
            'reward': round(reward, 2)
        })
        
        # 每5步输出并展示预测对比
        if agent.step_count % 5 == 0:
            print(f"\n步骤 {agent.step_count}:")
            print(f"  当前温度: {agent.current_temp:.1f}°C")
            print(f"  选择动作: {action}")
            print(f"  预测值: {value:.2f}")
            print(f"  真实奖励: {reward:.2f}")
            print("  ---")
            for act, traj in predictions.items():
                print(f"  {act}: {[round(t, 1) for t in traj]}")
    
    # 分析结果
    print("\n" + "=" * 70)
    print("📊 实验结果分析")
    print("=" * 70)
    
    temps = [r['temperature'] for r in results]
    print(f"\n📈 温度统计:")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  最低温度: {min(temps):.2f}°C")
    print(f"  最高温度: {max(temps):.2f}°C")
    
    print(f"\n📈 预测准确性:")
    values = [r['predicted_value'] for r in results]
    rewards = [r['reward'] for r in results]
    print(f"  平均预测值: {np.mean(values):.2f}")
    print(f"  平均真实奖励: {np.mean(rewards):.2f}")
    
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
    print("  如果规划时域设置为10步，会有什么变化？")
    print("  提示：修改 self.horizon = 10")
    print()
    print("📝 问题2：")
    print("  如果折扣因子设置为0.9（更短视），会有什么变化？")
    print("  提示：修改 self.gamma = 0.9")
    print()
    print("📝 问题3：")
    print("  如果动力学模型不准确（如heat_effect设置错误），会发生什么？")
    print("  提示：修改 self.dynamics_model['heat_effect'] = 0.1")
    print()
    print("🔧 扩展练习：")
    print("  实现模型更新机制：根据真实经验在线更新动力学模型参数")
    print("  提示：使用线性回归或神经网络学习delta_temp与action的关系")
    print()

if __name__ == "__main__":
    run_demo()
    learning_exercises()
    print("✅ 学习演示完成！")