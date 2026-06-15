#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================
 学习版：安全强化学习 (Safe Reinforcement Learning)
=============================================

📚 学习目标：
1. 理解安全强化学习的核心概念
2. 掌握双层架构设计（基础策略 + 安全层）
3. 学会实现安全裕度机制
4. 理解安全干预的重要性

🔧 技术要点：
- 安全边界 (Safety Boundary)
- 安全裕度 (Safety Margin)  
- 安全层 (Safety Layer)
- 危险动作过滤

💡 适用场景：
- 工业控制系统（温度、压力、速度等）
- 机器人导航（避障）
- 自动驾驶（碰撞避免）

🚀 运行方式：python learn_safe_rl.py
"""

import numpy as np
import os

# 设置matplotlib后端为非交互式，避免图形显示问题
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================
# 第1部分：安全强化学习智能体实现
# =============================================

class SafeRLAgent:
    """
    安全强化学习智能体
    
    架构说明：
    ┌─────────────────────────────────────┐
    │         环境 (Environment)          │
    │         当前温度: 21°C               │
    └────────────────────┬────────────────┘
                         │ 观测
                         ▼
    ┌─────────────────────────────────────┐
    │     基础策略 (Basic Policy)         │
    │  输入: 温度误差 → 输出: 动作        │
    │  例: 温度低 → 加热 (heat)           │
    └────────────────────┬────────────────┘
                         │ 动作建议
                         ▼
    ┌─────────────────────────────────────┐
    │     安全层 (Safety Layer)          │
    │  检查: 是否接近边界？              │
    │  过滤: 阻止危险动作                │
    │  例: 接近下限 → 禁止冷却           │
    └────────────────────┬────────────────┘
                         │ 安全动作
                         ▼
    ┌─────────────────────────────────────┐
    │         执行器 (Actuator)          │
    │       执行加热/冷却/保持            │
    └─────────────────────────────────────┘
    """
    
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化智能体参数
        
        参数解释：
        - temp_min: 温度下限（安全边界），低于此值可能损坏设备
        - temp_max: 温度上限（安全边界），高于此值可能引发危险
        - target_temp: 目标控制温度
        """
        self.temp_min = temp_min      # 安全边界：下限
        self.temp_max = temp_max      # 安全边界：上限
        self.target_temp = target_temp  # 控制目标
        
        # 安全裕度：距离边界的最小安全距离
        # 当温度进入此区域时，安全层开始干预
        self.safety_margin = 2.0
        
        # 当前状态
        self.current_temp = None
        self.step_count = 0
    
    def reset(self, initial_temp=None):
        """
        重置环境状态
        
        参数：
        - initial_temp: 可选，设置初始温度，用于测试特定场景
        """
        if initial_temp is None:
            # 随机初始温度（在安全范围内）
            self.current_temp = np.random.uniform(22, 28)
        else:
            self.current_temp = initial_temp
        self.step_count = 0
        print(f"🔄 环境已重置，初始温度: {self.current_temp:.1f}°C")
    
    def get_basic_action(self):
        """
        基础控制策略：基于温度误差的比例控制
        
        控制逻辑：
        - 如果温度低于目标1°C以上 → 加热
        - 如果温度高于目标1°C以上 → 冷却
        - 否则保持不变
        
        返回：'heat', 'cool', 或 'idle'
        """
        temp_error = self.current_temp - self.target_temp
        
        if temp_error < -1.0:
            action = 'heat'
        elif temp_error > 1.0:
            action = 'cool'
        else:
            action = 'idle'
        
        return action
    
    def safety_filter(self, proposed_action):
        """
        安全层：过滤危险动作
        
        安全规则：
        1. 接近下限（< 安全裕度）→ 禁止冷却
        2. 接近上限（< 安全裕度）→ 禁止加热
        3. 非常接近边界（< 1°C）→ 强制反向动作
        
        参数：
        - proposed_action: 基础策略建议的动作
        
        返回：
        - action: 经过安全过滤后的动作
        - is_filtered: 是否被安全层修改
        """
        # 计算到边界的距离
        distance_to_lower = self.current_temp - self.temp_min
        distance_to_upper = self.temp_max - self.current_temp
        
        action = proposed_action
        is_filtered = False
        
        # 规则1：接近下限时，禁止冷却动作
        if distance_to_lower < self.safety_margin:
            if action == 'cool':
                action = 'idle'  # 禁止冷却
                is_filtered = True
                print(f"⚠️ 安全干预: 温度接近下限 ({distance_to_lower:.1f}°C)，禁止冷却")
        
        # 规则2：接近上限时，禁止加热动作
        if distance_to_upper < self.safety_margin:
            if action == 'heat':
                action = 'idle'  # 禁止加热
                is_filtered = True
                print(f"⚠️ 安全干预: 温度接近上限 ({distance_to_upper:.1f}°C)，禁止加热")
        
        # 规则3：非常接近边界时，强制反向动作
        if distance_to_lower < 1.0:
            action = 'heat'  # 强制加热
            is_filtered = True
            print(f"🔒 强制保护: 温度极低 ({distance_to_lower:.1f}°C)，强制加热")
        
        if distance_to_upper < 1.0:
            action = 'cool'  # 强制冷却
            is_filtered = True
            print(f"🔒 强制保护: 温度极高 ({distance_to_upper:.1f}°C)，强制冷却")
        
        return action, is_filtered
    
    def step(self, action):
        """
        执行动作，更新环境状态
        
        参数：
        - action: 要执行的动作
        
        返回：
        - reward: 奖励值
        - done: 是否到达终止状态
        """
        # 温度变化模型
        delta_temp = -0.1  # 自然散热：每分钟下降0.1°C
        
        if action == 'heat':
            delta_temp += 0.3  # 加热：增加0.3°C
        elif action == 'cool':
            delta_temp -= 0.2  # 冷却：减少0.2°C
        
        # 更新温度（带边界约束）
        self.current_temp = max(
            self.temp_min, 
            min(self.temp_max, 
                self.current_temp + delta_temp)
        )
        self.step_count += 1
        
        # 计算奖励
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2  # 误差越小，奖励越高
        
        # 安全违规惩罚
        if self.current_temp <= self.temp_min or self.current_temp >= self.temp_max:
            reward -= 50.0  # 越界惩罚
        
        done = self.step_count >= 100
        
        return reward, done

# =============================================
# 第2部分：可视化工具
# =============================================

def plot_temperature_curve(results):
    """绘制温度变化曲线"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    targets = [r['target'] for r in results]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    plt.plot(steps, targets, label='目标温度', color='#10B981', linestyle='--', linewidth=2)
    plt.axhline(y=20, color='#EF4444', linestyle=':', label='安全下限')
    plt.axhline(y=35, color='#EF4444', linestyle=':', label='安全上限')
    
    # 标记安全裕度区域
    plt.fill_between(steps, 20, 22, color='#FEE2E2', alpha=0.3)
    plt.fill_between(steps, 33, 35, color='#FEE2E2', alpha=0.3)
    
    plt.xlabel('时间步')
    plt.ylabel('温度 (°C)')
    plt.title('安全强化学习 - 温度控制效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('temperature_curve.png')
    plt.close()

def plot_safety_intervention(results):
    """绘制安全干预记录"""
    steps = [r['step'] for r in results]
    filtered = [1 if r['was_filtered'] else 0 for r in results]
    
    plt.figure(figsize=(10, 3))
    plt.stem(steps, filtered, basefmt=' ', linefmt='r-', markerfmt='ro')
    plt.xlabel('时间步')
    plt.ylabel('安全干预')
    plt.title('安全层干预记录（1=被干预，0=正常）')
    plt.grid(True, alpha=0.3)
    plt.savefig('safety_intervention.png')
    plt.close()

# =============================================
# 第3部分：演示和学习练习
# =============================================

def run_demo():
    """完整演示流程"""
    print("=" * 70)
    print(" 📚 安全强化学习 - 学习演示")
    print("=" * 70)
    print()
    print("🎯 学习目标：")
    print("  1. 理解安全强化学习的双层架构")
    print("  2. 观察安全层如何阻止危险动作")
    print("  3. 分析安全裕度的作用")
    print()
    print("📋 实验设置：")
    print(f"  - 温度范围: {20}°C ~ {35}°C")
    print(f"  - 目标温度: {25}°C")
    print(f"  - 安全裕度: {2}°C")
    print("=" * 70)
    
    # 创建智能体
    agent = SafeRLAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 设置极端初始条件（靠近边界）
    print("\n🔬 测试场景：初始温度接近安全边界")
    agent.reset(initial_temp=21.0)  # 靠近下限
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = []
    
    while agent.step_count < 50:  # 缩短演示
        # 基础策略决策
        basic_action = agent.get_basic_action()
        
        # 安全层过滤
        action, is_filtered = agent.safety_filter(basic_action)
        
        # 执行动作
        reward, done = agent.step(action)
        
        # 记录
        results.append({
            'step': agent.step_count,
            'temperature': round(agent.current_temp, 2),
            'target': agent.target_temp,
            'basic_action': basic_action,
            'final_action': action,
            'was_filtered': is_filtered,
            'reward': round(reward, 2)
        })
        
        # 每5步输出一次状态
        if agent.step_count % 5 == 0:
            print(f"\n步骤 {agent.step_count}:")
            print(f"  温度: {agent.current_temp:.1f}°C | 目标: {agent.target_temp}°C")
            print(f"  基础动作: {basic_action} | 最终动作: {action}")
            print(f"  奖励: {reward:.1f}")
        
        if done:
            break
    
    # 统计分析
    print("\n" + "=" * 70)
    print("📊 实验结果分析")
    print("=" * 70)
    
    temps = [r['temperature'] for r in results]
    filtered_count = sum(1 for r in results if r['was_filtered'])
    
    print(f"\n📈 温度统计:")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  温度波动: {np.std(temps):.2f}°C")
    print(f"  最低温度: {min(temps):.2f}°C")
    print(f"  最高温度: {max(temps):.2f}°C")
    
    print(f"\n🔒 安全统计:")
    print(f"  安全干预次数: {filtered_count} 次")
    print(f"  干预率: {filtered_count/len(results)*100:.1f}%")
    
    print(f"\n✅ 安全验证:")
    print(f"  温度下限 ({20}°C): {'✓ 未越界' if min(temps) > 20 else '✗ 越界'}")
    print(f"  温度上限 ({35}°C): {'✓ 未越界' if max(temps) < 35 else '✗ 越界'}")
    
    # 可视化
    print("\n📉 生成可视化图表...")
    plot_temperature_curve(results)
    plot_safety_intervention(results)

def learning_exercises():
    """学习练习"""
    print("\n" + "=" * 70)
    print(" 💡 思考与练习")
    print("=" * 70)
    print()
    print("📝 问题1：")
    print("  如果安全裕度设置为3°C，会发生什么变化？")
    print("  提示：修改 self.safety_margin = 3.0 并运行")
    print()
    print("📝 问题2：")
    print("  如果初始温度设置为34°C（接近上限），安全层会如何反应？")
    print("  提示：修改 agent.reset(initial_temp=34.0)")
    print()
    print("📝 问题3：")
    print("  如果没有安全层，当温度接近边界时可能会发生什么？")
    print("  提示：注释掉安全层调用，直接使用 basic_action")
    print()
    print("🔧 扩展练习：")
    print("  尝试添加第三个安全约束：温度变化速率不能超过1°C/步")
    print()

if __name__ == "__main__":
    # 运行演示
    run_demo()
    
    # 显示练习
    learning_exercises()
    
    print("✅ 学习演示完成！")