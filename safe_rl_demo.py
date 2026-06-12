#!/usr/bin/env python3
"""
安全强化学习演示 Demo
Safe Reinforcement Learning Demo

技术路线：
1. 在标准 RL 策略基础上增加安全层（Safety Layer）
2. 当系统状态接近安全边界时，安全层会阻止危险动作
3. 双层架构：底层策略 + 安全约束层
4. 使用安全裕度（Safety Margin）概念

工程过程：
1. 定义安全边界和安全裕度
2. 实现基础控制策略（如规则控制或学习到的策略）
3. 添加安全层过滤危险动作
4. 模拟环境并验证安全性
"""

import numpy as np
import matplotlib.pyplot as plt

class SafeRLAgent:
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化安全强化学习智能体
        
        参数：
            temp_min: 温度下限（安全边界）
            temp_max: 温度上限（安全边界）
            target_temp: 目标温度
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.target_temp = target_temp
        
        # 安全裕度：距离边界的最小安全距离
        self.safety_margin = 2.0
        
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
        
    def get_basic_action(self):
        """
        基础策略：基于误差的简单控制
        返回：'heat', 'cool', 或 'idle'
        """
        temp_error = self.current_temp - self.target_temp
        
        if temp_error < -1.0:
            return 'heat'
        elif temp_error > 1.0:
            return 'cool'
        else:
            return 'idle'
    
    def safety_filter(self, action):
        """
        安全层：过滤危险动作
        
        当温度接近边界时，禁止可能导致越界的动作：
        - 接近下限时，禁止冷却动作
        - 接近上限时，禁止加热动作
        
        参数：
            action: 基础策略输出的动作
        
        返回：
            filtered_action: 经过安全过滤后的动作
            is_filtered: 是否被安全层修改
        """
        distance_to_lower = self.current_temp - self.temp_min
        distance_to_upper = self.temp_max - self.current_temp
        
        is_filtered = False
        
        # 接近下限：禁止冷却
        if distance_to_lower < self.safety_margin:
            if action == 'cool':
                action = 'idle'
                is_filtered = True
        
        # 接近上限：禁止加热
        if distance_to_upper < self.safety_margin:
            if action == 'heat':
                action = 'idle'
                is_filtered = True
        
        # 非常接近边界：主动采取反向动作
        if distance_to_lower < 1.0:
            action = 'heat'
            is_filtered = True
        if distance_to_upper < 1.0:
            action = 'cool'
            is_filtered = True
        
        return action, is_filtered
    
    def step(self, action):
        """
        执行动作，更新环境状态
        
        参数：
            action: 要执行的动作
        
        返回：
            reward: 奖励值
            done: 是否到达终止状态
        """
        # 应用动作效果
        delta_temp = -0.1  # 自然散热
        
        if action == 'heat':
            delta_temp += 0.3
        elif action == 'cool':
            delta_temp -= 0.2
        
        # 更新温度（带边界约束）
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        
        # 计算奖励
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        # 安全违规惩罚
        if self.current_temp <= self.temp_min or self.current_temp >= self.temp_max:
            reward -= 50.0
        
        done = self.step_count >= 100
        
        return reward, done
    
    def run_policy(self):
        """运行完整的安全强化学习策略"""
        results = []
        
        while self.step_count < 100:
            # 基础策略决策
            basic_action = self.get_basic_action()
            
            # 安全层过滤
            action, is_filtered = self.safety_filter(basic_action)
            
            # 执行动作
            reward, done = self.step(action)
            
            # 记录结果
            results.append({
                'step': self.step_count,
                'temperature': round(self.current_temp, 2),
                'target': self.target_temp,
                'basic_action': basic_action,
                'final_action': action,
                'was_filtered': is_filtered,
                'reward': round(reward, 2),
                'distance_to_lower': round(self.current_temp - self.temp_min, 2),
                'distance_to_upper': round(self.temp_max - self.current_temp, 2)
            })
            
            if done:
                break
        
        return results

def plot_results(results):
    """可视化演示结果"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    targets = [r['target'] for r in results]
    filtered = [1 if r['was_filtered'] else 0 for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 温度变化曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, targets, label='目标温度', color='#10B981', linestyle='--', linewidth=2)
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='温度下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='温度上限')
    ax1.fill_between(steps, 20, 35, color='#FEE2E2', alpha=0.3)
    
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('安全强化学习温度控制演示')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 安全层干预点
    ax2.stem(steps, filtered, basefmt=' ', use_line_collection=True)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('安全干预')
    ax2.set_title('安全层干预记录（1=被干预，0=未干预）')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """打印演示摘要"""
    print("=" * 60)
    print("安全强化学习演示摘要")
    print("=" * 60)
    
    temps = [r['temperature'] for r in results]
    rewards = [r['reward'] for r in results]
    filtered_count = sum(1 for r in results if r['was_filtered'])
    
    print(f"\n📊 统计信息:")
    print(f"  总步数: {len(results)}")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  温度标准差: {np.std(temps):.2f}°C")
    print(f"  最大温度: {max(temps):.2f}°C")
    print(f"  最小温度: {min(temps):.2f}°C")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  安全干预次数: {filtered_count} ({filtered_count/len(results)*100:.1f}%)")
    
    print("\n🔒 安全验证:")
    print(f"  温度下限 ({results[0]['target']-5}°C): {'✓ 未越界' if min(temps) > 20 else '✗ 越界'}")
    print(f"  温度上限 ({results[0]['target']+10}°C): {'✓ 未越界' if max(temps) < 35 else '✗ 越界'}")
    
    print("\n🎯 技术要点:")
    print("  1. 双层架构：基础策略 + 安全层")
    print("  2. 安全裕度：距离边界2°C时开始限制动作")
    print("  3. 主动保护：距离边界1°C时强制反向动作")
    print("  4. 过滤机制：禁止可能导致越界的动作")

if __name__ == "__main__":
    print("🚀 安全强化学习演示")
    print("=" * 60)
    print("技术路线：双层架构（基础策略 + 安全层）")
    print("安全裕度：2°C")
    print("目标温度：25°C")
    print("温度范围：20°C ~ 35°C")
    print("=" * 60)
    
    # 创建智能体
    agent = SafeRLAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 重置环境（设置一个靠近边界的初始温度，测试安全机制）
    agent.reset(initial_temp=21.0)  # 靠近下限
    # agent.reset(initial_temp=34.0)  # 靠近上限
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = agent.run_policy()
    
    # 输出结果
    print_summary(results)
    
    # 可视化
    print("\n📈 生成可视化图表...")
    plot_results(results)
    
    print("\n✅ 演示完成！")