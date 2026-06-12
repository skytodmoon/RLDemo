#!/usr/bin/env python3
"""
层级强化学习演示 Demo
Hierarchical Reinforcement Learning Demo

技术路线：
1. 将复杂任务分解为高层选项（Options）和低层动作
2. 高层策略：选择目标/选项（如"接近目标"或"维持温度"）
3. 低层策略：执行具体动作（加热/冷却/空闲）
4. 选项终止机制：决定何时切换选项

工程过程：
1. 定义高层选项空间
2. 实现选项策略（选择当前应执行的选项）
3. 实现低层动作策略
4. 实现选项终止判断机制
"""

import numpy as np
import matplotlib.pyplot as plt

class HierarchicalRLAgent:
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化层级强化学习智能体
        
        参数：
            temp_min: 温度下限
            temp_max: 温度上限
            target_temp: 目标温度
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.target_temp = target_temp
        
        # 选项空间（高层目标）
        self.options = ['approach', 'maintain', 'safe_guard']
        
        # 当前选项
        self.current_option = 'maintain'
        
        # 选项持续时间计数
        self.option_steps = 0
        
        # 选项最小持续时间
        self.min_option_duration = 3
        
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
        self.current_option = 'maintain'
        self.option_steps = 0
        
    def high_level_policy(self):
        """
        高层策略：选择当前选项（目标）
        
        返回：
            option: 选中的选项
        """
        temp_error = self.current_temp - self.target_temp
        abs_error = abs(temp_error)
        
        # 安全保护选项：当温度接近边界时
        distance_to_boundary = min(
            self.current_temp - self.temp_min,
            self.temp_max - self.current_temp
        )
        if distance_to_boundary < 2.0:
            return 'safe_guard'
        
        # 接近目标选项：当误差较大时
        if abs_error > 3.0:
            return 'approach'
        
        # 维持选项：当误差较小时
        return 'maintain'
    
    def low_level_policy(self, option):
        """
        低层策略：根据当前选项执行具体动作
        
        参数：
            option: 当前选项
        
        返回：
            action: 要执行的动作
        """
        temp_error = self.current_temp - self.target_temp
        
        if option == 'approach':
            # 快速接近目标
            if temp_error < 0:
                return 'heat'
            else:
                return 'cool'
        
        elif option == 'maintain':
            # 精细维持温度
            if temp_error < -0.5:
                return 'heat'
            elif temp_error > 0.5:
                return 'cool'
            else:
                return 'idle'
        
        elif option == 'safe_guard':
            # 安全保护：确保不越界
            if self.current_temp <= self.temp_min + 1.0:
                return 'heat'
            elif self.current_temp >= self.temp_max - 1.0:
                return 'cool'
            else:
                return 'idle'
    
    def option_termination(self):
        """
        判断是否应该终止当前选项
        
        返回：
            should_terminate: 是否应该终止
        """
        # 选项必须持续最小步数
        if self.option_steps < self.min_option_duration:
            return False
        
        # 根据状态判断是否需要切换选项
        new_option = self.high_level_policy()
        if new_option != self.current_option:
            return True
        
        return False
    
    def step(self, action):
        """
        执行动作，更新环境状态
        
        参数：
            action: 要执行的动作
        
        返回：
            reward: 奖励值
            done: 是否结束
        """
        delta_temp = -0.1
        
        if action == 'heat':
            delta_temp += 0.3
        elif action == 'cool':
            delta_temp -= 0.2
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        self.option_steps += 1
        
        # 计算奖励
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        if self.current_temp <= self.temp_min or self.current_temp >= self.temp_max:
            reward -= 50.0
        
        done = self.step_count >= 100
        
        return reward, done
    
    def run_policy(self):
        """运行完整的层级强化学习策略"""
        results = []
        
        while self.step_count < 100:
            # 检查是否需要切换选项
            if self.option_termination():
                self.current_option = self.high_level_policy()
                self.option_steps = 0
            
            # 根据当前选项选择动作
            action = self.low_level_policy(self.current_option)
            
            # 执行动作
            reward, done = self.step(action)
            
            # 记录结果
            results.append({
                'step': self.step_count,
                'temperature': round(self.current_temp, 2),
                'target': self.target_temp,
                'option': self.current_option,
                'action': action,
                'reward': round(reward, 2),
                'option_steps': self.option_steps
            })
            
            if done:
                break
        
        return results

def plot_results(results):
    """可视化演示结果"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    targets = [r['target'] for r in results]
    
    # 选项颜色映射
    option_colors = {
        'approach': '#F59E0B',
        'maintain': '#10B981',
        'safe_guard': '#EF4444'
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度变化曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, targets, label='目标温度', color='#10B981', linestyle='--', linewidth=2)
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='温度下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='温度上限')
    ax1.fill_between(steps, 20, 35, color='#FEE2E2', alpha=0.3)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('层级强化学习温度控制')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 选项切换可视化
    ax2.step(steps, [option_colors[r['option']] for r in results], 
             where='post', linewidth=3)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('选项')
    ax2.set_title('选项切换（高层策略）')
    ax2.grid(True, alpha=0.3)
    
    # 奖励变化
    rewards = [r['reward'] for r in results]
    ax3.plot(steps, rewards, label='奖励', color='#10B981')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('奖励值')
    ax3.set_title('奖励变化曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 选项统计
    option_counts = {'approach': 0, 'maintain': 0, 'safe_guard': 0}
    for r in results:
        option_counts[r['option']] += 1
    ax4.bar(option_counts.keys(), option_counts.values(), 
            color=[option_colors[k] for k in option_counts.keys()])
    ax4.set_xlabel('选项')
    ax4.set_ylabel('持续步数')
    ax4.set_title('选项执行分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """打印演示摘要"""
    print("=" * 60)
    print("层级强化学习演示摘要")
    print("=" * 60)
    
    temps = [r['temperature'] for r in results]
    rewards = [r['reward'] for r in results]
    
    print(f"\n📊 统计信息:")
    print(f"  总步数: {len(results)}")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  温度标准差: {np.std(temps):.2f}°C")
    print(f"  最大温度: {max(temps):.2f}°C")
    print(f"  最小温度: {min(temps):.2f}°C")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    
    print("\n🏰 选项统计:")
    option_counts = {'approach': 0, 'maintain': 0, 'safe_guard': 0}
    for r in results:
        option_counts[r['option']] += 1
    for option, count in option_counts.items():
        print(f"  {option.upper()}: {count}步 ({count/len(results)*100:.1f}%)")
    
    print("\n🔄 选项切换次数:")
    switch_count = 0
    prev_option = results[0]['option']
    for r in results[1:]:
        if r['option'] != prev_option:
            switch_count += 1
            prev_option = r['option']
    print(f"  切换次数: {switch_count}次")
    
    print("\n🎯 技术要点:")
    print("  1. 高层策略：选择选项/目标（approach/maintain/safe_guard）")
    print("  2. 低层策略：根据选项执行具体动作")
    print("  3. 选项终止：基于状态判断是否切换选项")
    print("  4. 层次分解：将复杂任务分解为简单子任务")

if __name__ == "__main__":
    print("🚀 层级强化学习演示")
    print("=" * 60)
    print("技术路线：分层任务分解（高层选项 + 低层动作）")
    print("选项空间：approach（接近）| maintain（维持）| safe_guard（安全保护）")
    print("目标温度：25°C")
    print("温度范围：20°C ~ 35°C")
    print("=" * 60)
    
    # 创建智能体
    agent = HierarchicalRLAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 重置环境（设置较大误差测试选项切换）
    agent.reset(initial_temp=29.0)
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = agent.run_policy()
    
    # 输出结果
    print_summary(results)
    
    # 可视化
    print("\n📈 生成可视化图表...")
    plot_results(results)
    
    print("\n✅ 演示完成！")