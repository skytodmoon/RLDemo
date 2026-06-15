#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================
 学习版：层级强化学习 (Hierarchical RL)
=============================================

📚 学习目标：
1. 理解层级决策架构
2. 掌握选项架构 (Option Architecture)
3. 学会分解复杂任务
4. 理解时间抽象的意义

🔧 技术要点：
- 高层策略 (High-level Policy)
- 低层策略 (Low-level Policy)
- 选项 (Option)
- 时间抽象 (Temporal Abstraction)

💡 适用场景：
- 复杂任务分解
- 长期规划任务
- 多阶段任务

🚀 运行方式：python learn_hierarchical_rl.py
"""

import numpy as np
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================
# 第1部分：层级强化学习智能体实现
# =============================================

class HierarchicalRLAgent:
    """
    层级强化学习智能体
    
    架构说明：
    ┌─────────────────────────────────────────────┐
    │           高层策略 (High-level)            │
    │     决策：选择"选项"或"任务模式"            │
    │     例如：'approach' vs 'maintain'         │
    │     频率：低（每N步决策一次）               │
    └────────────────────┬───────────────────────┘
                         │ 选项指令
                         ▼
    ┌─────────────────────────────────────────────┐
    │           低层策略 (Low-level)             │
    │     执行：根据选项执行具体动作              │
    │     例如：'heat', 'cool', 'idle'           │
    │     频率：高（每步决策）                    │
    └────────────────────┬───────────────────────┘
                         │ 具体动作
                         ▼
    ┌─────────────────────────────────────────────┐
    │              环境 (Environment)            │
    │           执行动作，更新状态                │
    └─────────────────────────────────────────────┘
    
    选项定义：
    - 'approach': 快速接近目标温度
    - 'maintain': 精细维持目标温度
    - 'recovery': 从边界恢复
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
        
        # 当前选项（由高层策略决定）
        self.current_option = 'maintain'
        
        # 选项切换阈值
        self.approach_threshold = 3.0  # 误差大于此值时进入approach模式
        self.maintain_threshold = 1.0  # 误差小于此值时进入maintain模式
        
        # 当前状态
        self.current_temp = None
        self.step_count = 0
        self.option_step_count = 0
    
    def reset(self, initial_temp=None):
        """重置环境"""
        if initial_temp is None:
            self.current_temp = np.random.uniform(22, 28)
        else:
            self.current_temp = initial_temp
        self.step_count = 0
        self.option_step_count = 0
        self.current_option = 'maintain'
        print(f"🔄 环境已重置，初始温度: {self.current_temp:.1f}°C")
    
    def high_level_policy(self):
        """
        高层策略：选择选项（任务模式）
        
        决策逻辑：
        - 如果温度误差 > approach_threshold → 'approach' 模式（快速接近）
        - 如果温度接近边界 → 'recovery' 模式（恢复）
        - 否则 → 'maintain' 模式（精细维持）
        
        返回：
        - option: 当前选项
        """
        temp_error = self.current_temp - self.target_temp
        abs_error = abs(temp_error)
        
        # 检查是否接近边界
        distance_to_lower = self.current_temp - self.temp_min
        distance_to_upper = self.temp_max - self.current_temp
        
        # 边界恢复模式
        if distance_to_lower < 2.0:
            option = 'recovery_heat'
        elif distance_to_upper < 2.0:
            option = 'recovery_cool'
        # 快速接近模式
        elif abs_error > self.approach_threshold:
            option = 'approach'
        # 精细维持模式
        else:
            option = 'maintain'
        
        # 记录选项切换
        if option != self.current_option:
            print(f"🔀 选项切换: {self.current_option} → {option}")
            self.current_option = option
            self.option_step_count = 0
        
        self.option_step_count += 1
        
        return self.current_option
    
    def low_level_policy(self, option):
        """
        低层策略：根据选项执行具体动作
        
        参数：
        - option: 当前选项
        
        返回：
        - action: 具体动作 ('heat', 'cool', 'idle')
        """
        temp_error = self.current_temp - self.target_temp
        
        if option == 'approach':
            # 快速接近：使用较大动作
            if temp_error < -1.0:
                return 'heat'
            elif temp_error > 1.0:
                return 'cool'
            else:
                return 'idle'
        
        elif option == 'maintain':
            # 精细维持：使用更小的阈值
            if temp_error < -0.5:
                return 'heat'
            elif temp_error > 0.5:
                return 'cool'
            else:
                return 'idle'
        
        elif option == 'recovery_heat':
            # 边界恢复：强制加热
            return 'heat'
        
        elif option == 'recovery_cool':
            # 边界恢复：强制冷却
            return 'cool'
        
        else:
            return 'idle'
    
    def select_action(self):
        """
        层级决策：先选选项，再选动作
        
        返回：
        - action: 最终动作
        - option: 当前选项
        """
        option = self.high_level_policy()
        action = self.low_level_policy(option)
        return action, option
    
    def step(self, action):
        """执行动作，更新状态"""
        delta_temp = -0.1  # 自然散热
        
        if action == 'heat':
            delta_temp += 0.3
        elif action == 'cool':
            delta_temp -= 0.2
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        
        # 计算奖励
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        if self.current_temp <= self.temp_min or self.current_temp >= self.temp_max:
            reward -= 50.0
        
        done = self.step_count >= 100
        
        return reward, done

# =============================================
# 第2部分：可视化工具
# =============================================

def plot_results(results):
    """绘制完整结果"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    options = [r['option'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, [25]*len(steps), label='目标温度', color='#10B981', linestyle='--')
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='上限')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('层级RL温度控制')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 选项切换
    option_codes = {'approach': 1, 'maintain': 2, 'recovery_heat': 3, 'recovery_cool': 4}
    option_values = [option_codes[opt] for opt in options]
    ax2.step(steps, option_values, where='post', color='#8B5CF6', linewidth=2)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels(['approach', 'maintain', 'recovery_heat', 'recovery_cool'])
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('选项')
    ax2.set_title('选项切换时序')
    ax2.grid(True, alpha=0.3)
    
    # 奖励变化
    rewards = [r['reward'] for r in results]
    ax3.plot(steps, rewards, label='奖励', color='#10B981')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('奖励')
    ax3.set_title('奖励变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 温度误差
    errors = [abs(r['temperature'] - 25) for r in results]
    ax4.plot(steps, errors, label='温度误差', color='#F59E0B')
    ax4.axhline(y=3.0, color='#EF4444', linestyle=':', label='切换阈值')
    ax4.axhline(y=1.0, color='#10B981', linestyle=':', label='维持阈值')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('温度误差 (°C)')
    ax4.set_title('温度误差与阈值对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hierarchical_rl_results.png')
    plt.close()

# =============================================
# 第3部分：演示和学习练习
# =============================================

def run_demo():
    """完整演示流程"""
    print("=" * 70)
    print(" 📚 层级强化学习 - 学习演示")
    print("=" * 70)
    print()
    print("🎯 学习目标：")
    print("  1. 理解层级决策架构")
    print("  2. 掌握选项切换逻辑")
    print("  3. 理解时间抽象的意义")
    print()
    print("📋 实验设置：")
    print(f"  - 温度范围: {20}°C ~ {35}°C")
    print(f"  - 目标温度: {25}°C")
    print(f"  - 接近阈值: {3.0}°C")
    print(f"  - 维持阈值: {1.0}°C")
    print()
    print("🔧 选项定义：")
    print("  - 'approach': 快速接近目标（误差>3°C）")
    print("  - 'maintain': 精细维持（误差<1°C）")
    print("  - 'recovery_heat': 边界恢复-加热")
    print("  - 'recovery_cool': 边界恢复-冷却")
    print("=" * 70)
    
    # 创建智能体
    agent = HierarchicalRLAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 设置极端初始条件
    print("\n🔬 测试场景：初始温度远离目标")
    agent.reset(initial_temp=32.0)
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = []
    
    while agent.step_count < 50:
        action, option = agent.select_action()
        reward, done = agent.step(action)
        
        results.append({
            'step': agent.step_count,
            'temperature': round(agent.current_temp, 2),
            'action': action,
            'option': option,
            'reward': round(reward, 2)
        })
        
        # 每5步输出
        if agent.step_count % 5 == 0:
            print(f"\n步骤 {agent.step_count}:")
            print(f"  温度: {agent.current_temp:.1f}°C | 目标: {agent.target_temp}°C")
            print(f"  误差: {abs(agent.current_temp - agent.target_temp):.1f}°C")
            print(f"  当前选项: {option}")
            print(f"  执行动作: {action}")
    
    # 分析结果
    print("\n" + "=" * 70)
    print("📊 实验结果分析")
    print("=" * 70)
    
    temps = [r['temperature'] for r in results]
    print(f"\n📈 温度统计:")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  温度波动: {np.std(temps):.2f}°C")
    print(f"  最低温度: {min(temps):.2f}°C")
    print(f"  最高温度: {max(temps):.2f}°C")
    
    print(f"\n📊 选项分布:")
    option_counts = {}
    for r in results:
        option_counts[r['option']] = option_counts.get(r['option'], 0) + 1
    for opt, count in option_counts.items():
        print(f"  {opt}: {count}步 ({count/len(results)*100:.1f}%)")
    
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
    print("  如果接近阈值设置为5°C，会有什么变化？")
    print("  提示：修改 self.approach_threshold = 5.0")
    print()
    print("📝 问题2：")
    print("  如果去掉边界恢复选项，会发生什么？")
    print("  提示：注释掉recovery相关的代码")
    print()
    print("📝 问题3：")
    print("  为什么需要层级架构？单层策略有什么缺点？")
    print("  提示：考虑复杂任务的学习效率")
    print()
    print("🔧 扩展练习：")
    print("  添加新选项 'explore'，用于探索温度动态")
    print("  提示：随机选择动作，收集环境信息")
    print()

if __name__ == "__main__":
    run_demo()
    learning_exercises()
    print("✅ 学习演示完成！")