#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================
 学习版：RL+MPC混合控制
=============================================

📚 学习目标：
1. 理解混合控制架构
2. 掌握模型预测控制原理
3. 学会自适应权重融合
4. 理解不确定性感知

🔧 技术要点：
- 强化学习策略 (RL Policy)
- 模型预测控制 (MPC)
- 自适应权重 (Adaptive Weight)
- 不确定性估计 (Uncertainty Estimation)

💡 适用场景：
- 需要高精度控制的场景
- 模型不完全已知的系统
- 需要兼顾探索与利用

🚀 运行方式：python learn_rl_mpc.py
"""

import numpy as np
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================
# 第1部分：RL+MPC混合控制智能体实现
# =============================================

class RLMPCHybridAgent:
    """
    RL+MPC混合控制智能体
    
    架构说明：
    ┌─────────────────────────────────────────────────┐
    │              状态观测                           │
    │           当前温度、误差、边界距离              │
    └──────────────────────┬────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │  RL策略   │    │  MPC策略   │    │ 不确定性估计│
    │ (探索性)  │    │ (精确控制) │    │ (风险评估)  │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                │                │
          └────────┬───────┴────────┬───────┘
                   ▼                ▼
            ┌─────────────┐  ┌─────────────┐
            │ 自适应权重   │  │ 不确定性    │
            │ w = f(unc)  │  │             │
            └──────┬──────┘  └─────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────┐
    │  最终动作 = w × RL_action + (1-w) × MPC_action │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
                    执行动作 → 更新状态
    
    自适应权重逻辑：
    - 当不确定性高时 → 偏向MPC（更可靠）
    - 当不确定性低时 → 偏向RL（更灵活）
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
        
        # MPC参数
        self.mpc_horizon = 10  # MPC规划时域
        self.mpc_gamma = 0.95   # MPC折扣因子
        
        # RL策略参数
        self.rl_kp = 0.5  # 比例系数
        
        # 自适应权重参数
        self.base_weight = 0.5  # 基础权重
        self.min_weight = 0.1   # 最小权重（RL）
        self.max_weight = 0.9   # 最大权重（RL）
        
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
    
    def rl_policy(self):
        """
        RL策略：基于误差的简单策略
        
        返回：
        - action: RL建议的动作
        """
        temp_error = self.current_temp - self.target_temp
        
        if temp_error < -self.rl_kp:
            return 'heat'
        elif temp_error > self.rl_kp:
            return 'cool'
        else:
            return 'idle'
    
    def predict_dynamics(self, temp, action):
        """
        动力学模型：预测温度变化
        
        参数：
        - temp: 当前温度
        - action: 执行的动作
        
        返回：
        - next_temp: 预测温度
        """
        delta = -0.1  # 自然散热
        
        if action == 'heat':
            delta += 0.3
        elif action == 'cool':
            delta -= 0.2
        
        return max(self.temp_min, min(self.temp_max, temp + delta))
    
    def mpc_policy(self):
        """
        MPC策略：多步优化选择最优动作序列
        
        返回：
        - best_action: 最优初始动作
        - best_value: 最优值
        """
        actions = ['heat', 'cool', 'idle']
        best_action = 'idle'
        best_value = float('-inf')
        
        for action in actions:
            # 模拟执行此动作后的轨迹
            temp = self.current_temp
            total_value = 0
            
            for t in range(self.mpc_horizon):
                temp = self.predict_dynamics(temp, action)
                error = abs(temp - self.target_temp)
                reward = 10.0 - error * 2
                total_value += reward * (self.mpc_gamma ** t)
            
            if total_value > best_value:
                best_value = total_value
                best_action = action
        
        return best_action, best_value
    
    def estimate_uncertainty(self):
        """
        估计系统不确定性
        
        不确定性来源：
        1. 温度误差越大，模型不确定性越高
        2. 接近边界时，安全风险增加
        
        返回：
        - uncertainty: 不确定性值 (0-1)
        """
        # 误差不确定性
        temp_error = abs(self.current_temp - self.target_temp)
        error_uncertainty = min(temp_error / 15.0, 1.0)
        
        # 边界不确定性
        distance_to_boundary = min(
            self.current_temp - self.temp_min,
            self.temp_max - self.current_temp
        )
        boundary_uncertainty = max(0, (2.0 - distance_to_boundary) / 2.0)
        
        # 综合不确定性
        uncertainty = (error_uncertainty + boundary_uncertainty) / 2.0
        
        return uncertainty
    
    def adaptive_weight(self):
        """
        计算自适应权重
        
        返回：
        - weight: RL动作的权重 (0-1)
                  MPC权重 = 1 - weight
        """
        uncertainty = self.estimate_uncertainty()
        
        # 不确定性越高，RL权重越低（更依赖MPC）
        weight = self.base_weight - uncertainty * 0.4
        
        # 限制权重范围
        weight = max(self.min_weight, min(self.max_weight, weight))
        
        return weight, uncertainty
    
    def select_action(self):
        """
        选择最终动作：融合RL和MPC
        
        返回：
        - final_action: 最终动作
        - rl_action: RL建议动作
        - mpc_action: MPC建议动作
        - weight: RL权重
        - uncertainty: 不确定性估计
        """
        # 获取两个策略的建议
        rl_action = self.rl_policy()
        mpc_action, mpc_value = self.mpc_policy()
        
        # 计算自适应权重
        weight, uncertainty = self.adaptive_weight()
        
        # 基于权重选择动作
        if np.random.random() < weight:
            final_action = rl_action
        else:
            final_action = mpc_action
        
        return final_action, rl_action, mpc_action, weight, uncertainty
    
    def step(self, action):
        """执行动作，更新状态"""
        delta_temp = -0.1
        
        if action == 'heat':
            delta_temp += 0.3
        elif action == 'cool':
            delta_temp -= 0.2
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        
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
    weights = [r['rl_weight'] for r in results]
    uncertainties = [r['uncertainty'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, [25]*len(steps), label='目标温度', color='#10B981', linestyle='--')
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='上限')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('RL+MPC混合控制')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 自适应权重
    ax2.plot(steps, weights, label='RL权重', color='#8B5CF6', linewidth=2)
    ax2.plot(steps, [1 - w for w in weights], label='MPC权重', color='#F59E0B', linewidth=2)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('权重')
    ax2.set_title('自适应权重变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 不确定性估计
    ax3.plot(steps, uncertainties, label='不确定性', color='#EF4444', linewidth=2)
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('不确定性')
    ax3.set_title('系统不确定性估计')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 策略选择对比
    rl_counts = sum(1 for r in results if r['source'] == 'RL')
    mpc_counts = sum(1 for r in results if r['source'] == 'MPC')
    ax4.bar(['RL', 'MPC'], [rl_counts, mpc_counts], color=['#8B5CF6', '#F59E0B'])
    ax4.set_xlabel('策略来源')
    ax4.set_ylabel('次数')
    ax4.set_title('策略选择分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_mpc_results.png')
    plt.close()

# =============================================
# 第3部分：演示和学习练习
# =============================================

def run_demo():
    """完整演示流程"""
    print("=" * 70)
    print(" 📚 RL+MPC混合控制 - 学习演示")
    print("=" * 70)
    print()
    print("🎯 学习目标：")
    print("  1. 理解混合控制架构")
    print("  2. 掌握自适应权重机制")
    print("  3. 理解不确定性感知")
    print()
    print("📋 实验设置：")
    print(f"  - 温度范围: {20}°C ~ {35}°C")
    print(f"  - 目标温度: {25}°C")
    print(f"  - MPC时域: {10}步")
    print(f"  - 基础权重: {0.5}")
    print()
    print("🔧 混合策略：")
    print("  - RL策略：基于误差的比例控制")
    print("  - MPC策略：多步优化规划")
    print("  - 自适应融合：根据不确定性动态调整权重")
    print("=" * 70)
    
    # 创建智能体
    agent = RLMPCHybridAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 设置极端初始条件
    print("\n🔬 测试场景：初始温度远离目标且接近边界")
    agent.reset(initial_temp=33.0)
    
    # 运行策略
    print("\n▶️ 开始运行...")
    results = []
    
    while agent.step_count < 50:
        final_action, rl_action, mpc_action, weight, uncertainty = agent.select_action()
        reward, done = agent.step(final_action)
        
        source = 'RL' if final_action == rl_action else 'MPC'
        
        results.append({
            'step': agent.step_count,
            'temperature': round(agent.current_temp, 2),
            'final_action': final_action,
            'rl_action': rl_action,
            'mpc_action': mpc_action,
            'rl_weight': round(weight, 3),
            'uncertainty': round(uncertainty, 3),
            'source': source,
            'reward': round(reward, 2)
        })
        
        # 每5步输出
        if agent.step_count % 5 == 0:
            print(f"\n步骤 {agent.step_count}:")
            print(f"  温度: {agent.current_temp:.1f}°C | 目标: {agent.target_temp}°C")
            print(f"  误差: {abs(agent.current_temp - agent.target_temp):.1f}°C")
            print(f"  RL动作: {rl_action} | MPC动作: {mpc_action}")
            print(f"  最终动作: {final_action} ({source})")
            print(f"  RL权重: {weight:.2f} | 不确定性: {uncertainty:.2f}")
    
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
    
    print(f"\n📊 策略使用统计:")
    rl_usage = sum(1 for r in results if r['source'] == 'RL')
    mpc_usage = sum(1 for r in results if r['source'] == 'MPC')
    print(f"  RL策略: {rl_usage}次 ({rl_usage/len(results)*100:.1f}%)")
    print(f"  MPC策略: {mpc_usage}次 ({mpc_usage/len(results)*100:.1f}%)")
    
    print(f"\n📊 不确定性统计:")
    uncertainties = [r['uncertainty'] for r in results]
    print(f"  平均不确定性: {np.mean(uncertainties):.2f}")
    print(f"  最大不确定性: {max(uncertainties):.2f}")
    print(f"  最小不确定性: {min(uncertainties):.2f}")
    
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
    print("  如果基础权重设置为0.8（更偏向RL），会发生什么？")
    print("  提示：修改 self.base_weight = 0.8")
    print()
    print("📝 问题2：")
    print("  如果去掉不确定性感知，固定权重为0.5，效果会如何？")
    print("  提示：注释掉自适应权重，直接使用固定值")
    print()
    print("📝 问题3：")
    print("  MPC的规划时域设置为3步会有什么影响？")
    print("  提示：修改 self.mpc_horizon = 3")
    print()
    print("🔧 扩展练习：")
    print("  添加更多不确定性来源：如环境噪声、模型误差估计")
    print("  提示：引入随机噪声到动力学模型")
    print()

if __name__ == "__main__":
    run_demo()
    learning_exercises()
    print("✅ 学习演示完成！")