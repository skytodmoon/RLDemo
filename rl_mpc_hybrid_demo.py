#!/usr/bin/env python3
"""
RL+MPC混合控制演示 Demo
Reinforcement Learning + Model Predictive Control Hybrid Demo

技术路线：
1. 强化学习策略：学习近似最优策略作为初始策略
2. 模型预测控制：基于动力学模型进行在线优化
3. 混合决策：融合RL策略和MPC优化结果
4. 自适应权重：根据状态不确定性调整权重

工程过程：
1. 训练/定义RL策略（规则策略作为代理）
2. 实现MPC控制器
3. 设计融合机制（加权混合）
4. 验证混合控制性能
"""

import numpy as np
import matplotlib.pyplot as plt

class RLMPCHybridAgent:
    def __init__(self, temp_min=20.0, temp_max=35.0, target_temp=25.0):
        """
        初始化RL+MPC混合控制智能体
        
        参数：
            temp_min: 温度下限
            temp_max: 温度上限
            target_temp: 目标温度
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.target_temp = target_temp
        
        # RL策略权重（0~1，0=纯MPC，1=纯RL）
        self.rl_weight = 0.3
        
        # MPC参数
        self.mpc_horizon = 3
        
        # 动力学模型参数
        self.model_params = {
            'cool_effect': -0.2,
            'heat_effect': 0.3,
            'natural_decay': -0.1
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
        
    def rl_policy(self):
        """
        强化学习策略（简化版，模拟学习到的策略）
        
        返回：
            action: RL策略输出的动作
        """
        temp_error = self.current_temp - self.target_temp
        
        # 学习到的策略：带滞后的比例控制
        if temp_error < -1.0:
            return 'heat'
        elif temp_error > 1.0:
            return 'cool'
        else:
            return 'idle'
    
    def mpc_policy(self):
        """
        模型预测控制策略
        
        返回：
            best_action: MPC优化后的动作
            best_cost: 最优代价
        """
        actions = ['heat', 'cool', 'idle']
        best_action = 'idle'
        best_cost = float('inf')
        
        for action in actions:
            temp = self.current_temp
            total_cost = 0
            
            for t in range(self.mpc_horizon):
                # 预测温度变化
                delta = self.model_params['natural_decay']
                if action == 'heat':
                    delta += self.model_params['heat_effect']
                elif action == 'cool':
                    delta += self.model_params['cool_effect']
                
                temp = max(self.temp_min, min(self.temp_max, temp + delta))
                
                # 计算代价（与目标的误差）
                error = abs(temp - self.target_temp)
                total_cost += error * (0.95 ** t)
                
                # 边界惩罚
                if temp <= self.temp_min or temp >= self.temp_max:
                    total_cost += 100
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = action
        
        return best_action, best_cost
    
    def hybrid_policy(self):
        """
        混合策略：融合RL和MPC
        
        返回：
            action: 最终动作
            source: 动作来源（'rl'或'mpc'）
        """
        rl_action = self.rl_policy()
        mpc_action, _ = self.mpc_policy()
        
        # 加权随机选择
        if np.random.random() < self.rl_weight:
            return rl_action, 'rl'
        else:
            return mpc_action, 'mpc'
    
    def adaptive_hybrid_policy(self):
        """
        自适应混合策略：根据状态调整权重
        
        当温度接近目标且远离边界时，更多依赖RL
        当温度接近边界或误差较大时，更多依赖MPC
        
        返回：
            action: 最终动作
            source: 动作来源
            weight: 当前RL权重
        """
        rl_action = self.rl_policy()
        mpc_action, _ = self.mpc_policy()
        
        # 计算状态不确定性指标
        temp_error = abs(self.current_temp - self.target_temp)
        distance_to_boundary = min(
            self.current_temp - self.temp_min,
            self.temp_max - self.current_temp
        )
        
        # 自适应权重：误差越大、距离边界越近，越依赖MPC
        uncertainty = (temp_error / 15.0) + ((2.0 - min(distance_to_boundary, 2.0)) / 2.0)
        uncertainty = min(1.0, max(0.0, uncertainty))
        
        adaptive_weight = max(0.1, 0.5 - uncertainty * 0.4)
        
        if np.random.random() < adaptive_weight:
            return rl_action, 'rl', adaptive_weight
        else:
            return mpc_action, 'mpc', adaptive_weight
    
    def step(self, action):
        """
        执行动作，更新环境状态
        
        参数：
            action: 要执行的动作
        
        返回：
            reward: 奖励值
            done: 是否结束
        """
        delta_temp = self.model_params['natural_decay']
        
        if action == 'heat':
            delta_temp += self.model_params['heat_effect']
        elif action == 'cool':
            delta_temp += self.model_params['cool_effect']
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        if self.current_temp <= self.temp_min or self.current_temp >= self.temp_max:
            reward -= 50.0
        
        done = self.step_count >= 100
        
        return reward, done
    
    def run_policy(self, use_adaptive=True):
        """运行完整的混合控制策略"""
        results = []
        
        while self.step_count < 100:
            if use_adaptive:
                action, source, weight = self.adaptive_hybrid_policy()
            else:
                action, source = self.hybrid_policy()
                weight = self.rl_weight
            
            reward, done = self.step(action)
            
            results.append({
                'step': self.step_count,
                'temperature': round(self.current_temp, 2),
                'target': self.target_temp,
                'action': action,
                'source': source,
                'rl_weight': round(weight, 3),
                'reward': round(reward, 2)
            })
            
            if done:
                break
        
        return results

def plot_results(results):
    """可视化演示结果"""
    steps = [r['step'] for r in results]
    temps = [r['temperature'] for r in results]
    targets = [r['target'] for r in results]
    weights = [r['rl_weight'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 温度变化曲线
    ax1.plot(steps, temps, label='实际温度', color='#3B82F6', linewidth=2)
    ax1.plot(steps, targets, label='目标温度', color='#10B981', linestyle='--', linewidth=2)
    ax1.axhline(y=20, color='#EF4444', linestyle=':', label='温度下限')
    ax1.axhline(y=35, color='#EF4444', linestyle=':', label='温度上限')
    ax1.fill_between(steps, 20, 35, color='#FEE2E2', alpha=0.3)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title('RL+MPC混合控制温度演示')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RL权重变化
    ax2.plot(steps, weights, label='RL权重', color='#8B5CF6', linewidth=2)
    ax2.fill_between(steps, 0, weights, color='#8B5CF6', alpha=0.2)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('RL权重')
    ax2.set_title('自适应RL权重变化')
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
    
    # 动作来源分布
    source_counts = {'rl': 0, 'mpc': 0}
    for r in results:
        source_counts[r['source']] += 1
    ax4.bar(source_counts.keys(), source_counts.values(), 
            color=['#3B82F6', '#F59E0B'])
    ax4.set_xlabel('动作来源')
    ax4.set_ylabel('次数')
    ax4.set_title('动作来源分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """打印演示摘要"""
    print("=" * 60)
    print("RL+MPC混合控制演示摘要")
    print("=" * 60)
    
    temps = [r['temperature'] for r in results]
    rewards = [r['reward'] for r in results]
    weights = [r['rl_weight'] for r in results]
    
    print(f"\n📊 统计信息:")
    print(f"  总步数: {len(results)}")
    print(f"  平均温度: {np.mean(temps):.2f}°C")
    print(f"  温度标准差: {np.std(temps):.2f}°C")
    print(f"  最大温度: {max(temps):.2f}°C")
    print(f"  最小温度: {min(temps):.2f}°C")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  平均RL权重: {np.mean(weights):.3f}")
    
    print("\n⚡ 动作来源统计:")
    source_counts = {'rl': 0, 'mpc': 0}
    for r in results:
        source_counts[r['source']] += 1
    for source, count in source_counts.items():
        print(f"  {source.upper()}: {count}次 ({count/len(results)*100:.1f}%)")
    
    print("\n🎯 技术要点:")
    print("  1. RL策略：学习近似最优策略，计算效率高")
    print("  2. MPC控制：基于模型在线优化，精度高")
    print("  3. 混合机制：加权融合两种策略")
    print("  4. 自适应权重：根据状态不确定性动态调整")
    print("  5. 优势互补：兼顾学习能力和在线优化能力")

if __name__ == "__main__":
    print("🚀 RL+MPC混合控制演示")
    print("=" * 60)
    print("技术路线：强化学习 + 模型预测控制混合")
    print("自适应权重：根据状态不确定性动态调整")
    print("MPC时域：3步")
    print("目标温度：25°C")
    print("温度范围：20°C ~ 35°C")
    print("=" * 60)
    
    # 创建智能体
    agent = RLMPCHybridAgent(temp_min=20.0, temp_max=35.0, target_temp=25.0)
    
    # 重置环境
    agent.reset(initial_temp=27.0)
    
    # 运行策略（使用自适应混合）
    print("\n▶️ 开始运行...")
    results = agent.run_policy(use_adaptive=True)
    
    # 输出结果
    print_summary(results)
    
    # 可视化
    print("\n📈 生成可视化图表...")
    plot_results(results)
    
    print("\n✅ 演示完成！")