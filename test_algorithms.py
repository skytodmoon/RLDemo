import numpy as np
from app import RuleBasedSimulator, SafeRLSimulator, ConstrainedRLSimulator, MBPO_Simulator, HierarchicalRLSimulator, RLMPCSimulator

def test_simulator(sim, name, steps=50):
    sim.reset()
    sim.current_temp = 22.0  # 从较低温度开始
    errors = []
    
    for _ in range(steps):
        action = sim.get_action()
        sim.apply_action(action)
        error = abs(sim.current_temp - sim.target_temp)
        errors.append(error)
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    final_error = errors[-1]
    
    print(f'{name}:')
    print(f'  平均误差: {avg_error:.3f}°C')
    print(f'  最大误差: {max_error:.3f}°C')
    print(f'  最终误差: {final_error:.3f}°C')
    print(f'  最终温度: {sim.current_temp:.2f}°C')
    print()
    return avg_error, final_error

print('=== 温度控制算法效果对比测试 ===')
print(f'目标温度: 25°C, 初始温度: 22°C, 测试步数: 50')
print('=' * 50)
print()

results = []
results.append(('规则控制', test_simulator(RuleBasedSimulator(), '规则控制')))
results.append(('安全强化学习', test_simulator(SafeRLSimulator(), '安全强化学习')))
results.append(('约束强化学习', test_simulator(ConstrainedRLSimulator(), '约束强化学习')))
results.append(('MBPO (模型预测)', test_simulator(MBPO_Simulator(), 'MBPO (模型预测)')))
results.append(('层级强化学习', test_simulator(HierarchicalRLSimulator(), '层级强化学习')))
results.append(('RL+MPC 混合控制', test_simulator(RLMPCSimulator(), 'RL+MPC 混合控制')))

print('=' * 50)
print('算法排名 (按平均误差):')
results.sort(key=lambda x: x[1][0])
for i, (name, (avg_err, final_err)) in enumerate(results, 1):
    print(f'{i}. {name}: 平均误差 {avg_err:.3f}°C, 最终误差 {final_err:.3f}°C')
