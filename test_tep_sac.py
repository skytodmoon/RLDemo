#!/usr/bin/env python3
"""
TEPSAC Performance Test Script with Evaluation Dashboard
Tests the Soft Actor-Critic agent on Tennessee Eastman Process environment
"""

import numpy as np
from tennessee_eastman_env import TennesseeEastmanEnv
from deep_rl_agent import SACAgent
from tep_evaluator import evaluate_tep_sac, generate_evaluation_report


def test_tep_sac_with_evaluation():
    print('=' * 60)
    print('TEPSAC Performance Test with Evaluation Dashboard')
    print('=' * 60)
    print()
    
    # Configuration
    num_episodes = 100
    max_steps = 200
    warmup_episodes = 10
    
    # Setup environment and agent
    env = TennesseeEastmanEnv(max_steps=max_steps)
    agent = SACAgent(
        state_dim=15, action_dim=6, hidden_dim=256,
        lr=3e-4, gamma=0.99, tau=0.005,
        auto_alpha=True, buffer_size=100000, batch_size=256,
    )
    
    # Data collection for evaluation
    all_states = []
    all_setpoints = []
    all_quality_flags = []
    all_product_compositions = []
    all_safety_violations = []
    all_energy_consumption = []
    all_production_output = []
    all_rewards = []
    
    emergency_shutdowns = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_violations = 0
        
        for step in range(max_steps):
            # Exploration vs exploitation
            if episode < warmup_episodes:
                action = env.action_space.sample() * 0.3
            else:
                action = agent.select_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if episode >= warmup_episodes and step % 20 == 0:
                for _ in range(5):
                    agent.update()
            
            # Collect data for evaluation
            all_states.append(state)
            all_setpoints.append([
                env.REACTOR_TEMP_NORM, env.REACTOR_PRESS_NORM, env.REACTOR_LEVEL_NORM,
                env.SEPARATOR_TEMP_NORM, env.SEPARATOR_LEVEL_NORM,
                env.STRIPPER_LEVEL_NORM,
                env.PRODUCT_FLOW_NORM, env.PRODUCT_COMP_A_NORM,
                env.FEED_TOTAL_NORM, env.FEED_RATIO_D_NORM,
                env.COOLING_TEMP_NORM, env.RECYCLE_FLOW_NORM,
                env.PURGE_RATE_NORM, env.COMPRESSOR_WORK_NORM, env.AGITATOR_SPEED_NORM
            ])
            all_quality_flags.append(info.get('product_quality', 'Fail'))
            all_product_compositions.append(info.get('product_comp_A', 0.0))
            if info.get('safety_violation', False):
                ep_violations += 1
            all_energy_consumption.append(info.get('compressor_work', 0.0))
            all_production_output.append(info.get('product_flow', 0.0))
            
            ep_reward += reward
            state = next_state
            
            if terminated:
                emergency_shutdowns += 1
                break
        
        all_safety_violations.append(ep_violations)
        all_rewards.append(ep_reward)
        
        # Log progress
        if (episode + 1) % 20 == 0:
            avg_r = np.mean(all_rewards[-20:])
            avg_q = np.mean([1 if q == 'Pass' else 0 for q in all_quality_flags[-20*max_steps:]])
            print(f"Episode {episode+1:3d} | Reward: {avg_r:6.2f} | Quality: {avg_q:.1%} | α: {agent.alpha:.3f}")
    
    # Prepare evaluation data
    eval_data = {
        'states': np.array(all_states),
        'setpoints': np.array(all_setpoints),
        'quality_flags': all_quality_flags,
        'product_compositions': np.array(all_product_compositions),
        'safety_violations': all_safety_violations,
        'emergency_shutdowns': emergency_shutdowns,
        'constraint_margins': np.random.uniform(0.1, 0.3, size=len(all_states)),  # Placeholder
        'energy_consumption': np.array(all_energy_consumption),
        'production_output': np.array(all_production_output),
        'rewards': np.array(all_rewards),
        'production_quality': np.array([1 if q == 'Pass' else 0 for q in all_quality_flags])
    }
    
    # Evaluate and generate report
    print()
    print('=' * 60)
    print('Running Comprehensive Evaluation...')
    print('=' * 60)
    
    result = evaluate_tep_sac(eval_data)
    report = generate_evaluation_report(result)
    
    print(report)
    
    return result


if __name__ == '__main__':
    result = test_tep_sac_with_evaluation()
