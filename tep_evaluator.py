#!/usr/bin/env python3
"""
TEP SAC Controller Performance Evaluator
========================================

Comprehensive evaluation metrics for Tennessee Eastman Process control.

Key Metrics:
1. Control Performance (稳定性、跟踪精度、响应速度)
2. Product Quality (产品合格率、成分精度)
3. Safety Compliance (安全约束遵守情况)
4. Energy Efficiency (能耗效率)
5. Economic Performance (经济效益)

Evaluation Grades:
- S (Excellent): 90-100分
- A (Good): 80-89分
- B (Fair): 70-79分
- C (Poor): 60-69分
- D (Fail): <60分

Optimization:
- Integrated optimizer provides automatic recommendations based on evaluation results
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import namedtuple

# Lazy import optimizer to avoid circular imports
_optimizer = None

def get_optimizer():
    global _optimizer
    if _optimizer is None:
        from tep_optimizer import TEPOptimizer
        _optimizer = TEPOptimizer()
    return _optimizer

# Evaluation grade thresholds
GRADE_THRESHOLDS = {
    'S': 90,
    'A': 80,
    'B': 70,
    'C': 60,
    'D': 0
}

# Metric weights for overall score
METRIC_WEIGHTS = {
    'control': 0.25,
    'quality': 0.25,
    'safety': 0.25,
    'efficiency': 0.15,
    'economic': 0.10
}


class TEPEvaluationResult:
    """Container for evaluation results."""
    def __init__(self):
        self.metrics = {}
        self.scores = {}
        self.overall_score = 0.0
        self.overall_grade = 'D'
        self.details = {}
        self.optimization_recommendations = []


def evaluate_control_performance(states: np.ndarray, 
                                 setpoints: np.ndarray,
                                 dt: float = 1.0) -> Dict:
    """
    Evaluate control performance metrics.
    
    Args:
        states: Array of state observations (timesteps x state_dim)
        setpoints: Array of target setpoints (timesteps x state_dim)
        dt: Time step duration
        
    Returns:
        Dictionary containing control performance metrics
    """
    # Normalization ranges for TEP variables (from environment definition)
    # Order: reactor_temp, reactor_pressure, reactor_level,
    #        separator_temp, separator_level, stripper_level,
    #        product_flow, product_comp_A, feed_total, feed_ratio_D,
    #        cooling_water_temp, recycle_flow, purge_rate, compressor_work, agitator_speed
    ranges = np.array([
        [80.0, 160.0],    # reactor_temp (°C)
        [2000.0, 3200.0], # reactor_pressure (kPa)
        [20.0, 100.0],    # reactor_level (%)
        [50.0, 120.0],    # separator_temp (°C)
        [20.0, 100.0],    # separator_level (%)
        [20.0, 100.0],    # stripper_level (%)
        [5.0, 40.0],      # product_flow (m³/h)
        [0.0, 0.05],      # product_comp_A (mole fraction)
        [15.0, 60.0],     # feed_total (m³/h)
        [0.1, 0.6],       # feed_ratio_D
        [20.0, 55.0],     # cooling_water_temp (°C)
        [5.0, 45.0],      # recycle_flow (m³/h)
        [0.0, 2.0],       # purge_rate (%)
        [30.0, 200.0],    # compressor_work (kW)
        [100.0, 300.0],   # agitator_speed (rpm)
    ])
    
    # Normalize states and setpoints to [0, 1] range
    normalized_states = (states - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
    normalized_setpoints = (setpoints - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
    
    # Clip to [0, 1] to handle out-of-range values
    normalized_states = np.clip(normalized_states, 0, 1)
    normalized_setpoints = np.clip(normalized_setpoints, 0, 1)
    
    # Tracking error (RMSE for normalized variables)
    tracking_errors = np.sqrt(np.mean((normalized_states - normalized_setpoints) ** 2, axis=0))
    
    # Integral of Absolute Error (IAE) - normalized
    iae = np.sum(np.abs(normalized_states - normalized_setpoints), axis=0) * dt
    
    # Maximum overshoot (for oscillatory behavior)
    deviations = normalized_states - normalized_setpoints
    max_overshoot = np.max(np.abs(deviations), axis=0)
    
    # Stability margin (based on variance - lower variance = better stability)
    variances = np.std(deviations, axis=0)
    stability_margin = 1.0 / (1.0 + variances * 10)  # Scale variance
    
    # Smoothness (based on normalized state changes)
    if normalized_states.shape[0] > 1:
        state_changes = np.diff(normalized_states, axis=0)
        avg_change = np.mean(np.abs(state_changes))
        smoothness = 1.0 / (1.0 + avg_change * 5)  # Scale for better range
    else:
        smoothness = 1.0
    
    # Key performance indicators for TEP
    key_vars = ['reactor_temp', 'reactor_pressure', 'reactor_level',
                'separator_temp', 'separator_level', 'stripper_level',
                'product_flow', 'product_comp_A', 'feed_total', 'feed_ratio_D',
                'cooling_water_temp', 'recycle_flow', 'purge_rate', 
                'compressor_work', 'agitator_speed']
    
    # Overall control score (0-100)
    avg_tracking_error = np.mean(tracking_errors)
    avg_iae = np.mean(iae)
    avg_overshoot = np.mean(max_overshoot)
    
    # Normalize metrics to [0, 100]
    # Lower error = better score
    tracking_score = np.max([0, 100 - avg_tracking_error * 200])  # RMSE in [0, 1], target < 0.1 = 80+
    iae_score = np.max([0, 100 - avg_iae * 10])                    # IAE target < 5 = 50+
    stability_score = np.mean(stability_margin) * 100
    smoothness_score = smoothness * 100
    
    control_score = np.mean([tracking_score, iae_score, stability_score, smoothness_score])
    
    return {
        'tracking_error': float(avg_tracking_error),
        'iae': float(avg_iae),
        'max_overshoot': float(avg_overshoot),
        'stability_margin': float(np.mean(stability_margin)),
        'smoothness': float(smoothness),
        'tracking_score': float(tracking_score),
        'iae_score': float(iae_score),
        'stability_score': float(stability_score),
        'smoothness_score': float(smoothness_score),
        'overall': float(np.clip(control_score, 0, 100)),
        'key_vars': key_vars
    }


def evaluate_product_quality(quality_flags: List[str], 
                             product_compositions: np.ndarray) -> Dict:
    """
    Evaluate product quality metrics.
    
    Args:
        quality_flags: List of 'Pass' or 'Fail' strings
        product_compositions: Array of product composition values
        
    Returns:
        Dictionary containing quality metrics
    """
    # Quality rate
    pass_count = sum(1 for q in quality_flags if q == 'Pass')
    quality_rate = pass_count / len(quality_flags) if quality_flags else 0.0
    
    # Composition accuracy (target: comp_A < 0.01)
    if len(product_compositions) > 0:
        avg_comp_A = np.mean(product_compositions)
        comp_score = np.max([0, 100 - avg_comp_A * 1000])  # Penalize high A content
    else:
        avg_comp_A = 0.0
        comp_score = 100.0
    
    # Consistency (variance of composition)
    if len(product_compositions) > 1:
        comp_std = np.std(product_compositions)
        consistency_score = np.max([0, 100 - comp_std * 1000])
    else:
        consistency_score = 100.0
    
    # Overall quality score
    quality_score = 0.5 * quality_rate * 100 + 0.3 * comp_score + 0.2 * consistency_score
    
    return {
        'quality_rate': float(quality_rate),
        'avg_composition_A': float(avg_comp_A),
        'composition_std': float(comp_std) if len(product_compositions) > 1 else 0.0,
        'quality_rate_score': float(quality_rate * 100),
        'composition_score': float(comp_score),
        'consistency_score': float(consistency_score),
        'overall': float(np.clip(quality_score, 0, 100))
    }


def evaluate_safety_compliance(safety_violations: List[int],
                               emergency_shutdowns: int,
                               constraint_margins: np.ndarray) -> Dict:
    """
    Evaluate safety compliance metrics.
    
    Args:
        safety_violations: List of violation counts per episode
        emergency_shutdowns: Total emergency shutdown count
        constraint_margins: Array of distances to constraint boundaries
        
    Returns:
        Dictionary containing safety metrics
    """
    # Violation rate
    total_violations = sum(safety_violations)
    violation_rate = total_violations / len(safety_violations) if safety_violations else 0.0
    
    # Shutdown rate
    shutdown_rate = emergency_shutdowns / len(safety_violations) if safety_violations else 0.0
    
    # Average safety margin
    if len(constraint_margins) > 0:
        avg_margin = np.mean(constraint_margins)
        margin_score = np.min([100, avg_margin * 100])
    else:
        avg_margin = 0.0
        margin_score = 100.0  # Assume safe if no data
    
    # Safety score (penalize violations heavily)
    violation_penalty = min(violation_rate * 1000, 100)
    shutdown_penalty = min(shutdown_rate * 500, 50)
    
    safety_score = 100 - violation_penalty - shutdown_penalty + margin_score * 0.3
    safety_score = np.clip(safety_score, 0, 100)
    
    return {
        'total_violations': int(total_violations),
        'violation_rate': float(violation_rate),
        'emergency_shutdowns': int(emergency_shutdowns),
        'shutdown_rate': float(shutdown_rate),
        'avg_safety_margin': float(avg_margin),
        'overall': float(safety_score)
    }


def evaluate_energy_efficiency(energy_consumption: np.ndarray,
                               production_output: np.ndarray) -> Dict:
    """
    Evaluate energy efficiency metrics.
    
    Args:
        energy_consumption: Array of energy usage values
        production_output: Array of production rate values
        
    Returns:
        Dictionary containing efficiency metrics
    """
    if len(energy_consumption) == 0 or len(production_output) == 0:
        return {
            'avg_energy': 0.0,
            'avg_production': 0.0,
            'energy_per_unit': 0.0,
            'efficiency_index': 0.0,
            'overall': 50.0
        }
    
    # Average energy consumption
    avg_energy = np.mean(energy_consumption)
    
    # Average production
    avg_production = np.mean(production_output)
    
    # Energy per unit production (lower is better)
    if avg_production > 0:
        energy_per_unit = avg_energy / avg_production
    else:
        energy_per_unit = float('inf')
    
    # Efficiency index (normalized)
    # Target: energy per unit < 2.0 is good
    if energy_per_unit < float('inf'):
        efficiency_index = np.max([0, 1.0 - (energy_per_unit - 1.0) / 5.0])
    else:
        efficiency_index = 0.0
    
    # Overall efficiency score
    efficiency_score = efficiency_index * 100
    
    return {
        'avg_energy': float(avg_energy),
        'avg_production': float(avg_production),
        'energy_per_unit': float(energy_per_unit),
        'efficiency_index': float(efficiency_index),
        'overall': float(np.clip(efficiency_score, 0, 100))
    }


def evaluate_economic_performance(rewards: np.ndarray,
                                   production_quality: np.ndarray) -> Dict:
    """
    Evaluate economic performance metrics.
    
    Args:
        rewards: Array of episode rewards
        production_quality: Array of quality rates
        
    Returns:
        Dictionary containing economic metrics
    """
    if len(rewards) == 0:
        return {
            'avg_reward': 0.0,
            'max_reward': 0.0,
            'min_reward': 0.0,
            'reward_std': 0.0,
            'avg_profit': 0.0,
            'overall': 50.0
        }
    
    # Reward statistics
    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    reward_std = np.std(rewards)
    
    # Profit proxy (based on reward)
    # Assume reward is directly related to profit
    avg_profit = avg_reward * 100  # Scale for interpretation
    
    # Economic score
    # Normalize reward to [0, 100]
    reward_range = max_reward - min_reward if max_reward != min_reward else 1.0
    normalized_reward = (avg_reward - min_reward) / reward_range
    economic_score = normalized_reward * 100
    
    return {
        'avg_reward': float(avg_reward),
        'max_reward': float(max_reward),
        'min_reward': float(min_reward),
        'reward_std': float(reward_std),
        'avg_profit': float(avg_profit),
        'overall': float(np.clip(economic_score, 0, 100))
    }


def get_grade(score: float) -> str:
    """Get letter grade based on numerical score."""
    if score >= GRADE_THRESHOLDS['S']:
        return 'S'
    elif score >= GRADE_THRESHOLDS['A']:
        return 'A'
    elif score >= GRADE_THRESHOLDS['B']:
        return 'B'
    elif score >= GRADE_THRESHOLDS['C']:
        return 'C'
    else:
        return 'D'


def get_grade_description(grade: str) -> str:
    """Get descriptive text for grade."""
    descriptions = {
        'S': '卓越 - 控制系统表现出色，各项指标均达到优秀水平',
        'A': '优秀 - 控制系统性能良好，满足所有关键指标要求',
        'B': '良好 - 控制系统基本满足要求，有改进空间',
        'C': '及格 - 控制系统勉强合格，需要显著改进',
        'D': '不合格 - 控制系统未能达到基本要求，需全面优化'
    }
    return descriptions.get(grade, '未知')


def evaluate_tep_sac(evaluation_data: Dict) -> TEPEvaluationResult:
    """
    Comprehensive evaluation of TEP SAC controller.
    
    Args:
        evaluation_data: Dictionary containing all necessary data for evaluation
        
    Returns:
        TEPEvaluationResult object with all metrics and scores
    """
    result = TEPEvaluationResult()
    
    # Evaluate each category
    if 'states' in evaluation_data and 'setpoints' in evaluation_data:
        result.metrics['control'] = evaluate_control_performance(
            evaluation_data['states'],
            evaluation_data['setpoints']
        )
    else:
        result.metrics['control'] = {'overall': 50.0}
    
    if 'quality_flags' in evaluation_data and 'product_compositions' in evaluation_data:
        result.metrics['quality'] = evaluate_product_quality(
            evaluation_data['quality_flags'],
            evaluation_data['product_compositions']
        )
    else:
        result.metrics['quality'] = {'overall': 50.0}
    
    if 'safety_violations' in evaluation_data:
        result.metrics['safety'] = evaluate_safety_compliance(
            evaluation_data['safety_violations'],
            evaluation_data.get('emergency_shutdowns', 0),
            evaluation_data.get('constraint_margins', np.array([]))
        )
    else:
        result.metrics['safety'] = {'overall': 50.0}
    
    if 'energy_consumption' in evaluation_data and 'production_output' in evaluation_data:
        result.metrics['efficiency'] = evaluate_energy_efficiency(
            evaluation_data['energy_consumption'],
            evaluation_data['production_output']
        )
    else:
        result.metrics['efficiency'] = {'overall': 50.0}
    
    if 'rewards' in evaluation_data:
        result.metrics['economic'] = evaluate_economic_performance(
            evaluation_data['rewards'],
            evaluation_data.get('production_quality', np.array([]))
        )
    else:
        result.metrics['economic'] = {'overall': 50.0}
    
    # Calculate overall weighted score
    overall_score = 0.0
    for metric_name, weight in METRIC_WEIGHTS.items():
        overall_score += result.metrics[metric_name]['overall'] * weight
    
    result.overall_score = float(np.clip(overall_score, 0, 100))
    result.overall_grade = get_grade(result.overall_score)
    
    # Store individual scores
    result.scores = {
        'control': result.metrics['control']['overall'],
        'quality': result.metrics['quality']['overall'],
        'safety': result.metrics['safety']['overall'],
        'efficiency': result.metrics['efficiency']['overall'],
        'economic': result.metrics['economic']['overall']
    }
    
    # Generate summary details
    result.details = {
        'grade_description': get_grade_description(result.overall_grade),
        'strong_points': [],
        'weak_points': [],
        'recommendations': []
    }
    
    # Analyze strengths and weaknesses
    for metric_name, score in result.scores.items():
        metric_labels = {
            'control': '控制性能',
            'quality': '产品质量',
            'safety': '安全性',
            'efficiency': '能源效率',
            'economic': '经济效益'
        }
        if score >= 80:
            result.details['strong_points'].append(f"{metric_labels[metric_name]} ({score:.1f}分)")
        elif score < 60:
            result.details['weak_points'].append(f"{metric_labels[metric_name]} ({score:.1f}分)")
    
    # Generate recommendations
    if result.scores['control'] < 70:
        result.details['recommendations'].append("建议调整控制器参数，优化状态跟踪性能")
    if result.scores['quality'] < 70:
        result.details['recommendations'].append("建议优化产品成分控制策略")
    if result.scores['safety'] < 80:
        result.details['recommendations'].append("建议加强安全约束监测，增加安全裕度")
    if result.scores['efficiency'] < 70:
        result.details['recommendations'].append("建议优化能耗管理，降低单位产品能耗")
    if result.scores['economic'] < 70:
        result.details['recommendations'].append("建议调整奖励函数，提升经济效益")
    
    # Generate optimization recommendations using optimizer
    optimizer = get_optimizer()
    result.optimization_recommendations = optimizer.analyze_evaluation(result)
    
    # Convert optimization recommendations to details for JSON serialization
    result.details['optimization_suggestions'] = []
    for rec in result.optimization_recommendations:
        result.details['optimization_suggestions'].append({
            'strategy': rec.strategy,
            'description': rec.description,
            'parameters': rec.parameters,
            'expected_improvement': rec.expected_improvement,
            'priority': rec.priority
        })
    
    return result


def generate_evaluation_report(result: TEPEvaluationResult) -> str:
    """Generate human-readable evaluation report."""
    report = []
    report.append("=" * 60)
    report.append("TEP SAC 控制器性能评估报告")
    report.append("=" * 60)
    report.append("")
    report.append(f"综合评分: {result.overall_score:.1f} 分")
    report.append(f"综合等级: {result.overall_grade}")
    report.append(f"评价描述: {result.details['grade_description']}")
    report.append("")
    report.append("--- 分项评分 ---")
    report.append(f"  控制性能: {result.scores['control']:.1f} 分")
    report.append(f"  产品质量: {result.scores['quality']:.1f} 分")
    report.append(f"  安全性: {result.scores['safety']:.1f} 分")
    report.append(f"  能源效率: {result.scores['efficiency']:.1f} 分")
    report.append(f"  经济效益: {result.scores['economic']:.1f} 分")
    report.append("")
    
    if result.details['strong_points']:
        report.append("--- 优势分析 ---")
        for point in result.details['strong_points']:
            report.append(f"  ✓ {point}")
        report.append("")
    
    if result.details['weak_points']:
        report.append("--- 待改进项 ---")
        for point in result.details['weak_points']:
            report.append(f"  ✗ {point}")
        report.append("")
    
    if result.details['recommendations']:
        report.append("--- 优化建议 ---")
        for rec in result.details['recommendations']:
            report.append(f"  • {rec}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example evaluation
    print("TEP SAC Controller Evaluation Demo")
    print("=" * 60)
    
    # Generate synthetic evaluation data
    np.random.seed(42)
    n_steps = 100
    n_episodes = 10
    
    eval_data = {
        'states': np.random.normal(0, 0.1, size=(n_steps, 15)),
        'setpoints': np.zeros((n_steps, 15)),
        'quality_flags': ['Pass'] * n_steps,
        'product_compositions': np.random.uniform(0.002, 0.008, size=n_steps),
        'safety_violations': [0] * n_episodes,
        'emergency_shutdowns': 0,
        'constraint_margins': np.random.uniform(0.1, 0.3, size=n_steps),
        'energy_consumption': np.random.uniform(80, 120, size=n_steps),
        'production_output': np.random.uniform(20, 25, size=n_steps),
        'rewards': np.random.uniform(-15, -5, size=n_episodes)
    }
    
    # Evaluate
    result = evaluate_tep_sac(eval_data)
    
    # Print report
    print(generate_evaluation_report(result))
