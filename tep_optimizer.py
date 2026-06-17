#!/usr/bin/env python3
"""
TEP SAC Controller Optimizer
=============================

Automatic optimization based on evaluation results.
This module provides recommendations and automatic adjustments
to improve TEP SAC controller performance.

Optimization Strategies:
1. Control Performance Optimization
2. Reward Function Tuning
3. Action Space Adjustment
4. Safety Constraint Adjustment
5. Learning Rate Adaptation
"""

import numpy as np
from typing import Dict, List, Any
from tep_evaluator import TEPEvaluationResult


class OptimizationRecommendation:
    """Container for optimization recommendations."""
    def __init__(self):
        self.strategy = ""
        self.parameters = {}
        self.expected_improvement = {}
        self.description = ""
        self.priority = "medium"  # high, medium, low


class TEPOptimizer:
    """Optimizer for TEP SAC controller based on evaluation results."""
    
    def __init__(self):
        # Default parameters (can be adjusted)
        self.default_params = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'hidden_dim': 256,
            'batch_size': 128,
            'buffer_size': 50000,
            'target_entropy': -6,  # -action_dim
        }
        
        # Reward weights for different objectives
        self.reward_weights = {
            'tracking': 0.3,
            'quality': 0.3,
            'safety': 0.2,
            'efficiency': 0.1,
            'profit': 0.1
        }
    
    def analyze_evaluation(self, result: TEPEvaluationResult) -> List[OptimizationRecommendation]:
        """
        Analyze evaluation results and generate optimization recommendations.
        
        Args:
            result: TEPEvaluationResult object
            
        Returns:
            List of OptimizationRecommendation objects
        """
        recommendations = []
        
        # Analyze each metric and generate recommendations
        if result.scores['control'] < 70:
            rec = self._optimize_control_performance(result)
            if rec:
                recommendations.append(rec)
        
        if result.scores['quality'] < 80:
            rec = self._optimize_product_quality(result)
            if rec:
                recommendations.append(rec)
        
        if result.scores['safety'] < 90:
            rec = self._optimize_safety(result)
            if rec:
                recommendations.append(rec)
        
        if result.scores['efficiency'] < 70:
            rec = self._optimize_energy_efficiency(result)
            if rec:
                recommendations.append(rec)
        
        if result.scores['economic'] < 70:
            rec = self._optimize_economic_performance(result)
            if rec:
                recommendations.append(rec)
        
        # Sort by priority
        recommendations.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x.priority])
        
        return recommendations
    
    def _optimize_control_performance(self, result: TEPEvaluationResult) -> OptimizationRecommendation:
        """Generate recommendations for control performance improvement."""
        rec = OptimizationRecommendation()
        rec.strategy = "control_tuning"
        rec.priority = "high"
        
        metrics = result.metrics['control']
        issues = []
        params = {}
        
        # Analyze specific control issues
        if metrics.get('tracking_error', 0) > 0.15:
            issues.append("跟踪误差过大")
            params['learning_rate'] = self.default_params['learning_rate'] * 1.5
            params['tau'] = min(0.01, self.default_params['tau'] * 2)
        
        if metrics.get('iae', 0) > 10:
            issues.append("积分误差过大")
            params['gamma'] = min(0.995, self.default_params['gamma'] + 0.003)
        
        if metrics.get('max_overshoot', 0) > 0.3:
            issues.append("超调量过大")
            params['alpha'] = max(0.1, self.default_params['alpha'] * 0.7)
        
        if metrics.get('smoothness', 1) < 0.7:
            issues.append("控制动作不够平滑")
            params['hidden_dim'] = max(128, self.default_params['hidden_dim'] * 0.8)
        
        if issues:
            rec.description = f"检测到控制性能问题: {', '.join(issues)}。建议调整控制器参数以改善状态跟踪。"
            rec.parameters = params
            rec.expected_improvement = {
                'control_score': '+5-15分',
                'tracking_error': '-10-20%',
                'stability': 'improved'
            }
            return rec
        
        return None
    
    def _optimize_product_quality(self, result: TEPEvaluationResult) -> OptimizationRecommendation:
        """Generate recommendations for product quality improvement."""
        rec = OptimizationRecommendation()
        rec.strategy = "quality_optimization"
        rec.priority = "high"
        
        metrics = result.metrics['quality']
        issues = []
        params = {}
        
        if metrics.get('quality_rate', 1) < 0.95:
            issues.append("产品合格率偏低")
            params['reward_weight_quality'] = min(0.4, self.reward_weights['quality'] + 0.1)
        
        if metrics.get('avg_composition_A', 0) > 0.008:
            issues.append("成分A含量偏高")
            params['target_comp_A'] = 0.006  # Lower target
        
        if metrics.get('composition_std', 0) > 0.001:
            issues.append("成分波动较大")
            params['reward_weight_tracking'] = min(0.4, self.reward_weights['tracking'] + 0.1)
        
        if issues:
            rec.description = f"检测到产品质量问题: {', '.join(issues)}。建议调整奖励函数权重和控制目标。"
            rec.parameters = params
            rec.expected_improvement = {
                'quality_score': '+3-10分',
                'quality_rate': '>98%',
                'composition_stability': 'improved'
            }
            return rec
        
        return None
    
    def _optimize_safety(self, result: TEPEvaluationResult) -> OptimizationRecommendation:
        """Generate recommendations for safety improvement."""
        rec = OptimizationRecommendation()
        rec.strategy = "safety_enhancement"
        rec.priority = "high" if result.scores['safety'] < 70 else "medium"
        
        metrics = result.metrics['safety']
        issues = []
        params = {}
        
        if metrics.get('violation_rate', 0) > 0.1:
            issues.append("安全违规率较高")
            params['reward_weight_safety'] = min(0.3, self.reward_weights['safety'] + 0.1)
            params['safety_penalty_factor'] = 2.0
        
        if metrics.get('emergency_shutdowns', 0) > 0:
            issues.append("发生紧急停机")
            params['constraint_margin'] = 0.15  # Increase margin
        
        if metrics.get('avg_safety_margin', 0) < 0.1:
            issues.append("安全裕度过小")
            params['constraint_margin'] = 0.12
        
        if issues:
            rec.description = f"检测到安全问题: {', '.join(issues)}。建议加强安全约束和惩罚机制。"
            rec.parameters = params
            rec.expected_improvement = {
                'safety_score': '+5-20分',
                'violation_rate': '<5%',
                'safety_margin': '>15%'
            }
            return rec
        
        return None
    
    def _optimize_energy_efficiency(self, result: TEPEvaluationResult) -> OptimizationRecommendation:
        """Generate recommendations for energy efficiency improvement."""
        rec = OptimizationRecommendation()
        rec.strategy = "efficiency_optimization"
        rec.priority = "medium"
        
        metrics = result.metrics['efficiency']
        issues = []
        params = {}
        
        if metrics.get('energy_per_unit', 0) > 3.0:
            issues.append("单位产品能耗偏高")
            params['reward_weight_efficiency'] = min(0.2, self.reward_weights['efficiency'] + 0.05)
            params['energy_penalty_factor'] = 1.5
        
        if metrics.get('avg_production', 0) < 20:
            issues.append("产量偏低")
            params['reward_weight_profit'] = min(0.15, self.reward_weights['profit'] + 0.05)
        
        if issues:
            rec.description = f"检测到能效问题: {', '.join(issues)}。建议调整能耗相关奖励权重。"
            rec.parameters = params
            rec.expected_improvement = {
                'efficiency_score': '+5-10分',
                'energy_per_unit': '-10-15%',
                'production': '+5-10%'
            }
            return rec
        
        return None
    
    def _optimize_economic_performance(self, result: TEPEvaluationResult) -> OptimizationRecommendation:
        """Generate recommendations for economic performance improvement."""
        rec = OptimizationRecommendation()
        rec.strategy = "economic_tuning"
        rec.priority = "medium"
        
        metrics = result.metrics['economic']
        issues = []
        params = {}
        
        if metrics.get('avg_reward', 0) < 50:
            issues.append("平均奖励偏低")
            params['reward_scaling'] = 1.2
            params['gamma'] = min(0.997, self.default_params['gamma'] + 0.005)
        
        if metrics.get('reward_std', 0) > 5:
            issues.append("奖励波动较大")
            params['target_entropy'] = self.default_params['target_entropy'] * 0.8
        
        if issues:
            rec.description = f"检测到经济效益问题: {', '.join(issues)}。建议调整奖励函数和折扣因子。"
            rec.parameters = params
            rec.expected_improvement = {
                'economic_score': '+5-10分',
                'avg_reward': '+10-20%',
                'profit': '+5-15%'
            }
            return rec
        
        return None
    
    def apply_optimization(self, agent, recommendations: List[OptimizationRecommendation]) -> Dict:
        """
        Apply optimization recommendations to the agent.
        
        Args:
            agent: SACAgent instance to optimize
            recommendations: List of OptimizationRecommendation objects
            
        Returns:
            Dictionary with applied changes
        """
        changes = {
            'applied': [],
            'skipped': [],
            'parameters_changed': {}
        }
        
        for rec in recommendations:
            try:
                if rec.strategy == 'control_tuning':
                    self._apply_control_tuning(agent, rec.parameters, changes)
                elif rec.strategy == 'quality_optimization':
                    self._apply_quality_optimization(agent, rec.parameters, changes)
                elif rec.strategy == 'safety_enhancement':
                    self._apply_safety_enhancement(agent, rec.parameters, changes)
                elif rec.strategy == 'efficiency_optimization':
                    self._apply_efficiency_optimization(agent, rec.parameters, changes)
                elif rec.strategy == 'economic_tuning':
                    self._apply_economic_tuning(agent, rec.parameters, changes)
                
                changes['applied'].append({
                    'strategy': rec.strategy,
                    'description': rec.description,
                    'expected_improvement': rec.expected_improvement
                })
            except Exception as e:
                changes['skipped'].append({
                    'strategy': rec.strategy,
                    'error': str(e)
                })
        
        return changes
    
    def _apply_control_tuning(self, agent, params, changes):
        """Apply control tuning parameters."""
        if 'learning_rate' in params:
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = params['learning_rate']
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] = params['learning_rate']
            changes['parameters_changed']['learning_rate'] = params['learning_rate']
        
        if 'tau' in params:
            agent.tau = params['tau']
            changes['parameters_changed']['tau'] = params['tau']
        
        if 'gamma' in params:
            agent.gamma = params['gamma']
            changes['parameters_changed']['gamma'] = params['gamma']
        
        if 'alpha' in params:
            agent.log_alpha.data.fill_(np.log(params['alpha']))
            changes['parameters_changed']['alpha'] = params['alpha']
    
    def _apply_quality_optimization(self, agent, params, changes):
        """Apply quality optimization parameters."""
        if hasattr(agent, 'reward_weights'):
            if 'reward_weight_quality' in params:
                agent.reward_weights['quality'] = params['reward_weight_quality']
                changes['parameters_changed']['reward_weight_quality'] = params['reward_weight_quality']
            if 'reward_weight_tracking' in params:
                agent.reward_weights['tracking'] = params['reward_weight_tracking']
                changes['parameters_changed']['reward_weight_tracking'] = params['reward_weight_tracking']
        
        if 'target_comp_A' in params:
            changes['parameters_changed']['target_comp_A'] = params['target_comp_A']
    
    def _apply_safety_enhancement(self, agent, params, changes):
        """Apply safety enhancement parameters."""
        if hasattr(agent, 'reward_weights'):
            if 'reward_weight_safety' in params:
                agent.reward_weights['safety'] = params['reward_weight_safety']
                changes['parameters_changed']['reward_weight_safety'] = params['reward_weight_safety']
        
        if hasattr(agent, 'safety_penalty_factor'):
            if 'safety_penalty_factor' in params:
                agent.safety_penalty_factor = params['safety_penalty_factor']
                changes['parameters_changed']['safety_penalty_factor'] = params['safety_penalty_factor']
    
    def _apply_efficiency_optimization(self, agent, params, changes):
        """Apply efficiency optimization parameters."""
        if hasattr(agent, 'reward_weights'):
            if 'reward_weight_efficiency' in params:
                agent.reward_weights['efficiency'] = params['reward_weight_efficiency']
                changes['parameters_changed']['reward_weight_efficiency'] = params['reward_weight_efficiency']
            if 'reward_weight_profit' in params:
                agent.reward_weights['profit'] = params['reward_weight_profit']
                changes['parameters_changed']['reward_weight_profit'] = params['reward_weight_profit']
    
    def _apply_economic_tuning(self, agent, params, changes):
        """Apply economic tuning parameters."""
        if 'gamma' in params:
            agent.gamma = params['gamma']
            changes['parameters_changed']['gamma'] = params['gamma']
        
        if 'target_entropy' in params:
            agent.target_entropy = params['target_entropy']
            changes['parameters_changed']['target_entropy'] = params['target_entropy']


def generate_optimization_report(recommendations: List[OptimizationRecommendation]) -> str:
    """Generate human-readable optimization report."""
    if not recommendations:
        return "✅ 评估结果优秀，暂无需优化建议。"
    
    report = []
    report.append("=" * 60)
    report.append("TEP SAC 控制器优化建议报告")
    report.append("=" * 60)
    report.append("")
    
    for i, rec in enumerate(recommendations, 1):
        priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[rec.priority]
        report.append(f"{priority_color} 优化策略 #{i}: {rec.strategy}")
        report.append(f"   优先级: {rec.priority}")
        report.append(f"   描述: {rec.description}")
        report.append(f"   建议调整参数: {rec.parameters}")
        report.append(f"   预期改进: {rec.expected_improvement}")
        report.append("")
    
    return "\n".join(report)
