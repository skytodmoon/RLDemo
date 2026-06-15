from flask import Flask, render_template, jsonify, request
import numpy as np
import threading
import time

app = Flask(__name__)

class BaseSimulator:
    def __init__(self):
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.target_temp = 25.0
        self.current_temp = 25.0
        self.heater_on = False
        self.cooler_on = False
        self.step_count = 0
    
    def reset(self):
        self.current_temp = np.random.uniform(22, 28)
        self.heater_on = False
        self.cooler_on = False
        self.step_count = 0
        return self.current_temp
    
    def apply_action(self, action):
        self.heater_on = False
        self.cooler_on = False
        
        delta_temp = -0.1
        
        if action == 'heat':
            self.heater_on = True
            delta_temp += 0.3
        elif action == 'cool':
            self.cooler_on = True
            delta_temp -= 0.2
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        self.step_count += 1
        
        return self.current_temp, action

class RuleBasedSimulator(BaseSimulator):
    def get_action(self):
        temp_error = self.current_temp - self.target_temp
        if temp_error < -1.0:
            return 'heat'
        elif temp_error > 1.0:
            return 'cool'
        else:
            return 'idle'

class SafeRLSimulator(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.safety_margin = 2.0
    
    def get_action(self):
        temp_error = self.current_temp - self.target_temp
        
        if self.current_temp <= self.temp_min + 1.0:
            return 'heat' if temp_error < 0 else 'idle'
        if self.current_temp >= self.temp_max - 1.0:
            return 'cool' if temp_error > 0 else 'idle'
        
        if temp_error < -1.0:
            return 'heat'
        elif temp_error > 1.0:
            return 'cool'
        else:
            return 'idle'

class ConstrainedRLSimulator(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.constraint_weight = 0.5
    
    def get_action(self):
        temp_error = self.current_temp - self.target_temp
        distance_to_boundary = min(
            self.current_temp - self.temp_min,
            self.temp_max - self.current_temp
        )
        
        if distance_to_boundary < 2.0:
            if self.current_temp < self.target_temp:
                return 'heat' if distance_to_boundary > 1.0 else 'idle'
            else:
                return 'cool' if distance_to_boundary > 1.0 else 'idle'
        
        if temp_error < -1.0:
            return 'heat'
        elif temp_error > 1.0:
            return 'cool'
        else:
            return 'idle'

class MBPO_Simulator(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.horizon = 5
    
    def predict_next_temp(self, temp, action):
        delta = -0.1
        if action == 'heat':
            delta += 0.3
        elif action == 'cool':
            delta -= 0.2
        return max(self.temp_min, min(self.temp_max, temp + delta))
    
    def get_action(self):
        best_action = 'idle'
        best_reward = float('-inf')
        
        for action in ['heat', 'cool', 'idle']:
            temp = self.current_temp
            total_reward = 0
            
            for _ in range(self.horizon):
                temp = self.predict_next_temp(temp, action)
                error = abs(temp - self.target_temp)
                reward = 10.0 - error * 2
                if temp <= self.temp_min or temp >= self.temp_max:
                    reward -= 50.0
                total_reward += reward * (0.99 ** _)
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action
        
        return best_action

class HierarchicalRLSimulator(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.current_goal = 'maintain'
    
    def get_action(self):
        temp_error = self.current_temp - self.target_temp
        
        if abs(temp_error) > 3.0:
            self.current_goal = 'approach'
        elif abs(temp_error) < 0.5:
            self.current_goal = 'maintain'
        
        if self.current_goal == 'approach':
            if temp_error < 0:
                return 'heat'
            else:
                return 'cool'
        else:
            if temp_error < -0.5:
                return 'heat'
            elif temp_error > 0.5:
                return 'cool'
            else:
                return 'idle'

class RLMPCSimulator(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.rl_weight = 0.3
        self.mpc_horizon = 3
    
    def get_rl_action(self):
        temp_error = self.current_temp - self.target_temp
        if temp_error < -1.0:
            return 'heat'
        elif temp_error > 1.0:
            return 'cool'
        else:
            return 'idle'
    
    def get_mpc_action(self):
        best_action = 'idle'
        best_reward = float('-inf')
        
        for action in ['heat', 'cool', 'idle']:
            temp = self.current_temp
            total_reward = 0
            
            for _ in range(self.mpc_horizon):
                delta = -0.1
                if action == 'heat':
                    delta += 0.3
                elif action == 'cool':
                    delta -= 0.2
                temp = max(self.temp_min, min(self.temp_max, temp + delta))
                
                error = abs(temp - self.target_temp)
                reward = 10.0 - error * 2
                total_reward += reward
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action
        
        return best_action
    
    def get_action(self):
        rl_action = self.get_rl_action()
        mpc_action = self.get_mpc_action()
        
        if np.random.random() < self.rl_weight:
            return rl_action
        else:
            return mpc_action

simulators = {
    'rule_based': RuleBasedSimulator(),
    'safe_rl': SafeRLSimulator(),
    'constrained_rl': ConstrainedRLSimulator(),
    'mbpo': MBPO_Simulator(),
    'hierarchical_rl': HierarchicalRLSimulator(),
    'rl_mpc': RLMPCSimulator()
}

current_simulator = simulators['rule_based']

pinns_solver = None
pinns_history = []

def init_pinns():
    global pinns_solver
    from pinns_heat_eq import HeatEquationSolver
    pinns_solver = HeatEquationSolver()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pinns')
def pinns():
    return render_template('pinns.html')

# 安全强化学习页面
@app.route('/safe-rl')
def safe_rl():
    return render_template('safe_rl.html')

# 约束强化学习页面
@app.route('/constrained-rl')
def constrained_rl():
    return render_template('constrained_rl.html')

# MBPO页面
@app.route('/mbpo')
def mbpo():
    return render_template('mbpo.html')

# 层级强化学习页面
@app.route('/hierarchical-rl')
def hierarchical_rl():
    return render_template('hierarchical_rl.html')

# RL+MPC页面
@app.route('/rl-mpc')
def rl_mpc():
    return render_template('rl_mpc.html')

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global current_simulator
    data = request.json
    algorithm = data.get('algorithm', 'rule_based')
    current_simulator = simulators.get(algorithm, simulators['rule_based'])
    initial_temp = current_simulator.reset()
    return jsonify({'status': 'success', 'initial_temp': round(initial_temp, 2)})

@app.route('/api/step', methods=['POST'])
def api_step():
    global current_simulator
    data = request.json
    algorithm = data.get('algorithm', 'rule_based')
    
    if simulators.get(algorithm) != current_simulator:
        current_simulator = simulators[algorithm]
    
    action = current_simulator.get_action()
    temp, action = current_simulator.apply_action(action)
    
    return jsonify({
        'step': current_simulator.step_count,
        'temp': round(temp, 2),
        'target': current_simulator.target_temp,
        'heater': current_simulator.heater_on,
        'cooler': current_simulator.cooler_on,
        'action': action
    })

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    global current_simulator
    data = request.json
    steps = data.get('steps', 50)
    algorithm = data.get('algorithm', 'rule_based')
    
    current_simulator = simulators.get(algorithm, simulators['rule_based'])
    current_simulator.reset()
    
    history = []
    for _ in range(steps):
        action = current_simulator.get_action()
        temp, action = current_simulator.apply_action(action)
        history.append({
            'step': current_simulator.step_count,
            'temp': round(temp, 2),
            'target': current_simulator.target_temp,
            'heater': current_simulator.heater_on,
            'cooler': current_simulator.cooler_on,
            'action': action
        })
    
    return jsonify({'status': 'success', 'history': history})

# 安全强化学习 API
@app.route('/api/safe-rl/reset', methods=['POST'])
def api_safe_rl_reset():
    sim = simulators['safe_rl']
    initial_temp = sim.reset()
    return jsonify({
        'temperature': round(initial_temp, 2),
        'target': sim.target_temp,
        'action': 'idle',
        'reward': 0,
        'safety_margin': sim.safety_margin,
        'in_danger_zone': False
    })

@app.route('/api/safe-rl/step', methods=['POST'])
def api_safe_rl_step():
    sim = simulators['safe_rl']
    action = sim.get_action()
    temp, _ = sim.apply_action(action)
    
    temp_error = abs(temp - sim.target_temp)
    reward = 10.0 - temp_error * 2
    
    # Check if in danger zone
    in_danger_zone = (temp <= sim.temp_min + sim.safety_margin or 
                      temp >= sim.temp_max - sim.safety_margin)
    
    return jsonify({
        'temperature': round(temp, 2),
        'target': sim.target_temp,
        'action': action,
        'final_action': action,
        'reward': round(reward, 2),
        'safety_margin': sim.safety_margin,
        'in_danger_zone': in_danger_zone
    })

# 约束强化学习 API
@app.route('/api/constrained-rl/reset', methods=['POST'])
def api_constrained_rl_reset():
    sim = simulators['constrained_rl']
    sim.reset()
    return jsonify({
        'temperature': round(sim.current_temp, 2),
        'target': sim.target_temp,
        'action': 'idle',
        'reward': 0,
        'constraint_weight': sim.constraint_weight,
        'distance_to_boundary': round(min(sim.current_temp - sim.temp_min, sim.temp_max - sim.current_temp), 2)
    })

@app.route('/api/constrained-rl/step', methods=['POST'])
def api_constrained_rl_step():
    sim = simulators['constrained_rl']
    action = sim.get_action()
    temp, _ = sim.apply_action(action)
    
    temp_error = abs(temp - sim.target_temp)
    reward = 10.0 - temp_error * 2
    
    distance_to_boundary = min(temp - sim.temp_min, sim.temp_max - temp)
    violation = 1.0 if distance_to_boundary < 2.0 else 0.0
    
    return jsonify({
        'temperature': round(temp, 2),
        'target': sim.target_temp,
        'action': action,
        'reward': round(reward, 2),
        'constraint_weight': sim.constraint_weight,
        'distance_to_boundary': round(distance_to_boundary, 2),
        'violation': violation
    })

# MBPO API
@app.route('/api/mbpo/reset', methods=['POST'])
def api_mbpo_reset():
    sim = simulators['mbpo']
    initial_temp = sim.reset()
    
    # Generate predictions for each action
    predictions = {'heat': [], 'cool': [], 'idle': []}
    for action in ['heat', 'cool', 'idle']:
        temp = initial_temp
        for _ in range(sim.horizon):
            temp = sim.predict_next_temp(temp, action)
            predictions[action].append(round(temp, 1))
    
    return jsonify({
        'temperature': round(initial_temp, 2),
        'target': sim.target_temp,
        'action': 'idle',
        'reward': 0,
        'predicted_value': 0,
        'predictions': predictions
    })

@app.route('/api/mbpo/step', methods=['POST'])
def api_mbpo_step():
    sim = simulators['mbpo']
    action = sim.get_action()
    temp, _ = sim.apply_action(action)
    
    temp_error = abs(temp - sim.target_temp)
    reward = 10.0 - temp_error * 2
    
    # Generate predictions for each action
    predictions = {'heat': [], 'cool': [], 'idle': []}
    for act in ['heat', 'cool', 'idle']:
        t = temp
        for _ in range(sim.horizon):
            t = sim.predict_next_temp(t, act)
            predictions[act].append(round(t, 1))
    
    return jsonify({
        'temperature': round(temp, 2),
        'target': sim.target_temp,
        'action': action,
        'reward': round(reward, 2),
        'predicted_value': round(reward * sim.horizon, 2),
        'predictions': predictions
    })

# 层级强化学习 API
@app.route('/api/hierarchical-rl/reset', methods=['POST'])
def api_hierarchical_rl_reset():
    sim = simulators['hierarchical_rl']
    initial_temp = sim.reset()
    sim.current_goal = 'maintain'
    
    return jsonify({
        'temperature': round(initial_temp, 2),
        'target': sim.target_temp,
        'action': 'idle',
        'option': sim.current_goal,
        'reward': 0
    })

@app.route('/api/hierarchical-rl/step', methods=['POST'])
def api_hierarchical_rl_step():
    sim = simulators['hierarchical_rl']
    action = sim.get_action()
    temp, _ = sim.apply_action(action)
    
    temp_error = abs(temp - sim.target_temp)
    reward = 10.0 - temp_error * 2
    
    return jsonify({
        'temperature': round(temp, 2),
        'target': sim.target_temp,
        'action': action,
        'option': sim.current_goal,
        'reward': round(reward, 2)
    })

# RL+MPC API
@app.route('/api/rl-mpc/reset', methods=['POST'])
def api_rl_mpc_reset():
    sim = simulators['rl_mpc']
    initial_temp = sim.reset()
    rl_action = sim.get_rl_action()
    mpc_action = sim.get_mpc_action()
    
    temp_error = abs(initial_temp - sim.target_temp)
    uncertainty = min(temp_error / 10.0, 1.0)
    
    return jsonify({
        'temperature': round(initial_temp, 2),
        'target': sim.target_temp,
        'action': 'idle',
        'rl_action': rl_action,
        'mpc_action': mpc_action,
        'final_action': 'idle',
        'source': 'RL',
        'rl_weight': sim.rl_weight,
        'uncertainty': round(uncertainty, 3),
        'reward': 0
    })

@app.route('/api/rl-mpc/step', methods=['POST'])
def api_rl_mpc_step():
    sim = simulators['rl_mpc']
    action = sim.get_action()
    temp, _ = sim.apply_action(action)
    
    rl_action = sim.get_rl_action()
    mpc_action = sim.get_mpc_action()
    source = 'RL' if action == rl_action else 'MPC'
    
    temp_error = abs(temp - sim.target_temp)
    uncertainty = min(temp_error / 10.0, 1.0)
    reward = 10.0 - temp_error * 2
    
    return jsonify({
        'temperature': round(temp, 2),
        'target': sim.target_temp,
        'action': action,
        'rl_action': rl_action,
        'mpc_action': mpc_action,
        'final_action': action,
        'source': source,
        'rl_weight': sim.rl_weight,
        'uncertainty': round(uncertainty, 3),
        'reward': round(reward, 2)
    })

@app.route('/api/pinns/train', methods=['POST'])
def api_pinns_train():
    global pinns_history
    epochs = request.json.get('epochs', 1000)
    
    if pinns_solver is None:
        init_pinns()
    
    pinns_history = []
    
    for epoch in range(epochs):
        x_pde = torch.rand(1000, 1).to(pinns_solver.device)
        t_pde = torch.rand(1000, 1).to(pinns_solver.device)
        
        x_bc = torch.rand(200, 1).to(pinns_solver.device)
        t_bc = torch.rand(200, 1).to(pinns_solver.device)
        x_bc_0 = torch.zeros(100, 1).to(pinns_solver.device)
        x_bc_1 = torch.ones(100, 1).to(pinns_solver.device)
        x_bc = torch.cat([x_bc_0, x_bc_1], dim=0)
        
        x_ic = torch.rand(200, 1).to(pinns_solver.device)
        t_ic = torch.zeros(200, 1).to(pinns_solver.device)
        
        total_loss, pde_loss, bc_loss, ic_loss = pinns_solver.loss_fn(x_pde, t_pde, x_bc, t_bc, x_ic, t_ic)
        
        pinns_solver.optimizer.zero_grad()
        total_loss.backward()
        pinns_solver.optimizer.step()
        
        if epoch % 10 == 0:
            pinns_history.append({
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'pde_loss': pde_loss,
                'bc_loss': bc_loss,
                'ic_loss': ic_loss
            })
    
    return jsonify({'status': 'success', 'history': pinns_history})

@app.route('/api/pinns/predict', methods=['POST'])
def api_pinns_predict():
    if pinns_solver is None:
        init_pinns()
    
    data = request.json
    x = np.array(data['x'])
    t = np.array(data['t'])
    
    u_pred = pinns_solver.predict(x, t)
    return jsonify({'u': u_pred.tolist()})

@app.route('/api/pinns/generate_data', methods=['POST'])
def api_pinns_generate_data():
    if pinns_solver is None:
        init_pinns()
    
    steps = request.json.get('steps', 50)
    x = np.linspace(0, 1, 100).tolist()
    times = np.linspace(0, 1, steps).tolist()
    
    results = []
    for t in times:
        u_pred = pinns_solver.predict(np.linspace(0, 1, 100), np.full(100, t))
        results.append({
            't': round(t, 3),
            'x': x,
            'u': u_pred.tolist()
        })
    
    return jsonify({'data': results})

import torch

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)