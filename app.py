from flask import Flask, render_template, jsonify, request
import numpy as np
import threading
import time

app = Flask(__name__)

class TempControlSim:
    def __init__(self):
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.target_temp = 25.0
        self.current_temp = 25.0
        self.heater_on = False
        self.cooler_on = False
        self.running = False
        self.history = []
        self.step_count = 0
    
    def reset(self):
        self.current_temp = np.random.uniform(22, 28)
        self.heater_on = False
        self.cooler_on = False
        self.history = []
        self.step_count = 0
    
    def get_action(self):
        temp_error = self.current_temp - self.target_temp
        if temp_error < -1.0:
            return 'heat'
        elif temp_error > 1.0:
            return 'cool'
        else:
            return 'idle'
    
    def step(self):
        action = self.get_action()
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
        
        self.history.append({
            'step': self.step_count,
            'temp': round(self.current_temp, 2),
            'target': self.target_temp,
            'heater': self.heater_on,
            'cooler': self.cooler_on,
            'action': action
        })
        
        return self.current_temp, action

sim = TempControlSim()

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

@app.route('/api/reset', methods=['POST'])
def api_reset():
    sim.reset()
    return jsonify({'status': 'success', 'initial_temp': round(sim.current_temp, 2)})

@app.route('/api/step', methods=['POST'])
def api_step():
    temp, action = sim.step()
    return jsonify({
        'step': sim.step_count,
        'temp': round(temp, 2),
        'target': sim.target_temp,
        'heater': sim.heater_on,
        'cooler': sim.cooler_on,
        'action': action
    })

@app.route('/api/history')
def api_history():
    return jsonify(sim.history)

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    steps = request.json.get('steps', 50)
    sim.reset()
    for _ in range(steps):
        sim.step()
    return jsonify({'status': 'success', 'history': sim.history})

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