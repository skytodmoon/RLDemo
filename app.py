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

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)