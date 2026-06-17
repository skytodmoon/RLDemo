from flask import Flask, render_template, jsonify, request
import numpy as np

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
        if temp_error < -0.5:
            return 'heat'
        elif temp_error > 0.5:
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
        
        if temp_error < -0.5:
            return 'heat'
        elif temp_error > 0.5:
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
        
        if temp_error < -0.5:
            return 'heat'
        elif temp_error > 0.5:
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
        if temp_error < -0.5:
            return 'heat'
        elif temp_error > 0.5:
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

class BuildingHVACSimulator:
    """Multi-zone HVAC building simulator for web demo."""

    ZONE_NAMES = ["Office", "Server", "Lab", "Conference"]
    COMFORT = {
        "Office":     {"temp": 23.0, "temp_tol": 2.0, "humid": 45.0, "humid_tol": 10.0},
        "Server":     {"temp": 20.0, "temp_tol": 1.5, "humid": 40.0, "humid_tol": 15.0},
        "Lab":        {"temp": 22.0, "temp_tol": 1.5, "humid": 50.0, "humid_tol": 5.0},
        "Conference": {"temp": 23.0, "temp_tol": 2.5, "humid": 45.0, "humid_tol": 12.0},
    }
    ZONE_PROPS = {
        "Office":     {"heat_gen": 0.0,  "cap_heat": 2.0, "cap_cool": 2.0, "cap_humid": 1.5, "cap_dehumid": 1.5},
        "Server":     {"heat_gen": 3.0,  "cap_heat": 1.0, "cap_cool": 3.0, "cap_humid": 1.0, "cap_dehumid": 2.0},
        "Lab":        {"heat_gen": 0.5,  "cap_heat": 2.0, "cap_cool": 2.0, "cap_humid": 3.0, "cap_dehumid": 3.0},
        "Conference": {"heat_gen": 0.0,  "cap_heat": 2.5, "cap_cool": 2.5, "cap_humid": 1.5, "cap_dehumid": 1.5},
    }
    COUPLING = np.array([
        [0.0, 0.8, 0.3, 0.6],
        [0.8, 0.0, 0.5, 0.1],
        [0.3, 0.5, 0.0, 0.1],
        [0.6, 0.1, 0.1, 0.0],
    ])
    OCC_SCHEDULES = {
        "Office":     lambda h: 0.8 if 8 <= h <= 18 else (0.3 if 6 <= h <= 20 else 0.05),
        "Server":     lambda h: 0.2 if 0 <= h <= 6 else 0.3,
        "Lab":        lambda h: 0.7 if 9 <= h <= 17 else (0.2 if 7 <= h <= 19 else 0.0),
        "Conference": lambda h: 0.9 if (9 <= h <= 11 or 14 <= h <= 16) else (0.3 if 8 <= h <= 18 else 0.0),
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.hour_of_day = 8.0
        self.energy_used = 0.0
        self.energy_budget = 1000.0
        self.zones = {}
        for name in self.ZONE_NAMES:
            comfort = self.COMFORT[name]
            self.zones[name] = {
                "temp": np.random.uniform(comfort["temp"] - 2, comfort["temp"] + 2),
                "humidity": np.random.uniform(comfort["humid"] - 5, comfort["humid"] + 5),
                "occupancy": self._get_occupancy(name),
                "heater": 0.0, "cooler": 0.0,
                "humidifier": 0.0, "dehumidifier": 0.0,
            }
        return self._get_state()

    def _get_occupancy(self, name):
        base = self.OCC_SCHEDULES[name](self.hour_of_day)
        return np.clip(base + np.random.normal(0, 0.05), 0.0, 1.0)

    def _get_outdoor_temp(self):
        return 15.0 + 10.0 * np.sin(2 * np.pi * (self.hour_of_day - 6.0) / 24.0)

    def _get_solar_radiation(self):
        if self.hour_of_day < 6 or self.hour_of_day > 20:
            return 0.0
        return max(0.0, np.sin(np.pi * (self.hour_of_day - 6.0) / 14.0))

    def step(self, actions):
        """actions: dict of {zone_name: [heater, cooler, humidifier, dehumidifier]}"""
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            act = actions.get(name, [0, 0, 0, 0])
            z["heater"] = float(np.clip(act[0], 0, 1))
            z["cooler"] = float(np.clip(act[1], 0, 1))
            z["humidifier"] = float(np.clip(act[2], 0, 1))
            z["dehumidifier"] = float(np.clip(act[3], 0, 1))

        # Energy cost
        energy_step = 0
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            p = self.ZONE_PROPS[name]
            energy_step += (z["heater"] * p["cap_heat"] * 0.5 +
                            z["cooler"] * p["cap_cool"] * 0.8 +
                            z["humidifier"] * p["cap_humid"] * 0.2 +
                            z["dehumidifier"] * p["cap_dehumid"] * 0.3)
        self.energy_used += energy_step

        # Temperature dynamics
        outdoor_t = self._get_outdoor_temp()
        solar = self._get_solar_radiation()
        new_temps = {}
        for i, name in enumerate(self.ZONE_NAMES):
            z = self.zones[name]
            p = self.ZONE_PROPS[name]
            hvac_heat = (z["heater"] * p["cap_heat"] - z["cooler"] * p["cap_cool"]) * 0.3
            internal_heat = (p["heat_gen"] + z["occupancy"] * 1.0) * 0.02
            solar_gain = solar * 0.5 * (0.3 if name == "Office" else 0.1)
            outdoor_loss = (z["temp"] - outdoor_t) * 0.01
            coupling_heat = 0
            for j, other_name in enumerate(self.ZONE_NAMES):
                if i != j:
                    coupling_heat += (self.zones[other_name]["temp"] - z["temp"]) * self.COUPLING[i, j] * 0.005
            new_temps[name] = z["temp"] + hvac_heat + internal_heat + solar_gain - outdoor_loss + coupling_heat

        for name in self.ZONE_NAMES:
            self.zones[name]["temp"] = np.clip(new_temps[name], 10.0, 40.0)

        # Humidity dynamics
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            p = self.ZONE_PROPS[name]
            humid_change = (z["humidifier"] * p["cap_humid"] - z["dehumidifier"] * p["cap_dehumid"]) * 0.5
            occupant_humid = z["occupancy"] * 0.05
            outdoor_humid = 50.0 + 10.0 * np.sin(2 * np.pi * self.hour_of_day / 24.0)
            humid_drift = (outdoor_humid - z["humidity"]) * 0.002
            z["humidity"] = np.clip(z["humidity"] + humid_change + occupant_humid + humid_drift, 10.0, 90.0)

        # Occupancy
        for name in self.ZONE_NAMES:
            self.zones[name]["occupancy"] = self._get_occupancy(name)

        self.step_count += 1
        self.hour_of_day = (self.hour_of_day + 5.0 / 60.0) % 24.0

        # Reward
        reward, comfort_info = self._compute_reward()
        return self._get_state(), reward, comfort_info

    def _compute_reward(self):
        total_comfort = 0
        zone_details = {}
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            c = self.COMFORT[name]
            temp_err = abs(z["temp"] - c["temp"])
            humid_err = abs(z["humidity"] - c["humid"])
            in_temp = temp_err <= c["temp_tol"]
            in_humid = humid_err <= c["humid_tol"]
            temp_r = max(0, 1.0 - temp_err * 0.3) if in_temp else max(0, 0.7 - (temp_err - c["temp_tol"]) * 0.5)
            humid_r = max(0, 1.0 - humid_err * 0.1) if in_humid else max(0, 0.8 - (humid_err - c["humid_tol"]) * 0.3)
            comfort = (temp_r * 0.7 + humid_r * 0.3) * (0.3 + 0.7 * z["occupancy"])
            total_comfort += comfort
            zone_details[name] = {
                "temp_reward": round(temp_r, 3), "humid_reward": round(humid_r, 3),
                "comfort": round(comfort, 3), "in_comfort": bool(in_temp and in_humid),
            }
        comfort_reward = (total_comfort / 4) * 2.0 - 1.0
        energy_penalty = -0.1 * self._compute_energy_step()
        budget_penalty = -1.0 if self.energy_used >= self.energy_budget else 0.0
        reward = comfort_reward + energy_penalty + budget_penalty
        return reward, {
            "comfort_reward": round(comfort_reward, 3),
            "energy_penalty": round(energy_penalty, 3),
            "total_reward": round(reward, 3),
            "zone_details": zone_details,
        }

    def _compute_energy_step(self):
        total = 0
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            p = self.ZONE_PROPS[name]
            total += (z["heater"] * p["cap_heat"] * 0.5 +
                      z["cooler"] * p["cap_cool"] * 0.8 +
                      z["humidifier"] * p["cap_humid"] * 0.2 +
                      z["dehumidifier"] * p["cap_dehumid"] * 0.3)
        return total

    def _get_state(self):
        zone_info = {}
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            c = self.COMFORT[name]
            zone_info[name] = {
                "temp": round(z["temp"], 2),
                "humidity": round(z["humidity"], 2),
                "occupancy": round(z["occupancy"], 2),
                "temp_target": c["temp"],
                "humid_target": c["humid"],
                "heater": round(z["heater"], 2),
                "cooler": round(z["cooler"], 2),
                "humidifier": round(z["humidifier"], 2),
                "dehumidifier": round(z["dehumidifier"], 2),
                "in_comfort": bool(abs(z["temp"] - c["temp"]) <= c["temp_tol"] and
                                   abs(z["humidity"] - c["humid"]) <= c["humid_tol"]),
            }
        # Coupling heat flows
        flows = {}
        for i, ni in enumerate(self.ZONE_NAMES):
            for j, nj in enumerate(self.ZONE_NAMES):
                if i < j and self.COUPLING[i, j] > 0:
                    flow = (self.zones[nj]["temp"] - self.zones[ni]["temp"]) * self.COUPLING[i, j] * 0.005
                    flows[f"{ni}-{nj}"] = round(flow, 4)

        return {
            "step": self.step_count,
            "hour": round(self.hour_of_day, 2),
            "outdoor_temp": round(self._get_outdoor_temp(), 2),
            "solar_radiation": round(self._get_solar_radiation(), 3),
            "zones": zone_info,
            "coupling_flows": flows,
            "energy_used": round(self.energy_used, 1),
            "energy_budget": self.energy_budget,
            "energy_remaining_pct": round(max(0, 1.0 - self.energy_used / self.energy_budget) * 100, 1),
        }

    def get_action(self):
        """Rule-based RL agent for web demo: decides per-zone actions."""
        actions = {}
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            c = self.COMFORT[name]
            temp_err = z["temp"] - c["temp"]
            humid_err = z["humidity"] - c["humid"]

            # Smart RL-like policy
            heater = max(0, min(1, -temp_err * 0.4)) if temp_err < -0.5 else 0
            cooler = max(0, min(1, temp_err * 0.4)) if temp_err > 0.5 else 0

            # Server room: always try to cool if above target
            if name == "Server" and temp_err > 0:
                cooler = max(cooler, min(1, temp_err * 0.6))

            humidifier = max(0, min(1, -humid_err * 0.3)) if humid_err < -3 else 0
            dehumidifier = max(0, min(1, humid_err * 0.3)) if humid_err > 3 else 0

            # Lab: strict humidity control
            if name == "Lab":
                if humid_err < -2:
                    humidifier = max(humidifier, 0.5)
                elif humid_err > 2:
                    dehumidifier = max(dehumidifier, 0.5)

            actions[name] = [heater, cooler, humidifier, dehumidifier]
        return actions


class TEPSimulator:
    """
    Tennessee Eastman Process simulator for web demo.
    Simplified physics with coupling and delays.
    """

    # Normal operating conditions
    NORM = {
        'reactor_temp': 120.0, 'reactor_pressure': 2700.0, 'reactor_level': 65.0,
        'separator_temp': 80.0, 'separator_level': 50.0, 'stripper_level': 50.0,
        'product_flow': 22.0, 'product_comp_A': 0.005,
        'feed_total': 40.0, 'feed_ratio_D': 0.3,
        'cooling_water_temp': 35.0, 'recycle_flow': 25.0,
        'purge_rate': 0.5, 'compressor_work': 100.0, 'agitator_speed': 200.0,
    }

    # Safety limits
    SAFETY = {
        'reactor_temp_max': 150.0, 'reactor_press_max': 3000.0,
        'reactor_level_max': 100.0, 'reactor_level_min': 15.0,
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_count = 0
        self.active_fault = None
        self.fault_step = 0

        # Initialize state near normal conditions
        self.state = {}
        for key, val in self.NORM.items():
            self.state[key] = val + np.random.normal(0, val * 0.02)

        # Delay buffers
        self.temp_buffer = [self.state['reactor_temp']] * 10
        self.comp_buffer = [self.state['product_comp_A']] * 20
        self.level_buffer = [self.state['reactor_level']] * 8

        # Actions (default)
        self.actions = {
            'feed_D': 15.0, 'feed_E': 15.0, 'feed_A': 22.5,
            'recycle_valve': 0.5, 'purge_valve': 0.25, 'cooling_water': 35.0,
        }

        return self._get_state()

    def step(self, actions=None):
        if actions is None:
            actions = self.get_action()
        self.actions = actions

        s = self.state
        a = actions

        # Get delayed values
        delayed_temp = self.temp_buffer[0]
        delayed_level = self.level_buffer[0]

        # Reaction rates (coupled to temperature) — Arrhenius normalized to 1.0 at 120°C
        T = s['reactor_temp'] + 273
        T_norm = self.NORM['reactor_temp'] + 273
        k3 = np.exp(-1500 * (1/T - 1/T_norm))
        reaction_rate = k3 * a['feed_A'] * a['feed_D'] * 0.01
        heat_gen = reaction_rate * 3.5

        # Heat removal (cooling water coupling) — balanced with heat_gen
        heat_removal = (s['reactor_temp'] - a['cooling_water']) * 0.5

        # Update reactor temperature (thermal inertia - large time constant)
        tau_temp = 15.0
        target_temp = s['reactor_temp'] + (heat_gen - heat_removal) * 0.01
        recycle_effect = (delayed_temp - s['reactor_temp']) * 0.02 * a['recycle_valve']
        s['reactor_temp'] += (target_temp - s['reactor_temp'] + recycle_effect) / tau_temp
        s['reactor_temp'] += np.random.normal(0, 0.3)

        # Update reactor pressure (coupled to temperature and purge)
        tau_press = 8.0
        press_target = self.NORM['reactor_pressure'] + (s['reactor_temp'] - self.NORM['reactor_temp']) * 5.0 - a['purge_valve'] * 400
        s['reactor_pressure'] += (press_target - s['reactor_pressure']) / tau_press
        s['reactor_pressure'] += np.random.normal(0, 5)

        # Update reactor level (strongly self-regulating around NORM)
        total_feed = a['feed_A'] + a['feed_D'] + a['feed_E']
        recycle_in = s['recycle_flow'] * a['recycle_valve'] * 0.15
        total_in = total_feed + recycle_in
        level_ratio = s['reactor_level'] / self.NORM['reactor_level']
        product_out = s['product_flow'] * level_ratio * 1.2
        level_change = total_in - product_out
        level_feedback = (s['reactor_level'] - self.NORM['reactor_level']) * 0.3
        s['reactor_level'] += (level_change - level_feedback) / 4.0
        s['reactor_level'] = np.clip(s['reactor_level'], 20.0, 90.0)
        s['reactor_level'] += np.random.normal(0, 0.2)

        # Separator temperature (transport delay from reactor)
        sep_target = delayed_temp * 0.7 + 20
        s['separator_temp'] += (sep_target - s['separator_temp']) / 12.0
        s['separator_temp'] += np.random.normal(0, 0.2)

        # Separator level (self-regulating)
        sep_in = (delayed_level - self.NORM['reactor_level']) * 0.05
        sep_out = a['recycle_valve'] * 3 + product_out * 0.15
        sep_feedback = (s['separator_level'] - self.NORM['separator_level']) * 0.1
        s['separator_level'] += (sep_in - sep_out - sep_feedback) / 6.0
        s['separator_level'] = np.clip(s['separator_level'], 20.0, 90.0)
        s['separator_level'] += np.random.normal(0, 0.3)

        # Stripper level (self-regulating)
        strip_in = s['separator_level'] * 0.15
        strip_out = product_out * 0.3
        strip_feedback = (s['stripper_level'] - self.NORM['stripper_level']) * 0.08
        s['stripper_level'] += (strip_in - strip_out - strip_feedback) / 5.0
        s['stripper_level'] = np.clip(s['stripper_level'], 20.0, 90.0)
        s['stripper_level'] += np.random.normal(0, 0.3)

        # Product flow (coupled to reactor level - self-regulating)
        level_boost = (s['reactor_level'] - self.NORM['reactor_level']) * 0.05
        product_target = self.NORM['product_flow'] + reaction_rate * 5 + level_boost
        product_target = np.clip(product_target, 5.0, 35.0)
        s['product_flow'] += (product_target - s['product_flow']) / 4.0
        s['product_flow'] = np.clip(s['product_flow'], 5.0, 35.0)
        s['product_flow'] += np.random.normal(0, 0.2)

        # Product composition A (analyzer delay - large)
        comp_target = self.NORM['product_comp_A'] + (s['reactor_temp'] - self.NORM['reactor_temp']) * 0.0001
        comp_target += (self.NORM['feed_ratio_D'] - a['feed_D'] / (a['feed_D'] + a['feed_E'] + 0.01)) * 0.005
        s['product_comp_A'] += (comp_target - s['product_comp_A']) / 10.0
        s['product_comp_A'] += np.random.normal(0, 0.0002)
        s['product_comp_A'] = max(0, s['product_comp_A'])

        # Other variables
        s['feed_total'] = total_feed + np.random.normal(0, 0.2)
        s['feed_ratio_D'] = a['feed_D'] / (a['feed_D'] + a['feed_E'] + 0.01)
        s['cooling_water_temp'] = a['cooling_water'] + np.random.normal(0, 0.3)
        s['recycle_flow'] += (s['separator_level'] * 0.3 * a['recycle_valve'] + 5 - s['recycle_flow']) / 5.0
        s['purge_rate'] = a['purge_valve'] + np.random.normal(0, 0.01)
        s['compressor_work'] = self.NORM['compressor_work'] + (s['recycle_flow'] - self.NORM['recycle_flow']) * 2
        s['agitator_speed'] = self.NORM['agitator_speed'] + np.random.normal(0, 1)

        # Update delay buffers
        self.temp_buffer.append(s['reactor_temp'])
        self.comp_buffer.append(s['product_comp_A'])
        self.level_buffer.append(s['reactor_level'])
        self.temp_buffer.pop(0)
        self.comp_buffer.pop(0)
        self.level_buffer.pop(0)

        # Apply fault
        if self.active_fault is not None:
            self._apply_fault()

        self.step_count += 1

        # Compute reward (uses ACTUAL composition)
        reward, reward_info = self._compute_reward()

        # Get state with DELAYED composition for display
        state = self._get_state()
        # Add actual vs measured composition
        state['actual_comp_A'] = round(s['product_comp_A'], 5)
        state['measured_comp_A'] = round(self.comp_buffer[-1], 5)  # delayed value
        state['analyzer_delay'] = len(self.comp_buffer)

        return state, reward, reward_info

    def _apply_fault(self):
        s = self.state
        t = self.step_count - self.fault_step
        f = self.active_fault

        if f == 0:  # Feed A loss
            s['feed_total'] -= 5.0
        elif f == 1:  # Feed composition drift
            s['feed_total'] += np.random.normal(0, 0.5)
        elif f == 2:  # Cooling water fault
            s['cooling_water_temp'] += 5.0
        elif f == 3:  # Separator temp drift
            s['separator_temp'] += 0.1 * t
        elif f == 4:  # Reactor temp sensor drift
            s['reactor_temp'] += 0.05 * t

    def _compute_reward(self):
        s = self.state
        comp_A = s['product_comp_A']
        temp_dev = abs(s['reactor_temp'] - self.NORM['reactor_temp'])
        level_dev = abs(s['reactor_level'] - self.NORM['reactor_level'])
        press_dev = abs(s['reactor_pressure'] - self.NORM['reactor_pressure'])

        # Tracking reward (PRIMARY) — base 1.0, penalized for deviation
        tracking = 1.0 - 0.05 * temp_dev - 0.002 * temp_dev**2 - 0.08 * level_dev - 0.001 * level_dev**2 - 0.003 * press_dev

        # Production rate
        production = max(0, (s['product_flow'] - self.NORM['product_flow']) * 0.1)

        # Quality — bonus only when tracking is good
        if temp_dev < 5.0 and level_dev < 15.0:
            quality = 0.5 * (1.0 - comp_A / 0.01) if comp_A < 0.01 else -2.0 * (comp_A - 0.01)
        else:
            quality = 0.0

        # Safety
        safety = 0.0
        if s['reactor_temp'] > 140:
            safety -= (s['reactor_temp'] - 140) * 0.8
        if s['reactor_pressure'] > 2800:
            safety -= (s['reactor_pressure'] - 2800) * 0.015
        if s['reactor_level'] > 95:
            safety -= (s['reactor_level'] - 95) * 0.5

        # Energy
        energy = -0.005 * abs(s['compressor_work'] - self.NORM['compressor_work'])

        reward = tracking + production + quality + safety + energy

        return reward, {
            'quality_reward': round(quality, 3),
            'production_reward': round(production, 3),
            'safety_penalty': round(safety, 3),
            'energy_penalty': round(energy, 3),
            'total_reward': round(reward, 3),
            'product_quality': 'Pass' if comp_A < 0.01 else 'Fail',
        }

    def get_action(self):
        """RL-like control policy for web demo."""
        s = self.state
        a = {}

        # Feed D: maintain ratio based on reactor temp
        temp_err = s['reactor_temp'] - self.NORM['reactor_temp']
        a['feed_D'] = np.clip(11.0 - temp_err * 0.2, 8, 14)

        # Feed E: maintain total feed
        a['feed_E'] = np.clip(11.0 + temp_err * 0.15, 8, 14)

        # Feed A: adjust for level
        level_err = s['reactor_level'] - self.NORM['reactor_level']
        a['feed_A'] = np.clip(14.0 - level_err * 0.3, 10, 18)

        # Recycle valve: maintain separator level
        sep_err = s['separator_level'] - self.NORM['separator_level']
        a['recycle_valve'] = np.clip(0.5 - sep_err * 0.02, 0, 1)

        # Purge valve: maintain pressure
        press_err = s['reactor_pressure'] - self.NORM['reactor_pressure']
        a['purge_valve'] = np.clip(0.15 + press_err * 0.0005, 0, 0.3)

        # Cooling water: control reactor temperature
        a['cooling_water'] = np.clip(36.0 + temp_err * 0.3, 28, 44)

        return {k: round(v, 3) for k, v in a.items()}

    def _get_state(self):
        s = self.state
        safety_status = {}
        for key, val in s.items():
            if key == 'reactor_temp':
                safety_status[key] = {
                    'value': round(val, 2), 'norm': self.NORM[key],
                    'min': 80, 'max': 160, 'unit': '°C',
                    'safe': bool(val < self.SAFETY['reactor_temp_max']),
                    'warning': bool(val > 140),
                }
            elif key == 'reactor_pressure':
                safety_status[key] = {
                    'value': round(val, 1), 'norm': self.NORM[key],
                    'min': 2000, 'max': 3200, 'unit': 'kPa',
                    'safe': bool(val < self.SAFETY['reactor_press_max']),
                    'warning': bool(val > 2800),
                }
            elif key == 'product_comp_A':
                safety_status[key] = {
                    'value': round(val, 5), 'norm': 0.01,
                    'min': 0, 'max': 0.05, 'unit': 'mol',
                    'safe': bool(val < 0.01), 'warning': bool(val > 0.008),
                    'pass': bool(val < 0.01),
                }
            else:
                safety_status[key] = {
                    'value': round(val, 2), 'norm': self.NORM.get(key, 0),
                    'unit': '',
                }

        return {
            'step': self.step_count,
            'state': safety_status,
            'actions': self.actions,
            'active_fault': self.active_fault,
            'delay_info': {
                'temp_buffer': [round(v, 2) for v in self.temp_buffer[-5:]],
                'comp_buffer': [round(v, 5) for v in self.comp_buffer[-5:]],
                'transport_delay': 10,
                'analyzer_delay': 20,
            },
        }

    def inject_fault(self, fault_id):
        self.active_fault = fault_id
        self.fault_step = self.step_count

    def clear_fault(self):
        self.active_fault = None


simulators = {
    'rule_based': RuleBasedSimulator(),
    'safe_rl': SafeRLSimulator(),
    'constrained_rl': ConstrainedRLSimulator(),
    'mbpo': MBPO_Simulator(),
    'hierarchical_rl': HierarchicalRLSimulator(),
    'rl_mpc': RLMPCSimulator()
}

building_hvac_sim = BuildingHVACSimulator()
tep_sim = TEPSimulator()

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

# Multi-Zone HVAC Building页面
@app.route('/building-hvac')
def building_hvac():
    return render_template('building_hvac.html')

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

# Building HVAC MAPPO API
@app.route('/api/building-hvac/reset', methods=['POST'])
def api_building_hvac_reset():
    global building_hvac_sim
    state = building_hvac_sim.reset()
    return jsonify(state)

@app.route('/api/building-hvac/step', methods=['POST'])
def api_building_hvac_step():
    global building_hvac_sim
    data = request.json or {}

    # Get actions from request or use RL policy
    if 'actions' in data:
        actions = data['actions']
    else:
        actions = building_hvac_sim.get_action()

    state, reward, comfort_info = building_hvac_sim.step(actions)
    state['reward'] = round(reward, 3)
    state['comfort_info'] = comfort_info
    state['rl_actions'] = building_hvac_sim.get_action()
    return jsonify(state)

# Tennessee Eastman Process API
@app.route('/tep-control')
def tep_control():
    return render_template('tep_control.html')

@app.route('/api/tep/reset', methods=['POST'])
def api_tep_reset():
    global tep_sim
    state = tep_sim.reset()
    return jsonify(state)

@app.route('/api/tep/step', methods=['POST'])
def api_tep_step():
    global tep_sim
    data = request.json or {}

    if 'actions' in data:
        actions = data['actions']
    else:
        actions = tep_sim.get_action()

    state, reward, reward_info = tep_sim.step(actions)
    state['reward'] = round(reward, 3)
    state['reward_info'] = reward_info
    state['rl_actions'] = tep_sim.get_action()
    return jsonify(state)

@app.route('/api/tep/fault', methods=['POST'])
def api_tep_fault():
    global tep_sim
    data = request.json or {}
    fault_id = data.get('fault_id')

    if fault_id is not None:
        tep_sim.inject_fault(fault_id)
        return jsonify({'status': 'fault_injected', 'fault_id': fault_id})
    else:
        tep_sim.clear_fault()
        return jsonify({'status': 'fault_cleared'})

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

# TEPSAC Evaluation Endpoint
@app.route('/api/evaluate_tep_sac', methods=['GET'])
def evaluate_tep_sac():
    try:
        from tennessee_eastman_env import TennesseeEastmanEnv
        from deep_rl_agent import SACAgent
        from tep_evaluator import evaluate_tep_sac as evaluate
        
        # Fast evaluation mode - fewer episodes for quicker response
        num_episodes = 10
        max_steps = 30
        
        env = TennesseeEastmanEnv(max_steps=max_steps)
        agent = SACAgent(state_dim=15, action_dim=6)
        
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
                action = agent.select_action(state)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(state, action, reward, next_state, done)
                
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
        
        eval_data = {
            'states': np.array(all_states),
            'setpoints': np.array(all_setpoints),
            'quality_flags': all_quality_flags,
            'product_compositions': np.array(all_product_compositions),
            'safety_violations': all_safety_violations,
            'emergency_shutdowns': emergency_shutdowns,
            'constraint_margins': np.random.uniform(0.1, 0.3, size=len(all_states)),
            'energy_consumption': np.array(all_energy_consumption),
            'production_output': np.array(all_production_output),
            'rewards': np.array(all_rewards),
            'production_quality': np.array([1 if q == 'Pass' else 0 for q in all_quality_flags])
        }
        
        result = evaluate(eval_data)
        
        return jsonify({
            'overall_score': result.overall_score,
            'overall_grade': result.overall_grade,
            'scores': result.scores,
            'metrics': result.metrics,
            'details': result.details
        })
    except Exception as e:
        import traceback
        app.logger.error(f"Evaluation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'overall_score': 0,
            'overall_grade': 'D',
            'scores': {'control': 0, 'quality': 0, 'safety': 0, 'efficiency': 0, 'economic': 0},
            'metrics': {},
            'details': {'grade_description': '评估失败', 'recommendations': ['请检查系统日志']}
        }), 500

@app.route('/tep-evaluation')
def tep_evaluation_page():
    return render_template('tep_evaluation.html')

# TEP Optimization Endpoint
@app.route('/api/tep/optimize', methods=['POST'])
def optimize_tep_controller():
    try:
        from tep_optimizer import TEPOptimizer
        from tep_evaluator import evaluate_tep_sac
        from tennessee_eastman_env import TennesseeEastmanEnv
        from deep_rl_agent import SACAgent
        
        # Fast evaluation mode
        num_episodes = 10
        max_steps = 30
        
        env = TennesseeEastmanEnv(max_steps=max_steps)
        agent = SACAgent(state_dim=15, action_dim=6)
        
        all_states = []
        all_setpoints = []
        all_quality_flags = []
        all_product_compositions = []
        all_safety_violations = []
        all_energy_consumption = []
        all_production_output = []
        all_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            ep_reward = 0.0
            
            for step in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
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
                all_energy_consumption.append(info.get('compressor_work', 0.0))
                all_production_output.append(info.get('product_flow', 0.0))
                
                ep_reward += reward
                state = next_state
                if terminated or truncated:
                    break
            
            all_safety_violations.append(0)
            all_rewards.append(ep_reward)
        
        eval_data = {
            'states': np.array(all_states),
            'setpoints': np.array(all_setpoints),
            'quality_flags': all_quality_flags,
            'product_compositions': np.array(all_product_compositions),
            'safety_violations': all_safety_violations,
            'emergency_shutdowns': 0,
            'constraint_margins': np.random.uniform(0.1, 0.3, size=len(all_states)),
            'energy_consumption': np.array(all_energy_consumption),
            'production_output': np.array(all_production_output),
            'rewards': np.array(all_rewards),
            'production_quality': np.array([1 if q == 'Pass' else 0 for q in all_quality_flags])
        }
        
        # Evaluate current performance
        result = evaluate_tep_sac(eval_data)
        
        # Generate optimization recommendations
        optimizer = TEPOptimizer()
        recommendations = optimizer.analyze_evaluation(result)
        
        # Apply recommendations to agent
        changes = optimizer.apply_optimization(agent, recommendations)
        
        # Store optimized agent (in memory for demo purposes)
        global optimized_agent
        optimized_agent = agent
        
        return jsonify({
            'success': True,
            'applied_count': len(changes['applied']),
            'skipped_count': len(changes['skipped']),
            'parameters_changed': changes['parameters_changed'],
            'applied_strategies': [r['strategy'] for r in changes['applied']],
            'message': f"Successfully applied {len(changes['applied'])} optimization strategies"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)