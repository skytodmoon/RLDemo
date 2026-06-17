"""
Tennessee Eastman Process (TEP) Control Environment
=====================================================

A simplified but physically-grounded simulation of the Tennessee Eastman Process,
the gold standard benchmark for industrial process control and fault detection.

Reference: Downs & Vogel (1993), "A plant-wide industrial process control problem"

Process Overview:
  Reactor: A+B→C, A+C→D, A+D→E (desired), 3A→F (byproduct)
  Condenser: Vapor condensation
  Separator: Gas-liquid separation
  Stripper: Product purification
  Recycle: Unreacted material recovery

Key Characteristics:
  - Multi-variable: 15 state variables, 6 manipulated variables
  - Multi-coupling: Temperature↔reaction rate↔composition, flow↔level↔pressure
  - Large delays: Analyzer lag (20 steps), transport delay (10 steps), thermal inertia
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional


class TennesseeEastmanEnv(gym.Env):
    """
    Simplified Tennessee Eastman Process environment.

    State (15 vars):
      [reactor_temp, reactor_pressure, reactor_level,
       separator_temp, separator_level,
       stripper_level,
       product_flow, product_comp_A, product_comp_B, product_comp_C,
       feed_A, feed_B, feed_C, feed_D, feed_E,
       cooling_water_temp, recycle_flow]

    Wait, that's 17. Let me redefine to exactly 15:
      [reactor_temp, reactor_pressure, reactor_level,
       separator_temp, separator_level,
       stripper_level,
       product_flow, product_comp_A,
       feed_total, feed_ratio_D,
       cooling_water_temp, recycle_flow,
       purge_rate, compressor_work, agitator_speed]

    Action (6 vars, continuous [-1, 1]):
      [feed_D, feed_E, feed_A, recycle_valve, purge_valve, cooling_water]
    """

    metadata = {"render_modes": ["human"]}

    # ---- Normal operating conditions (scaled) ----
    # Reactor
    REACTOR_TEMP_NORM = 120.0      # °C
    REACTOR_PRESS_NORM = 2700.0    # kPa
    REACTOR_LEVEL_NORM = 65.0      # %

    # Separator
    SEPARATOR_TEMP_NORM = 80.0     # °C
    SEPARATOR_LEVEL_NORM = 50.0    # %

    # Stripper
    STRIPPER_LEVEL_NORM = 50.0     # %

    # Product
    PRODUCT_FLOW_NORM = 22.0       # m³/h
    PRODUCT_COMP_A_NORM = 0.005    # mole fraction (target < 0.01)

    # Feed
    FEED_TOTAL_NORM = 40.0         # m³/h
    FEED_RATIO_D_NORM = 0.3        # D/(D+E) ratio

    # Utilities
    COOLING_TEMP_NORM = 35.0       # °C
    RECYCLE_FLOW_NORM = 25.0       # m³/h
    PURGE_RATE_NORM = 0.5          # %
    COMPRESSOR_WORK_NORM = 100.0   # kW
    AGITATOR_SPEED_NORM = 200.0    # rpm

    # ---- Safety limits ----
    REACTOR_TEMP_MAX = 150.0       # °C (emergency shutdown)
    REACTOR_PRESS_MAX = 3000.0     # kPa (emergency shutdown)
    REACTOR_LEVEL_MAX = 100.0      # %
    REACTOR_LEVEL_MIN = 15.0       # %

    def __init__(self, max_steps: int = 500, dt: float = 1.0,
                 delay_analyzer: int = 20, delay_transport: int = 10,
                 seed: Optional[int] = None):
        super().__init__()
        self.max_steps = max_steps
        self.dt = dt
        self.delay_analyzer = delay_analyzer
        self.delay_transport = delay_transport

        # State: 15 continuous variables
        self.observation_space = spaces.Box(
            low=np.array([
                80.0,   # reactor_temp
                2000.0, # reactor_pressure
                20.0,   # reactor_level
                50.0,   # separator_temp
                20.0,   # separator_level
                20.0,   # stripper_level
                5.0,    # product_flow
                0.0,    # product_comp_A
                15.0,   # feed_total
                0.1,    # feed_ratio_D
                20.0,   # cooling_water_temp
                5.0,    # recycle_flow
                0.0,    # purge_rate
                30.0,   # compressor_work
                100.0,  # agitator_speed
            ], dtype=np.float32),
            high=np.array([
                160.0,  # reactor_temp
                3200.0, # reactor_pressure
                100.0,  # reactor_level
                120.0,  # separator_temp
                100.0,  # separator_level
                100.0,  # stripper_level
                40.0,   # product_flow
                0.05,   # product_comp_A
                60.0,   # feed_total
                0.6,    # feed_ratio_D
                55.0,   # cooling_water_temp
                45.0,   # recycle_flow
                2.0,    # purge_rate
                200.0,  # compressor_work
                300.0,  # agitator_speed
            ], dtype=np.float32),
        )

        # Action: 6 continuous [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Delay buffers (for simulating large delays)
        self.reactor_temp_buffer = deque(maxlen=delay_transport + 1)
        self.composition_buffer = deque(maxlen=delay_analyzer + 1)
        self.separator_input_buffer = deque(maxlen=delay_transport // 2 + 1)

        # Fault system
        self.active_fault = None
        self.fault_step = 0

        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.state = None
        self.actions = None

    def _init_state(self) -> np.ndarray:
        """Initialize state near normal operating conditions with small perturbation."""
        state = np.array([
            self.REACTOR_TEMP_NORM + np.random.normal(0, 2),
            self.REACTOR_PRESS_NORM + np.random.normal(0, 30),
            self.REACTOR_LEVEL_NORM + np.random.normal(0, 3),
            self.SEPARATOR_TEMP_NORM + np.random.normal(0, 2),
            self.SEPARATOR_LEVEL_NORM + np.random.normal(0, 3),
            self.STRIPPER_LEVEL_NORM + np.random.normal(0, 3),
            self.PRODUCT_FLOW_NORM + np.random.normal(0, 1),
            self.PRODUCT_COMP_A_NORM + np.random.normal(0, 0.001),
            self.FEED_TOTAL_NORM + np.random.normal(0, 1),
            self.FEED_RATIO_D_NORM + np.random.normal(0, 0.02),
            self.COOLING_TEMP_NORM + np.random.normal(0, 1),
            self.RECYCLE_FLOW_NORM + np.random.normal(0, 1),
            self.PURGE_RATE_NORM + np.random.normal(0, 0.05),
            self.COMPRESSOR_WORK_NORM + np.random.normal(0, 5),
            self.AGITATOR_SPEED_NORM + np.random.normal(0, 5),
        ], dtype=np.float32)

        # Fill delay buffers with initial state
        for _ in range(self.delay_analyzer + 1):
            self.reactor_temp_buffer.append(state[0])
            self.composition_buffer.append(state[7])
        for _ in range(self.delay_transport // 2 + 1):
            self.separator_input_buffer.append(state[0])

        return state

    def _apply_actions(self, action: np.ndarray) -> Dict[str, float]:
        """Convert normalized action [-1,1] to physical values."""
        action = np.clip(action, -1.0, 1.0)

        # Map [-1, 1] to physical ranges (balanced for steady-state)
        feed_D = 8.0 + (action[0] + 1) * 3.0          # [8, 14] m³/h
        feed_E = 8.0 + (action[1] + 1) * 3.0          # [8, 14] m³/h
        feed_A = 10.0 + (action[2] + 1) * 4.0          # [10, 18] m³/h
        recycle_valve = (action[3] + 1) / 2             # [0, 1]
        purge_valve = (action[4] + 1) / 2 * 0.3        # [0, 0.3]
        cooling_water = 28.0 + (action[5] + 1) * 8.0   # [28, 44] °C

        self.actions = {
            'feed_D': feed_D, 'feed_E': feed_E, 'feed_A': feed_A,
            'recycle_valve': recycle_valve, 'purge_valve': purge_valve,
            'cooling_water': cooling_water,
        }
        return self.actions

    def _compute_reaction_rate(self, temp: float, feed_A: float, feed_D: float) -> Dict:
        """
        Compute reaction rates using Arrhenius-type kinetics.
        Coupling: temperature affects all reaction rates nonlinearly.
        Activation energies calibrated so reactions are significant at 120°C.
        """
        T = temp + 273  # Convert to Kelvin
        T_norm = self.REACTOR_TEMP_NORM + 273  # Reference temperature

        # Arrhenius rates normalized to 1.0 at T_norm
        k1 = np.exp(-2000 * (1/T - 1/T_norm))    # A+B→C
        k2 = np.exp(-2500 * (1/T - 1/T_norm))    # A+C→D
        k3 = np.exp(-1500 * (1/T - 1/T_norm))    # A+D→E (desired)
        k4 = np.exp(-1000 * (1/T - 1/T_norm))    # 3A→F (byproduct)

        # Reaction rates (proportional to feed concentrations)
        r1 = k1 * feed_A * 0.5
        r2 = k2 * feed_A * 0.3
        r3 = k3 * feed_A * feed_D * 0.01  # desired reaction
        r4 = k4 * feed_A ** 3 * 0.001     # byproduct

        return {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'desired': r3, 'byproduct': r4}

    def _compute_dynamics(self, state: np.ndarray, actions: Dict) -> np.ndarray:
        """
        Compute state derivatives with coupling and delays.

        Coupling structure:
        - Reactor temp ↔ reaction rate ↔ composition
        - Feed flow ↔ reactor level ↔ separator level
        - Cooling water ↔ reactor temp ↔ reaction selectivity
        - Purge ↔ recycle ↔ reactor pressure
        - Product flow ↔ stripper level ↔ separator level
        """
        s = state
        a = actions
        dt = self.dt

        # Get delayed values (large delays)
        delayed_reactor_temp = self.reactor_temp_buffer[0]  # pure delay
        delayed_composition = self.composition_buffer[0]    # analyzer delay

        # ---- Coupling calculations ----

        # Reaction rates (coupled to temperature)
        reactions = self._compute_reaction_rate(s[0], a['feed_A'], a['feed_D'])

        # Heat generation from reactions (exothermic) — balanced with heat removal
        # At T=120°C, CW=36°C: heat_gen ≈ heat_removal ≈ 42
        heat_gen = reactions['r1'] * 3.5 + reactions['r2'] * 2.0 + reactions['r3'] * 1.5
        # Cooling water effect — calibrated so equilibrium is at T_norm with CW=36°C
        heat_removal = (s[0] - a['cooling_water']) * 0.5

        # Feed effect on level (balanced: total feed ≈ product out at normal conditions)
        total_feed_in = a['feed_A'] + a['feed_D'] + a['feed_E']
        recycle_in = s[11] * a['recycle_valve'] * 0.15  # recycle is partial
        total_in = total_feed_in + recycle_in

        # Product out (strongly coupled to level - self-regulating around NORM)
        # At normal level (65), product_out ≈ total_in (mass balance)
        level_ratio = s[2] / self.REACTOR_LEVEL_NORM
        product_out = s[6] * level_ratio * 1.2  # scaled so equilibrium is at NORM level

        # Pressure (coupled to temperature and purge)
        temp_pressure_coupling = (s[0] - self.REACTOR_TEMP_NORM) * 5.0
        purge_pressure_relief = a['purge_valve'] * 200

        # ---- State updates (with time constants for inertia) ----

        new_state = np.copy(s)

        # 0: Reactor temperature (large thermal inertia)
        tau_temp = 15.0
        target_temp = s[0] + (heat_gen - heat_removal) * 0.01
        # Coupling: delayed reactor temp affects current via recycle
        recycle_temp_effect = (delayed_reactor_temp - s[0]) * 0.02 * a['recycle_valve']
        new_state[0] = s[0] + (target_temp - s[0] + recycle_temp_effect) / tau_temp * dt
        new_state[0] += np.random.normal(0, 0.3)  # process noise

        # 1: Reactor pressure (coupled to temperature and purge)
        tau_press = 8.0
        target_press = self.REACTOR_PRESS_NORM + temp_pressure_coupling - purge_pressure_relief
        new_state[1] = s[1] + (target_press - s[1]) / tau_press * dt
        new_state[1] += np.random.normal(0, 5)

        # 2: Reactor level (strongly self-regulating around NORM)
        tau_level = 4.0
        level_change = total_in - product_out
        # Strong feedback: level is pulled back toward NORM
        level_feedback = (s[2] - self.REACTOR_LEVEL_NORM) * 0.3
        new_state[2] = s[2] + (level_change - level_feedback) / tau_level * dt
        new_state[2] = np.clip(new_state[2], 20.0, 90.0)
        new_state[2] += np.random.normal(0, 0.2)

        # 3: Separator temperature (coupled to reactor temp via transport delay)
        tau_sep_temp = 12.0
        sep_temp_target = delayed_reactor_temp * 0.7 + 20  # heat loss in transport
        new_state[3] = s[3] + (sep_temp_target - s[3]) / tau_sep_temp * dt
        new_state[3] += np.random.normal(0, 0.2)

        # 4: Separator level (coupled to reactor level via transport, self-regulating)
        tau_sep_level = 6.0
        delayed_sep_input = self.separator_input_buffer[0]
        sep_in = (delayed_sep_input - self.REACTOR_LEVEL_NORM) * 0.05  # inflow from reactor
        sep_out = a['recycle_valve'] * 3 + product_out * 0.15  # outflow via recycle and product
        sep_feedback = (s[4] - self.SEPARATOR_LEVEL_NORM) * 0.1
        new_state[4] = s[4] + (sep_in - sep_out - sep_feedback) / tau_sep_level * dt
        new_state[4] = np.clip(new_state[4], 20.0, 90.0)
        new_state[4] += np.random.normal(0, 0.3)

        # 5: Stripper level (coupled to separator and product flow)
        tau_strip_level = 5.0
        strip_in = s[4] * 0.15  # from separator
        strip_out = product_out * 0.3
        # Stripper level feedback
        strip_feedback = (s[5] - self.STRIPPER_LEVEL_NORM) * 0.08
        new_state[5] = s[5] + (strip_in - strip_out - strip_feedback) / tau_strip_level * dt
        new_state[5] = np.clip(new_state[5], 20.0, 90.0)
        new_state[5] += np.random.normal(0, 0.3)

        # 6: Product flow (coupled to reactor level and reactions - self-regulating)
        tau_product = 4.0
        # Product flow increases with reactor level (more material available)
        level_boost = (s[2] - self.REACTOR_LEVEL_NORM) * 0.05
        product_target = self.PRODUCT_FLOW_NORM + reactions['desired'] * 5 + level_boost
        product_target = np.clip(product_target, 5.0, 35.0)
        new_state[6] = s[6] + (product_target - s[6]) / tau_product * dt
        new_state[6] = np.clip(new_state[6], 5.0, 35.0)
        new_state[6] += np.random.normal(0, 0.2)

        # 7: Product composition A (large analyzer delay - uses delayed value)
        tau_comp = 10.0
        # High temp → more byproduct → more A in product
        comp_A_effect = (s[0] - self.REACTOR_TEMP_NORM) * 0.0001
        # Low feed D → incomplete reaction → more A
        comp_feed_effect = (self.FEED_RATIO_D_NORM - a['feed_D'] / (a['feed_D'] + a['feed_E'] + 0.01)) * 0.005
        comp_target = self.PRODUCT_COMP_A_NORM + comp_A_effect + comp_feed_effect
        # Use delayed composition for control (analyzer lag)
        new_state[7] = s[7] + (comp_target - s[7]) / tau_comp * dt
        new_state[7] += np.random.normal(0, 0.0002)
        new_state[7] = max(0, new_state[7])

        # 8: Total feed (directly controlled)
        new_state[8] = total_feed_in + np.random.normal(0, 0.2)

        # 9: Feed ratio D/(D+E)
        if a['feed_D'] + a['feed_E'] > 0.01:
            new_state[9] = a['feed_D'] / (a['feed_D'] + a['feed_E'])
        else:
            new_state[9] = 0.5

        # 10: Cooling water temperature (directly controlled)
        new_state[10] = a['cooling_water'] + np.random.normal(0, 0.3)

        # 11: Recycle flow (coupled to separator level and recycle valve)
        tau_recycle = 5.0
        recycle_target = s[4] * 0.3 * a['recycle_valve'] + 5
        new_state[11] = s[11] + (recycle_target - s[11]) / tau_recycle * dt
        new_state[11] += np.random.normal(0, 0.2)

        # 12: Purge rate (directly controlled)
        new_state[12] = a['purge_valve'] + np.random.normal(0, 0.01)

        # 13: Compressor work (coupled to recycle flow and pressure)
        new_state[13] = self.COMPRESSOR_WORK_NORM + (s[11] - self.RECYCLE_FLOW_NORM) * 2 + (s[1] - self.REACTOR_PRESS_NORM) * 0.01
        new_state[13] += np.random.normal(0, 1)

        # 14: Agitator speed (relatively independent)
        new_state[14] = self.AGITATOR_SPEED_NORM + np.random.normal(0, 1)

        # Update delay buffers
        self.reactor_temp_buffer.append(new_state[0])
        self.composition_buffer.append(new_state[7])
        self.separator_input_buffer.append(new_state[2])  # reactor level → separator

        # Apply fault if active
        if self.active_fault is not None:
            new_state = self._apply_fault(new_state)

        return new_state.astype(np.float32)

    def _apply_fault(self, state: np.ndarray) -> np.ndarray:
        """Apply one of the TEP fault scenarios."""
        s = state
        fault = self.active_fault
        t = self.step_count - self.fault_step

        if fault == 0:  # Fault 0: Feed A loss (step change)
            s[8] -= 5.0  # sudden feed loss
        elif fault == 1:  # Fault 1: Feed B composition (random walk)
            s[8] += np.random.normal(0, 0.5)
        elif fault == 2:  # Fault 2: Reactor cooling water inlet temp (step)
            s[10] += 5.0
        elif fault == 3:  # Fault 3: Condenser cooling water inlet temp (drift)
            s[3] += 0.1 * t  # slow drift
        elif fault == 4:  # Fault 4: Reactor temperature sensor drift
            s[0] += 0.05 * t  # slow sensor drift

        return s

    def _compute_reward(self, state: np.ndarray, actions: Dict) -> Tuple[float, Dict]:
        """
        Hierarchical reward: tracking first, then production, then quality.
        The agent must maintain normal conditions before optimizing production.
        """
        s = state
        comp_A = s[7]
        temp_dev = abs(s[0] - self.REACTOR_TEMP_NORM)
        level_dev = abs(s[2] - self.REACTOR_LEVEL_NORM)
        press_dev = abs(s[1] - self.REACTOR_PRESS_NORM)

        # 1. Tracking reward (PRIMARY) — must maintain normal conditions
        #    Base reward of 1.0, penalized for deviation
        temp_penalty = 0.05 * temp_dev + 0.002 * temp_dev ** 2
        level_penalty = 0.08 * level_dev + 0.001 * level_dev ** 2
        press_penalty = 0.003 * press_dev
        tracking_reward = 1.0 - temp_penalty - level_penalty - press_penalty

        # 2. Production rate — bonus for higher throughput
        production_reward = max(0, (s[6] - self.PRODUCT_FLOW_NORM) * 0.1)

        # 3. Quality — bonus only when tracking is good
        if temp_dev < 5.0 and level_dev < 15.0:
            # Good tracking → quality matters
            if comp_A < 0.01:
                quality_reward = 0.5 * (1.0 - comp_A / 0.01)
            else:
                quality_reward = -2.0 * (comp_A - 0.01)
        else:
            # Poor tracking → no quality bonus
            quality_reward = 0.0

        # 4. Safety penalties — hard boundaries
        safety_penalty = 0.0
        if s[0] > self.REACTOR_TEMP_MAX - 10:
            safety_penalty -= (s[0] - (self.REACTOR_TEMP_MAX - 10)) * 0.8
        if s[0] > self.REACTOR_TEMP_MAX:
            safety_penalty -= 15.0
        if s[1] > self.REACTOR_PRESS_MAX - 200:
            safety_penalty -= (s[1] - (self.REACTOR_PRESS_MAX - 200)) * 0.015
        if s[2] > self.REACTOR_LEVEL_MAX - 5:
            safety_penalty -= (s[2] - (self.REACTOR_LEVEL_MAX - 5)) * 0.5
        if s[2] < self.REACTOR_LEVEL_MIN + 5:
            safety_penalty -= ((self.REACTOR_LEVEL_MIN + 5) - s[2]) * 0.5

        # 5. Energy penalty
        energy_penalty = -0.005 * abs(s[13] - self.COMPRESSOR_WORK_NORM)

        reward = tracking_reward + production_reward + quality_reward + safety_penalty + energy_penalty

        info = {
            'quality_reward': round(quality_reward, 3),
            'production_reward': round(production_reward, 3),
            'safety_penalty': round(safety_penalty, 3),
            'energy_penalty': round(energy_penalty, 3),
            'tracking_reward': round(tracking_reward, 3),
            'total_reward': round(reward, 3),
            'product_comp_A': round(float(s[7]), 5),
            'product_quality': 'Pass' if s[7] < 0.01 else 'Fail',
        }

        return reward, info

    def _check_safety(self, state: np.ndarray) -> bool:
        """Check if emergency shutdown is needed."""
        return (state[0] > self.REACTOR_TEMP_MAX or
                state[1] > self.REACTOR_PRESS_MAX or
                state[2] > self.REACTOR_LEVEL_MAX or
                state[2] < self.REACTOR_LEVEL_MIN)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.active_fault = None
        self.fault_step = 0
        self.reactor_temp_buffer.clear()
        self.composition_buffer.clear()
        self.separator_input_buffer.clear()

        self.state = self._init_state()
        self.actions = {
            'feed_D': 15.0, 'feed_E': 15.0, 'feed_A': 22.5,
            'recycle_valve': 0.5, 'purge_valve': 0.25, 'cooling_water': 35.0,
        }

        return self.state, self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        actions = self._apply_actions(action)

        # Compute dynamics with delays and coupling
        new_state = self._compute_dynamics(self.state, actions)
        self.state = new_state
        self.step_count += 1

        # Compute reward (uses ACTUAL composition for product quality)
        reward, reward_info = self._compute_reward(self.state, actions)

        # Check termination
        emergency = self._check_safety(self.state)
        terminated = emergency
        truncated = self.step_count >= self.max_steps

        # Build observation with DELAYED composition (analyzer delay)
        # The agent sees the delayed measurement, not the actual value
        obs = self.state.copy()
        delayed_comp = self.composition_buffer[0]  # 20-step old measurement
        obs[7] = delayed_comp  # Replace actual comp with delayed measurement

        info = self._get_info()
        info.update(reward_info)
        info['actual_comp_A'] = round(float(self.state[7]), 5)
        info['measured_comp_A'] = round(float(delayed_comp), 5)
        info['analyzer_delay'] = self.delay_analyzer

        if emergency:
            info['emergency_shutdown'] = True
            reward -= 20.0  # heavy penalty for emergency

        return obs, reward, terminated, truncated, info

    def inject_fault(self, fault_id: int):
        """Inject a fault scenario."""
        self.active_fault = fault_id
        self.fault_step = self.step_count

    def clear_fault(self):
        """Clear active fault."""
        self.active_fault = None

    def _get_info(self) -> Dict:
        """Build info dict for current state."""
        s = self.state
        return {
            'step': self.step_count,
            'reactor_temp': round(float(s[0]), 2),
            'reactor_pressure': round(float(s[1]), 1),
            'reactor_level': round(float(s[2]), 2),
            'separator_temp': round(float(s[3]), 2),
            'separator_level': round(float(s[4]), 2),
            'stripper_level': round(float(s[5]), 2),
            'product_flow': round(float(s[6]), 2),
            'product_comp_A': round(float(s[7]), 5),
            'feed_total': round(float(s[8]), 2),
            'feed_ratio_D': round(float(s[9]), 3),
            'cooling_water_temp': round(float(s[10]), 2),
            'recycle_flow': round(float(s[11]), 2),
            'purge_rate': round(float(s[12]), 3),
            'compressor_work': round(float(s[13]), 1),
            'agitator_speed': round(float(s[14]), 1),
            'actions': self.actions,
            'active_fault': self.active_fault,
            'delay_buffer_size': len(self.reactor_temp_buffer),
        }

    def get_delay_analysis(self) -> Dict:
        """Analyze delay characteristics for visualization."""
        if len(self.reactor_temp_buffer) < 2:
            return {'reactor_temp_delay': [], 'composition_delay': []}

        return {
            'reactor_temp_delay': list(self.reactor_temp_buffer),
            'composition_delay': list(self.composition_buffer),
            'transport_delay_steps': self.delay_transport,
            'analyzer_delay_steps': self.delay_analyzer,
        }

    def render(self):
        s = self.state
        print(f"\n=== TEP Step {self.step_count} ===")
        print(f"Reactor:  T={s[0]:.1f}°C  P={s[1]:.0f}kPa  L={s[2]:.1f}%")
        print(f"Separator: T={s[3]:.1f}°C  L={s[4]:.1f}%")
        print(f"Stripper: L={s[5]:.1f}%")
        print(f"Product:  F={s[6]:.1f}m³/h  A={s[7]:.4f} {'✓' if s[7] < 0.01 else '✗'}")
        if self.actions:
            a = self.actions
            print(f"Actions:  D={a['feed_D']:.1f} E={a['feed_E']:.1f} A={a['feed_A']:.1f}")
            print(f"          Recycle={a['recycle_valve']:.2f} Purge={a['purge_valve']:.2f} Cool={a['cooling_water']:.1f}")


if __name__ == "__main__":
    env = TennesseeEastmanEnv(max_steps=100, seed=42)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"State: {obs}")

    total_reward = 0
    for step in range(100):
        action = env.action_space.sample() * 0.3  # gentle random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 20 == 0:
            env.render()
            print(f"  Reward: {reward:.3f}")
        if terminated:
            print("EMERGENCY SHUTDOWN!")
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Delay analysis: {env.get_delay_analysis()}")
