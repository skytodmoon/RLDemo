import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IndustrialTempEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(IndustrialTempEnv, self).__init__()
        
        self.temp_min = 20.0  
        self.temp_max = 35.0  
        self.target_temp = 25.0  
        self.current_temp = 25.0  
        self.time_step = 0
        self.max_steps = 100
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([self.temp_min], dtype=np.float32),
            high=np.array([self.temp_max], dtype=np.float32),
            dtype=np.float32
        )

    def _get_obs(self):
        return np.array([self.current_temp], dtype=np.float32)

    def _get_info(self):
        return {
            'temp': self.current_temp,
            'time_step': self.time_step
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        delta_temp = -0.1
        
        if action == 1:
            delta_temp += 0.3
        elif action == 2:
            delta_temp -= 0.2
        
        proposed_temp = self.current_temp + delta_temp
        boundary_violation = proposed_temp < self.temp_min or proposed_temp > self.temp_max
        self.current_temp = float(np.clip(
            proposed_temp,
            self.temp_min,
            self.temp_max
        ))
        
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        if boundary_violation:
            reward -= 50.0
        
        self.time_step += 1
        terminated = False
        truncated = self.time_step >= self.max_steps
        
        info = self._get_info()
        info['boundary_violation'] = bool(boundary_violation)
        info['temperature_error'] = temp_error
        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = float(self.np_random.uniform(22, 28))
        if options and 'initial_temp' in options:
            initial_temp = float(options['initial_temp'])
            if not np.isfinite(initial_temp) or not self.temp_min <= initial_temp <= self.temp_max:
                raise ValueError('initial_temp must be within the temperature bounds')
            self.current_temp = initial_temp
        self.time_step = 0
        return self._get_obs(), self._get_info()

    def render(self, mode='human'):
        actions = ['IDLE', 'HEAT', 'COOL']
        print(f"Step: {self.time_step:3d} | Temp: {self.current_temp:5.2f}C | Target: {self.target_temp}C")
