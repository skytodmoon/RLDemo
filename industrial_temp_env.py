import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IndustrialTempEnv(gym.Env):
    metadata = {'render.modes': ['human']}

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
        delta_temp = -0.1
        
        if action == 1:
            delta_temp += 0.3
        elif action == 2:
            delta_temp -= 0.2
        
        self.current_temp = np.clip(
            self.current_temp + delta_temp,
            self.temp_min,
            self.temp_max
        )
        
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        if self.current_temp < self.temp_min or self.current_temp > self.temp_max:
            reward -= 50.0
        
        self.time_step += 1
        terminated = self.time_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = np.random.uniform(22, 28)
        self.time_step = 0
        return self._get_obs(), self._get_info()

    def render(self, mode='human'):
        actions = ['IDLE', 'HEAT', 'COOL']
        print(f"Step: {self.time_step:3d} | Temp: {self.current_temp:5.2f}C | Target: {self.target_temp}C")