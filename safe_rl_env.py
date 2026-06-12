import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SafeIndustrialTempEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.target_temp = 25.0
        self.safety_margin = 2.0
        self.current_temp = 25.0
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([self.temp_min, 0.0]),
            high=np.array([self.temp_max, 1.0]),
            dtype=np.float32
        )
        
        self.safety_layer_active = False
        self.violation_count = 0
        self.episode_steps = 0
    
    def _get_obs(self):
        safety_status = 1.0 if self._is_unsafe() else 0.0
        return np.array([self.current_temp, safety_status], dtype=np.float32)
    
    def _is_unsafe(self):
        return self.current_temp < self.temp_min + self.safety_margin or \
               self.current_temp > self.temp_max - self.safety_margin
    
    def _get_info(self):
        return {
            'violation_count': self.violation_count,
            'safety_layer_active': self.safety_layer_active,
            'distance_to_boundary': min(
                self.current_temp - self.temp_min,
                self.temp_max - self.current_temp
            )
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = np.random.uniform(22, 28)
        self.safety_layer_active = False
        self.violation_count = 0
        self.episode_steps = 0
        return self._get_obs(), self._get_info()
    
    def _safety_filter(self, action):
        if self.current_temp <= self.temp_min + 1.0 and action == 1:
            return 0
        if self.current_temp >= self.temp_max - 1.0 and action == 2:
            return 0
        return action
    
    def step(self, action):
        self.episode_steps += 1
        
        original_action = action
        action = self._safety_filter(action)
        
        if action != original_action:
            self.safety_layer_active = True
        else:
            self.safety_layer_active = False
        
        delta_temp = -0.1
        if action == 1:
            delta_temp += 0.3
        elif action == 2:
            delta_temp -= 0.2
        
        self.current_temp = np.clip(self.current_temp + delta_temp, self.temp_min, self.temp_max)
        
        if self.current_temp <= self.temp_min or self.current_temp >= self.temp_max:
            self.violation_count += 1
        
        temp_error = abs(self.current_temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        
        if self._is_unsafe():
            reward -= 10.0
        if self.current_temp <= self.temp_min or self.current_temp >= self.temp_max:
            reward -= 50.0
        
        terminated = self.episode_steps >= 100
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

class SafeLayer:
    def __init__(self, env):
        self.env = env
        self.safe_actions = {
            'heat_allowed': True,
            'cool_allowed': True
        }
    
    def update_safety_constraints(self, obs):
        temp = obs[0]
        self.safe_actions['heat_allowed'] = temp < self.env.temp_max - 1.0
        self.safe_actions['cool_allowed'] = temp > self.env.temp_min + 1.0
    
    def filter_action(self, action):
        if action == 1 and not self.safe_actions['heat_allowed']:
            return 0
        if action == 2 and not self.safe_actions['cool_allowed']:
            return 0
        return action

if __name__ == '__main__':
    env = SafeIndustrialTempEnv()
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward:.2f}, Info: {info}")
        if terminated:
            break