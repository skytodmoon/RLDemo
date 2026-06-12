import torch
import torch.nn as nn
import numpy as np
from collections import deque

class RLPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, state_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLMPCAgent:
    def __init__(self, state_dim, action_dim, mpc_horizon=10, lambda_rl=0.5):
        self.rl_policy = RLPolicy(state_dim, action_dim)
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        
        self.rl_optimizer = torch.optim.Adam(self.rl_policy.parameters(), lr=3e-4)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=1e-3)
        
        self.gamma = 0.99
        self.mpc_horizon = mpc_horizon
        self.lambda_rl = lambda_rl
        self.buffer = deque(maxlen=10000)
        
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.target_temp = 25.0
    
    def select_action_rl(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.rl_policy(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def mpc_plan(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        best_action = 0
        best_reward = float('-inf')
        
        for action in range(3):
            current_state = state.clone()
            total_reward = 0
            
            for h in range(self.mpc_horizon):
                action_onehot = torch.zeros(1, 3)
                action_onehot[0, action] = 1
                
                next_state = self.dynamics_model(current_state, action_onehot)
                next_state = torch.clamp(next_state, self.temp_min, self.temp_max)
                
                temp_error = torch.abs(next_state[0, 0] - self.target_temp)
                reward = 10.0 - temp_error * 2
                
                if next_state[0, 0] < self.temp_min or next_state[0, 0] > self.temp_max:
                    reward -= 50.0
                
                total_reward += (self.gamma ** h) * reward
                current_state = next_state
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action
        
        return best_action
    
    def select_action(self, state):
        rl_action = self.select_action_rl(state)
        mpc_action = self.mpc_plan(state)
        
        if np.random.random() < self.lambda_rl:
            return rl_action
        else:
            return mpc_action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def train_dynamics(self, epochs=5):
        if len(self.buffer) < 128:
            return
        
        for _ in range(epochs):
            batch = np.random.choice(len(self.buffer), 128, replace=False)
            states = torch.tensor([self.buffer[i]['state'] for i in batch], dtype=torch.float32)
            actions = torch.tensor([[self.buffer[i]['action']] for i in batch], dtype=torch.float32)
            next_states = torch.tensor([self.buffer[i]['next_state'] for i in batch], dtype=torch.float32)
            
            actions_onehot = torch.zeros(128, 3)
            actions_onehot.scatter_(1, actions.long(), 1)
            
            self.dynamics_optimizer.zero_grad()
            pred_next_states = self.dynamics_model(states, actions_onehot)
            loss = nn.MSELoss()(pred_next_states, next_states)
            loss.backward()
            self.dynamics_optimizer.step()
        
        return {'dynamics_loss': loss.item()}
    
    def train_rl(self):
        if len(self.buffer) < 32:
            return
        
        batch = np.random.choice(len(self.buffer), 32, replace=False)
        states = torch.tensor([self.buffer[i]['state'] for i in batch], dtype=torch.float32)
        actions = torch.tensor([self.buffer[i]['action'] for i in batch], dtype=torch.long)
        rewards = torch.tensor([self.buffer[i]['reward'] for i in batch], dtype=torch.float32)
        
        self.rl_optimizer.zero_grad()
        probs = self.rl_policy(states)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        loss = -(action_probs * advantages.detach()).mean()
        loss.backward()
        self.rl_optimizer.step()
        
        return {'rl_loss': loss.item()}

class MPCController:
    def __init__(self, horizon=10):
        self.horizon = horizon
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.target_temp = 25.0
    
    def simulate_step(self, temp, action):
        delta_temp = -0.1
        if action == 1:
            delta_temp += 0.3
        elif action == 2:
            delta_temp -= 0.2
        return max(self.temp_min, min(self.temp_max, temp + delta_temp))
    
    def compute_reward(self, temp):
        temp_error = abs(temp - self.target_temp)
        reward = 10.0 - temp_error * 2
        if temp < self.temp_min or temp > self.temp_max:
            reward -= 50.0
        return reward
    
    def plan(self, current_temp):
        best_action = 0
        best_total_reward = float('-inf')
        
        for action in range(3):
            temp = current_temp
            total_reward = 0
            
            for h in range(self.horizon):
                temp = self.simulate_step(temp, action)
                reward = self.compute_reward(temp)
                total_reward += (0.99 ** h) * reward
            
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_action = action
        
        return best_action

if __name__ == '__main__':
    from safe_rl_env import SafeIndustrialTempEnv
    
    env = SafeIndustrialTempEnv()
    agent = RLMPCAgent(state_dim=2, action_dim=3)
    
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated)
            
            total_reward += reward
            obs = next_obs
            
            if terminated:
                break
        
        if episode >= 5:
            agent.train_dynamics()
            agent.train_rl()
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={total_reward:.2f}")