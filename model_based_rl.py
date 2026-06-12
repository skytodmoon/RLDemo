import torch
import torch.nn as nn
import numpy as np
from collections import deque

class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3_mean = nn.Linear(128, state_dim)
        self.fc3_std = nn.Linear(128, state_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        std = torch.sigmoid(self.fc3_std(x)) + 0.01
        return mean, std

class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MBPOPolicy(nn.Module):
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

class MBPOAgent:
    def __init__(self, state_dim, action_dim):
        self.transition_model = TransitionModel(state_dim, action_dim)
        self.reward_model = RewardModel(state_dim, action_dim)
        self.policy = MBPOPolicy(state_dim, action_dim)
        
        self.transition_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=1e-3)
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.horizon = 15
        self.num_particles = 10
        self.buffer = deque(maxlen=10000)
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def train_model(self, epochs=10):
        if len(self.buffer) < 128:
            return
        
        for _ in range(epochs):
            batch = np.random.choice(len(self.buffer), 128, replace=False)
            states = torch.tensor([self.buffer[i]['state'] for i in batch], dtype=torch.float32)
            actions = torch.tensor([[self.buffer[i]['action']] for i in batch], dtype=torch.float32)
            next_states = torch.tensor([self.buffer[i]['next_state'] for i in batch], dtype=torch.float32)
            rewards = torch.tensor([[self.buffer[i]['reward']] for i in batch], dtype=torch.float32)
            
            self.transition_optimizer.zero_grad()
            mean, std = self.transition_model(states, actions)
            transition_loss = nn.MSELoss()(mean, next_states)
            transition_loss.backward()
            self.transition_optimizer.step()
            
            self.reward_optimizer.zero_grad()
            pred_rewards = self.reward_model(states, actions)
            reward_loss = nn.MSELoss()(pred_rewards, rewards)
            reward_loss.backward()
            self.reward_optimizer.step()
        
        return {'transition_loss': transition_loss.item(), 'reward_loss': reward_loss.item()}
    
    def plan(self, initial_state):
        initial_state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)
        states = initial_state.repeat(self.num_particles, 1)
        total_rewards = torch.zeros(self.num_particles)
        
        for _ in range(self.horizon):
            probs = self.policy(states)
            actions = torch.multinomial(probs, 1)
            actions_onehot = torch.zeros(self.num_particles, 3)
            actions_onehot.scatter_(1, actions, 1)
            
            mean, std = self.transition_model(states, actions_onehot)
            states = mean + std * torch.randn_like(std)
            
            rewards = self.reward_model(states, actions_onehot)
            total_rewards += rewards.squeeze()
        
        best_action = torch.argmax(total_rewards).item()
        return best_action
    
    def train_policy(self, iterations=5):
        if len(self.buffer) < 128:
            return
        
        for _ in range(iterations):
            batch = np.random.choice(len(self.buffer), 64, replace=False)
            states = torch.tensor([self.buffer[i]['state'] for i in batch], dtype=torch.float32)
            
            self.policy_optimizer.zero_grad()
            
            total_return = 0
            for _ in range(self.num_particles):
                current_states = states.clone()
                episode_return = 0
                
                for h in range(self.horizon):
                    probs = self.policy(current_states)
                    actions = torch.multinomial(probs, 1)
                    actions_onehot = torch.zeros(64, 3)
                    actions_onehot.scatter_(1, actions, 1)
                    
                    mean, std = self.transition_model(current_states, actions_onehot)
                    current_states = mean + std * torch.randn_like(std)
                    
                    rewards = self.reward_model(current_states, actions_onehot)
                    episode_return += (self.gamma ** h) * rewards.mean()
                
                total_return += episode_return
            
            loss = -total_return / self.num_particles
            loss.backward()
            self.policy_optimizer.step()
        
        return {'policy_loss': loss.item()}

if __name__ == '__main__':
    from safe_rl_env import SafeIndustrialTempEnv
    
    env = SafeIndustrialTempEnv()
    agent = MBPOAgent(state_dim=2, action_dim=3)
    
    for episode in range(200):
        obs, _ = env.reset()
        total_reward = 0
        
        while True:
            if episode < 10:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated)
            
            total_reward += reward
            obs = next_obs
            
            if terminated:
                break
        
        if episode >= 10:
            model_loss = agent.train_model()
            policy_loss = agent.train_policy()
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Reward={total_reward:.2f}")