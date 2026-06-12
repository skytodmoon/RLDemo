import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class HighLevelPolicy(nn.Module):
    def __init__(self, state_dim, num_options):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_options)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class LowLevelPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + 1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, state, option):
        x = torch.cat([state, option], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class OptionCritic(nn.Module):
    def __init__(self, state_dim, num_options):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_options)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class TerminationClassifier(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class HRLAgent:
    def __init__(self, state_dim, action_dim, num_options=3):
        self.num_options = num_options
        self.high_level_policy = HighLevelPolicy(state_dim, num_options)
        self.low_level_policy = LowLevelPolicy(state_dim, action_dim)
        self.option_critic = OptionCritic(state_dim, num_options)
        self.termination_classifier = TerminationClassifier(state_dim)
        
        self.optimizers = {
            'high': optim.Adam(self.high_level_policy.parameters(), lr=3e-4),
            'low': optim.Adam(self.low_level_policy.parameters(), lr=3e-4),
            'critic': optim.Adam(self.option_critic.parameters(), lr=3e-4),
            'term': optim.Adam(self.termination_classifier.parameters(), lr=3e-4)
        }
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.buffer = []
        self.current_option = 0
    
    def select_high_level_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.high_level_policy(state)
        option = torch.multinomial(probs, 1).item()
        return option
    
    def select_low_level_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        option = torch.tensor([[self.current_option]], dtype=torch.float32)
        probs = self.low_level_policy(state, option)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def check_termination(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        prob = self.termination_classifier(state).item()
        return prob > 0.5
    
    def store_transition(self, state, option, action, reward, next_state, done, terminated):
        self.buffer.append({
            'state': state,
            'option': option,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'terminated': terminated
        })
    
    def train(self):
        if len(self.buffer) < 32:
            return
        
        states = torch.tensor([t['state'] for t in self.buffer], dtype=torch.float32)
        options = torch.tensor([[t['option']] for t in self.buffer], dtype=torch.float32)
        actions = torch.tensor([t['action'] for t in self.buffer], dtype=torch.long)
        rewards = torch.tensor([t['reward'] for t in self.buffer], dtype=torch.float32)
        next_states = torch.tensor([t['next_state'] for t in self.buffer], dtype=torch.float32)
        dones = torch.tensor([t['done'] for t in self.buffer], dtype=torch.float32)
        terminateds = torch.tensor([t['terminated'] for t in self.buffer], dtype=torch.float32)
        
        option_values = self.option_critic(states)
        next_option_values = self.option_critic(next_states)
        
        option_indices = options.squeeze().long()
        current_values = option_values.gather(1, option_indices.unsqueeze(1)).squeeze()
        
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 1 - terminateds[i]
            delta = rewards[i] + self.gamma * mask * next_option_values[i][option_indices[i]] - current_values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.optimizers['low'].zero_grad()
        low_probs = self.low_level_policy(states, options)
        low_action_probs = low_probs.gather(1, actions.unsqueeze(1)).squeeze()
        low_loss = -(low_action_probs * advantages.detach()).mean()
        low_loss.backward()
        self.optimizers['low'].step()
        
        self.optimizers['high'].zero_grad()
        high_probs = self.high_level_policy(states)
        high_action_probs = high_probs.gather(1, option_indices.unsqueeze(1)).squeeze()
        high_loss = -(high_action_probs * advantages.detach()).mean()
        high_loss.backward()
        self.optimizers['high'].step()
        
        self.optimizers['critic'].zero_grad()
        target_values = rewards + self.gamma * (1 - dones) * next_option_values.gather(1, option_indices.unsqueeze(1)).squeeze()
        critic_loss = nn.MSELoss()(current_values, target_values.detach())
        critic_loss.backward()
        self.optimizers['critic'].step()
        
        self.optimizers['term'].zero_grad()
        term_probs = self.termination_classifier(states).squeeze()
        term_targets = terminateds
        term_loss = nn.BCELoss()(term_probs, term_targets)
        term_loss.backward()
        self.optimizers['term'].step()
        
        self.buffer = []
        
        return {
            'low_loss': low_loss.item(),
            'high_loss': high_loss.item(),
            'critic_loss': critic_loss.item(),
            'term_loss': term_loss.item()
        }

if __name__ == '__main__':
    from safe_rl_env import SafeIndustrialTempEnv
    
    env = SafeIndustrialTempEnv()
    agent = HRLAgent(state_dim=2, action_dim=3, num_options=3)
    
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        agent.current_option = agent.select_high_level_action(obs)
        
        while True:
            action = agent.select_low_level_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            should_terminate = agent.check_termination(next_obs) or terminated
            
            agent.store_transition(obs, agent.current_option, action, reward, next_obs, terminated, should_terminate)
            
            total_reward += reward
            obs = next_obs
            
            if should_terminate and not terminated:
                agent.current_option = agent.select_high_level_action(obs)
            
            if terminated:
                break
        
        result = agent.train()
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={total_reward:.2f}")