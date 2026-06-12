import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ConstrainedActor(nn.Module):
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

class ConstrainedCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class ConstraintCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class CPOAgent:
    def __init__(self, state_dim, action_dim, lam=0.01, c=1.0):
        self.actor = ConstrainedActor(state_dim, action_dim)
        self.critic = ConstrainedCritic(state_dim)
        self.constraint_critic = ConstraintCritic(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.constraint_optimizer = optim.Adam(self.constraint_critic.parameters(), lr=3e-4)
        
        self.lam = lam
        self.c = c
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        self.buffer = []
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def store_transition(self, state, action, reward, constraint_cost, next_state, done):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'constraint_cost': constraint_cost,
            'next_state': next_state,
            'done': done
        })
    
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return advantages
    
    def train(self):
        if len(self.buffer) < 32:
            return
        
        states = torch.tensor([t['state'] for t in self.buffer], dtype=torch.float32)
        actions = torch.tensor([t['action'] for t in self.buffer], dtype=torch.long)
        rewards = torch.tensor([t['reward'] for t in self.buffer], dtype=torch.float32)
        constraint_costs = torch.tensor([t['constraint_cost'] for t in self.buffer], dtype=torch.float32)
        next_states = torch.tensor([t['next_state'] for t in self.buffer], dtype=torch.float32)
        dones = torch.tensor([t['done'] for t in self.buffer], dtype=torch.float32)
        
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        constraint_values = self.constraint_critic(states).squeeze()
        
        advantages = self.compute_gae(rewards, values, next_values[-1], dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        constraint_advantages = self.compute_gae(constraint_costs, constraint_values, 0, dones)
        constraint_advantages = torch.tensor(constraint_advantages, dtype=torch.float32)
        
        self.actor_optimizer.zero_grad()
        probs = self.actor(states)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        old_probs = action_probs.detach()
        
        ratio = action_probs / (old_probs + 1e-8)
        
        constraint_cost = constraint_advantages.mean()
        if constraint_cost > self.c:
            lagrangian = self.lam * (constraint_cost - self.c)
            objective = -((ratio * advantages).mean() + lagrangian)
        else:
            objective = -(ratio * advantages).mean()
        
        objective.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        value_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))
        value_loss.backward()
        self.critic_optimizer.step()
        
        self.constraint_optimizer.zero_grad()
        constraint_loss = nn.MSELoss()(constraint_values, constraint_costs)
        constraint_loss.backward()
        self.constraint_optimizer.step()
        
        self.buffer = []
        
        return {
            'objective': objective.item(),
            'value_loss': value_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'constraint_cost': constraint_cost.item()
        }

if __name__ == '__main__':
    from safe_rl_env import SafeIndustrialTempEnv
    
    env = SafeIndustrialTempEnv()
    agent = CPOAgent(state_dim=2, action_dim=3)
    
    for episode in range(100):
        obs, _ = env.reset()
        total_reward = 0
        total_constraint_cost = 0
        
        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            constraint_cost = 1.0 if info['violation_count'] > 0 else 0.0
            agent.store_transition(obs, action, reward, constraint_cost, next_obs, terminated)
            
            total_reward += reward
            total_constraint_cost += constraint_cost
            obs = next_obs
            
            if terminated:
                break
        
        result = agent.train()
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={total_reward:.2f}, Constraint Cost={total_constraint_cost:.2f}")