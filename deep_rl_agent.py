"""
Soft Actor-Critic (SAC) for Tennessee Eastman Process Control
==============================================================

SAC is ideal for TEP because:
  1. Off-policy: sample efficient (reuses past data via replay buffer)
  2. Maximum entropy: naturally explores, robust to local optima
  3. Automatic temperature tuning: no manual entropy coefficient tuning
  4. Stable: twin Q-networks prevent overestimation

Architecture:
  - Actor: Gaussian policy with reparameterization trick
  - Twin Critics: two independent Q-networks (take minimum)
  - Target networks: soft update for stability
  - Replay buffer: 100K transitions
  - Auto α tuning: learns entropy coefficient automatically
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
import os
import json


# ============================================================
# Neural Network Modules
# ============================================================

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """
    Squashed Gaussian Policy for SAC.

    Outputs mean and log_std for a Gaussian distribution,
    then squashes through tanh to bound actions in [-1, 1].
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 2, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference."""
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        return torch.tanh(x_t)


class TwinQCritic(nn.Module):
    """
    Twin Q-Networks for SAC.
    Takes (state, action) as input, outputs Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + action_dim

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.q1.modules()) + list(self.q2.modules()):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """Fixed-size replay buffer with numpy arrays for efficiency."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]).unsqueeze(-1),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx]).unsqueeze(-1),
        )


# ============================================================
# SAC Agent
# ============================================================

class SACAgent:
    """
    Soft Actor-Critic agent with automatic temperature tuning.

    Key features:
    - Squashed Gaussian policy
    - Twin Q-networks (minimum for stability)
    - Automatic α (entropy coefficient) tuning
    - Soft target network updates
    - Replay buffer for off-policy learning
    """

    def __init__(self, state_dim: int = 15, action_dim: int = 6,
                 hidden_dim: int = 256, lr_actor: float = 3e-4, lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, auto_alpha: bool = True,
                 buffer_size: int = 100_000, batch_size: int = 256,
                 device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Networks
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinQCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers with separate LRs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic temperature tuning with slower alpha learning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # Target entropy = -dim(action) * 0.5 (less aggressive exploration reduction)
            self.target_entropy = -float(action_dim) * 0.5
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

        # Training stats
        self.stats = {
            'critic_loss': [], 'actor_loss': [], 'alpha_loss': [],
            'alpha': [], 'q_values': [],
        }

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for environment interaction."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_t, deterministic)
            return action.cpu().numpy().flatten()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """Perform one SAC update step."""
        if self.replay_buffer.size < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ---- Critic update ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---- Actor update ----
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ---- Alpha update ----
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        # ---- Soft target update ----
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Record stats
        stats = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss,
            'alpha': self.alpha,
            'q_values': q1.mean().item(),
        }
        for k, v in stats.items():
            self.stats[k].append(v)

        return stats

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha': self.alpha,
            'stats': self.stats,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.auto_alpha and checkpoint.get('log_alpha') is not None:
            self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint['alpha']


# ============================================================
# Training Script
# ============================================================

def train_sac(num_episodes: int = 200, max_steps: int = 500,
              update_every: int = 1, updates_per_step: int = 1,
              warmup_steps: int = 2000, save_path: str = "sac_tep_model.pt",
              log_interval: int = 10, verbose: bool = True) -> Dict:
    """
    Train SAC on Tennessee Eastman Process environment.

    Args:
        num_episodes: number of training episodes
        max_steps: steps per episode
        update_every: steps between policy updates
        updates_per_step: SAC updates per step
        warmup_steps: random exploration steps before training
        save_path: path to save trained model
        log_interval: episodes between logging
        verbose: print progress

    Returns:
        Training history dictionary
    """
    from tennessee_eastman_env import TennesseeEastmanEnv

    env = TennesseeEastmanEnv(max_steps=max_steps)
    agent = SACAgent(
        state_dim=15, action_dim=6, hidden_dim=256,
        lr_actor=3e-4, lr_critic=5e-4, lr_alpha=3e-4,
        gamma=0.99, tau=0.005,
        auto_alpha=True, buffer_size=100_000, batch_size=256,
    )

    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'avg_q_values': [],
        'product_quality': [],
        'critic_losses': [],
        'actor_losses': [],
        'alphas': [],
        'temp_tracking': [],
        'level_tracking': [],
    }

    total_steps = 0
    best_avg_reward = float('-inf')

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        quality_pass_count = 0
        temp_devs = []
        level_devs = []

        for step in range(max_steps):
            # Warmup: random actions
            if total_steps < warmup_steps:
                action = env.action_space.sample() * 0.3
            else:
                action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if total_steps >= warmup_steps and total_steps % update_every == 0:
                for _ in range(updates_per_step):
                    stats = agent.update()

            episode_reward += reward
            total_steps += 1

            if info.get('product_quality') == 'Pass':
                quality_pass_count += 1
            temp_devs.append(abs(info['reactor_temp'] - 120.0))
            level_devs.append(abs(info['reactor_level'] - 65.0))

            state = next_state
            if done:
                break

        # Episode stats
        quality_rate = quality_pass_count / (step + 1) if step > 0 else 0
        avg_temp_dev = np.mean(temp_devs) if temp_devs else 0
        avg_level_dev = np.mean(level_devs) if level_devs else 0

        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(step + 1)
        history['product_quality'].append(quality_rate)
        history['temp_tracking'].append(avg_temp_dev)
        history['level_tracking'].append(avg_level_dev)

        if agent.stats['q_values']:
            history['avg_q_values'].append(np.mean(agent.stats['q_values'][-10:]))
        if agent.stats['critic_loss']:
            history['critic_losses'].append(np.mean(agent.stats['critic_loss'][-10:]))
        if agent.stats['actor_loss']:
            history['actor_losses'].append(np.mean(agent.stats['actor_loss'][-10:]))
        history['alphas'].append(agent.alpha)

        # Save best model (composite score: reward + quality + tracking)
        if episode >= 10:
            avg_reward = np.mean(history['episode_rewards'][-10:])
            avg_quality = np.mean(history['product_quality'][-10:])
            avg_temp = np.mean(history['temp_tracking'][-10:])
            avg_level = np.mean(history['level_tracking'][-10:])
            # Composite score: reward weighted by quality and penalized by tracking error
            composite = avg_reward * avg_quality - avg_temp * 0.5 - avg_level * 0.3
            if composite > best_avg_reward:
                best_avg_reward = composite
                agent.save(save_path)

        # Logging
        if verbose and (episode + 1) % log_interval == 0:
            avg_r = np.mean(history['episode_rewards'][-log_interval:])
            avg_q = quality_rate
            avg_loss = np.mean(history['critic_losses'][-5:]) if history['critic_losses'] else 0
            print(f"Ep {episode+1:4d} | R: {avg_r:7.2f} | Q: {avg_q:.1%} | "
                  f"T_dev: {avg_temp_dev:.1f} | L_dev: {avg_level_dev:.1f} | "
                  f"α: {agent.alpha:.3f}")

    agent.save(save_path)
    if verbose:
        print(f"\nTraining complete. Best composite score: {best_avg_reward:.2f}")

    return history


if __name__ == "__main__":
    print("=" * 60)
    print("SAC Training for Tennessee Eastman Process")
    print("=" * 60)

    history = train_sac(
        num_episodes=30,
        max_steps=100,
        update_every=10,
        warmup_steps=200,
        log_interval=5,
        verbose=True,
        save_path="sac_tep_test.pt",
    )

    print(f"\nFinal avg reward (last 5): {np.mean(history['episode_rewards'][-5:]):.2f}")
    print(f"Final quality rate (last 5): {np.mean(history['product_quality'][-5:]):.1%}")

    if os.path.exists("sac_tep_test.pt"):
        os.remove("sac_tep_test.pt")
        print("Test model cleaned up.")
