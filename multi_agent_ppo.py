"""
Multi-Agent PPO (MAPPO) for Building HVAC Control
===================================================

Centralized Training, Decentralized Execution (CTDE):
  - Centralized Critic: sees full building state for value estimation
  - Decentralized Actors: each zone has its own policy network

Architecture:
  - SharedFeatureExtractor: processes zone-local observations
  - CentralizedCritic: estimates V(s) from full state
  - DecentralizedActor: π(a_i | o_i) for each zone
  - MultiAgentPPO: training loop with GAE, clipped objective

Reference: Yu et al. "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games" (2022) - MAPPO paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import os

from building_hvac_env import BuildingHVACEnv


# ============================================================
# Neural Network Modules
# ============================================================

class SharedEncoder(nn.Module):
    """Shared feature encoder for zone observations."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecentralizedActor(nn.Module):
    """
    Per-zone actor network. Outputs parameters for a Beta distribution
    over continuous actions [0, 1].

    Input: zone-local observation (3 vars) + context (outdoor temp, solar, hour)
    Output: 4 actions (heating, cooling, humidify, dehumidify) as Beta(alpha, beta)
    """

    def __init__(self, obs_dim: int = 6, hidden_dim: int = 64, num_actions: int = 4):
        super().__init__()
        self.encoder = SharedEncoder(obs_dim, hidden_dim * 2, hidden_dim)

        # Alpha and Beta parameters for Beta distribution
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_actions),
            nn.Softplus(),  # ensure positive
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_actions),
            nn.Softplus(),  # ensure positive
        )

        # Initialize for near-uniform distribution
        self._init_weights()

    def _init_weights(self):
        for m in [self.alpha_head, self.beta_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 1.0)  # alpha=beta=1 → uniform

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        alpha = self.alpha_head(features) + 1.0  # minimum 1.0 for stability
        beta = self.beta_head(features) + 1.0
        return alpha, beta

    def get_action_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, beta = self.forward(obs)
        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, beta = self.forward(obs)
        dist = Beta(alpha, beta)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized value function. Sees the full building state
    to estimate V(s) for advantage computation.

    Input: full 15-dim state (4 zones × 3 vars + 3 env vars)
    Output: scalar value V(s)
    """

    def __init__(self, state_dim: int = 15, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ============================================================
# Rollout Buffer for Multi-Agent
# ============================================================

class MultiAgentRolloutBuffer:
    """Stores trajectories for MAPPO training."""

    def __init__(self, num_zones: int = 4, zone_obs_dim: int = 6,
                 full_state_dim: int = 15, num_actions: int = 4):
        self.num_zones = num_zones
        self.zone_obs_dim = zone_obs_dim
        self.full_state_dim = full_state_dim
        self.num_actions = num_actions
        self.clear()

    def clear(self):
        self.zone_obs = []       # List[List[Tensor]] - per-zone obs per step
        self.full_states = []    # List[Tensor] - full state per step
        self.actions = []        # List[Tensor] - joint actions per step
        self.log_probs = []      # List[Tensor] - joint log probs per step
        self.rewards = []        # List[float]
        self.values = []         # List[float]
        self.dones = []          # List[bool]

    def add(self, zone_obs: List[np.ndarray], full_state: np.ndarray,
            action: np.ndarray, log_prob: float, reward: float,
            value: float, done: bool):
        self.zone_obs.append([torch.FloatTensor(o) for o in zone_obs])
        self.full_states.append(torch.FloatTensor(full_state))
        self.actions.append(torch.FloatTensor(action))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, gamma: float = 0.99, lam: float = 0.95,
                                         last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)

        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values[:-1]
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    def get_batches(self, advantages: torch.Tensor, returns: torch.Tensor,
                    batch_size: int = 64):
        """Yield random mini-batches for training."""
        T = len(self.rewards)
        indices = np.random.permutation(T)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            idx = indices[start:end]

            batch_zone_obs = []
            for z in range(self.num_zones):
                zone_batch = torch.stack([self.zone_obs[i][z] for i in idx])
                batch_zone_obs.append(zone_batch)

            batch_full_state = torch.stack([self.full_states[i] for i in idx])
            batch_actions = torch.stack([self.actions[i] for i in idx])
            batch_log_probs = torch.tensor([self.log_probs[i] for i in idx], dtype=torch.float32)
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            yield {
                "zone_obs": batch_zone_obs,
                "full_state": batch_full_state,
                "actions": batch_actions,
                "old_log_probs": batch_log_probs,
                "advantages": batch_advantages,
                "returns": batch_returns,
            }


# ============================================================
# MAPPO Agent
# ============================================================

class MultiAgentPPO:
    """
    Multi-Agent PPO with Centralized Training, Decentralized Execution.

    Key features:
    - Per-zone actor networks (Beta distribution for continuous actions)
    - Shared centralized critic
    - GAE advantage estimation
    - PPO clipped objective with entropy bonus
    - Gradient clipping for stability
    """

    def __init__(self, num_zones: int = 4, zone_obs_dim: int = 6,
                 full_state_dim: int = 15, num_actions: int = 4,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5):
        self.num_zones = num_zones
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Create actor per zone
        self.actors = nn.ModuleList([
            DecentralizedActor(zone_obs_dim, hidden_dim=64, num_actions=num_actions)
            for _ in range(num_zones)
        ])

        # Shared centralized critic
        self.critic = CentralizedCritic(full_state_dim, hidden_dim=128)

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor, eps=1e-5)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Rollout buffer
        self.buffer = MultiAgentRolloutBuffer(num_zones, zone_obs_dim, full_state_dim, num_actions)

        # Training stats
        self.training_stats = defaultdict(list)

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Select joint action for all zones.

        Args:
            obs: full observation from environment

        Returns:
            joint_action: 16-dim array
            zone_log_probs: per-zone log probabilities
            joint_log_prob: summed log probability
            value: critic value estimate
        """
        # Parse zone observations (each zone: 3 vars)
        zone_obs_list = []
        for z in range(self.num_zones):
            start = z * 3
            zone_obs_list.append(obs[start:start + 3])

        # Add context: outdoor_temp, solar, hour (last 4 vars minus energy_rem)
        context = obs[-4:-1]  # outdoor_temp, solar, hour

        # Build per-zone input (zone local + context)
        zone_inputs = []
        for z in range(self.num_zones):
            zone_input = np.concatenate([zone_obs_list[z], context])
            zone_inputs.append(zone_input)

        # Get actions from each actor
        joint_action = []
        joint_log_prob = 0.0

        with torch.no_grad():
            for z, actor in enumerate(self.actors):
                zone_tensor = torch.FloatTensor(zone_inputs[z]).unsqueeze(0)
                action, log_prob = actor.get_action_and_logprob(zone_tensor)
                joint_action.append(action.squeeze(0).numpy())
                joint_log_prob += log_prob.item()

            # Critic value
            full_state = torch.FloatTensor(obs[:15]).unsqueeze(0)  # exclude energy_rem for critic
            value = self.critic(full_state).item()

        joint_action = np.concatenate(joint_action)
        return joint_action, zone_inputs, joint_log_prob, value

    def store_transition(self, zone_obs: List[np.ndarray], full_state: np.ndarray,
                          action: np.ndarray, log_prob: float, reward: float,
                          value: float, done: bool):
        """Store a transition in the rollout buffer."""
        self.buffer.add(zone_obs, full_state, action, log_prob, reward, value, done)

    def update(self, num_epochs: int = 4, batch_size: int = 64) -> Dict:
        """
        PPO update using collected rollout data.

        Returns:
            Dictionary of training metrics
        """
        # Compute advantages
        advantages, returns = self.buffer.compute_returns_and_advantages(
            self.gamma, self.lam
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        stats = defaultdict(float)
        num_updates = 0

        for epoch in range(num_epochs):
            for batch in self.buffer.get_batches(advantages, returns, batch_size):
                # --- Update each zone's actor ---
                total_actor_loss = 0
                total_entropy = 0

                for z, actor in enumerate(self.actors):
                    zone_obs = batch["zone_obs"][z]
                    actions = batch["actions"][:, z * 4:(z + 1) * 4]
                    old_log_probs = batch["old_log_probs"]

                    # Evaluate current policy
                    new_log_probs, entropy = actor.evaluate_actions(zone_obs, actions)

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * batch["advantages"]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch["advantages"]
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Entropy bonus
                    entropy_loss = -entropy.mean() * self.entropy_coef

                    total_loss = actor_loss + entropy_loss

                    self.actor_optimizers[z].zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    self.actor_optimizers[z].step()

                    total_actor_loss += actor_loss.item()
                    total_entropy += entropy.mean().item()

                # --- Update critic ---
                full_states = batch["full_state"]
                values = self.critic(full_states)
                critic_loss = F.mse_loss(values, batch["returns"])

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                stats["actor_loss"] += total_actor_loss
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += total_entropy
                num_updates += 1

        # Average stats
        for key in stats:
            stats[key] /= max(num_updates, 1)

        # Clear buffer
        self.buffer.clear()

        return dict(stats)

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'actors': self.actors.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, weights_only=True)
        self.actors.load_state_dict(checkpoint['actors'])
        self.critic.load_state_dict(checkpoint['critic'])


# ============================================================
# Training Script
# ============================================================

def train_mappo(num_episodes: int = 200, max_steps: int = 288,
                update_interval: int = 288, num_update_epochs: int = 4,
                save_path: str = "mappo_hvac_model.pt",
                log_interval: int = 10, verbose: bool = True) -> Dict:
    """
    Train MAPPO on the BuildingHVAC environment.

    Args:
        num_episodes: number of training episodes
        max_steps: steps per episode (288 = 24h at 5-min intervals)
        update_interval: steps between policy updates
        num_update_epochs: PPO epochs per update
        save_path: path to save trained model
        log_interval: episodes between logging
        verbose: print progress

    Returns:
        Training history dictionary
    """
    env = BuildingHVACEnv(max_steps=max_steps)
    agent = MultiAgentPPO(
        num_zones=4,
        zone_obs_dim=6,
        full_state_dim=15,
        num_actions=4,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
    )

    history = {
        "episode_rewards": [],
        "avg_comfort_rates": [],
        "avg_energy_usage": [],
        "actor_losses": [],
        "critic_losses": [],
        "entropies": [],
    }

    best_avg_reward = float('-inf')

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        comfort_count = 0
        total_steps = 0

        for step in range(max_steps):
            # Select action
            action, zone_obs, log_prob, value = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            full_state = obs[:15]
            agent.store_transition(zone_obs, full_state, action, log_prob, reward, value, done)

            episode_reward += reward
            total_steps += 1

            # Count comfort
            for zone_name, zone_info in info.get("zone_details", {}).items():
                if zone_info.get("in_comfort", False):
                    comfort_count += 1

            obs = next_obs

            # Update policy
            if (step + 1) % update_interval == 0:
                update_stats = agent.update(num_epochs=num_update_epochs)
                history["actor_losses"].append(update_stats.get("actor_loss", 0))
                history["critic_losses"].append(update_stats.get("critic_loss", 0))
                history["entropies"].append(update_stats.get("entropy", 0))

            if done:
                break

        # Episode stats
        avg_comfort = comfort_count / (total_steps * 4) if total_steps > 0 else 0
        energy_usage = info.get("energy_used", 0) / info.get("energy_budget", 1000)

        history["episode_rewards"].append(episode_reward)
        history["avg_comfort_rates"].append(avg_comfort)
        history["avg_energy_usage"].append(energy_usage)

        # Save best model
        if episode >= 10:
            avg_reward = np.mean(history["episode_rewards"][-10:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(save_path)

        # Logging
        if verbose and (episode + 1) % log_interval == 0:
            avg_r = np.mean(history["episode_rewards"][-log_interval:])
            avg_c = np.mean(history["avg_comfort_rates"][-log_interval:])
            avg_e = np.mean(history["avg_energy_usage"][-log_interval:])
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {avg_r:7.2f} | "
                  f"Comfort: {avg_c:.1%} | "
                  f"Energy: {avg_e:.1%}")

    # Final save
    agent.save(save_path)
    if verbose:
        print(f"\nTraining complete. Model saved to {save_path}")
        print(f"Best avg reward: {best_avg_reward:.2f}")

    return history


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MAPPO Training for Multi-Zone HVAC Control")
    print("=" * 60)

    # Quick test with few episodes
    history = train_mappo(
        num_episodes=20,
        max_steps=50,  # shorter episodes for quick test
        update_interval=50,
        log_interval=5,
        verbose=True,
        save_path="mappo_hvac_model_test.pt",
    )

    print(f"\nFinal avg reward (last 5): {np.mean(history['episode_rewards'][-5:]):.2f}")
    print(f"Final avg comfort (last 5): {np.mean(history['avg_comfort_rates'][-5:]):.1%}")

    # Clean up test model
    if os.path.exists("mappo_hvac_model_test.pt"):
        os.remove("mappo_hvac_model_test.pt")
        print("Test model cleaned up.")
