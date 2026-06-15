"""
Multi-Zone HVAC Building Energy Management Environment
=======================================================

A Gymnasium environment simulating a 4-zone building with coupled thermal dynamics,
occupancy patterns, weather disturbances, and energy constraints.

Zones:
  - Office:    standard comfort, moderate load
  - Server:    high heat generation, strict cooling requirements
  - Lab:       strict humidity control, chemical ventilation needs
  - Conference: variable occupancy, rapid load changes

State per zone (12 vars total):
  - temperature (°C)
  - humidity (% RH)
  - occupancy (0-1 normalized)

External disturbances:
  - outdoor temperature (sinusoidal day/night)
  - solar radiation (bell curve midday)
  - occupancy schedule (square wave with noise)

Actions per zone (continuous [0,1], 4 per zone, 16 total):
  - heating power
  - cooling power
  - humidification power
  - dehumidification power

Coupling:
  - Adjacent zones exchange heat via conductance matrix
  - Server room radiates heat to Office and Lab
  - Conference room affects Office via shared wall
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Tuple


class BuildingHVACEnv(gym.Env):
    """
    Multi-zone HVAC building environment with coupled thermal dynamics.

    Observation space: 23-dim vector
      [zone0_temp, zone0_humid, zone0_occ,
       zone1_temp, zone1_humid, zone1_occ,
       zone2_temp, zone2_humid, zone2_occ,
       zone3_temp, zone3_humid, zone3_occ,
       outdoor_temp, solar_rad, hour_of_day,
       energy_remaining, total_energy_used]

    Action space: 16-dim continuous [0,1]
      Per zone: [heating, cooling, humidify, dehumidify]
    """

    metadata = {"render_modes": ["human"]}

    # Zone definitions
    ZONE_NAMES = ["Office", "Server", "Lab", "Conference"]
    NUM_ZONES = 4

    # Comfort targets per zone: (temp_target, temp_tol, humid_target, humid_tol)
    COMFORT = {
        "Office":     {"temp": 23.0, "temp_tol": 2.0, "humid": 45.0, "humid_tol": 10.0},
        "Server":     {"temp": 20.0, "temp_tol": 1.5, "humid": 40.0, "humid_tol": 15.0},
        "Lab":        {"temp": 22.0, "temp_tol": 1.5, "humid": 50.0, "humid_tol": 5.0},
        "Conference": {"temp": 23.0, "temp_tol": 2.5, "humid": 45.0, "humid_tol": 12.0},
    }

    # Zone thermal properties: (mass, base_heat_gen, capacity)
    ZONE_PROPS = {
        "Office":     {"mass": 500.0, "heat_gen": 0.2,  "cap_heat": 2.0, "cap_cool": 2.0, "cap_humid": 1.5, "cap_dehumid": 1.5},
        "Server":     {"mass": 400.0, "heat_gen": 1.5,  "cap_heat": 1.0, "cap_cool": 3.5, "cap_humid": 1.0, "cap_dehumid": 2.0},
        "Lab":        {"mass": 400.0, "heat_gen": 0.3,  "cap_heat": 2.0, "cap_cool": 2.0, "cap_humid": 3.0, "cap_dehumid": 3.0},
        "Conference": {"mass": 350.0, "heat_gen": 0.1,  "cap_heat": 2.5, "cap_cool": 2.5, "cap_humid": 1.5, "cap_dehumid": 1.5},
    }

    # Adjacency coupling matrix (thermal conductance W/°C between zones)
    # Index: 0=Office, 1=Server, 2=Lab, 3=Conference
    COUPLING = np.array([
        [0.0, 0.8, 0.3, 0.6],  # Office <-> Server, Lab, Conference
        [0.8, 0.0, 0.5, 0.1],  # Server <-> Office, Lab
        [0.3, 0.5, 0.0, 0.1],  # Lab <-> Office, Server
        [0.6, 0.1, 0.1, 0.0],  # Conference <-> Office
    ])

    # Occupancy schedules (hour-based, normalized 0-1)
    OCC_SCHEDULES = {
        "Office":     lambda h: 0.8 if 8 <= h <= 18 else (0.3 if 6 <= h <= 20 else 0.05),
        "Server":     lambda h: 0.2 if 0 <= h <= 6 else 0.3,  # always some maintenance
        "Lab":        lambda h: 0.7 if 9 <= h <= 17 else (0.2 if 7 <= h <= 19 else 0.0),
        "Conference": lambda h: 0.9 if (9 <= h <= 11 or 14 <= h <= 16) else (0.3 if 8 <= h <= 18 else 0.0),
    }

    def __init__(self, max_steps: int = 288, dt: float = 300.0, seed: Optional[int] = None):
        """
        Args:
            max_steps: Maximum steps per episode (288 = 24h at 5-min intervals)
            dt: Time step in seconds (300 = 5 minutes)
            seed: Random seed
        """
        super().__init__()
        self.max_steps = max_steps
        self.dt = dt  # seconds per step
        self.energy_budget = 1000.0  # total energy units per episode

        # Observation: 4 zones × 3 vars + 4 env vars = 16
        # [t0,h0,o0, t1,h1,o1, t2,h2,o2, t3,h3,o3, outdoor_t, solar, hour, energy_rem]
        obs_low = np.array([15.0, 20.0, 0.0] * 4 + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([35.0, 80.0, 1.0] * 4 + [45.0, 1.0, 24.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: 4 zones × 4 controls = 16 continuous [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)

        self.step_count = 0
        self.hour_of_day = 8.0  # start at 8 AM
        self.zones = {}
        self.energy_used = 0.0

        if seed is not None:
            np.random.seed(seed)

    def _get_outdoor_temp(self) -> float:
        """Sinusoidal outdoor temperature: peaks at 14:00, trough at 02:00."""
        return 15.0 + 10.0 * np.sin(2 * np.pi * (self.hour_of_day - 6.0) / 24.0)

    def _get_solar_radiation(self) -> float:
        """Bell-curve solar radiation, 0 at night."""
        if self.hour_of_day < 6 or self.hour_of_day > 20:
            return 0.0
        return max(0.0, np.sin(np.pi * (self.hour_of_day - 6.0) / 14.0))

    def _get_occupancy(self, zone_name: str) -> float:
        """Get occupancy for a zone at current hour, with noise."""
        base = self.OCC_SCHEDULES[zone_name](self.hour_of_day)
        noise = np.random.normal(0, 0.05)
        return np.clip(base + noise, 0.0, 1.0)

    def _init_zones(self) -> Dict:
        """Initialize zone states with random temperatures."""
        zones = {}
        for name in self.ZONE_NAMES:
            comfort = self.COMFORT[name]
            zones[name] = {
                "temp": np.random.uniform(comfort["temp"] - 2, comfort["temp"] + 2),
                "humidity": np.random.uniform(comfort["humid"] - 5, comfort["humid"] + 5),
                "occupancy": self._get_occupancy(name),
                "heater": 0.0,
                "cooler": 0.0,
                "humidifier": 0.0,
                "dehumidifier": 0.0,
            }
        return zones

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        obs = []
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            obs.extend([z["temp"], z["humidity"], z["occupancy"]])

        outdoor_t = self._get_outdoor_temp()
        solar = self._get_solar_radiation()
        energy_rem = max(0.0, 1.0 - self.energy_used / self.energy_budget)

        obs.extend([outdoor_t, solar, self.hour_of_day, energy_rem])
        return np.array(obs, dtype=np.float32)

    def _apply_actions(self, actions: np.ndarray):
        """Parse and apply 16-dim action vector to zones."""
        action_names = ["heater", "cooler", "humidifier", "dehumidifier"]
        for i, name in enumerate(self.ZONE_NAMES):
            zone = self.zones[name]
            for j, prop_name in enumerate(action_names):
                zone[prop_name] = float(np.clip(actions[i * 4 + j], 0.0, 1.0))

    def _compute_energy_cost(self) -> float:
        """Total energy consumed this step across all zones."""
        total = 0.0
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            props = self.ZONE_PROPS[name]
            total += (z["heater"] * props["cap_heat"] * 0.5 +
                      z["cooler"] * props["cap_cool"] * 0.8 +
                      z["humidifier"] * props["cap_humid"] * 0.2 +
                      z["dehumidifier"] * props["cap_dehumid"] * 0.3)
        return total * (self.dt / 300.0)

    def _update_dynamics(self):
        """Update zone temperatures and humidity with coupling."""
        outdoor_t = self._get_outdoor_temp()
        solar = self._get_solar_radiation()
        dt_factor = self.dt / 300.0  # normalize to 5-min base

        # --- Temperature dynamics ---
        new_temps = {}
        for i, name in enumerate(self.ZONE_NAMES):
            z = self.zones[name]
            props = self.ZONE_PROPS[name]

            # HVAC effect
            hvac_heat = (z["heater"] * props["cap_heat"] - z["cooler"] * props["cap_cool"]) * 0.6 * dt_factor

            # Internal heat generation (server, lab equipment, occupancy)
            internal_heat = (props["heat_gen"] + z["occupancy"] * 0.6) * 0.12 * dt_factor

            # Solar gain (through windows) - reduced for better control
            solar_gain = solar * 0.4 * dt_factor * (0.3 if name == "Office" else 0.1)

            # Heat loss to outdoor (through walls) - increased for better heat dissipation
            wall_loss_coeff = 0.03  # thermal loss coefficient
            outdoor_loss = (z["temp"] - outdoor_t) * wall_loss_coeff * dt_factor

            # Coupling heat from adjacent zones
            coupling_heat = 0.0
            for j, other_name in enumerate(self.ZONE_NAMES):
                if i != j:
                    conductance = self.COUPLING[i, j] * 0.01 * dt_factor
                    coupling_heat += (self.zones[other_name]["temp"] - z["temp"]) * conductance

            new_temps[name] = z["temp"] + hvac_heat + internal_heat + solar_gain - outdoor_loss + coupling_heat

        # Apply new temperatures
        for name in self.ZONE_NAMES:
            self.zones[name]["temp"] = np.clip(new_temps[name], 10.0, 40.0)

        # --- Humidity dynamics ---
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            props = self.ZONE_PROPS[name]

            # HVAC humidity effect
            humid_change = (z["humidifier"] * props["cap_humid"] - z["dehumidifier"] * props["cap_dehumid"]) * 0.5 * dt_factor

            # Occupancy adds moisture
            occupant_humid = z["occupancy"] * 0.05 * dt_factor

            # Natural drift toward outdoor humidity (simplified)
            outdoor_humid = 50.0 + 10.0 * np.sin(2 * np.pi * self.hour_of_day / 24.0)
            humid_drift = (outdoor_humid - z["humidity"]) * 0.002 * dt_factor

            z["humidity"] = np.clip(z["humidity"] + humid_change + occupant_humid + humid_drift, 10.0, 90.0)

        # --- Update occupancy ---
        for name in self.ZONE_NAMES:
            self.zones[name]["occupancy"] = self._get_occupancy(name)

    def _compute_reward(self) -> Tuple[float, Dict]:
        """
        Multi-objective reward:
          - Comfort: penalize deviation from target temp/humidity, weighted by occupancy
          - Energy: penalize total energy consumption
          - Constraint: heavy penalty if energy budget exceeded
        """
        total_comfort_reward = 0.0
        zone_details = {}

        for name in self.ZONE_NAMES:
            z = self.zones[name]
            comfort = self.COMFORT[name]
            occ = z["occupancy"]

            # Temperature comfort (exponential penalty outside tolerance)
            temp_err = abs(z["temp"] - comfort["temp"])
            if temp_err <= comfort["temp_tol"]:
                temp_reward = 1.0 - (temp_err / comfort["temp_tol"]) * 0.3
            else:
                temp_reward = max(0.0, 0.7 - (temp_err - comfort["temp_tol"]) * 0.5)

            # Humidity comfort
            humid_err = abs(z["humidity"] - comfort["humid"])
            if humid_err <= comfort["humid_tol"]:
                humid_reward = 1.0 - (humid_err / comfort["humid_tol"]) * 0.2
            else:
                humid_reward = max(0.0, 0.8 - (humid_err - comfort["humid_tol"]) * 0.3)

            # Occupancy-weighted comfort (comfort matters more when occupied)
            zone_comfort = (temp_reward * 0.7 + humid_reward * 0.3) * (0.3 + 0.7 * occ)
            total_comfort_reward += zone_comfort

            zone_details[name] = {
                "temp_reward": round(temp_reward, 3),
                "humid_reward": round(humid_reward, 3),
                "comfort": round(zone_comfort, 3),
                "in_comfort": temp_err <= comfort["temp_tol"] and humid_err <= comfort["humid_tol"],
            }

        # Normalize comfort to [-1, 1]
        comfort_reward = (total_comfort_reward / self.NUM_ZONES) * 2.0 - 1.0

        # Energy penalty
        energy_step = self._compute_energy_cost()
        energy_penalty = -0.1 * energy_step

        # Budget constraint
        budget_penalty = 0.0
        if self.energy_used >= self.energy_budget:
            budget_penalty = -1.0  # heavy penalty

        reward = comfort_reward + energy_penalty + budget_penalty

        info = {
            "comfort_reward": round(comfort_reward, 3),
            "energy_penalty": round(energy_penalty, 3),
            "budget_penalty": round(budget_penalty, 3),
            "total_reward": round(reward, 3),
            "zone_details": zone_details,
            "energy_step": round(energy_step, 3),
            "energy_used": round(self.energy_used, 1),
            "energy_budget": self.energy_budget,
        }

        return reward, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.hour_of_day = 8.0 + np.random.uniform(-1, 1)
        self.energy_used = 0.0
        self.zones = self._init_zones()

        obs = self._get_obs()
        info = self._get_step_info()
        return obs, info

    def _get_step_info(self) -> Dict:
        """Build info dict for current step."""
        outdoor_t = self._get_outdoor_temp()
        solar = self._get_solar_radiation()

        zone_info = {}
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            comfort = self.COMFORT[name]
            zone_info[name] = {
                "temp": round(z["temp"], 2),
                "humidity": round(z["humidity"], 2),
                "occupancy": round(z["occupancy"], 2),
                "temp_target": comfort["temp"],
                "humid_target": comfort["humid"],
                "heater": round(z["heater"], 2),
                "cooler": round(z["cooler"], 2),
                "humidifier": round(z["humidifier"], 2),
                "dehumidifier": round(z["dehumidifier"], 2),
            }

        return {
            "step": self.step_count,
            "hour": round(self.hour_of_day, 2),
            "outdoor_temp": round(outdoor_t, 2),
            "solar_radiation": round(solar, 3),
            "zones": zone_info,
            "energy_used": round(self.energy_used, 1),
            "energy_budget": self.energy_budget,
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        Args:
            action: 16-dim continuous [0,1] array

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.clip(action, 0.0, 1.0)
        self._apply_actions(action)

        # Update dynamics
        self._update_dynamics()

        # Track energy
        self.energy_used += self._compute_energy_cost()

        # Advance time
        self.step_count += 1
        self.hour_of_day = (self.hour_of_day + self.dt / 3600.0) % 24.0

        # Compute reward
        reward, reward_info = self._compute_reward()

        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_step_info()
        info.update(reward_info)

        return obs, reward, terminated, truncated, info

    def get_coupling_heat_flows(self) -> Dict:
        """Calculate heat flow between zones for visualization."""
        flows = {}
        dt_factor = self.dt / 300.0
        for i, name_i in enumerate(self.ZONE_NAMES):
            for j, name_j in enumerate(self.ZONE_NAMES):
                if i < j and self.COUPLING[i, j] > 0:
                    conductance = self.COUPLING[i, j] * 0.005 * dt_factor
                    heat_flow = (self.zones[name_j]["temp"] - self.zones[name_i]["temp"]) * conductance
                    flows[f"{name_i}-{name_j}"] = round(heat_flow, 4)
        return flows

    def render(self):
        """Print current state to console."""
        print(f"\n=== Step {self.step_count} | Hour {self.hour_of_day:.1f} ===")
        print(f"Outdoor: {self._get_outdoor_temp():.1f}°C | Solar: {self._get_solar_radiation():.2f}")
        print(f"Energy: {self.energy_used:.1f}/{self.energy_budget:.0f}")
        for name in self.ZONE_NAMES:
            z = self.zones[name]
            c = self.COMFORT[name]
            temp_ok = "✓" if abs(z["temp"] - c["temp"]) <= c["temp_tol"] else "✗"
            humid_ok = "✓" if abs(z["humidity"] - c["humid"]) <= c["humid_tol"] else "✗"
            print(f"  {name:12s}: {z['temp']:.1f}°C{temp_ok} | {z['humidity']:.1f}%{humid_ok} | "
                  f"Occ:{z['occupancy']:.2f} | H:{z['heater']:.1f} C:{z['cooler']:.1f}")


# --- Standalone test ---
if __name__ == "__main__":
    env = BuildingHVACEnv(max_steps=50)
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()  # random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            env.render()
            print(f"  Reward: {reward:.3f}")

    print(f"\nTotal reward over 50 steps: {total_reward:.3f}")
    print(f"Coupling flows: {env.get_coupling_heat_flows()}")
