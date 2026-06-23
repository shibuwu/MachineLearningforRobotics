import json
import os
import gymnasium as gym
import numpy as np
import torch

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Ensure directories exist
for d in [VIDEOS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

class ObservationNormalizer:
    """Online running mean/std normalizer. Essential for stable A2C/PPO."""
    def __init__(self, obs_dim, epsilon=1e-8):
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.m2 = np.zeros(obs_dim, dtype=np.float32)

    def update(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        self.count += 1
        delta = observation - self.mean
        self.mean += delta / self.count
        delta2 = observation - self.mean
        self.m2 += delta * delta2

    def normalize(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if self.count < 2:
            return observation
        variance = self.m2 / max(self.count - 1, 1)
        std = np.sqrt(np.maximum(variance, self.epsilon))
        # Clip to prevent outliers from exploding gradients
        return np.clip((observation - self.mean) / std, -5.0, 5.0)

    def state_dict(self):
        return {
            "obs_dim": self.obs_dim,
            "epsilon": self.epsilon,
            "count": self.count,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
        }

    def load_state_dict(self, state):
        self.obs_dim = state["obs_dim"]
        self.epsilon = state["epsilon"]
        self.count = state["count"]
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.m2 = np.asarray(state["m2"], dtype=np.float32)

def load_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        # Default stable parameters if file is missing
        return {
            "env_id": "LunarLander-v3",
            "actor_lr": 3e-4,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "num_episodes": 10000,
            "max_ep_steps": 600,
            "hidden_dim": 256,
            "random_seed": 42
        }
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(env_id, render_mode="rgb_array"):
    """Corrected environment creation for stable rendering."""
    try:
        return gym.make(env_id, render_mode=render_mode)
    except Exception as exc:
        raise RuntimeError(f"Failed to create {env_id}. Use: pip install 'gymnasium[box2d]'") from exc

def reset_env(env, seed=None):
    """Standardized reset for Gymnasium."""
    obs, _ = env.reset(seed=seed)
    return obs

def step_env(env, action):
    """Standardized step for Gymnasium."""
    obs, reward, terminated, truncated, info = env.step(action)
    # Stability Trick: Reward Scaling
    # Dividing rewards by a constant helps the Critic converge faster
    scaled_reward = reward / 10.0 
    return obs, scaled_reward, terminated, truncated, info

def render_frame(env):
    """Safe render call for Gymnasium."""
    return env.render()