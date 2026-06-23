import numpy as np
from collections import deque


class MetricsTracker:
    def __init__(self, window_size=100):
        self.rewards = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)

    def add_episode_reward(self, reward):
        self.rewards.append(reward)

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_episode_length(self, length):
        self.episode_lengths.append(length)

    def get_average_reward(self):
        return np.mean(self.rewards) if self.rewards else 0.0

    def get_average_loss(self):
        return np.mean(self.losses) if self.losses else 0.0

    def get_average_episode_length(self):
        return np.mean(self.episode_lengths) if self.episode_lengths else 0.0
