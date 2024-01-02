import gym as legacy_gym
import gymnasium as gym
import numpy as np


class Box(legacy_gym.spaces.Box, gym.spaces.Box):
  def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
    # legacy_gym.spaces.Box.__init__(self, low, high, shape, dtype, seed)
    gym.spaces.Box.__init__(self, low, high, shape, dtype, seed)
