import gymnasium as gym
import numpy as np


class Box(gym.spaces.Box):
  def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
    super(Box, self).__init__(low=low, high=high, shape=shape, dtype=dtype,
                              seed=seed)
