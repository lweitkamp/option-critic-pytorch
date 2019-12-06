import numpy as np
import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, option, reward, next_obs, done):
        obs      = np.expand_dims(obs, 0)
        next_obs = np.expand_dims(next_obs, 0)
        self.buffer.append((obs, option, reward, next_obs, done))
    
    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
        return np.concatenate(obs), option, reward, np.concatenate(next_obs), done
    
    def __len__(self):
        return len(self.buffer)
