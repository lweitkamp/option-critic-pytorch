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

if __name__=="__main__":
    # just a small test to check if the buffer wont overflow and can sample properly
    state  = np.random.rand(4, 84, 84)
    action = [i for i in range(32)]
    reward = [i for i in range(32)]
    next_state = state + 1
    done = [np.random.randint(0,2) for _ in range(32)]

    buffer = ReplayBuffer(int(1e3), seed=42)
    for j in range(int(1e3) + 10):
        buffer.push(
            np.random.rand(4, 84, 84),  # observation
            np.random.randint(0, 5),    # option
            np.random.randint(-1, 2),   # reward
            np.random.rand(4, 84, 84),  # next observation
            np.random.randint(0, 2)     # episode done
        )
        if j % 1000 == 0:
            print(len(buffer))


    assert len(buffer) == int(1e3), "overflow encountered"

    for l in range(40):
        items = buffer.sample(batch_size=32)
        assert items[0].shape == (32, 4, 84, 84)
        if l % 20 == 0:
            assert len(buffer) == int(1e3)
