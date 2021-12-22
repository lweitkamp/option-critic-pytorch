import numpy as np
import random
from collections import deque


class ReplayBuffer(object):
    rng : random.SystemRandom

    def __init__(self, capacity, seed=42):
        self.buffer = deque(maxlen=capacity)
        self.seed(seed)

    def seed(self, seed):
        random.seed(seed)

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.stack(obs), option, reward, np.stack(next_obs), done

    def __len__(self):
        return len(self.buffer)


def collect_random_experience(env, size, n_options, seed=42):
    rng = np.random.RandomState(seed)
    env.action_space.np_random.seed(seed)

    obs = env.reset()
    experience = []

    for _ in range(size):
        option = rng.randint(0, n_options)
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        experience.append((obs, option, reward, next_obs, done))
        obs = env.reset() if done else next_obs

    return experience
