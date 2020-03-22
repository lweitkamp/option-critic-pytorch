import gym
import numpy as np
import torch

from fourrooms import Fourrooms


class LinearSchedule:
    """
    Linear scheduler, for epsilon-greedy option selection.
    """
    def __init__(self, e_start=1.0, e_min=0.1, e_decay=int(1e6)):
        self.num_steps = 0
        self.e_min   = e_min
        self.e_start = e_start
        self.e_decay = e_decay

    def __next__(self):
        epsilon = self.e_min + (self.e_start - self.e_min) * np.exp(-self.num_steps / self.e_decay)
        self.num_steps += 1
        return epsilon


def make_env(env_name):

    if env_name == 'fourrooms':
        env = Fourrooms()

    env = gym.make(env_name)
    return env

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs
