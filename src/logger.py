import os.path
from collections import defaultdict
import gym
from torch.utils.tensorboard import SummaryWriter

from typing import Dict


class ReturnWrapper(gym.Wrapper):
    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################
    def __init__(self, env):
        super().__init__(env)
        self.total_rewards = 0
        self.steps = 0
        self.n_episodes = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        self.steps += 1
        if done:
            info['returns/episodic_reward'] = self.total_rewards
            info['returns/episodic_length'] = self.steps
            self.total_rewards = 0
            self.steps = 0
            self.n_episodes += 1
        else:
            info['returns/episodic_reward'] = None
            info['returns/episodic_length'] = None
        return obs, reward, done, info


class Logger:
    def __init__(self, dir: str, name: str, add_timestamp: bool = True):
        if add_timestamp:
            name = f"{name}"

        self.writer = SummaryWriter(os.path.join(dir, name))
        self.option_dict = defaultdict(list)
        self.prev_option = 0
        self.option_counter = 0

    def log(self, info: Dict[str, float], step: int) -> None:
        if 'option' in info:
            self.log_options(info['option'], step)
            del info['option']

        for key, value in info.items():
            if value is not None:
                self.writer.add_scalar(
                    tag=key, scalar_value=value, global_step=step)

    def log_options(self, option: int, step) -> None:
        if option == self.prev_option:
            self.option_counter += 1
            return

        self.option_dict[self.prev_option].append(self.option_counter)
        self.writer.add_scalar(
            f'options/avg_length_{self.prev_option}', self.option_counter, step)
        self.prev_option = option
        self.option_counter = 0
