from os import stat
import numpy as np
import torch
import abc

from torch.distributions import Distribution
from torch.distributions.bernoulli import Bernoulli

from typing import Dict

class Policy(abc.ABC):

    @abc.abstractmethod
    def sample(self, **kwargs):
        return

    @abc.abstractmethod
    def state_dict(self):
        return


class EpsilonGreedy(Policy):

    def __init__(self,
                 eps_start: float = 1.0,
                 eps_min: float = 0.1,
                 eps_decay: int = int(1e6),
                 seed: int = 0):

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.n_steps: int = 0

        self.rng = np.random.RandomState(seed)

    def sample(self,
               terminations: Distribution,
               q_values: torch.Tensor,
               current_option: int) -> int:
        termination_dist = Bernoulli(terminations[current_option])
        is_terminated = termination_dist.sample()

        if not is_terminated:
            return current_option

        # Now epsilon-greedy
        option = None

        if self.rng.rand() < self.epsilon():
            option = np.random.choice(q_values.shape[-1])
        else:
            option = q_values.argmax(-1)

        return int(option)

    def state_dict(self) -> Dict[str, float]:
        return {"eps_start": self.eps_start,
                "eps_min": self.eps_min,
                "eps_decay": self.eps_decay,
                "seed": self.rng.seed,
                "n_steps": self.n_steps}

    def from_state_dict(self, state_dict: Dict[str, float]) -> None:
        self.eps_start = state_dict['eps_start']
        self.eps_min = state_dict['eps_min']
        self.eps_decay = state_dict['eps_decay']
        self.rng = np.random.RandomState(state_dict['seed'])
        self.n_steps = state_dict['n_steps']
        return None

    def epsilon(self) -> float:
        eps = self.eps_min + (self.eps_start - self.eps_min) * \
            np.exp(-self.n_steps / self.eps_decay)
        self.n_steps += 1
        return eps
