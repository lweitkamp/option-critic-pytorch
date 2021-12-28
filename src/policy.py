from os import stat
import numpy as np
import torch
import abc

from torch.distributions import Distribution
from torch.distributions.bernoulli import Bernoulli

from typing import Dict, Optional


class Policy(abc.ABC):

    @abc.abstractmethod
    def sample(self, **kwargs):
        return

    @abc.abstractmethod
    def greedy_action(self, **kwargs):
        return

    @abc.abstractmethod
    def state_dict(self):
        return

    def seed(self, seed: int = 42):
        return


class EpsilonGreedy(Policy):

    rng: np.random.RandomState = None

    def __init__(self,
                 eps_start: float = 1.0,
                 eps_min: float = 0.1,
                 eps_decay: int = int(1e6),
                 seed: int = 42):

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.n_steps: int = 0
        self.seed(seed)

    def greedy_action(self,
                      terminations: Distribution,
                      q_values: torch.Tensor,
                      current_option: int):
        """Take actions according to the greedy policy. In the case
        of e-greedy, this is equal to sampling with the e-min value."""
        return self.sample(terminations, q_values, current_option, epsilon=self.eps_min)

    def sample(self,
               terminations: Distribution,
               q_values: torch.Tensor,
               current_option: int,
               epsilon: Optional[float] = None) -> int:
        termination_dist = Bernoulli(terminations[current_option])
        is_terminated = termination_dist.sample()

        if not is_terminated:
            return current_option

        if epsilon is None:
            epsilon = self.epsilon()

        if self.rng.rand() < epsilon:
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

    def epsilon(self, update: bool = True) -> float:
        eps = self.eps_min + (self.eps_start - self.eps_min) * \
              np.exp(-self.n_steps / self.eps_decay)

        if update:
            self.n_steps += 1
        return eps

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
