import torch
import numpy as np
import torch.nn as nn

from typing import Dict, Tuple


class OptionCriticBase(nn.Module):

    def __init__(self,
                 net: nn.Module,
                 q: nn.Module,
                 terminations: nn.Module,
                 options: nn.Module,
                 n_options: int,
                 n_actions: int,
                 device: torch.device):
        super().__init__()

        self.net = net
        self.q = q
        self.terminations = terminations
        self.options = options
        self.n_options = n_options
        self.n_actions = n_actions
        self.device = device
        self.to(device)

    def reshape(self, x: np.ndarray) -> np.ndarray:
        return x

    def forward(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        obs = self.reshape(obs)
        obs = torch.Tensor(obs).to(self.device)
        features = self.net(obs)

        # Policy-Over-Options
        q = self.q(features)

        # Option-Termination
        terminations = torch.sigmoid(self.terminations(features))

        # Option-Policies
        options = self.options(features)
        options = options.reshape(-1, self.n_options, self.n_actions)
        option_logits = torch.log_softmax(options, dim=-1)

        out = {'q': q.squeeze(),
               'terminations': terminations.squeeze(),
               'option_logits': option_logits.squeeze()}

        return out


class OptionCriticFeatures(OptionCriticBase):
    def __init__(self,
                 obs_dim: int,
                 n_options: int,
                 n_actions: int,
                 device: torch.device,
                 hidden_dims: Tuple[int] = (32, 64)):

        if len(hidden_dims) == 0:
            def net(x): return x
            hidden_dims = None, obs_dim
        else:
            net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU()
            )

        q = nn.Linear(hidden_dims[1], n_options)
        terminations = nn.Linear(hidden_dims[1], n_options)
        options = nn.Linear(hidden_dims[1], n_options * n_actions)

        super().__init__(net=net,
                         q=q,
                         terminations=terminations,
                         options=options,
                         n_options=n_options,
                         n_actions=n_actions,
                         device=device)

    def reshape(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        return x


class OptionCriticConv(OptionCriticBase):
    def __init__(self,
                 in_channels: int,
                 n_options: int,
                 n_actions: int,
                 device: torch.device,
                 channel_first: bool = True):

        net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(7 * 7 * 64, 512),  # magic number based on Atari
            nn.ReLU()
        )

        q = nn.Linear(512, n_options)
        terminations = nn.Linear(512, n_options)
        options = nn.Linear(512, n_options * n_actions)

        self.channel_first = channel_first

        super().__init__(net=net,
                         q=q,
                         terminations=terminations,
                         options=options,
                         n_options=n_options,
                         n_actions=n_actions,
                         device=device)

    def reshape(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:  # No channels, add singleton dims.
            x = np.expand_dims(x, axis=(0, 1))
        elif x.ndim == 3:  # Not a batch, add batch dim.
            x = np.expand_dims(x, axis=0)

        if self.channel_first:
            x = np.transpose(x, (0, 3, 1, 2))
        return x
