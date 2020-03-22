import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from utils import to_tensor


class OptionCritic(nn.Module):
    """
    Implementation of the Option Critic model (harb, et al. 2017).
    This specific version has a wholly independent network per option,
    to try and answer the following question:

    - Why do options, in deep end-to-end learning, reach a local optima in which
    each option is 'exactly' the same?

    My hypothesis is that this is due to the following points:

    (1) end-to-end learning with shared features might pull the options together in earlier layers.
        It will take 'too much' advantage of a shared representation.
    (2) There is no effort at all to make options dissimilar, hence there is no reason it should go
        and learn different interpretable and useful options.

    This implementation is meant to fix (1), and see if that works. This might be backed up by the fact
    that the tabular implementation 'does' seem to learn interpretable options.
    """
    def __init__(self,
                in_features,
                num_options,
                num_actions,
                temperature=1.0,
                device='cpu',
                testing=False):

        super(OptionCritic, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.temperature = temperature
        self.option_idx  = [i for i in range(num_options)]

        self.features = [
            nn.Sequential(
                nn.Linear(in_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            ) for _ in range(num_options)
        ]

        self.Q    = [nn.Linear(64, 1) for _ in range(4)]
        self.beta = [nn.Linear(64, 1) for _ in range(4)]
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.device = device
        self.to(device)
        self.train(not testing)

    def get_state(self, observations, options=None):
        """
        observations: batch_size x feature_dim OR feature_dim
        options     : integer or list of integers

        Retrieve the 'state' from an observation of the environment.
        """
        if options is not None:
            states = [self.features[op](obs) for obs, op in zip(observations, options)]
        else:
            states = [self.features[op](observations) for op in self.option_idx]
        return torch.stack(states)

    def get_Q(self, states, options):
        """
        States:  batch_size x num_features
        options: integer or list of integers

        Retrieve the Q_omega (state-option) value.
        """
        if not isinstance(options, list):
            options = [options]

        Qso = torch.cat([option_critic.Q[o](s) for s, o in zip(states, options)])
        return Qso

    def predict_option_termination(self, state, option):
        """
        state:  single state (num_features)
        option: integer

        Predict if the (current) option will terminate given a state.
        """
        beta = self.beta[option](state[option]).sigmoid()
        termination = bool(Bernoulli(beta).sample().item())
        return termination

    def get_beta(self, states, options=None):
        """
        states:  batch_size x num_features
        options: integer or list of integers

        Retrieve the beta termination probability of an option given a state.
        """
        options = options or self.option_idx

        betas = [self.beta[o](s) for s, o in zip(states, options)]
        betas = torch.cat(betas)
        return betas.sigmoid()

    def get_action(self, state, option):
        """
        state:  single state (num_features)
        option: integer

        Select an action to take, conditioned on state and option.
        """
        logits = state[option] @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action  = action_dist.sample()
        logp    = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), logp, entropy

    def greedy_option(self, observation):
        """
        observations: single observation (feature_dim)

        Choose the option that maximizes the Q_omega value of observation.
        """
        with torch.no_grad():
            states = [self.features[option](observation) for option in self.option_idx]
            qs     = [self.Q[option](states[option]) for option in self.option_idx]
            qs     = torch.stack(qs, dim=1).squeeze(-1)
            return qs.argmax(-1).item()

    def critic_loss(self, data_batch, args):
        """
        data_batch: buffer with states,...
        args:       argparse

        Given a batch of experience, calculate the critic loss.
        """
        obs, options, rewards, next_obs, dones = data_batch
        obs         = to_tensor(obs)
        options     = list(options)
        next_obs    = to_tensor(next_obs)
        rewards     = to_tensor(rewards)
        masks       = 1 - to_tensor(dones)

        states = option_critic.get_state(obs, options)
        qs     = option_critic.get_Q(states, options)

        next_states_prime = option_critic.get_state(next_obs, options)
        next_q_prime      = option_critic.get_Q(next_states_prime, options)

        next_states = option_critic.get_state(next_obs, options)
        next_beta   = option_critic.get_beta(next_states, options).detach()

        gt = rewards + masks * args.gamma * \
            ((1 - next_beta) * next_q_prime + next_beta * next_q_prime)

        td_err = (qs - gt.detach()).pow(2).mul(0.5).mean()
        return td_err

