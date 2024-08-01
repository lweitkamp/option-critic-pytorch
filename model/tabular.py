from torch import nn
import torch

from torch.distributions import Categorical, Bernoulli

from math import exp


class TabularOptionCritic(nn.Module):
    def __init__(self, in_features, num_actions, num_options):

        super().__init__()
        self.Q = nn.Parameter(torch.zeros((in_features, num_options)))
        self.terminations = nn.Parameter(torch.zeros((in_features, num_options)))
        self.options_W = nn.Parameter(torch.zeros(in_features, num_options, num_actions))

        self.temperature = 1.0
        self.eps_min   = 0.1
        self.eps_start = 1.0
        self.eps_decay = int(1e6)
        self.eps_test  = 0.05
        self.num_steps = 0

    def get_state(self, obs):
        return obs

    def get_Q(self, state):
        return self.Q[state]

    def predict_option_termination(self, state, current_option):
        termination = self.terminations[state, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations[state].sigmoid() 

    def get_action(self, state, option):
        logits = self.options_W[state, option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps