import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from utils import to_tensor

class OptionCritic(nn.Module):
    def __init__(self,
                in_channels,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCritic, self).__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )

        self.Q            = nn.Linear(512, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
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


def critic_loss(model, model_prime, data_batch, optim, args):
    """
    
    Args:
        model ([type]): [description]
        model_prime ([type]): [description]
        data_batch ([type]): [description]
        optim ([type]): [description]
        args ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options)
    rewards   = torch.FloatTensor(rewards)
    masks     = 1 - torch.FloatTensor(dones)

    # get next state normal and prime (probably target network?)
    states = model.get_state(to_tensor(obs))
    next_states = model.get_state(to_tensor(next_obs))
    next_states_prime = model_prime.get_state(to_tensor(next_obs))

    # Get the termination probabilities of current and next state, and of the specific option
    termination_probs = model.get_terminations(states).detach()
    options_term_prob = termination_probs[batch_idx, options]
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # sample 0/1 from the option of the state
    termination_sample = Bernoulli(options_term_prob)

    # Get corresponding Q values for current and next state
    Q = model.get_Q(states)
    next_Q       = model.get_Q(next_states)
    next_Q_prime = model_prime.get_Q(next_states_prime) # detach?

    # So for the return we use the prime network which should learn slower
    y = rewards + masks * args.gamma * \
        (1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + \
             next_options_term_prob  * next_Q_prime.max(dim=-1)[0]

    y = y.detach()

    # to update Q we want to use the actual network, not theprime
    td_err = y - Q[batch_idx, options]

    # optionally clip delta.. what is it? TODO
    quadratic_part = torch.min(td_err.abs(), torch.zeros_like(td_err).add(args.clip_delta))
    td_cost = 0.5 * quadratic_part.pow(2) + args.clip_delta * (torch.abs(td_err) - quadratic_part)
    td_cost = td_cost.sum()

    optim.zero_grad()
    td_cost.backward()
    optim.step()

    return td_cost.item()

def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, optim, args):
    """[summary]
    
    Args:
        obs ([type]): [description]
        option ([type]): [description]
        logp ([type]): [description]
        entropy ([type]): [description]
        reward ([type]): [description]
        done (function): [description]
        next_obs ([type]): [description]
        model ([type]): [description]
        model_prime ([type]): [description]
        optim ([type]): [description]
        args ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    state = model.get_state(to_tensor(obs))
    next_state = model.get_state(to_tensor(next_obs))
    next_state_prime = model_prime.get_state(to_tensor(next_obs))

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    y = reward + (1 - done) * args.gamma * \
        (1 - next_option_term_prob) * next_Q_prime[option] + \
             next_option_term_prob  * next_Q_prime.max(dim=-1)[0]

    y = y.detach()

    termination_loss = option_term_prob * (Q[option] - Q.max(dim=-1)[0] + args.termination_reg)
    policy_loss = (-logp * (y - Q[option])).sum() - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    optim.zero_grad()
    actor_loss.backward()
    optim.step()
    return actor_loss.item()
