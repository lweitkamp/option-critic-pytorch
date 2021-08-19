import json
import torch
import gym
import supersuit

from gym.wrappers import AtariPreprocessing
from gym.spaces import Discrete

from option_critic_pytorch.experience_replay import ReplayBuffer, collect_random_experience
from option_critic_pytorch.logger import Logger, ReturnWrapper
from option_critic_pytorch.oc import OptionCriticConv, OptionCriticFeatures
from option_critic_pytorch.loss import actor_loss, critic_loss
from option_critic_pytorch.policy import EpsilonGreedy
from option_critic_pytorch.fourrooms import Fourrooms

POSSIBLE_LOSS_ACTOR = {
    'option_critic': actor_loss
}

POSSIBLE_LOSS_CRITIC = {
    'option_critic': critic_loss
}

POSSIBLE_OPTIMIZER = {
    'rmsprop': torch.optim.RMSprop,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def make_env(env_name: str, seed: int):
    if env_name == 'fourrooms':
        env = Fourrooms()
    else:
        env = gym.make(env_name)

    if len(env.observation_space.shape) > 1:
        env = AtariPreprocessing(env)
        env = supersuit.frame_stack_v1(env, 4)

    env = ReturnWrapper(env)
    env.seed(seed)
    return env


def from_config(filename: str, seed: int = 0):
    with open(filename, 'r') as f:
        config = json.load(f)

    # Set torch device.
    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    # Create the environment.
    env = make_env(config['env_name'], seed=seed)

    # Select between discrete and continuous action spaces.
    if isinstance(env.action_space, Discrete):
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0]

    # Select model based on observation space.
    if len(env.observation_space.shape) > 1:
        model = OptionCriticConv(in_channels=env.observation_space.shape[-1],
                                 n_options=config['n_options'],
                                 n_actions=env.action_space.n,
                                 device=device)
    else:
        model = OptionCriticFeatures(obs_dim=env.observation_space.shape[0],
                                     n_options=config['n_options'],
                                     n_actions=n_actions,
                                     hidden_dims=config['hidden_dims'],
                                     device=device)

    optimizer_args = {key.replace('optimizer_', ''): value for key,
                      value in config.items() if 'optimizer_' in key}
    optimizer = POSSIBLE_OPTIMIZER[config['optimizer']]
    optimizer = optimizer(model.parameters(), **optimizer_args)

    option_policy = EpsilonGreedy(eps_start=config['eps_start'],
                                  eps_min=config['eps_min'],
                                  eps_decay=config['eps_decay'],
                                  seed=seed)

    replay_buffer = ReplayBuffer(config['replay_capacity'], seed=seed)
    for experience in collect_random_experience(env, 32, 2):
        replay_buffer.push(*experience)

    logname = config['logger_name'] + f"_seed={seed}"
    logger = Logger(dir=config['logger_dir'], name=logname)

    out = {'env': env,
           'model': model,
           'option_policy': option_policy,
           'optimizer': optimizer,
           'replay_buffer': replay_buffer,
           'logger': logger,
           'init_step': config['init_step'],
           'max_steps': config['max_steps'],
           'update_critic': config['update_critic'],
           'batch_size': config['batch_size'],
           'gamma': config['gamma'],
           'term_reg': config['term_reg'],
           'ent_reg': config['ent_reg'],
           'polyak': config['polyak'],
           'actor_loss': POSSIBLE_LOSS_ACTOR[config['actor_loss']],
           'critic_loss': POSSIBLE_LOSS_CRITIC[config['critic_loss']]}

    return out


if __name__ == "__main__":
    out = from_config('config_files/CartPole-v0.json')
    print(out)

    from src.loop import train_loop
    train_loop(**out)
