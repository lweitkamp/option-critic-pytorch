import copy
import gym
import torch
import numpy as np
import random
import time

from src.experience_replay import ReplayBuffer
from src.loss import actor_loss, critic_loss
from src.policy import Policy
from src.logger import Logger

from typing import Callable, Optional


def train_loop(env: gym.Env,
               model: torch.nn,
               option_policy: Policy,
               optimizer: torch.optim.Optimizer,
               replay_buffer: ReplayBuffer,
               logger: Logger,
               init_step: int = 1,
               max_steps: int = int(1e6),
               update_critic: int = 4,
               batch_size: int = 32,
               gamma: float = 0.99,
               term_reg: float = 0.01,
               ent_reg: float = 0.01,
               polyak: float = 1,
               actor_loss: Callable = actor_loss,
               critic_loss: Callable = critic_loss,
               print_every: int = 100000,
               eval_env: Optional[gym.Env] = None,
               seed: int = 42):

    # Seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)
    eval_env.seed(seed)
    replay_buffer.seed(seed)
    option_policy.seed(seed)


    # Create target model.
    model_target = copy.deepcopy(model)
    obs = env.reset()
    option = 0

    for step in range(init_step, max_steps):
        out = model(obs)

        # Sample option.
        option = option_policy.sample(out['terminations'], out['q'], option)


        # Create action distribution, sample action, calculate log_prob & entropy.
        action_dist = torch.distributions.Categorical(
            logits=out['option_logits'][option])
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Take a step in the environment.
        next_obs, reward, done, info = env.step(action.item())
        info.update({'option': option})

        # Add experience to buffer.
        replay_buffer.push(obs, option, reward, next_obs, done)

        # Update actor.
        info.update(actor_loss(obs, option, logp, entropy, reward, done,
                    next_obs, model, model_target, optimizer, gamma, term_reg, ent_reg))

        # Update critic if condition is met.
        if step % update_critic == 0 and len(replay_buffer) > batch_size:
            data_batch = replay_buffer.sample(batch_size)
            info.update(critic_loss(model, model_target,
                        optimizer, data_batch, gamma))

        # Polyak average target model.
        for mt, m in zip(model_target.parameters(), model.parameters()):
            mt.data.copy_(mt.data * polyak + m.data * (1 - polyak))

        obs = env.reset() if done else next_obs
        logger.log(info, step)

        if step % print_every == 0 and eval_env:
            eval_out = eval_loop(model, option_policy, eval_env)
            logger.log(eval_out, step)
            print(f'Step {step} | Average Return {eval_out["eval/avg_ret"]}')

    return {'model': model,
            'option_policy': option_policy,
            'optimizer': optimizer,
            'steps': step}


@torch.no_grad()
def eval_loop(model: torch.nn,
              option_policy: Policy,
              env: gym.Env,
              num_episodes: int = 10):
    """With Greedy Option selection"""
    start = time.time()

    returns = []
    for episode in range(num_episodes):
        episodic_return = 0
        done = False
        obs = env.reset()
        option = 0
        while not done:
            out = model(obs)
            option = option_policy.greedy_action(out['terminations'], out['q'], option)

            action_dist = torch.distributions.Categorical(
                logits=out['option_logits'][option])
            action = action_dist.sample()

            obs, reward, done, info = env.step(action.item())

            episodic_return += reward
        returns.append(episodic_return)
    total_time = time.time() - start
    return {'eval/avg_ret': sum(returns) / num_episodes, 'eval/computational_time': total_time}

