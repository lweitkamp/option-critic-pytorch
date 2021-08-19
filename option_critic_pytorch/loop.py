import copy
import gym
import torch

from option_critic_pytorch.experience_replay import ReplayBuffer
from option_critic_pytorch.loss import actor_loss, critic_loss
from option_critic_pytorch.policy import Policy
from option_critic_pytorch.logger import Logger

from typing import Callable, Optional


def train_loop(env: gym.Env,
               model: torch.nn,
               option_policy: Policy,
               optimizer: torch.optim.Optimizer,
               replay_buffer: ReplayBuffer,
               logger: Logger,
               init_step: int = 1,
               max_steps: int = int(1e6),
               max_episodes: Optional[int] = None,
               update_critic: int = 4,
               batch_size: int = 32,
               gamma: float = 0.99,
               term_reg: float = 0.01,
               ent_reg: float = 0.01,
               polyak: float = 1,
               actor_loss: Callable = actor_loss,
               critic_loss: Callable = critic_loss,
               env_fn: Callable = None):

    # Create target model.
    model_target = copy.deepcopy(model)

    # Create logger

    obs = env.reset()
    option = 0

    for step in range(init_step, max_steps):
        out = model(obs)

        # Sample option.
        option = option_policy.sample(out['terminations'], out['q'], option)

        # Create action distribution, sample action, calculate log_prob & entropy.
        action_dist = torch.distributions.Categorical(
            logits=out['option_logits'][:, option])
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
        if step % update_critic == 0:
            data_batch = replay_buffer.sample(batch_size)
            info.update(critic_loss(model, model_target,
                        optimizer, data_batch, gamma))

        # Polyak average target model.
        for mt, m in zip(model_target.parameters(), model.parameters()):
            mt.data.copy_(mt.data * (1 - polyak) + m.data * polyak)

        obs = env.reset() if done else next_obs
        logger.log(info, step)

        # If we have episode-based limit, set it here.
        if max_episodes is not None and env.n_episodes >= max_episodes:
            break

    return {'model': model,
            'option_policy': option_policy,
            'optimizer': optimizer,
            'steps': step}
