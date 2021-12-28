# Option Critic
This repository is a PyTorch implementation of the paper "The Option-Critic Architecture" by Pierre-Luc Bacon, Jean Harb and Doina Precup [arXiv](https://arxiv.org/abs/1609.05140). It is mostly a rewriting of the original Theano code found [here](https://github.com/jeanharb/option_critic) into PyTorch. The main difference is that this implementation uses a single optimizer for both options and critic.

## Experiments

Configuration files for `CartPole-v0`, `CartPole-v1` and `PongNoFrameskip-v4` are available in the folder `config_files`. 
Results on these environments are shown below, with 10 seeds averaged for the CartPole environments and 2 seeds for the Pong environment.
Each plot displays the average sum of rewards of 10 episodes on the y-axis, evaluated with the greedy policy at #frames at the x-axis.

| **CartPole-v0** | **CartPole-v1** | **PongNoFrameskip-v4** |
|---|---|---|
|![](images/CartPole-v0_eval.png) | | |

## Examples
To make it easy, you can write a config file similar to those in `config_files` and load it using `src/from_config.py`. An example of this is provided in `main.py`.

Alternatively, you can run the following code snippet.

```python
import torch

from src.experience_replay import ReplayBuffer, collect_random_experience
from src.logger import Logger
from src.oc import OptionCriticConv
from src.loss import actor_loss, critic_loss
from src.policy import EpsilonGreedy
from src.from_config import make_env
from src.loop import train_loop

device = torch.device('cuda')
env = make_env('PongNoFrameskip-v4', seed=0)

model = OptionCriticConv(in_channels=env.observation_space.shape[-1],
                         n_options=4,
                         n_actions=env.action_space.n,
                         device=device)

optimizer = torch.optim.RMSprop(params=model.parameters(),
                                lr=0.00025)

# Create the option-selection policy (e-greedy)
option_policy = EpsilonGreedy(eps_start=1.0,
                              eps_min=0.05,
                              eps_decay=900000,
                              seed=0)

# Create the replay buffer and push a batch of experience in.
replay_buffer = ReplayBuffer(10000, seed=0)
for experience in collect_random_experience(env, 32, 2):
  replay_buffer.push(*experience)

logger = Logger(dir='tb', name='PongNoFrameskip-v4_seed=0')

out = train_loop(env,
                 model,
                 option_policy,
                 optimizer,
                 replay_buffer,
                 logger,
                 init_step=1,
                 max_steps=1000000,
                 max_episodes=None,
                 update_critic=4,
                 batch_size=32,
                 gamma=0.99,
                 term_reg=0.01,
                 ent_reg=0.01,
                 polyak=0.9,
                 actor_loss=actor_loss,
                 critic_loss=critic_loss,
                 env_fn=lambda: make_env('PongNoFrameskip-v4'))
```
