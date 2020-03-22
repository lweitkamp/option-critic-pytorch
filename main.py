import numpy as np
import argparse
import torch
from copy import deepcopy

from option_critic import OptionCritic

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor, LinearSchedule
from logger import Logger

import time

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--xi', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')

def run(args):
    env = make_env(args.env)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    agent = OptionCritic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        device=device
    )
    # Create a prime network for more stable Q values
    agent_prime = deepcopy(agent)

    optim = torch.optim.RMSprop(agent.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCritic.__name__}-{args.env}-{args.exp}-{time.ctime()}")
    e_greedy = LinearSchedule(args.epsilon_start, args.epsilon_min, args.epsilon_decay)

    steps = 0 ;
    while steps < args.max_steps_total:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}

        obs   = env.reset()
        state = agent.get_state(to_tensor(obs))
        greedy_option = agent.greedy_option(state)
        option = 0

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = next(e_greedy)

            if option_termination:
                option_lengths[option].append(curr_op_len)
                option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = agent.get_action(state, option)

            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, option, reward, next_obs, done)

            next_state = agent.get_state(to_tensor(next_obs))

            option_termination = agent.predict_option_termination(next_state, option)
            greedy_option      = agent.greedy_option(next_state)

            optim.zero_grad()
            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = agent.actor_loss(state, option,
                    logp, entropy, reward, done, next_state, epsilon, args)
                actor_loss.backward()

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = agent.critic_loss(data_batch, args)
                    critic_loss.backward()

                optim.step()

                if steps % args.freeze_interval == 0:
                    agent_prime.load_state_dict(agent.state_dict())

            # update global steps etc
            rewards += reward
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            state = next_state
            obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)

if __name__=="__main__":
    args = parser.parse_args()
    run(args)
