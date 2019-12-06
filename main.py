import numpy as np
import argparse
import torch
from copy import deepcopy

from option_critic import OptionCritic
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor, Logger

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
#parser.add_argument('--rom', default=defaults.ROM, help='ROM to run (default: %(default)s)') # 'bug.bin?
parser.add_argument('--env', default='BreakoutNoFrameskip-v4', help='ROM to run')
parser.add_argument('--epochs', type=int, default=8000, help='Number of training epochs')
parser.add_argument('--steps-per-epoch', type=int, default=250000, help='Number of steps per epoch')
parser.add_argument('--test-length', type=int, default=130000, help='Number of steps per test')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
#parser.add_argument('--display-screen', dest="display_screen", action='store_true', default=False, help='Show the game screen.')
#parser.add_argument('--testing', dest="testing", action='store_true', default=False, help='Signals running test.')
parser.add_argument('--experiment-prefix', dest="experiment_prefix", default=None, help='Experiment name prefix ' '(default is the name of the game)')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
#parser.add_argument('--update-rule', type=str, default='rmsprop', help=('adam|adadelta|rmsprop|sgd)'))
parser.add_argument('--learning-rate',type=float, default=.00025, help='Learning rate')
parser.add_argument('--rms-decay', type=float, default=.95, help='Decay rate for rms_prop')
parser.add_argument('--rms-epsilon', type=float, default=.01, help='Denominator epsilson for rms_prop')
parser.add_argument('--clip-delta', type=float, default=1.0, help=('Max absolute value for Q-update delta value.'))
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=1000000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--phi-length', type=int, default=4, help=('Number of recent frames used to represent'))
parser.add_argument('--max-history', type=int, default=1000000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=10000, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
#parser.add_argument('--replay-start-size', type=int, default=50000, help=('Number of random steps before training.'))
parser.add_argument('--resize-method', type=str, default='scale', help=('crop|scale'))
parser.add_argument('--crop-offset', type=str, default=18, help=('crop offset.'))
parser.add_argument('--nn-file', type=str, default=None, help='Pickle file containing trained net.')
parser.add_argument('--cap-reward', type=bool, default=True, help=('true|false'))
parser.add_argument('--death-ends-episode', type=bool, default=True, help=('true|false'))
parser.add_argument('--max-start-nullops', type=int, default=30, help=('Maximum number of null-ops at the start'))
parser.add_argument('--folder-name', type=str, default="", help='Name of pkl files destination (within models/)')
parser.add_argument('--termination-reg', type=float, default=0.0, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.0, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=8, help=('Number of options to create.'))
parser.add_argument('--actor-lr', type=float, default=0.00025, help=('Actor network learning rate'))
#parser.add_argument('--double-q', type=bool, default=False, help='Train using Double Q networks.')
parser.add_argument('--mean-frame', type=bool, default=False, help='Use pixel-wise mean consecutive frames as images.')
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')
parser.add_argument('--baseline', type=bool, default=False, help='use baseline in actor gradient function.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e7), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for numpy, torch, random.')

parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')


def run(args):
    env = make_env(args.env)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    option_critic = OptionCritic(
        in_channels=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim_critic = torch.optim.RMSprop(option_critic.parameters(),
                                        lr=args.learning_rate,
                                        eps=args.rms_epsilon,
                                        weight_decay=args.rms_decay)
    optim_actor  = torch.optim.SGD(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCritic.__name__}-{args.env}") # [TODO]: add time.time() to name

    steps = 0
    while steps < args.max_steps_total:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}

        obs   = env.reset()
        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done or ep_steps > args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = option_critic.get_action(state, current_option)

            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)

            old_state = state
            state = option_critic.get_state(to_tensor(next_obs))

            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                #actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                #    reward, done, next_obs, option_critic, option_critic_prime, optim_actor, args)
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic, optim_actor, args)

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    #critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, optim_critic, args)
                    critic_loss = critic_loss_fn(option_critic, option_critic, data_batch, optim_critic, args)

                if steps % args.freeze_interval == 0:
                    # if freeze_interval > 999..?
                    # update target parameters
                    pass

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1

            # LOGGING
            ## what to log on a step-by-step basis?
            # critic_loss, actor_loss, ...

        # EP LOGGING
        logger.log_episode(steps, rewards, option_lengths, ep_steps)

if __name__=="__main__":
    args = parser.parse_args()
    run(args)