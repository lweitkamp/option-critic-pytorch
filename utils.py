import gym
import numpy as np
import torch
import logging
import os

from gym.wrappers import AtariPreprocessing, TransformReward
from gym.wrappers import FrameStack as FrameStack_

# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

def make_env(env):
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(
            env,
            screen_size=84,
            grayscale_obs=True,
            frame_skip=1,
            scale_obs=True, # It also limits memory optimization benefits of FrameStack Wrapper. <-- ?
            terminal_on_life_loss=True, 
    )
    env = TransformReward(env, lambda r: np.clip(r, -1, 1)) # option-critic uses clipping
    env = FrameStack(env, 4)
    return env

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).unsqueeze(0)
    return obs


class Logger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
            )

    def log_episode(self, steps, reward, option_lenghts, ep_steps):
        logging.info(f"> ep done. total_steps={steps} | reward={reward} | episode_steps={ep_steps}")
        # TODO: log to tensorflow

    def log_data(self, data):
        # log per step
        # - reward 
        # - entropy
        # - Q values..?
        pass

if __name__=="__main__":
    logger = Logger(logdir='runs/', run_name='test_model-test_env')
    steps = 200 ; reward = 5 ; option_lengths = {opt: np.random.randint(0,5,size=(5)) for opt in range(5)} ; ep_steps = 50
    logger.log_episode(steps, reward, option_lengths, ep_steps)