import gym
import numpy as np
import torch

from gym.wrappers import AtariPreprocessing, TransformReward
from gym.wrappers import FrameStack as FrameStack_


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
