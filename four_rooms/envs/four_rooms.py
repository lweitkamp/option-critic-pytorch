import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FourRoomsEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.layout = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        self.observation_space = spaces.Box(
            low=0, high=max(self.layout.shape), shape=(2,), dtype=int
        )

        # Create a quick lookup table and one for sampling.
        self.states = np.stack(np.where(self.layout == 1), axis=1)
        self.legal_states = set([(tuple([int(x), int(y)])) for (x, y) in self.states])

        # We have 4 actions, up, down, left, right.
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array((-1, 0)),
            1: np.array((1, 0)),
            2: np.array((0, -1)),
            3: np.array((0, 1)),
        }

        self.window = None
        self.clock = None

        self._agent_location: tuple[int, int] | None = None
        self._target_location: tuple[int, int] | None = None
        self._steps: int = 0

    def _get_info(self):
        return {
            "distance": float(
                np.linalg.norm(self._agent_location - self._target_location, ord=1)
            ),
            "target_location": self._target_location,
        }

    def reset(self, seed: int | None = None, options: None = None) -> tuple[int, dict]:
        super().reset(seed=seed)
        self._steps = 0

        # Randomize the target location if it is not fixed at init.
        if self._target_location is None:
            self.set_target_location(self.np_random.choice(self.states))

        # Ensure target and agent are not in the same location.
        self._agent_location = self.np_random.choice(self.states)
        while np.array_equal(self._target_location, self._agent_location):
            self._agent_location = self.np_random.choice(self.states)

        info = self._get_info()

        return self._agent_location, info

    def render(self):
        grid = np.copy(self.layout == 0).astype(int)
        grid[*self._agent_location] = -1
        grid[*self._target_location] = -2
        return grid

    def step(self, action):
        self._steps += 1
        direction = self._action_to_direction[action]

        # Check if the agent is not running against a wall (treat as no-op).
        new_x, new_y = self._agent_location + direction
        if (int(new_x), int(new_y)) in self.legal_states:
            self._agent_location = np.array([new_x, new_y])

        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if done else 0.0

        info = self._get_info()

        return self._agent_location, reward, done, False, info

    def set_target_location(self, location: tuple[int, int]):
        self._target_location = location