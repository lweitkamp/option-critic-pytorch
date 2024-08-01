import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FourRoomsEnv(gym.Env):
    """A simple Four Rooms environment as described in
    Between MDPs and semi-MDPs by Sutton, Precup, & Singh (1999).

    The code itself is a slightly modified version of the original:
    https://github.com/jeanharb/option_critic/blob/master/fourrooms/fourrooms.py.

    Each state is defined as an index that has a semantic reference to a coordinate,
    the agent can perform one of four actions (up, down, left, right), and the goal
    is to reach the target location. Each episode is truncated after 1000(!) steps
    total are taken - it is highly likely that even a random policy finishes.
    """

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

        coordinates = np.stack(np.where(self.layout == 1), axis=1)
        self.to_coordinate = {i: tuple(x) for i, x in enumerate(coordinates)}
        self.to_state = {coord: i for i, coord in self.to_coordinate.items()}

        self.observation_space = spaces.Discrete(len(self.to_coordinate))
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.agent_location: int | None = None  # Randomized at .reset()
        self.target_location: int = 62

        self.init_states = set(range(self.observation_space.n))
        self.init_states.remove(self.target_location)
        
        self._steps: int = 0

    def _get_info(self) -> dict:
        return {"steps": self._steps}

    def render(self):
        grid = (self.layout == 0).astype(int)
        grid[*self.to_coordinate[self.agent_location]] = 2
        grid[*self.to_coordinate[self.target_location]] = 3
        return grid

    def reset(self, seed: int | None = None, options: None = None) -> tuple[int, dict]:
        """Reset the environment to start a new episode. This only resets the agent
        location, the target location is fixed."""
        super().reset(seed=seed)
        self._steps = 0

        # Ensure target and agent are not in the same location.
        self.agent_location = self.np_random.choice(list(self.init_states))
        while self.target_location == self.agent_location:
            self.agent_location = self.np_random.choice(list(self.init_states))

        info = self._get_info()
        return self.agent_location, info

    def step(self, action):
        """The environment takes a step given an action.

        If the agent hits a wall, the agent stays in the same location.
        Otherwise, there is a 1/3 chance the agent will move in a random direction.
        This leaves a 2/3 chance the agent will move in the intended direction.
        """
        self._steps += 1

        direction = self._action_to_direction[action]
        next_cell = self.to_coordinate[self.agent_location] + np.array(direction)

        if tuple(next_cell) in self.to_state:
            if self.np_random.uniform() < 1 / 3:
                next_cell = self.np_random.choice(self.get_neighbours())
            self.agent_location = self.to_state[tuple(next_cell)]

        done = self.agent_location == self.target_location
        reward = 1.0 if done else 0.0

        info = self._get_info()

        return self.agent_location, reward, done, False, info

    def get_neighbours(self):
        """Get the neighbours of the agent."""
        neighbours = [
            self.to_coordinate[self.agent_location] + np.array(d)
            for d in self._action_to_direction
        ]
        return [n for n in neighbours if tuple(n) in self.to_state]
