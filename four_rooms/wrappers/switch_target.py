from gymnasium import Wrapper
from four_rooms.envs.four_rooms import FourRoomsEnv


class SwitchTarget(Wrapper):
    def __init__(self, env: FourRoomsEnv, new_target: tuple[int, int] = 3):
        """Switches the target location after 1000 episodes."""
        super().__init__(env)
        self._new_target = new_target
        self._episodes: int = 0

    def reset(self, seed: int | None = None, options: None = None):
        self._episodes += 1

        if self._episodes > 1000:
            self.env.unwrapped.target_location = self.new_target

        return self.env.reset(seed=seed, options=options)
