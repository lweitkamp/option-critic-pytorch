from gymnasium import Wrapper
from four_rooms.envs.four_rooms import FourRoomsEnv


class SwitchTarget(Wrapper):
    def __init__(
        self,
        env: FourRoomsEnv,
        original_target: tuple[int, int] = (7, 9),
        new_target: tuple[int, int] = (3, 3),
    ):
        super().__init__(env)
        self._original_target = original_target
        self._new_target = new_target
        self._episodes: int = 0
        
        self.env.unwrapped.set_target_location(self._original_target)

    def reset(self, seed: int | None = None, options: None = None):
        self._episodes += 1

        if self._episodes > 40:
            self.env.unwrapped.set_target_location(self._new_target)

        return self.env.reset(seed=seed, options=options)
