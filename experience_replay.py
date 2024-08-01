import numpy as np


class ReplayBuffer:
    """
    A replay buffer for storing and sampling experiences.

    Args:
        obs_shape (tuple[int, ...]): The shape of the observation.
        capacity (int): The maximum capacity of the buffer.
        seed (int, optional): The seed for the random number generator. Defaults to 42.
    """

    def __init__(self, obs_shape: tuple[int, ...], capacity: int, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.obs_dim = np.prod(obs_shape)
        self.buffer = np.zeros((capacity, self.obs_dim * 2 + 3))
        self.start: int = 0

    def push(self, obs, option, reward, next_obs, done):
        """
        Pushes an experience into the buffer.

        Args:
            obs: The current observation.
            option: The chosen option.
            reward: The received reward.
            next_obs: The next observation.
            done: Whether the episode is done.
        """
        self.buffer[self.start] = np.array([obs, next_obs, option, reward, done])
        self.start = (self.start + 1) % self.buffer.shape

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple[np.ndarray, ...]: A tuple containing the sampled observations, options, rewards,
            next observations, and done flags.
        """
        experience = self.rng.choice(self.buffer, batch_size, axis=0)
        obs = experience[:, :self.obs_dim]
        next_obs = experience[:, self.obs_dim:2 * self.obs_dim]
        option, reward, done = experience[:, -3:]
        return obs, option, reward, next_obs, done

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The size of the buffer.
        """
        return len(self.buffer)
