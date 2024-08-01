import four_rooms  # noqa
import gymnasium as gym

from four_rooms.wrappers import SwitchTarget
from model import TabularOptionCritic
import numpy as np

def train(n_options: int = 4):
    """Reproduce"""

    env = SwitchTarget(gym.make("four_rooms/FourRooms-v0"))

    model = TabularOptionCritic(
        env.observation_space.n, n_options, env.action_space.n
    )

    for episode in range(2000):

        state, info = env.reset()
        greedy_option = model.greedy_option(state)
        current_option = 0
        option_termination = True

        while True:
            epsilon = model.epsilon

            if option_termination:
                current_option = np.random.choice(n_options) if np.random.rand() < epsilon else greedy_option
            
            action, logp, entropy = model.get_action(state, current_option)
            
            next_state, reward, done, truncated, info = env.step(action)


            if model.sample_option_termination(state, option):
                option = model.sample_option(state)

            action = model.sample_action(state, option)
            model.update(state, option, action, reward, done)

            if done or truncated:
                break

        print(f"Episode: {episode}, Steps: {info['steps']}")

if __name__ == "__main__":
    train()
