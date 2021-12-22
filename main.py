from src.loop import train_loop
from src.from_config import from_config


def run_cartpole_v0(n_trials=10):
    for seed in range(n_trials):
        training_objects = from_config('config_files/CartPole-v0.json', seed=seed)
        train_loop(**training_objects)


if __name__ == "__main__":
    run_cartpole_v0()
