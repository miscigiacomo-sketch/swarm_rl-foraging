import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.foraging_env import ForagingEnv


def main():
    env = ForagingEnv(random_obstacles=True)

    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
    )

    model.learn(total_timesteps=200000)

    model.save("models/ppo_foraging_random_obstacles_200k")

    print("Training completed.")
    print("Model saved to models/ppo_foraging_random_obstacles_200k.zip")


if __name__ == "__main__":
    main()