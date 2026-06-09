import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO

from env.foraging_env import ForagingEnv


def main():
    env = ForagingEnv(grid_size=5, max_steps=50)
    model = PPO.load("models/ppo_foraging_obstacles_50k")

    observation, _ = env.reset()

    print("Initial observation:", observation)
    env.render()

    for step in range(env.max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)

        print()
        print("Step:", step)
        print("Action:", action)
        print("Observation:", observation)
        print("Reward:", reward)

        env.render()

        if terminated:
            print()
            print("Food collected!")
            break

        if truncated:
            print()
            print("Maximum steps reached!")
            break

    print()
    print("Obstacle agent test finished.")


if __name__ == "__main__":
    main()