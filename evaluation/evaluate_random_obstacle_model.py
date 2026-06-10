import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from env.foraging_env import ForagingEnv
import numpy as np


def evaluate_model(model_path, episodes=100):
    env = ForagingEnv(random_obstacles=True)
    model = PPO.load(model_path)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        if total_reward > 0:
            successes += 1

        total_rewards.append(total_reward)
        episode_lengths.append(steps)

    success_rate = successes / episodes
    average_reward = np.mean(total_rewards)
    average_episode_length = np.mean(episode_lengths)

    print("===== PPO TRAINED ON RANDOMIZED OBSTACLES =====")
    print(f"Episodes: {episodes}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Average Episode Length: {average_episode_length:.2f}")


if __name__ == "__main__":
    evaluate_model(
        model_path="models/ppo_foraging_random_obstacles.zip",
        episodes=100,
    )