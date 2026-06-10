import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from env.foraging_env import ForagingEnv
import numpy as np


def evaluate_model_on_seeded_episodes(model, seeds, episodes=1000):
    env = ForagingEnv(random_obstacles=True)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seeds[episode])
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

    return success_rate, average_reward, average_episode_length


def main():
    episodes = 1000
    seeds = list(range(episodes))

    fixed_obstacle_model = PPO.load("models/ppo_foraging_obstacles_state.zip")
    random_obstacle_model = PPO.load("models/ppo_foraging_random_obstacles.zip")

    fixed_results = evaluate_model_on_seeded_episodes(
        model=fixed_obstacle_model,
        seeds=seeds,
        episodes=episodes,
    )

    random_results = evaluate_model_on_seeded_episodes(
        model=random_obstacle_model,
        seeds=seeds,
        episodes=episodes,
    )

    print("===== RANDOM OBSTACLE GENERALIZATION COMPARISON =====")
    print(f"Episodes: {episodes}")
    print()

    print("Fixed-obstacle PPO evaluated on randomized obstacles:")
    print(f"Success Rate: {fixed_results[0] * 100:.2f}%")
    print(f"Average Reward: {fixed_results[1]:.2f}")
    print(f"Average Episode Length: {fixed_results[2]:.2f}")
    print()

    print("Random-obstacle PPO evaluated on randomized obstacles:")
    print(f"Success Rate: {random_results[0] * 100:.2f}%")
    print(f"Average Reward: {random_results[1]:.2f}")
    print(f"Average Episode Length: {random_results[2]:.2f}")


if __name__ == "__main__":
    main()