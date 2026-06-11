import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from env.foraging_env import ForagingEnv


def get_agent_position(obs):
    return tuple(obs[0:2].astype(int))


def evaluate_model(
    model,
    episodes=1000,
    grid_size=5,
    deterministic=True,
    action_seed=42,
):
    set_random_seed(action_seed)

    env = ForagingEnv(grid_size=grid_size, random_obstacles=True)

    successes = 0
    total_rewards = []
    episode_lengths = []

    total_no_move_steps = 0
    failed_no_move_steps = 0
    failures = 0

    for seed in range(episodes):
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0
        steps = 0
        no_move_steps = 0

        while not done:
            previous_position = get_agent_position(obs)

            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)

            current_position = get_agent_position(obs)

            if current_position == previous_position:
                no_move_steps += 1

            total_reward += reward
            steps += 1
            done = terminated or truncated

        if total_reward > 0:
            successes += 1
        else:
            failures += 1
            failed_no_move_steps += no_move_steps

        total_no_move_steps += no_move_steps
        total_rewards.append(total_reward)
        episode_lengths.append(steps)

    success_rate = successes / episodes
    average_reward = np.mean(total_rewards)
    average_episode_length = np.mean(episode_lengths)
    average_no_move_steps = total_no_move_steps / episodes

    if failures > 0:
        average_no_move_steps_per_failed_episode = failed_no_move_steps / failures
    else:
        average_no_move_steps_per_failed_episode = 0

    return {
        "success_rate": success_rate,
        "average_reward": average_reward,
        "average_episode_length": average_episode_length,
        "failures": failures,
        "average_no_move_steps": average_no_move_steps,
        "average_no_move_steps_per_failed_episode": average_no_move_steps_per_failed_episode,
    }


def print_results(title, results):
    print(title)
    print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Average Episode Length: {results['average_episode_length']:.2f}")
    print(f"Failures: {results['failures']}")
    print(f"Average No-Move Steps: {results['average_no_move_steps']:.2f}")
    print(
        "Average No-Move Steps per Failed Episode: "
        f"{results['average_no_move_steps_per_failed_episode']:.2f}"
    )
    print()


def main():
    model_path = "models/ppo_foraging_random_obstacles_100k.zip"
    episodes = 1000
    grid_size = 5

    model = PPO.load(model_path)

    print("===== PPO POLICY SAMPLING MODE COMPARISON =====")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print()

    deterministic_results = evaluate_model(
        model=model,
        episodes=episodes,
        grid_size=grid_size,
        deterministic=True,
        action_seed=42,
    )

    stochastic_results = evaluate_model(
        model=model,
        episodes=episodes,
        grid_size=grid_size,
        deterministic=False,
        action_seed=42,
    )

    print_results(
        "Deterministic PPO evaluation:",
        deterministic_results,
    )

    print_results(
        "Stochastic PPO evaluation:",
        stochastic_results,
    )


if __name__ == "__main__":
    main()