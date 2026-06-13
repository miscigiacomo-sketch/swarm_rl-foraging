import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from env.multi_agent_foraging_env import MultiAgentForagingEnv


def evaluate_ppo_model(
    model_path,
    num_agents=2,
    grid_size=5,
    episodes=1000,
    deterministic=True,
    action_seed=42,
):
    env = MultiAgentForagingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=50,
    )

    set_random_seed(action_seed)

    model = PPO.load(model_path)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(
                obs,
                deterministic=deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        if total_reward > 0:
            successes += 1

        total_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    return {
        "success_rate": successes / episodes,
        "average_reward": np.mean(total_rewards),
        "average_episode_length": np.mean(episode_lengths),
    }


def evaluate_stochastic_ppo(
    model_path,
    num_agents=2,
    grid_size=5,
    episodes=1000,
    action_seeds=None,
):
    if action_seeds is None:
        action_seeds = [0, 1, 2, 3, 4, 42, 100, 123, 999, 2024]

    success_rates = []
    average_rewards = []
    average_lengths = []

    for action_seed in action_seeds:
        results = evaluate_ppo_model(
            model_path=model_path,
            num_agents=num_agents,
            grid_size=grid_size,
            episodes=episodes,
            deterministic=False,
            action_seed=action_seed,
        )

        success_rates.append(results["success_rate"])
        average_rewards.append(results["average_reward"])
        average_lengths.append(results["average_episode_length"])

    return {
        "success_rate_mean": np.mean(success_rates),
        "success_rate_std": np.std(success_rates),
        "average_reward_mean": np.mean(average_rewards),
        "average_episode_length_mean": np.mean(average_lengths),
    }


def save_results_to_file(
    deterministic_results,
    stochastic_results,
    output_path,
):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MULTI-AGENT PPO EVALUATION\n")
        file.write("==========================\n\n")

        file.write("Environment:\n")
        file.write("Agents: 2\n")
        file.write("Grid Size: 5x5\n")
        file.write("Episodes: 1000\n\n")

        file.write("Deterministic PPO:\n")
        file.write(
            f"Success Rate: {deterministic_results['success_rate'] * 100:.2f}%\n"
        )
        file.write(
            f"Average Reward: {deterministic_results['average_reward']:.2f}\n"
        )
        file.write(
            f"Average Episode Length: "
            f"{deterministic_results['average_episode_length']:.2f}\n\n"
        )

        file.write("Stochastic PPO:\n")
        file.write(
            f"Mean Success Rate: "
            f"{stochastic_results['success_rate_mean'] * 100:.2f}%\n"
        )
        file.write(
            f"Success Rate Std: "
            f"{stochastic_results['success_rate_std'] * 100:.2f}%\n"
        )
        file.write(
            f"Mean Average Reward: "
            f"{stochastic_results['average_reward_mean']:.2f}\n"
        )
        file.write(
            f"Mean Average Episode Length: "
            f"{stochastic_results['average_episode_length_mean']:.2f}\n"
        )


def main():
    model_path = "models/ppo_multi_agent_2agents.zip"
    episodes = 1000

    print("===== MULTI-AGENT PPO EVALUATION =====")
    print("Model:", model_path)
    print("Agents: 2")
    print("Grid Size: 5x5")
    print(f"Episodes: {episodes}")
    print()

    deterministic_results = evaluate_ppo_model(
        model_path=model_path,
        num_agents=2,
        grid_size=5,
        episodes=episodes,
        deterministic=True,
        action_seed=42,
    )

    stochastic_results = evaluate_stochastic_ppo(
        model_path=model_path,
        num_agents=2,
        grid_size=5,
        episodes=episodes,
    )

    output_path = "results/multi_agent_ppo_summary.txt"
    save_results_to_file(
        deterministic_results=deterministic_results,
        stochastic_results=stochastic_results,
        output_path=output_path,
    )

    print("===== PPO RESULTS =====")
    print("Deterministic PPO:")
    print(f"Success Rate: {deterministic_results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {deterministic_results['average_reward']:.2f}")
    print(
        f"Average Episode Length: "
        f"{deterministic_results['average_episode_length']:.2f}"
    )
    print()

    print("Stochastic PPO:")
    print(
        f"Mean Success Rate: "
        f"{stochastic_results['success_rate_mean'] * 100:.2f}%"
    )
    print(
        f"Success Rate Std: "
        f"{stochastic_results['success_rate_std'] * 100:.2f}%"
    )
    print(
        f"Mean Average Reward: "
        f"{stochastic_results['average_reward_mean']:.2f}"
    )
    print(
        f"Mean Average Episode Length: "
        f"{stochastic_results['average_episode_length_mean']:.2f}"
    )
    print()

    print("===== BASELINE REFERENCE =====")
    print("| Method | Success Rate | Average Episode Length |")
    print("|---|---:|---:|")
    print("| Random baseline | 84.30% | 21.30 |")
    print("| Greedy decentralized baseline | 100.00% | 2.41 |")
    print(
        f"| PPO deterministic | "
        f"{deterministic_results['success_rate'] * 100:.2f}% | "
        f"{deterministic_results['average_episode_length']:.2f} |"
    )
    print(
        f"| PPO stochastic mean | "
        f"{stochastic_results['success_rate_mean'] * 100:.2f}% | "
        f"{stochastic_results['average_episode_length_mean']:.2f} |"
    )

    print()
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()