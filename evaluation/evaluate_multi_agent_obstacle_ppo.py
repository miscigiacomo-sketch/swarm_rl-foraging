import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from env.multi_agent_foraging_env import MultiAgentForagingEnv


FIXED_OBSTACLES = [
    (2, 1),
    (2, 2),
    (2, 3),
]


BASELINE_RESULTS = {
    "random": {
        "success_rate": 0.764,
        "average_reward": 0.76,
        "average_episode_length": 24.63,
    },
    "greedy": {
        "success_rate": 0.931,
        "average_reward": 0.93,
        "average_episode_length": 5.83,
    },
}


def evaluate_ppo_model(
    model,
    episodes=1000,
    grid_size=5,
    num_agents=2,
    deterministic=True,
    action_seed=42,
):
    set_random_seed(action_seed)

    env = MultiAgentForagingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=50,
        obstacles=FIXED_OBSTACLES,
    )

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

            action = int(action)

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
    model,
    episodes=1000,
    grid_size=5,
    num_agents=2,
    action_seeds=None,
):
    if action_seeds is None:
        action_seeds = [0, 1, 2, 3, 4, 42, 100, 123, 999, 2024]

    success_rates = []
    average_rewards = []
    average_lengths = []
    per_seed_results = []

    for action_seed in action_seeds:
        results = evaluate_ppo_model(
            model=model,
            episodes=episodes,
            grid_size=grid_size,
            num_agents=num_agents,
            deterministic=False,
            action_seed=action_seed,
        )

        success_rates.append(results["success_rate"])
        average_rewards.append(results["average_reward"])
        average_lengths.append(results["average_episode_length"])
        per_seed_results.append((action_seed, results))

    return {
        "success_rate_mean": np.mean(success_rates),
        "success_rate_std": np.std(success_rates),
        "success_rate_min": np.min(success_rates),
        "success_rate_max": np.max(success_rates),
        "average_reward_mean": np.mean(average_rewards),
        "average_episode_length_mean": np.mean(average_lengths),
        "per_seed_results": per_seed_results,
    }


def print_baseline_results():
    print("Baseline references from Task 2:")
    print(
        "Random baseline: "
        f"{BASELINE_RESULTS['random']['success_rate'] * 100:.2f}% success, "
        f"average reward {BASELINE_RESULTS['random']['average_reward']:.2f}, "
        f"average episode length "
        f"{BASELINE_RESULTS['random']['average_episode_length']:.2f}"
    )
    print(
        "Greedy obstacle-aware baseline: "
        f"{BASELINE_RESULTS['greedy']['success_rate'] * 100:.2f}% success, "
        f"average reward {BASELINE_RESULTS['greedy']['average_reward']:.2f}, "
        f"average episode length "
        f"{BASELINE_RESULTS['greedy']['average_episode_length']:.2f}"
    )
    print()


def print_ppo_results(deterministic_results, stochastic_results):
    print("PPO deterministic:")
    print(f"Success Rate: {deterministic_results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {deterministic_results['average_reward']:.2f}")
    print(
        "Average Episode Length: "
        f"{deterministic_results['average_episode_length']:.2f}"
    )
    print()

    print("PPO stochastic summary:")
    print(f"Mean Success Rate: {stochastic_results['success_rate_mean'] * 100:.2f}%")
    print(f"Success Rate Std: {stochastic_results['success_rate_std'] * 100:.2f}%")
    print(f"Min Success Rate: {stochastic_results['success_rate_min'] * 100:.2f}%")
    print(f"Max Success Rate: {stochastic_results['success_rate_max'] * 100:.2f}%")
    print(f"Mean Average Reward: {stochastic_results['average_reward_mean']:.2f}")
    print(
        "Mean Average Episode Length: "
        f"{stochastic_results['average_episode_length_mean']:.2f}"
    )
    print()


def save_results_to_file(
    model_path,
    deterministic_results,
    stochastic_results,
    output_path,
):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MULTI-AGENT FIXED-OBSTACLE PPO EVALUATION\n")
        file.write("=========================================\n\n")

        file.write("Environment setup:\n")
        file.write("- Grid size: 5x5\n")
        file.write("- Number of agents: 2\n")
        file.write("- Max steps: 50\n")
        file.write(f"- Obstacles: {FIXED_OBSTACLES}\n")
        file.write("- Control type: centralized joint action space\n")
        file.write("- Joint action space: 16 actions\n")
        file.write("- Observation size: 12\n\n")

        file.write(f"Model: {model_path}\n")
        file.write("Evaluation episodes: 1000 seeded episodes\n\n")

        file.write("Baseline references from Task 2:\n")
        file.write(
            f"Random baseline: "
            f"{BASELINE_RESULTS['random']['success_rate'] * 100:.2f}% success, "
            f"average reward {BASELINE_RESULTS['random']['average_reward']:.2f}, "
            f"average episode length "
            f"{BASELINE_RESULTS['random']['average_episode_length']:.2f}\n"
        )
        file.write(
            f"Greedy obstacle-aware baseline: "
            f"{BASELINE_RESULTS['greedy']['success_rate'] * 100:.2f}% success, "
            f"average reward {BASELINE_RESULTS['greedy']['average_reward']:.2f}, "
            f"average episode length "
            f"{BASELINE_RESULTS['greedy']['average_episode_length']:.2f}\n\n"
        )

        file.write("PPO deterministic:\n")
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

        file.write("PPO stochastic:\n")
        file.write(
            f"Mean Success Rate: "
            f"{stochastic_results['success_rate_mean'] * 100:.2f}%\n"
        )
        file.write(
            f"Success Rate Std: "
            f"{stochastic_results['success_rate_std'] * 100:.2f}%\n"
        )
        file.write(
            f"Min Success Rate: "
            f"{stochastic_results['success_rate_min'] * 100:.2f}%\n"
        )
        file.write(
            f"Max Success Rate: "
            f"{stochastic_results['success_rate_max'] * 100:.2f}%\n"
        )
        file.write(
            f"Mean Average Reward: "
            f"{stochastic_results['average_reward_mean']:.2f}\n"
        )
        file.write(
            f"Mean Average Episode Length: "
            f"{stochastic_results['average_episode_length_mean']:.2f}\n\n"
        )

        file.write("Per-action-seed stochastic results:\n")
        for action_seed, results in stochastic_results["per_seed_results"]:
            file.write(
                f"Action Seed {action_seed}: "
                f"Success Rate {results['success_rate'] * 100:.2f}%, "
                f"Average Reward {results['average_reward']:.2f}, "
                f"Average Episode Length "
                f"{results['average_episode_length']:.2f}\n"
            )

        file.write("\nInterpretation:\n")
        file.write(
            "The centralized PPO policy is evaluated in the same fixed-obstacle "
            "multi-agent environment used for the Task 2 baselines. "
            "The deterministic PPO result shows the performance of argmax action "
            "selection, while the stochastic PPO result measures whether sampling "
            "from the learned policy improves robustness. "
            "The PPO results should be compared against the random baseline and "
            "the greedy obstacle-aware baseline.\n"
        )


def main():
    model_path = "models/ppo_multi_agent_2agents_obstacles.zip"
    output_path = "results/multi_agent_obstacle_ppo_summary.txt"
    episodes = 1000

    print("===== MULTI-AGENT FIXED-OBSTACLE PPO EVALUATION =====")
    print(f"Model: {model_path}")
    print("Grid Size: 5x5")
    print("Number of Agents: 2")
    print(f"Fixed Obstacles: {FIXED_OBSTACLES}")
    print(f"Episodes: {episodes}")
    print()

    model = PPO.load(model_path)

    deterministic_results = evaluate_ppo_model(
        model=model,
        episodes=episodes,
        grid_size=5,
        num_agents=2,
        deterministic=True,
        action_seed=42,
    )

    stochastic_results = evaluate_stochastic_ppo(
        model=model,
        episodes=episodes,
        grid_size=5,
        num_agents=2,
    )

    print_baseline_results()
    print_ppo_results(
        deterministic_results=deterministic_results,
        stochastic_results=stochastic_results,
    )

    save_results_to_file(
        model_path=model_path,
        deterministic_results=deterministic_results,
        stochastic_results=stochastic_results,
        output_path=output_path,
    )

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()