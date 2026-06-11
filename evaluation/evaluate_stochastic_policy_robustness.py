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
    deterministic=False,
    action_seed=42,
):
    set_random_seed(action_seed)

    env = ForagingEnv(grid_size=grid_size, random_obstacles=True)

    successes = 0
    total_rewards = []
    episode_lengths = []
    total_no_move_steps = 0
    failures = 0
    failed_no_move_steps = 0

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

        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        total_no_move_steps += no_move_steps

    success_rate = successes / episodes
    average_reward = np.mean(total_rewards)
    average_episode_length = np.mean(episode_lengths)
    average_no_move_steps = total_no_move_steps / episodes

    if failures > 0:
        average_no_move_steps_failed = failed_no_move_steps / failures
    else:
        average_no_move_steps_failed = 0

    env.close()

    return {
        "success_rate": success_rate,
        "average_reward": average_reward,
        "average_episode_length": average_episode_length,
        "failures": failures,
        "average_no_move_steps": average_no_move_steps,
        "average_no_move_steps_failed": average_no_move_steps_failed,
    }


def evaluate_stochastic_across_seeds(
    model,
    action_seeds,
    episodes=1000,
    grid_size=5,
):
    success_rates = []
    average_rewards = []
    average_lengths = []
    average_no_move_steps = []
    failures = []

    per_seed_results = []

    for action_seed in action_seeds:
        results = evaluate_model(
            model=model,
            episodes=episodes,
            grid_size=grid_size,
            deterministic=False,
            action_seed=action_seed,
        )

        success_rates.append(results["success_rate"])
        average_rewards.append(results["average_reward"])
        average_lengths.append(results["average_episode_length"])
        average_no_move_steps.append(results["average_no_move_steps"])
        failures.append(results["failures"])

        per_seed_results.append((action_seed, results))

    summary = {
        "mean_success_rate": np.mean(success_rates),
        "std_success_rate": np.std(success_rates),
        "min_success_rate": np.min(success_rates),
        "max_success_rate": np.max(success_rates),
        "mean_average_reward": np.mean(average_rewards),
        "mean_average_episode_length": np.mean(average_lengths),
        "mean_average_no_move_steps": np.mean(average_no_move_steps),
        "mean_failures": np.mean(failures),
        "per_seed_results": per_seed_results,
    }

    return summary


def print_deterministic_results(model_name, results):
    print(f"{model_name} - Deterministic evaluation")
    print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Average Episode Length: {results['average_episode_length']:.2f}")
    print(f"Failures: {results['failures']}")
    print(f"Average No-Move Steps: {results['average_no_move_steps']:.2f}")
    print()


def print_stochastic_summary(model_name, summary):
    print(f"{model_name} - Stochastic evaluation summary")
    print(f"Mean Success Rate: {summary['mean_success_rate'] * 100:.2f}%")
    print(f"Std Success Rate: {summary['std_success_rate'] * 100:.2f}%")
    print(f"Min Success Rate: {summary['min_success_rate'] * 100:.2f}%")
    print(f"Max Success Rate: {summary['max_success_rate'] * 100:.2f}%")
    print(f"Mean Average Reward: {summary['mean_average_reward']:.2f}")
    print(f"Mean Average Episode Length: {summary['mean_average_episode_length']:.2f}")
    print(f"Mean Average No-Move Steps: {summary['mean_average_no_move_steps']:.2f}")
    print(f"Mean Failures: {summary['mean_failures']:.2f}")
    print()


def save_summary_to_file(all_results, output_path):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("PPO POLICY SAMPLING ROBUSTNESS SUMMARY\n")
        file.write("======================================\n\n")

        for model_name, results in all_results.items():
            deterministic = results["deterministic"]
            stochastic = results["stochastic"]

            file.write(f"{model_name}\n")
            file.write("-" * len(model_name) + "\n\n")

            file.write("Deterministic evaluation:\n")
            file.write(f"Success Rate: {deterministic['success_rate'] * 100:.2f}%\n")
            file.write(f"Average Reward: {deterministic['average_reward']:.2f}\n")
            file.write(f"Average Episode Length: {deterministic['average_episode_length']:.2f}\n")
            file.write(f"Failures: {deterministic['failures']}\n")
            file.write(f"Average No-Move Steps: {deterministic['average_no_move_steps']:.2f}\n\n")

            file.write("Stochastic evaluation across action seeds:\n")
            file.write(f"Mean Success Rate: {stochastic['mean_success_rate'] * 100:.2f}%\n")
            file.write(f"Std Success Rate: {stochastic['std_success_rate'] * 100:.2f}%\n")
            file.write(f"Min Success Rate: {stochastic['min_success_rate'] * 100:.2f}%\n")
            file.write(f"Max Success Rate: {stochastic['max_success_rate'] * 100:.2f}%\n")
            file.write(f"Mean Average Reward: {stochastic['mean_average_reward']:.2f}\n")
            file.write(f"Mean Average Episode Length: {stochastic['mean_average_episode_length']:.2f}\n")
            file.write(f"Mean Average No-Move Steps: {stochastic['mean_average_no_move_steps']:.2f}\n")
            file.write(f"Mean Failures: {stochastic['mean_failures']:.2f}\n\n")

            file.write("Per-action-seed stochastic results:\n")
            for action_seed, seed_results in stochastic["per_seed_results"]:
                file.write(
                    f"Action Seed {action_seed}: "
                    f"Success Rate {seed_results['success_rate'] * 100:.2f}%, "
                    f"Average Episode Length {seed_results['average_episode_length']:.2f}, "
                    f"Failures {seed_results['failures']}, "
                    f"No-Move Steps {seed_results['average_no_move_steps']:.2f}\n"
                )

            file.write("\n\n")


def main():
    episodes = 1000
    grid_size = 5
    action_seeds = [0, 1, 2, 3, 4, 42, 100, 123, 999, 2024]

    models_to_evaluate = {
        "PPO Random Obstacles 100k": "models/ppo_foraging_random_obstacles_100k.zip",
        "PPO Random Obstacles 200k": "models/ppo_foraging_random_obstacles_200k.zip",
    }

    all_results = {}

    print("===== PPO POLICY SAMPLING ROBUSTNESS EVALUATION =====")
    print(f"Episodes: {episodes}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Action Seeds: {action_seeds}")
    print()

    for model_name, model_path in models_to_evaluate.items():
        print(f"Evaluating: {model_name}")
        print(f"Model path: {model_path}")
        print()

        model = PPO.load(model_path)

        deterministic_results = evaluate_model(
            model=model,
            episodes=episodes,
            grid_size=grid_size,
            deterministic=True,
            action_seed=42,
        )

        stochastic_summary = evaluate_stochastic_across_seeds(
            model=model,
            action_seeds=action_seeds,
            episodes=episodes,
            grid_size=grid_size,
        )

        all_results[model_name] = {
            "deterministic": deterministic_results,
            "stochastic": stochastic_summary,
        }

        print_deterministic_results(model_name, deterministic_results)
        print_stochastic_summary(model_name, stochastic_summary)

    output_path = "results/policy_sampling_robustness_summary.txt"
    save_summary_to_file(all_results, output_path)

    print("===== FINAL COMPARISON =====")
    print(
        "| Model | Deterministic Success | Stochastic Mean Success | "
        "Stochastic Std | Mean Episode Length | Mean No-Move Steps |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|"
    )

    for model_name, results in all_results.items():
        deterministic = results["deterministic"]
        stochastic = results["stochastic"]

        print(
            f"| {model_name} | "
            f"{deterministic['success_rate'] * 100:.2f}% | "
            f"{stochastic['mean_success_rate'] * 100:.2f}% | "
            f"{stochastic['std_success_rate'] * 100:.2f}% | "
            f"{stochastic['mean_average_episode_length']:.2f} | "
            f"{stochastic['mean_average_no_move_steps']:.2f} |"
        )

    print()
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()