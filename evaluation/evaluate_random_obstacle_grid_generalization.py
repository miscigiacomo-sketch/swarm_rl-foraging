import os
import sys
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from env.foraging_env import ForagingEnv


ACTIONS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
}


def get_agent_position(obs):
    return tuple(obs[0:2].astype(int))


def parse_observation(obs):
    agent_pos = tuple(obs[0:2].astype(int))
    food_pos = tuple(obs[2:4].astype(int))

    obstacles = []
    for i in range(4, len(obs), 2):
        obstacles.append(tuple(obs[i:i + 2].astype(int)))

    return agent_pos, food_pos, obstacles


def find_shortest_path(agent_pos, food_pos, obstacles, grid_size):
    obstacle_set = set(obstacles)

    queue = deque()
    queue.append((agent_pos, []))

    visited = set()
    visited.add(agent_pos)

    while queue:
        current_pos, path = queue.popleft()

        if current_pos == food_pos:
            return path

        current_x, current_y = current_pos

        for action, (dx, dy) in ACTIONS.items():
            next_pos = (current_x + dx, current_y + dy)
            next_x, next_y = next_pos

            if next_x < 0 or next_x >= grid_size:
                continue

            if next_y < 0 or next_y >= grid_size:
                continue

            if next_pos in obstacle_set:
                continue

            if next_pos in visited:
                continue

            visited.add(next_pos)
            queue.append((next_pos, path + [action]))

    return None


def evaluate_oracle(episodes=1000, grid_size=5):
    env = ForagingEnv(grid_size=grid_size, random_obstacles=True)

    successes = 0
    path_lengths = []
    failed_seeds = []

    for seed in range(episodes):
        obs, info = env.reset(seed=seed)

        agent_pos, food_pos, obstacles = parse_observation(obs)

        path = find_shortest_path(
            agent_pos=agent_pos,
            food_pos=food_pos,
            obstacles=obstacles,
            grid_size=grid_size,
        )

        if path is None:
            failed_seeds.append(seed)
            continue

        done = False
        total_reward = 0

        for action in path:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if done:
                break

        if total_reward > 0:
            successes += 1

        path_lengths.append(len(path))

    env.close()

    return {
        "success_rate": successes / episodes,
        "average_shortest_path_length": np.mean(path_lengths),
        "failed_seeds": failed_seeds,
    }


def evaluate_ppo(
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

        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        total_no_move_steps += no_move_steps

    env.close()

    return {
        "success_rate": successes / episodes,
        "average_reward": np.mean(total_rewards),
        "average_episode_length": np.mean(episode_lengths),
        "failures": failures,
        "average_no_move_steps": total_no_move_steps / episodes,
    }


def evaluate_stochastic_across_seeds(
    model,
    action_seeds,
    episodes=1000,
    grid_size=5,
):
    success_rates = []
    rewards = []
    episode_lengths = []
    no_move_steps = []
    failures = []

    for action_seed in action_seeds:
        results = evaluate_ppo(
            model=model,
            episodes=episodes,
            grid_size=grid_size,
            deterministic=False,
            action_seed=action_seed,
        )

        success_rates.append(results["success_rate"])
        rewards.append(results["average_reward"])
        episode_lengths.append(results["average_episode_length"])
        no_move_steps.append(results["average_no_move_steps"])
        failures.append(results["failures"])

    return {
        "mean_success_rate": np.mean(success_rates),
        "std_success_rate": np.std(success_rates),
        "min_success_rate": np.min(success_rates),
        "max_success_rate": np.max(success_rates),
        "mean_average_reward": np.mean(rewards),
        "mean_average_episode_length": np.mean(episode_lengths),
        "mean_no_move_steps": np.mean(no_move_steps),
        "mean_failures": np.mean(failures),
    }


def print_grid_results(grid_size, oracle_results, deterministic_results, stochastic_results):
    print(f"===== RANDOM OBSTACLE GRID GENERALIZATION: {grid_size}x{grid_size} =====")
    print()

    print("BFS Oracle:")
    print(f"Success Rate: {oracle_results['success_rate'] * 100:.2f}%")
    print(f"Average Shortest Path Length: {oracle_results['average_shortest_path_length']:.2f}")
    print(f"Failed Seeds: {oracle_results['failed_seeds']}")
    print()

    print("PPO Deterministic:")
    print(f"Success Rate: {deterministic_results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {deterministic_results['average_reward']:.2f}")
    print(f"Average Episode Length: {deterministic_results['average_episode_length']:.2f}")
    print(f"Failures: {deterministic_results['failures']}")
    print(f"Average No-Move Steps: {deterministic_results['average_no_move_steps']:.2f}")
    print()

    print("PPO Stochastic Summary:")
    print(f"Mean Success Rate: {stochastic_results['mean_success_rate'] * 100:.2f}%")
    print(f"Std Success Rate: {stochastic_results['std_success_rate'] * 100:.2f}%")
    print(f"Min Success Rate: {stochastic_results['min_success_rate'] * 100:.2f}%")
    print(f"Max Success Rate: {stochastic_results['max_success_rate'] * 100:.2f}%")
    print(f"Mean Average Reward: {stochastic_results['mean_average_reward']:.2f}")
    print(f"Mean Average Episode Length: {stochastic_results['mean_average_episode_length']:.2f}")
    print(f"Mean No-Move Steps: {stochastic_results['mean_no_move_steps']:.2f}")
    print(f"Mean Failures: {stochastic_results['mean_failures']:.2f}")
    print()


def save_results_to_file(all_results, output_path):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("RANDOM OBSTACLE GRID-SIZE GENERALIZATION\n")
        file.write("========================================\n\n")

        for grid_size, results in all_results.items():
            oracle = results["oracle"]
            deterministic = results["deterministic"]
            stochastic = results["stochastic"]

            file.write(f"Grid Size: {grid_size}x{grid_size}\n")
            file.write("-" * 30 + "\n\n")

            file.write("BFS Oracle:\n")
            file.write(f"Success Rate: {oracle['success_rate'] * 100:.2f}%\n")
            file.write(
                f"Average Shortest Path Length: "
                f"{oracle['average_shortest_path_length']:.2f}\n"
            )
            file.write(f"Failed Seeds: {oracle['failed_seeds']}\n\n")

            file.write("PPO Deterministic:\n")
            file.write(f"Success Rate: {deterministic['success_rate'] * 100:.2f}%\n")
            file.write(f"Average Reward: {deterministic['average_reward']:.2f}\n")
            file.write(
                f"Average Episode Length: "
                f"{deterministic['average_episode_length']:.2f}\n"
            )
            file.write(f"Failures: {deterministic['failures']}\n")
            file.write(
                f"Average No-Move Steps: "
                f"{deterministic['average_no_move_steps']:.2f}\n\n"
            )

            file.write("PPO Stochastic Summary:\n")
            file.write(
                f"Mean Success Rate: "
                f"{stochastic['mean_success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Std Success Rate: "
                f"{stochastic['std_success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Min Success Rate: "
                f"{stochastic['min_success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Max Success Rate: "
                f"{stochastic['max_success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Mean Average Reward: "
                f"{stochastic['mean_average_reward']:.2f}\n"
            )
            file.write(
                f"Mean Average Episode Length: "
                f"{stochastic['mean_average_episode_length']:.2f}\n"
            )
            file.write(
                f"Mean No-Move Steps: "
                f"{stochastic['mean_no_move_steps']:.2f}\n"
            )
            file.write(f"Mean Failures: {stochastic['mean_failures']:.2f}\n\n")


def main():
    model_path = "models/ppo_foraging_random_obstacles_200k.zip"
    episodes = 1000
    grid_sizes = [5, 10]
    action_seeds = [0, 1, 2, 3, 4, 42, 100, 123, 999, 2024]

    model = PPO.load(model_path)

    print("===== RANDOM OBSTACLE GRID-SIZE GENERALIZATION EVALUATION =====")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Grid Sizes: {grid_sizes}")
    print(f"Action Seeds: {action_seeds}")
    print()

    all_results = {}

    for grid_size in grid_sizes:
        oracle_results = evaluate_oracle(
            episodes=episodes,
            grid_size=grid_size,
        )

        deterministic_results = evaluate_ppo(
            model=model,
            episodes=episodes,
            grid_size=grid_size,
            deterministic=True,
            action_seed=42,
        )

        stochastic_results = evaluate_stochastic_across_seeds(
            model=model,
            action_seeds=action_seeds,
            episodes=episodes,
            grid_size=grid_size,
        )

        all_results[grid_size] = {
            "oracle": oracle_results,
            "deterministic": deterministic_results,
            "stochastic": stochastic_results,
        }

        print_grid_results(
            grid_size=grid_size,
            oracle_results=oracle_results,
            deterministic_results=deterministic_results,
            stochastic_results=stochastic_results,
        )

    output_path = "results/random_obstacle_grid_generalization_summary.txt"
    save_results_to_file(all_results, output_path)

    print("===== FINAL GRID GENERALIZATION COMPARISON =====")
    print(
        "| Grid | Oracle Success | PPO Deterministic Success | "
        "PPO Stochastic Mean Success | Stochastic Std | "
        "PPO Stochastic Avg Length |"
    )
    print("|---|---:|---:|---:|---:|---:|")

    for grid_size, results in all_results.items():
        oracle = results["oracle"]
        deterministic = results["deterministic"]
        stochastic = results["stochastic"]

        print(
            f"| {grid_size}x{grid_size} | "
            f"{oracle['success_rate'] * 100:.2f}% | "
            f"{deterministic['success_rate'] * 100:.2f}% | "
            f"{stochastic['mean_success_rate'] * 100:.2f}% | "
            f"{stochastic['std_success_rate'] * 100:.2f}% | "
            f"{stochastic['mean_average_episode_length']:.2f} |"
        )

    print()
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()