import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from env.multi_agent_foraging_env import MultiAgentForagingEnv


def make_env(grid_size=5, num_agents=2, max_steps=50):
    return MultiAgentForagingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
        random_obstacles=True,
        num_obstacles=3,
    )


def decode_joint_action(action, num_agents, num_actions_per_agent=4):
    action = int(action)
    individual_actions = []

    for _ in range(num_agents):
        individual_actions.append(action % num_actions_per_agent)
        action //= num_actions_per_agent

    return individual_actions


def propose_position(position, action, grid_size, obstacle_set):
    proposed_position = np.array(position, dtype=np.int32).copy()

    if action == 0:      # up
        proposed_position[1] -= 1
    elif action == 1:    # down
        proposed_position[1] += 1
    elif action == 2:    # left
        proposed_position[0] -= 1
    elif action == 3:    # right
        proposed_position[0] += 1

    proposed_position = np.clip(
        proposed_position,
        0,
        grid_size - 1,
    )

    if tuple(proposed_position) in obstacle_set:
        return np.array(position, dtype=np.int32).copy()

    return proposed_position


def has_collision(positions):
    position_tuples = [tuple(position) for position in positions]
    return len(position_tuples) != len(set(position_tuples))


def has_position_swap(old_positions, proposed_positions):
    old_tuples = [tuple(position) for position in old_positions]
    proposed_tuples = [tuple(position) for position in proposed_positions]

    for i in range(len(old_positions)):
        for j in range(i + 1, len(old_positions)):
            if (
                proposed_tuples[i] == old_tuples[j]
                and proposed_tuples[j] == old_tuples[i]
            ):
                return True

    return False


def choose_greedy_joint_action(env):
    """
    Obstacle-aware one-step greedy baseline.

    It chooses the joint action that minimizes the closest agent-food
    Manhattan distance after one simulated step.

    This is not full path planning.
    """
    best_action = 0
    best_distance = float("inf")

    old_positions = [
        position.copy()
        for position in env.agent_positions
    ]

    food_pos = env.food_pos.copy()
    obstacle_set = {
        tuple(obstacle)
        for obstacle in env.obstacles
    }

    for joint_action in range(env.action_space.n):
        individual_actions = decode_joint_action(
            action=joint_action,
            num_agents=env.num_agents,
        )

        proposed_positions = []

        for agent_index, agent_action in enumerate(individual_actions):
            proposed_position = propose_position(
                position=old_positions[agent_index],
                action=agent_action,
                grid_size=env.grid_size,
                obstacle_set=obstacle_set,
            )

            proposed_positions.append(proposed_position)

        if has_collision(proposed_positions) or has_position_swap(
            old_positions,
            proposed_positions,
        ):
            proposed_positions = old_positions

        distances = [
            abs(position[0] - food_pos[0]) + abs(position[1] - food_pos[1])
            for position in proposed_positions
        ]

        closest_distance = min(distances)

        if closest_distance < best_distance:
            best_distance = closest_distance
            best_action = joint_action

    return best_action


def evaluate_random_baseline(
    episodes=1000,
    grid_size=5,
    num_agents=2,
    max_steps=50,
    action_seed=42,
):
    env = make_env(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
    )

    env.action_space.seed(action_seed)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()

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


def evaluate_greedy_baseline(
    episodes=1000,
    grid_size=5,
    num_agents=2,
    max_steps=50,
):
    env = make_env(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
    )

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = choose_greedy_joint_action(env)

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


def evaluate_ppo_model(
    model,
    episodes=1000,
    grid_size=5,
    num_agents=2,
    max_steps=50,
    deterministic=True,
    action_seed=42,
):
    set_random_seed(action_seed)

    env = make_env(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
    )

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0.0
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
    max_steps=50,
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
            max_steps=max_steps,
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


def evaluate_grid_size(
    model,
    grid_size,
    episodes=1000,
    num_agents=2,
    max_steps=50,
):
    print(f"Evaluating grid size: {grid_size}x{grid_size}")

    random_results = evaluate_random_baseline(
        episodes=episodes,
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
        action_seed=42,
    )

    greedy_results = evaluate_greedy_baseline(
        episodes=episodes,
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
    )

    deterministic_results = evaluate_ppo_model(
        model=model,
        episodes=episodes,
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
        deterministic=True,
        action_seed=42,
    )

    stochastic_results = evaluate_stochastic_ppo(
        model=model,
        episodes=episodes,
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=max_steps,
    )

    return {
        "random": random_results,
        "greedy": greedy_results,
        "ppo_deterministic": deterministic_results,
        "ppo_stochastic": stochastic_results,
    }


def print_grid_results(grid_size, results):
    print()
    print(f"===== RESULTS: {grid_size}x{grid_size} =====")

    print(
        "Random baseline: "
        f"{results['random']['success_rate'] * 100:.2f}% success, "
        f"average reward {results['random']['average_reward']:.2f}, "
        f"average episode length "
        f"{results['random']['average_episode_length']:.2f}"
    )

    print(
        "Greedy obstacle-aware baseline: "
        f"{results['greedy']['success_rate'] * 100:.2f}% success, "
        f"average reward {results['greedy']['average_reward']:.2f}, "
        f"average episode length "
        f"{results['greedy']['average_episode_length']:.2f}"
    )

    print("PPO deterministic:")
    print(
        f"Success Rate: "
        f"{results['ppo_deterministic']['success_rate'] * 100:.2f}%"
    )
    print(
        f"Average Reward: "
        f"{results['ppo_deterministic']['average_reward']:.2f}"
    )
    print(
        "Average Episode Length: "
        f"{results['ppo_deterministic']['average_episode_length']:.2f}"
    )

    print("PPO stochastic summary:")
    print(
        "Mean Success Rate: "
        f"{results['ppo_stochastic']['success_rate_mean'] * 100:.2f}%"
    )
    print(
        "Success Rate Std: "
        f"{results['ppo_stochastic']['success_rate_std'] * 100:.2f}%"
    )
    print(
        "Min Success Rate: "
        f"{results['ppo_stochastic']['success_rate_min'] * 100:.2f}%"
    )
    print(
        "Max Success Rate: "
        f"{results['ppo_stochastic']['success_rate_max'] * 100:.2f}%"
    )
    print(
        "Mean Average Reward: "
        f"{results['ppo_stochastic']['average_reward_mean']:.2f}"
    )
    print(
        "Mean Average Episode Length: "
        f"{results['ppo_stochastic']['average_episode_length_mean']:.2f}"
    )


def save_results_to_file(
    model_path,
    all_results,
    output_path,
    episodes,
    max_steps,
):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MULTI-AGENT RANDOM-OBSTACLE GRID GENERALIZATION\n")
        file.write("================================================\n\n")

        file.write("Experiment setup:\n")
        file.write("- Training grid size: 5x5\n")
        file.write("- Evaluation grid sizes: 5x5 and 10x10\n")
        file.write("- Number of agents: 2\n")
        file.write("- Random obstacles: True\n")
        file.write("- Number of obstacles: 3\n")
        file.write(f"- Max steps: {max_steps}\n")
        file.write("- Control type: centralized joint action space\n")
        file.write("- Joint action space: 16 actions\n")
        file.write("- Observation size: 12\n")
        file.write(f"- Evaluation episodes: {episodes} seeded episodes\n")
        file.write(f"- Model: {model_path}\n\n")

        for grid_size, results in all_results.items():
            file.write(f"RESULTS: {grid_size}x{grid_size}\n")
            file.write("-" * 40)
            file.write("\n")

            file.write(
                f"Random baseline: "
                f"{results['random']['success_rate'] * 100:.2f}% success, "
                f"average reward {results['random']['average_reward']:.2f}, "
                f"average episode length "
                f"{results['random']['average_episode_length']:.2f}\n"
            )

            file.write(
                f"Greedy obstacle-aware baseline: "
                f"{results['greedy']['success_rate'] * 100:.2f}% success, "
                f"average reward {results['greedy']['average_reward']:.2f}, "
                f"average episode length "
                f"{results['greedy']['average_episode_length']:.2f}\n"
            )

            file.write("PPO deterministic:\n")
            file.write(
                f"Success Rate: "
                f"{results['ppo_deterministic']['success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Average Reward: "
                f"{results['ppo_deterministic']['average_reward']:.2f}\n"
            )
            file.write(
                "Average Episode Length: "
                f"{results['ppo_deterministic']['average_episode_length']:.2f}\n"
            )

            file.write("PPO stochastic:\n")
            file.write(
                "Mean Success Rate: "
                f"{results['ppo_stochastic']['success_rate_mean'] * 100:.2f}%\n"
            )
            file.write(
                "Success Rate Std: "
                f"{results['ppo_stochastic']['success_rate_std'] * 100:.2f}%\n"
            )
            file.write(
                "Min Success Rate: "
                f"{results['ppo_stochastic']['success_rate_min'] * 100:.2f}%\n"
            )
            file.write(
                "Max Success Rate: "
                f"{results['ppo_stochastic']['success_rate_max'] * 100:.2f}%\n"
            )
            file.write(
                "Mean Average Reward: "
                f"{results['ppo_stochastic']['average_reward_mean']:.2f}\n"
            )
            file.write(
                "Mean Average Episode Length: "
                f"{results['ppo_stochastic']['average_episode_length_mean']:.2f}\n"
            )

            file.write("\nPer-action-seed stochastic results:\n")
            for action_seed, seed_results in results["ppo_stochastic"][
                "per_seed_results"
            ]:
                file.write(
                    f"Action Seed {action_seed}: "
                    f"Success Rate {seed_results['success_rate'] * 100:.2f}%, "
                    f"Average Reward {seed_results['average_reward']:.2f}, "
                    f"Average Episode Length "
                    f"{seed_results['average_episode_length']:.2f}\n"
                )

            file.write("\n")

        file.write("Interpretation:\n")
        file.write(
            "This experiment evaluates whether the centralized PPO policy trained "
            "on 5x5 randomized-obstacle environments can generalize to a larger "
            "10x10 grid while keeping the same observation structure and number "
            "of random obstacles. Comparing 5x5 and 10x10 performance measures "
            "spatial generalization. Random and greedy baselines are included to "
            "separate learned-policy performance from uninformed exploration and "
            "local obstacle-aware heuristics.\n"
        )


def main():
    model_path = "models/ppo_multi_agent_2agents_random_obstacles.zip"
    output_path = "results/multi_agent_random_obstacle_grid_generalization_summary.txt"

    episodes = 1000
    num_agents = 2
    max_steps = 50
    grid_sizes = [5, 10]

    print("===== MULTI-AGENT RANDOM-OBSTACLE GRID GENERALIZATION =====")
    print(f"Model: {model_path}")
    print("Training Grid Size: 5x5")
    print(f"Evaluation Grid Sizes: {grid_sizes}")
    print("Number of Agents: 2")
    print("Random Obstacles: True")
    print("Number of Obstacles: 3")
    print(f"Episodes per evaluation: {episodes}")
    print()

    model = PPO.load(model_path)

    all_results = {}

    for grid_size in grid_sizes:
        results = evaluate_grid_size(
            model=model,
            grid_size=grid_size,
            episodes=episodes,
            num_agents=num_agents,
            max_steps=max_steps,
        )

        all_results[grid_size] = results
        print_grid_results(
            grid_size=grid_size,
            results=results,
        )

    save_results_to_file(
        model_path=model_path,
        all_results=all_results,
        output_path=output_path,
        episodes=episodes,
        max_steps=max_steps,
    )

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()