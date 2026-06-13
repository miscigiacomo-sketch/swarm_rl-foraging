import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from env.multi_agent_foraging_env import MultiAgentForagingEnv


ACTIONS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
}


def parse_observation(obs, num_agents):
    agent_positions = []

    for agent_index in range(num_agents):
        x = int(obs[2 * agent_index])
        y = int(obs[2 * agent_index + 1])
        agent_positions.append((x, y))

    food_pos = (
        int(obs[2 * num_agents]),
        int(obs[2 * num_agents + 1]),
    )

    return agent_positions, food_pos


def manhattan_distance(pos_a, pos_b):
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def apply_action(position, action, grid_size):
    dx, dy = ACTIONS[action]

    new_x = position[0] + dx
    new_y = position[1] + dy

    new_x = min(max(new_x, 0), grid_size - 1)
    new_y = min(max(new_y, 0), grid_size - 1)

    return (new_x, new_y)


def encode_joint_action(individual_actions):
    joint_action = 0

    for agent_index, action in enumerate(individual_actions):
        joint_action += int(action) * (4 ** agent_index)

    return joint_action


def greedy_decentralized_action(obs, num_agents, grid_size):
    agent_positions, food_pos = parse_observation(obs, num_agents)

    proposed_positions = []
    individual_actions = []

    for agent_position in agent_positions:
        candidate_actions = []

        for action in ACTIONS:
            next_position = apply_action(agent_position, action, grid_size)
            distance_to_food = manhattan_distance(next_position, food_pos)

            no_move_penalty = 1 if next_position == agent_position else 0

            candidate_actions.append(
                (
                    distance_to_food,
                    no_move_penalty,
                    action,
                    next_position,
                )
            )

        candidate_actions.sort(key=lambda item: (item[0], item[1]))

        selected_action = candidate_actions[0][2]
        selected_position = candidate_actions[0][3]

        for _, _, action, next_position in candidate_actions:
            if next_position in proposed_positions:
                continue

            selected_action = action
            selected_position = next_position
            break

        individual_actions.append(selected_action)
        proposed_positions.append(selected_position)

    return encode_joint_action(individual_actions)


def evaluate_random_baseline(num_agents, grid_size, episodes=1000, action_seed=42):
    env = MultiAgentForagingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=50,
    )

    rng = np.random.default_rng(action_seed)

    successes = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = int(rng.integers(env.action_space.n))

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


def evaluate_greedy_baseline(num_agents, grid_size, episodes=1000):
    env = MultiAgentForagingEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        max_steps=50,
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
            action = greedy_decentralized_action(
                obs=obs,
                num_agents=num_agents,
                grid_size=grid_size,
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


def evaluate_ppo_model(
    model,
    num_agents,
    grid_size,
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
    model,
    num_agents,
    grid_size,
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
            model=model,
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


def save_results_to_file(all_results, output_path):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MULTI-AGENT GRID-SIZE GENERALIZATION\n")
        file.write("====================================\n\n")
        file.write("Model: models/ppo_multi_agent_2agents.zip\n")
        file.write("Training environment: 2 agents, 5x5 grid\n")
        file.write("Evaluation: 2 agents, 5x5 and 10x10 grids\n\n")

        for grid_size, results in all_results.items():
            file.write(f"{grid_size}x{grid_size} grid\n")
            file.write("-" * 30 + "\n\n")

            file.write("Random baseline:\n")
            file.write(
                f"Success Rate: {results['random']['success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Average Episode Length: "
                f"{results['random']['average_episode_length']:.2f}\n\n"
            )

            file.write("Greedy decentralized baseline:\n")
            file.write(
                f"Success Rate: {results['greedy']['success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Average Episode Length: "
                f"{results['greedy']['average_episode_length']:.2f}\n\n"
            )

            file.write("PPO deterministic:\n")
            file.write(
                f"Success Rate: "
                f"{results['ppo_deterministic']['success_rate'] * 100:.2f}%\n"
            )
            file.write(
                f"Average Episode Length: "
                f"{results['ppo_deterministic']['average_episode_length']:.2f}\n\n"
            )

            file.write("PPO stochastic:\n")
            file.write(
                f"Mean Success Rate: "
                f"{results['ppo_stochastic']['success_rate_mean'] * 100:.2f}%\n"
            )
            file.write(
                f"Success Rate Std: "
                f"{results['ppo_stochastic']['success_rate_std'] * 100:.2f}%\n"
            )
            file.write(
                f"Mean Average Episode Length: "
                f"{results['ppo_stochastic']['average_episode_length_mean']:.2f}\n\n"
            )


def main():
    model_path = "models/ppo_multi_agent_2agents.zip"
    episodes = 1000
    num_agents = 2
    grid_sizes = [5, 10]

    model = PPO.load(model_path)

    all_results = {}

    print("===== MULTI-AGENT GRID-SIZE GENERALIZATION =====")
    print(f"Model: {model_path}")
    print("Training environment: 2 agents, 5x5")
    print(f"Evaluation grid sizes: {grid_sizes}")
    print(f"Episodes: {episodes}")
    print()

    for grid_size in grid_sizes:
        print(f"===== {grid_size}x{grid_size} GRID =====")

        random_results = evaluate_random_baseline(
            num_agents=num_agents,
            grid_size=grid_size,
            episodes=episodes,
            action_seed=42,
        )

        greedy_results = evaluate_greedy_baseline(
            num_agents=num_agents,
            grid_size=grid_size,
            episodes=episodes,
        )

        ppo_deterministic_results = evaluate_ppo_model(
            model=model,
            num_agents=num_agents,
            grid_size=grid_size,
            episodes=episodes,
            deterministic=True,
            action_seed=42,
        )

        ppo_stochastic_results = evaluate_stochastic_ppo(
            model=model,
            num_agents=num_agents,
            grid_size=grid_size,
            episodes=episodes,
        )

        all_results[grid_size] = {
            "random": random_results,
            "greedy": greedy_results,
            "ppo_deterministic": ppo_deterministic_results,
            "ppo_stochastic": ppo_stochastic_results,
        }

        print("Random baseline:")
        print(f"Success Rate: {random_results['success_rate'] * 100:.2f}%")
        print(
            f"Average Episode Length: "
            f"{random_results['average_episode_length']:.2f}"
        )
        print()

        print("Greedy decentralized baseline:")
        print(f"Success Rate: {greedy_results['success_rate'] * 100:.2f}%")
        print(
            f"Average Episode Length: "
            f"{greedy_results['average_episode_length']:.2f}"
        )
        print()

        print("PPO deterministic:")
        print(
            f"Success Rate: "
            f"{ppo_deterministic_results['success_rate'] * 100:.2f}%"
        )
        print(
            f"Average Episode Length: "
            f"{ppo_deterministic_results['average_episode_length']:.2f}"
        )
        print()

        print("PPO stochastic:")
        print(
            f"Mean Success Rate: "
            f"{ppo_stochastic_results['success_rate_mean'] * 100:.2f}%"
        )
        print(
            f"Success Rate Std: "
            f"{ppo_stochastic_results['success_rate_std'] * 100:.2f}%"
        )
        print(
            f"Mean Average Episode Length: "
            f"{ppo_stochastic_results['average_episode_length_mean']:.2f}"
        )
        print()

    output_path = "results/multi_agent_grid_generalization_summary.txt"
    save_results_to_file(all_results, output_path)

    print("===== FINAL MULTI-AGENT GRID GENERALIZATION COMPARISON =====")
    print(
        "| Grid | Random Success | Greedy Success | "
        "PPO Deterministic Success | PPO Stochastic Mean Success | "
        "PPO Stochastic Avg Length |"
    )
    print("|---:|---:|---:|---:|---:|---:|")

    for grid_size, results in all_results.items():
        print(
            f"| {grid_size}x{grid_size} | "
            f"{results['random']['success_rate'] * 100:.2f}% | "
            f"{results['greedy']['success_rate'] * 100:.2f}% | "
            f"{results['ppo_deterministic']['success_rate'] * 100:.2f}% | "
            f"{results['ppo_stochastic']['success_rate_mean'] * 100:.2f}% | "
            f"{results['ppo_stochastic']['average_episode_length_mean']:.2f} |"
        )

    print()
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()