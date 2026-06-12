import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

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
    """
    Encode individual actions into the centralized joint action used by the environment.

    The environment decodes actions using base-4 representation:
    action = a1 + 4*a2 + 16*a3 + ...
    """
    joint_action = 0

    for agent_index, action in enumerate(individual_actions):
        joint_action += int(action) * (4 ** agent_index)

    return joint_action


def greedy_decentralized_action(obs, num_agents, grid_size):
    """
    Simple decentralized greedy baseline.

    Each agent tries to move closer to the food.
    Agents are selected sequentially and avoid choosing already occupied proposed cells.
    """
    agent_positions, food_pos = parse_observation(obs, num_agents)

    proposed_positions = []
    individual_actions = []

    for agent_index, agent_position in enumerate(agent_positions):
        candidate_actions = []

        for action in ACTIONS:
            next_position = apply_action(agent_position, action, grid_size)
            distance_to_food = manhattan_distance(next_position, food_pos)

            # Penalize actions that do not move the agent, usually wall hits.
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


def evaluate_random_baseline(num_agents, grid_size=5, episodes=1000, action_seed=42):
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


def evaluate_greedy_baseline(num_agents, grid_size=5, episodes=1000):
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


def print_results(title, results):
    print(title)
    print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Average Episode Length: {results['average_episode_length']:.2f}")
    print()


def save_results_to_file(all_results, output_path):
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MULTI-AGENT BASELINE EVALUATION\n")
        file.write("================================\n\n")

        for num_agents, results in all_results.items():
            file.write(f"{num_agents} agents, 5x5 grid\n")
            file.write("-" * 30 + "\n\n")

            for method_name, method_results in results.items():
                file.write(f"{method_name}:\n")
                file.write(
                    f"Success Rate: {method_results['success_rate'] * 100:.2f}%\n"
                )
                file.write(
                    f"Average Reward: {method_results['average_reward']:.2f}\n"
                )
                file.write(
                    f"Average Episode Length: "
                    f"{method_results['average_episode_length']:.2f}\n\n"
                )


def main():
    episodes = 1000
    grid_size = 5
    agent_counts = [2, 3]

    all_results = {}

    print("===== MULTI-AGENT BASELINE EVALUATION =====")
    print(f"Episodes: {episodes}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Agent counts: {agent_counts}")
    print()

    for num_agents in agent_counts:
        print(f"===== {num_agents} AGENTS =====")

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

        all_results[num_agents] = {
            "Random baseline": random_results,
            "Greedy decentralized baseline": greedy_results,
        }

        print_results("Random baseline:", random_results)
        print_results("Greedy decentralized baseline:", greedy_results)

    output_path = "results/multi_agent_baseline_summary.txt"
    save_results_to_file(all_results, output_path)

    print("===== FINAL MULTI-AGENT BASELINE COMPARISON =====")
    print("| Agents | Method | Success Rate | Average Reward | Average Episode Length |")
    print("|---:|---|---:|---:|---:|")

    for num_agents, results in all_results.items():
        for method_name, method_results in results.items():
            print(
                f"| {num_agents} | "
                f"{method_name} | "
                f"{method_results['success_rate'] * 100:.2f}% | "
                f"{method_results['average_reward']:.2f} | "
                f"{method_results['average_episode_length']:.2f} |"
            )

    print()
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()